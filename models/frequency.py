"""Frequency-aware prototype utilities for training-free FSCIL.

The functions in this module do not depend on CLIP internals.  This keeps the
frequency decomposition, prototype estimation, semantic calibration, and
logit fusion independently testable.
"""

import warnings
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RadialFrequencyDecomposer(nn.Module):
    """Build low-, middle-, and high-frequency views of normalized images.

    ``disjoint`` reproduces the original residual-band experiment, ``natural``
    constructs cumulative/enhanced image-like views, and ``original`` repeats
    the input image as a control. Exact FFT components sum to the pixel image.
    """

    def __init__(
        self,
        low_cutoff: float,
        high_cutoff: float,
        mean: Sequence[float] = (0.48145466, 0.4578275, 0.40821073),
        std: Sequence[float] = (0.26862954, 0.26130258, 0.27577711),
        center_residual_bands: bool = True,
        fft_batch_size: int = 8,
        fft_device: str = "auto",
        view_mode: str = "disjoint",
        high_enhance: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0.0 < low_cutoff < high_cutoff < 1.0:
            raise ValueError(
                "Frequency cutoffs must satisfy 0 < low_cutoff < "
                "high_cutoff < 1."
            )
        self.low_cutoff = float(low_cutoff)
        self.high_cutoff = float(high_cutoff)
        self.center_residual_bands = bool(center_residual_bands)
        if fft_batch_size <= 0:
            raise ValueError("FFT batch size must be positive.")
        self.fft_batch_size = int(fft_batch_size)
        fft_device = fft_device.lower()
        if fft_device not in ("auto", "cuda", "cpu"):
            raise ValueError("FFT device must be one of: auto, cuda, cpu.")
        self.fft_device = fft_device
        self._cpu_fallback_warned = False
        view_mode = view_mode.lower()
        if view_mode not in ("disjoint", "natural", "original"):
            raise ValueError(
                "Frequency view mode must be one of: disjoint, natural, original."
            )
        if high_enhance < 0.0:
            raise ValueError("High-frequency enhancement must be non-negative.")
        self.view_mode = view_mode
        self.high_enhance = float(high_enhance)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    @property
    def num_bands(self) -> int:
        return 3

    def _radial_masks(
        self, height: int, width: int, device: torch.device
    ) -> torch.Tensor:
        # Normalize the radius by the corner Nyquist frequency, so the two
        # cutoffs always lie in (0, 1), independent of image resolution.
        fy = torch.fft.fftfreq(height, device=device)
        fx = torch.fft.fftfreq(width, device=device)
        radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
        radius = radius / (0.5 ** 2 + 0.5 ** 2) ** 0.5

        low = radius <= self.low_cutoff
        middle = (radius > self.low_cutoff) & (radius <= self.high_cutoff)
        high = radius > self.high_cutoff
        return torch.stack((low, middle, high), dim=0)

    def split_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """Return exact band components with shape ``[N, 3, C, H, W]``."""
        if pixels.ndim != 4:
            raise ValueError("Expected pixels with shape [N, C, H, W].")
        original_device = pixels.device
        if self.fft_device == "cpu" and pixels.is_cuda:
            cpu_components = self.split_pixels(pixels.cpu())
            return cpu_components.to(original_device)
        if self.fft_device == "cuda" and not pixels.is_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("FFT_DEVICE is cuda, but CUDA is unavailable.")
            cuda_components = self.split_pixels(pixels.cuda())
            return cuda_components.to(original_device)

        height, width = pixels.shape[-2:]
        masks = self._radial_masks(height, width, pixels.device)

        def split_with_chunk_size(chunk_size: int) -> torch.Tensor:
            output_chunks = []
            for pixel_chunk in torch.split(pixels, chunk_size, dim=0):
                # FFT on fp16 tensors is restricted for some CUDA sizes (e.g.
                # 224), so always decompose in fp32. Chunking also avoids
                # fragile, high-workspace cuFFT plans on older CUDA stacks.
                spectrum = torch.fft.fft2(
                    pixel_chunk.float(), dim=(-2, -1), norm="ortho"
                )
                band_chunks = []
                for mask in masks:
                    component = torch.fft.ifft2(
                        spectrum * mask[None, None], dim=(-2, -1), norm="ortho"
                    ).real
                    band_chunks.append(component)
                output_chunks.append(torch.stack(band_chunks, dim=1))
            return torch.cat(output_chunks, dim=0)

        try:
            return split_with_chunk_size(self.fft_batch_size)
        except RuntimeError as error:
            is_cufft_error = pixels.is_cuda and "CUFFT" in str(error).upper()
            if not is_cufft_error or self.fft_batch_size == 1:
                raise
            # A failed large cuFFT plan can remain cached. Clear it and retry
            # sample-by-sample, which uses the smallest possible plan.
            torch.backends.cuda.cufft_plan_cache.clear()
            torch.cuda.empty_cache()
            try:
                return split_with_chunk_size(1)
            except RuntimeError as retry_error:
                if "CUFFT" not in str(retry_error).upper():
                    raise
                if self.fft_device == "auto":
                    if not self._cpu_fallback_warned:
                        warnings.warn(
                            "cuFFT failed with chunk size 1; falling back to "
                            "exact CPU FFT. Set FFT_DEVICE: cpu to skip the "
                            "failed CUDA attempt on subsequent runs.",
                            RuntimeWarning,
                        )
                        self._cpu_fallback_warned = True
                    # Persist the fallback so later batches do not repeatedly
                    # create and fail the same cuFFT plans.
                    self.fft_device = "cpu"
                    cpu_components = self.split_pixels(pixels.cpu())
                    return cpu_components.to(original_device)
                raise RuntimeError(
                    "cuFFT failed even with FFT_BATCH_SIZE=1. Check that the "
                    "NVIDIA driver matches the CUDA runtime used by PyTorch, "
                    "or set FFT_DEVICE: cpu."
                ) from retry_error

    def forward(self, normalized_images: torch.Tensor) -> torch.Tensor:
        """Create normalized, image-like inputs for each frequency band."""
        input_dtype = normalized_images.dtype
        mean = self.mean.to(device=normalized_images.device, dtype=torch.float32)
        std = self.std.to(device=normalized_images.device, dtype=torch.float32)
        pixels = normalized_images.float() * std + mean

        if self.view_mode == "original":
            # Control experiment: retain the complete frequency pipeline while
            # replacing all three frequency views with the original image.
            band_images = pixels.unsqueeze(1).expand(-1, self.num_bands, -1, -1, -1)
        else:
            components = self.split_pixels(pixels)
            if self.view_mode == "disjoint":
                band_images = components.clone()
                if self.center_residual_bands:
                    channel_mean = pixels.mean(dim=(-2, -1), keepdim=True)
                    band_images[:, 1:] = band_images[:, 1:] + channel_mean[:, None]
            else:
                # Natural CLIP-compatible views:
                #   low    = low-pass image,
                #   middle = low + middle = image without high-frequency noise,
                #   high   = original image with its high-frequency residual enhanced.
                low = components[:, 0]
                middle = components[:, 0] + components[:, 1]
                high = pixels + self.high_enhance * components[:, 2]
                band_images = torch.stack((low, middle, high), dim=1)
        band_images = band_images.clamp(0.0, 1.0)
        band_images = (band_images - mean[:, None]) / std[:, None]
        return band_images.to(dtype=input_dtype)


def compute_frequency_prototypes(
    features: torch.Tensor,
    labels: torch.Tensor,
    class_index: Sequence[int],
    return_counts: bool = False,
):
    """Estimate per-class, per-band prototypes and angular uncertainty.

    Args:
        features: L2-normalized features of shape ``[N, B, D]``.
        labels: Global class labels of shape ``[N]``.
        class_index: Desired output class order.

    Returns:
        Prototypes with shape ``[C, B, D]`` and uncertainty ``[C, B]``.
    """
    if features.ndim != 3:
        raise ValueError("Expected frequency features with shape [N, B, D].")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("Labels must have shape [N] and match the features.")

    prototypes = []
    uncertainties = []
    sample_counts = []
    for class_id in class_index:
        class_mask = labels == int(class_id)
        if not torch.any(class_mask):
            raise ValueError("No samples found for class {}.".format(class_id))
        class_features = F.normalize(features[class_mask], dim=-1)
        prototype = F.normalize(class_features.mean(dim=0), dim=-1)
        similarity = torch.sum(class_features * prototype.unsqueeze(0), dim=-1)
        prototypes.append(prototype)
        uncertainties.append(1.0 - similarity.mean(dim=0))
        sample_counts.append(class_mask.sum())

    prototype_tensor = torch.stack(prototypes, dim=0)
    uncertainty_tensor = torch.stack(uncertainties, dim=0)
    if return_counts:
        return (
            prototype_tensor,
            uncertainty_tensor,
            torch.stack(sample_counts).to(dtype=features.dtype),
        )
    return prototype_tensor, uncertainty_tensor


def route_description_embeddings(
    descriptions: Sequence[str],
    embeddings: torch.Tensor,
    keyword_groups: Sequence[Sequence[str]],
) -> torch.Tensor:
    """Route class descriptions into semantic bands using keyword matching.

    Each band's prototype is the normalized average of matching descriptions.
    When no description matches a band, the complete class description set is
    used as a conservative fallback.
    """
    if embeddings.ndim != 2 or embeddings.shape[0] != len(descriptions):
        raise ValueError("Description embeddings must have shape [num_prompts, D].")
    if len(descriptions) == 0:
        raise ValueError("At least one description is required.")

    lower_descriptions = [description.lower() for description in descriptions]
    band_embeddings = []
    for keywords in keyword_groups:
        normalized_keywords = [keyword.lower() for keyword in keywords]
        selected_indices = [
            index
            for index, description in enumerate(lower_descriptions)
            if any(keyword in description for keyword in normalized_keywords)
        ]
        if selected_indices:
            selected = embeddings[selected_indices]
        else:
            selected = embeddings
        band_embeddings.append(F.normalize(selected.mean(dim=0), dim=-1))
    return torch.stack(band_embeddings, dim=0)


def select_frequency_description_prototypes(
    text_candidates: torch.Tensor,
    visual_prototypes: torch.Tensor,
    top_k: int = 3,
    temperature: float = 0.07,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ground explicit frequency descriptions with support-set prototypes.

    Args:
        text_candidates: Normalized or unnormalized text features with shape
            ``[C, B, K, D]``.
        visual_prototypes: Support-set visual prototypes with shape
            ``[C, B, D]``.
        top_k: Number of visually aligned descriptions retained per class and
            band. Values larger than the available candidate count retain all
            candidates.
        temperature: Softmax temperature used to combine the selected text
            features.

    Returns:
        The grounded semantic prototype ``[C, B, D]``, selected similarities
        ``[C, B, top_k]``, and selected candidate indices
        ``[C, B, top_k]``.
    """
    if text_candidates.ndim != 4:
        raise ValueError("Text candidates must have shape [C, B, K, D].")
    if visual_prototypes.ndim != 3:
        raise ValueError("Visual prototypes must have shape [C, B, D].")
    if text_candidates.shape[:2] != visual_prototypes.shape[:2]:
        raise ValueError("Text and visual class/band dimensions must match.")
    if text_candidates.shape[-1] != visual_prototypes.shape[-1]:
        raise ValueError("Text and visual feature dimensions must match.")
    if text_candidates.shape[2] == 0:
        raise ValueError("At least one text candidate is required.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if temperature <= 0.0:
        raise ValueError("Description temperature must be positive.")

    text = F.normalize(text_candidates, dim=-1)
    visual = F.normalize(visual_prototypes, dim=-1)
    similarities = torch.einsum("cbkd,cbd->cbk", text, visual)
    retained = min(int(top_k), text.shape[2])
    top_scores, top_indices = similarities.topk(retained, dim=-1)
    gather_indices = top_indices.unsqueeze(-1).expand(
        -1, -1, -1, text.shape[-1]
    )
    selected = torch.gather(text, dim=2, index=gather_indices)
    weights = torch.softmax(top_scores / temperature, dim=-1)
    semantic = F.normalize(
        torch.sum(weights.unsqueeze(-1) * selected, dim=2), dim=-1
    )
    return semantic, top_scores, top_indices


def calibrate_frequency_prototypes(
    visual_prototypes: torch.Tensor,
    semantic_prototypes: torch.Tensor,
    uncertainty: torch.Tensor,
    semantic_weight: float,
    max_semantic_weight: float,
    uncertainty_scale: float,
    alignment_scale: float,
    fusion_temperature: float,
    adaptive_fusion: bool = True,
    band_prior: Optional[torch.Tensor] = None,
    semantic_gate_mode: str = "amplify",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Semantically calibrate frequency prototypes and estimate band weights.

    Semantic correction is agreement-gated and can either grow with or be
    attenuated by uncertainty. Band fusion prefers semantically aligned and
    visually compact bands.
    """
    if visual_prototypes.shape != semantic_prototypes.shape:
        raise ValueError("Visual and semantic prototypes must have equal shapes.")
    if visual_prototypes.ndim != 3:
        raise ValueError("Expected prototypes with shape [C, B, D].")
    if uncertainty.shape != visual_prototypes.shape[:2]:
        raise ValueError("Uncertainty must have shape [C, B].")
    if not 0.0 <= semantic_weight <= max_semantic_weight <= 1.0:
        raise ValueError(
            "Weights must satisfy 0 <= semantic_weight <= "
            "max_semantic_weight <= 1."
        )
    if fusion_temperature <= 0.0:
        raise ValueError("Fusion temperature must be positive.")

    visual = F.normalize(visual_prototypes, dim=-1)
    semantic = F.normalize(semantic_prototypes, dim=-1)
    alignment = torch.sum(visual * semantic, dim=-1)
    agreement = ((alignment + 1.0) * 0.5).clamp(0.0, 1.0)

    semantic_gate_mode = semantic_gate_mode.lower()
    if semantic_gate_mode == "amplify":
        gates = semantic_weight * (1.0 + uncertainty_scale * uncertainty)
    elif semantic_gate_mode == "attenuate":
        gates = semantic_weight * torch.exp(
            -uncertainty_scale * uncertainty.clamp_min(0.0)
        )
    else:
        raise ValueError("Semantic gate mode must be amplify or attenuate.")
    gates = (gates * agreement).clamp(0.0, max_semantic_weight)
    calibrated = F.normalize(
        (1.0 - gates.unsqueeze(-1)) * visual
        + gates.unsqueeze(-1) * semantic,
        dim=-1,
    )

    num_bands = visual.shape[1]
    if band_prior is None:
        prior = visual.new_ones(num_bands)
    else:
        prior = band_prior.to(device=visual.device, dtype=visual.dtype)
        if (
            prior.ndim != 1
            or prior.numel() != num_bands
            or torch.any(prior < 0)
            or prior.sum() <= 0
        ):
            raise ValueError(
                "Band prior must contain one non-negative value per band "
                "and at least one positive value."
            )
    prior = prior / prior.sum()

    if adaptive_fusion:
        scores = (
            alignment_scale * alignment
            - uncertainty_scale * uncertainty
            + torch.log(prior.clamp_min(1e-12)).unsqueeze(0)
        )
        band_weights = F.softmax(scores / fusion_temperature, dim=1)
    else:
        band_weights = prior.unsqueeze(0).expand(visual.shape[0], -1)

    return calibrated, band_weights, gates, alignment


def compute_frequency_class_alpha(
    alignment: torch.Tensor,
    uncertainty: torch.Tensor,
    band_weights: torch.Tensor,
    sample_counts: torch.Tensor,
    max_alpha: float,
    min_alpha: float = 0.0,
    uncertainty_scale: float = 2.0,
    shot_tau: float = 5.0,
    reliability_power: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estimate a reliable frequency mixing coefficient for every class.

    Reliability combines visual-semantic agreement, within-class frequency
    compactness, and support-set size.  Few-shot or inconsistent prototypes
    therefore fall back toward the original BiMC probability distribution.
    """
    if alignment.shape != uncertainty.shape or alignment.shape != band_weights.shape:
        raise ValueError(
            "Alignment, uncertainty, and band weights must have shape [C, B]."
        )
    if sample_counts.ndim != 1 or sample_counts.shape[0] != alignment.shape[0]:
        raise ValueError("Sample counts must have shape [C].")
    if not 0.0 <= min_alpha <= max_alpha <= 1.0:
        raise ValueError("Frequency alphas must satisfy 0 <= min <= max <= 1.")
    if uncertainty_scale < 0.0 or shot_tau < 0.0 or reliability_power <= 0.0:
        raise ValueError("Reliability scales must be non-negative and power positive.")

    agreement = ((alignment + 1.0) * 0.5).clamp(0.0, 1.0)
    compactness = torch.exp(-uncertainty_scale * uncertainty.clamp_min(0.0))
    band_reliability = agreement * compactness
    class_reliability = torch.sum(band_weights * band_reliability, dim=1)

    if shot_tau > 0.0:
        counts = sample_counts.to(
            device=class_reliability.device, dtype=class_reliability.dtype
        )
        shot_reliability = counts / (counts + shot_tau)
        class_reliability = class_reliability * shot_reliability

    class_reliability = class_reliability.clamp(0.0, 1.0)
    class_reliability = class_reliability.pow(reliability_power)
    class_alpha = min_alpha + (max_alpha - min_alpha) * class_reliability
    return class_alpha, class_reliability


def compute_frequency_logits(
    query_features: torch.Tensor,
    class_prototypes: torch.Tensor,
    class_band_weights: torch.Tensor,
) -> torch.Tensor:
    """Compute class logits from matched query/prototype frequency bands."""
    band_similarity = compute_frequency_band_logits(
        query_features, class_prototypes
    )
    if class_band_weights.shape != class_prototypes.shape[:2]:
        raise ValueError("Class band weights must have shape [C, B].")
    return torch.sum(
        band_similarity * class_band_weights.unsqueeze(0), dim=-1
    )


def compute_frequency_band_logits(
    query_features: torch.Tensor,
    class_prototypes: torch.Tensor,
) -> torch.Tensor:
    """Return the un-fused cosine logit of every frequency band.

    The result has shape ``[N, C, B]``.  Keeping the band dimension is useful
    for trainable routers, which must decide whether a query benefits from a
    particular band before the logits are fused.
    """
    if query_features.ndim != 3 or class_prototypes.ndim != 3:
        raise ValueError("Query and prototype tensors must have shape [N/C, B, D].")
    if query_features.shape[1:] != class_prototypes.shape[1:]:
        raise ValueError("Query and prototype band/feature dimensions must match.")

    query = F.normalize(query_features, dim=-1)
    prototypes = F.normalize(class_prototypes, dim=-1)
    return torch.einsum("nbd,cbd->ncb", query, prototypes)


def mix_frequency_probabilities(
    original_probabilities: torch.Tensor,
    frequency_probabilities: torch.Tensor,
    class_alpha: torch.Tensor,
) -> torch.Tensor:
    """Mix frequency predictions with class-wise coefficients and renormalize."""
    if original_probabilities.shape != frequency_probabilities.shape:
        raise ValueError("Original and frequency probabilities must have equal shapes.")
    if original_probabilities.ndim != 2:
        raise ValueError("Probability tensors must have shape [N, C].")
    if (
        class_alpha.ndim != 1
        or class_alpha.shape[0] != original_probabilities.shape[1]
    ):
        raise ValueError("Class alpha must have shape [C].")
    if torch.any(class_alpha < 0.0) or torch.any(class_alpha > 1.0):
        raise ValueError("Class alpha values must lie in [0, 1].")

    alpha = class_alpha.to(
        device=frequency_probabilities.device,
        dtype=frequency_probabilities.dtype,
    ).unsqueeze(0)
    mixed = (
        (1.0 - alpha) * original_probabilities
        + alpha * frequency_probabilities
    )
    return mixed / mixed.sum(dim=-1, keepdim=True).clamp_min(1e-12)
