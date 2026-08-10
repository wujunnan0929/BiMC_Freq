"""Frequency-aware prototype utilities for training-free FSCIL.

The functions in this module do not depend on CLIP internals.  This keeps the
frequency decomposition, prototype estimation, semantic calibration, and
logit fusion independently testable.
"""

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RadialFrequencyDecomposer(nn.Module):
    """Split CLIP-normalized images into low, middle, and high frequencies.

    The exact FFT components sum to the input image in pixel space.  Before the
    middle/high components are passed to CLIP, their zero-frequency component
    is replaced with the image mean.  This produces valid image-like inputs
    while retaining the selected band details.
    """

    def __init__(
        self,
        low_cutoff: float,
        high_cutoff: float,
        mean: Sequence[float] = (0.48145466, 0.4578275, 0.40821073),
        std: Sequence[float] = (0.26862954, 0.26130258, 0.27577711),
        center_residual_bands: bool = True,
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
        height, width = pixels.shape[-2:]
        # FFT on fp16 tensors is restricted for some CUDA sizes (e.g. 224), so
        # always decompose in fp32 and cast the CLIP inputs later.
        spectrum = torch.fft.fft2(pixels.float(), dim=(-2, -1), norm="ortho")
        masks = self._radial_masks(height, width, pixels.device)
        components = []
        for mask in masks:
            component = torch.fft.ifft2(
                spectrum * mask[None, None], dim=(-2, -1), norm="ortho"
            ).real
            components.append(component)
        return torch.stack(components, dim=1)

    def forward(self, normalized_images: torch.Tensor) -> torch.Tensor:
        """Create normalized, image-like inputs for each frequency band."""
        input_dtype = normalized_images.dtype
        mean = self.mean.to(device=normalized_images.device, dtype=torch.float32)
        std = self.std.to(device=normalized_images.device, dtype=torch.float32)
        pixels = normalized_images.float() * std + mean
        components = self.split_pixels(pixels)

        band_images = components.clone()
        if self.center_residual_bands:
            channel_mean = pixels.mean(dim=(-2, -1), keepdim=True)
            band_images[:, 1:] = band_images[:, 1:] + channel_mean[:, None]
        band_images = band_images.clamp(0.0, 1.0)
        band_images = (band_images - mean[:, None]) / std[:, None]
        return band_images.to(dtype=input_dtype)


def compute_frequency_prototypes(
    features: torch.Tensor,
    labels: torch.Tensor,
    class_index: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    for class_id in class_index:
        class_mask = labels == int(class_id)
        if not torch.any(class_mask):
            raise ValueError("No samples found for class {}.".format(class_id))
        class_features = F.normalize(features[class_mask], dim=-1)
        prototype = F.normalize(class_features.mean(dim=0), dim=-1)
        similarity = torch.sum(class_features * prototype.unsqueeze(0), dim=-1)
        prototypes.append(prototype)
        uncertainties.append(1.0 - similarity.mean(dim=0))

    return torch.stack(prototypes, dim=0), torch.stack(uncertainties, dim=0)


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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Semantically calibrate frequency prototypes and estimate band weights.

    Semantic correction grows with visual uncertainty, but is suppressed when
    the visual and semantic band prototypes disagree.  Band fusion prefers
    semantically aligned and visually compact bands.
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

    gates = semantic_weight * (1.0 + uncertainty_scale * uncertainty)
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
        if prior.ndim != 1 or prior.numel() != num_bands or torch.any(prior <= 0):
            raise ValueError("Band prior must contain one positive value per band.")
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


def compute_frequency_logits(
    query_features: torch.Tensor,
    class_prototypes: torch.Tensor,
    class_band_weights: torch.Tensor,
) -> torch.Tensor:
    """Compute class logits from matched query/prototype frequency bands."""
    if query_features.ndim != 3 or class_prototypes.ndim != 3:
        raise ValueError("Query and prototype tensors must have shape [N/C, B, D].")
    if query_features.shape[1:] != class_prototypes.shape[1:]:
        raise ValueError("Query and prototype band/feature dimensions must match.")
    if class_band_weights.shape != class_prototypes.shape[:2]:
        raise ValueError("Class band weights must have shape [C, B].")

    query = F.normalize(query_features, dim=-1)
    prototypes = F.normalize(class_prototypes, dim=-1)
    band_similarity = torch.einsum("nbd,cbd->ncb", query, prototypes)
    return torch.sum(band_similarity * class_band_weights.unsqueeze(0), dim=-1)
