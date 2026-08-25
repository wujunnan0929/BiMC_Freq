"""Independent frequency modality for tri-modal BiMC.

Unlike the legacy frequency-view branch, this module never sends filtered
images through CLIP's visual encoder.  It describes an image directly from
the log-magnitude of its Fourier spectrum and maps that descriptor into the
shared CLIP embedding space with a small base-session projection network.
"""

import warnings
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencySpectrumDescriptor(nn.Module):
    """Extract phase-free spectral descriptors from normalized RGB images.

    The descriptor concatenates a pooled two-dimensional log-magnitude map
    with a radial power profile.  Phase is intentionally discarded so this
    branch captures frequency statistics rather than duplicating CLIP's
    spatial representation.
    """

    def __init__(
        self,
        grid_size: int = 16,
        radial_bins: int = 16,
        mean: Sequence[float] = (0.48145466, 0.4578275, 0.40821073),
        std: Sequence[float] = (0.26862954, 0.26130258, 0.27577711),
        fft_batch_size: int = 16,
        fft_device: str = "auto",
    ) -> None:
        super().__init__()
        if grid_size <= 0 or radial_bins <= 0:
            raise ValueError("grid_size and radial_bins must be positive.")
        if fft_batch_size <= 0:
            raise ValueError("fft_batch_size must be positive.")
        fft_device = str(fft_device).lower()
        if fft_device not in ("auto", "cpu", "cuda"):
            raise ValueError("fft_device must be one of: auto, cpu, cuda.")

        self.grid_size = int(grid_size)
        self.radial_bins = int(radial_bins)
        self.fft_batch_size = int(fft_batch_size)
        self.fft_device = fft_device
        self._cpu_fallback_warned = False
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    @property
    def descriptor_dim(self) -> int:
        return self.grid_size * self.grid_size + self.radial_bins

    def _radial_profile(self, power: torch.Tensor) -> torch.Tensor:
        height, width = power.shape[-2:]
        fy = torch.fft.fftshift(torch.fft.fftfreq(height, device=power.device))
        fx = torch.fft.fftshift(torch.fft.fftfreq(width, device=power.device))
        radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
        radius = radius / radius.max().clamp_min(1e-8)
        bin_ids = torch.clamp(
            torch.floor(radius * self.radial_bins).long(),
            max=self.radial_bins - 1,
        ).reshape(-1)

        flat_power = power.reshape(power.shape[0], -1)
        profile = power.new_zeros(power.shape[0], self.radial_bins)
        profile.scatter_add_(
            1, bin_ids.unsqueeze(0).expand(power.shape[0], -1), flat_power
        )
        return profile / profile.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def _extract(self, pixels: torch.Tensor) -> torch.Tensor:
        # Luminance keeps the branch compact while retaining texture/edge
        # frequency statistics that are complementary to CLIP image features.
        luminance_weights = pixels.new_tensor(
            (0.299, 0.587, 0.114)
        ).view(1, 3, 1, 1)
        luminance = (pixels * luminance_weights).sum(dim=1, keepdim=True)
        luminance = luminance - luminance.mean(dim=(-2, -1), keepdim=True)

        height, width = luminance.shape[-2:]
        window_y = torch.hann_window(
            height, periodic=False, device=pixels.device, dtype=pixels.dtype
        )
        window_x = torch.hann_window(
            width, periodic=False, device=pixels.device, dtype=pixels.dtype
        )
        window = (window_y[:, None] * window_x[None, :]).view(
            1, 1, height, width
        )
        spectrum = torch.fft.fft2(
            luminance * window, dim=(-2, -1), norm="ortho"
        )
        magnitude = torch.fft.fftshift(spectrum.abs(), dim=(-2, -1))

        log_magnitude = torch.log1p(magnitude)
        log_magnitude = log_magnitude - log_magnitude.mean(
            dim=(-2, -1), keepdim=True
        )
        log_magnitude = log_magnitude / log_magnitude.std(
            dim=(-2, -1), keepdim=True, unbiased=False
        ).clamp_min(1e-6)
        pooled_map = F.adaptive_avg_pool2d(
            log_magnitude, (self.grid_size, self.grid_size)
        ).flatten(1)
        pooled_map = F.normalize(pooled_map, dim=-1)

        radial_profile = self._radial_profile(magnitude.square().squeeze(1))
        radial_profile = torch.sqrt(radial_profile.clamp_min(0.0))
        radial_profile = F.normalize(radial_profile, dim=-1)
        return torch.cat((pooled_map, radial_profile), dim=-1)

    def _extract_chunked(self, pixels: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self._extract(chunk) for chunk in torch.split(
                pixels, self.fft_batch_size, dim=0
            )],
            dim=0,
        )

    def forward(self, normalized_images: torch.Tensor) -> torch.Tensor:
        if normalized_images.ndim != 4 or normalized_images.shape[1] != 3:
            raise ValueError("Expected normalized RGB images with shape [N, 3, H, W].")
        original_device = normalized_images.device
        mean = self.mean.to(device=original_device, dtype=torch.float32)
        std = self.std.to(device=original_device, dtype=torch.float32)
        pixels = (normalized_images.float() * std + mean).clamp(0.0, 1.0)

        if self.fft_device == "cpu" and pixels.is_cuda:
            return self._extract_chunked(pixels.cpu()).to(original_device)
        if self.fft_device == "cuda" and not pixels.is_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("fft_device is cuda, but CUDA is unavailable.")
            return self._extract_chunked(pixels.cuda()).to(original_device)

        try:
            return self._extract_chunked(pixels)
        except RuntimeError as error:
            if not pixels.is_cuda or "CUFFT" not in str(error).upper():
                raise
            if self.fft_device != "auto":
                raise
            if not self._cpu_fallback_warned:
                warnings.warn(
                    "cuFFT failed for the frequency modality; falling back to CPU FFT.",
                    RuntimeWarning,
                )
                self._cpu_fallback_warned = True
            self.fft_device = "cpu"
            torch.cuda.empty_cache()
            return self._extract_chunked(pixels.cpu()).to(original_device)


class FrequencyModalityEncoder(nn.Module):
    """Project handcrafted spectral descriptors into CLIP embedding space."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0 or hidden_dim <= 0:
            raise ValueError("Encoder dimensions must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1).")
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        if descriptors.ndim != 2:
            raise ValueError("Expected frequency descriptors with shape [N, D].")
        return F.normalize(self.network(descriptors.float()), dim=-1)


def compute_modality_prototypes(
    features: torch.Tensor,
    labels: torch.Tensor,
    class_index: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute frequency prototypes, angular uncertainty, and sample counts."""
    if features.ndim != 2:
        raise ValueError("Expected modality features with shape [N, D].")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("Labels must have shape [N] and match features.")

    prototypes = []
    uncertainties = []
    counts = []
    for class_id in class_index:
        mask = labels == int(class_id)
        if not torch.any(mask):
            raise ValueError("No samples found for class {}.".format(class_id))
        class_features = F.normalize(features[mask], dim=-1)
        prototype = F.normalize(class_features.mean(dim=0), dim=-1)
        similarity = class_features @ prototype
        prototypes.append(prototype)
        uncertainties.append(1.0 - similarity.mean())
        counts.append(mask.sum())
    return (
        torch.stack(prototypes),
        torch.stack(uncertainties),
        torch.stack(counts).to(dtype=features.dtype),
    )


def compute_modality_alpha(
    uncertainty: torch.Tensor,
    sample_counts: torch.Tensor,
    max_alpha: float,
    min_alpha: float = 0.0,
    uncertainty_scale: float = 2.0,
    shot_tau: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Downweight noisy and few-shot frequency prototypes."""
    if uncertainty.ndim != 1 or sample_counts.shape != uncertainty.shape:
        raise ValueError("uncertainty and sample_counts must have shape [C].")
    if not 0.0 <= min_alpha <= max_alpha <= 1.0:
        raise ValueError("Alphas must satisfy 0 <= min_alpha <= max_alpha <= 1.")
    if uncertainty_scale < 0.0 or shot_tau < 0.0:
        raise ValueError("Reliability scales must be non-negative.")

    reliability = torch.exp(-uncertainty_scale * uncertainty.clamp_min(0.0))
    if shot_tau > 0.0:
        reliability = reliability * sample_counts / (sample_counts + shot_tau)
    reliability = reliability.clamp(0.0, 1.0)
    alpha = min_alpha + (max_alpha - min_alpha) * reliability
    return alpha, reliability


def fuse_modality_probabilities(
    clip_probabilities: torch.Tensor,
    frequency_probabilities: torch.Tensor,
    class_alpha: torch.Tensor,
) -> torch.Tensor:
    """Late-fuse CLIP image/text evidence with the frequency modality."""
    if clip_probabilities.shape != frequency_probabilities.shape:
        raise ValueError("Both probability tensors must have shape [N, C].")
    if class_alpha.shape != (clip_probabilities.shape[1],):
        raise ValueError("class_alpha must have shape [C].")
    if torch.any(class_alpha < 0.0) or torch.any(class_alpha > 1.0):
        raise ValueError("class_alpha values must lie in [0, 1].")
    alpha = class_alpha.to(
        device=clip_probabilities.device, dtype=clip_probabilities.dtype
    ).unsqueeze(0)
    fused = (
        (1.0 - alpha) * clip_probabilities
        + alpha * frequency_probabilities.to(clip_probabilities.dtype)
    )
    return fused / fused.sum(dim=-1, keepdim=True).clamp_min(1e-12)
