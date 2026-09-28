"""Discrete stochastic AFD image views, without FFT or a learned backbone.

Szego kernels, stochastic maximum selection and the generalized backward
shift follow Qian et al., https://arxiv.org/abs/2205.00844. Each image supplies
an ensemble of RGB scanlines; poles are shared across those signals, never
across unrelated query images. The separable image projection and view
construction are adaptations for BiMC, not a reproduction of SAFDNet.
"""

import math
from collections import OrderedDict

import torch
from torch import nn


class SAFDFrequencyDecomposer(nn.Module):
    """Construct coarse, intermediate and detail views from nested SAFD spans.

    These are adaptive approximation scales, not disjoint radial frequencies.
    A fixed DC atom preserves color. Subsequent poles maximize mean squared
    coefficients over scanlines of one image. Real/imaginary TM atoms are
    reorthogonalized on the finite grid before separable spatial projection.
    No query labels, batch statistics, optimizer, or dataset fitting is used.
    """

    def __init__(
        self, low_terms=8, middle_terms=24, radii=(0.0, 0.3, 0.6, 0.8),
        angles=16, max_signals=96, batch_size=2,
        view_mode="natural", high_enhance=0.25,
        original_blend=(0.25, 0.1), max_detail_rms=0.04,
        center_residual_bands=True,
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ):
        super().__init__()
        for name, value in (("low_terms", low_terms), ("middle_terms", middle_terms),
                            ("angles", angles), ("max_signals", max_signals),
                            ("batch_size", batch_size)):
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(name + " must be a positive integer.")
        if low_terms >= middle_terms:
            raise ValueError("SAFD requires 0 < low_terms < middle_terms.")
        if not radii or any(not math.isfinite(r) or not 0 <= r <= 0.95 for r in radii):
            raise ValueError("SAFD radii must be finite and lie in [0, 0.95] for discrete stability.")
        if len(set(radii)) != len(radii):
            raise ValueError("SAFD radii must be distinct.")
        if len(original_blend) != 2 or any(
                not math.isfinite(x) or not 0 <= x <= 1 for x in original_blend):
            raise ValueError("SAFD original_blend needs two values in [0, 1].")
        if not math.isfinite(high_enhance) or high_enhance < 0:
            raise ValueError("high_enhance must be finite and nonnegative.")
        if not math.isfinite(max_detail_rms) or max_detail_rms <= 0:
            raise ValueError("max_detail_rms must be finite and positive.")
        if view_mode not in ("natural", "disjoint", "original"):
            raise ValueError("Unknown SAFD view mode: " + view_mode)
        if len(mean) != 3 or len(std) != 3 or any(x <= 0 for x in std):
            raise ValueError("Expected RGB mean/std with positive std.")
        self.low_terms, self.middle_terms = int(low_terms), int(middle_terms)
        self.radii, self.angles = tuple(radii), int(angles)
        self.max_signals, self.batch_size = int(max_signals), int(batch_size)
        self.view_mode, self.high_enhance = view_mode, float(high_enhance)
        self.original_blend = tuple(original_blend)
        self.max_detail_rms = float(max_detail_rms)
        self.center_residual_bands = bool(center_residual_bands)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        self._grids = OrderedDict()

    @property
    def num_bands(self):
        return 3

    def _apply(self, fn):
        self._grids.clear()
        return super()._apply(fn)

    def _grid(self, length, device):
        key = (length, device)
        if key in self._grids:
            self._grids.move_to_end(key)
            return self._grids[key]
        # Construct the discrete periodic Hilbert transform directly. For
        # row vectors x, x @ H is the imaginary part of the analytic signal.
        # DC and even-length Nyquist have zero Hilbert response.
        theta = torch.arange(length, device=device, dtype=torch.float64) * (2 * math.pi / length)
        frequencies = torch.arange(1, (length + 1) // 2, device=device, dtype=torch.float64)
        phase = theta[:, None] * frequencies[None, :]
        c, s = phase.cos(), phase.sin()
        hilbert = (2 / length * (c @ s.T - s @ c.T)).float()
        z = torch.polar(torch.ones_like(theta), theta).to(torch.complex64)
        poles = []
        for radius in self.radii:
            count = 1 if radius == 0 else self.angles
            for angle in range(count):
                poles.append(radius * complex(math.cos(2 * math.pi * angle / count),
                                              math.sin(2 * math.pi * angle / count)))
        poles = torch.tensor(poles, device=device, dtype=torch.complex64)
        kernels = (1 - poles.abs().square()).sqrt()[:, None] / (1 - poles.conj()[:, None] * z)
        # Discrete normalization avoids bias toward poorly sampled near-boundary poles.
        kernels = kernels / kernels.abs().square().mean(-1, keepdim=True).sqrt()
        value = z, poles, kernels, hilbert
        self._grids[key] = value
        if len(self._grids) > 4:
            self._grids.popitem(last=False)
        return value

    def _axis_basis(self, signals):
        """Return nested real bases [N, length, rank] for an image's scanlines."""
        batch, count, length = signals.shape
        z, poles, kernels, hilbert = self._grid(length, signals.device)
        if count > self.max_signals:
            indices = torch.linspace(0, count - 1, self.max_signals,
                                     device=signals.device).round().long()
            signals = signals[:, indices]
        analytic = torch.complex(signals, signals @ hilbert)
        # a_0 = 0 fixes the constant atom; g is its backward-shift remainder.
        g = (analytic - analytic.mean(-1, keepdim=True)) / z
        product = z.expand(batch, -1).clone()
        atoms = []
        terms = min(self.middle_terms, (length - 1) // 2)
        for _ in range(terms):
            coefficients = (g @ kernels.conj().T) / length
            scores = coefficients.abs().square().mean(1)
            selected = scores.argmax(-1)
            pole, kernel = poles[selected], kernels[selected]
            coefficient = coefficients.gather(
                2, selected[:, None, None].expand(-1, g.shape[1], 1))
            atoms.append(product * kernel)
            factor = (z - pole[:, None]) / (1 - pole.conj()[:, None] * z)
            g = (g - coefficient * kernel[:, None]) / factor[:, None]
            product = product * factor

        # Discrete TM atoms need numerical orthogonalization (the continuous
        # orthogonality identity is not exact on a small pixel grid). Two-pass
        # Gram-Schmidt drops dependent directions instead of filling them with
        # arbitrary QR columns. Inserting DC first makes all details zero-mean.
        basis = signals.new_full((batch, length, 1), 1 / math.sqrt(length))
        coarse = basis
        for index, atom in enumerate(atoms):
            for part in (atom.real, atom.imag):
                vector = part.unsqueeze(-1)
                for _ in range(2):
                    vector = vector - basis @ (basis.transpose(1, 2) @ vector)
                norm = vector.norm(dim=1, keepdim=True)
                vector = torch.where(norm > 1e-5 * math.sqrt(length),
                                     vector / norm.clamp_min(1e-12), torch.zeros_like(vector))
                basis = torch.cat((basis, vector), dim=-1)
            if index + 1 == min(self.low_terms, terms):
                coarse = basis
        return coarse, basis

    @staticmethod
    def _project(pixels, vertical, horizontal):
        # The same per-image poles are used for every color and both levels.
        coefficients = vertical[:, None].transpose(-1, -2) @ pixels @ horizontal[:, None]
        return vertical[:, None] @ coefficients @ horizontal[:, None].transpose(-1, -2)

    @staticmethod
    def _validate(pixels):
        if pixels.ndim != 4 or pixels.shape[1] != 3 or min(pixels.shape[-2:]) < 2:
            raise ValueError("SAFD expects RGB [N, 3, H, W] with H, W >= 2.")
        if not pixels.is_floating_point():
            raise ValueError("SAFD expects floating-point pixels.")
        if not torch.isfinite(pixels).all():
            raise ValueError("SAFD pixels must be finite.")

    @torch.no_grad()
    def split_pixels(self, pixels):
        """Return FP32 components [N, 3, C, H, W] summing to the input."""
        self._validate(pixels)
        if pixels.shape[0] == 0:
            return pixels.float().unsqueeze(1).expand(-1, 3, -1, -1, -1).clone()
        chunks = []
        # Explicitly disable external autocast: small projection differences
        # affect greedy pole selection and must not be computed in fp16.
        with torch.autocast(device_type=pixels.device.type, enabled=False):
            for chunk in pixels.float().split(self.batch_size):
                n, c, h, w = chunk.shape
                horizontal = self._axis_basis(chunk.reshape(n, c * h, w))
                vertical = self._axis_basis(chunk.transpose(-1, -2).reshape(n, c * w, h))
                low = self._project(chunk, vertical[0], horizontal[0])
                middle = self._project(chunk, vertical[1], horizontal[1])
                chunks.append(torch.stack((low, middle - low, chunk - middle), dim=1))
        return torch.cat(chunks)

    @torch.no_grad()
    def forward(self, normalized_images):
        self._validate(normalized_images)
        if self.view_mode == "original":
            return normalized_images[:, None].expand(-1, 3, -1, -1, -1).clone()
        dtype = normalized_images.dtype
        mean = self.mean.to(device=normalized_images.device, dtype=torch.float32)
        std = self.std.to(device=normalized_images.device, dtype=torch.float32)
        pixels = normalized_images.float() * std + mean
        components = self.split_pixels(pixels)
        if self.view_mode == "disjoint":
            views = components.clone()
            if self.center_residual_bands:
                views[:, 1:] += pixels.mean((-2, -1), keepdim=True)[:, None]
        else:
            low = torch.lerp(components[:, 0], pixels, self.original_blend[0])
            middle = torch.lerp(components[:, 0] + components[:, 1], pixels, self.original_blend[1])
            detail = self.high_enhance * components[:, 2]
            rms = detail.square().mean((-2, -1), keepdim=True).sqrt()
            detail = detail * (self.max_detail_rms / rms.clamp_min(1e-12)).clamp(max=1)
            views = torch.stack((low, middle, pixels + detail), dim=1)
        return ((views.clamp(0, 1) - mean[:, None]) / std[:, None]).to(dtype)
