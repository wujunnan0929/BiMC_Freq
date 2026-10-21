"""Nested adaptive image interventions for reusable frozen-CLIP features.

SAFD produces nested approximation spaces. They are not disjoint radial
frequency bands, and this module makes no causal-identification claim.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


SOURCES = ('safd', 'fft', 'augmentation', 'repeat')
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class NestedInterventionViews(nn.Module):
    """Return original/coarse/intermediate images, in that order.

    Ordinary augmentations are a compute-budget control, not a claim of
    distortion matching to the adaptive or FFT interventions.
    """

    def __init__(self, source='safd', frequency_cfg=None, *, low_terms=8,
                 middle_terms=24, radii=(0., .3, .6, .8), angles=16,
                 max_signals=96, batch_size=2, low_cutoff=.18,
                 high_cutoff=.45, fft_device='auto', fft_batch_size=8,
                 mean=CLIP_MEAN, std=CLIP_STD):
        super().__init__()
        if source not in SOURCES:
            raise ValueError('Unknown intervention source: ' + str(source))
        if len(mean) != 3 or len(std) != 3 or any(not math.isfinite(v) for v in (*mean, *std)) or min(std) <= 0:
            raise ValueError('Expected finite RGB mean/std with positive std.')
        self.source = source
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
        if frequency_cfg is not None:
            settings = frequency_cfg.SAFD
            low_terms, middle_terms = settings.LOW_TERMS, settings.MIDDLE_TERMS
            radii, angles = settings.RADII, settings.ANGLES
            max_signals, batch_size = settings.MAX_SIGNALS, settings.BATCH_SIZE
            low_cutoff, high_cutoff = frequency_cfg.LOW_CUTOFF, frequency_cfg.HIGH_CUTOFF
            fft_device, fft_batch_size = frequency_cfg.FFT_DEVICE, frequency_cfg.FFT_BATCH_SIZE
        self.settings = {'low_terms': int(low_terms), 'middle_terms': int(middle_terms),
                         'radii': list(radii), 'angles': int(angles),
                         'max_signals': int(max_signals), 'safd_batch_size': int(batch_size),
                         'low_cutoff': float(low_cutoff), 'high_cutoff': float(high_cutoff),
                         'fft_device': fft_device, 'fft_batch_size': int(fft_batch_size)}
        if source == 'safd':
            from models.safd import SAFDFrequencyDecomposer
            self.decomposer = SAFDFrequencyDecomposer(
                low_terms=low_terms, middle_terms=middle_terms, radii=radii,
                angles=angles, max_signals=max_signals, batch_size=batch_size,
                mean=mean, std=std)
        elif source == 'fft':
            from models.frequency import RadialFrequencyDecomposer
            self.decomposer = RadialFrequencyDecomposer(
                low_cutoff, high_cutoff, mean=mean, std=std,
                fft_device=fft_device, fft_batch_size=fft_batch_size)
        else:
            self.decomposer = None

    def specification(self):
        names = {'safd': ['original', 'safd_coarse', 'safd_intermediate'],
                 'fft': ['original', 'fft_low', 'fft_low_middle'],
                 'augmentation': ['original', 'blur_7x7', 'center_crop_90pct'],
                 'repeat': ['original', 'original_repeat_1', 'original_repeat_2']}
        return {'source': self.source, 'view_order': names[self.source],
                'construction': ('Clamp RGB L and L+M separately; no original blend or high enhancement.'
                                 if self.source in ('safd', 'fft') else
                                 '7x7 replicate-pad average blur and bilinear center-crop resize (90%).'
                                 if self.source == 'augmentation' else 'Exact copies; auxiliary encoding is skipped.'),
                'settings': self.settings,
                'response_order': ['intermediate_minus_coarse', 'original_minus_intermediate'],
                'interpretation': 'SAFD scales are adaptive approximation spaces, not fixed physical Fourier bands.'}

    @torch.no_grad()
    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3 or min(images.shape[-2:]) < 2:
            raise ValueError('Expected normalized RGB [N,3,H,W] with H,W >= 2.')
        if not images.is_floating_point() or not torch.isfinite(images).all():
            raise ValueError('Images must be finite floating-point tensors.')
        if self.source == 'repeat':
            return images[:, None].expand(-1, 3, -1, -1, -1).clone()
        mean, std = self.mean.to(images.device), self.std.to(images.device)
        pixels = images.float() * std + mean
        if not len(images):
            return images[:, None].expand(-1, 3, -1, -1, -1).clone()
        if self.decomposer is not None:
            components = self.decomposer.split_pixels(pixels)
            coarse, intermediate = components[:, 0], components[:, 0] + components[:, 1]
        else:
            coarse = F.avg_pool2d(F.pad(pixels, (3, 3, 3, 3), mode='replicate'), 7, stride=1)
            h, w = pixels.shape[-2:]
            ch, cw = max(1, round(.9 * h)), max(1, round(.9 * w))
            top, left = (h - ch) // 2, (w - cw) // 2
            intermediate = F.interpolate(pixels[:, :, top:top+ch, left:left+cw],
                                         (h, w), mode='bilinear', align_corners=False, antialias=True)
        auxiliary = torch.stack((coarse, intermediate), 1).clamp(0, 1)
        auxiliary = (auxiliary - mean[:, None]) / std[:, None]
        # Retain the original input exactly, including dtype/roundoff.
        return torch.cat((images[:, None], auxiliary.to(images.dtype)), 1)

    @torch.no_grad()
    def pixel_rmse(self, views):
        return ((views[:, 1:].float() - views[:, :1].float()) * self.std.to(views.device)[:, None]).square().mean((2, 3, 4)).sqrt()


@torch.no_grad()
def encode_intervention_features(images, views, original_encoder, auxiliary_encoder=None):
    """Encode three feature positions; repeated originals require one call."""
    transformed = views(images)
    auxiliary_encoder = auxiliary_encoder or original_encoder
    def normalized(value):
        value = value.float()
        if value.ndim != 2 or len(value) != len(images) or not torch.isfinite(value).all() or (value.norm(dim=-1) <= 1e-12).any():
            raise ValueError('Encoder must return nonzero finite [N,D] features.')
        return F.normalize(value, dim=-1)
    original = normalized(original_encoder(transformed[:, 0]))
    if views.source == 'repeat':
        features = original[:, None].expand(-1, 3, -1).clone()
    else:
        features = torch.stack((original, normalized(auxiliary_encoder(transformed[:, 1])),
                                normalized(auxiliary_encoder(transformed[:, 2]))), 1)
    return features, views.pixel_rmse(transformed)


def counterfactual_responses(features):
    """Return two signed CLIP responses; no per-response unit normalization."""
    if features.ndim != 3 or features.shape[1] != 3 or not torch.isfinite(features).all():
        raise ValueError('Expected finite features [N,3,D].')
    return torch.stack((features[:, 2] - features[:, 1],
                        features[:, 0] - features[:, 2]), 1)
