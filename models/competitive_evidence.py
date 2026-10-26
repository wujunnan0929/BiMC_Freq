"""A-fit-only evidence transforms for bounded diagnostic experiments."""
import math

import torch

from models.competitive_fusion import CompetitiveFusion, FEATURE_DIM


REPRESENTATIONS = ('raw', 'standardized', 'prior', 'prior_standardized')


def remove_count_prior(features):
    """Correct evidence inputs, leaving the classifier scores/priors unchanged.

    Recover known group sizes from the existing log1p count features. Replace
    group-mass odds by per-class mean odds; express conditional confidence and
    entropy relative to a uniform class distribution; mask explicit counts.
    """
    x = features.detach().clone()
    old = torch.expm1(5*x[:, 27]).round().clamp_min(1)
    novel = torch.expm1(5*x[:, 28]).round().clamp_min(1)
    prior = (old.log()-novel.log())/6.
    for start in (0, 8, 16):
        x[:, start] = x[:, start]-prior
        for offset, count in ((1, old), (4, novel)):
            x[:, start+offset] = (count*x[:, start+offset]).clamp_min(1e-30).log()
            # Original entropy divides by max(1, log(group size)).
            logn = count.log()
            x[:, start+offset+1] = (logn-x[:, start+offset+1]*logn.clamp_min(1.)).clamp_min(0.)
            x[:, start+offset+2] = torch.log1p(count*x[:, start+offset+2].clamp_min(0.))
    x[:, 27:30] = 0.
    return x


class EvidenceFusion(CompetitiveFusion):
    """Same prediction head/budget; normalizer buffers are fitted on A only."""
    def __init__(self, hidden=16, method='joint', use_support_quality=True, group_limit=2., mix_limit=.5,
                 representation='raw', standardization_floor=1e-4, standardization_clip=5.):
        super().__init__(hidden, method, use_support_quality, group_limit, mix_limit)
        if representation not in REPRESENTATIONS:
            raise ValueError('Unknown diagnostic evidence representation.')
        if not math.isfinite(standardization_floor) or standardization_floor <= 0:
            raise ValueError('standardization_floor must be finite and positive.')
        if not math.isfinite(standardization_clip) or standardization_clip <= 0:
            raise ValueError('standardization_clip must be finite and positive.')
        self.representation = representation
        self.standardization_floor = float(standardization_floor)
        self.standardization_clip = float(standardization_clip)
        self.register_buffer('evidence_mean', torch.zeros(FEATURE_DIM))
        self.register_buffer('evidence_scale', torch.ones(FEATURE_DIM))
        self.statistics_fitted = False

    def before_standardization(self, features):
        return remove_count_prior(features) if self.representation.startswith('prior') else features.detach()

    @torch.no_grad()
    def fit_statistics(self, fitting_features):
        x = self.before_standardization(fitting_features)
        if x.ndim != 2 or x.shape[1] != FEATURE_DIM or len(x) < 2 or not torch.isfinite(x).all():
            raise ValueError('Fit evidence statistics on a finite Nx36 A-fitting matrix.')
        mu = x.double().mean(0)
        std = x.double().std(0, unbiased=False)
        scale = torch.where(std < 1e-8, torch.ones_like(std), std.clamp_min(self.standardization_floor))
        self.evidence_mean.copy_(mu.to(self.evidence_mean))
        self.evidence_scale.copy_(scale.to(self.evidence_scale))
        self.statistics_fitted = True

    def input_features(self, features):
        x = self.before_standardization(features)
        if 'standardized' in self.representation:
            if not self.statistics_fitted:
                raise RuntimeError('Fit the A-only evidence normalizer before standardized inference.')
            x = ((x-self.evidence_mean)/self.evidence_scale).clamp(
                -self.standardization_clip, self.standardization_clip)
        return x.detach()

    def forward(self, features):
        return super().forward(self.input_features(features))

    def diagnostic_configuration(self):
        return {**self.configuration(), 'representation': self.representation,
                'standardization_floor': self.standardization_floor,
                'standardization_clip': self.standardization_clip}
