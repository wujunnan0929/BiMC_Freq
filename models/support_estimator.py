"""Class-agnostic support-conditioned prototype estimation on frozen features.

The small networks see scalar geometry, never class indices or learned feature
coordinates. They may be fitted on base episodes, then frozen for incremental
inference. A bounded prototype movement is not an accuracy guarantee.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def _real(value, name, *, device, dtype):
    try:
        tensor = torch.as_tensor(value, device=device).detach()
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError('Invalid ' + name) from error
    if tensor.is_complex() or tensor.dtype == torch.bool or not torch.isfinite(tensor).all():
        raise ValueError(name + ' must contain finite real values.')
    return tensor.to(dtype=dtype)


def _unit(value, name):
    if (value.norm(dim=-1) <= 1e-12).any():
        raise ValueError(name + ' must contain nonzero feature vectors.')
    return F.normalize(value, dim=-1)


def _blend(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or value > 1):
        raise ValueError('blend must be a finite number in [0,1].')
    return float(value)


class SupportPrototypeEstimator(nn.Module):
    """Predict bounded corrections using support quality and competitors.

    All input tensors are detached. Gradients enter only the shared MLPs. The
    sample head predicts normalized weights; its uniform component is removed
    before aggregating samples. The class head controls a bounded movement
    toward or away from text. Zero-initialized final layers give an exactly
    zero pre-normalization delta with useful output-layer gradients.
    """

    sample_feature_dim = 12
    class_feature_dim = 7

    def __init__(self, hidden=32, max_delta=.15, use_competitors=True):
        super().__init__()
        if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden < 1:
            raise ValueError('hidden must be a positive integer.')
        if (isinstance(max_delta, bool) or not isinstance(max_delta, (int, float))
                or not math.isfinite(max_delta) or not 0 < max_delta < 1):
            raise ValueError('max_delta must be finite and strictly between zero and one.')
        if not isinstance(use_competitors, bool):
            raise ValueError('use_competitors must be boolean.')
        self.hidden = hidden
        self.max_delta = float(max_delta)
        self.use_competitors = use_competitors
        self.sample_head = nn.Sequential(nn.Linear(self.sample_feature_dim, hidden),
                                         nn.Tanh(), nn.Linear(hidden, hidden),
                                         nn.Tanh(), nn.Linear(hidden, 1))
        self.class_head = nn.Sequential(nn.Linear(self.class_feature_dim, hidden),
                                        nn.Tanh(), nn.Linear(hidden, 1))
        for layer in (self.sample_head[-1], self.class_head[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def configuration(self):
        return {'hidden': self.hidden, 'max_delta': self.max_delta,
                'use_competitors': self.use_competitors}

    def _inputs(self, support, reference, text, competitors):
        parameter = next(self.parameters())
        kwargs = {'device': parameter.device, 'dtype': parameter.dtype}
        x = _real(support, 'support', **kwargs)
        ref = _real(reference, 'reference', **kwargs)
        semantic = _real(text, 'text', **kwargs)
        if x.ndim != 2 or min(x.shape) < 1:
            raise ValueError('support must be nonempty [N,D].')
        if ref.shape != (x.shape[1],) or semantic.shape != ref.shape:
            raise ValueError('reference/text must be [D] matching support.')
        x, ref, semantic = (_unit(x, 'support'), _unit(ref, 'reference'),
                            _unit(semantic, 'text'))
        if competitors is None:
            other = x.new_empty(0, x.shape[1])
        else:
            other = _real(competitors, 'competitors', **kwargs)
            if other.ndim != 2 or other.shape[1] != x.shape[1]:
                raise ValueError('competitors must be [K,D] matching support.')
            other = _unit(other, 'competitors')
        return x, ref, semantic, other

    def _geometry(self, x, ref, text, competitors):
        count = len(x)
        raw = _unit(x.mean(0), 'support mean')
        residual = x - raw
        distances = residual.square().sum(1)
        dispersion = distances.mean()
        if count > 1:
            # A cancelling leave-one-out mean is permitted: it represents an
            # undefined direction, rather than invalidating otherwise valid S.
            leave_out = F.normalize(x.sum(0, keepdim=True) - x, dim=-1)
            leave_out_variation = (x - leave_out).square().sum(1)
        else:
            leave_out_variation = distances.new_zeros(count)
        zeros = x.new_zeros(count)
        class_competitor = x.new_zeros(())
        sample_competitor, semantic_gap = zeros, zeros
        if self.use_competitors and len(competitors):
            sample_competitor = (x @ competitors.T).max(1).values
            class_competitor = (competitors @ raw).max()
            semantic_gap = x @ text - sample_competitor
        shot = x.new_tensor(math.log1p(count) / math.log(501.))
        align = raw @ text
        ref_align = ref @ text
        repeat = lambda value: value.expand(count)
        features = torch.stack((x @ raw, x @ ref, x @ text, distances,
            leave_out_variation, sample_competitor, semantic_gap,
            repeat(class_competitor), repeat(dispersion), repeat(align),
            repeat(shot), repeat(ref_align)), dim=1)
        class_features = torch.stack((dispersion, leave_out_variation.mean(),
            align, raw @ ref, shot, class_competitor, ref_align))
        return raw, features, class_features

    def forward(self, support, reference, text, competitors=None, *, return_details=False):
        x, ref, semantic, other = self._inputs(support, reference, text, competitors)
        raw, features, class_features = self._geometry(x, ref, semantic, other)
        logits = self.sample_head(features).squeeze(-1)
        weights = logits.softmax(0)
        uniform = weights.new_full(weights.shape, 1. / len(x))
        sample_delta = ((weights - uniform)[:, None] * x).sum(0)
        text_coefficient = self.class_head(class_features).squeeze(-1).tanh()
        delta = sample_delta + text_coefficient * (semantic - raw)
        scale = (self.max_delta / delta.norm().clamp_min(1e-12)).clamp(max=1.)
        delta = delta * scale
        corrected = F.normalize(ref + delta, dim=-1)
        if not torch.isfinite(corrected).all():
            raise ValueError('Support estimator produced non-finite output.')
        if return_details:
            return corrected, {'delta': delta, 'sample_weights': weights,
                               'text_coefficient': text_coefficient,
                               'raw_prototype': raw}
        return corrected


def correct_current(model, supports, reference, text, past_prototypes=None, blend=1.):
    """Correct one arrival jointly, keeping all competitor inputs fixed.

    Past prototypes and the unmodified current references form the context.
    No correction from another current class is fed back within this call.
    Reordering current classes therefore only reorders outputs. The caller
    saves these outputs at arrival and need not retain individual supports.
    Zero blend returns the original reference tensor object exactly.
    """
    blend = _blend(blend)
    if not torch.is_tensor(reference):
        raise ValueError('reference must be a tensor [C,D].')
    parameter = next(model.parameters())
    kwargs = {'device': parameter.device, 'dtype': parameter.dtype}
    ref = _real(reference, 'reference', **kwargs)
    if ref.ndim != 2 or min(ref.shape) < 1:
        raise ValueError('reference must be nonempty [C,D].')
    ref = _unit(ref, 'reference')
    if blend == 0:
        return reference
    semantic = _real(text, 'text', **kwargs)
    if semantic.shape != ref.shape:
        raise ValueError('text must match current reference [C,D].')
    semantic = _unit(semantic, 'text')
    if not isinstance(supports, (tuple, list)) or len(supports) != len(ref):
        raise ValueError('supports must contain one [N,D] tensor per current class.')
    if past_prototypes is None:
        past = ref.new_empty(0, ref.shape[1])
    else:
        past = _real(past_prototypes, 'past_prototypes', **kwargs)
        if past.ndim != 2 or past.shape[1] != ref.shape[1]:
            raise ValueError('past_prototypes must be [K,D] matching reference.')
        past = _unit(past, 'past_prototypes')
    context = torch.cat((past, ref), dim=0)
    results = []
    for j, support in enumerate(supports):
        position = len(past) + j
        competitors = torch.cat((context[:position], context[position + 1:]), dim=0)
        estimated = model(support, ref[j], semantic[j], competitors)
        results.append(F.normalize((1 - blend) * ref[j] + blend * estimated, dim=-1))
    return torch.stack(results)
