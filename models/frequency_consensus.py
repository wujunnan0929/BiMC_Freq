"""Bounded, gradient-free visual/semantic evidence on the original top-two pair."""

import math

import torch
from torch.nn import functional as F


MODES = ('visual', 'semantic', 'average', 'consensus')


def pair_candidates(scores):
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError('Expected positive votes [N,C] with C >= 2.')
    if not torch.isfinite(scores).all() or (scores <= 0).any():
        raise ValueError('BiMC votes must be finite and strictly positive, not logits.')
    first = scores.argmax(1)
    remaining = scores.clone()
    remaining.scatter_(1, first[:, None], -torch.inf)
    second = remaining.argmax(1)
    votes = scores.float()
    gap = (votes.gather(1, first[:, None]).log()
           - votes.gather(1, second[:, None]).log()).squeeze(1)
    return first, second, gap


def pairwise_margins(scores, frequency, visual, semantic, permutation=(0, 1, 2)):
    first, second, _ = pair_candidates(scores)
    if sorted(permutation) != [0, 1, 2]:
        raise ValueError('Semantic permutation must contain 0, 1, 2 exactly once.')
    if (frequency.ndim != 3 or frequency.shape[:2] != (len(scores), 3)
            or visual.shape != semantic.shape
            or visual.shape != (scores.shape[1], 3, frequency.shape[-1])):
        raise ValueError('Expected query [N,3,D] and visual/text prototypes [C,3,D].')
    for value in (frequency, visual, semantic):
        if not torch.isfinite(value).all():
            raise ValueError('Features and prototypes must be finite.')
    query = F.normalize(frequency.float(), dim=-1)
    visual = F.normalize(visual.float(), dim=-1)
    semantic = F.normalize(semantic.float(), dim=-1)[:, list(permutation)]
    return ((query * (visual[first] - visual[second])).sum(-1),
            (query * (semantic[first] - semantic[second])).sum(-1))


def fit_margin_scales(visual_margin, semantic_margin, eps=1e-4):
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError('Scale epsilon must be finite and positive.')
    if (visual_margin.shape != semantic_margin.shape or visual_margin.ndim != 2
            or visual_margin.shape[1] != 3 or not len(visual_margin)):
        raise ValueError('Scale fitting needs nonempty matching margins [N,3].')
    raw = torch.stack((visual_margin.abs().median(0).values,
                       semantic_margin.abs().median(0).values)).float()
    if not torch.isfinite(raw).all():
        raise ValueError('Scale-fitting margins must be finite.')
    # A zero scale disables a degenerate source instead of amplifying noise.
    return torch.where(raw >= eps, raw, torch.zeros_like(raw))


def bounded_evidence(visual_margin, semantic_margin, scales, mode='consensus'):
    if mode not in MODES:
        raise ValueError('Unknown evidence mode: ' + str(mode))
    scales = torch.as_tensor(scales, device=visual_margin.device, dtype=torch.float32)
    if scales.shape != (2, 3) or not torch.isfinite(scales).all() or (scales < 0).any():
        raise ValueError('Scales must be finite nonnegative [2,3]; zero disables a source.')
    margins = torch.stack((visual_margin, semantic_margin)).float()
    if margins.ndim != 3 or margins.shape[-1] != 3 or not torch.isfinite(margins).all():
        raise ValueError('Expected finite visual/semantic margins [N,3].')
    scaled = torch.tanh(margins / scales.clamp_min(1e-12)[:, None])
    scaled = torch.where(scales[:, None] > 0, scaled, torch.zeros_like(scaled))
    visual, semantic = scaled.unbind(0)
    if mode == 'visual':
        bands = visual
    elif mode == 'semantic':
        bands = semantic
    elif mode == 'average':
        bands = (visual + semantic) / 2
    else:
        bands = torch.where(visual * semantic > 0,
                            visual.sign() * torch.minimum(visual.abs(), semantic.abs()),
                            torch.zeros_like(visual))
    return bands.mean(1)


def rerank_with_evidence(scores, evidence, strength):
    if not math.isfinite(strength) or strength < 0:
        raise ValueError('Reranking strength must be finite and nonnegative.')
    first, second, gap = pair_candidates(scores)
    if (evidence.shape != gap.shape or not torch.isfinite(evidence).all()
            or (evidence.abs() > 1.000001).any()):
        raise ValueError('Evidence must be finite [N] within [-1,1].')
    margin = gap + strength * evidence
    changed = (gap < strength) & (margin < 0)
    detail = {'first': first, 'second': second, 'log_ratio': gap,
              'evidence': evidence, 'eligible': gap < strength, 'changed': changed}
    if strength == 0 or not changed.any():
        return scores, detail
    output = scores.float().clone()
    rows = torch.where(changed)[0]
    a, b = first[rows], second[rows]
    mass = output[rows, a] + output[rows, b]
    new_a = mass * torch.sigmoid(margin[rows])
    new_b = mass - new_a
    # A tiny negative margin can round sigmoid to .5. Preserve intended winner.
    tied = new_b <= new_a
    new_b = torch.where(tied, torch.nextafter(mass / 2, torch.full_like(mass, torch.inf)), new_b)
    new_a = torch.where(tied, mass - new_b, new_a)
    output[rows, a], output[rows, b] = new_a, new_b
    return output, detail


def rerank_frequency_consensus(scores, frequency, visual, semantic, scales,
                               strength, mode='consensus', permutation=(0, 1, 2)):
    visual_margin, semantic_margin = pairwise_margins(
        scores, frequency, visual, semantic, permutation,
    )
    evidence = bounded_evidence(visual_margin, semantic_margin, scales, mode)
    return rerank_with_evidence(scores, evidence, strength)
