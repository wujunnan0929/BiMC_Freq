"""Frozen CLIP scale alignment and bounded visual/semantic residual evidence.

Semantic bands are LLM descriptions of spatial scales, not a Fourier transform
of text coordinates. All class statistics/description selection use support;
query batches never fit a prototype, scale, weight, or hyperparameter.
"""

import math

import torch
from torch.nn import functional as F

from models.frequency_consensus import (
    MODES, pair_candidates, rerank_with_evidence,
)
from models.frequency_uncertainty import class_statistics


def remove_global(values, global_features, enabled=True, eps=1e-6):
    """Remove each sample/class's global direction; preserve zero residuals."""
    if (values.ndim < 3 or global_features.ndim != 2
            or values.shape[0] != global_features.shape[0]
            or values.shape[-1] != global_features.shape[-1]
            or not torch.isfinite(values).all() or not torch.isfinite(global_features).all()):
        raise ValueError('Expected finite aligned scale/global features.')
    values = F.normalize(values.detach().float(), dim=-1)
    if enabled:
        global_features = F.normalize(global_features.detach().float(), dim=-1)
        direction = global_features.reshape(len(values), *([1] * (values.ndim - 2)), -1)
        values = values - (values * direction).sum(-1, keepdim=True) * direction
    energy = values.norm(dim=-1)
    # Do not normalize floating-point cancellation into a confident direction.
    values = torch.where((energy > eps).unsqueeze(-1),
                         F.normalize(values, dim=-1), torch.zeros_like(values))
    return values, energy


@torch.no_grad()
def support_statistics(original, frequency, labels, class_ids, residualize=True):
    residual, _ = remove_global(frequency, original, residualize)
    return class_statistics(residual, labels, class_ids)


@torch.no_grad()
def align_semantics(stats, candidates, global_text, *, top_k=3, temperature=.07,
                    shot_tau=5., residualize=True, enabled=True,
                    permutation=(0, 1, 2)):
    """Ground candidates by own-class versus strongest competing support.

Same-class cross-band confusion is also penalized. Reliability combines
positive cross-modal agreement, separation, mean concentration and shot count.
It is an attenuation in [0,1], never a learned router or a probability claim.
"""
    if (candidates.ndim != 4 or candidates.shape[1] != 3 or candidates.shape[2] < 1
            or stats['mean'].shape != (candidates.shape[0], 3, candidates.shape[-1])
            or candidates.shape[0] < 2 or sorted(permutation) != [0, 1, 2]):
        raise ValueError('Alignment needs candidates [C,3,K,D], statistics [C,3,D], C>=2.')
    if (not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1
            or not math.isfinite(temperature) or temperature <= 0
            or not math.isfinite(shot_tau) or shot_tau < 0):
        raise ValueError('Invalid description selection or shot reliability settings.')
    text, _ = remove_global(candidates[:, list(permutation)], global_text, residualize)
    visual = F.normalize(stats['mean'].float(), dim=-1)
    concentration = stats['mean'].float().norm(dim=-1).clamp(0, 1)
    # [target class, band, description candidate, competing visual class]
    scores = torch.einsum('cbkd,jbd->cbkj', text, visual)
    own = (text * visual[:, :, None]).sum(-1)
    mask = torch.eye(len(visual), device=visual.device, dtype=torch.bool)[:, None, None]
    competitor = scores.masked_fill(mask, -torch.inf).amax(-1)
    cross = torch.einsum('cbkd,cad->cbka', text, visual)
    band_mask = torch.eye(3, device=visual.device, dtype=torch.bool)[None, :, None]
    cross_competitor = cross.masked_fill(band_mask, -torch.inf).amax(-1)
    quality = own - torch.maximum(competitor, cross_competitor)
    if enabled:
        selected_quality, indices = quality.topk(min(top_k, text.shape[2]), dim=2)
        selected = text.gather(2, indices[..., None].expand(-1, -1, -1, text.shape[-1]))
        weights = F.softmax(selected_quality / temperature, dim=2)
        semantic = F.normalize((selected * weights[..., None]).sum(2), dim=-1)
    else:
        indices = torch.arange(text.shape[2], device=text.device).expand(*text.shape[:2], -1)
        semantic = F.normalize(text.mean(2), dim=-1)
    similarity = torch.einsum('cbd,jbd->cbj', semantic, visual)
    agreement = (semantic * visual).sum(-1)
    competing = similarity.masked_fill(mask.squeeze(2), -torch.inf).amax(-1)
    cross_similarity = torch.einsum('cbd,cad->cba', semantic, visual)
    mismatched = cross_similarity.masked_fill(band_mask.squeeze(2), -torch.inf).amax(-1)
    class_margin, band_margin = agreement - competing, agreement - mismatched
    counts = stats['count'].float()[:, None]
    shot_reliability = counts / (counts + shot_tau)
    reliability = (agreement.clamp(0, 1) * concentration * shot_reliability
                   * (2 * torch.sigmoid(class_margin / temperature)).clamp(max=1)
                   * (2 * torch.sigmoid(band_margin / temperature)).clamp(max=1)
                   if enabled else torch.ones_like(agreement))
    return {'visual': visual, 'semantic': semantic, 'reliability': reliability,
            'visual_reliability': concentration * shot_reliability if enabled else torch.ones_like(agreement),
            'agreement': agreement, 'class_margin': class_margin, 'band_margin': band_margin,
            'concentration': concentration, 'selected_indices': indices}


@torch.no_grad()
def pair_margins(anchor, original, frequency, aligned, residualize=True, mode='consensus'):
    first, second, _ = pair_candidates(anchor)
    query, energy = remove_global(frequency, original, residualize)
    if aligned['visual'].shape != (anchor.shape[1], 3, query.shape[-1]):
        raise ValueError('Anchor classes and aligned prototypes do not match.')
    visual = (query * (aligned['visual'][first] - aligned['visual'][second])).sum(-1)
    semantic = (query * (aligned['semantic'][first] - aligned['semantic'][second])).sum(-1)
    class_reliability = aligned['visual_reliability'] if mode == 'visual' else aligned['reliability']
    reliability = torch.minimum(class_reliability[first], class_reliability[second])
    # Tiny view changes provide no reliable residual evidence.
    reliability = reliability * (energy / .02).clamp(0, 1)
    return visual, semantic, reliability


def frequency_evidence(visual, semantic, reliability, scales, mode='consensus', bands=(1., 1., 1.)):
    if mode not in MODES:
        raise ValueError('Unknown dual-frequency evidence mode.')
    scales = torch.as_tensor(scales, device=visual.device, dtype=torch.float32)
    bands = torch.as_tensor(bands, device=visual.device, dtype=torch.float32)
    if (scales.shape != (2, 3) or (scales < 0).any() or not torch.isfinite(scales).all()
            or bands.shape != (3,) or (bands < 0).any() or not torch.isfinite(bands).all()
            or bands.sum() <= 0 or visual.shape != semantic.shape
            or visual.ndim != 2 or visual.shape[1] != 3 or reliability.shape != visual.shape
            or not all(torch.isfinite(value).all() for value in (visual, semantic, reliability))
            or (reliability < 0).any() or (reliability > 1).any()):
        raise ValueError('Invalid margins, reliability, scales or band weights.')
    margins = torch.stack((visual, semantic)).float()
    bounded = torch.where(scales[:, None] > 0,
                          torch.tanh(margins / scales.clamp_min(1e-12)[:, None]),
                          torch.zeros_like(margins))
    v, t = bounded.unbind(0)
    if mode == 'visual':
        evidence = v
    elif mode == 'semantic':
        evidence = t
    elif mode == 'average':
        evidence = (v + t) / 2
    else:
        evidence = torch.where(v * t > 0, v.sign() * torch.minimum(v.abs(), t.abs()),
                               torch.zeros_like(v))
    # Fixed denominator prevents weak/absent bands being renormalized to confidence.
    return (evidence * reliability * (bands / bands.sum())).sum(-1)


def apply_evidence(anchor, original, frequency, aligned, scales, strength, *,
                   residualize=True, mode='consensus', bands=(1., 1., 1.)):
    visual, semantic, reliability = pair_margins(anchor, original, frequency, aligned, residualize, mode)
    evidence = frequency_evidence(visual, semantic, reliability, scales, mode, bands)
    return rerank_with_evidence(anchor, evidence, strength)
