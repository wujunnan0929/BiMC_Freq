"""Support-only relations for training-free visual prototype guidance.

All distributions are over the original base bank, never over query images.
Scale fusion is arithmetic *after* within-scale normalization, so pairing is
observable. No parameters or image samples are retained by this module.
"""

import math

import torch
import torch.nn.functional as F


def validate_settings(cfg):
    b, g = cfg.TRAINER.BiMC, cfg.TRAINER.BiMC.GUIDANCE
    if not g.ENABLED:
        return
    if (b.FREQUENCY.ENABLED or b.FREQUENCY.ROUTER.ENABLED or b.CONSENSUS.ENABLED
            or b.UNCERTAINTY.ENABLED or b.DUAL_FREQUENCY.ENABLED or b.RESIDUAL.ENABLED):
        raise ValueError('GUIDANCE requires legacy frequency fusion, router, GDA, dual, consensus and residual disabled.')
    if not b.VISION_CALIBRATION:
        raise ValueError('GUIDANCE requires VISION_CALIBRATION=True; use G0 for the disabled control.')
    if g.MODE not in ('visual', 'marginal', 'paired'):
        raise ValueError('GUIDANCE.MODE must be visual, marginal, or paired.')
    if g.VIEW_CONTROL not in ('frequency', 'repeat', 'augmentation'):
        raise ValueError('GUIDANCE.VIEW_CONTROL must be frequency, repeat, or augmentation.')
    if g.SEMANTIC_SOURCE not in ('explicit', 'ordinary', 'uniform'):
        raise ValueError('GUIDANCE.SEMANTIC_SOURCE must be explicit, ordinary, or uniform.')
    if sorted(g.SEMANTIC_PERMUTATION) != [0, 1, 2]:
        raise ValueError('GUIDANCE.SEMANTIC_PERMUTATION must be a permutation of [0,1,2].')
    for name, value in [('RHO', g.RHO), ('ETA', g.ETA), ('LAMBDA_I', b.LAMBDA_I)]:
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(name + ' must be finite in [0,1].')
    if not math.isfinite(g.TAU_TEXT) or g.TAU_TEXT <= 0 or b.TAU <= 0:
        raise ValueError('Guidance temperatures must be positive and finite.')
    if (len(g.BAND_WEIGHTS) != 3 or any(not math.isfinite(v) or v < 0 for v in g.BAND_WEIGHTS)
            or sum(g.BAND_WEIGHTS) <= 0):
        raise ValueError('GUIDANCE.BAND_WEIGHTS needs three nonnegative weights with positive sum.')
    if (not g.RHO_GRID or 0 not in g.RHO_GRID
            or any(not math.isfinite(v) or not 0 <= v <= 1 for v in g.RHO_GRID)):
        raise ValueError('GUIDANCE.RHO_GRID must include zero and finite values in [0,1].')
    for name in ('VAL_SEQUENCES', 'SHOT', 'QUERY', 'AUDIT_IMAGES', 'CANDIDATES_PER_BAND'):
        if getattr(g, name) < 1:
            raise ValueError('GUIDANCE.' + name + ' must be positive.')
    if g.SHOT != cfg.DATASET.NUM_INC_SHOT:
        raise ValueError('GUIDANCE.SHOT must match DATASET.NUM_INC_SHOT.')
    if not math.isfinite(g.MAX_GROUP_DROP_PP) or g.MAX_GROUP_DROP_PP < 0:
        raise ValueError('GUIDANCE.MAX_GROUP_DROP_PP must be finite and nonnegative.')


@torch.no_grad()
def relation_distributions(base_visual, novel_visual, base_bands, novel_bands,
                           base_text=None, novel_text=None, *, mode='paired',
                           tau=16., tau_text=16., eta=.5, permutation=(0, 1, 2),
                           band_weights=(1., 1., 1.), uniform_semantics=False):
    base_visual = F.normalize(base_visual.float(), dim=-1)
    novel_visual = F.normalize(novel_visual.float(), dim=-1)
    base_bands = F.normalize(base_bands.float(), dim=-1)
    novel_bands = F.normalize(novel_bands.float(), dim=-1)
    if (base_visual.ndim != 2 or novel_visual.ndim != 2 or not len(base_visual)
            or base_bands.shape != (len(base_visual), 3, base_visual.shape[-1])
            or novel_bands.shape != (len(novel_visual), 3, base_visual.shape[-1])):
        raise ValueError('Guidance expects visual [classes,D] and bands [classes,3,D].')
    a0 = (tau * novel_visual @ base_visual.T).softmax(-1)
    # [novel, scales, base]
    log_p = (tau * torch.einsum('csd,bsd->csb', novel_bands, base_bands)).log_softmax(-1)
    p = log_p.exp()
    w = p.new_tensor(band_weights)
    w = w / w.sum()
    if mode == 'visual':
        q, r = None, p
        frequency = (p * w[None, :, None]).sum(1)
    else:
        if uniform_semantics:
            log_q = torch.full_like(log_p, -math.log(len(base_visual)))
        else:
            if base_text is None or novel_text is None:
                raise ValueError('Cross-modal guidance needs class/scale text prototypes.')
            base_text = F.normalize(base_text.float(), dim=-1)[:, list(permutation)]
            novel_text = F.normalize(novel_text.float(), dim=-1)[:, list(permutation)]
            log_q = (tau_text * torch.einsum('csd,bsd->csb', novel_text, base_text)).log_softmax(-1)
        q = log_q.exp()
        r = ((1 - eta) * log_p + eta * log_q).softmax(-1)
        if mode == 'paired':
            frequency = (r * w[None, :, None]).sum(1)
        elif mode == 'marginal':
            # logsumexp avoids underflow for sharply peaked scale relations.
            log_w = w.log()[None, :, None]
            mean_p = torch.logsumexp(log_p + log_w, dim=1)
            mean_q = torch.logsumexp(log_q + log_w, dim=1)
            frequency = ((1 - eta) * mean_p + eta * mean_q).softmax(-1)
        else:
            raise ValueError('Unknown guidance mode: ' + mode)
    return {'original': a0, 'visual': p, 'semantic': q, 'paired': r, 'frequency': frequency}


@torch.no_grad()
def update_prototypes(base_visual, novel_visual, relations, rho, shift=.1,
                      reference=None):
    # Use the original implementation for exact zero-strength regression.
    if shift == 0:
        return novel_visual
    if rho == 0 and reference is not None:
        return reference(base_visual, novel_visual)
    base = F.normalize(base_visual, dim=-1)
    novel = F.normalize(novel_visual, dim=-1)
    weights = ((1 - rho) * relations['original'] + rho * relations['frequency']).to(base)
    guidance = F.normalize(weights @ base, dim=-1)
    return F.normalize((1 - shift) * novel + shift * guidance, dim=-1)


def settings_kwargs(cfg):
    g = cfg.TRAINER.BiMC.GUIDANCE
    return dict(mode=g.MODE, tau=cfg.TRAINER.BiMC.TAU, tau_text=g.TAU_TEXT,
                eta=g.ETA, permutation=g.SEMANTIC_PERMUTATION,
                band_weights=g.BAND_WEIGHTS, uniform_semantics=g.SEMANTIC_SOURCE == 'uniform')


@torch.no_grad()
def diagnostics(relations, raw, updated, base_ids, rho):
    def entropy(value):
        return None if value is None else (-(value * value.clamp_min(1e-30).log()).sum(-1)).cpu().tolist()
    weight = (1 - rho) * relations['original'] + rho * relations['frequency']
    values, positions = weight.topk(min(5, len(base_ids)), dim=-1)
    p, q = relations['visual'], relations['semantic']
    k = min(5, len(base_ids))
    neighbors = p.topk(k, dim=-1).indices
    overlap = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        overlap.append((neighbors[:, a, :, None] == neighbors[:, b, None, :]).any(-1).float().mean(-1))
    result = {
        'rho': float(rho), 'fallback_to_G1': rho == 0,
        'top_base_ids': [[int(base_ids[i]) for i in row] for row in positions.cpu().tolist()],
        'top_base_weights': values.cpu().tolist(),
        'entropy_visual': entropy(p), 'entropy_semantic': entropy(q),
        'entropy_paired': entropy(relations['paired']),
        'interscale_top5_overlap': torch.stack(overlap, 1).cpu().tolist(),
        'movement_degrees': (F.cosine_similarity(raw.float(), updated.float(), dim=-1)
                             .clamp(-1, 1).acos() * (180 / math.pi)).cpu().tolist(),
        'raw_prototypes': raw.cpu().tolist(), 'guided_prototypes': updated.cpu().tolist(),
    }
    if q is not None:
        result['visual_semantic_relation_cosine'] = F.cosine_similarity(p, q, dim=-1).cpu().tolist()
    return result
