"""Frozen, attribute-scoped prototype transfer inspired by FDR.

This is an explicit four-group residual transfer experiment, not a faithful
reimplementation of the FDR paper. A donor bank contains fitting-class means
and pooled sampling statistics only. Predictions use the current class's
support, its known text, and a fixed channel mask; no queries are accepted.

The optional reliability rule is a conservative heuristic. It is not claimed
to be an optimal Gaussian estimator or an independently calibrated posterior.
"""

import hashlib
import math

import torch
from torch.nn import functional as F


FORMAT = 'attribute_channel_residual_transfer_v1'
GROUPS = ('shape', 'part', 'color', 'pattern')
EPS = 1e-12


def _real(value, name, device=None):
    result = torch.as_tensor(value, device=device).detach()
    if result.dtype == torch.bool or result.is_complex() or not torch.isfinite(result).all():
        raise ValueError(name+' must contain finite real values.')
    return result.to(torch.float64)


def _ids(value, name, device=None):
    result = torch.as_tensor(value, device=device).detach()
    if (result.ndim != 1 or not result.numel() or result.dtype == torch.bool
            or result.is_complex() or not torch.isfinite(result).all()
            or (result.is_floating_point() and not torch.equal(result, result.round()))
            or (result < 0).any()):
        raise ValueError(name+' must be a nonempty vector of nonnegative integers.')
    return result.long()


def _binary(value, shape, name, device):
    result = torch.as_tensor(value, device=device).detach()
    if (tuple(result.shape) != tuple(shape) or result.is_complex()
            or not torch.isfinite(result).all() or not torch.all((result == 0) | (result == 1))):
        raise ValueError(name+' must be a binary array of shape '+str(tuple(shape))+'.')
    return result.bool()


def _inputs(features, labels, text, available, class_ids):
    values = _real(features, 'features')
    if values.ndim != 2 or min(values.shape) < 1 or (values.norm(dim=-1) <= EPS).any():
        raise ValueError('features must be a nonzero [samples, dimension] matrix.')
    y = _ids(labels, 'labels', values.device)
    ids = _ids(class_ids, 'class_ids', values.device)
    if len(y) != len(values) or len(ids.unique()) != len(ids):
        raise ValueError('labels must match feature rows and class_ids must be distinct.')
    groups = [torch.where(y == c)[0] for c in ids]
    if any(not len(g) for g in groups):
        raise ValueError('Each requested class needs at least one sample.')
    texts = _real(text, 'text', values.device)
    if texts.shape != (len(ids), len(GROUPS), values.shape[1]):
        raise ValueError('text must have shape [classes, 4, feature_dimension].')
    mask = _binary(available, (len(ids), len(GROUPS)), 'available', values.device)
    norms = texts.norm(dim=-1)
    if (norms[mask] <= EPS).any():
        raise ValueError('Available attribute text must be nonzero.')
    texts = torch.where(mask[..., None], F.normalize(texts, dim=-1), torch.zeros_like(texts))
    return values, y, texts, mask, ids, groups


@torch.no_grad()
def fit_attribute_transfer(features, labels, text, available, class_ids, *, topk=5, temperature=16.):
    """Fit an A-only bank; text/availability rows follow explicit class_ids.

    Feature rows with labels outside class_ids are excluded from all statistics.
    The pooled diagonal covariance uses the actual within-class degrees of
    freedom. No image rows, descriptions, or query observations are retained.
    """
    if isinstance(topk, bool) or not isinstance(topk, int) or topk < 1:
        raise ValueError('topk must be a positive integer.')
    if (isinstance(temperature, bool) or not isinstance(temperature, (float, int))
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError('temperature must be a finite positive scalar.')
    values, _, texts, mask, ids, groups = _inputs(features, labels, text, available, class_ids)
    means = torch.stack([values[g].mean(0) for g in groups])
    if (means.norm(dim=-1) <= EPS).any():
        raise ValueError('Fitting class means must be nonzero.')
    counts = torch.tensor([len(g) for g in groups], device=values.device, dtype=torch.long)
    degrees = int(counts.sum())-len(ids)
    scatter = torch.zeros(values.shape[1], dtype=values.dtype, device=values.device)
    for mean, group in zip(means, groups):
        scatter += ((values[group]-mean)**2).sum(0)
    within = scatter/max(1, degrees)
    return {'format':FORMAT, 'class_ids':ids.detach().clone(), 'counts':counts,
            'means':means.detach(), 'prototypes':F.normalize(means, dim=-1).detach(),
            'text':texts.detach(), 'available':mask.detach(), 'within_diag':within.detach(),
            'feature_dimension':values.shape[1], 'topk':topk, 'temperature':float(temperature),
            'fit_class_ids':ids.detach().clone(), 'fit_sample_count':int(counts.sum()),
            'within_degrees_of_freedom':degrees, 'retained_training_samples':0,
            'gradient_training_performed':False,
            'reliability_rule':'semantic_agreement_times_support_noise_over_noise_plus_donor_and_residual_uncertainty',
            'reliability_is_heuristic':True}


def _state(state, dimension, device):
    if not isinstance(state, dict) or state.get('format') != FORMAT:
        raise ValueError('Invalid attribute transfer state format.')
    if state.get('feature_dimension') != dimension:
        raise ValueError('Prediction feature dimension differs from the donor bank.')
    ids = _ids(state['class_ids'], 'state class_ids', device)
    if len(ids.unique()) != len(ids):
        raise ValueError('Donor IDs must be distinct.')
    result = {name:_real(state[name], 'state '+name, device)
              for name in ('means','prototypes','counts','text','within_diag')}
    if (result['means'].shape != (len(ids),dimension)
            or result['prototypes'].shape != (len(ids),dimension)
            or result['counts'].shape != (len(ids),)
            or result['text'].shape != (len(ids),4,dimension)
            or result['within_diag'].shape != (dimension,)
            or (result['counts'] < 1).any() or (result['within_diag'] < 0).any()):
        raise ValueError('The donor bank has inconsistent array shapes or sampling statistics.')
    result['available'] = _binary(state['available'], (len(ids),4), 'state available', device)
    if not torch.allclose(result['prototypes'].norm(dim=-1), torch.ones(len(ids),device=device,dtype=torch.float64), atol=1e-7, rtol=1e-6):
        raise ValueError('Donor prototypes must be unit normalized.')
    if (result['text'].norm(dim=-1)[result['available']] <= EPS).any():
        raise ValueError('Available donor texts must be nonzero.')
    if (isinstance(state['topk'],bool) or not isinstance(state['topk'],int) or state['topk'] < 1
            or not math.isfinite(state['temperature']) or state['temperature'] <= 0):
        raise ValueError('Donor retrieval settings are invalid.')
    result['class_ids'] = ids
    return result


def _random_mask(mask, seed, class_id, group):
    # Global class identity, rather than sequence/order or support content,
    # makes the random-capacity control stable across incremental arrivals.
    key = f'{seed}:class={class_id}:group={group}:dimension={len(mask)}'.encode()
    generator = torch.Generator(device='cpu').manual_seed(int.from_bytes(hashlib.sha256(key).digest()[:8], 'little') % (2**63-1))
    indices = torch.randperm(len(mask), generator=generator)[:int(mask.sum())].to(mask.device)
    result = torch.zeros_like(mask)
    result[indices] = True
    return result


@torch.no_grad()
def complete_attribute_transfer(state, features, labels, text, available, channel_masks,
                                class_ids, *, mode='semantic', reliability=False,
                                seed=3101, return_diagnostics=False):
    """Return raw-scale means with independently computed classwise residuals.

    ``whole`` is a whole-prototype donor-transfer control. ``random`` matches
    each target mask's cardinality. ``shuffled`` cyclically deranges available
    masks within each target class while retaining the original text groups.
    Overlapping unique evidence is averaged per coordinate. Exact duplicate
    text/mask/donor blocks are merged before this average, never counted as
    additional support images. A zero correction returns the raw mean exactly.
    """
    if mode not in ('whole','semantic','random','shuffled'):
        raise ValueError('Unknown attribute transfer mode.')
    if not isinstance(reliability, bool) or not isinstance(return_diagnostics, bool):
        raise ValueError('reliability and return_diagnostics must be boolean.')
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('seed must be a nonnegative integer.')
    values, _, texts, active, ids, groups = _inputs(features, labels, text, available, class_ids)
    masks = _binary(channel_masks, texts.shape, 'channel_masks', values.device)
    bank = _state(state, values.shape[1], values.device)
    output_dtype = torch.as_tensor(features).dtype
    if output_dtype not in (torch.float32, torch.float64):
        output_dtype = torch.float32
    source_values = torch.as_tensor(features,device=values.device).detach().to(output_dtype)
    results, class_reports = [], []
    for row, (class_id, group) in enumerate(zip(ids.tolist(), groups)):
        # Use the classifier's original-dtype arithmetic for the untouched
        # mean. A double-mean/cast could differ by an ulp and fabricate a tiny
        # correction for unknown/no-donor classes at a nonzero outer blend.
        raw = source_values[group].mean(0).to(torch.float64)
        scale = raw.norm()
        if scale <= EPS:
            raise ValueError('Support class means must be nonzero.')
        direction = raw/scale
        chosen_masks = masks[row].clone()
        source_groups = list(range(4))
        usable = [g for g in range(4) if bool(active[row,g]) and bool(masks[row,g].any())]
        aliases = {}
        canonical = []
        for g in usable:
            old = next((h for h in canonical if torch.equal(masks[row,g],masks[row,h])
                and torch.equal(texts[row,g],texts[row,h])
                and torch.equal(bank['available'][:,g],bank['available'][:,h])
                and torch.equal(bank['text'][:,g],bank['text'][:,h])), None)
            if old is None:
                canonical.append(g)
            else:
                aliases[g] = old
        if mode == 'shuffled' and len(canonical) > 1:
            for position,g in enumerate(canonical):
                source_groups[g] = canonical[(position+1) % len(canonical)]
                chosen_masks[g] = masks[row,source_groups[g]]
        if mode == 'whole':
            for g in usable:
                chosen_masks[g] = True
        elif mode == 'random':
            for g in canonical:
                chosen_masks[g] = _random_mask(masks[row,g],seed,class_id,g)
        for g,old in aliases.items():
            chosen_masks[g] = chosen_masks[old]
            source_groups[g] = source_groups[old]
        delta = torch.zeros_like(raw)
        hits = torch.zeros_like(raw)
        reports, evidence = [], []
        for g,name in enumerate(GROUPS):
            mask = chosen_masks[g]
            report = {'group':name,'available':bool(active[row,g]),
                      'mask_source_group':GROUPS[source_groups[g]],
                      'mask_count':int(mask.sum()),'original_mask_count':int(masks[row,g].sum()),
                      'mask_channel_indices':torch.where(mask)[0].cpu().tolist(),
                      'gate':0.,'support_noise':None,'donor_disagreement':None,
                      'donor_mean_uncertainty':None,'residual_energy':None,
                      'semantic_similarity':None,'donor_ids':[],'donor_weights':[],
                      'donor_similarities':[],
                      'reason':None,'duplicate_of':None,
                      'shuffled':mode == 'shuffled' and source_groups[g] != g}
            reports.append(report)
            if not bool(active[row,g]):
                report['reason'] = 'unknown_attribute'
                continue
            if not masks[row,g].any() or not mask.any():
                report['reason'] = 'empty_channel_mask'
                continue
            duplicate = next((old_g for old_g,old_mask in evidence
                if torch.equal(mask,old_mask) and torch.equal(texts[row,g],texts[row,old_g])
                and torch.equal(bank['available'][:,g],bank['available'][:,old_g])
                and torch.equal(bank['text'][:,g],bank['text'][:,old_g])), None)
            if duplicate is not None:
                report.update(reason='duplicate_evidence_merged', duplicate_of=GROUPS[duplicate])
                continue
            target = texts[row,g]*mask
            if target.norm() <= EPS:
                report['reason'] = 'zero_masked_text'
                continue
            donor_text = bank['text'][:,g]*mask
            eligible = torch.where(bank['available'][:,g] & (bank['class_ids'] != class_id)
                                   & (donor_text.norm(dim=-1) > EPS))[0]
            if not len(eligible):
                report['reason'] = 'no_eligible_donor'
                continue
            # Tie resolution uses global donor identity, not storage order.
            eligible = eligible[torch.argsort(bank['class_ids'][eligible])]
            similarities = F.normalize(donor_text[eligible],dim=-1) @ F.normalize(target,dim=-1)
            order = torch.argsort(similarities,descending=True,stable=True)[:state['topk']]
            indices, similarities = eligible[order], similarities[order]
            weights = torch.softmax(state['temperature']*similarities,dim=0)
            donors = bank['prototypes'][indices]
            combination = (weights[:,None]*donors).sum(0)
            if combination.norm() <= EPS:
                report['reason'] = 'cancelled_donor_combination'
                continue
            prior = F.normalize(combination,dim=-1)
            residual = mask*(prior-direction)
            support_noise = bank['within_diag'][mask].sum()/len(group)/(scale**2)
            disagreement = (weights*((donors-combination)**2)[:,mask].sum(-1)).sum()
            donor_uncertainty = (weights.square()/bank['counts'][indices]).sum()*bank['within_diag'][mask].sum()
            residual_energy = residual.square().sum()
            semantic = (weights*similarities).sum()
            gate = 1.
            if reliability:
                # Large support noise permits borrowing; donor disagreement,
                # finite donor means, and a large direction mismatch oppose it.
                # This is deliberately a conservative, documented heuristic.
                semantic_confidence = semantic.clamp(0.,1.)
                gate = float(semantic_confidence*support_noise /
                             (support_noise+disagreement+donor_uncertainty+residual_energy+EPS))
            report.update(gate=gate, support_noise=float(support_noise),
                          donor_disagreement=float(disagreement),
                          donor_mean_uncertainty=float(donor_uncertainty),
                          residual_energy=float(residual_energy),
                          semantic_similarity=float(semantic),
                          donor_ids=bank['class_ids'][indices].cpu().tolist(),
                          donor_weights=weights.cpu().tolist(),
                          donor_similarities=similarities.cpu().tolist())
            if gate <= EPS:
                report['reason'] = 'reliability_rejected'
                continue
            report['reason'] = 'transferred'
            evidence.append((g,mask))
            delta += gate*residual
            hits += mask
        delta = delta/hits.clamp_min(1)
        completed = raw+scale*delta
        if completed.norm() <= EPS or not torch.isfinite(completed).all():
            # A residual that cancels the entire prototype is rejected, rather
            # than passing a zero direction to the classifier.
            completed = raw.clone()
            class_reason = 'degenerate_direction_rejected'
        else:
            class_reason = 'transferred' if delta.abs().sum() > 0 else 'no_correction'
        results.append(completed)
        class_reports.append({'class_id':class_id,'support_count':len(group),
            'reason':class_reason,'raw_norm':float(scale),'completed_norm':float(completed.norm()),
            'raw_completed_cosine':float(F.cosine_similarity(raw[None],completed[None]).clamp(-1.,1.)),
            'correction_norm':float((completed-raw).norm()),
            'covered_channels':int((hits>0).sum()),'overlap_channels':int((hits>1).sum()),
            'maximum_unique_mask_hits':int(hits.max()),
            'unshufflable_single_group':mode == 'shuffled' and len(canonical) < 2,
            'groups':reports})
    result = torch.stack(results).to(output_dtype).detach()
    diagnostics = {'format':FORMAT,'mode':mode,'reliability':reliability,
                   'reliability_is_heuristic':True,'donor_class_ids':bank['class_ids'].cpu().tolist(),
                   'self_donors_excluded':True,'sampling_count_rule':'actual_support_count_per_class',
                   'overlap_rule':'mean_over_unique_active_masks_per_coordinate',
                   'new_gradient_training':False,'classes':class_reports}
    return (result,diagnostics) if return_diagnostics else result
