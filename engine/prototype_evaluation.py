"""Shared base-only prototype evaluation and calibrated BiMC scoring.

No CLIP model is constructed. A/V selection tasks use cached original images;
the V teacher bank is reserved globally and used only by diagnostic oracles.
"""
import itertools
import math
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from engine.residual_training import _group_covariance
from engine.safd_metric_training import (
    reserve_selection_split, _reference_proxy, _text_subset,
)


def identity_calibration():
    return {'fused_temperature': 1., 'cov_temperature': 1.,
            'description_temperature': 1., 'novel_bias': 0.}


def validate_calibration(calibration):
    values = identity_calibration()
    if calibration is not None:
        if set(calibration) != set(values):
            raise ValueError('Calibration must specify exactly three temperatures and novel_bias.')
        values.update(calibration)
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Calibration values must be finite real scalars.')
        if key != 'novel_bias' and value <= 0:
            raise ValueError('Branch temperatures must be positive.')
    return values


def prepare_context(arrays, cfg, seed=1, sequences=20, teacher_shot=15):
    if min(seed, sequences, teacher_shot) < 1:
        raise ValueError('seed, sequences and teacher_shot must be positive.')
    local = cfg.clone()
    local.defrost()
    local.TRAINER.BiMC.GUIDANCE.AUDIT_IMAGES = int(teacher_shot)
    local.freeze()
    labels = torch.as_tensor(arrays['labels']).long()
    return reserve_selection_split(local, labels, arrays['sample_id'],
                                   SimpleNamespace(seed=int(seed), sequences=int(sequences)))


@torch.no_grad()
def validation_records(arrays, cfg, protocol, device='cpu'):
    """Build staged original-image tasks; no teacher feature enters reference votes."""
    x = F.normalize(torch.as_tensor(arrays['features'][:, 0], device=device).float(), dim=-1)
    proxy, records = _reference_proxy(cfg), []
    b = cfg.TRAINER.BiMC
    for sequence_id, seq in enumerate(protocol['sequences']):
        classes, supports, queries = seq['class_ids'], seq['support_indices'], seq['query_indices']
        old = len(protocol['pseudo_base_class_ids'])
        support_features = [x[rows] for rows in supports]
        raw = F.normalize(torch.stack([v.mean(0) for v in support_features]), dim=-1)
        reference = raw.clone()
        if b.VISION_CALIBRATION:
            reference[old:] = proxy.soft_calibration(raw[:old], raw[old:])
        teachers = reference.clone()
        for j, c in enumerate(classes[old:], old):
            rows = protocol['audit_indices'][str(c)]
            if (set(rows) & set(supports[j])) or (set(rows) & set(queries[j])):
                raise ValueError('Diagnostic teacher bank overlaps support/query images.')
            teachers[j] = F.normalize(x[rows].mean(0), dim=-1)
        covariance_sum = old * _group_covariance(
            x[[i for rows in supports[:old] for i in rows]], b.GAMMA_BASE)
        start = old
        for stage, size in enumerate(seq['stage_sizes'], 1):
            end = start + size
            covariance_sum = covariance_sum + size * _group_covariance(
                x[[i for rows in supports[start:end] for i in rows]], b.GAMMA_INC)
            covariance = covariance_sum / end
            cov_inverse = torch.pinverse(covariance.float())
            dp, df, dt, text = _text_subset(arrays, classes[:end], device)
            query_indices = [i for rows in queries[:end] for i in rows]
            query = x[query_indices]
            targets = torch.cat([torch.full((len(rows),), j, dtype=torch.long, device=device)
                                 for j, rows in enumerate(queries[:end])])
            if not torch.equal(torch.as_tensor(classes[:end], device=device)[targets],
                               torch.as_tensor(arrays['labels'], device=device)[query_indices]):
                raise ValueError('Cached global labels and episode-local class order disagree.')
            votes = proxy.reference_scores_from_features(query, end, old, reference[:end], covariance,
                dp, df, dt, text, cfg.DATASET.BETA, cov_inverse=cov_inverse)
            lambda_t = b.LAMBDA_T if b.TEXT_CALIBRATION else 0.
            semantic = F.normalize((1 - lambda_t) * text + lambda_t * dp, dim=-1)
            row = {'class_ids': classes[:end], 'num_old': old,
                   'arrival_sizes': list(seq['stage_sizes'][:stage]),
                   'supports': support_features[:end], 'raw_prototypes': raw[:end],
                   'reference_prototypes': reference[:end], 'semantic_prototypes': semantic,
                   'teacher_prototypes': teachers[:end], 'query': query, 'targets': targets,
                   'covariance': covariance, 'cov_inverse': cov_inverse,
                   'description_proto': dp, 'description_features': df, 'description_targets': dt,
                   'text_features': text, 'reference_votes': votes,
                   'sequence': sequence_id, 'stage': stage,
                   'support_indices': supports[:end], 'query_indices': queries[:end],
                   'teacher_indices': [protocol['audit_indices'][str(c)] for c in classes[old:end]],
                   'query_sample_ids': np.asarray(arrays['sample_id'])[query_indices].tolist()}
            records.append(row)
            start = end
    return records


def _prototypes(row):
    return row['reference_prototypes'] if 'reference_prototypes' in row else row['image_proto']


def branch_logits(row, cfg, image_prototypes=None):
    """Return raw prototype/description cosines and covariance scores /512.

    The class-common query quadratic term is omitted from covariance logits;
    this leaves softmax probabilities unchanged and avoids per-class matmuls.
    All products involving the inverse covariance use float64 internally.
    """
    if image_prototypes is None and '_reference_branch_logits' in row:
        return row['_reference_branch_logits']
    q = F.normalize(row['query'].float(), dim=-1)
    proto = _prototypes(row) if image_prototypes is None else image_prototypes
    proto = proto.to(device=q.device, dtype=torch.float32)
    lambda_t = cfg.TRAINER.BiMC.LAMBDA_T if cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.
    fused = cfg.DATASET.BETA * ((1 - lambda_t) * row['text_features'].to(q.device).float()
                + lambda_t * row['description_proto'].to(q.device).float()) + (1 - cfg.DATASET.BETA) * proto
    fused_logits = q @ F.normalize(fused, dim=-1).T
    inverse = row.get('cov_inverse')
    if inverse is None:
        covariance = row['covariance'] if 'covariance' in row else row['cov_image']
        inverse = torch.pinverse(covariance.float())
    inv = inverse.to(device=q.device, dtype=torch.float64)
    p = proto.double()
    cov_logits = (q.double() @ (inv + inv.T) @ p.T - (p * (p @ inv)).sum(-1)) / 512.
    cov_logits = cov_logits.float()
    if '_description_logits' in row:
        description = row['_description_logits']
    else:
        desc = row['description_features'].to(q.device).float()
        targets = row['description_targets'].to(q.device).long()
        similarity = q @ desc.T
        columns = []
        for c in range(len(proto)):
            values = similarity[:, targets == c]
            if not values.shape[1]:
                raise ValueError(f'Missing ordinary description features for local class {c}.')
            columns.append(values.max(1).values)
        description = torch.stack(columns, dim=1)
        if not description.requires_grad:
            row['_description_logits'] = description
    result = {'fused': fused_logits, 'cov': cov_logits, 'description': description}
    if image_prototypes is None and not any(v.requires_grad for v in result.values()):
        row['_reference_branch_logits'] = result
    return result


def scores_from_logits(logits, num_old, cfg, calibration=None):
    c = validate_calibration(calibration)
    p_f = F.softmax(logits['fused'] / c['fused_temperature'], dim=-1)
    p_cov = F.softmax(logits['cov'] / c['cov_temperature'], dim=-1)
    p_desc = F.softmax(logits['description'] / c['description_temperature'], dim=-1)
    alpha = cfg.DATASET.ENSEMBLE_ALPHA if cfg.TRAINER.BiMC.USING_ENSEMBLE else 1.
    scores = torch.cat((alpha * p_f[:, :num_old] + (1 - alpha) * p_cov[:, :num_old],
                        alpha * p_f[:, num_old:] + (1 - alpha) * p_desc[:, num_old:]), dim=-1)
    if c['novel_bias']:
        scores = torch.cat((scores[:, :num_old], scores[:, num_old:] * math.exp(c['novel_bias'])), dim=-1)
    if not torch.isfinite(scores).all():
        raise ValueError('Calibrated scores must remain finite.')
    return scores.clamp_min(torch.nextafter(scores.new_tensor(0.), scores.new_tensor(1.)))


def score_record(row, cfg, image_prototypes=None, calibration=None):
    c = validate_calibration(calibration)
    if c == identity_calibration() and (image_prototypes is None or torch.equal(image_prototypes, _prototypes(row))):
        return row['reference_votes']
    return scores_from_logits(branch_logits(row, cfg, image_prototypes), row['num_old'], cfg, c)


def description_count_scores(row, cfg, per_class_count, calibration=None):
    """Equalize only description-max candidate counts, keeping text means fixed.

    Selection is evenly spaced in the existing text order, not image/query
    grounded. This is a diagnostic control, not a new fitted predictor.
    """
    if isinstance(per_class_count, bool) or not isinstance(per_class_count, int) or per_class_count < 1:
        raise ValueError('per_class_count must be a positive integer.')
    targets = row['description_targets']
    selected, changed = [], False
    for c in range(len(_prototypes(row))):
        indices = torch.where(targets == c)[0]
        if len(indices) < per_class_count:
            raise ValueError('Description-count control exceeds an available class bank.')
        positions = torch.linspace(0, len(indices) - 1, per_class_count, device=indices.device).round().long()
        selected.append(indices[positions])
        changed = changed or len(indices) != per_class_count
    if not changed:
        return score_record(row, cfg, calibration=calibration)
    reduced = dict(row)
    rows = torch.cat(selected)
    reduced['description_features'] = row['description_features'][rows]
    reduced['description_targets'] = targets[rows]
    reduced.pop('_description_logits', None)
    reduced.pop('_reference_branch_logits', None)
    return scores_from_logits(branch_logits(reduced, cfg), row['num_old'], cfg, calibration)


@torch.no_grad()
def evaluate_records(records, score_fn, reference_score_fn=None):
    """Mean stage accuracies and paired counts against an explicit anchor.

    The default anchor is the original BiMC, preserving existing reports.
    Counts are repeated stage-query observations, not distinct images.
    """
    rows, corrected, damaged, changed = [], 0, 0, 0
    for row in records:
        scores = score_fn(row)
        target = row['targets']
        old = target < row['num_old']
        if not old.any() or not (~old).any():
            raise ValueError('A validation stage must contain both pseudo-old and pseudo-novel queries.')
        correct = scores.argmax(-1) == target
        reference = row['reference_votes'] if reference_score_fn is None else reference_score_fn(row)
        ref_prediction = reference.argmax(-1)
        ref_correct = ref_prediction == target
        base = 100 * correct[old].float().mean().item()
        novel = 100 * correct[~old].float().mean().item()
        rows.append((base, novel, 2 * base * novel / (base + novel) if base + novel else 0.))
        corrected += int((correct & ~ref_correct).sum())
        damaged += int((~correct & ref_correct).sum())
        changed += int((scores.argmax(-1) != ref_prediction).sum())
    if not rows:
        raise ValueError('No validation stages.')
    means = np.asarray(rows).mean(0)
    return {'base_accuracy': float(means[0]), 'novel_accuracy': float(means[1]),
            'harmonic_accuracy': float(means[2]), 'stages': len(rows),
            'corrected_observations': corrected, 'damaged_observations': damaged,
            'changed_observations': changed}


def _feasible(metrics, reference, maximum_drop):
    return all(metrics[k] >= reference[k] - maximum_drop - 1e-12
               for k in ('base_accuracy', 'novel_accuracy'))


@torch.no_grad()
def select_calibration(records, cfg, maximum_drop=.2, grid=None):
    """Select only on base-derived validation tasks; no oracle is considered."""
    if not math.isfinite(maximum_drop) or maximum_drop < 0:
        raise ValueError('maximum_drop must be finite and nonnegative.')
    reference = evaluate_records(records, lambda r: r['reference_votes'])
    if grid is None:
        grid = [dict(zip(identity_calibration(), values)) for values in itertools.product(
            (1., .1, .03), (1., .5, 2.), (1., .1, .03), (0., -.05, .05))]
    definitions = [identity_calibration()] + [validate_calibration(v) for v in grid]
    unique, candidates = [], []
    for definition in definitions:
        if definition not in unique:
            unique.append(definition)
    best, best_metrics = identity_calibration(), reference
    for index, c in enumerate(unique):
        metrics = evaluate_records(records, lambda r: score_record(r, cfg, calibration=c))
        feasible = _feasible(metrics, reference, maximum_drop)
        row = {'index': index, 'calibration': c, **metrics, 'feasible': feasible,
               'H_delta_pp': metrics['harmonic_accuracy'] - reference['harmonic_accuracy']}
        candidates.append(row)
        if feasible and metrics['harmonic_accuracy'] > best_metrics['harmonic_accuracy'] + 1e-12:
            best, best_metrics = dict(c), metrics
    report = {'reference': reference, 'selected_validation': best_metrics,
              'selected_calibration': best, 'candidates': candidates,
              'candidate_count': len(candidates), 'maximum_group_drop_pp': maximum_drop,
              'selection_source': 'base_training_class_disjoint_validation_only',
              'selection_rule': 'maximum mean-stage H subject to mean old/novel drop guard; identity wins ties',
              'oracle_candidates_allowed': False}
    return best, report


@torch.no_grad()
def expert_oracle_scores(row, cfg):
    """Label-using diagnostic: correct if reference OR any raw branch is correct."""
    logits = branch_logits(row, cfg)
    hit = row['reference_votes'].argmax(-1) == row['targets']
    for values in logits.values():
        hit = hit | (values.argmax(-1) == row['targets'])
    scores = row['reference_votes'].clone()
    scores[hit] = F.one_hot(row['targets'][hit], scores.shape[1]).to(scores.dtype)
    return scores


def prototype_control_scores(row, cfg, variant, calibration=None):
    """Cross support/teacher banks with the same visual guidance operation.

    Covariance, semantic features, old prototypes and score calibration stay
    fixed. Teacher variants use extra labels and are never fitted candidates.
    """
    if variant == 'support_guided':
        return score_record(row, cfg, calibration=calibration)
    if variant == 'support_raw':
        proto = row['raw_prototypes']
    elif variant == 'teacher_raw':
        proto = row['teacher_prototypes']
    elif variant == 'teacher_guided':
        proto = row['teacher_prototypes']
        if cfg.TRAINER.BiMC.VISION_CALIBRATION:
            old = row['num_old']
            proto = proto.clone()
            proto[old:] = _reference_proxy(cfg).soft_calibration(_prototypes(row)[:old], proto[old:])
    else:
        raise ValueError(f'Unknown prototype control: {variant}')
    return score_record(row, cfg, image_prototypes=proto, calibration=calibration)


def _accuracy_deltas(metrics, reference):
    return {key + '_delta_pp': metrics[key] - reference[key]
            for key in ('base_accuracy', 'novel_accuracy', 'harmonic_accuracy')}


@torch.no_grad()
def prototype_factorial_diagnostics(records, cfg, calibration=None):
    """A diagnostic 2x2 comparison; no teacher result selects a model."""
    calibration = validate_calibration(calibration)
    def scores(row, variant):
        return prototype_control_scores(row, cfg, variant, calibration)
    anchor = lambda row: scores(row, 'support_guided')
    reference = evaluate_records(records, anchor, reference_score_fn=anchor)
    variants = {}
    for name in ('support_guided', 'support_raw', 'teacher_guided', 'teacher_raw'):
        metrics = evaluate_records(records, lambda row: scores(row, name), reference_score_fn=anchor)
        teacher = name.startswith('teacher_')
        variants[name] = {'metrics': metrics, 'delta_vs_support_guided': _accuracy_deltas(metrics, reference),
                          'uses_extra_labeled_teacher_images': teacher,
                          'deployable': not teacher, 'selection_candidate': False}
    contrasts = {}
    comparisons = (
        ('disable_guidance_support', 'support_raw', 'support_guided'),
        ('disable_guidance_teacher', 'teacher_raw', 'teacher_guided'),
        ('teacher_vs_support_guided', 'teacher_guided', 'support_guided'),
        ('teacher_vs_support_raw', 'teacher_raw', 'support_raw'),
    )
    for name, candidate, control in comparisons:
        metrics = evaluate_records(records, lambda row: scores(row, candidate),
                                   reference_score_fn=lambda row: scores(row, control))
        contrasts[name] = {'candidate': candidate, 'control': control,
                          'delta_pp': _accuracy_deltas(metrics, variants[control]['metrics']),
                          'corrected_observations': metrics['corrected_observations'],
                          'damaged_observations': metrics['damaged_observations'],
                          'changed_observations': metrics['changed_observations']}
    support_counts = sorted({len(indices) for row in records
                             for indices in row['support_indices'][row['num_old']:]})
    teacher_counts = sorted({len(indices) for row in records for indices in row['teacher_indices']})
    return {'anchor': 'support_guided_with_selected_score_calibration',
            'fixed_score_calibration': calibration, 'vision_calibration_enabled': bool(cfg.TRAINER.BiMC.VISION_CALIBRATION),
            'support_counts': support_counts, 'teacher_counts': teacher_counts,
            'old_prototypes_covariance_and_text_fixed': True,
            'teacher_and_support_banks_disjoint': True, 'variants': variants, 'contrasts': contrasts,
            'selection_candidate': False,
            'limitations': ['Teacher banks contain extra labeled images and are not nested support sets.',
                'Teacher/support contrasts compare independent banks; they are not strict accuracy upper bounds.',
                'Disabling novel guidance also changes old-class accuracy through class competition.']}


def _expert_scores(row, cfg, calibration):
    logits = branch_logits(row, cfg)
    return {'original_BiMC': row['reference_votes'],
            'calibrated_BiMC': score_record(row, cfg, calibration=calibration),
            'fused_prototype': logits['fused'], 'covariance': logits['cov'],
            'description_max': logits['description']}


@torch.no_grad()
def expert_profile_diagnostics(records, cfg, calibration=None):
    """Profile fixed predictors before interpreting a label-using union.

    Every single expert predicts across all seen classes. Pairwise counts use
    labels only for evaluation; they do not define an inference-time router.
    """
    calibration = validate_calibration(calibration)
    names = ('original_BiMC', 'calibrated_BiMC', 'fused_prototype', 'covariance', 'description_max')
    anchor = lambda row: score_record(row, cfg, calibration=calibration)
    reference = evaluate_records(records, anchor, reference_score_fn=anchor)
    profiles = {}
    for name in names:
        metrics = evaluate_records(records, lambda row: _expert_scores(row, cfg, calibration)[name],
                                   reference_score_fn=anchor)
        profiles[name] = {'metrics': metrics, 'delta_vs_calibrated': _accuracy_deltas(metrics, reference)}
    pairs = []
    for left, right in itertools.combinations(names, 2):
        groups = {name: dict.fromkeys(('observations', 'both_correct', 'left_only_correct',
                    'right_only_correct', 'both_wrong'), 0) for name in ('all', 'old', 'novel')}
        for row in records:
            scores = _expert_scores(row, cfg, calibration)
            target = row['targets']
            lc, rc = (scores[name].argmax(-1) == target for name in (left, right))
            old = target < row['num_old']
            for group, mask in (('all', torch.ones_like(old)), ('old', old), ('novel', ~old)):
                counts = groups[group]
                counts['observations'] += int(mask.sum())
                counts['both_correct'] += int((lc & rc & mask).sum())
                counts['left_only_correct'] += int((lc & ~rc & mask).sum())
                counts['right_only_correct'] += int((~lc & rc & mask).sum())
                counts['both_wrong'] += int((~lc & ~rc & mask).sum())
        pairs.append({'left': left, 'right': right, 'groups': groups})
    def union_scores(row):
        experts = _expert_scores(row, cfg, calibration)
        hit = torch.stack([scores.argmax(-1) == row['targets'] for scores in experts.values()]).any(0)
        result = row['reference_votes'].clone()
        result[hit] = F.one_hot(row['targets'][hit], result.shape[1]).to(result.dtype)
        return result
    union = evaluate_records(records, union_scores, reference_score_fn=anchor)
    best = max(names, key=lambda name: profiles[name]['metrics']['harmonic_accuracy'])
    return {'anchor': 'calibrated_BiMC', 'fixed_score_calibration': calibration,
            'prediction_scope': 'each_expert_predicts_over_all_seen_classes', 'experts': profiles,
            'pairwise_correctness': pairs, 'count_unit': 'repeated_stage_query_observations',
            'best_fixed_expert_by_development_H': {'name': best, 'metrics': profiles[best]['metrics'],
                'selection_source': 'same_development_bank; not_an_independent_performance_estimate'},
            'label_oracle_union': {'metrics': union, 'experts': list(names),
                'H_gap_vs_best_fixed_expert_pp': union['harmonic_accuracy'] - profiles[best]['metrics']['harmonic_accuracy'],
                'uses_query_labels_to_choose_expert': True, 'deployable': False,
                'strict_mathematical_upper_bound_for_other_methods': False},
            'selection_candidate': False,
            'limitations': ['An oracle union gap does not imply a realizable gain for a learned router.',
                'A single covariance/description expert uses all classes, unlike the group-specific BiMC mixture.',
                'Stage-query observations overlap across stages and sequences.']}
