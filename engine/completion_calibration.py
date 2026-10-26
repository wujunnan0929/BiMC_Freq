"""Base-only selection and frozen support-only statistical prototype completion.

Selection fits covariance/PCA on A classes, excluding globally reserved A
queries. V classes provide support/query validation only. After selecting a
blend, statistics are refitted on all original base training images. No query
image, benchmark label, or historical support image enters incremental fitting.
"""

import hashlib
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from engine.guidance_calibration import digest, sample_protocol, _text_state
from engine.residual_training import _group_covariance
from models.prototype_completion import fit_completion, complete_prototypes


def validate_completion_settings(cfg):
    p, b = cfg.TRAINER.BiMC.PROTOTYPE_COMPLETION, cfg.TRAINER.BiMC
    if not p.ENABLED:
        return
    if b.PREC != 'fp32':
        raise ValueError('PROTOTYPE_COMPLETION requires fp32 frozen CLIP features.')
    if any((b.FREQUENCY.ENABLED, b.FREQUENCY.ROUTER.ENABLED, b.RESIDUAL.ENABLED,
            b.CONSENSUS.ENABLED, b.UNCERTAINTY.ENABLED, b.DUAL_FREQUENCY.ENABLED,
            b.GUIDANCE.ENABLED, b.SAFD_METRIC.ENABLED,
            b.PROTOTYPE_ESTIMATOR.ENABLED, b.COMPETITIVE_FUSION.ENABLED)):
        raise ValueError('PROTOTYPE_COMPLETION requires other optional branches disabled.')
    if p.VIEW_CONTROL not in ('frequency', 'original', 'augmentation', 'repeat'):
        raise ValueError('Unsupported PROTOTYPE_COMPLETION.VIEW_CONTROL.')
    if p.SEMANTIC_SOURCE not in ('structured', 'ordinary', 'explicit'):
        raise ValueError('Unsupported PROTOTYPE_COMPLETION.SEMANTIC_SOURCE.')
    if p.AUXILIARY_INDEX not in (1, 2):
        raise ValueError('PROTOTYPE_COMPLETION.AUXILIARY_INDEX must be 1 or 2.')
    if sorted(p.SEMANTIC_PERMUTATION) != [0, 1, 2]:
        raise ValueError('SEMANTIC_PERMUTATION must permute the three semantic scales.')
    for name in ('RANK', 'CANDIDATES_PER_BAND', 'VAL_SEQUENCES', 'SHOT', 'QUERY'):
        value = getattr(p, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('PROTOTYPE_COMPLETION.' + name + ' must be a positive integer.')
    if p.SHOT != cfg.DATASET.NUM_INC_SHOT:
        raise ValueError('PROTOTYPE_COMPLETION.SHOT must match DATASET.NUM_INC_SHOT.')
    for name in ('SHRINKAGE', 'RIDGE', 'BLEND', 'MAX_GROUP_DROP_PP'):
        value = getattr(p, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('PROTOTYPE_COMPLETION.' + name + ' must be finite.')
    if not 0 <= p.SHRINKAGE <= 1 or p.RIDGE <= 0 or not 0 <= p.BLEND <= 1 or p.MAX_GROUP_DROP_PP < 0:
        raise ValueError('Invalid prototype completion covariance/blend settings.')
    if (not p.BLEND_GRID or 0 not in p.BLEND_GRID
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) or not 0 <= v <= 1 for v in p.BLEND_GRID)):
        raise ValueError('PROTOTYPE_COMPLETION.BLEND_GRID must include zero and values in [0,1].')


def completion_protocol(cfg, labels, sample_ids):
    """Return paired class-disjoint tasks, with A queries excluded from all fitting."""
    p = cfg.TRAINER.BiMC.PROTOTYPE_COMPLETION
    local = cfg.clone()
    local.defrost()
    g = local.TRAINER.BiMC.GUIDANCE
    g.CALIBRATION_SEED = p.CALIBRATION_SEED
    g.VAL_SEQUENCES, g.SHOT, g.QUERY, g.AUDIT_IMAGES = p.VAL_SEQUENCES, p.SHOT, p.QUERY, 0
    local.freeze()
    protocol = sample_protocol(local, labels, sample_ids)
    a = protocol['pseudo_base_class_ids']
    reserved = {c: list(rows) for c, rows in zip(a, protocol['sequences'][0]['query_indices'][:len(a)])}
    pool = {c: [int(i) for i in torch.where(labels.cpu() == c)[0].tolist()
                if int(i) not in set(reserved[c])] for c in a}
    for seq in protocol['sequences']:
        for j, c in enumerate(a):
            seq['support_indices'][j], seq['query_indices'][j] = list(pool[c]), list(reserved[c])
    protocol.update(fitting_indices=[i for c in a for i in pool[c]],
                    reserved_pseudo_base_query_indices={str(c): rows for c, rows in reserved.items()},
                    fitting_rule='A_classes_only_excluding_globally_reserved_A_queries',
                    selection_class_count_note='Half-base-class tasks; no claim of matching full benchmark class count.')
    protocol.pop('sha256', None)
    protocol['sha256'] = digest(protocol)
    return protocol


def _fit(cfg, current, indices=None, classes=None):
    p = cfg.TRAINER.BiMC.PROTOTYPE_COMPLETION
    ids = [int(c) for c in current['class_index']]
    classes = ids if classes is None else list(classes)
    positions = [ids.index(c) for c in classes]
    x, y = current['completion_features'], current['images_targets']
    if indices is not None:
        x, y = x[indices], y[indices]
    return fit_completion(x, y, current['completion_text'][positions], classes,
                          rank=p.RANK, shrinkage=p.SHRINKAGE, ridge=p.RIDGE,
                          cross_view=p.CROSS_VIEW)


def blend_completion(reference, raw, completed, blend):
    """Apply a unit-direction correction to exactly the same BiMC reference.

    The zero path deliberately returns the reference unchanged (no normalize).
    This isolates completion from the original visual calibration mechanism.
    """
    if blend == 0:
        return reference
    delta = F.normalize(completed.float(), dim=-1) - F.normalize(raw.float(), dim=-1)
    corrected = reference.float() + float(blend) * delta
    if not torch.isfinite(corrected).all() or (corrected.norm(dim=-1) <= 1e-12).any():
        raise ValueError('Completion produced a non-finite or zero prototype.')
    return F.normalize(corrected, dim=-1).to(reference)


@torch.no_grad()
def completion_sequence_records(model, cfg, current, sequence, fitted):
    """Build support-only candidates before reading any validation queries."""
    x, z, y = current['images_features'], current['completion_features'], current['images_targets']
    classes, supports = sequence['class_ids'], sequence['support_indices']
    old = len(classes) - sum(sequence['stage_sizes'])
    raw = torch.stack([x[rows].mean(0) for rows in supports])
    reference = F.normalize(raw, dim=-1)
    if cfg.TRAINER.BiMC.VISION_CALIBRATION:
        reference[old:] = model.soft_calibration(reference[:old], reference[old:])
    novel_rows = [i for rows in supports[old:] for i in rows]
    positions = {int(c): i for i, c in enumerate(current['class_index'])}
    anchors = current['completion_text'][[positions[c] for c in classes[old:]]]
    completed = complete_prototypes(fitted, z[novel_rows], y[novel_rows], anchors, classes[old:])
    covariance_sum = old * _group_covariance(
        x[[i for rows in supports[:old] for i in rows]], cfg.TRAINER.BiMC.GAMMA_BASE)
    start = old
    for stage, size in enumerate(sequence['stage_sizes'], 1):
        end = start + size
        covariance_sum = covariance_sum + size * _group_covariance(
            x[[i for rows in supports[start:end] for i in rows]], cfg.TRAINER.BiMC.GAMMA_INC)
        query_rows = [i for rows in sequence['query_indices'][:end] for i in rows]
        target = torch.cat([torch.full((len(rows),), j, dtype=torch.long, device=x.device)
                            for j, rows in enumerate(sequence['query_indices'][:end])])
        yield {'stage': stage, 'class_ids': classes[:end], 'num_base': old,
               'reference': reference[:end], 'raw': raw[:end],
               'completed': completed[:end-old], 'query': x[query_rows], 'targets': target,
               'covariance': covariance_sum / end,
               'text': _text_state(current, classes[:end], x.device),
               'support_indices': supports[:end], 'query_indices': sequence['query_indices'][:end]}
        start = end


def select_completion_candidate(candidates, maximum_drop):
    reference = next(c for c in candidates if c['blend'] == 0)
    for c in candidates:
        c['base_delta_pp'] = c['mean_base_accuracy'] - reference['mean_base_accuracy']
        c['novel_delta_pp'] = c['mean_novel_accuracy'] - reference['mean_novel_accuracy']
        c['overall_delta_pp'] = c['mean_accuracy'] - reference['mean_accuracy']
        c['harmonic_delta_pp'] = c['mean_harmonic_accuracy'] - reference['mean_harmonic_accuracy']
        c['feasible'] = min(c['base_delta_pp'], c['novel_delta_pp']) >= -maximum_drop - 1e-12
    feasible = [c for c in candidates if c['feasible']]
    best = max(c['mean_accuracy'] for c in feasible)
    return min((c for c in feasible if c['mean_accuracy'] >= best - 1e-12), key=lambda c: c['blend'])


@torch.no_grad()
def _select(model, cfg, current, protocol, fitted):
    p = cfg.TRAINER.BiMC.PROTOTYPE_COMPLETION
    grid = sorted(set(float(v) for v in p.BLEND_GRID))
    results = {v: [] for v in grid}
    for index, sequence in enumerate(protocol['sequences']):
        for rec in completion_sequence_records(model, cfg, current, sequence, fitted):
            old = rec['num_base']
            inverse = torch.pinverse(rec['covariance'].float())
            predictions = {}
            for value in grid:
                prototypes = torch.cat((rec['reference'][:old], blend_completion(
                    rec['reference'][old:], rec['raw'][old:], rec['completed'], value)))
                t = rec['text']
                scores = model.reference_scores_from_features(
                    rec['query'], len(prototypes), old, prototypes, rec['covariance'],
                    t['description_proto'], t['description_features'], t['description_targets'],
                    t['text_features'], cfg.DATASET.BETA, cov_inverse=inverse)
                if not torch.isfinite(scores).all():
                    raise ValueError('Non-finite prototype completion validation scores.')
                predictions[value] = scores.argmax(-1)
            ref_correct = predictions[0.] == rec['targets']
            for value in grid:
                correct = predictions[value] == rec['targets']
                mask = rec['targets'] < old
                a = 100 * correct[mask].float().mean().item()
                n = 100 * correct[~mask].float().mean().item()
                results[value].append({
                    'sequence': index, 'stage': rec['stage'],
                    'accuracy': 100 * correct.float().mean().item(),
                    'base_accuracy': a, 'novel_accuracy': n,
                    'harmonic_accuracy': 2 * a * n / (a + n) if a + n else 0.,
                    'query_occurrences': len(correct),
                    'corrected': int((correct & ~ref_correct).sum()),
                    'damaged': int((~correct & ref_correct).sum()),
                })
        print(f'Prototype completion base validation {index+1}/{len(protocol["sequences"])}', flush=True)
    candidates = []
    for value, rows in results.items():
        row = {'blend': value, 'stages': rows,
               'corrected_observations': sum(r['corrected'] for r in rows),
               'damaged_observations': sum(r['damaged'] for r in rows)}
        for name in ('accuracy', 'base_accuracy', 'novel_accuracy', 'harmonic_accuracy'):
            row['mean_' + name] = sum(r[name] for r in rows) / len(rows)
        candidates.append(row)
    return select_completion_candidate(candidates, p.MAX_GROUP_DROP_PP), candidates


@torch.no_grad()
def initialize_completion(model, cfg, current):
    validate_completion_settings(cfg)
    if getattr(model, 'completion_state', None) is not None:
        raise RuntimeError('Initialize prototype completion only once, in the base session.')
    ids = [int(c) for c in current['class_index']]
    if (current.get('data_source') != 'base_training_only' or current.get('task_id') != 0
            or ids != list(range(cfg.DATASET.NUM_INIT_CLS))
            or sorted(int(c) for c in torch.unique(current['images_targets'])) != ids):
        raise ValueError('Prototype completion requires the complete original base training session.')
    p = cfg.TRAINER.BiMC.PROTOTYPE_COMPLETION
    start = time.perf_counter()
    seed = int(p.CALIBRATION_SEED if p.CALIBRATION_SEED >= 0 else cfg.SEED + 10101)
    report = {'schema_version': 1, 'method': 'joint_statistical_prototype_completion',
              'data_source': 'base_training_only', 'seed': seed, 'run_seed': int(cfg.SEED),
              'auto_calibrate': bool(p.AUTO_CALIBRATE), 'view_control': p.VIEW_CONTROL,
              'semantic_source': p.SEMANTIC_SOURCE, 'semantic_permutation': list(p.SEMANTIC_PERMUTATION),
              'auxiliary_index': int(p.AUXILIARY_INDEX), 'cross_view': bool(p.CROSS_VIEW),
              'rank_requested': int(p.RANK), 'shrinkage': float(p.SHRINKAGE), 'ridge': float(p.RIDGE),
              'selection_objective': 'mean_incremental_stage_overall_accuracy',
              'maximum_mean_group_drop_pp': float(p.MAX_GROUP_DROP_PP),
              'candidate_results': [], 'protocol_sha256': None,
              'fit_class_ids': [], 'validation_class_ids': [],
              'query_auxiliary_encodings': 0, 'gradient_training_performed': False,
              'incremental_trainable_parameters': 0,
              'assumption': 'Empirical Gaussian approximation in base-fitted low-rank joint view coordinates.'}
    blend, fitted = float(p.BLEND), None
    if p.AUTO_CALIBRATE:
        sample_ids = [int(i) for i in model._last_support_ids]
        protocol = completion_protocol(cfg, current['images_targets'], sample_ids)
        a = protocol['pseudo_base_class_ids']
        fitted = _fit(cfg, current, protocol['fitting_indices'], a)
        winner, candidates = _select(model, cfg, current, protocol, fitted)
        blend = winner['blend']
        report.update(selection_status='base_class_disjoint_validation', protocol=protocol,
                      protocol_sha256=protocol['sha256'], candidate_results=candidates,
                      fit_class_ids=a, validation_class_ids=protocol['pseudo_validation_class_ids'],
                      validation_query_occurrences=sum(s['query_occurrences'] for s in winner['stages']))
    else:
        report['selection_status'] = 'manual_no_validation'
    # With a zero selected correction, storing/fitting unused statistics is unnecessary.
    fitted = _fit(cfg, current) if blend > 0 else None
    model.completion_state = {'blend': blend, 'statistics': fitted,
                              'base_class_ids': ids, 'gradient_training_performed': False}
    report.update(selected_blend=blend, fallback_to_BiMC=blend == 0,
                  refit_source='all_original_base_training_support' if fitted is not None else 'skipped_zero_blend',
                  refit_class_ids=ids if fitted is not None else [],
                  retained_training_samples=0,
                  retained_statistic_scalars=(sum(v.numel() for v in fitted.values()
                                                  if torch.is_tensor(v)) if fitted is not None else 0),
                  fitted_rank=int(fitted['rank']) if fitted is not None else None,
                  fit_seconds=time.perf_counter() - start)
    report['calibration_seconds'] = report['fit_seconds']
    report['provenance'] = {
        'clip_weight_sha256': getattr(model.clip_model, 'clip_weight_sha256', None),
        'config_sha256': hashlib.sha256(cfg.dump().encode()).hexdigest(),
        'base_sample_ids_sha256': digest([int(i) for i in model._last_support_ids]),
    }
    descriptions = {'ordinary': cfg.DATASET.GPT_PATH}
    if p.SEMANTIC_SOURCE == 'explicit' and p.VIEW_CONTROL != 'original':
        descriptions['frequency'] = cfg.TRAINER.BiMC.FREQUENCY.EXPLICIT_DESCRIPTION_PATH
    report['provenance']['description_files'] = {
        name: {'path': str(path), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        for name, path in descriptions.items() if path and Path(path).is_file()
    }
    views = getattr(model, 'completion_views', None)
    if views is not None and hasattr(views, 'specification'):
        report['view_specification'] = views.specification()
    return report


@torch.no_grad()
def prepare_completion(model, cfg, current):
    state = getattr(model, 'completion_state', None)
    if state is None:
        raise RuntimeError('Initialize completion on base training data before novel adaptation.')
    if current.get('task_id') == 0 or state['blend'] == 0:
        return
    if current.get('data_source') != 'incremental_training_only':
        raise ValueError('Completion adaptation accepts current incremental support only.')
    ids = [int(c) for c in current['class_index']]
    if any(c in state['base_class_ids'] for c in ids):
        raise ValueError('Incremental completion cannot update original base classes.')
    x, y = current['completion_features'], current['images_targets']
    posterior = complete_prototypes(state['statistics'], x, y, current['completion_text'], ids)
    raw = torch.stack([x[y == c, 0].mean(0) for c in ids])
    current['image_proto'] = blend_completion(current['image_proto'], raw, posterior, state['blend']).detach()
