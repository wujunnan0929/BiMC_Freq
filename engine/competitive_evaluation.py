"""Full BiMC fixed-fusion controls and group/within-group error diagnostics."""
import itertools

import numpy as np
import torch

from engine.prototype_evaluation import branch_logits, evaluate_records, score_record, validate_calibration
from models.competitive_fusion import branch_probabilities, fixed_scores, competitive_scores, support_quality, validate_fixed


def identity_fixed(cfg):
    alpha = float(cfg.DATASET.ENSEMBLE_ALPHA) if cfg.TRAINER.BiMC.USING_ENSEMBLE else 1.
    return {'alpha_old': alpha, 'alpha_novel': alpha, 'novel_bias_delta': 0.}


@torch.no_grad()
def prepare_records(records, cfg, calibration):
    for row in records:
        row['fusion_probabilities'] = branch_probabilities(branch_logits(row, cfg), calibration)
        row['fusion_calibration'] = dict(calibration)
        row['fusion_quality'] = support_quality(row['supports'], row['reference_prototypes'], row['semantic_prototypes'])
    return records


def fusion_record_scores(row, cfg, calibration, fixed, model=None, blend=0.):
    if row.get('fusion_calibration') != calibration:
        raise ValueError('Cached fusion probabilities use different calibration.')
    anchor = (score_record(row, cfg, calibration=calibration) if fixed == identity_fixed(cfg) else
              fixed_scores(row['fusion_probabilities'], row['num_old'], fixed, calibration))
    if model is None or blend == 0:
        return anchor
    return competitive_scores(model, row['fusion_probabilities'], row['num_old'], row['fusion_quality'],
                              fixed, calibration, blend, anchor=anchor)


def feasible(metrics, references, maximum_drop):
    return all(metrics[k] >= ref[k] - maximum_drop - 1e-12 for ref in references
               for k in ('base_accuracy', 'novel_accuracy'))


@torch.no_grad()
def select_fixed_fusion(records, cfg, calibration, maximum_drop=.2,
                        weight_grid=(0., .25, .5, .75, .8, 1.), bias_grid=(0., -.2, -.1, .1, .2)):
    validate_calibration(calibration)
    if not np.isfinite(maximum_drop) or maximum_drop < 0:
        raise ValueError('maximum_drop must be finite and nonnegative.')
    original = evaluate_records(records, lambda row: row['reference_votes'])
    calibrated = evaluate_records(records, lambda row: score_record(row, cfg, calibration=calibration))
    definitions = [identity_fixed(cfg)]
    for a, b, bias in itertools.product(weight_grid, weight_grid, bias_grid):
        definition = validate_fixed({'alpha_old': a, 'alpha_novel': b, 'novel_bias_delta': bias})
        if definition not in definitions:
            definitions.append(definition)
    best, best_metrics = definitions[0], calibrated
    candidates = []
    for definition in definitions:
        metrics = evaluate_records(records, lambda row: fusion_record_scores(row, cfg, calibration, definition))
        ok = feasible(metrics, (original, calibrated), maximum_drop)
        candidates.append({'fixed': definition, 'metrics': metrics, 'feasible': ok})
        if ok and metrics['harmonic_accuracy'] > best_metrics['harmonic_accuracy'] + 1e-12:
            best, best_metrics = definition, metrics
    return best, {'reference': original, 'calibrated_reference': calibrated,
        'selected_fixed': best, 'selected_validation': best_metrics, 'candidates': candidates,
        'candidate_count': len(candidates), 'maximum_group_drop_pp': maximum_drop,
        'selection_source': 'original_base_class_disjoint_validation_only',
        'selection_rule': 'max mean-stage H; guard against original and calibrated B/N; identity wins ties',
        'teacher_or_query_label_router_candidates': False}


@torch.no_grad()
def group_diagnostics(records, score_fn):
    """Targets are read only for reporting errors, never for producing scores."""
    if not records:
        raise ValueError('No validation stages for competition diagnostics.')
    stages, counts = [], dict.fromkeys(('queries', 'old_queries', 'novel_queries', 'old_to_novel',
                'novel_to_old', 'old_within_group_errors', 'novel_within_group_errors'), 0)
    for row in records:
        prediction = score_fn(row).argmax(-1)
        y, old_count = row['targets'], row['num_old']
        old, pred_old = y < old_count, prediction < old_count
        wrong = prediction != y
        masks = {'old_to_novel': old & ~pred_old, 'novel_to_old': ~old & pred_old,
                 'old_within_group_errors': old & pred_old & wrong,
                 'novel_within_group_errors': ~old & ~pred_old & wrong}
        record = {'sequence': row.get('sequence'), 'stage': row.get('stage'),
                  'old_classes': old_count, 'novel_classes': len(row['reference_prototypes'])-old_count}
        counts['queries'] += len(y)
        counts['old_queries'] += int(old.sum()); counts['novel_queries'] += int((~old).sum())
        for name, mask in masks.items():
            number = int(mask.sum())
            counts[name] += number
            denominator = old.sum() if name.startswith('old_') else (~old).sum()
            record[name + '_rate'] = 100. * number / int(denominator)
        evidence = {}
        for name, p in row['fusion_probabilities'].items():
            mass = p[:, :old_count].sum(-1)
            evidence[name] = {'old_mass_on_old_queries': float(mass[old].mean()),
                              'old_mass_on_novel_queries': float(mass[~old].mean()),
                              'top1_on_old_queries': float(p[old].max(-1).values.mean()),
                              'top1_on_novel_queries': float(p[~old].max(-1).values.mean())}
        record['branch_evidence'] = evidence
        stages.append(record)
    return {'count_unit': 'repeated_stage_query_observations', 'counts': counts,
            'mean_stage_rates': {name+'_rate': float(np.mean([r[name+'_rate'] for r in stages])) for name in masks},
            'stages': stages, 'uses_labels_for_evaluation_only': True}
