"""Calibrate diagonal predictive uncertainty using base training classes only.

The prior is fitted on classes disjoint from calibration episodes. Episode class
statistics and the BiMC reference use support images only. Once hyperparameters
are selected, only the pooled prior is refitted on all base training images.
"""

import hashlib
import json
import math

import torch

from engine.consensus_calibration import build_support_reference, sample_sequence
from models.frequency_uncertainty import (
    class_statistics, mix_uncertainty_probabilities, pooled_variance,
    uncertainty_logits,
)


GROUPS = ('all', 'old', 'new', 'historical_incremental', 'current_new')


def _number(value, name, *, lower=0., upper=None, strict=False):
    if isinstance(value, bool):
        raise ValueError(name + ' must be a finite number.')
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(name + ' must be a finite number.') from error
    if (not math.isfinite(value) or (value <= lower if strict else value < lower)
            or (upper is not None and value > upper)):
        raise ValueError(name + ' is outside its valid range.')
    return value


def _grid(values, name, **limits):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(name + ' must be a nonempty list.')
    return sorted(set(_number(value, name, **limits) for value in values))


def _validate_settings(settings):
    if settings.VIEW_CONTROL not in ('frequency', 'original'):
        raise ValueError('UNCERTAINTY.VIEW_CONTROL must be frequency or original.')
    if settings.COVARIANCE not in ('shrinkage', 'shared'):
        raise ValueError('UNCERTAINTY.COVARIANCE must be shrinkage or shared.')
    for name in ('AUTO_CALIBRATE', 'MEAN_UNCERTAINTY'):
        if not isinstance(getattr(settings, name), bool):
            raise ValueError('UNCERTAINTY.' + name + ' must be boolean.')
    for name in ('VAL_EPISODES', 'OLD_WAY', 'NEW_WAY', 'STAGES', 'OLD_SHOT', 'SHOT', 'QUERY'):
        value = getattr(settings, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('UNCERTAINTY.' + name + ' must be a positive integer.')
    seed = settings.CALIBRATION_SEED
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < -1:
        raise ValueError('CALIBRATION_SEED must be -1 or a nonnegative integer.')
    _number(settings.PRIOR_STRENGTH, 'PRIOR_STRENGTH')
    _number(settings.VAR_FLOOR, 'VAR_FLOOR', strict=True)
    _number(settings.ALPHA, 'ALPHA', upper=1.)
    _number(settings.TEMPERATURE, 'TEMPERATURE', strict=True)
    fraction = _number(settings.FIT_FRACTION, 'FIT_FRACTION', strict=True)
    if fraction >= 1:
        raise ValueError('FIT_FRACTION must lie strictly between zero and one.')
    priors = _grid(settings.PRIOR_GRID, 'PRIOR_GRID')
    alphas = _grid(settings.ALPHA_GRID, 'ALPHA_GRID', upper=1.)
    temperatures = _grid(settings.TEMPERATURE_GRID, 'TEMPERATURE_GRID', strict=True)
    if 0. not in alphas:
        raise ValueError('ALPHA_GRID must include zero as the unchanged reference.')
    return priors, alphas, temperatures


def _validate_base_state(cfg, state):
    required = ('images_features', 'images_targets', 'uncertainty_features', 'class_index')
    if any(name not in state for name in required):
        raise ValueError('Base training state is missing required uncertainty calibration data.')
    features, labels, auxiliary = (state[name] for name in required[:3])
    if (not all(torch.is_tensor(value) for value in (features, labels, auxiliary))
            or features.ndim != 2 or labels.ndim != 1 or auxiliary.ndim != 3
            or len(labels) == 0 or features.shape[0] != len(labels)
            or auxiliary.shape[0] != len(labels) or min(auxiliary.shape[1:]) < 1
            or features.shape[1] < 1):
        raise ValueError('Expected aligned base features [N,D], labels [N], auxiliary [N,B,D].')
    if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError('Base-class training labels must be integers.')
    if not torch.isfinite(features).all() or not torch.isfinite(auxiliary).all():
        raise ValueError('Base training features must be finite.')
    if features.device != labels.device or features.device != auxiliary.device:
        raise ValueError('Base training tensors must share one device.')
    expected_bands = 1 if cfg.TRAINER.BiMC.UNCERTAINTY.VIEW_CONTROL == 'original' else 3
    if auxiliary.shape[1] != expected_bands:
        raise ValueError('Auxiliary band count does not match UNCERTAINTY.VIEW_CONTROL.')
    class_ids = sorted(int(value) for value in torch.unique(labels))
    try:
        provided = [int(value) for value in state['class_index']]
        if any(float(value) != converted for value, converted in zip(state['class_index'], provided)):
            raise ValueError
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('class_index must contain integer base-class identifiers.') from error
    if (len(provided) != len(set(provided)) or sorted(provided) != class_ids
            or any(value < 0 or value >= cfg.DATASET.NUM_INIT_CLS for value in class_ids)):
        raise ValueError('Calibration accepts only matching base-class training samples.')
    # Explicit provenance is honored when a caller provides it. This entry point
    # never loads files, test sets, or future sessions to fill in missing data.
    if state.get('data_source', 'base_training_only') not in ('base_training_only', 'base_train', 'train'):
        raise ValueError('Calibration accepts base training data only.')
    if 'task_id' in state and state['task_id'] != 0:
        raise ValueError('Uncertainty initialization is allowed only at the base session.')
    return features.detach().float(), labels, auxiliary.detach().float(), class_ids


def _fit_prior(auxiliary, labels, class_ids, var_floor):
    mask = torch.isin(labels, labels.new_tensor(class_ids))
    stats = class_statistics(features=auxiliary[mask], labels=labels[mask], class_ids=class_ids)
    prior = pooled_variance(stats=stats, var_floor=var_floor).detach()
    if prior.shape != auxiliary.shape[1:] or not torch.isfinite(prior).all():
        raise ValueError('The fitted uncertainty prior must be finite with shape [B,D].')
    return prior


def _sequence_records(model, cfg, state, sequence):
    """Yield support-only statistics with independently held-out query features."""
    settings = cfg.TRAINER.BiMC.UNCERTAINTY
    features, labels = state['images_features'], state['images_targets']
    auxiliary = state['uncertainty_features']
    for stage in range(settings.STAGES + 1):
        seen = settings.OLD_WAY + stage * settings.NEW_WAY
        selected = sequence['class_ids'][:seen]
        support = [index for row in sequence['support_indices'][:seen] for index in row]
        query = [index for row in sequence['query_indices'][:seen] for index in row]
        if (len(set(support)) != len(support) or len(set(query)) != len(query)
                or set(support).intersection(query)):
            raise ValueError('Calibration support and query samples must be distinct without replacement.')
        support_indices = torch.tensor(support, device=labels.device, dtype=torch.long)
        query_indices = torch.tensor(query, device=labels.device, dtype=torch.long)
        scorer, _, _, reference_state = build_support_reference(
            model, cfg, state, support, selected, settings.OLD_WAY, settings.NEW_WAY,
            include_frequency=False,
        )
        stats = class_statistics(features=auxiliary[support_indices].float(),
                                 labels=labels[support_indices], class_ids=selected)
        local = {class_id: position for position, class_id in enumerate(selected)}
        targets = labels.new_tensor([local[int(value)] for value in labels[query_indices]])
        yield {'stage': stage, 'scores': scorer(features[query_indices].float()),
               'features': auxiliary[query_indices].float(), 'stats': stats,
               'targets': targets, 'reference_state': reference_state}


def _empty_counts():
    return dict.fromkeys(('count', 'correct', 'reference_correct', 'corrected', 'damaged',
                          'changed', 'wrong_to_wrong'), 0)


def _update_counts(counts, mask, before, after, changed):
    for name, values in (
        ('count', mask), ('correct', mask & after), ('reference_correct', mask & before),
        ('corrected', mask & ~before & after), ('damaged', mask & before & ~after),
        ('changed', mask & changed), ('wrong_to_wrong', mask & ~before & ~after & changed),
    ):
        counts[name] += int(values.sum())


def _summarize(counts):
    count = counts['count']
    return {**counts, 'accuracy': 100 * counts['correct'] / count if count else None,
            'reference_accuracy': 100 * counts['reference_correct'] / count if count else None,
            'delta_accuracy_pp': 100 * (counts['corrected'] - counts['damaged']) / count
            if count else None}


def _candidate_grid(settings, priors, alphas, temperatures):
    # Shared covariance ignores support sample variances and prior strength.
    if settings.COVARIANCE == 'shared':
        priors = [float(settings.PRIOR_STRENGTH)]
    # Alpha zero is exactly BiMC, so neither its strength nor temperature matter.
    candidates = [{'prior_strength': float(settings.PRIOR_STRENGTH),
                   'alpha': 0., 'temperature': float(settings.TEMPERATURE)}]
    candidates.extend({'prior_strength': prior, 'alpha': alpha, 'temperature': temperature}
                      for alpha in alphas if alpha > 0
                      for prior in priors for temperature in temperatures)
    return candidates


def _evaluate(model, cfg, state, sequences, prior, candidates):
    settings = cfg.TRAINER.BiMC.UNCERTAINTY
    stages = [[{'stage': stage, 'candidate_classes': settings.OLD_WAY + stage * settings.NEW_WAY,
                'groups': {name: _empty_counts() for name in GROUPS}}
               for stage in range(settings.STAGES + 1)] for _ in candidates]
    for sequence in sequences:
        for record in _sequence_records(model, cfg, state, sequence):
            reference, targets = record['scores'], record['targets']
            stage = record['stage']
            original_prediction = reference.argmax(1)
            before = original_prediction == targets
            old = targets < settings.OLD_WAY
            current = (~old & (targets >= settings.OLD_WAY + (stage - 1) * settings.NEW_WAY)
                       if stage else torch.zeros_like(old))
            masks = {'all': torch.ones_like(old), 'old': old, 'new': ~old,
                     'historical_incremental': ~old & ~current, 'current_new': current}
            logits_by_prior = {}
            for index, candidate in enumerate(candidates):
                strength = candidate['prior_strength']
                if candidate['alpha'] == 0:
                    output = reference
                else:
                    if strength not in logits_by_prior:
                        logits_by_prior[strength] = uncertainty_logits(
                            features=record['features'], stats=record['stats'], prior=prior,
                            prior_strength=strength, var_floor=settings.VAR_FLOOR,
                            covariance=settings.COVARIANCE, mean_uncertainty=settings.MEAN_UNCERTAINTY,
                        )
                    output = mix_uncertainty_probabilities(
                        reference=reference, logits=logits_by_prior[strength],
                        alpha=candidate['alpha'], temperature=candidate['temperature'],
                    )
                prediction = output.argmax(1)
                after, changed = prediction == targets, prediction != original_prediction
                for name, mask in masks.items():
                    _update_counts(stages[index][stage]['groups'][name], mask, before, after, changed)
    results = []
    for candidate, stage_rows in zip(candidates, stages):
        for row in stage_rows:
            row['groups'] = {name: _summarize(counts) for name, counts in row['groups'].items()}
        balanced = []
        for row in stage_rows[1:]:
            if any(row['groups'][name]['count'] <= 0 for name in ('old', 'new')):
                raise ValueError('Balanced calibration requires old and new queries at every stage.')
            balanced.append(sum(row['groups'][name]['accuracy'] for name in ('old', 'new')) / 2)
        totals = {name: sum(row['groups']['all'][name] for row in stage_rows)
                  for name in _empty_counts()}
        results.append({**candidate, **_summarize(totals), 'stages': stage_rows,
                        'balanced_objective': sum(balanced) / len(balanced)})
    return results


@torch.no_grad()
def initialize_uncertainty(model, cfg, base_state):
    """Fit/select once; return a finite JSON audit, storing tensors on the model."""
    if getattr(model, 'uncertainty_state', None) is not None:
        raise RuntimeError('Uncertainty statistics must be initialized once, at base only.')
    settings = cfg.TRAINER.BiMC.UNCERTAINTY
    priors, alphas, temperatures = _validate_settings(settings)
    if (getattr(model, 'frequency_fusion_enabled', False)
            or cfg.TRAINER.BiMC.RESIDUAL.ENABLED or cfg.TRAINER.BiMC.CONSENSUS.ENABLED):
        raise ValueError('Uncertainty calibration requires a plain BiMC reference.')
    _, labels, auxiliary, class_ids = _validate_base_state(cfg, base_state)
    seed = settings.CALIBRATION_SEED if settings.CALIBRATION_SEED >= 0 else int(cfg.SEED) + 10101
    report = {'schema_version': 1, 'data_source': 'base_training_only',
              'seed': seed, 'run_seed': int(cfg.SEED), 'auto_calibrate': settings.AUTO_CALIBRATE,
              'view_control': settings.VIEW_CONTROL, 'covariance': settings.COVARIANCE,
              'mean_uncertainty': settings.MEAN_UNCERTAINTY, 'var_floor': float(settings.VAR_FLOOR),
              'selection_objective': 'balanced_incremental',
              'objective_definition': 'Equal weight for old and all incremental classes at each '
                                      'incremental stage, then equal weight across stages; stage 0 excluded.',
              'tie_break': 'Smaller alpha, then smaller prior_strength, then smaller temperature.',
              'base_class_ids': class_ids, 'base_sample_count': len(labels),
              'fit_class_ids': [], 'validation_class_ids': [], 'validation_sequences': [],
              'validation_query_occurrences': 0, 'candidate_results': []}
    if settings.AUTO_CALIBRATE:
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(class_ids), generator=generator).tolist()
        split = int(len(class_ids) * settings.FIT_FRACTION)
        fit_ids = [class_ids[index] for index in order[:split]]
        validation_ids = [class_ids[index] for index in order[split:]]
        if not fit_ids or not validation_ids:
            raise ValueError('Class-disjoint prior fitting and validation need nonempty splits.')
        required = settings.OLD_WAY + settings.NEW_WAY * settings.STAGES
        if len(validation_ids) < required:
            raise ValueError(f'Uncertainty validation needs {required} classes, got {len(validation_ids)}.')
        # Every validation class can be sampled in either role across episodes.
        minimum = max(settings.OLD_SHOT, settings.SHOT) + settings.QUERY
        for class_id in validation_ids:
            if int((labels == class_id).sum()) < minimum:
                raise ValueError(f'Base class {class_id} needs {minimum} distinct support/query images.')
        fit_prior = _fit_prior(auxiliary, labels, fit_ids, settings.VAR_FLOOR)
        sequences = [sample_sequence(labels, validation_ids, settings, generator)
                     for _ in range(settings.VAL_EPISODES)]
        candidates = _candidate_grid(settings, priors, alphas, temperatures)
        results = _evaluate(model, cfg, base_state, sequences, fit_prior, candidates)
        best = max(candidate['balanced_objective'] for candidate in results)
        winner = min((candidate for candidate in results
                      if candidate['balanced_objective'] >= best - 1e-12),
                     key=lambda candidate: (candidate['alpha'], candidate['prior_strength'],
                                            candidate['temperature']))
        selected = {key: winner[key] for key in ('prior_strength', 'alpha', 'temperature')}
        report.update(selection_status='base_class_disjoint_validation',
                      fit_class_ids=fit_ids, validation_class_ids=validation_ids,
                      fit_sample_count=int(torch.isin(labels, labels.new_tensor(fit_ids)).sum()),
                      selection_prior=fit_prior.cpu().tolist(), validation_sequences=sequences,
                      validation_query_occurrences=winner['count'], candidate_results=results,
                      selected_balanced_objective=winner['balanced_objective'],
                      reference_accuracy=results[0]['accuracy'])
    else:
        selected = {'prior_strength': float(settings.PRIOR_STRENGTH),
                    'alpha': float(settings.ALPHA), 'temperature': float(settings.TEMPERATURE)}
        report.update(selection_status='manual_unvalidated', fit_class_ids=class_ids,
                      fit_sample_count=len(labels), selected_balanced_objective=None,
                      reference_accuracy=None)
    prior = _fit_prior(auxiliary, labels, class_ids, settings.VAR_FLOOR)
    report.update(selected=selected, selected_alpha=selected['alpha'],
                  selected_prior_strength=selected['prior_strength'],
                  selected_temperature=selected['temperature'], prior=prior.cpu().tolist(),
                  refit={'class_ids': class_ids, 'sample_count': len(labels),
                         'hyperparameters_frozen': True,
                         'description': 'Pooled within-class diagonal prior fitted on all base training '
                                        'images after freezing hyperparameters. Former validation support/query '
                                        'images rejoin this fit; no benchmark test or future-class images are used.'},
                  limitation='Validation queries recur across stages; counts are occurrences, not '
                             'independent trials. Validation has fewer candidate classes than the benchmark. '
                             'Refitting the prior changes its values after hyperparameter selection.')
    audit = {key: report[key] for key in ('seed', 'fit_class_ids', 'validation_class_ids',
                                        'validation_sequences')}
    report['protocol_sha256'] = hashlib.sha256(json.dumps(audit, sort_keys=True).encode()).hexdigest()
    # Check serialization before committing state, so failures leave initialization retryable.
    json.dumps(report, allow_nan=False)
    model.uncertainty_state = {'prior': prior, **selected, 'covariance': settings.COVARIANCE,
                               'mean_uncertainty': settings.MEAN_UNCERTAINTY}
    return report
