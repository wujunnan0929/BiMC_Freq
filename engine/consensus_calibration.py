"""Fit six scales and one strength using disjoint base-class pseudo sequences.

All episode image statistics use support only. Benchmark test data and future
incremental classes are not accepted by this interface.
"""

import hashlib
import json
import math

import torch
from torch.nn import functional as F

from engine.residual_training import _group_covariance, make_reference_scorer
from models.frequency_consensus import (
    MODES, bounded_evidence, fit_margin_scales, pairwise_margins, rerank_with_evidence,
)


def build_support_reference(model, cfg, base_state, support_indices, selected_ids,
                            old_way, new_way):
    """Build the same class-weighted per-session covariance as the real runner."""
    features = base_state['images_features'].detach().float()
    labels = base_state['images_targets'].long()
    support_indices = torch.as_tensor(support_indices, device=labels.device)
    support = features[support_indices]
    support_labels = labels[support_indices]
    selected_ids = [int(value) for value in selected_ids]
    positions = {int(value): i for i, value in enumerate(base_state['class_index'])}
    local_positions = torch.tensor([positions[value] for value in selected_ids], device=labels.device)
    means = []
    frequency_means = []
    for class_id in selected_ids:
        mask = support_labels == class_id
        if not mask.any():
            raise ValueError('Every episode class needs its own support.')
        means.append(support[mask].mean(0))
        frequency_means.append(base_state['frequency_features'][support_indices[mask]].float().mean(0))
    prototypes = F.normalize(torch.stack(means), dim=-1)
    if len(selected_ids) > old_way and cfg.TRAINER.BiMC.VISION_CALIBRATION:
        prototypes[old_way:] = model.soft_calibration(prototypes[:old_way], prototypes[old_way:])
    covariance = features.new_zeros(features.shape[1], features.shape[1])
    groups = [selected_ids[:old_way]]
    groups.extend(selected_ids[start:start+new_way]
                  for start in range(old_way, len(selected_ids), new_way))
    for index, group in enumerate(groups):
        mask = torch.isin(support_labels, labels.new_tensor(group))
        gamma = cfg.TRAINER.BiMC.GAMMA_BASE if index == 0 else cfg.TRAINER.BiMC.GAMMA_INC
        covariance += len(group) * _group_covariance(support[mask], gamma)
    covariance /= len(selected_ids)
    descriptions, description_labels = [], []
    for local, class_id in enumerate(selected_ids):
        values = base_state['description_features'][base_state['description_targets'] == class_id].float()
        if not len(values):
            raise ValueError('An episode class has no descriptions.')
        descriptions.append(values)
        description_labels.append(labels.new_full((len(values),), local))
    reference_state = {
        'image_proto': prototypes, 'cov_image': covariance,
        'description_proto': base_state['description_proto'][local_positions].float(),
        'text_features': base_state['text_features'][local_positions].float(),
        'description_features': torch.cat(descriptions),
        'description_targets': torch.cat(description_labels),
    }
    return (make_reference_scorer(model, cfg, reference_state, old_way),
            F.normalize(torch.stack(frequency_means), dim=-1),
            base_state['frequency_consensus_text_proto'][local_positions].float(), reference_state)


def sample_sequence(labels, class_ids, settings, generator):
    total = settings.OLD_WAY + settings.NEW_WAY * settings.STAGES
    if len(class_ids) < total:
        raise ValueError(f'Each calibration split needs {total} classes, got {len(class_ids)}; '
                         'reduce OLD_WAY/NEW_WAY/STAGES using base data only.')
    order = torch.randperm(len(class_ids), generator=generator)[:total].tolist()
    selected = [class_ids[index] for index in order]
    support, query = [], []
    for local, class_id in enumerate(selected):
        available = torch.where(labels.cpu() == class_id)[0]
        shot = settings.OLD_SHOT if local < settings.OLD_WAY else settings.SHOT
        required = shot + settings.QUERY
        if len(available) < required:
            raise ValueError(f'Base class {class_id} needs {required} distinct support/query '
                             f'images, has {len(available)}. No replacement is allowed.')
        indices = available[torch.randperm(len(available), generator=generator)[:required]]
        support.append(indices[:shot].tolist())
        query.append(indices[shot:].tolist())
    return {'class_ids': selected, 'support_indices': support, 'query_indices': query}


def sequence_records(model, cfg, state, sequence):
    settings = cfg.TRAINER.BiMC.CONSENSUS
    features = state['images_features']
    labels = state['images_targets']
    for stage in range(settings.STAGES + 1):
        seen = settings.OLD_WAY + stage * settings.NEW_WAY
        selected = sequence['class_ids'][:seen]
        support = [i for group in sequence['support_indices'][:seen] for i in group]
        query = [i for group in sequence['query_indices'][:seen] for i in group]
        if set(support).intersection(query):
            raise ValueError('Calibration support/query overlap is forbidden.')
        scorer, visual, semantic, _ = build_support_reference(
            model, cfg, state, support, selected, settings.OLD_WAY, settings.NEW_WAY,
        )
        query_indices = torch.tensor(query, device=features.device)
        scores = scorer(features[query_indices].float())
        visual_margin, semantic_margin = pairwise_margins(
            scores, state['frequency_features'][query_indices], visual, semantic,
            settings.SEMANTIC_PERMUTATION,
        )
        mapping = {class_id: local for local, class_id in enumerate(selected)}
        targets = labels.new_tensor([mapping[int(label)] for label in labels[query_indices]])
        yield {'scores': scores, 'visual_margin': visual_margin,
               'semantic_margin': semantic_margin, 'targets': targets,
               'stage': stage, 'old_way': settings.OLD_WAY}


@torch.no_grad()
def initialize_consensus(model, cfg, base_state):
    settings = cfg.TRAINER.BiMC.CONSENSUS
    if model.consensus_state is not None:
        raise RuntimeError('Consensus statistics must be initialized once, at base only.')
    if model.frequency_fusion_enabled or cfg.TRAINER.BiMC.RESIDUAL.ENABLED:
        raise ValueError('Calibration requires a plain BiMC candidate reference.')
    if settings.MODE not in MODES or sorted(settings.SEMANTIC_PERMUTATION) != [0, 1, 2]:
        raise ValueError('Invalid consensus mode or semantic permutation.')
    for key in ('FIT_EPISODES', 'VAL_EPISODES', 'OLD_WAY', 'NEW_WAY', 'OLD_SHOT',
                'SHOT', 'QUERY', 'STAGES'):
        if getattr(settings, key) < 1:
            raise ValueError('CONSENSUS.' + key + ' must be positive.')
    if settings.OLD_WAY < 2:
        raise ValueError('CONSENSUS.OLD_WAY must be >= 2 for base top-two candidates.')
    if not 0 < settings.FIT_FRACTION < 1:
        raise ValueError('FIT_FRACTION must lie strictly between 0 and 1.')
    grid = sorted(set(float(value) for value in settings.LAMBDA_GRID))
    if not grid or grid[0] != 0 or any(not math.isfinite(value) or value < 0 for value in grid):
        raise ValueError('LAMBDA_GRID must include zero and contain finite nonnegative values.')
    if not math.isfinite(settings.LAMBDA) or settings.LAMBDA < 0:
        raise ValueError('LAMBDA must be finite and nonnegative.')
    features, labels = base_state['images_features'], base_state['images_targets']
    class_ids = sorted(int(value) for value in torch.unique(labels))
    if (class_ids != sorted(int(value) for value in base_state['class_index'])
            or any(value < 0 or value >= cfg.DATASET.NUM_INIT_CLS for value in class_ids)):
        raise ValueError('Calibration accepts only base-class training samples.')
    seed = int(settings.CALIBRATION_SEED)
    seed = seed if seed >= 0 else int(cfg.SEED) + 9101
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(class_ids), generator=generator).tolist()
    count = int(len(class_ids) * settings.FIT_FRACTION)
    fit_ids, val_ids = ([class_ids[index] for index in order[:count]],
                       [class_ids[index] for index in order[count:]])
    fit_sequences = [sample_sequence(labels, fit_ids, settings, generator)
                     for _ in range(settings.FIT_EPISODES)]
    val_sequences = [sample_sequence(labels, val_ids, settings, generator)
                     for _ in range(settings.VAL_EPISODES)]
    visual_margins, semantic_margins = [], []
    for sequence in fit_sequences:
        for record in sequence_records(model, cfg, base_state, sequence):
            visual_margins.append(record['visual_margin'])
            semantic_margins.append(record['semantic_margin'])
    scales = fit_margin_scales(torch.cat(visual_margins), torch.cat(semantic_margins),
                              eps=settings.SCALE_EPS).detach()
    fit_query_occurrences = sum(len(values) for values in visual_margins)
    del visual_margins, semantic_margins
    evaluated_grid = sorted(set(grid + ([float(settings.LAMBDA)] if not settings.AUTO_CALIBRATE else [])))
    totals = {value: {'correct': 0, 'corrected': 0, 'damaged': 0,
                      'old_corrected': 0, 'old_damaged': 0,
                      'new_corrected': 0, 'new_damaged': 0} for value in evaluated_grid}
    count = top2_hits = reference_correct = 0
    from models.frequency_consensus import pair_candidates
    for sequence in val_sequences:
        for record in sequence_records(model, cfg, base_state, sequence):
            scores, targets = record['scores'], record['targets']
            first, second, _ = pair_candidates(scores)
            before = first == targets
            old = targets < settings.OLD_WAY
            count += len(targets)
            reference_correct += int(before.sum())
            top2_hits += int((before | (second == targets)).sum())
            evidence = bounded_evidence(record['visual_margin'], record['semantic_margin'],
                                        scales, settings.MODE)
            for strength in evaluated_grid:
                output, _ = rerank_with_evidence(scores, evidence, strength)
                after = output.argmax(1) == targets
                corrected, damaged = ~before & after, before & ~after
                values = totals[strength]
                values['correct'] += int(after.sum())
                values['corrected'] += int(corrected.sum())
                values['damaged'] += int(damaged.sum())
                for name, mask in (('old', old), ('new', ~old)):
                    values[name + '_corrected'] += int((corrected & mask).sum())
                    values[name + '_damaged'] += int((damaged & mask).sum())
    strength = (min(grid, key=lambda value: (-totals[value]['correct'], value))
                if settings.AUTO_CALIBRATE else float(settings.LAMBDA))
    model.consensus_state = {'scales': scales, 'lambda': strength,
                             'mode': settings.MODE,
                             'semantic_permutation': list(settings.SEMANTIC_PERMUTATION)}
    audit = {'fit_sequences': fit_sequences, 'validation_sequences': val_sequences}
    return {
        'schema_version': 1, 'data_source': 'base_training_only', 'seed': seed,
        'fit_class_ids': fit_ids, 'validation_class_ids': val_ids,
        'fit_query_occurrences': fit_query_occurrences, 'validation_query_occurrences': count,
        'selected_lambda': strength, 'auto_calibrate': settings.AUTO_CALIBRATE,
        'scales': scales.cpu().tolist(), 'active_sources': (scales > 0).cpu().tolist(),
        'mode': settings.MODE, 'view_control': settings.VIEW_CONTROL,
        'semantic_permutation': list(settings.SEMANTIC_PERMUTATION),
        'reference_accuracy': 100 * reference_correct / count,
        'top2_recall': 100 * top2_hits / count,
        'candidate_results': [{'lambda': value, 'accuracy': 100 * values['correct'] / count,
                               **values} for value, values in totals.items()],
        'candidate_class_counts': [settings.OLD_WAY + step * settings.NEW_WAY
                                   for step in range(settings.STAGES + 1)],
        'limitation': 'Pseudo sequences have fewer candidates than the full benchmark; '
                      'queries recur across stages, so occurrences are not independent trials.',
        'protocol_sha256': hashlib.sha256(json.dumps(audit, sort_keys=True).encode()).hexdigest(),
        **audit,
    }
