"""Base-only calibration of dual-modal scale evidence above original-view GDA."""

import hashlib
import json
import math
from pathlib import Path

import torch

from engine.consensus_calibration import sample_sequence
from engine.uncertainty_calibration import (
    GROUPS, _empty_counts, _fit_prior, _sequence_records, _summarize, _update_counts, _validate_base_state,
)
from models.dual_frequency import align_semantics, frequency_evidence, pair_margins, support_statistics
from models.frequency_consensus import MODES, fit_margin_scales, pair_candidates, rerank_with_evidence
from models.frequency_discriminant import discriminant_logits, precision_from_covariance
from models.frequency_uncertainty import mix_uncertainty_probabilities
from utils.consensus_selection import score_candidate


def validate_dual_settings(cfg):
    settings = cfg.TRAINER.BiMC.DUAL_FREQUENCY
    u = cfg.TRAINER.BiMC.UNCERTAINTY
    if (not u.ENABLED or u.VIEW_CONTROL != 'original' or u.COVARIANCE != 'full_shared'
            or u.MEAN_UNCERTAINTY or cfg.TRAINER.BiMC.FREQUENCY.ENABLED
            or cfg.TRAINER.BiMC.FREQUENCY.ROUTER.ENABLED
            or cfg.TRAINER.BiMC.CONSENSUS.ENABLED or cfg.TRAINER.BiMC.RESIDUAL.ENABLED):
        raise ValueError('DUAL_FREQUENCY requires original-view full_shared GDA and frozen CLIP; '
                         'legacy fusion, router, consensus and residual must be disabled.')
    if settings.MODE not in MODES or settings.SEMANTIC_SOURCE not in ('explicit', 'ordinary', 'structured'):
        raise ValueError('Invalid dual-frequency mode or semantic source.')
    if sorted(settings.SEMANTIC_PERMUTATION) != [0, 1, 2]:
        raise ValueError('Dual semantic permutation must contain 0, 1, 2.')
    if settings.SEMANTIC_SOURCE == 'explicit' and not cfg.TRAINER.BiMC.FREQUENCY.EXPLICIT_DESCRIPTION_PATH:
        raise ValueError('Explicit dual semantics require a frequency description file.')
    for key in ('RESIDUALIZE', 'ALIGNMENT', 'AUTO_CALIBRATE', 'BASE_ONLY', 'SAVE_PREDICTIONS'):
        if not isinstance(getattr(settings, key), bool):
            raise ValueError('DUAL_FREQUENCY.' + key + ' must be boolean.')
    if (isinstance(settings.FIT_EPISODES, bool) or not isinstance(settings.FIT_EPISODES, int)
            or settings.FIT_EPISODES < 1 or u.OLD_WAY < 2):
        raise ValueError('Dual fitting needs positive FIT_EPISODES and OLD_WAY>=2.')
    if (isinstance(settings.ORDINARY_CANDIDATES, bool) or not isinstance(settings.ORDINARY_CANDIDATES, int)
            or settings.ORDINARY_CANDIDATES < 1):
        raise ValueError('ORDINARY_CANDIDATES must be a positive integer.')
    for key in ('SHOT_TAU', 'LAMBDA', 'MAX_GROUP_DROP_PP', 'SCALE_EPS'):
        value = getattr(settings, key)
        if isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError('DUAL_FREQUENCY.' + key + ' must be finite nonnegative.')
    if settings.SCALE_EPS <= 0:
        raise ValueError('Dual SCALE_EPS must be positive.')
    for key, length in (('BAND_WEIGHTS', 3), ('MANUAL_SCALES', 6)):
        values = getattr(settings, key)
        if len(values) != length or any(not math.isfinite(v) or v < 0 for v in values) or sum(values) <= 0:
            raise ValueError('Invalid DUAL_FREQUENCY.' + key)
    grid = settings.LAMBDA_GRID
    if not grid or 0 not in grid or any(not math.isfinite(v) or v < 0 for v in grid):
        raise ValueError('Dual LAMBDA_GRID must include zero and finite nonnegative values.')
    if settings.AUTO_CALIBRATE and not u.AUTO_CALIBRATE:
        raise ValueError('Automatic dual calibration requires automatic base anchor calibration.')


def _records(model, cfg, state, sequences, precision):
    """Rebuild all visual/semantic grounding with episode support only."""
    d, f = cfg.TRAINER.BiMC.DUAL_FREQUENCY, cfg.TRAINER.BiMC.FREQUENCY
    positions = {int(value): i for i, value in enumerate(state['class_index'])}
    selected_anchor = model.uncertainty_state
    for sequence in sequences:
        for record in _sequence_records(model, cfg, state, sequence):
            support, query = record['support_indices'], record['query_indices']
            local = torch.tensor([positions[value] for value in record['class_ids']], device=support.device)
            stats = support_statistics(
                state['images_features'][support], state['frequency_features'][support],
                state['images_targets'][support], record['class_ids'], d.RESIDUALIZE,
            )
            aligned = align_semantics(
                stats, state['dual_description_candidates'][local], state['description_proto'][local],
                top_k=f.DESCRIPTION_TOPK, temperature=f.DESCRIPTION_TEMPERATURE,
                shot_tau=d.SHOT_TAU, residualize=d.RESIDUALIZE, enabled=d.ALIGNMENT,
                permutation=d.SEMANTIC_PERMUTATION,
            )
            anchor = record['scores']
            if selected_anchor['alpha'] > 0:
                logits = discriminant_logits(record['features'], record['stats'], precision,
                                             validate_precision=False)
                anchor = mix_uncertainty_probabilities(
                    anchor, logits, alpha=selected_anchor['alpha'],
                    temperature=selected_anchor['temperature'],
                )
            margins = pair_margins(anchor, state['images_features'][query],
                                   state['frequency_features'][query], aligned, d.RESIDUALIZE, d.MODE)
            yield {**record, 'anchor': anchor, 'margins': margins, 'alignment': aligned}


def _evaluate(records, cfg, scales, grid):
    settings = cfg.TRAINER.BiMC.UNCERTAINTY
    d = cfg.TRAINER.BiMC.DUAL_FREQUENCY
    stage_rows = [[{'stage': stage, 'groups': {name: _empty_counts() for name in GROUPS}}
                   for stage in range(settings.STAGES + 1)] for _ in grid]
    for record in records:
        anchor, targets, stage = record['anchor'], record['targets'], record['stage']
        evidence = frequency_evidence(*record['margins'], scales, d.MODE, d.BAND_WEIGHTS)
        first = anchor.argmax(1)
        old = targets < settings.OLD_WAY
        current = (~old & (targets >= settings.OLD_WAY + (stage - 1) * settings.NEW_WAY)
                   if stage else torch.zeros_like(old))
        masks = {'all': torch.ones_like(old), 'old': old, 'new': ~old,
                 'historical_incremental': ~old & ~current, 'current_new': current}
        for i, strength in enumerate(grid):
            output, _ = rerank_with_evidence(anchor, evidence, strength)
            after = output.argmax(1)
            for name, mask in masks.items():
                _update_counts(stage_rows[i][stage]['groups'][name], mask, first == targets,
                               after == targets, first != after)
    candidates = []
    for strength, rows in zip(grid, stage_rows):
        for row in rows:
            row['groups'] = {name: _summarize(counts) for name, counts in row['groups'].items()}
        selection = score_candidate(rows, 'balanced_incremental', d.MAX_GROUP_DROP_PP)
        # Also guard the current and historical incremental groups when present.
        for row in rows[1:]:
            for name in ('historical_incremental', 'current_new'):
                group = row['groups'][name]
                if group['count'] and group['delta_accuracy_pp'] < -d.MAX_GROUP_DROP_PP - 1e-12:
                    selection['constraint_violations'].append(
                        {'stage': row['stage'], 'group': name, 'delta_accuracy_pp': group['delta_accuracy_pp']})
        selection['feasible'] = not selection['constraint_violations']
        candidates.append({'lambda': strength, 'stages': rows, 'selection': selection})
    feasible = [row for row in candidates if row['selection']['feasible']]
    best = max(row['selection']['score'] for row in feasible)
    winner = min((row for row in feasible if row['selection']['score'] >= best - 1e-12),
                 key=lambda row: row['lambda'])
    return candidates, winner


@torch.no_grad()
def initialize_dual_frequency(model, cfg, base_state, anchor_report):
    validate_dual_settings(cfg)
    if model.dual_frequency_state is not None or model.uncertainty_state is None:
        raise RuntimeError('Initialize the GDA anchor first; initialize dual frequency only once.')
    d, u = cfg.TRAINER.BiMC.DUAL_FREQUENCY, cfg.TRAINER.BiMC.UNCERTAINTY
    _validate_base_state(cfg, base_state)
    if (base_state.get('data_source', 'base_training_only') not in ('base_training_only', 'base_train', 'train')
            or base_state.get('task_id', 0) != 0
            or sorted(int(v) for v in base_state['class_index']) != anchor_report['base_class_ids']
            or any(int(v) >= cfg.DATASET.NUM_INIT_CLS for v in base_state['class_index'])):
        raise ValueError('Dual calibration accepts matching base training classes only.')
    report = {'schema_version': 1, 'data_source': 'base_training_only',
              'method': 'dual_modal_scale_residual_consensus', 'auto_calibrate': d.AUTO_CALIBRATE,
              'anchor_selected': anchor_report['selected'], 'anchor_protocol_sha256': anchor_report['protocol_sha256'],
              'semantic_source': d.SEMANTIC_SOURCE, 'semantic_permutation': list(d.SEMANTIC_PERMUTATION),
              'residualize': d.RESIDUALIZE, 'alignment': d.ALIGNMENT, 'mode': d.MODE,
              'band_weights': list(d.BAND_WEIGHTS), 'max_group_drop_pp': d.MAX_GROUP_DROP_PP,
              'selection_objective': 'balanced_incremental',
              'guard_groups': ['old', 'new', 'historical_incremental', 'current_new'],
              'seed': anchor_report['seed'], 'run_seed': int(cfg.SEED),
              'fit_seed': anchor_report['seed'] + 271,
              'fit_class_ids': anchor_report['fit_class_ids'],
              'validation_class_ids': anchor_report['validation_class_ids'],
              'validation_sequences': anchor_report['validation_sequences'],
              'fit_sequences': [], 'candidate_results': []}
    if d.SEMANTIC_SOURCE == 'explicit':
        path = Path(cfg.TRAINER.BiMC.FREQUENCY.EXPLICIT_DESCRIPTION_PATH)
        report['description_file'] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    if d.AUTO_CALIBRATE:
        fit_ids = anchor_report['fit_class_ids']
        if set(fit_ids) & set(anchor_report['validation_class_ids']):
            raise ValueError('Dual fit/validation classes must be disjoint.')
        generator = torch.Generator().manual_seed(anchor_report['seed'] + 271)
        fit_sequences = [sample_sequence(base_state['images_targets'], fit_ids, u, generator)
                         for _ in range(d.FIT_EPISODES)]
        # Reconstruct the selection metric on fit classes. The full-base refit
        # stored on the model must never leak validation queries into this prior.
        prior = _fit_prior(base_state['uncertainty_features'], base_state['images_targets'],
                           fit_ids, u.VAR_FLOOR, u.COVARIANCE)
        precision = precision_from_covariance(prior, ridge=model.uncertainty_state['ridge'],
                                              var_floor=u.VAR_FLOOR)
        visual, semantic, alignments = [], [], []
        for row in _records(model, cfg, base_state, fit_sequences, precision):
            visual.append(row['margins'][0])
            semantic.append(row['margins'][1])
            alignments.append(row['alignment'])
        scales = fit_margin_scales(torch.cat(visual), torch.cat(semantic), eps=d.SCALE_EPS)
        candidates, winner = _evaluate(
            _records(model, cfg, base_state, anchor_report['validation_sequences'], precision),
            cfg, scales, sorted(set(float(v) for v in d.LAMBDA_GRID)),
        )
        strength = winner['lambda']
        diagnostics = {key: torch.cat([a[key] for a in alignments]).mean(0).cpu().tolist()
                       for key in ('agreement', 'class_margin', 'band_margin', 'concentration', 'reliability')}
        report.update(selection_status='base_class_disjoint_validation', fit_sequences=fit_sequences,
                      candidate_results=candidates, selected_objective=winner['selection']['score'],
                      fit_alignment_means=diagnostics,
                      validation_query_occurrences=sum(row['groups']['all']['count'] for row in winner['stages']))
    else:
        scales = torch.tensor(d.MANUAL_SCALES, device=base_state['images_features'].device).reshape(2, 3)
        strength = float(d.LAMBDA)
        report.update(selection_status='manual_unvalidated', selected_objective=None)
    report.update(selected_lambda=strength, scales=scales.cpu().tolist(),
                  hyperparameters_frozen=True,
                  limitation='Anchor and residual strengths reuse the same base validation episodes; '
                             'these are tuning results, not independent accuracy estimates. '
                             'Scale-fit classes also fit the anchor covariance. The anchor covariance '
                             'is refitted on all base training data for deployment. Group guards are '
                             'relative to the selected anchor and do not guarantee benchmark gains.')
    audit = {key: report[key] for key in ('seed', 'fit_class_ids', 'validation_class_ids', 'validation_sequences')}
    report['protocol_sha256'] = hashlib.sha256(json.dumps(audit, sort_keys=True).encode()).hexdigest()
    json.dumps(report, allow_nan=False)
    model.dual_frequency_state = {'lambda': strength, 'scales': scales.detach()}
    return report
