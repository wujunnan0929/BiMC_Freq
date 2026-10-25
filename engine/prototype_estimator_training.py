"""Base-only support estimator training, class-disjoint selection and fresh refit.

No CLIP model or benchmark query loader is constructed. The teacher, support
and query sets in fitting episodes are disjoint. During model selection,
validation teachers are diagnostic only. After selection, all original base
rows may be reused in the fresh fixed-duration refit.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F


@dataclass
class EstimatorOptions:
    method: str = 'boundary'
    auxiliary: str = 'none'
    auxiliary_probability: float = .25
    seed: int = 2
    sequences: int = 20
    teacher_shot: int = 15
    query: int = 5
    old_shot: int = 20
    old_ways: tuple = (10, 20, 30)
    novel_ways: tuple = (5, 10)
    novel_shots: tuple = (2, 5)
    episode_pool: int = 100
    steps: int = 300
    validate_every: int = 50
    early_checkpoints: tuple = (1, 5, 10, 20)
    hidden: int = 32
    max_delta: float = .15
    lr: float = .001
    weight_decay: float = .0001
    center_weight: float = 1.
    margin_weight: float = 1.
    anchor_weight: float = .5
    blend_grid: tuple = (0., .25, .5, 1.)
    maximum_group_drop_pp: float = .2
    calibrate: bool = True


def validate_options(o):
    if o.method not in ('center', 'boundary') or o.auxiliary not in ('none', 'safd', 'fft', 'augmentation'):
        raise ValueError('Unknown method or auxiliary source.')
    for name in ('seed', 'sequences', 'teacher_shot', 'query', 'old_shot', 'episode_pool', 'validate_every', 'hidden'):
        value = getattr(o, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(name + ' must be a positive integer.')
    if isinstance(o.steps, bool) or not isinstance(o.steps, int) or o.steps < 0:
        raise ValueError('steps must be nonnegative.')
    for name in ('old_ways', 'novel_ways', 'novel_shots'):
        if not getattr(o, name) or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in getattr(o, name)):
            raise ValueError(name + ' must contain positive integers.')
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in o.early_checkpoints):
        raise ValueError('early_checkpoints must contain positive integers.')
    for name in ('max_delta', 'lr'):
        if not np.isfinite(getattr(o, name)) or getattr(o, name) <= 0:
            raise ValueError(name + ' must be finite and positive.')
    if o.max_delta >= 1:
        raise ValueError('max_delta must be smaller than one.')
    for name in ('weight_decay', 'center_weight', 'margin_weight', 'anchor_weight', 'maximum_group_drop_pp'):
        if not np.isfinite(getattr(o, name)) or getattr(o, name) < 0:
            raise ValueError(name + ' must be finite and nonnegative.')
    if not o.center_weight or (o.method == 'boundary' and not o.margin_weight):
        raise ValueError('Use positive center supervision, and positive margin supervision for boundary.')
    if not np.isfinite(o.auxiliary_probability) or not 0 <= o.auxiliary_probability <= 1:
        raise ValueError('auxiliary_probability must be in [0,1].')
    if not o.blend_grid or 0 not in o.blend_grid or any(not np.isfinite(v) or not 0 <= v <= 1 for v in o.blend_grid):
        raise ValueError('blend_grid must include zero and contain values in [0,1].')


def check_auxiliary(arrays, metadata, aux_arrays, aux_metadata, source):
    """The ordinary support bank and its perturbation rows must be identical."""
    if source == 'none':
        return
    if aux_arrays is None or aux_metadata is None or aux_metadata.get('source') != source:
        raise ValueError('Auxiliary source must match the supplied feature cache.')
    for key in ('dataset', 'clip_weight_sha256', 'base_class_ids', 'ordinary_description_sha256',
                'text_template', 'support_transform', 'input_provenance'):
        if metadata.get(key) != aux_metadata.get(key):
            raise ValueError('Auxiliary cache provenance differs: ' + key)
    for key in ('labels', 'sample_id'):
        if not np.array_equal(arrays[key], aux_arrays[key]):
            raise ValueError('Auxiliary cache row order differs: ' + key)
    if not np.allclose(arrays['features'][:, 0], aux_arrays['features'][:, 0], atol=1e-6, rtol=1e-5):
        raise ValueError('Auxiliary cache original features differ.')


def _semantics(arrays, classes, cfg, device):
    ids = torch.tensor(classes, device=device)
    text = torch.as_tensor(arrays['text_features'], device=device).float()[ids]
    desc = torch.as_tensor(arrays['description_proto'], device=device).float()[ids]
    lam = cfg.TRAINER.BiMC.LAMBDA_T if cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.
    return F.normalize((1 - lam) * text + lam * desc, dim=-1)


@torch.no_grad()
def training_episodes(arrays, cfg, pool, options, device='cpu', auxiliary_features=None, seed_offset=0):
    """Create a deterministic finite pool; all fitting rows belong to A only."""
    from engine.safd_metric_training import _reference_proxy
    rng = random.Random(options.seed + seed_offset)
    augmentation_rng = random.Random(options.seed + seed_offset + 30303)
    x = F.normalize(torch.as_tensor(arrays['features'][:, 0], device=device).float(), dim=-1)
    aux = torch.as_tensor(auxiliary_features, device=device).float() if auxiliary_features is not None else None
    proxy = _reference_proxy(cfg)
    classes = sorted(pool)
    records, trace = [], []
    way_pairs = [(old, novel) for old in options.old_ways for novel in options.novel_ways
                 if old + novel <= len(classes)]
    if not way_pairs:
        old = min(min(options.old_ways), len(classes) - 1)
        way_pairs = [(old, min(min(options.novel_ways), len(classes) - old))]
    for episode in range(options.episode_pool):
        old, novel = rng.choice(way_pairs)
        if min(old, novel) < 1:
            raise ValueError('Training needs at least two fitting classes.')
        selected = rng.sample(classes, old + novel)
        support_rows, teacher_rows, query_rows, supports, teachers, edits = [], [], [], [], [], []
        for j, c in enumerate(selected):
            available = list(pool[c])
            if j < old:
                shot = min(options.old_shot, len(available) - options.query)
                teacher = 0
            else:
                choices = [s for s in options.novel_shots if s + options.query + 2 <= len(available)]
                if not choices:
                    raise ValueError(f'Class {c} lacks disjoint support/teacher/query fitting rows.')
                shot = rng.choice(choices)
                teacher = min(options.teacher_shot, len(available) - shot - options.query)
            if shot < 1 or shot + teacher + options.query > len(available):
                raise ValueError(f'Class {c} lacks fitting rows; lower old-shot/query.')
            chosen = rng.sample(available, shot + teacher + options.query)
            s, t, q = chosen[:shot], chosen[shot:shot + teacher], chosen[shot + teacher:]
            value = x[s].clone()
            if j >= old and options.auxiliary != 'none' and augmentation_rng.random() < options.auxiliary_probability:
                if aux is None:
                    raise ValueError('Auxiliary training requires cached intervention features.')
                position, view = augmentation_rng.randrange(shot), augmentation_rng.choice((1, 2))
                value[position] = F.normalize(aux[s[position], view], dim=-1)
                edits.append({'class_id': c, 'support_position': position, 'row': s[position], 'view': view})
            support_rows.append(s); teacher_rows.append(t); query_rows.append(q)
            supports.append(value)
            teachers.append(F.normalize(x[t].mean(0), dim=-1) if t else None)
        raw = F.normalize(torch.stack([s.mean(0) for s in supports]), dim=-1)
        reference = raw.clone()
        if cfg.TRAINER.BiMC.VISION_CALIBRATION:
            reference[old:] = proxy.soft_calibration(raw[:old], raw[old:])
        teacher_prototypes = torch.cat((reference[:old], torch.stack(teachers[old:])))
        query = torch.cat([x[q] for q in query_rows])
        target = torch.cat([torch.full((len(q),), j, dtype=torch.long, device=device) for j, q in enumerate(query_rows)])
        records.append({'supports': supports, 'reference_prototypes': reference,
                        'semantic_prototypes': _semantics(arrays, selected, cfg, device),
                        'teacher_prototypes': teacher_prototypes, 'query': query, 'targets': target,
                        'num_old': old, 'class_ids': selected})
        trace.append({'episode': episode, 'class_ids': selected, 'num_old': old,
                      'support_indices': support_rows, 'teacher_indices': teacher_rows,
                      'query_indices': query_rows, 'auxiliary_edits': edits})
    return records, trace


def episode_prototypes(model, row):
    from models.support_estimator import correct_current
    old = row['num_old']
    reference = row['reference_prototypes']
    current = correct_current(model, row['supports'][old:], reference[old:],
                              row['semantic_prototypes'][old:], reference[:old], blend=1.)
    return torch.cat((reference[:old], current))


def _balanced(values, old):
    groups = [values[mask].mean() for mask in (old, ~old) if bool(mask.any())]
    return torch.stack(groups).mean()


def episode_loss(model, row, options):
    prototypes = episode_prototypes(model, row)
    teacher = row['teacher_prototypes'].detach()
    old_count = row['num_old']
    center = (1 - (prototypes[old_count:] * teacher[old_count:]).sum(-1)).mean()
    reference = row['reference_prototypes'].detach()
    query, y = row['query'].detach(), row['targets']
    student_scores = query @ prototypes.T
    with torch.no_grad():
        teacher_scores = query @ teacher.T
        reference_scores = query @ reference.T
        ids = torch.arange(len(y), device=y.device)
        competitors = teacher_scores.clone()
        competitors[ids, y] = -torch.inf
        negative = competitors.argmax(1)
        target_margin = teacher_scores[ids, y] - teacher_scores[ids, negative]
    margin = student_scores[ids, y] - student_scores[ids, negative]
    boundary = _balanced(F.smooth_l1_loss(margin, target_margin, reduction='none', beta=.05), y < old_count)
    # Protect every competing column on fitting old-class queries.
    other = student_scores.clone()
    other[ids, y] = -torch.inf
    student_true_margin = student_scores[ids, y] - other.max(1).values
    with torch.no_grad():
        ref_other = reference_scores.clone()
        ref_other[ids, y] = -torch.inf
        reference_margin = reference_scores[ids, y] - ref_other.max(1).values
        protected = (y < old_count) & (reference_margin > 0)
    anchor = (F.relu(reference_margin[protected] - student_true_margin[protected]).mean()
              if bool(protected.any()) else center.new_zeros(()))
    loss = options.center_weight * center
    if options.method == 'boundary':
        loss = loss + options.margin_weight * boundary + options.anchor_weight * anchor
    metrics = {'loss': float(loss.detach()), 'center_loss': float(center.detach()),
               'margin_loss': float(boundary.detach()), 'anchor_loss': float(anchor.detach()),
               'base_accuracy': 100 * float((student_scores.argmax(1)[y < old_count] == y[y < old_count]).float().mean()),
               'novel_accuracy': 100 * float((student_scores.argmax(1)[y >= old_count] == y[y >= old_count]).float().mean())}
    return loss, metrics


def optimize(model, records, options, steps, callback=None):
    if not steps:
        return []
    optimizer = torch.optim.Adam(model.parameters(), lr=options.lr, weight_decay=options.weight_decay)
    history = []
    for step in range(1, steps + 1):
        loss, metrics = episode_loss(model, records[(step - 1) % len(records)], options)
        if not torch.isfinite(loss):
            raise ValueError('Non-finite support-estimator loss.')
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        history.append(metrics)
        if callback and (step in options.early_checkpoints or step % options.validate_every == 0 or step == steps):
            callback(step, model, metrics)
    return history


@torch.no_grad()
def estimate_record(model, row, blend):
    """Replay prototype arrival in order, never revisiting an old correction."""
    from models.support_estimator import correct_current
    if blend == 0:
        return None
    reference = row['reference_prototypes']
    blocks = [reference[:row['num_old']]]
    start = row['num_old']
    for size in row['arrival_sizes']:
        end = start + size
        blocks.append(correct_current(model, row['supports'][start:end], reference[start:end],
                      row['semantic_prototypes'][start:end], torch.cat(blocks), blend=blend))
        start = end
    if start != len(reference):
        raise ValueError('Validation arrival sizes do not cover all novel classes.')
    return torch.cat(blocks)


@torch.no_grad()
def evaluate_estimator(model, records, cfg, calibration, blend):
    from engine.prototype_evaluation import score_record, evaluate_records
    return evaluate_records(records, lambda row: score_record(
        row, cfg, image_prototypes=estimate_record(model, row, blend), calibration=calibration))


def _feasible(metrics, references, maximum_drop):
    return all(metrics[k] >= ref[k] - maximum_drop - 1e-12 for ref in references
               for k in ('base_accuracy', 'novel_accuracy'))


def _model(options, device):
    from models.support_estimator import SupportPrototypeEstimator
    torch.manual_seed(options.seed)
    return SupportPrototypeEstimator(hidden=options.hidden, max_delta=options.max_delta,
                                    use_competitors=options.method == 'boundary').to(device)


def fit_estimator(arrays, metadata, cfg, options=None, device='cpu', auxiliary_arrays=None,
                  auxiliary_metadata=None):
    from engine.prototype_evaluation import (prepare_context, validation_records, select_calibration,
                                             identity_calibration, score_record, evaluate_records)
    started = time.perf_counter()
    options = options or EstimatorOptions()
    validate_options(options)
    if not metadata.get('complete_base_bank') or not metadata.get('has_text_features'):
        raise ValueError('Use a complete base feature bank with ordinary text features.')
    if options.auxiliary != 'none' and auxiliary_arrays is None:
        auxiliary_arrays, auxiliary_metadata = arrays, metadata
    check_auxiliary(arrays, metadata, auxiliary_arrays, auxiliary_metadata, options.auxiliary)
    cfg = cfg.clone()
    cfg.defrost()
    cfg.TRAINER.BiMC.GUIDANCE.QUERY = options.query
    cfg.freeze()
    cfg, protocol, pool = prepare_context(arrays, cfg, seed=options.seed, sequences=options.sequences,
                                         teacher_shot=options.teacher_shot)
    val = validation_records(arrays, cfg, protocol, device=device)
    reference = evaluate_records(val, lambda row: score_record(row, cfg))
    if options.calibrate:
        calibration, calibration_report = select_calibration(val, cfg, maximum_drop=options.maximum_group_drop_pp)
    else:
        calibration, calibration_report = identity_calibration(), {'status': 'disabled'}
    anchor = evaluate_records(val, lambda row: score_record(row, cfg, calibration=calibration))
    model = _model(options, device)
    best = {'step': 0, 'blend': 0., **anchor}
    candidates = [dict(best, feasible=True)]
    train, trace = training_episodes(arrays, cfg, pool, options, device,
                    auxiliary_arrays['features'] if options.auxiliary != 'none' else None)

    def validate(step, fitted, metrics):
        nonlocal best
        for blend in sorted(set(float(v) for v in options.blend_grid if v > 0)):
            result = evaluate_estimator(fitted, val, cfg, calibration, blend)
            feasible = _feasible(result, (reference, anchor), options.maximum_group_drop_pp)
            row = {'step': step, 'blend': blend, **result, 'feasible': feasible, 'training_episode': metrics}
            candidates.append(row)
            if feasible and result['harmonic_accuracy'] > best['harmonic_accuracy'] + 1e-12:
                best = dict(row)
            print(f"Estimator step={step} blend={blend:g}: B={result['base_accuracy']:.4f} "
                  f"N={result['novel_accuracy']:.4f} H={result['harmonic_accuracy']:.4f} feasible={feasible}", flush=True)
    history = optimize(model, train, options, options.steps, validate)
    selection_finished = time.perf_counter()
    # Selection has ended. Start over with the fixed selected duration on all base images.
    final = _model(options, device)
    final_trace, final_history = [], []
    if best['step'] and best['blend']:
        labels = np.asarray(arrays['labels'])
        full_pool = {int(c): np.flatnonzero(labels == c).tolist() for c in np.unique(labels)}
        final_train, final_trace = training_episodes(arrays, cfg, full_pool, options, device,
            auxiliary_arrays['features'] if options.auxiliary != 'none' else None, seed_offset=1000003)
        final_history = optimize(final, final_train, options, best['step'])
    final.eval().requires_grad_(False)
    report = {'schema_version': 1,
              'training_protocol': 'base_only_class_disjoint_teacher_margin_selection_then_all_base_fixed_step_refit',
              'options': asdict(options), 'reference': reference, 'calibrated_reference': anchor,
              'calibration': calibration, 'calibration_report': calibration_report,
              'selected_validation': best, 'selected_step': best['step'], 'blend': best['blend'],
              'checkpoints': candidates, 'selection_protocol': protocol,
              'selection_training_episodes': trace, 'final_refit_training_episodes': final_trace,
              'selection_training_history': history, 'final_training_history': final_history,
              'trainable_parameters': sum(p.numel() for p in final.parameters()),
              'phase_order': ['reserve_A_queries_and_V_teachers', 'select_reference_calibration',
                              'fit_A_only_and_select', 'selection_ended', 'fresh_all_base_fixed_step_refit', 'freeze'],
              'selection_wall_seconds': selection_finished - started,
              'final_refit_wall_seconds': time.perf_counter() - selection_finished,
              'fit_wall_seconds': time.perf_counter() - started,
              'teacher_role': 'selection_phase_A_teachers_only; V_teachers_diagnostic_only_before_all_base_refit',
              'limitations': ['Base-supervised adaptation, not training-free.',
                              'A finite episode pool is reused across optimizer passes.',
                              'Validation sequences share images and classes.',
                              'Zero-step/zero-blend selection is allowed and must be reported.']}
    return final, report, cfg


def export_fitted_estimator(path, model, report, cfg, metadata, cache_path, sample_ids,
                            auxiliary_cache_path=None):
    from engine.safd_metric_runtime import reference_settings, sample_identity_hash
    from engine.prototype_estimator_runtime import export_estimator
    root = Path(__file__).resolve().parents[1]
    source_files = ('engine/prototype_estimator_training.py', 'engine/prototype_evaluation.py',
                    'engine/prototype_estimator_runtime.py', 'models/support_estimator.py', 'models/bimc.py')
    content = {k: metadata.get(k) for k in ('ordinary_description_sha256', 'text_template',
               'support_transform', 'input_provenance', 'base_class_names')}
    saved = {'dataset': metadata['dataset'], 'data_source': 'base_training_only',
             'clip_weight_sha256': metadata['clip_weight_sha256'], 'feature_dim': int(metadata['feature_dim']),
             'base_class_ids': metadata['base_class_ids'],
             'base_sample_ids_sorted_sha256': sample_identity_hash(sample_ids),
             'reference_settings': reference_settings(cfg), 'precision': cfg.TRAINER.BiMC.PREC,
             **content, 'model_config': {'hidden': report['options']['hidden'],
                 'max_delta': report['options']['max_delta'], 'use_competitors': report['options']['method'] == 'boundary'},
             'blend': report['blend'], 'calibration': report['calibration'],
             'selected_step': report['selected_step'], 'trainable_parameters': report['trainable_parameters'],
             'training_protocol': report['training_protocol'], 'method': report['options']['method'],
             'fitting_seed': report['options']['seed'],
             'selection_protocol_sha256': report['selection_protocol']['sha256'],
             'auxiliary': report['options']['auxiliary'], 'auxiliary_training': report['options']['auxiliary'],
             'complete_base_bank': metadata['complete_base_bank'],
             'cache_sha256': hashlib.sha256(Path(cache_path).read_bytes()).hexdigest(),
             'cache_path': str(Path(cache_path).resolve()),
             'auxiliary_cache_sha256': hashlib.sha256(Path(auxiliary_cache_path).read_bytes()).hexdigest() if auxiliary_cache_path else None,
             'fit_wall_seconds': report['fit_wall_seconds'], 'cache_extraction_wall_seconds': metadata.get('wall_seconds'),
             'source_files_sha256': {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in source_files}}
    return export_estimator(path, model, saved, report)
