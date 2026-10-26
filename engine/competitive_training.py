"""Base-only competitive fusion fitting; class-disjoint selection then fresh refit."""
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from engine.competitive_evaluation import (prepare_records, select_fixed_fusion, fusion_record_scores,
                                          group_diagnostics, feasible, identity_fixed)
from engine.prototype_evaluation import (prepare_context, validation_records, evaluate_records,
                                         select_calibration, score_record, validate_calibration)
from engine.prototype_estimator_training import check_auxiliary
from models.competitive_fusion import CompetitiveFusion, validate_fixed


@dataclass
class FusionOptions:
    method: str = 'joint'
    use_support_quality: bool = True
    auxiliary: str = 'none'
    auxiliary_probability: float = .25
    seed: int = 2
    sequences: int = 20
    teacher_shot: int = 15  # reserve the same diagnostic bank as previous screenings; never train on it during selection
    query: int = 5
    old_shot: int = 20
    old_ways: tuple = (10, 20, 30)
    novel_ways: tuple = (5, 10, 20)
    novel_shots: tuple = (2, 5)
    episode_pool: int = 100
    steps: int = 300
    validate_every: int = 50
    early_checkpoints: tuple = (1, 5, 10, 20)
    hidden: int = 16
    group_limit: float = 2.
    mix_limit: float = .5
    lr: float = .001
    weight_decay: float = .0001
    group_weight: float = .5
    protection_weight: float = .5
    distillation_weight: float = .1
    blend_grid: tuple = (0., .25, .5, 1.)
    weight_grid: tuple = (0., .25, .5, .75, .8, 1.)
    bias_grid: tuple = (0., -.2, -.1, .1, .2)
    maximum_group_drop_pp: float = .2


def validate_options(o):
    if o.method not in ('group', 'joint') or o.auxiliary not in ('none', 'safd', 'fft', 'augmentation'):
        raise ValueError('Unknown fusion method or auxiliary.')
    if not isinstance(o.use_support_quality, bool):
        raise ValueError('use_support_quality must be boolean.')
    for key in ('seed', 'sequences', 'teacher_shot', 'query', 'old_shot', 'episode_pool', 'validate_every', 'hidden'):
        if isinstance(getattr(o, key), bool) or not isinstance(getattr(o, key), int) or getattr(o, key) < 1:
            raise ValueError(key + ' must be a positive integer.')
    if isinstance(o.steps, bool) or not isinstance(o.steps, int) or o.steps < 0:
        raise ValueError('steps must be nonnegative.')
    for key in ('old_ways', 'novel_ways', 'novel_shots', 'early_checkpoints'):
        if not getattr(o, key) or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in getattr(o, key)):
            raise ValueError(key + ' must contain positive integers.')
    for key in ('lr', 'group_limit', 'mix_limit'):
        if not math.isfinite(getattr(o, key)) or getattr(o, key) <= 0:
            raise ValueError(key + ' must be positive and finite.')
    if o.group_limit > 10 or o.mix_limit > 1:
        raise ValueError('Competition update limits are too large.')
    for key in ('weight_decay', 'group_weight', 'protection_weight', 'distillation_weight', 'maximum_group_drop_pp'):
        if not math.isfinite(getattr(o, key)) or getattr(o, key) < 0:
            raise ValueError(key + ' must be finite and nonnegative.')
    if not math.isfinite(o.auxiliary_probability) or not 0 <= o.auxiliary_probability <= 1:
        raise ValueError('auxiliary_probability must be in [0,1].')
    for key in ('blend_grid', 'weight_grid'):
        if not getattr(o, key) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in getattr(o, key)):
            raise ValueError(key + ' must contain weights in [0,1].')
    if 0 not in o.blend_grid or not o.bias_grid or any(not math.isfinite(v) or abs(v) > 20 for v in o.bias_grid):
        raise ValueError('Include blend=0 and finite supported bias candidates.')


@torch.no_grad()
def training_episodes(arrays, cfg, pool, options, calibration, device='cpu', auxiliary_features=None, seed_offset=0):
    from engine.safd_metric_training import _reference_proxy, _text_subset
    from engine.residual_training import _group_covariance
    rng = random.Random(options.seed + seed_offset)
    aux_rng = random.Random(options.seed + seed_offset + 30303)
    x = F.normalize(torch.as_tensor(arrays['features'][:, 0], device=device).float(), dim=-1)
    aux = torch.as_tensor(auxiliary_features, device=device).float() if auxiliary_features is not None else None
    classes, records, trace = sorted(pool), [], []
    pairs = [(a, b) for a in options.old_ways for b in options.novel_ways if a+b <= len(classes)]
    if not pairs:
        a = min(min(options.old_ways), len(classes)-1)
        pairs = [(a, min(min(options.novel_ways), len(classes)-a))]
    if min(pairs[0]) < 1:
        raise ValueError('Fitting requires at least two fitting classes.')
    proxy, b = _reference_proxy(cfg), cfg.TRAINER.BiMC
    for episode in range(options.episode_pool):
        old, novel = rng.choice(pairs)
        selected = rng.sample(classes, old+novel)
        supports, support_rows, query_rows, edits = [], [], [], []
        for j, c in enumerate(selected):
            available = list(pool[c])
            choices = [s for s in options.novel_shots if s+options.query <= len(available)]
            shot = min(options.old_shot, len(available)-options.query) if j < old else rng.choice(choices) if choices else 0
            if shot < 1:
                raise ValueError('Insufficient disjoint support/query fitting rows.')
            indices = rng.sample(available, shot+options.query)
            s, q = indices[:shot], indices[shot:]
            values = x[s].clone()
            if j >= old and options.auxiliary != 'none' and aux_rng.random() < options.auxiliary_probability:
                if aux is None:
                    raise ValueError('Auxiliary features are required.')
                position, view = aux_rng.randrange(shot), aux_rng.choice((1, 2))
                values[position] = F.normalize(aux[s[position], view], dim=-1)
                edits.append({'class_id': c, 'row': s[position], 'support_position': position, 'view': view})
            supports.append(values); support_rows.append(s); query_rows.append(q)
        raw = F.normalize(torch.stack([v.mean(0) for v in supports]), dim=-1)
        reference = raw.clone()
        if b.VISION_CALIBRATION:
            reference[old:] = proxy.soft_calibration(raw[:old], raw[old:])
        covariance_sum = old * _group_covariance(torch.cat(supports[:old]), b.GAMMA_BASE)
        for start in range(old, old+novel, cfg.DATASET.NUM_INC_CLS):
            end = min(start+cfg.DATASET.NUM_INC_CLS, old+novel)
            covariance_sum += (end-start) * _group_covariance(torch.cat(supports[start:end]), b.GAMMA_INC)
        covariance = covariance_sum / (old+novel)
        inverse = torch.pinverse(covariance.float())
        dp, df, dt, text = _text_subset(arrays, selected, device)
        query = torch.cat([x[q] for q in query_rows])
        target = torch.arange(old+novel, device=device).repeat_interleave(options.query)
        votes = proxy.reference_scores_from_features(query, old+novel, old, reference, covariance,
            dp, df, dt, text, cfg.DATASET.BETA, cov_inverse=inverse)
        lam = b.LAMBDA_T if b.TEXT_CALIBRATION else 0.
        records.append({'class_ids': selected, 'num_old': old, 'supports': supports,
            'reference_prototypes': reference, 'semantic_prototypes': F.normalize((1-lam)*text+lam*dp, dim=-1),
            'covariance': covariance, 'cov_inverse': inverse, 'query': query, 'targets': target,
            'reference_votes': votes, 'text_features': text, 'description_proto': dp,
            'description_features': df, 'description_targets': dt})
        trace.append({'episode': episode, 'class_ids': selected, 'num_old': old,
                      'support_indices': support_rows, 'query_indices': query_rows, 'auxiliary_edits': edits})
    return prepare_records(records, cfg, calibration), trace


def balanced(values, old):
    return torch.stack([values[mask].mean() for mask in (old, ~old) if bool(mask.any())]).mean()


def episode_loss(model, row, cfg, calibration, fixed, options):
    anchor = fusion_record_scores(row, cfg, calibration, fixed)
    scores = fusion_record_scores(row, cfg, calibration, fixed, model, blend=1.)
    logp = scores.log()-scores.sum(-1, keepdim=True).log()
    ref = (anchor.log()-anchor.sum(-1, keepdim=True).log()).detach()
    y, old_count = row['targets'], row['num_old']
    old = y < old_count
    ce = balanced(F.nll_loss(logp, y, reduction='none'), old)
    group = torch.stack((scores[:, :old_count].sum(-1), scores[:, old_count:].sum(-1)), dim=-1)
    group_ce = balanced(F.cross_entropy(group.log(), (~old).long(), reduction='none'), old)
    ids = torch.arange(len(y), device=y.device)
    def margins(values):
        other = values.clone()
        other[ids, y] = -torch.inf
        return values[ids, y]-other.max(-1).values
    ref_margin = margins(ref)
    protected = old & (anchor.argmax(-1) == y)
    protection = F.relu(ref_margin[protected]-margins(logp)[protected]).mean() if bool(protected.any()) else ce.new_zeros(())
    kl = balanced((ref.exp() * (ref-logp)).sum(-1), old)
    loss = ce + options.group_weight*group_ce + options.protection_weight*protection + options.distillation_weight*kl
    return loss, {'loss': float(loss.detach()), 'classification_loss': float(ce.detach()),
        'group_loss': float(group_ce.detach()), 'old_protection_loss': float(protection.detach()),
        'distillation_loss': float(kl.detach())}


def optimize(model, records, cfg, calibration, fixed, options, steps, callback=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=options.lr, weight_decay=options.weight_decay)
    history = []
    for step in range(1, steps+1):
        loss, metrics = episode_loss(model, records[(step-1) % len(records)], cfg, calibration, fixed, options)
        if not torch.isfinite(loss):
            raise ValueError('Non-finite competitive fusion loss.')
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step(); history.append(metrics)
        if callback and (step in options.early_checkpoints or step % options.validate_every == 0 or step == steps):
            callback(step, model, metrics)
    return history


def make_model(options, device):
    torch.manual_seed(options.seed)
    return CompetitiveFusion(options.hidden, options.method, options.use_support_quality,
                             options.group_limit, options.mix_limit).to(device)


def fit_competition(arrays, metadata, cfg, options=None, device='cpu', auxiliary_arrays=None, auxiliary_metadata=None,
                    fixed_anchor=None):
    options = options or FusionOptions()
    validate_options(options)
    if not metadata.get('complete_base_bank') or not metadata.get('has_text_features'):
        raise ValueError('Competitive fitting requires a complete original base/text cache.')
    if options.auxiliary != 'none' and auxiliary_arrays is None:
        auxiliary_arrays, auxiliary_metadata = arrays, metadata
    check_auxiliary(arrays, metadata, auxiliary_arrays, auxiliary_metadata, options.auxiliary)
    started = time.perf_counter()
    cfg = cfg.clone(); cfg.defrost()
    cfg.TRAINER.BiMC.GUIDANCE.QUERY = options.query
    cfg.freeze()
    cfg, protocol, pool = prepare_context(arrays, cfg, options.seed, options.sequences, options.teacher_shot)
    val = validation_records(arrays, cfg, protocol, device)
    if fixed_anchor is None:
        calibration, calibration_report = select_calibration(val, cfg, options.maximum_group_drop_pp)
        prepare_records(val, cfg, calibration)
        fixed, fixed_report = select_fixed_fusion(val, cfg, calibration, options.maximum_group_drop_pp,
                                                 options.weight_grid, options.bias_grid)
    else:
        from engine.safd_metric_runtime import reference_settings, sample_identity_hash
        expected = {'cache_sha256': metadata.get('cache_sha256'), 'fitting_seed': options.seed,
            'selection_protocol_sha256': protocol['sha256'], 'reference_settings': reference_settings(cfg),
            'clip_weight_sha256': metadata.get('clip_weight_sha256'),
            'base_sample_ids_sorted_sha256': sample_identity_hash(arrays['sample_id'])}
        if not expected['cache_sha256'] or any(fixed_anchor.get(k) != v for k, v in expected.items()):
            raise ValueError('Fixed anchor cache/seed/protocol/reference inputs differ from this fitting run.')
        calibration = validate_calibration(fixed_anchor['calibration'])
        fixed = validate_fixed(fixed_anchor['fixed'])
        prepare_records(val, cfg, calibration)
        original = evaluate_records(val, lambda row: row['reference_votes'])
        calibrated = evaluate_records(val, lambda row: score_record(row, cfg, calibration=calibration))
        anchor = evaluate_records(val, lambda row: fusion_record_scores(row, cfg, calibration, fixed))
        if not feasible(calibrated, (original,), options.maximum_group_drop_pp) or not feasible(
                anchor, (original, calibrated), options.maximum_group_drop_pp):
            raise ValueError('Replayed fixed anchor violates the requested B/N protection guard.')
        calibration_report = {'status': 'replayed_checked_base_selected_anchor',
                              'source_model_sha256': fixed_anchor['model_sha256']}
        fixed_report = {'reference': original, 'calibrated_reference': calibrated, 'selected_validation': anchor,
            'selected_fixed': fixed, 'candidates': [], 'candidate_count': 0,
            'source_candidate_count': fixed_anchor.get('fixed_candidate_count'),
            'source_model_sha256': fixed_anchor['model_sha256'],
            'source_anchor_selection_wall_seconds': fixed_anchor.get('fit_wall_seconds'),
            'selection_source': 'replayed_same_base_cache_seed_protocol_anchor',
            'maximum_group_drop_pp': options.maximum_group_drop_pp}
    print('Selected calibration:', calibration, 'fixed:', fixed,
          'fixed H=', fixed_report['selected_validation']['harmonic_accuracy'], flush=True)
    original, calibrated = fixed_report['reference'], fixed_report['calibrated_reference']
    anchor = fixed_report['selected_validation']
    references = (original, calibrated, anchor)
    model = make_model(options, device)
    best = {'step': 0, 'blend': 0., **anchor}
    candidates = [dict(best, feasible=True)]
    aux = auxiliary_arrays['features'] if options.auxiliary != 'none' else None
    train, trace = (training_episodes(arrays, cfg, pool, options, calibration, device, aux)
                    if options.steps else ([], []))
    def validate(step, fitted, losses):
        nonlocal best
        with torch.no_grad():
            for blend in sorted(set(v for v in options.blend_grid if v > 0)):
                metrics = evaluate_records(val, lambda row: fusion_record_scores(row, cfg, calibration, fixed, fitted, blend))
                ok = feasible(metrics, references, options.maximum_group_drop_pp)
                candidate = {'step': step, 'blend': blend, **metrics, 'feasible': ok, 'training_episode': losses}
                candidates.append(candidate)
                if ok and metrics['harmonic_accuracy'] > best['harmonic_accuracy'] + 1e-12:
                    best = dict(candidate)
                print(f"Fusion step={step} blend={blend:g}: B={metrics['base_accuracy']:.4f} "
                      f"N={metrics['novel_accuracy']:.4f} H={metrics['harmonic_accuracy']:.4f} feasible={ok}", flush=True)
    snapshots = {}
    def checkpoint(step, fitted, losses):
        validate(step, fitted, losses)
        if best['step'] == step:
            snapshots['selected'] = {k: v.detach().clone() for k, v in fitted.state_dict().items()}
    history = optimize(model, train, cfg, calibration, fixed, options, options.steps, checkpoint)
    if best['step']:
        model.load_state_dict(snapshots['selected'])
    diagnostics = {
        'original': group_diagnostics(val, lambda row: row['reference_votes']),
        'calibrated': group_diagnostics(val, lambda row: fusion_record_scores(row, cfg, calibration, identity_fixed(cfg))),
        'fixed': group_diagnostics(val, lambda row: fusion_record_scores(row, cfg, calibration, fixed)),
        'selected': group_diagnostics(val, lambda row: fusion_record_scores(row, cfg, calibration, fixed, model, best['blend']))}
    selection_finished = time.perf_counter()
    final = make_model(options, device)
    final_trace, final_history = [], []
    if best['step']:
        labels = np.asarray(arrays['labels'])
        full = {int(c): np.flatnonzero(labels == c).tolist() for c in np.unique(labels)}
        final_train, final_trace = training_episodes(arrays, cfg, full, options, calibration, device, aux, seed_offset=1000003)
        final_history = optimize(final, final_train, cfg, calibration, fixed, options, best['step'])
    final.eval().requires_grad_(False)
    report = {'schema_version': 1, 'training_protocol': 'base_only_class_disjoint_competition_selection_then_fresh_all_base_refit',
        'options': asdict(options), 'reference': original, 'calibrated_reference': calibrated, 'fixed_reference': anchor,
        'calibration': calibration, 'calibration_report': calibration_report, 'fixed': fixed, 'fixed_report': fixed_report,
        'selected_validation': best, 'selected_step': best['step'], 'blend': best['blend'], 'checkpoints': candidates,
        'selection_protocol': protocol, 'selection_training_episodes': trace, 'final_refit_training_episodes': final_trace,
        'selection_training_history': history, 'final_training_history': final_history, 'group_diagnostics': diagnostics,
        'model_config': final.configuration(), 'trainable_parameters': sum(p.numel() for p in final.parameters()),
        'selection_wall_seconds': selection_finished-started, 'fit_wall_seconds': time.perf_counter()-started,
        'anchor_selection_wall_seconds': (fixed_anchor.get('fit_wall_seconds') if fixed_anchor else 0.),
        'phase_order': ['reserve_A_queries_and_V_teachers', 'select_calibration', 'select_fixed_fusion',
                        'fit_A_only_and_select', 'selection_ended', 'fresh_all_base_fixed_step_refit', 'freeze'],
        'limitations': ['Base-supervised fitting, not training-free.', 'No V teacher features enter fitting or selection scores.',
            'Validation stages overlap.', 'Zero-step/blend must be reported as no learned gain.',
            'Fixed fusion and learned candidates are development selections, not benchmark results.']}
    return final, report, cfg


def export_fitted_fusion(path, model, report, cfg, metadata, cache_path, sample_ids, auxiliary_cache_path=None):
    from engine.safd_metric_runtime import reference_settings, sample_identity_hash
    from engine.competitive_runtime import export_fusion
    root = Path(__file__).resolve().parents[1]
    content = {key: metadata.get(key) for key in ('ordinary_description_sha256', 'text_template',
               'support_transform', 'input_provenance', 'base_class_names')}
    source_files = ('models/competitive_fusion.py', 'engine/competitive_evaluation.py', 'engine/competitive_training.py',
                    'engine/competitive_runtime.py', 'engine/prototype_evaluation.py', 'models/bimc.py')
    saved = {'dataset': metadata['dataset'], 'data_source': 'base_training_only', 'precision': cfg.TRAINER.BiMC.PREC,
        'feature_dim': int(metadata['feature_dim']), 'base_class_ids': metadata['base_class_ids'],
        'base_sample_ids_sorted_sha256': sample_identity_hash(sample_ids), 'reference_settings': reference_settings(cfg),
        'clip_weight_sha256': metadata['clip_weight_sha256'], **content,
        'model_config': model.configuration(), 'trainable_parameters': report['trainable_parameters'],
        'calibration': report['calibration'], 'fixed': report['fixed'], 'blend': report['blend'],
        'selected_step': report['selected_step'], 'fitting_seed': report['options']['seed'],
        'fixed_candidate_count': report['fixed_report']['candidate_count'],
        'selection_protocol_sha256': report['selection_protocol']['sha256'],
        'training_protocol': report['training_protocol'], 'auxiliary': report['options']['auxiliary'],
        'fit_wall_seconds': report['fit_wall_seconds'], 'cache_extraction_wall_seconds': metadata.get('wall_seconds'),
        'anchor_selection_wall_seconds': report['anchor_selection_wall_seconds'],
        'cache_sha256': hashlib.sha256(Path(cache_path).read_bytes()).hexdigest(),
        'auxiliary_cache_sha256': hashlib.sha256(Path(auxiliary_cache_path).read_bytes()).hexdigest() if auxiliary_cache_path else None,
        'source_files_sha256': {f: hashlib.sha256((root/f).read_bytes()).hexdigest() for f in source_files}}
    return export_fusion(path, model, saved, report)
