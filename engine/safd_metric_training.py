"""Base-only episodic fitting of a small SAFD-informed metric, then frozen export.

The class-disjoint selection split reserves the pseudo-base queries globally.
Neither those rows nor any pseudo-validation class fits a basis or a parameter.
Only after selecting a training duration is a fresh model refitted on all base
training images. No CLIP model is constructed or loaded by this module.
"""

from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from engine.guidance_calibration import sample_protocol, digest
from engine.residual_training import _group_covariance
from models.safd_metric import (SAFDMetricAdapter, scale_responses,
                                fit_explained_scatter, build_response_basis,
                                prepare_residual_classifier, apply_metric_residual,
                                metric_from_scatter)


@dataclass
class TrainingOptions:
    steps: int = 1000
    validate_every: int = 100
    episode_pool: int = 100
    rank: int = 64
    ridge: float = .1
    lr: float = .01
    weight_decay: float = .001
    seed: int = 1
    sequences: int = 20
    old_way: int = 10
    novel_way: int = 5
    old_shot: int = 10
    query: int = 5
    maximum_group_drop_pp: float = .2
    gain: float = 16.
    basis: str = 'explained'
    method: str = 'learned'
    strength_grid: tuple = (0., .1, 1., 10.)


def validate_options(options):
    for name in ('validate_every', 'episode_pool', 'rank', 'seed', 'sequences',
                 'old_way', 'novel_way', 'old_shot', 'query'):
        value = getattr(options, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(name + ' must be a positive integer.')
    if isinstance(options.steps, bool) or not isinstance(options.steps, int) or options.steps < 0:
        raise ValueError('steps must be a nonnegative integer.')
    for name in ('ridge', 'lr', 'gain'):
        value = getattr(options, name)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(name + ' must be finite and positive.')
    if not np.isfinite(options.weight_decay) or options.weight_decay < 0:
        raise ValueError('weight_decay must be finite and nonnegative.')
    if not np.isfinite(options.maximum_group_drop_pp) or options.maximum_group_drop_pp < 0:
        raise ValueError('maximum_group_drop_pp must be finite and nonnegative.')
    if options.basis not in ('explained', 'original_within'):
        raise ValueError('basis must be explained or original_within.')
    if options.method not in ('learned', 'closed_form'):
        raise ValueError('method must be learned or closed_form.')
    if (not options.strength_grid or 0 not in options.strength_grid
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not np.isfinite(v) or v < 0 for v in options.strength_grid)):
        raise ValueError('strength_grid must contain zero and finite nonnegative strengths.')


def reserve_selection_split(cfg, labels, sample_ids, options):
    """Use the existing class split, with one fixed A-query bank for all tasks."""
    local = cfg.clone()
    local.defrost()
    local.SEED = options.seed
    local.TRAINER.BiMC.GUIDANCE.CALIBRATION_SEED = options.seed + 10101
    local.TRAINER.BiMC.GUIDANCE.VAL_SEQUENCES = options.sequences
    local.freeze()
    protocol = sample_protocol(local, labels.cpu(), [int(value) for value in sample_ids])
    count_a = len(protocol['pseudo_base_class_ids'])
    queries = {cls: list(rows) for cls, rows in zip(
        protocol['pseudo_base_class_ids'], protocol['sequences'][0]['query_indices'][:count_a])}
    groups = {int(c): torch.where(labels.cpu() == c)[0].tolist() for c in torch.unique(labels.cpu())}
    pool = {cls: [i for i in groups[cls] if i not in set(queries[cls])]
            for cls in protocol['pseudo_base_class_ids']}
    for sequence in protocol['sequences']:
        for j, cls in enumerate(protocol['pseudo_base_class_ids']):
            sequence['support_indices'][j] = list(pool[cls])
            sequence['query_indices'][j] = list(queries[cls])
    fit = [i for cls in protocol['pseudo_base_class_ids'] for i in pool[cls]]
    protocol['reserved_pseudo_base_query_indices'] = {str(k): v for k, v in queries.items()}
    protocol['fitting_indices'] = fit
    protocol['fitting_rule'] = 'A_only_excluding_globally_reserved_A_queries; no_V_rows'
    protocol.pop('sha256', None)
    protocol['sha256'] = digest(protocol)
    return local, protocol, pool


def _reference_proxy(cfg):
    # Importing the class defines methods but never calls its constructor.
    from models.bimc import BiMC
    proxy = SimpleNamespace(cfg=cfg, frequency_fusion_enabled=False,
                            frequency_router=None)
    proxy.soft_calibration = BiMC.soft_calibration.__get__(proxy)
    proxy.reference_scores_from_features = BiMC.reference_scores_from_features.__get__(proxy)
    return proxy


def _text_subset(arrays, class_ids, device):
    index = torch.tensor(class_ids, dtype=torch.long, device=device)
    description = torch.as_tensor(arrays['description_features'], device=device).float()
    targets = torch.as_tensor(arrays['description_targets'], device=device).long()
    features, local_targets = [], []
    for j, cls in enumerate(class_ids):
        rows = description[targets == cls]
        if not len(rows):
            raise ValueError(f'No ordinary descriptions for base class {cls}.')
        features.append(rows)
        local_targets.append(torch.full((len(rows),), j, dtype=torch.long, device=device))
    return (torch.as_tensor(arrays['description_proto'], device=device).float()[index],
            torch.cat(features), torch.cat(local_targets),
            torch.as_tensor(arrays['text_features'], device=device).float()[index])


@torch.no_grad()
def _record(proxy, cfg, x, arrays, classes, supports, queries, num_old, covariance):
    raw = F.normalize(torch.stack([x[rows].mean(0) for rows in supports]), dim=-1)
    guided = raw.clone()
    if cfg.TRAINER.BiMC.VISION_CALIBRATION and len(classes) > num_old:
        guided[num_old:] = proxy.soft_calibration(raw[:num_old], raw[num_old:])
    query = torch.cat([x[rows] for rows in queries])
    target = torch.cat([torch.full((len(rows),), i, dtype=torch.long, device=x.device)
                        for i, rows in enumerate(queries)])
    dp, df, dt, text = _text_subset(arrays, classes, x.device)
    votes = proxy.reference_scores_from_features(
        query, len(classes), num_old, guided, covariance, dp, df, dt,
        text, cfg.DATASET.BETA, cov_inverse=torch.pinverse(covariance.float()))
    if not torch.isfinite(votes).all() or (votes <= 0).any():
        raise ValueError('BiMC reference votes must be finite and positive.')
    return {'query': query.detach(), 'prototypes': raw.detach(),
            'reference_votes': votes.detach(),
            'log_reference': votes.float().clamp_min(1e-8).log().detach(),
            'targets': target, 'num_old': num_old}


@torch.no_grad()
def validation_records(cfg, x, arrays, protocol):
    """Support-only statistics, old calibrated prototypes frozen at arrival."""
    proxy, records = _reference_proxy(cfg), []
    b = cfg.TRAINER.BiMC
    for sequence_id, seq in enumerate(protocol['sequences']):
        classes, supports, queries = seq['class_ids'], seq['support_indices'], seq['query_indices']
        old = len(protocol['pseudo_base_class_ids'])
        covariance_sum = old * _group_covariance(x[[i for rows in supports[:old] for i in rows]], b.GAMMA_BASE)
        start = old
        for stage, size in enumerate(seq['stage_sizes'], 1):
            end = start + size
            covariance_sum = covariance_sum + size * _group_covariance(
                x[[i for rows in supports[start:end] for i in rows]], b.GAMMA_INC)
            row = _record(proxy, cfg, x, arrays, classes[:end], supports[:end], queries[:end],
                          old, covariance_sum / end)
            row.update(sequence=sequence_id, stage=stage)
            records.append(row)
            start = end
    return records


@torch.no_grad()
def training_records(cfg, x, arrays, pool, options, seed_offset=0):
    rng, proxy, records, trace = random.Random(options.seed + seed_offset), _reference_proxy(cfg), [], []
    classes = sorted(pool)
    old = min(options.old_way, len(classes) - 1)
    novel = min(options.novel_way, len(classes) - old)
    if old < 1 or novel < 1:
        raise ValueError('Episodic fitting needs at least two fitting classes.')
    shot = int(cfg.DATASET.NUM_INC_SHOT)
    for episode in range(options.episode_pool):
        selected = rng.sample(classes, old + novel)
        supports, queries = [], []
        for j, cls in enumerate(selected):
            count = options.old_shot if j < old else shot
            if len(pool[cls]) < count + options.query:
                raise ValueError(f'Class {cls} needs {count + options.query} fitting rows; reduce old-shot/query.')
            chosen = rng.sample(pool[cls], count + options.query)
            supports.append(chosen[:count])
            queries.append(chosen[count:])
        covariance = (old * _group_covariance(x[[i for rows in supports[:old] for i in rows]], cfg.TRAINER.BiMC.GAMMA_BASE)
                      + novel * _group_covariance(x[[i for rows in supports[old:] for i in rows]], cfg.TRAINER.BiMC.GAMMA_INC)) / (old + novel)
        records.append(_record(proxy, cfg, x, arrays, selected, supports, queries, old, covariance))
        trace.append({'class_ids': selected, 'support_indices': supports, 'query_indices': queries})
    return records, trace


@torch.no_grad()
def evaluate(adapter, records, gain):
    values = []
    metric = adapter.to_metric()
    if int(torch.linalg.cholesky_ex(metric.double()).info) != 0:
        raise ValueError('Export-rounded adapter metric must be positive definite.')
    for row in records:
        prepared = prepare_residual_classifier(row['prototypes'], metric, validate_metric=False)
        votes = apply_metric_residual(row['reference_votes'], row['query'], prepared, gain)
        correct, old = votes.argmax(-1) == row['targets'], row['targets'] < row['num_old']
        base, novel = (100 * correct[old].float().mean().item(),
                       100 * correct[~old].float().mean().item())
        values.append((base, novel, 2 * base * novel / (base + novel) if base + novel else 0.))
    if not values:
        raise ValueError('No pseudo-incremental validation stages.')
    a = np.asarray(values)
    return {'base_accuracy': float(a[:, 0].mean()), 'novel_accuracy': float(a[:, 1].mean()),
            'harmonic_accuracy': float(a[:, 2].mean()), 'stages': len(values)}


def train_steps(adapter, records, options, steps, callback=None):
    """Only theta is optimized; all frozen feature records are detached."""
    if not steps or not adapter.theta.numel():
        return []
    optimizer = torch.optim.Adam([adapter.theta], lr=options.lr, weight_decay=options.weight_decay)
    losses = []
    for step in range(1, steps + 1):
        row = records[(step - 1) % len(records)]
        logits = row['log_reference'] + options.gain * adapter.delta_scores(row['query'], row['prototypes'])
        old = row['targets'] < row['num_old']
        # Equal group weight prevents a larger pseudo-base class bank from
        # dominating optimization simply because it supplies more queries.
        groups = [mask for mask in (old, ~old) if bool(mask.any())]
        loss = torch.stack([F.cross_entropy(logits[mask], row['targets'][mask])
                            for mask in groups]).mean()
        if not torch.isfinite(loss):
            raise ValueError('Non-finite episodic metric loss.')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if callback and (step % options.validate_every == 0 or step == steps):
            callback(step, adapter, losses[-1])
    return losses


def _fit_scatter(x, features, labels, indices, options):
    chosen = torch.tensor(indices, dtype=torch.long)
    return fit_explained_scatter(x.cpu()[chosen], scale_responses(features.cpu()[chosen]),
                                 labels.cpu()[chosen], rank=options.rank, ridge=options.ridge,
                                 mode=options.basis)


def _fit_basis(x, features, labels, indices, options):
    state = _fit_scatter(x, features, labels, indices, options)
    return build_response_basis(state, options.rank), state['diagnostics']


class FrozenMetric(torch.nn.Module):
    """Export-compatible analytic control with zero trainable parameters."""

    def __init__(self, metric):
        super().__init__()
        self.register_buffer('metric', metric.detach().float())
        self.register_buffer('basis', self.metric.new_empty(len(metric), 0))
        self.register_buffer('theta', self.metric.new_empty(0))

    def to_metric(self):
        return self.metric.detach()


def _feasible(metrics, reference, maximum_drop):
    return (metrics['base_accuracy'] >= reference['base_accuracy'] - maximum_drop - 1e-12
            and metrics['novel_accuracy'] >= reference['novel_accuracy'] - maximum_drop - 1e-12)


def _fit_closed_form(arrays, cfg, options, x, features, labels, protocol, device, started):
    """Select an analytic strength on exactly the learned method's A/V split."""
    selection_state = _fit_scatter(x, features, labels, protocol['fitting_indices'], options)
    val = validation_records(cfg, x.to(device), arrays, protocol)
    zero = FrozenMetric(torch.eye(x.shape[1])).to(device)
    reference = evaluate(zero, val, options.gain)
    best, checkpoints = {'strength': 0., 'step': 0, **reference}, []
    for strength in sorted(set(float(v) for v in options.strength_grid)):
        fitted = FrozenMetric(metric_from_scatter(selection_state, strength)).to(device)
        metrics = evaluate(fitted, val, options.gain)
        feasible = _feasible(metrics, reference, options.maximum_group_drop_pp)
        row = {'strength': strength, 'step': 0, **metrics, 'feasible': feasible}
        checkpoints.append(row)
        if feasible and metrics['harmonic_accuracy'] > best['harmonic_accuracy'] + 1e-12:
            best = dict(row)
        print(f'Metric validation strength {strength}: H={metrics["harmonic_accuracy"]:.6f} feasible={feasible}', flush=True)
    selection_finished = time.perf_counter()
    final_indices = list(range(len(labels)))
    final_state = _fit_scatter(x, features, labels, final_indices, options)
    adapter = FrozenMetric(metric_from_scatter(final_state, best['strength'])).to(device)
    report = {'schema_version': 1,
              'training_protocol': 'base_only_class_disjoint_analytic_selection_then_all_base_fixed_strength_refit',
              'options': asdict(options), 'selected_step': 0, 'selected_strength': best['strength'],
              'gain': options.gain if best['strength'] else 0., 'reference': reference,
              'selected_validation': best, 'checkpoints': checkpoints,
              'selection_protocol': protocol,
              'selection_basis_diagnostics': selection_state['diagnostics'],
              'final_basis_diagnostics': final_state['diagnostics'],
              'selection_training_episodes': [], 'final_refit_training_episodes': [],
              'final_refit_indices': final_indices, 'trainable_parameters': 0,
              'phase_order': ['reserve_validation', 'fit_A_scatter', 'select_strength_on_V',
                              'selection_ended', 'refit_all_base_scatter', 'freeze'],
              'loss_weighting': 'not_applicable_no_gradient_training',
              'selection_wall_seconds': selection_finished - started,
              'final_refit_wall_seconds': time.perf_counter() - selection_finished,
              'fit_wall_seconds': time.perf_counter() - started,
              'limitations': ['Supervised base-only analytic fitting; not data-free.',
                              'Frozen incremental parameters do not guarantee unchanged old-class accuracy.',
                              'Validation stages share images; they are not independent statistical replicates.']}
    return adapter, report, cfg


def fit_metric(arrays, cfg, options=None, device='cpu'):
    started = time.perf_counter()
    options = options or TrainingOptions()
    validate_options(options)
    features = torch.as_tensor(arrays['features']).detach().float().cpu()
    labels = torch.as_tensor(arrays['labels']).detach().long().cpu()
    sample_ids = [int(i) for i in arrays['sample_id']]
    x = F.normalize(features[:, 0], dim=-1)
    cfg, protocol, pool = reserve_selection_split(cfg, labels, sample_ids, options)
    if options.method == 'closed_form':
        return _fit_closed_form(arrays, cfg, options, x, features, labels, protocol, device, started)
    fitting = protocol['fitting_indices']
    basis, diagnostics = _fit_basis(x, features, labels, fitting, options)
    adapter = SAFDMetricAdapter(basis).to(device)
    xx = x.to(device)
    val = validation_records(cfg, xx, arrays, protocol)
    reference = evaluate(adapter, val, options.gain)
    best, checkpoints = {'step': 0, **reference}, []

    def validate(step, fitted, loss=None):
        nonlocal best
        metrics = evaluate(fitted, val, options.gain)
        feasible = _feasible(metrics, reference, options.maximum_group_drop_pp)
        row = {'step': step, **metrics, 'feasible': feasible, 'loss': loss}
        checkpoints.append(row)
        if feasible and metrics['harmonic_accuracy'] > best['harmonic_accuracy'] + 1e-12:
            best = dict(row)
        print(f'Metric validation step {step}: H={metrics["harmonic_accuracy"]:.6f} feasible={feasible}', flush=True)

    validate(0, adapter)
    if options.steps and adapter.theta.numel():
        train, trace = training_records(cfg, xx, arrays, pool, options)
        train_steps(adapter, train, options, options.steps, validate)
    else:
        trace = []
    # Model selection has ended. A fresh basis and theta may now use all
    # original base images, including the previously reserved selection rows.
    selected_step = best['step']
    selection_finished = time.perf_counter()
    del val
    final_indices = list(range(len(labels)))
    final_basis, final_diagnostics = _fit_basis(x, features, labels, final_indices, options)
    final_adapter = SAFDMetricAdapter(final_basis).to(device)
    final_trace = []
    if selected_step and final_adapter.theta.numel():
        final_pool = {int(c): torch.where(labels == c)[0].tolist() for c in torch.unique(labels)}
        final_records, final_trace = training_records(cfg, xx, arrays, final_pool, options, seed_offset=1000003)
        train_steps(final_adapter, final_records, options, selected_step)
    gain = options.gain if selected_step and final_adapter.theta.numel() else 0.
    report = {'schema_version': 1, 'training_protocol': 'base_only_class_disjoint_episodic_selection_then_all_base_fixed_step_refit',
              'options': asdict(options), 'selected_step': selected_step, 'gain': gain,
              'reference': reference, 'selected_validation': best, 'checkpoints': checkpoints,
              'selection_protocol': protocol, 'selection_basis_diagnostics': diagnostics,
              'final_basis_diagnostics': final_diagnostics,
              'selection_training_episodes': trace, 'final_refit_training_episodes': final_trace,
              'final_refit_indices': final_indices,
              'phase_order': ['reserve_validation', 'fit_A_basis', 'train_A_theta_and_select_step',
                              'selection_ended', 'refit_all_base_basis', 'train_all_base_fixed_selected_steps', 'freeze'],
              'trainable_parameters': int(final_adapter.theta.numel()),
              'loss_weighting': 'mean_of_nonempty_old_and_novel_group_cross_entropies',
              'selection_wall_seconds': selection_finished - started,
              'final_refit_wall_seconds': time.perf_counter() - selection_finished,
              'fit_wall_seconds': time.perf_counter() - started,
              'limitations': ['Base-only supervised gradient adaptation; not training-free.',
                              'Frozen incremental parameters do not guarantee unchanged old-class accuracy.',
                              'Validation stages share images; they are not independent statistical replicates.']}
    return final_adapter, report, cfg


def export_metric(path, adapter, report, cfg, cache_metadata, cache_path, sample_ids):
    from engine.safd_metric_runtime import reference_settings, sample_identity_hash, validate_reference_metadata
    path = Path(path)
    if path.suffix != '.npz':
        raise ValueError('Metric output must end in .npz.')
    if any(p.exists() for p in (path, path.with_suffix('.json'), path.with_suffix('.training.json'))):
        raise ValueError('Refusing to overwrite an existing fitted metric; choose a new --output.')
    content = {key: cache_metadata.get(key) for key in
               ('ordinary_description_sha256', 'text_template', 'support_transform',
                'input_provenance', 'base_class_names')}
    content['precision'] = cfg.TRAINER.BiMC.PREC
    validate_reference_metadata(content)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, metric=adapter.to_metric().cpu().numpy().astype(np.float32),
                        basis=adapter.basis.detach().cpu().numpy(), theta=adapter.theta.detach().cpu().numpy())
    root = Path(__file__).resolve().parents[1]
    source_files = ('engine/safd_metric_training.py', 'engine/safd_metric_runtime.py',
                    'models/safd_metric.py', 'models/bimc.py', 'tools/train_safd_metric.py')
    analytic = report['options']['method'] == 'closed_form'
    method_name = ('analytic metric control' if analytic else 'learned metric')
    metadata = {'schema_version': 1, 'dataset': cache_metadata['dataset'],
                'source_dataset': cache_metadata['dataset'], 'source': cache_metadata['source'],
                'data_source': 'base_training_only',
                'clip_weight_sha256': cache_metadata['clip_weight_sha256'],
                'feature_dim': int(adapter.basis.shape[0]), 'base_class_ids': cache_metadata['base_class_ids'],
                'base_sample_ids_sorted_sha256': sample_identity_hash(sample_ids),
                'gain': report['gain'], 'selected_gain': report['gain'], 'selected_step': report['selected_step'],
                'selected_strength': report.get('selected_strength'),
                'reference_settings': reference_settings(cfg), 'training_protocol': report['training_protocol'],
                'method': ((cache_metadata['source'] + '-response ') if report['options']['basis'] == 'explained' else 'original-within ') + method_name,
                'fit_method': report['options']['method'],
                'basis_mode': report['options']['basis'], 'trainable_parameters': report['trainable_parameters'],
                'cache_path': str(Path(cache_path).resolve()), 'cache_sha256': hashlib.sha256(Path(cache_path).read_bytes()).hexdigest(),
                'model_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'selection_report': {k: report[k] for k in ('selected_validation', 'reference', 'checkpoints')},
                'complete_base_bank': cache_metadata.get('complete_base_bank'),
                'fit_wall_seconds': report['fit_wall_seconds'],
                'cache_extraction_wall_seconds': cache_metadata.get('wall_seconds'),
                **content,
                'source_files_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                         for name in source_files},
                'adaptation': ('supervised_base_only_analytic_fitting; incremental_parameters_frozen' if analytic
                               else 'supervised_base_only_gradient_training; incremental_parameters_frozen')}
    path.with_suffix('.json').write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
    path.with_suffix('.training.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return metadata
