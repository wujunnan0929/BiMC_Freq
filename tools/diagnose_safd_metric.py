"""Replay base-only SAFD metric selection from its cache, without loading CLIP.

This diagnostic never exports a fitted model or evaluates benchmark test data.
Gain sweeps are post-hoc scoring probes; training uses the report's original gain.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', required=True)
    p.add_argument('--training-report', required=True)
    p.add_argument('--output-dir', required=True, help='Fresh diagnostic directory; existing files are never overwritten.')
    p.add_argument('--mode', choices=('closed_form', 'learned'), default='closed_form')
    p.add_argument('--gains', type=float, nargs='+', default=[0., .01, .1, 1., 16.])
    p.add_argument('--strengths', type=float, nargs='+', default=[0., .001, .01, .1, 1., 10.])
    p.add_argument('--checkpoints', type=int, nargs='+', default=[0, 1, 5, 10, 20, 50, 100, 1000])
    p.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--reference-tolerance-pp', type=float, default=.01)
    p.add_argument('--save-score-probe', action='store_true', help='Save first-stage features, scores and metric snapshots for numerical inspection.')
    a = p.parse_args(argv)
    import math
    if a.threads < 1 or not math.isfinite(a.reference_tolerance_pp) or a.reference_tolerance_pp < 0:
        p.error('threads must be positive and reference tolerance finite/nonnegative.')
    for name in ('gains', 'strengths'):
        values = getattr(a, name)
        if 0 not in values or any(not math.isfinite(v) or v < 0 for v in values):
            p.error(name + ' must include zero and contain finite nonnegative values.')
        setattr(a, name, sorted(set(values)))
    if 0 not in a.checkpoints or min(a.checkpoints) < 0:
        p.error('checkpoints must include zero and be nonnegative.')
    a.checkpoints = sorted(set(a.checkpoints))
    return a


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def _percentiles(tensor):
    import torch
    v = tensor.detach().cpu().double().flatten()
    if not len(v):
        return None
    return dict(zip(('min', 'p50', 'p90', 'p99', 'max'),
                    torch.quantile(v, v.new_tensor([0., .5, .9, .99, 1.])).tolist()))


def metric_diagnostics(matrix):
    import torch
    m = matrix.detach().cpu().double()
    eig = torch.linalg.eigvalsh(m)
    delta = m - torch.eye(len(m), dtype=m.dtype)
    return {'finite': bool(torch.isfinite(m).all()), 'eigenvalue_min': float(eig.min()),
            'positive_definite': bool(eig.min() > 0),
            'eigenvalue_max': float(eig.max()), 'condition_number': float(eig.max() / eig.min()),
            'trace': float(m.trace()), 'identity_difference_frobenius': float(delta.norm()),
            'identity_difference_max_abs': float(delta.abs().max()),
            'is_identity': bool(torch.count_nonzero(delta) == 0),
            'symmetric_max_error': float((m - m.T).abs().max())}


def score_sweep(records, matrix, gains):
    """Compile once per stage, then sweep gains using runtime-equivalent scores.

    Each stage also checks a nonzero gain against the actual public scorer.
    Candidate prediction counts describe repeated stage-query observations.
    """
    import numpy as np
    import torch
    from torch.nn import functional as F
    from models.safd_metric import prepare_residual_classifier, apply_metric_residual
    buckets = {g: {'stages': [], 'margins': [], 'pair_changes': [], 'true_before': [],
                   'true_after': [], 'residual_abs_sum': 0., 'residual_count': 0,
                   'residual_max_abs': 0.} for g in gains}
    probe = None
    for stage_index, row in enumerate(records):
        prepared = prepare_residual_classifier(row['prototypes'], matrix, validate_metric=False)
        device = row['query'].device
        query = F.normalize(row['query'].detach().double(), dim=-1)
        delta = query @ prepared['weights'].to(device) + prepared['bias'].to(device)
        reference = row['reference_votes']
        log_reference = reference.double().log()
        y = row['targets']
        idx = torch.arange(len(y), device=device)
        old = y < row['num_old']
        first = reference.argmax(1)
        # Match argmax tie handling of reference_votes rather than topk ties.
        other = log_reference.clone()
        other[idx, first] = -torch.inf
        second = other.argmax(1)
        margin = log_reference[idx, first] - log_reference[idx, second]
        true_other = log_reference.clone()
        true_other[idx, y] = -torch.inf
        true_before = log_reference[idx, y] - true_other.max(1).values
        baseline_correct = first == y
        if probe is None:
            pick = torch.cat((torch.where(old)[0][:16], torch.where(~old)[0][:16]))
            probe = {'query': row['query'][pick].detach().cpu().numpy(),
                     'raw_prototypes': row['prototypes'].detach().cpu().numpy(),
                     'query_row_indices_in_stage': pick.cpu().numpy(),
                     'targets_local': y[pick].cpu().numpy(),
                     'reference_votes': reference[pick].cpu().numpy(),
                     'metric_residual_before_gain': delta[pick].cpu().numpy()}
            if 'class_ids' in row:
                probe['class_ids_global'] = np.asarray(row['class_ids'], dtype=np.int64)
                probe['query_sample_ids'] = np.asarray(row['query_sample_ids'], dtype=np.int64)[pick.cpu().numpy()]
                probe['targets_global'] = np.asarray(row['class_ids'], dtype=np.int64)[y[pick].cpu().numpy()]
        nonzero_gain = next((g for g in reversed(gains) if g > 0), 0.)
        for gain, b in buckets.items():
            logits = log_reference + gain * delta
            if gain == 0 or prepared['is_identity']:
                votes = reference
            else:
                votes = (logits.softmax(-1) * reference.double().sum(-1, keepdim=True)).to(
                    torch.float64 if reference.dtype == torch.float64 else torch.float32)
                votes = votes.clamp_min(torch.nextafter(votes.new_tensor(0.), votes.new_tensor(1.)))
            if gain == nonzero_gain:
                actual = apply_metric_residual(reference, row['query'], prepared, gain)
                if not torch.equal(votes, actual):
                    raise ValueError('Diagnostic scores differ from frozen runtime; stop interpreting this sweep.')
            if not torch.isfinite(votes).all() or not torch.isfinite(logits).all():
                raise ValueError('Non-finite metric/log scores.')
            prediction = votes.argmax(1)
            correct = prediction == y
            base = 100 * correct[old].float().mean().item()
            novel = 100 * correct[~old].float().mean().item()
            corrected = ~baseline_correct & correct
            damaged = baseline_correct & ~correct
            item = {'sequence': row.get('sequence'), 'stage': row.get('stage', stage_index),
                    'base_accuracy': base, 'novel_accuracy': novel,
                    'harmonic_accuracy': 2 * base * novel / (base + novel) if base + novel else 0.,
                    'query_observations': len(y), 'num_old': row['num_old'],
                    'num_classes': reference.shape[1],
                    'base_queries': int(old.sum()), 'novel_queries': int((~old).sum()),
                    'reference_correct': int(baseline_correct.sum()), 'correct': int(correct.sum()),
                    'corrected': int(corrected.sum()), 'damaged': int(damaged.sum()),
                    'changed': int((prediction != first).sum()),
                    'base_corrected': int((corrected & old).sum()),
                    'base_damaged': int((damaged & old).sum()),
                    'novel_corrected': int((corrected & ~old).sum()),
                    'novel_damaged': int((damaged & ~old).sum()),
                    'predicted_old': int((prediction < row['num_old']).sum()),
                    'novel_to_old': int(((prediction < row['num_old']) & ~old).sum()),
                    'old_to_novel': int(((prediction >= row['num_old']) & old).sum())}
            b['stages'].append(item)
            b['margins'].append(margin.cpu())
            b['pair_changes'].append((gain * (delta[idx, first] - delta[idx, second])).cpu())
            b['true_before'].append(true_before.cpu())
            output_other = logits.clone()
            output_other[idx, y] = -torch.inf
            b['true_after'].append((logits[idx, y] - output_other.max(1).values).cpu())
            b['residual_abs_sum'] += float((gain * delta).abs().sum())
            b['residual_count'] += delta.numel()
            b['residual_max_abs'] = max(b['residual_max_abs'], float((gain * delta).abs().max()))
    summaries = []
    for gain, b in buckets.items():
        stages = b['stages']
        margin, change = torch.cat(b['margins']), torch.cat(b['pair_changes'])
        result = {'gain': gain, 'stages': len(stages),
                  **{k: float(np.mean([s[k] for s in stages])) for k in
                     ('base_accuracy', 'novel_accuracy', 'harmonic_accuracy')},
                  **{k: sum(s[k] for s in stages) for k in ('query_observations', 'reference_correct',
                     'correct', 'corrected', 'damaged', 'changed', 'base_corrected', 'base_damaged',
                     'novel_corrected', 'novel_damaged', 'predicted_old', 'base_queries',
                     'novel_queries', 'novel_to_old', 'old_to_novel')},
                  'reference_log_top2_margin': _percentiles(margin),
                  'reference_top2_residual_change': _percentiles(change),
                  'abs_pair_change_over_reference_margin': _percentiles(change.abs() / margin.clamp_min(1e-12)),
                  'fraction_pair_change_exceeds_margin': float((change.abs() > margin).double().mean()),
                  'fraction_reference_pair_reversed': float((margin + change < 0).double().mean()),
                  'true_class_log_margin_before': _percentiles(torch.cat(b['true_before'])),
                  'true_class_log_margin_after': _percentiles(torch.cat(b['true_after'])),
                  'mean_abs_scaled_residual': b['residual_abs_sum'] / b['residual_count'],
                  'max_abs_scaled_residual': b['residual_max_abs'],
                  'runtime_score_parity_checked': True}
        result['net_correct'] = result['corrected'] - result['damaged']
        summaries.append(result)
    return summaries, {str(g): b['stages'] for g, b in buckets.items()}, probe


def training_metrics(adapter, records, gain):
    import torch
    from torch.nn import functional as F
    values = []
    with torch.no_grad():
        for row in records:
            logits = row['log_reference'] + gain * adapter.delta_scores(row['query'], row['prototypes'])
            old = row['targets'] < row['num_old']
            correct = logits.argmax(1) == row['targets']
            values.append([float(.5 * (F.cross_entropy(logits[old], row['targets'][old]) +
                                      F.cross_entropy(logits[~old], row['targets'][~old]))),
                           100 * correct[old].float().mean().item(),
                           100 * correct[~old].float().mean().item()])
    import numpy as np
    means = np.asarray(values).mean(0)
    return dict(zip(('mean_balanced_CE', 'base_accuracy', 'novel_accuracy'), means.tolist()))


def diagnose(arrays, metadata, report, cfg, args, output):
    import numpy as np
    import torch
    from dataclasses import replace
    from torch.nn import functional as F
    from engine.safd_metric_training import (TrainingOptions, reserve_selection_split,
        validation_records, training_records, _fit_scatter, train_steps, validate_options)
    from models.safd_metric import (SAFDMetricAdapter, metric_from_scatter, build_response_basis)
    started = time.perf_counter()
    options = TrainingOptions(**report['options'])
    validate_options(options)
    if arrays['sample_id'].tolist() != report['selection_protocol']['sample_ids']:
        raise ValueError('Cache row order/sample IDs differ from the training report.')
    features = torch.as_tensor(arrays['features']).float()
    labels = torch.as_tensor(arrays['labels']).long()
    cfg, protocol, pool = reserve_selection_split(cfg, labels, arrays['sample_id'], options)
    if protocol != report['selection_protocol']:
        raise ValueError('Class/query protocol does not replay exactly; check cache/config/report pairing.')
    x = F.normalize(features[:, 0], dim=-1)
    state = _fit_scatter(x, features, labels, protocol['fitting_indices'], options)
    records = validation_records(cfg, x.to(args.device), arrays, protocol)
    for row in records:
        seq = protocol['sequences'][row['sequence']]
        count = row['reference_votes'].shape[1]
        indices = [i for rows in seq['query_indices'][:count] for i in rows]
        row['class_ids'] = seq['class_ids'][:count]
        row['query_sample_ids'] = arrays['sample_id'][indices].tolist()
        global_targets = torch.tensor(row['class_ids'])[row['targets'].cpu()]
        if not torch.equal(global_targets, labels[indices]):
            raise ValueError('Local/global validation label order is inconsistent.')
    zero, _, _ = score_sweep(records, torch.eye(x.shape[1]), [0.])
    metrics = ('base_accuracy', 'novel_accuracy', 'harmonic_accuracy')
    drift = {k: zero[0][k] - report['reference'][k] for k in metrics}
    checks = {'protocol_exact': True, 'local_global_query_labels_consistent': True,
              'base_only': True, 'benchmark_test_evaluation': False,
              'cache_complete_base_bank': metadata.get('complete_base_bank'),
              'reference_replay': {k: zero[0][k] for k in metrics},
              'reference_recorded': report['reference'], 'reference_delta_pp': drift,
              'reference_within_tolerance': max(abs(v) for v in drift.values()) <= args.reference_tolerance_pp,
              'basis_mode': options.basis, 'fitting_scatter_diagnostics': state['diagnostics'],
              'recorded_scatter_diagnostics': report['selection_basis_diagnostics'],
              'training_gain': options.gain,
              'gain_sweep_scope': 'posthoc_validation_scoring_not_retraining_at_each_gain'}
    write_json(output / 'checks.json', checks)
    if not checks['reference_within_tolerance']:
        raise ValueError('Reference replay differs from recorded values; inspect checks.json before further diagnostics.')
    print('Reference replay and exact A/V protocol: passed', flush=True)
    rows, stages, spectra, train_log, probes = [], {}, {}, [], {}

    def snapshot(key, matrix, extra):
        matrix = matrix.detach().float()  # Same rounding as exported models.
        spectra[key] = metric_diagnostics(matrix)
        summary, stage_details, probe = score_sweep(records, matrix, args.gains)
        zero_h = zero[0]['harmonic_accuracy']
        for row in summary:
            row.update(candidate=key, **extra)
            row['H_delta_pp'] = row['harmonic_accuracy'] - zero_h
            row['feasible'] = (row['base_accuracy'] >= zero[0]['base_accuracy'] - options.maximum_group_drop_pp - 1e-12
                               and row['novel_accuracy'] >= zero[0]['novel_accuracy'] - options.maximum_group_drop_pp - 1e-12)
            print(f"{key} gain={row['gain']:g}: B={row['base_accuracy']:.4f} N={row['novel_accuracy']:.4f} "
                  f"H={row['harmonic_accuracy']:.4f} C={row['corrected']} D={row['damaged']}", flush=True)
        rows.extend(summary)
        stages[key] = stage_details
        if args.save_score_probe:
            probes[key + '_metric'] = matrix.cpu().numpy()
            for k, v in probe.items():
                probes[key + '_' + k] = v
        # Keep useful partial results if a long diagnostic is interrupted.
        write_json(output / 'candidate_summaries.json', rows)
        write_json(output / 'matrix_diagnostics.json', spectra)

    if args.mode == 'closed_form':
        for strength in args.strengths:
            snapshot('strength_' + format(strength, '.8g'), metric_from_scatter(state, strength), {'strength': strength})
    else:
        if report['options']['method'] != 'learned':
            raise ValueError('Learned replay requires a learned training report.')
        if max(args.checkpoints) > options.steps:
            raise ValueError('Diagnostic checkpoints exceed the original training budget.')
        train, trace = training_records(cfg, x.to(args.device), arrays, pool, options)
        if trace != report['selection_training_episodes']:
            raise ValueError('Training episode identities do not match the recorded run.')
        checks['training_episode_replay_exact'] = True
        write_json(output / 'checks.json', checks)
        adapter = SAFDMetricAdapter(build_response_basis(state, options.rank)).to(args.device)

        def capture(step, fitted, loss=None):
            if step not in args.checkpoints:
                return
            train_log.append({'step': step, 'last_episode_loss': loss,
                              **training_metrics(fitted, train, options.gain),
                              'theta_l2': float(fitted.theta.detach().norm()),
                              'coefficient_min': float(fitted.coefficients().min()) if fitted.theta.numel() else 0.,
                              'coefficient_max': float(fitted.coefficients().max()) if fitted.theta.numel() else 0.})
            snapshot('step_' + str(step), fitted.to_metric(), {'step': step})
        capture(0, adapter)
        # One uninterrupted optimizer trajectory, retaining the original episode order.
        replay = replace(options, validate_every=1)
        train_steps(adapter, train, replay, max(args.checkpoints), capture)
    write_json(output / 'stage_details.json', stages)
    write_json(output / 'training_curve.json', train_log)
    if probes:
        np.savez_compressed(output / 'score_probe.npz', **probes)
    comparisons = []
    for candidate in rows:
        if candidate['gain'] != options.gain:
            continue
        match = next((old for old in report['checkpoints'] if
            (args.mode == 'closed_form' and old.get('strength') == candidate.get('strength')) or
            (args.mode == 'learned' and old.get('step') == candidate.get('step'))), None)
        if match is not None:
            comparisons.append({'candidate': candidate['candidate'],
                'delta_pp': {k: candidate[k] - match[k] for k in metrics},
                'within_tolerance': max(abs(candidate[k] - match[k]) for k in metrics) <= args.reference_tolerance_pp})
    replay_ok = all(c['within_tolerance'] for c in comparisons) if comparisons else None
    result = {'status': 'completed', 'mode': args.mode, 'basis_mode': options.basis,
              'scope': 'base_training_class_disjoint_validation_only',
              'benchmark_test_evaluation': False, 'new_CLIP_or_SAFD_encodings': 0,
              'frozen_model_exported': False, 'checks': checks, 'candidates': rows,
              'recorded_checkpoint_replay': comparisons,
              'all_compared_checkpoints_within_tolerance': replay_ok,
              'training_curve': train_log, 'wall_seconds': time.perf_counter() - started,
              'limitations': ['Stage-query observations are repeated and not independent replicates.',
                              'Gain sweeps diagnose scoring scale; gains do not alter the training trajectory.',
                              'No model is selected or exported by this diagnostic.']}
    write_json(output / 'summary.json', result)
    return result


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError('Choose a fresh --output-dir; existing diagnostic directories are not overwritten.')
    import yaml
    import numpy as np
    import torch
    from yacs.config import CfgNode
    from tools.cache_safd_base_features import load_cache
    torch.set_num_threads(args.threads)
    arrays, metadata = load_cache(args.cache)
    if not metadata.get('complete_base_bank') or not metadata.get('has_text_features'):
        raise ValueError('Use the complete original base cache with ordinary BiMC text features.')
    report_path = Path(args.training_report)
    report = json.loads(report_path.read_text(encoding='utf-8'))
    model_sidecar = report_path.with_name(report_path.name.replace('.training.json', '.json'))
    if model_sidecar != report_path and model_sidecar.is_file():
        saved = json.loads(model_sidecar.read_text(encoding='utf-8'))
        if saved.get('cache_sha256') != metadata['cache_sha256']:
            raise ValueError('Training model sidecar names a different feature cache.')
    cfg = CfgNode(yaml.safe_load(metadata['config_yaml']))
    cfg.freeze()
    output.mkdir(parents=True)
    provenance = {'arguments': vars(args), 'cache_sha256': metadata['cache_sha256'],
                  'runtime': {'torch': torch.__version__, 'numpy': np.__version__,
                              'python': sys.version, 'torch_cuda': torch.version.cuda},
                  'cache_metadata_payload_sha256': metadata['metadata_payload_sha256'],
                  'training_report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
                  'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                    ('tools/diagnose_safd_metric.py', 'engine/safd_metric_training.py', 'models/safd_metric.py', 'models/bimc.py')}}
    write_json(output / 'provenance.json', provenance)
    try:
        diagnose(arrays, metadata, report, cfg, args, output)
    except Exception as exc:
        write_json(output / 'failure.json', {'status': 'failed', 'error': str(exc)})
        raise
    print('Diagnostic results:', output / 'summary.json', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
