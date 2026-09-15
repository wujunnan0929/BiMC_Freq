"""Plan or run paired frequency-uncertainty experiments; dry-run is the default.

Each nonzero branch selects its own parameters from the same base-only episode
protocol. Benchmark metrics are collected only after selection has finished.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from tools import run_incremental_residual_experiments as common

U = 'TRAINER.BiMC.UNCERTAINTY.'
F = 'TRAINER.BiMC.FREQUENCY.'
C = 'TRAINER.BiMC.CONSENSUS.'
R = 'TRAINER.BiMC.RESIDUAL.'
MANAGED = {
    'SEED', 'OUTPUT_DIR', 'DATASET.SUPPORT_SEED', 'DATASET.SUPPORT_MANIFEST',
    U+'ENABLED', U+'VIEW_CONTROL', U+'COVARIANCE', U+'AUTO_CALIBRATE',
    U+'ALPHA', U+'CALIBRATION_SEED', U+'BASE_ONLY',
    F+'ENABLED', F+'ROUTER.ENABLED', C+'ENABLED', R+'ENABLED', R+'BASE_ONLY',
}


def variants_for_suite(suite):
    options = {
        U+'ENABLED': True, U+'VIEW_CONTROL': 'frequency', U+'COVARIANCE': 'shrinkage',
        U+'AUTO_CALIBRATE': True, U+'ALPHA': 0.0, U+'BASE_ONLY': False,
        F+'ENABLED': False, F+'ROUTER.ENABLED': False,
        C+'ENABLED': False, R+'ENABLED': False, R+'BASE_ONLY': False,
    }
    definitions = [
        ('baseline', {U+'ENABLED': False}),
        ('zero', {U+'AUTO_CALIBRATE': False, U+'ALPHA': 0.0}),
        ('original_shared', {U+'VIEW_CONTROL': 'original', U+'COVARIANCE': 'shared'}),
        ('original_shrinkage', {U+'VIEW_CONTROL': 'original'}),
        ('frequency_shared', {U+'COVARIANCE': 'shared'}),
        ('frequency_shrinkage', {}),
    ]
    if suite == 'all':
        definitions.append(('router', {
            U+'ENABLED': False, F+'ENABLED': True, F+'ROUTER.ENABLED': True,
        }))
    elif suite != 'core':
        raise ValueError('Unknown suite: ' + suite)
    return [(name, {**options, **overrides}) for name, overrides in definitions]


def build_plan(args):
    if len(args.opts) % 2 or len(set(args.opts[::2])) != len(args.opts[::2]):
        raise ValueError('--opts requires distinct KEY VALUE pairs and must be last.')
    collision = set(args.opts[::2]) & MANAGED
    if collision:
        raise ValueError('Matrix-managed keys cannot be overridden: ' + ', '.join(sorted(collision))
                         + '. For base-only calibration, call main.py directly.')
    if not args.seeds or min(args.seeds) < 1 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Use distinct positive seeds; SEED=0 is nondeterministic.')
    if args.variants is not None and len(set(args.variants)) != len(args.variants):
        raise ValueError('--variants must contain distinct names.')
    data_cfg, train_cfg = Path(args.data_cfg).resolve(), Path(args.train_cfg).resolve()
    configs = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in (data_cfg, train_cfg)}
    options = dict(zip(args.opts[::2], args.opts[1::2]))
    command_opts = []
    for key, value in options.items():
        # Preserve a literal string through YACS decoding and subprocess argv.
        command_opts.extend([key, repr(value) if key == 'DEVICE.GPU_ID' else value])
    auxiliary_files = list((REPO_ROOT / 'description').glob('*.json'))
    auxiliary_files.extend(Path(value).resolve() for key, value in options.items()
                           if key.endswith('PATH') and value and Path(value).is_file())
    source_hash = common.sha256_json({
        'implementation': common.source_digest(REPO_ROOT),
        'launcher': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'auxiliary_files': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in auxiliary_files},
    })
    # Same identity as the consensus runner, allowing explicitly shared support roots.
    support_id = common.sha256_json({
        'dataset_config': configs[str(data_cfg)],
        'dataset_options': {key: value for key, value in options.items() if key.startswith('DATASET.')},
    })[:12]
    matrix = variants_for_suite(args.suite)
    selected = set(args.variants or [name for name, _ in matrix])
    unknown = selected - {name for name, _ in matrix}
    if unknown:
        raise ValueError('Variants not in this suite: ' + ', '.join(sorted(unknown)))
    root = Path(args.output_root).resolve()
    support_root = Path(args.support_root).resolve() if args.support_root else root / 'support'
    plan = []
    for seed in args.seeds:
        support = support_root / (data_cfg.stem + '_' + support_id) / f'seed_{seed}.json'
        for name, overrides in matrix:
            if name not in selected:
                continue
            output = root / data_cfg.stem / name / f'seed_{seed}'
            overrides = {**overrides, U+'CALIBRATION_SEED': seed + 10101}
            command = [args.python, '-u', str(REPO_ROOT / 'main.py'),
                       '--data_cfg', str(data_cfg), '--train_cfg', str(train_cfg), '--opts',
                       *command_opts, 'SEED', str(seed), 'DATASET.SUPPORT_SEED', str(seed),
                       'DATASET.SUPPORT_MANIFEST', str(support), 'OUTPUT_DIR', str(output)]
            for key, value in overrides.items():
                command.extend([key, str(value)])
            fingerprint = common.sha256_json({'command': command, 'configs': configs, 'source': source_hash})
            plan.append({'variant': name, 'seed': seed, 'support_seed': seed,
                         'support_manifest': str(support), 'output_dir': str(output),
                         'command': command, 'cwd': str(REPO_ROOT), 'fingerprint': fingerprint,
                         'config_hashes': configs, 'source_hash': source_hash})
    return plan


def audit_pairing(rows):
    """Audit saved support, calibration episodes and predictions without selecting parameters."""
    result = {}
    successful = [row for row in rows if row['status'] in ('completed', 'skipped')]
    for seed in sorted({row['seed'] for row in successful}):
        group = [row for row in successful if row['seed'] == seed]
        support_hashes = {}
        protocols = {}
        calibration_errors = {}
        for row in group:
            name = row['variant']
            support_path = Path(row['output_dir']) / 'support.json'
            if support_path.is_file():
                support_hashes[name] = hashlib.sha256(support_path.read_bytes()).hexdigest()
            report = row.get('uncertainty_calibration')
            if report is not None:
                errors = []
                if report.get('data_source') != 'base_training_only':
                    errors.append('Calibration source is not certified base_training_only.')
                if report.get('seed') != seed + 10101:
                    errors.append('Calibration seed does not match the paired plan.')
                if report.get('run_seed', seed) != seed:
                    errors.append('Run seed does not match the paired plan.')
                fit, validation = report.get('fit_class_ids'), report.get('validation_class_ids')
                if fit is not None and validation is not None and set(fit) & set(validation):
                    errors.append('Fit and validation classes overlap.')
                if name != 'zero' and report.get('auto_calibrate') is not True:
                    errors.append('Nonzero matrix controls require automatic base calibration.')
                if report.get('auto_calibrate') is True:
                    # The manual zero control deliberately samples no episodes.
                    # Its protocol cannot be compared to automatic calibration.
                    protocols[name] = report.get('protocol_sha256')
                    if not protocols[name]:
                        errors.append('Automatic calibration is missing its protocol hash.')
                if errors:
                    calibration_errors[name] = errors
            elif name not in ('baseline', 'router'):
                calibration_errors[name] = ['Missing uncertainty calibration artifact.']
        report = {
            'support_manifest_match': len(support_hashes) == len(group) and len(set(support_hashes.values())) == 1,
            'support_artifacts_found': len(support_hashes),
            'calibration_protocols': protocols,
            'calibration_protocol_match': (all(protocols.values()) and len(set(protocols.values())) == 1)
                                          if protocols else None,
            'calibration_errors': calibration_errors,
            'sample_comparisons': {},
        }
        baseline = next((row for row in group if row['variant'] == 'baseline'), None)
        if baseline is not None:
            reference_paths = {path.name: path for path in Path(baseline['output_dir']).glob('predictions_session_*.npz')}
            for row in group:
                if row is baseline:
                    continue
                candidate_paths = {path.name: path for path in Path(row['output_dir']).glob('predictions_session_*.npz')}
                common_names = sorted(reference_paths.keys() & candidate_paths.keys())
                check = {'sessions_checked': len(common_names),
                         'complete_session_set': bool(reference_paths) and reference_paths.keys() == candidate_paths.keys(),
                         'missing_sessions': sorted(reference_paths.keys() - candidate_paths.keys()),
                         'sample_alignment': True if common_names else None,
                         'reference_prediction_mismatches': 0, 'output_prediction_mismatches': 0}
                if common_names:
                    import numpy as np  # Planning and summary without predictions stay standard-library only.
                    for name in common_names:
                        with np.load(reference_paths[name], allow_pickle=False) as reference, \
                                np.load(candidate_paths[name], allow_pickle=False) as candidate:
                            aligned = (np.array_equal(reference['sample_id'], candidate['sample_id'])
                                       and np.array_equal(reference['target'], candidate['target']))
                            check['sample_alignment'] &= aligned
                            if aligned:
                                check['reference_prediction_mismatches'] += int(np.count_nonzero(
                                    reference['prediction'] != candidate['reference_prediction']))
                                check['output_prediction_mismatches'] += int(np.count_nonzero(
                                    reference['prediction'] != candidate['prediction']))
                report['sample_comparisons'][row['variant']] = check
        result[str(seed)] = report
    return result


def save_summary(path, args, rows, planned_runs):
    enriched = []
    for original in rows:
        row = dict(original)
        if row['status'] in ('completed', 'skipped'):
            directory = Path(row['output_dir'])
            metrics = common.load_completed_metrics(directory / 'metrics.json')
            calibration_path = directory / 'uncertainty_calibration.json'
            calibration = common.read_json(calibration_path)
            if isinstance(calibration, dict):
                row['uncertainty_calibration'] = calibration
                row['uncertainty_calibration_sha256'] = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
            else:
                row['uncertainty_calibration'] = None
            if metrics is not None:
                last = metrics['sessions'][-1]
                for key in ('prediction_diagnostics', 'query_auxiliary_encodings', 'eval_seconds', 'retained_tensor_bytes'):
                    row[key] = last.get(key)
                row['exact_reference_all_sessions'] = all(
                    item.get('reference_scores_exact_equal') is True for item in metrics['sessions'])
        enriched.append(row)
    aggregates, paired = common.aggregate_results(enriched)
    successful = [row for row in enriched if row['status'] in ('completed', 'skipped')]
    comparisons = {}
    for reference_name in ('original_shared', 'original_shrinkage', 'frequency_shared', 'router'):
        reference = {row['seed']: row for row in successful if row['variant'] == reference_name}
        if not reference:
            continue
        comparisons[reference_name] = {
            name: {key: common.metric_statistics([
                row['summary'][key] - reference[row['seed']]['summary'][key]
                for row in successful if row['variant'] == name and row['seed'] in reference
                and row['summary'].get(key) is not None
                and reference[row['seed']]['summary'].get(key) is not None
            ]) for key in common.METRIC_NAMES}
            for name in sorted({row['variant'] for row in successful})
        }
    common.write_json(path, {
        'schema_version': 1, 'generated_at': common.utc_now(), 'suite': args.suite,
        'seeds': args.seeds, 'metrics_unit': 'percent; paired deltas in percentage points',
        'planned_runs': planned_runs,
        'completed_runs': sum(row['status'] in ('completed', 'skipped') for row in enriched),
        'failed_runs': sum(row['status'] == 'failed' for row in enriched),
        'runs': enriched, 'aggregates': aggregates, 'paired_delta_vs_baseline': paired,
        'paired_delta_vs_reference': comparisons, 'pairing_audit': audit_pairing(enriched),
        'selection_policy': 'Each nonzero uncertainty variant calibrates separately on the same base-only '
                            'class/episode protocol and parameter grids; redundant shared-prior candidates may be deduplicated.',
        'warning': 'No benchmark result selects hyperparameters or variants. Inspect pairing, base/novel tradeoffs, '
                   'and seed variation before making an accuracy claim.',
    })


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-cfg', default=str(REPO_ROOT / 'configs/datasets/cub200.yaml'))
    parser.add_argument('--train-cfg', default=str(REPO_ROOT / 'configs/trainers/bimc_frequency_uncertainty.yaml'))
    parser.add_argument('--output-root', default=str(REPO_ROOT / 'outputs/frequency_uncertainty'))
    parser.add_argument('--support-root', help='Shared support manifest root; default: OUTPUT/support.')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--suite', choices=('core', 'all'), default='core')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--variants', nargs='+')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--rerun', action='store_true')
    parser.add_argument('--keep-going', action='store_true')
    parser.add_argument('--opts', nargs=argparse.REMAINDER, default=[], help='Additional YACS KEY VALUE pairs; place last.')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        plan = build_plan(args)
    except (OSError, ValueError) as error:
        print('Invalid experiment plan: ' + str(error), file=sys.stderr)
        return 2
    if args.summarize_only:
        rows = []
        for run in plan:
            metrics = common.completed_result(run)
            row = {key: run[key] for key in ('variant', 'seed', 'support_seed', 'output_dir', 'support_manifest', 'fingerprint')}
            row.update(status='skipped', summary=metrics['summary']) if metrics else row.update(status='missing_or_stale')
            rows.append(row)
        path = Path(args.output_root).resolve() / 'summary.json'
        save_summary(path, args, rows, len(plan))
        print('Summary: ' + str(path))
        return 0
    if not args.execute:
        for run in plan:
            print(f"{run['variant']} seed={run['seed']}: " + common.describe_command(run['command']))
        print(f'Dry run: {len(plan)} commands; no files or training processes created.')
        return 0
    for run in plan:
        directory = Path(run['output_dir'])
        if directory.exists() and any(directory.iterdir()) and not (directory / 'run_spec.json').is_file():
            print('Refusing unmanaged nonempty output directory: ' + str(directory), file=sys.stderr)
            return 2
    return common.execute_plan(plan, args, summary_writer=save_summary)


if __name__ == '__main__':
    raise SystemExit(main())
