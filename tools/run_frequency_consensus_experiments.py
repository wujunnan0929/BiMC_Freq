"""Paired consensus validation on Ubuntu/Windows; dry-run is the default.

No benchmark labels enter calibration or variant selection. Existing runs are
reused only with matching source/config/command and support-manifest hashes.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from tools import run_incremental_residual_experiments as common

C = 'TRAINER.BiMC.CONSENSUS.'
F = 'TRAINER.BiMC.FREQUENCY.'
R = 'TRAINER.BiMC.RESIDUAL.'
MANAGED = {'SEED', 'OUTPUT_DIR', 'DATASET.SUPPORT_SEED', 'DATASET.SUPPORT_MANIFEST',
           C+'ENABLED', C+'MODE', C+'VIEW_CONTROL', C+'SEMANTIC_PERMUTATION',
           C+'AUTO_CALIBRATE', C+'LAMBDA', C+'CALIBRATION_SEED',
           C+'OBJECTIVE', C+'MAX_GROUP_DROP_PP',
           F+'ENABLED', F+'ROUTER.ENABLED', R+'ENABLED'}


def variants_for_suite(suite, fixed_lambda=None):
    common_opts = {R+'ENABLED': False, F+'ENABLED': False, F+'ROUTER.ENABLED': False,
                   C+'ENABLED': True, C+'MODE': 'consensus', C+'VIEW_CONTROL': 'frequency',
                   C+'SEMANTIC_PERMUTATION': [0, 1, 2],
                   C+'OBJECTIVE': 'micro_all', C+'MAX_GROUP_DROP_PP': -1.0,
                   C+'AUTO_CALIBRATE': fixed_lambda is None,
                   C+'LAMBDA': float(fixed_lambda or 0.)}
    definitions = [
        ('baseline', {C+'ENABLED': False}),
        ('frequency_v2', {C+'ENABLED': False, F+'ENABLED': True}),
        ('router', {C+'ENABLED': False, F+'ENABLED': True, F+'ROUTER.ENABLED': True}),
        ('zero', {C+'AUTO_CALIBRATE': False, C+'LAMBDA': 0.}),
        ('visual', {C+'MODE': 'visual'}),
        ('semantic', {C+'MODE': 'semantic'}),
        ('average', {C+'MODE': 'average'}),
        ('consensus', {}),
    ]
    if suite == 'calibration':
        if fixed_lambda is not None:
            raise ValueError('The calibration suite must select lambda, not fix it.')
        definitions = [item for item in definitions if item[0] in
                       ('baseline', 'router', 'zero', 'consensus')]
        definitions.extend([
            ('consensus_inc_micro', {C+'OBJECTIVE': 'micro_incremental'}),
            ('consensus_balanced', {C+'OBJECTIVE': 'balanced_incremental'}),
            ('consensus_safe', {C+'OBJECTIVE': 'balanced_incremental',
                                C+'MAX_GROUP_DROP_PP': 0.0}),
        ])
    elif suite == 'all':
        definitions.extend([
            ('shuffle_120', {C+'SEMANTIC_PERMUTATION': [1, 2, 0]}),
            ('shuffle_201', {C+'SEMANTIC_PERMUTATION': [2, 0, 1]}),
            ('original_views', {C+'VIEW_CONTROL': 'original'}),
            ('ordinary_views', {C+'VIEW_CONTROL': 'augmentation'}),
        ])
    elif suite != 'core':
        raise ValueError('Unknown suite: ' + suite)
    return [(name, {**common_opts, **changes}) for name, changes in definitions]


def read_base_lambda(args, seed, data_cfg, configs, source_hash):
    """Read only a base calibration artifact, never a benchmark accuracy."""
    directory = (Path(args.lambda_from_root).resolve() / data_cfg.stem
                 / args.lambda_from_variant / f'seed_{seed}')
    path = directory / 'consensus_calibration.json'
    report_bytes = path.read_bytes()
    report = json.loads(report_bytes.decode('utf-8'))
    spec = json.loads((directory / 'run_spec.json').read_text(encoding='utf-8'))
    if not isinstance(report, dict) or not isinstance(spec, dict):
        raise ValueError('Lambda source report and run_spec must be JSON objects.')
    if (report.get('data_source') != 'base_training_only'
            or report.get('seed') != seed + 9101
            or report.get('run_seed', seed) != seed or not report.get('auto_calibrate')):
        raise ValueError('Lambda source must be an auto-calibrated base-only report for this seed: ' + str(path))
    if spec.get('config_hashes') != configs or spec.get('source_hash') != source_hash:
        raise ValueError('Lambda source config/source differs; keep phase A/B code and configs unchanged.')
    if not isinstance(spec.get('command'), list) or '--opts' not in spec['command']:
        raise ValueError('Lambda source needs a valid run_spec command.')
    source_opts = spec['command'][spec['command'].index('--opts') + 1:]
    source_opts = dict(zip(source_opts[::2], source_opts[1::2]))
    current_opts = dict(zip(args.opts[::2], args.opts[1::2]))
    # Reject asymmetric user overrides (dataset root, shot, precision, scales,
    # episodic protocol, descriptions, etc.), even if only one side sets a key.
    for key in (set(source_opts) | set(current_opts)) - MANAGED:
        expected = current_opts.get(key)
        if key == 'DEVICE.GPU_ID' and expected is not None:
            expected = repr(expected)
        if source_opts.get(key) != expected:
            raise ValueError('Lambda source override differs for ' + key)
    value = report.get('selected_lambda')
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('Invalid base-selected lambda: ' + str(path))
    return float(value), {'path': str(path), 'sha256': hashlib.sha256(report_bytes).hexdigest(),
                          'selected_lambda': float(value), 'protocol_sha256': report.get('protocol_sha256'),
                          'selection_objective': report.get('selection_objective', 'micro_all')}


def build_plan(args):
    if len(args.opts) % 2 or len(set(args.opts[::2])) != len(args.opts[::2]):
        raise ValueError('--opts needs distinct KEY VALUE pairs.')
    collision = set(args.opts[::2]) & MANAGED
    if collision:
        raise ValueError('Matrix-managed keys cannot be overridden: ' + ', '.join(sorted(collision)))
    if not args.seeds or min(args.seeds) < 1 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Use distinct positive seeds (legacy SEED=0 is nondeterministic).')
    if args.fixed_lambda is not None and (not math.isfinite(args.fixed_lambda) or args.fixed_lambda < 0):
        raise ValueError('--fixed-lambda must be finite and nonnegative, chosen using base data only.')
    if args.lambda_from_root and args.suite == 'calibration':
        raise ValueError('--lambda-from-root is for fixed-strength controls, not the calibration suite.')
    data_cfg, train_cfg = Path(args.data_cfg).resolve(), Path(args.train_cfg).resolve()
    configs = {}
    for path in (data_cfg, train_cfg):
        configs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    options = dict(zip(args.opts[::2], args.opts[1::2]))
    command_opts = []
    for key, value in options.items():
        # YACS otherwise interprets a bare GPU_ID 0 as int, although the field
        # is a string. Pass literal quotes through subprocess argv, not a shell.
        command_opts.extend([key, repr(value) if key == 'DEVICE.GPU_ID' else value])
    auxiliary_files = list((REPO_ROOT / 'description').glob('*.json'))
    auxiliary_files.extend(Path(value).resolve() for key, value in options.items()
                           if key.endswith('PATH') and value and Path(value).is_file())
    auxiliary_hash = common.sha256_json({str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                         for path in auxiliary_files})
    source_hash = common.sha256_json({
        'implementation': common.source_digest(REPO_ROOT),
        'launcher': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'descriptions': auxiliary_hash,
    })
    support_id = common.sha256_json({
        'dataset_config': configs[str(data_cfg)],
        'dataset_options': {key: value for key, value in options.items() if key.startswith('DATASET.')},
    })[:12]
    matrix = variants_for_suite(args.suite, args.fixed_lambda)
    selected = set(args.variants or [name for name, _ in matrix])
    if selected - {name for name, _ in matrix}:
        raise ValueError('Requested variant is not part of this suite.')
    root = Path(args.output_root).resolve()
    support_root = Path(args.support_root).resolve() if args.support_root else root / 'support'
    plan = []
    for seed in args.seeds:
        support = support_root / (data_cfg.stem + '_' + support_id) / f'seed_{seed}.json'
        strength, provenance = args.fixed_lambda, None
        if args.lambda_from_root:
            strength, provenance = read_base_lambda(args, seed, data_cfg, configs, source_hash)
        for name, overrides in matrix:
            if name not in selected:
                continue
            output = root / data_cfg.stem / name / f'seed_{seed}'
            overrides = {**overrides, C+'CALIBRATION_SEED': seed + 9101}
            if args.lambda_from_root:
                overrides.update({C+'AUTO_CALIBRATE': False,
                                  C+'LAMBDA': 0.0 if name == 'zero' else strength})
            command = [args.python, '-u', str(REPO_ROOT / 'main.py'),
                       '--data_cfg', str(data_cfg), '--train_cfg', str(train_cfg), '--opts',
                       *command_opts, 'SEED', str(seed), 'DATASET.SUPPORT_SEED', str(seed),
                       'DATASET.SUPPORT_MANIFEST', str(support), 'OUTPUT_DIR', str(output)]
            for key, value in overrides.items():
                command.extend([key, str(value)])
            fingerprint_data = {'command': command, 'configs': configs, 'source': source_hash}
            if provenance:
                fingerprint_data['lambda_provenance'] = provenance
            fingerprint = common.sha256_json(fingerprint_data)
            plan.append({'variant': name, 'seed': seed, 'support_seed': seed,
                         'support_manifest': str(support), 'output_dir': str(output),
                         'command': command, 'cwd': str(REPO_ROOT), 'fingerprint': fingerprint,
                         'config_hashes': configs, 'source_hash': source_hash,
                         'lambda_provenance': provenance})
    return plan


def save_summary(path, args, rows, planned_runs):
    enriched = []
    for original in rows:
        row = dict(original)
        if row['status'] in ('completed', 'skipped'):
            metrics = common.load_completed_metrics(Path(row['output_dir']) / 'metrics.json')
            if metrics is not None:
                last = metrics['sessions'][-1]
                calibration = metrics.get('consensus')
                row.update(consensus_calibration={
                    key: calibration[key] for key in (
                        'selected_lambda', 'scales', 'active_sources', 'protocol_sha256',
                        'fit_class_ids', 'validation_class_ids', 'candidate_class_counts',
                        'selection_objective', 'max_group_drop_pp', 'selection_audit',
                        'balanced_safe_lambda', 'candidate_results',
                        'mode', 'view_control', 'semantic_permutation',
                    ) if key in calibration
                } if calibration else None,
                           prediction_diagnostics=last.get('prediction_diagnostics'),
                           query_auxiliary_encodings=last.get('query_auxiliary_encodings'),
                           eval_seconds=last.get('eval_seconds'),
                           retained_tensor_bytes=last.get('retained_tensor_bytes'),
                           exact_reference_all_sessions=all(
                               item.get('reference_scores_exact_equal') is True for item in metrics['sessions']))
        enriched.append(row)
    pairing_audit = audit_pairing(enriched)
    aggregates, paired = common.aggregate_results(enriched)
    # In addition to baseline, compare the proposed method directly to strong alternatives.
    comparisons = {}
    metric_comparisons = {}
    successful = [row for row in enriched if row['status'] in ('completed', 'skipped')]
    for reference_name in ('router', 'average', 'consensus', 'consensus_balanced'):
        reference = {row['seed']: row for row in successful if row['variant'] == reference_name}
        comparisons[reference_name] = {}
        metric_comparisons[reference_name] = {}
        for name in sorted({row['variant'] for row in successful}):
            differences = [row['summary']['final_accuracy'] - reference[row['seed']]['summary']['final_accuracy']
                           for row in successful if row['variant'] == name and row['seed'] in reference]
            comparisons[reference_name][name] = common.metric_statistics(differences)
            metric_comparisons[reference_name][name] = {
                key: common.metric_statistics([
                    row['summary'][key] - reference[row['seed']]['summary'][key]
                    for row in successful if row['variant'] == name and row['seed'] in reference
                    and row['summary'].get(key) is not None
                    and reference[row['seed']]['summary'].get(key) is not None
                ]) for key in common.METRIC_NAMES
            }
    common.write_json(path, {
        'schema_version': 2, 'generated_at': common.utc_now(), 'suite': args.suite,
        'seeds': args.seeds, 'metrics_unit': 'percent; paired deltas in percentage points',
        'planned_runs': planned_runs,
        'completed_runs': sum(row['status'] in ('completed', 'skipped') for row in enriched),
        'failed_runs': sum(row['status'] == 'failed' for row in enriched),
        'runs': enriched, 'aggregates': aggregates, 'paired_delta_vs_baseline': paired,
        'pairing_audit': pairing_audit,
        'paired_final_delta_vs_reference': comparisons,
        'paired_delta_vs_reference': metric_comparisons,
        'selection_policy': f'Per-seed base lambda from {args.lambda_from_root}/{args.lambda_from_variant}'
                            if args.lambda_from_root else
                            'Base-only per-variant calibration' if args.fixed_lambda is None
                            else f'User-fixed base-selected lambda={args.fixed_lambda}',
        'warning': 'No test-set variant selection is performed. Inspect paired variation and mechanisms; '
                   'a positive single-seed result does not establish an improvement.',
    })


def audit_pairing(rows):
    """Verify same support and saved test identities; never use labels to select a method."""
    result = {}
    successful = [row for row in rows if row['status'] in ('completed', 'skipped')]
    for seed in sorted({row['seed'] for row in successful}):
        group = [row for row in successful if row['seed'] == seed]
        support_hashes = set()
        calibration_hashes = set()
        scale_hashes = {}
        for row in group:
            path = Path(row['output_dir']) / 'support.json'
            if path.is_file():
                support_hashes.add(hashlib.sha256(path.read_bytes()).hexdigest())
            report = row.get('consensus_calibration')
            if report:
                calibration_hashes.add(report['protocol_sha256'])
                if 'scales' in report and 'view_control' in report:
                    control = str((report['view_control'], report.get('semantic_permutation')))
                    scale_hashes.setdefault(control, set()).add(common.sha256_json(report['scales']))
        report = {'support_manifest_match': len(support_hashes) == 1,
                  'calibration_protocol_match': len(calibration_hashes) <= 1,
                  'calibration_scales_match_within_view': {
                      name: len(hashes) == 1 for name, hashes in scale_hashes.items()
                  },
                  'sample_comparisons': {}}
        baseline = next((row for row in group if row['variant'] == 'baseline'), None)
        if baseline is not None:
            import numpy as np  # Needed only after execution; planning stays standard-library only.
            reference_files = sorted(Path(baseline['output_dir']).glob('predictions_session_*.npz'))
            for row in group:
                if row is baseline:
                    continue
                check = {'sessions_checked': 0, 'sample_alignment': True,
                         'reference_prediction_mismatches': 0, 'output_prediction_mismatches': 0}
                for reference_path in reference_files:
                    candidate_path = Path(row['output_dir']) / reference_path.name
                    if not candidate_path.is_file():
                        continue
                    with np.load(reference_path, allow_pickle=False) as reference, \
                            np.load(candidate_path, allow_pickle=False) as candidate:
                        aligned = (np.array_equal(reference['sample_id'], candidate['sample_id'])
                                   and np.array_equal(reference['target'], candidate['target']))
                        check['sessions_checked'] += 1
                        check['sample_alignment'] &= aligned
                        if aligned:
                            check['reference_prediction_mismatches'] += int(np.count_nonzero(
                                reference['prediction'] != candidate['reference_prediction']))
                            check['output_prediction_mismatches'] += int(np.count_nonzero(
                                reference['prediction'] != candidate['prediction']))
                if not check['sessions_checked']:
                    check['sample_alignment'] = None
                report['sample_comparisons'][row['variant']] = check
        result[str(seed)] = report
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-cfg', default=str(REPO_ROOT / 'configs/datasets/cub200.yaml'))
    parser.add_argument('--train-cfg', default=str(REPO_ROOT / 'configs/trainers/bimc_frequency_consensus.yaml'))
    parser.add_argument('--output-root', default=str(REPO_ROOT / 'outputs/frequency_consensus'))
    parser.add_argument('--support-root', help='Shared manifest root across output directories (default: OUTPUT/support).')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--suite', choices=('core', 'all', 'calibration'), default='core')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--variants', nargs='+')
    strength = parser.add_mutually_exclusive_group()
    strength.add_argument('--fixed-lambda', type=float, help='Base-selected value only; applies to nonzero rerank variants.')
    strength.add_argument('--lambda-from-root', help='Read each seed\'s base-selected lambda from this experiment root.')
    parser.add_argument('--lambda-from-variant', default='consensus_balanced')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--execute', action='store_true')
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--rerun', action='store_true')
    parser.add_argument('--keep-going', action='store_true')
    parser.add_argument('--opts', nargs=argparse.REMAINDER, default=[])
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
            row = {key: run[key] for key in ('variant', 'seed', 'support_seed', 'output_dir',
                                           'support_manifest', 'fingerprint')}
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
