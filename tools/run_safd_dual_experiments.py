"""Paired frozen-CLIP dual-modal scale experiments; planning is the default."""

import argparse
import hashlib
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from tools import run_frequency_uncertainty_experiments as shared
from utils.frequency_descriptions import (
    DATASET_SPECS, config_scalar, dataset_class_names, dataset_slug,
    description_path, load_explicit_descriptions,
)

common = shared.common
F, U, D = shared.F, shared.U, 'TRAINER.BiMC.DUAL_FREQUENCY.'


def variants_for_suite(suite):
    base = {
        U+'ENABLED': True, U+'VIEW_CONTROL': 'original', U+'COVARIANCE': 'full_shared',
        U+'MEAN_UNCERTAINTY': False, U+'AUTO_CALIBRATE': True, U+'ALPHA': 0.0, U+'BASE_ONLY': False,
        F+'ENABLED': False, F+'BACKEND': 'safd', F+'VIEW_MODE': 'natural', F+'HIGH_ENHANCE': .25,
        F+'ROUTER.ENABLED': False, shared.C+'ENABLED': False, shared.R+'ENABLED': False,
        shared.R+'BASE_ONLY': False, F+'USE_EXPLICIT_DESCRIPTIONS': True,
        D+'ENABLED': True, D+'AUTO_CALIBRATE': True, D+'LAMBDA': 0.0, D+'BASE_ONLY': False,
        D+'MODE': 'consensus', D+'ALIGNMENT': True, D+'RESIDUALIZE': True,
        D+'SEMANTIC_SOURCE': 'explicit', D+'SEMANTIC_PERMUTATION': [0, 1, 2],
        D+'BAND_WEIGHTS': [1., 1., 1.],
    }
    definitions = [
        ('baseline', {U+'ENABLED': False, D+'ENABLED': False}),
        ('zero', {D+'ENABLED': False, U+'AUTO_CALIBRATE': False}),
        ('original_gda', {D+'ENABLED': False}),
        ('fft_gda', {D+'ENABLED': False, U+'VIEW_CONTROL': 'joint', F+'BACKEND': 'fft'}),
        ('safd_gda', {D+'ENABLED': False, U+'VIEW_CONTROL': 'joint'}),
        ('zero_dual', {D+'AUTO_CALIBRATE': False}),
        ('fft_dual', {F+'BACKEND': 'fft'}),
        ('safd_dual', {}),
    ]
    if suite == 'all':
        definitions.extend([
            ('visual_only', {D+'MODE': 'visual'}),
            ('semantic_only', {D+'MODE': 'semantic'}),
            ('average', {D+'MODE': 'average'}),
            ('no_alignment', {D+'ALIGNMENT': False}),
            ('no_residual', {D+'RESIDUALIZE': False}),
            ('ordinary_text', {D+'SEMANTIC_SOURCE': 'ordinary'}),
            ('structured_text', {D+'SEMANTIC_SOURCE': 'structured'}),
            ('permuted_text', {D+'SEMANTIC_PERMUTATION': [2, 0, 1]}),
            ('low_only', {D+'BAND_WEIGHTS': [1., 0., 0.]}),
            ('middle_only', {D+'BAND_WEIGHTS': [0., 1., 0.]}),
            ('high_only', {D+'BAND_WEIGHTS': [0., 0., 1.]}),
        ])
    elif suite != 'core':
        raise ValueError('Unknown dual suite: ' + suite)
    return [(name, {**base, **changes}) for name, changes in definitions]


def build_plan(args):
    matrix = variants_for_suite(args.suite)
    managed = shared.MANAGED | {key for _, values in matrix for key in values}
    collision = set(args.opts[::2]) & managed
    if collision:
        raise ValueError('Matrix-managed keys cannot be overridden: ' + ', '.join(sorted(collision)))
    return shared.build_plan(args, matrix=matrix, launcher_path=Path(__file__))


def validate_plan_inputs(args, plan):
    """Check full-dataset candidates before launching even the first baseline."""
    supplied = dict(zip(args.opts[::2], args.opts[1::2]))
    dataset = supplied.get('DATASET.NAME') or config_scalar(args.data_cfg, 'NAME')
    dataset = dataset_slug(dataset)
    train_path = config_scalar(args.train_cfg, 'EXPLICIT_DESCRIPTION_PATH', 'auto')
    path = description_path(dataset, supplied.get(F+'EXPLICIT_DESCRIPTION_PATH', train_path))
    required = []
    for run in plan:
        options = run['command'][run['command'].index('--opts') + 1:]
        options = dict(zip(options[::2], options[1::2]))
        if options.get(D+'ENABLED') == 'True' and options.get(D+'SEMANTIC_SOURCE') == 'explicit':
            required.append(run['variant'])
    if not required:
        return None
    candidates = load_explicit_descriptions(path, dataset_class_names(dataset), exact=True, dataset=dataset)
    return {'dataset': dataset, 'path': str(path), 'classes': len(candidates),
            'candidates_per_band': len(candidates[0]['low']), 'required_variants': sorted(set(required))}


def _compare_predictions(reference, candidate, candidate_key='prediction'):
    import numpy as np
    ref_paths = {p.name: p for p in Path(reference).glob('predictions_session_*.npz')}
    paths = {p.name: p for p in Path(candidate).glob('predictions_session_*.npz')}
    result = {'complete_session_set': bool(ref_paths) and ref_paths.keys() == paths.keys(),
              'sessions_checked': 0, 'sample_alignment': True, 'prediction_mismatches': 0}
    for name in sorted(ref_paths.keys() & paths.keys()):
        with np.load(ref_paths[name], allow_pickle=False) as ref, np.load(paths[name], allow_pickle=False) as run:
            aligned = np.array_equal(ref['sample_id'], run['sample_id']) and np.array_equal(ref['target'], run['target'])
            result['sample_alignment'] &= aligned
            result['sessions_checked'] += 1
            if aligned and candidate_key in run:
                result['prediction_mismatches'] += int(np.count_nonzero(ref['prediction'] != run[candidate_key]))
            else:
                result['sample_alignment'] = False
    return result


def save_summary(path, args, rows, planned_runs):
    shared.save_summary(path, args, rows, planned_runs)
    report = common.read_json(path)
    completed = [r for r in report['runs'] if r['status'] in ('completed', 'skipped')]
    dual_audit = {}
    for row in completed:
        directory = Path(row['output_dir'])
        calibration_path = directory / 'dual_frequency_calibration.json'
        calibration = common.read_json(calibration_path)
        if isinstance(calibration, dict):
            row['dual_frequency_calibration'] = calibration
            row['dual_frequency_calibration_sha256'] = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
        metrics = common.load_completed_metrics(directory / 'metrics.json')
        if metrics:
            row['eval_seconds_total'] = sum(s.get('eval_seconds', 0) for s in metrics['sessions'])
            row['fit_seconds_total'] = sum(s.get('fit_seconds', 0) for s in metrics['sessions'])
            row['frequency_prediction_diagnostics'] = metrics['sessions'][-1].get('frequency_prediction_diagnostics')
            row['anchor_accuracy'] = metrics['sessions'][-1].get('anchor_accuracy')
    for seed in sorted({r['seed'] for r in completed}):
        group = {r['variant']: r for r in completed if r['seed'] == seed}
        anchor = group.get('original_gda')
        if not anchor:
            continue
        checks = {}
        for name, row in group.items():
            if row.get('dual_frequency_calibration'):
                checks[name] = {'anchor_comparison': _compare_predictions(
                    anchor['output_dir'], row['output_dir'], 'anchor_prediction'),
                    'same_anchor_protocol': row['dual_frequency_calibration']['anchor_protocol_sha256']
                    == (anchor.get('uncertainty_calibration') or {}).get('protocol_sha256')}
        if 'zero_dual' in group:
            checks.setdefault('zero_dual', {})['zero_output_comparison'] = _compare_predictions(
                anchor['output_dir'], group['zero_dual']['output_dir'])
        dual_audit[str(seed)] = checks
    comparisons = report.setdefault('paired_delta_vs_reference', {})
    for ref_name in ('fft_dual', 'safd_gda'):
        reference = {r['seed']: r for r in completed if r['variant'] == ref_name}
        if reference:
            comparisons[ref_name] = {
                name: {key: common.metric_statistics([
                    r['summary'][key] - reference[r['seed']]['summary'][key]
                    for r in completed if r['variant'] == name and r['seed'] in reference
                    and r['summary'].get(key) is not None and reference[r['seed']]['summary'].get(key) is not None
                ]) for key in common.METRIC_NAMES} for name in sorted({r['variant'] for r in completed})}
    report.update(dual_pairing_audit=dual_audit,
                  selection_policy='The original GDA anchor is selected independently of the frequency evidence. '
                                   'Dual variants reuse its base validation protocol to select residual strength; '
                                   'all variants are recalibrated with the same grids and supports.',
                  method_note='SAFD scales are adaptive approximation ranks. Text bands describe spatial semantics. '
                              'Validation guards compare against the anchor; no test-time gain is guaranteed. '
                              'FFT and SAFD natural views have different distortion recipes.')
    common.write_json(path, report)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=tuple(DATASET_SPECS),
                        help='Convenience selection of dataset YAML and output directory.')
    parser.add_argument('--data-cfg')
    parser.add_argument('--train-cfg', default=str(REPO_ROOT / 'configs/trainers/bimc_safd_dual.yaml'))
    parser.add_argument('--output-root')
    parser.add_argument('--support-root')
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
    parser.add_argument('--opts', nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args(argv)
    args.data_cfg = args.data_cfg or str(REPO_ROOT / 'configs/datasets' / ((args.dataset or 'cub200') + '.yaml'))
    supplied = dict(zip(args.opts[::2], args.opts[1::2]))
    resolved = dataset_slug(supplied.get('DATASET.NAME') or config_scalar(args.data_cfg, 'NAME'))
    if args.dataset and resolved != args.dataset:
        parser.error('--dataset disagrees with DATASET.NAME in --data-cfg/--opts.')
    suffix = 'cub' if resolved == 'cub200' else resolved
    args.output_root = args.output_root or str(REPO_ROOT / 'outputs' / ('safd_dual_' + suffix))
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        plan = build_plan(args)
    except (ValueError, OSError) as error:
        print('Invalid dual experiment plan: ' + str(error), file=sys.stderr)
        return 2
    if args.summarize_only:
        rows = []
        for run in plan:
            metrics = common.completed_result(run)
            row = {key: run[key] for key in ('variant', 'seed', 'support_seed', 'output_dir',
                                             'support_manifest', 'fingerprint')}
            row.update(status='skipped', summary=metrics['summary']) if metrics else row.update(status='missing_or_stale')
            rows.append(row)
        save_summary(Path(args.output_root).resolve() / 'summary.json', args, rows, len(plan))
        return 0
    if not args.execute:
        for run in plan:
            print(f"{run['variant']} seed={run['seed']}: " + common.describe_command(run['command']))
        try:
            checked = validate_plan_inputs(args, plan)
            if checked:
                print('Frequency text preflight passed: ' + str(checked))
        except (ValueError, OSError) as error:
            print('Frequency text preflight failed: ' + str(error), file=sys.stderr)
            return 2
        print(f'Dry run: {len(plan)} commands; no experiments started.')
        return 0
    try:
        checked = validate_plan_inputs(args, plan)
        if checked:
            print('Frequency text preflight passed: ' + str(checked), flush=True)
    except (ValueError, OSError) as error:
        print('Frequency text preflight failed: ' + str(error), file=sys.stderr)
        return 2
    for run in plan:
        directory = Path(run['output_dir'])
        if directory.exists() and any(directory.iterdir()) and not (directory / 'run_spec.json').is_file():
            print('Refusing unmanaged nonempty output directory: ' + str(directory), file=sys.stderr)
            return 2
    return common.execute_plan(plan, args, summary_writer=save_summary)


if __name__ == '__main__':
    raise SystemExit(main())
