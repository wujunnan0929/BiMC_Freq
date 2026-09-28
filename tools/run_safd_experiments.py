"""Paired SAFD/FFT evaluation with base-only calibration and shared supports."""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from tools import run_frequency_uncertainty_experiments as shared

common = shared.common
F, U = shared.F, shared.U
MANAGED = shared.MANAGED | {F + 'BACKEND', F + 'VIEW_MODE', U + 'MEAN_UNCERTAINTY',
                            F + 'SAFD.ORIGINAL_BLEND', F + 'HIGH_ENHANCE'}


def variants_for_suite(suite):
    base = {
        U + 'ENABLED': True, U + 'VIEW_CONTROL': 'joint',
        U + 'COVARIANCE': 'full_shared', U + 'MEAN_UNCERTAINTY': False,
        U + 'AUTO_CALIBRATE': True, U + 'ALPHA': 0.0, U + 'BASE_ONLY': False,
        F + 'ENABLED': False, F + 'ROUTER.ENABLED': False,
        shared.C + 'ENABLED': False, shared.R + 'ENABLED': False,
        shared.R + 'BASE_ONLY': False,
        F + 'BACKEND': 'safd', F + 'VIEW_MODE': 'natural',
        F + 'HIGH_ENHANCE': 0.25, F + 'SAFD.ORIGINAL_BLEND': [0.25, 0.1],
    }
    definitions = [
        ('baseline', {U + 'ENABLED': False}),
        ('zero', {U + 'AUTO_CALIBRATE': False}),
        ('original_gda', {U + 'VIEW_CONTROL': 'original'}),
        ('fft_gda', {F + 'BACKEND': 'fft'}),
        ('safd_gda', {}),
    ]
    if suite == 'all':
        definitions.extend([
            ('safd_no_blend', {F + 'SAFD.ORIGINAL_BLEND': [0.0, 0.0]}),
            ('safd_no_enhance', {F + 'HIGH_ENHANCE': 0.0}),
            ('repeat_gda', {U + 'VIEW_CONTROL': 'repeat'}),
        ])
    elif suite != 'core':
        raise ValueError('Unknown SAFD suite: ' + suite)
    return [(name, {**base, **changes}) for name, changes in definitions]


def build_plan(args):
    collision = set(args.opts[::2]) & MANAGED
    if collision:
        raise ValueError('Matrix-managed keys cannot be overridden: ' + ', '.join(sorted(collision)))
    return shared.build_plan(args, matrix=variants_for_suite(args.suite), launcher_path=Path(__file__))


def save_summary(path, args, rows, planned_runs):
    shared.save_summary(path, args, rows, planned_runs)
    report = common.read_json(path)
    completed = [r for r in report['runs'] if r['status'] in ('completed', 'skipped')]
    reference = {r['seed']: r for r in completed if r['variant'] == 'fft_gda'}
    report['paired_delta_vs_fft'] = {
        name: {key: common.metric_statistics([
            r['summary'][key] - reference[r['seed']]['summary'][key]
            for r in completed if r['variant'] == name and r['seed'] in reference
            and r['summary'].get(key) is not None and reference[r['seed']]['summary'].get(key) is not None
        ]) for key in common.METRIC_NAMES}
        for name in sorted({r['variant'] for r in completed})
    }
    report['method_note'] = ('SAFD scales are adaptive approximation ranks, not radial frequency bands. '
                             'Configuration is an unvalidated starting point, not a claimed accuracy gain.')
    common.write_json(path, report)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-cfg', default=str(REPO_ROOT / 'configs/datasets/cub200.yaml'))
    parser.add_argument('--train-cfg', default=str(REPO_ROOT / 'configs/trainers/bimc_safd.yaml'))
    parser.add_argument('--output-root', default=str(REPO_ROOT / 'outputs/safd_cub'))
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
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        plan = build_plan(args)
    except (ValueError, OSError) as error:
        print('Invalid SAFD experiment plan: ' + str(error), file=sys.stderr)
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
        print(f'Dry run: {len(plan)} commands; no experiments started.')
        return 0
    for run in plan:
        directory = Path(run['output_dir'])
        if directory.exists() and any(directory.iterdir()) and not (directory / 'run_spec.json').is_file():
            print('Refusing unmanaged nonempty output directory: ' + str(directory), file=sys.stderr)
            return 2
    return common.execute_plan(plan, args, summary_writer=save_summary)


if __name__ == '__main__':
    raise SystemExit(main())
