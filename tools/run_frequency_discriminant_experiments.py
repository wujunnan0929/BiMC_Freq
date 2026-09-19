"""Plan paired training-free GDA experiments; execution is opt-in.

Joint features contain the original image and three fixed frequency views.
All variants reuse the uncertainty runner's support, resume, and audit logic.
"""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from tools import run_frequency_uncertainty_experiments as uncertainty

common = uncertainty.common
U, F, C, R = uncertainty.U, uncertainty.F, uncertainty.C, uncertainty.R
MANAGED = uncertainty.MANAGED | {U + 'MEAN_UNCERTAINTY', U + 'RIDGE'}
audit_pairing = uncertainty.audit_pairing
save_summary = uncertainty.save_summary


def variants_for_suite(suite):
    options = {
        U + 'ENABLED': True, U + 'VIEW_CONTROL': 'joint',
        U + 'COVARIANCE': 'full_shared', U + 'MEAN_UNCERTAINTY': False,
        U + 'RIDGE': 0.1, U + 'AUTO_CALIBRATE': True,
        U + 'ALPHA': 0.0, U + 'BASE_ONLY': False,
        F + 'ENABLED': False, F + 'ROUTER.ENABLED': False,
        C + 'ENABLED': False, R + 'ENABLED': False, R + 'BASE_ONLY': False,
    }
    definitions = [
        ('baseline', {U + 'ENABLED': False}),
        ('original_shared', {
            U + 'VIEW_CONTROL': 'original', U + 'COVARIANCE': 'shared',
            U + 'MEAN_UNCERTAINTY': True,
        }),
        ('frequency_shrinkage', {
            U + 'VIEW_CONTROL': 'frequency', U + 'COVARIANCE': 'shrinkage',
            U + 'MEAN_UNCERTAINTY': True,
        }),
        ('original_gda', {U + 'VIEW_CONTROL': 'original'}),
        ('frequency_gda_block', {U + 'COVARIANCE': 'block_shared'}),
        ('frequency_gda_joint', {}),
    ]
    if suite == 'all':
        definitions.extend([
            ('original_diagonal', {
                U + 'VIEW_CONTROL': 'original', U + 'COVARIANCE': 'shared',
            }),
            ('original_repeat_gda', {U + 'VIEW_CONTROL': 'repeat'}),
        ])
    elif suite != 'core':
        raise ValueError('Unknown suite: ' + suite)
    return [(name, {**options, **overrides}) for name, overrides in definitions]


def build_plan(args):
    collision = set(args.opts[::2]) & MANAGED
    if collision:
        raise ValueError('Matrix-managed keys cannot be overridden: ' + ', '.join(sorted(collision))
                         + '. For base-only calibration, call main.py directly.')
    return uncertainty.build_plan(args, matrix=variants_for_suite(args.suite),
                                  launcher_path=Path(__file__))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-cfg', default=str(REPO_ROOT / 'configs/datasets/cub200.yaml'))
    parser.add_argument('--train-cfg', default=str(REPO_ROOT / 'configs/trainers/bimc_frequency_discriminant.yaml'))
    parser.add_argument('--output-root', default=str(REPO_ROOT / 'outputs/frequency_discriminant_cub'))
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
    parser.add_argument('--opts', nargs=argparse.REMAINDER, default=[],
                        help='Additional YACS KEY VALUE pairs; place last.')
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
            if metrics:
                row.update(status='skipped', summary=metrics['summary'])
            else:
                row.update(status='missing_or_stale')
            rows.append(row)
        path = Path(args.output_root).resolve() / 'summary.json'
        save_summary(path, args, rows, len(plan))
        print('Summary: ' + str(path))
        return 0
    if not args.execute:
        for run in plan:
            print(f"{run['variant']} seed={run['seed']}: " + common.describe_command(run['command']))
        print(f'Dry run: {len(plan)} commands; no files or experiment processes created.')
        return 0
    for run in plan:
        directory = Path(run['output_dir'])
        if directory.exists() and any(directory.iterdir()) and not (directory / 'run_spec.json').is_file():
            print('Refusing unmanaged nonempty output directory: ' + str(directory), file=sys.stderr)
            return 2
    return common.execute_plan(plan, args, summary_writer=save_summary)


if __name__ == '__main__':
    raise SystemExit(main())
