"""Plan or execute base-only attribute-channel transfer experiments.

The default is a standard-library-only dry run. Numerical imports and input
cache validation occur only with --execute. No formal FSCIL test data is used.
"""

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True, help='Certified complete base visual feature .npz cache.')
    parser.add_argument('--grounding', required=True, help='Matching certified attribute grounding .npz sidecar.')
    parser.add_argument('--output-root', required=True, help='New or matching resumable study directory.')
    parser.add_argument('--fit-seeds', type=int, nargs='+', default=[2, 3, 4])
    parser.add_argument('--sequences', type=int, default=20)
    parser.add_argument('--inner-sequences', type=int, default=5,
                        help='A-only pseudo-incremental sequences used to fit the optional score offset.')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--blend-grid', type=float, nargs='+', default=[0., .05, .1, .25, .5, 1.])
    parser.add_argument('--offset-grid', type=float, nargs='+', default=[-.1, -.05, -.02, 0., .02, .05, .1])
    parser.add_argument('--max-group-drop', type=float, default=.2,
                        help='Maximum mean Base/Novel decrease in percentage points.')
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=16.)
    parser.add_argument('--no-resume', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    parser.add_argument('--package', help='Create a NEW compact analysis .zip after execution.')
    parser.add_argument('--allow-incomplete-package', action='store_true')
    args = parser.parse_args(argv)
    if min(args.fit_seeds) < 1 or len(set(args.fit_seeds)) != len(args.fit_seeds):
        parser.error('--fit-seeds must be distinct positive integers.')
    if min(args.sequences, args.inner_sequences, args.threads, args.topk) < 1:
        parser.error('sequences, inner-sequences, threads and topk must be positive.')
    if (0. not in args.blend_grid or len(set(args.blend_grid)) != len(args.blend_grid)
            or any(not math.isfinite(v) or not 0 <= v <= 1 for v in args.blend_grid)):
        parser.error('--blend-grid requires distinct finite values in [0,1], including zero.')
    if (0. not in args.offset_grid or len(set(args.offset_grid)) != len(args.offset_grid)
            or any(not math.isfinite(v) or abs(v)>10 for v in args.offset_grid)):
        parser.error('--offset-grid requires distinct finite values in [-10,10], including zero.')
    if not math.isfinite(args.max_group_drop) or args.max_group_drop < 0:
        parser.error('--max-group-drop must be finite and nonnegative.')
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error('--temperature must be finite and positive.')
    for key in ('cache', 'grounding'):
        if Path(getattr(args, key)).suffix.lower() != '.npz':
            parser.error('--'+key+' must end in .npz.')
    if args.package and Path(args.package).suffix.lower() != '.zip':
        parser.error('--package must be a new .zip file.')
    if args.allow_incomplete_package and not args.package:
        parser.error('--allow-incomplete-package requires --package.')
    return args


def build_plan(args):
    from engine.attribute_transfer_experiments import experiment_variants
    variants = experiment_variants()
    runs = [{'fit_seed':seed, 'stage':'transfer', 'variant':variant['name'],
             'configuration':variant} for seed in args.fit_seeds for variant in variants]
    return {'method':'attribute_transfer_development', 'data_source':'base_training_only',
            'benchmark_evaluation_performed':False,
            'cache':str(Path(args.cache).resolve()), 'grounding':str(Path(args.grounding).resolve()),
            'output_root':str(Path(args.output_root).resolve()), 'fit_seeds':args.fit_seeds,
            'sequences':args.sequences, 'inner_sequences':args.inner_sequences,
            'device':args.device, 'threads':args.threads, 'topk':args.topk,
            'temperature':args.temperature, 'blend_grid':args.blend_grid,
            'offset_grid':args.offset_grid, 'maximum_group_drop_pp':args.max_group_drop,
            'planned_configurations':len(runs), 'runs':runs,
            'package':str(Path(args.package).resolve()) if args.package else None,
            'note':'Planning does not read or authenticate either cache. The score offset is fit '
                   'only on A-inner episodes; outer V is used for development selection.'}


def main(argv=None):
    args = parse_args(argv)
    try:
        plan = build_plan(args)
    except (ImportError, OSError, ValueError) as error:
        print('Invalid attribute transfer plan: '+str(error), file=sys.stderr)
        return 2
    if not args.execute:
        print(json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False))
        print(f'Dry run: {plan["planned_configurations"]} base-only configurations; no caches loaded or files written.')
        return 0
    if args.package and (Path(args.package).exists() or Path(args.package).is_symlink()):
        print('Refusing existing analysis bundle: '+str(args.package), file=sys.stderr)
        return 2
    try:
        from engine.attribute_transfer_experiments import run_experiments
        summary = run_experiments(args.cache, args.grounding, args.output_root,
            fit_seeds=tuple(args.fit_seeds), sequences=args.sequences,
            inner_sequences=args.inner_sequences, device=args.device, threads=args.threads,
            blend_grid=tuple(args.blend_grid), offset_grid=tuple(args.offset_grid),
            maximum_group_drop_pp=args.max_group_drop, topk=args.topk,
            temperature=args.temperature, resume=not args.no_resume)
        print('Base-only study summary:', str(Path(args.output_root).resolve()/'summary.json'))
        print('Completed configurations:', summary.get('completed_runs', 'unknown'))
        if args.package:
            from tools.package_attribute_transfer_results import package_results
            manifest = package_results(args.output_root, args.package,
                                       allow_incomplete=args.allow_incomplete_package)
            print('Analysis bundle:', str(Path(args.package).resolve()),
                  'complete=',manifest['complete'], 'files=',len(manifest['files']))
        return 0 if summary.get('study_complete') is True else 1
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print('Attribute transfer experiment failed: '+str(error), file=sys.stderr)
        print('Existing results are retained. Use tools/package_attribute_transfer_results.py '
              '--allow-incomplete for an explicitly partial diagnostic bundle.', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
