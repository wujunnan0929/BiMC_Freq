"""Plan or run paired BASE-ONLY multilayer and multiattribute completion studies.

Planning is the default, uses only the standard library, and writes nothing.
The numerical engine and cached features are loaded only with --execute.
"""

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STAGES = ('repeat', 'layers', 'attributes')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True, help='Authenticated complete original-base .npz cache.')
    parser.add_argument('--output-root', required=True, help='New or matching resumable experiment directory.')
    parser.add_argument('--fit-seeds', nargs='+', type=int, default=[2, 3, 4],
                        help='Base class-split seeds, not formal FSCIL support/test seeds.')
    parser.add_argument('--stages', nargs='+', choices=STAGES, default=list(STAGES))
    parser.add_argument('--sequences', type=int, default=20)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--selected-layer', type=int, choices=(3, 6, 9),
                        help='Optional predeclared manual layer; otherwise selected using base validation.')
    parser.add_argument('--blend-grid', type=float, nargs='+', default=[0., .25, .5, 1.])
    parser.add_argument('--max-group-drop', type=float, default=.2,
                        help='Maximum mean Base/Novel decrease, in percentage points.')
    parser.add_argument('--no-resume', action='store_true',
                        help='Do not reuse completed matching runs; never authorizes overwriting artifacts.')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    parser.add_argument('--package', help='After successful execution create a NEW analysis .zip.')
    parser.add_argument('--allow-incomplete-package', action='store_true',
                        help='Permit an explicitly marked incomplete analysis bundle.')
    args = parser.parse_args(argv)
    if (not args.fit_seeds or min(args.fit_seeds) < 1
            or len(set(args.fit_seeds)) != len(args.fit_seeds)):
        parser.error('--fit-seeds must be distinct positive integers.')
    if len(set(args.stages)) != len(args.stages):
        parser.error('--stages must be distinct.')
    if min(args.sequences, args.threads) < 1:
        parser.error('--sequences and --threads must be positive.')
    if (not args.blend_grid or 0 not in args.blend_grid
            or len(set(args.blend_grid)) != len(args.blend_grid)
            or any(not math.isfinite(v) or not 0 <= v <= 1 for v in args.blend_grid)):
        parser.error('--blend-grid must contain zero and distinct finite values in [0,1].')
    if not math.isfinite(args.max_group_drop) or args.max_group_drop < 0:
        parser.error('--max-group-drop must be finite and nonnegative.')
    if Path(args.cache).suffix.lower() != '.npz':
        parser.error('--cache must be a .npz file.')
    if args.package and Path(args.package).suffix.lower() != '.zip':
        parser.error('--package must be a new .zip file.')
    if args.allow_incomplete_package and not args.package:
        parser.error('--allow-incomplete-package requires --package.')
    args.stages = [name for name in STAGES if name in args.stages]
    return args


def build_plan(args):
    """Describe engine variants without opening the cache or importing torch."""
    from engine.multilayer_completion_experiments import experiment_variants
    rows = []
    for seed in args.fit_seeds:
        for stage in args.stages:
            for variant in experiment_variants(stage, args.selected_layer):
                name = variant['name']
                rows.append({'fit_seed': seed, 'stage': stage, 'variant': name,
                             'configuration': variant})
    return {'method': 'multilayer_multiattribute_completion_development',
            'data_source': 'base_training_only', 'benchmark_evaluation_performed': False,
            'cache': str(Path(args.cache).resolve()),
            'output_root': str(Path(args.output_root).resolve()),
            'fit_seeds': list(args.fit_seeds), 'stages': list(args.stages),
            'sequences': args.sequences, 'device': args.device, 'threads': args.threads,
            'blend_grid': list(args.blend_grid), 'maximum_group_drop_pp': args.max_group_drop,
            'selected_layer': args.selected_layer,
            'layer_selection': ('manual_predeclared' if args.selected_layer is not None
                                else 'base_validation_or_matching_saved_layer_selection'),
            'planned_configurations': len(rows), 'runs': rows,
            'package': str(Path(args.package).resolve()) if args.package else None,
            'note': 'Planning does not validate cache contents. Attributes require a matching base-only '
                    'layer selection or an explicitly predeclared --selected-layer.'}


def main(argv=None):
    args = parse_args(argv)
    try:
        plan = build_plan(args)
    except (ValueError, OSError, ImportError) as error:
        print('Invalid multilayer completion plan: ' + str(error), file=sys.stderr)
        return 2
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False))
        print(f'Dry run: {plan["planned_configurations"]} base-only configurations; no cache loaded or files written.')
        return 0
    # Refuse a doomed bundle request before doing numerical work.
    if args.package and Path(args.package).exists():
        print('Refusing existing analysis bundle: ' + str(args.package), file=sys.stderr)
        return 2
    try:
        from engine.multilayer_completion_experiments import run_experiments
        summary = run_experiments(
            args.cache, args.output_root, fit_seeds=tuple(args.fit_seeds),
            stages=tuple(args.stages), sequences=args.sequences,
            device=args.device, threads=args.threads,
            maximum_group_drop_pp=args.max_group_drop,
            blend_grid=tuple(args.blend_grid), selected_layer=args.selected_layer,
            resume=not args.no_resume)
        print('Base-only study summary:', str(Path(args.output_root).resolve()/'summary.json'))
        print('Completed configurations:', summary.get('completed_runs', 'unknown'))
        if args.package:
            from tools.package_multilayer_completion_results import package_results
            manifest = package_results(args.output_root, args.package,
                                       allow_incomplete=args.allow_incomplete_package)
            print('Analysis bundle:', str(Path(args.package).resolve()),
                  'complete=', manifest['complete'], 'files=', len(manifest['files']))
        return 0 if not summary.get('failed_runs', 0) else 1
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print('Multilayer completion experiment failed: ' + str(error), file=sys.stderr)
        print('Existing results are retained. To package partial diagnostics, use '
              'tools/package_multilayer_completion_results.py --allow-incomplete.', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
