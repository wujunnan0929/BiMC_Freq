"""Plan or run paired support-only statistical prototype completion experiments.

Planning uses only the standard library and is the default. Each automatic
variant fits statistics on A classes, selects its blend on held-out V classes,
and refits on all original base training classes before benchmark evaluation.
"""

import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import run_frequency_uncertainty_experiments as shared

common = shared.common
B = 'TRAINER.BiMC.'
P = B + 'PROTOTYPE_COMPLETION.'
F = B + 'FREQUENCY.'


def variants_for_suite(suite='core', semantic_source='structured'):
    if suite not in ('core', 'all'):
        raise ValueError('Unknown completion suite: ' + suite)
    if semantic_source not in ('structured', 'explicit'):
        raise ValueError('Use structured or explicit semantics for the main method.')
    base = {
        B+'PREC': 'fp32', B+'VISION_CALIBRATION': True,
        B+'LAMBDA_I': .1, B+'TAU': 16, B+'TEXT_CALIBRATION': True,
        B+'LAMBDA_T': .5, B+'USING_ENSEMBLE': True,
        B+'GAMMA_BASE': 1., B+'GAMMA_INC': 5.,
        F+'ENABLED': False, F+'ROUTER.ENABLED': False,
        F+'BACKEND': 'fft', F+'VIEW_MODE': 'natural',
        F+'USE_EXPLICIT_DESCRIPTIONS': False,
        B+'RESIDUAL.ENABLED': False, B+'RESIDUAL.BASE_ONLY': False,
        B+'CONSENSUS.ENABLED': False, B+'CONSENSUS.SAVE_PREDICTIONS': True,
        B+'UNCERTAINTY.ENABLED': False, B+'UNCERTAINTY.BASE_ONLY': False,
        B+'DUAL_FREQUENCY.ENABLED': False, B+'DUAL_FREQUENCY.BASE_ONLY': False,
        B+'GUIDANCE.ENABLED': False, B+'GUIDANCE.BASE_ONLY': False,
        B+'SAFD_METRIC.ENABLED': False, B+'PROTOTYPE_ESTIMATOR.ENABLED': False,
        B+'COMPETITIVE_FUSION.ENABLED': False,
        P+'ENABLED': True, P+'VIEW_CONTROL': 'frequency',
        P+'SEMANTIC_SOURCE': semantic_source, P+'SEMANTIC_PERMUTATION': [0, 1, 2],
        P+'CANDIDATES_PER_BAND': 1 if semantic_source == 'structured' else 5,
        P+'CROSS_VIEW': True, P+'AUTO_CALIBRATE': True, P+'BLEND': .5,
        P+'BASE_ONLY': False, P+'SAVE_PREDICTIONS': True,
    }
    changes = [
        ('baseline', {P+'ENABLED': False}),
        ('zero', {P+'BLEND': 0., P+'AUTO_CALIBRATE': False}),
        ('original', {P+'VIEW_CONTROL': 'original'}),
        ('frequency', {}),
        ('ordinary_text', {P+'SEMANTIC_SOURCE': 'ordinary'}),
        ('permuted_text', {P+'SEMANTIC_PERMUTATION': [2, 0, 1]}),
        ('augmentation', {P+'VIEW_CONTROL': 'augmentation'}),
        ('repeat', {P+'VIEW_CONTROL': 'repeat'}),
        ('block', {P+'CROSS_VIEW': False}),
    ]
    # "all" is an alias for this first, entirely gradient-free matrix.
    return [(name, {**base, **options}) for name, options in changes]


MANAGED = (shared.MANAGED
           | set(variants_for_suite()[0][1])
           | {P+'CALIBRATION_SEED', 'DATASET.NAME'})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('cub200', 'cifar100', 'miniimagenet'), default='cub200')
    parser.add_argument('--data-cfg', help='Defaults to configs/datasets/DATASET.yaml.')
    parser.add_argument('--train-cfg', default=str(ROOT/'configs/trainers/bimc_prototype_completion.yaml'))
    parser.add_argument('--output-root')
    parser.add_argument('--support-root', help='Shared support-manifest root; defaults to OUTPUT/support.')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--suite', choices=('core', 'all'), default='core')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--variants', nargs='+')
    parser.add_argument('--semantic-source', choices=('structured', 'explicit'), default='structured',
                        help='Main-method semantics; ordinary_text always uses ordinary class text.')
    parser.add_argument('--description-path', help='Validated frequency descriptions; required with explicit semantics.')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--rerun', action='store_true')
    parser.add_argument('--keep-going', action='store_true')
    parser.add_argument('--opts', nargs=argparse.REMAINDER, default=[], help='YACS KEY VALUE pairs; place last.')
    args = parser.parse_args(argv)
    args.data_cfg = args.data_cfg or str(ROOT/f'configs/datasets/{args.dataset}.yaml')
    args.output_root = args.output_root or str(ROOT/f'outputs/prototype_completion_{args.dataset}')
    return args


def build_plan(args):
    if len(args.opts) % 2 or len(set(args.opts[::2])) != len(args.opts[::2]):
        raise ValueError('--opts requires distinct KEY VALUE pairs and must be last.')
    collision = set(args.opts[::2]) & MANAGED
    if collision:
        raise ValueError('Matrix-managed keys cannot be overridden: ' + ', '.join(sorted(collision))
                         + '. For manual blend or base-only calibration, call main.py directly.')
    if args.semantic_source == 'explicit' and not args.description_path:
        raise ValueError('--semantic-source explicit requires --description-path.')
    if args.description_path and args.semantic_source != 'explicit':
        raise ValueError('--description-path requires --semantic-source explicit.')
    opts = list(args.opts)
    if args.description_path:
        if F+'EXPLICIT_DESCRIPTION_PATH' in opts[::2]:
            raise ValueError('Supply the description path once with --description-path.')
        path = Path(args.description_path).resolve()
        if not path.is_file():
            raise ValueError('Explicit description file does not exist: ' + str(path))
        opts.extend((F+'EXPLICIT_DESCRIPTION_PATH', str(path)))
    # The shared launcher supplies support identities, source/config fingerprints,
    # executable argv quoting and completion/resume checks without importing torch.
    import copy
    local = copy.copy(args)
    local.opts = opts
    plan = shared.build_plan(local, matrix=variants_for_suite(args.suite, args.semantic_source),
                             launcher_path=Path(__file__))
    for run in plan:
        run['command'].extend((P+'CALIBRATION_SEED', str(run['seed'] + 10101)))
        run['fingerprint'] = common.sha256_json({
            'shared_fingerprint': run['fingerprint'], 'command': run['command'],
            'method': 'joint_statistical_prototype_completion_v1',
        })
    return plan


def audit_pairing(rows):
    adapted = []
    for row in rows:
        adapted.append({**row, 'uncertainty_calibration': row.get('completion_calibration')})
    result = shared.audit_pairing(adapted)
    for report in result.values():
        for name, errors in report['calibration_errors'].items():
            report['calibration_errors'][name] = [message.replace('uncertainty', 'completion') for message in errors]
    for row in rows:
        if row['status'] not in ('completed', 'skipped'):
            continue
        report = result[str(row['seed'])]
        errors = []
        calibration = row.get('completion_calibration')
        if isinstance(calibration, dict) and calibration.get('auto_calibrate') is True:
            errors.extend(_protocol_errors(calibration))
        if row.get('query_auxiliary_encodings') not in (None, 0):
            errors.append('Query auxiliary encodings must be zero for support-only completion.')
        if errors:
            report['calibration_errors'].setdefault(row['variant'], []).extend(errors)
    return result


def _protocol_errors(calibration):
    """Check saved selection provenance without importing the numerical runtime."""
    protocol = calibration.get('protocol')
    if not isinstance(protocol, dict):
        return ['Automatic completion calibration is missing its full selection protocol.']
    errors = []
    content = {key: value for key, value in protocol.items() if key != 'sha256'}
    computed = common.sha256_json(content)
    if computed != protocol.get('sha256') or computed != calibration.get('protocol_sha256'):
        errors.append('Selection protocol content/checksum mismatch.')
    if protocol.get('source') != 'base_training_only' or protocol.get('seed') != calibration.get('seed'):
        errors.append('Selection protocol source/seed does not match its report.')
    a, v = protocol.get('pseudo_base_class_ids'), protocol.get('pseudo_validation_class_ids')
    if a != calibration.get('fit_class_ids') or v != calibration.get('validation_class_ids'):
        errors.append('Selection protocol classes do not match the fit/validation report.')
    try:
        sample_ids = protocol['sample_ids']
        fit = set(protocol['fitting_indices'])
        reserved = {int(key): set(values) for key, values in protocol['reserved_pseudo_base_query_indices'].items()}
        all_a_query = set().union(*reserved.values())
        if len(sample_ids) != len(set(sample_ids)):
            errors.append('Selection protocol contains duplicate sample identities.')
        if not fit or fit & all_a_query:
            errors.append('A fitting rows overlap reserved A queries, or the fitting pool is empty.')
        for seq in protocol['sequences']:
            ids, support, query = seq['class_ids'], seq['support_indices'], seq['query_indices']
            if len(ids) != len(support) or len(ids) != len(query) or ids[:len(a)] != a:
                errors.append('Selection sequence classes/support/query have incompatible structure.')
                break
            support_rows = set(i for values in support for i in values)
            query_rows = set(i for values in query for i in values)
            if support_rows & query_rows:
                errors.append('Selection support/query rows overlap.')
                break
            all_rows = fit | all_a_query | support_rows | query_rows
            if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= len(sample_ids) for i in all_rows):
                errors.append('Selection protocol contains invalid sample indices.')
                break
            if set(i for values in support[:len(a)] for i in values) != fit:
                errors.append('A fitting pool differs from the paired A support rows.')
                break
            if any(set(query[j]) != reserved[c] for j, c in enumerate(a)):
                errors.append('A reserved queries change between selection sequences.')
                break
            v_rows = set(i for values in support[len(a):]+query[len(a):] for i in values)
            if v_rows & fit:
                errors.append('Validation classes contribute rows to the A fitting pool.')
                break
        if not protocol['sequences']:
            errors.append('Automatic selection protocol has no validation sequences.')
    except (KeyError, TypeError, ValueError):
        errors.append('Selection protocol is missing valid fitting/support/query indices.')
    return errors


def _paired(rows, reference_name):
    successful = [r for r in rows if r['status'] in ('completed', 'skipped')]
    reference = {r['seed']: r for r in successful if r['variant'] == reference_name}
    return {name: {key: common.metric_statistics([
        row['summary'][key] - reference[row['seed']]['summary'][key]
        for row in successful if row['variant'] == name and row['seed'] in reference
        and row['summary'].get(key) is not None
        and reference[row['seed']]['summary'].get(key) is not None
    ]) for key in common.METRIC_NAMES} for name in sorted({r['variant'] for r in successful})}


def save_summary(path, args, rows, planned_runs):
    enriched = []
    for source in rows:
        row = dict(source)
        if row['status'] in ('completed', 'skipped'):
            directory = Path(row['output_dir'])
            metrics = common.load_completed_metrics(directory/'metrics.json')
            calibration_path = directory/'completion_calibration.json'
            calibration = common.read_json(calibration_path)
            row['completion_calibration'] = calibration if isinstance(calibration, dict) else None
            if isinstance(calibration, dict):
                row['completion_calibration_sha256'] = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
            if metrics is not None:
                row['prototype_completion'] = metrics.get('prototype_completion')
                row['query_auxiliary_encodings'] = sum(s.get('query_auxiliary_encodings', 0) for s in metrics['sessions'])
                row['exact_reference_all_sessions'] = all(
                    s.get('reference_scores_exact_equal') is True for s in metrics['sessions'])
                row['prediction_diagnostics'] = metrics['sessions'][-1].get('prediction_diagnostics')
        enriched.append(row)
    aggregates, paired = common.aggregate_results(enriched)
    common.write_json(path, {
        'schema_version': 1, 'generated_at': common.utc_now(), 'suite': args.suite,
        'method': 'joint_statistical_prototype_completion_v1',
        'dataset': args.dataset, 'seeds': args.seeds, 'semantic_source': args.semantic_source,
        'metrics_unit': 'percent; paired deltas in percentage points',
        'planned_runs': planned_runs,
        'completed_runs': sum(r['status'] in ('completed', 'skipped') for r in enriched),
        'failed_runs': sum(r['status'] == 'failed' for r in enriched),
        'runs': enriched, 'aggregates': aggregates, 'paired_delta_vs_baseline': paired,
        'paired_delta_vs_original': _paired(enriched, 'original'),
        'paired_delta_vs_augmentation': _paired(enriched, 'augmentation'),
        'pairing_audit': audit_pairing(enriched),
        'selection_policy': 'Each automatic control fits A only, selects blend on the same held-out '
                            'base-class protocol including zero, then refits statistics on all base training data.',
        'cost_note': 'No new gradients; base statistics and base validation are data fitting. '
                     'Frequency views are support-only; query auxiliary encodings must be zero.',
        'warning': 'No measured accuracy gain is assumed. Benchmark results do not select hyperparameters; '
                   'inspect old/new tradeoffs and matched ordinary-view/text controls.',
    })


def main(argv=None):
    args = parse_args(argv)
    try:
        plan = build_plan(args)
    except (OSError, ValueError) as exc:
        print('Invalid completion experiment plan: ' + str(exc), file=sys.stderr)
        return 2
    if args.summarize_only:
        rows = []
        for run in plan:
            metrics = common.completed_result(run)
            row = {key: run[key] for key in ('variant', 'seed', 'support_seed', 'output_dir',
                                             'support_manifest', 'fingerprint')}
            row.update(status='skipped', summary=metrics['summary']) if metrics else row.update(status='missing_or_stale')
            rows.append(row)
        save_summary(Path(args.output_root).resolve()/'summary.json', args, rows, len(plan))
        return 0
    if not args.execute:
        for run in plan:
            print(f"{run['variant']} seed={run['seed']}: " + common.describe_command(run['command']))
        print(f'Dry run: {len(plan)} evaluations; no experiments or output writes. Pass --execute to run.')
        return 0
    for run in plan:
        directory = Path(run['output_dir'])
        if directory.exists() and any(directory.iterdir()) and not (directory/'run_spec.json').is_file():
            print('Refusing unmanaged nonempty output directory: ' + str(directory), file=sys.stderr)
            return 2
    return common.execute_plan(plan, args, summary_writer=save_summary)


if __name__ == '__main__':
    raise SystemExit(main())
