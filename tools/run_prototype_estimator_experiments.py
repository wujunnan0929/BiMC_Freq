"""Plan paired frozen-estimator FSCIL evaluations; dry-run by default."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import run_frequency_uncertainty_experiments as shared
from utils.frequency_descriptions import dataset_slug

common = shared.common
B = 'TRAINER.BiMC.'
P = B + 'PROTOTYPE_ESTIMATOR.'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=('cub200', 'cifar100', 'miniimagenet'), default='cub200')
    p.add_argument('--model', action='append', required=True, help='NAME=PATH.npz; repeat for matched method/auxiliary controls.')
    p.add_argument('--reference-model', help='Name supplying the calibration-only anchor; default is the first named model.')
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    p.add_argument('--variants', nargs='+')
    p.add_argument('--support-root')
    p.add_argument('--output-root')
    p.add_argument('--python', default=sys.executable)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--summarize-only', action='store_true')
    p.add_argument('--keep-going', action='store_true')
    p.add_argument('--opts', nargs=argparse.REMAINDER, default=[])
    a = p.parse_args(argv)
    a.models = {}
    for supplied in a.model:
        if '=' not in supplied:
            p.error('--model must be NAME=PATH.npz')
        name, path = supplied.split('=', 1)
        if not re.fullmatch('[a-z][a-z0-9_]*', name) or name in ('baseline', 'zero_full', 'calibrated') or name in a.models:
            p.error('Use distinct simple model names; baseline/zero_full/calibrated are reserved.')
        a.models[name] = str(Path(path).resolve())
    a.reference_model = a.reference_model or next(iter(a.models))
    if a.reference_model not in a.models:
        p.error('--reference-model must name a supplied model.')
    a.data_cfg = str(ROOT / f'configs/datasets/{a.dataset}.yaml')
    a.train_cfg = str(ROOT / 'configs/trainers/bimc_prototype_estimator.yaml')
    a.output_root = a.output_root or str(ROOT / f'outputs/prototype_estimator_{a.dataset}')
    a.suite, a.rerun = 'core', False
    return a


def check_models(args):
    artifacts, metadata = {}, {}
    for name, supplied in args.models.items():
        path = Path(supplied)
        if path.suffix != '.npz':
            raise ValueError('Estimator model must end in .npz.')
        sidecar = path.with_suffix('.json')
        d = json.loads(sidecar.read_text(encoding='utf-8'))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if d.get('model_sha256') != digest:
            raise ValueError('Model checksum mismatch: ' + name)
        if d.get('format') != 'support_prototype_estimator_v1' or d.get('data_source') != 'base_training_only':
            raise ValueError('Use an exported base-only support estimator: ' + name)
        if dataset_slug(d.get('dataset', '')) != args.dataset:
            raise ValueError('Model dataset does not match --dataset: ' + name)
        if not isinstance(d.get('fitting_seed'), int) or not d.get('selection_protocol_sha256'):
            raise ValueError('Model must record its fitting seed and selection protocol: ' + name)
        metadata[name] = d
        artifacts[name] = {'model': digest, 'sidecar': hashlib.sha256(sidecar.read_bytes()).hexdigest()}
    reference = metadata[args.reference_model]
    for name, d in metadata.items():
        for key in ('cache_sha256', 'clip_weight_sha256', 'base_sample_ids_sorted_sha256',
                    'reference_settings', 'calibration', 'ordinary_description_sha256', 'text_template',
                    'support_transform', 'input_provenance', 'fitting_seed', 'selection_protocol_sha256'):
            if d.get(key) != reference.get(key):
                raise ValueError(f'Models must share the reference inputs and calibration ({key} differs for {name}). Compare fitting seeds in separate matrices.')
    return artifacts


def build_plan(args):
    if len(args.opts) % 2 or len(set(args.opts[::2])) != len(args.opts[::2]):
        raise ValueError('--opts must contain distinct KEY VALUE pairs and be last.')
    artifacts = check_models(args)
    base = {B+'PREC': 'fp32', B+'VISION_CALIBRATION': True, B+'LAMBDA_I': .1,
            B+'TAU': 16, B+'TEXT_CALIBRATION': True, B+'LAMBDA_T': .5,
            B+'USING_ENSEMBLE': True, B+'GAMMA_BASE': 1., B+'GAMMA_INC': 5.,
            B+'FREQUENCY.ENABLED': False, B+'FREQUENCY.ROUTER.ENABLED': False,
            B+'RESIDUAL.ENABLED': False, B+'RESIDUAL.BASE_ONLY': False,
            B+'UNCERTAINTY.ENABLED': False, B+'UNCERTAINTY.BASE_ONLY': False,
            B+'CONSENSUS.ENABLED': False, B+'DUAL_FREQUENCY.ENABLED': False,
            B+'DUAL_FREQUENCY.BASE_ONLY': False, B+'GUIDANCE.ENABLED': False,
            B+'GUIDANCE.BASE_ONLY': False, B+'SAFD_METRIC.ENABLED': False,
            P+'ENABLED': True, P+'MODEL_PATH': args.models[args.reference_model],
            P+'BLEND': -1., P+'CALIBRATION': True}
    supplied = dict(zip(args.opts[::2], args.opts[1::2]))
    if set(supplied) & set(base):
        raise ValueError('Do not override matrix-managed reference, model or branch options.')
    if dataset_slug(supplied.get('DATASET.NAME', args.dataset)) != args.dataset:
        raise ValueError('--dataset and DATASET.NAME disagree.')
    matrix = [('baseline', {**base, P+'ENABLED': False}),
              ('zero_full', {**base, P+'BLEND': 0., P+'CALIBRATION': False}),
              ('calibrated', {**base, P+'BLEND': 0.})]
    matrix += [(name, {**base, P+'MODEL_PATH': path}) for name, path in args.models.items()]
    plan = shared.build_plan(args, matrix=matrix, launcher_path=Path(__file__))
    for run in plan:
        run['prototype_artifact_hashes'] = artifacts
        run['fingerprint'] = common.sha256_json({'prior': run['fingerprint'], 'estimator_artifacts': artifacts})
    return plan


def save_summary(path, args, rows, planned_runs):
    common.save_summary(path, args, rows, planned_runs)
    result = common.read_json(path)
    result['method'] = 'base_trained_frozen_support_prototype_estimator'
    result['reference_model'] = args.reference_model
    result['cost_note'] = 'Cache extraction, base selection and all-base refit are separate offline costs; add them to evaluation cost.'
    for row in result['runs']:
        if row['status'] not in ('completed', 'skipped'):
            continue
        metrics = common.load_completed_metrics(Path(row['output_dir']) / 'metrics.json')
        row['prototype_estimator'] = metrics.get('prototype_estimator')
        row['query_auxiliary_encodings'] = sum(s.get('query_auxiliary_encodings', 0) for s in metrics['sessions'])
    # Compare against the strong calibrated anchor as well as the original BiMC.
    reference = {r['seed']: r for r in result['runs'] if r['variant'] == 'calibrated' and r['status'] in ('completed', 'skipped')}
    paired = {}
    for name in args.models:
        group = [r for r in result['runs'] if r['variant'] == name and r['status'] in ('completed', 'skipped')]
        paired[name] = {k: common.metric_statistics([r['summary'][k] - reference[r['seed']]['summary'][k]
                      for r in group if r['seed'] in reference and r['summary'][k] is not None
                      and reference[r['seed']]['summary'][k] is not None]) for k in common.METRIC_NAMES}
    result['paired_delta_vs_calibrated'] = paired
    common.write_json(path, result)


def main(argv=None):
    args = parse_args(argv)
    try:
        plan = build_plan(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print('Invalid estimator experiment inputs: ' + str(exc), file=sys.stderr)
        return 2
    if args.summarize_only:
        rows = []
        for run in plan:
            metrics = common.completed_result(run)
            row = {k: run[k] for k in ('variant', 'seed', 'support_seed', 'output_dir', 'support_manifest', 'fingerprint')}
            row.update(status='skipped', summary=metrics['summary']) if metrics else row.update(status='missing_or_stale')
            rows.append(row)
        save_summary(Path(args.output_root) / 'summary.json', args, rows, len(plan))
        return 0
    if not args.execute:
        for run in plan:
            print(f"{run['variant']} seed={run['seed']}: " + common.describe_command(run['command']))
        print(f'Dry run: {len(plan)} evaluations; none started.')
        return 0
    for run in plan:
        folder = Path(run['output_dir'])
        if folder.exists() and any(folder.iterdir()) and not (folder / 'run_spec.json').is_file():
            print('Refusing unmanaged nonempty output: ' + str(folder), file=sys.stderr)
            return 2
    return common.execute_plan(plan, args, summary_writer=save_summary)


if __name__ == '__main__':
    raise SystemExit(main())
