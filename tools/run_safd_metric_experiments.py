"""Paired evaluation of previously base-fitted SAFD metrics; dry-run by default."""

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
from utils.frequency_descriptions import dataset_slug, config_scalar

common = shared.common
M = 'TRAINER.BiMC.SAFD_METRIC.'
B = 'TRAINER.BiMC.'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('cub200', 'cifar100', 'miniimagenet'), default='cub200')
    parser.add_argument('--model', action='append', required=True, help='NAME=PATH.npz; include safd=...; repeat for controls.')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--variants', nargs='+', help='Subset of baseline, zero, and named models.')
    parser.add_argument('--support-root')
    parser.add_argument('--output-root')
    parser.add_argument('--python', default=sys.executable)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--keep-going', action='store_true')
    parser.add_argument('--opts', nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args(argv)
    args.models = {}
    for specification in args.model:
        if '=' not in specification:
            parser.error('--model must be NAME=PATH.npz')
        name, path = specification.split('=', 1)
        if not re.fullmatch('[a-z][a-z0-9_]*', name) or name in ('baseline', 'zero') or name in args.models:
            parser.error('Model names must be distinct simple names; baseline/zero are reserved.')
        args.models[name] = str(Path(path).resolve())
    if 'safd' not in args.models:
        parser.error('Include --model safd=PATH.npz for the primary SAFD candidate.')
    args.data_cfg = str(ROOT / f'configs/datasets/{args.dataset}.yaml')
    args.train_cfg = str(ROOT / 'configs/trainers/bimc_safd_metric.yaml')
    args.output_root = args.output_root or str(ROOT / f'outputs/safd_metric_{args.dataset}')
    args.suite, args.rerun = 'core', False
    return args


def check_models(args):
    artifacts = {}
    for name, supplied in args.models.items():
        path = Path(supplied)
        if path.suffix != '.npz':
            raise ValueError('Metric model must be a .npz: ' + str(path))
        sidecar = path.with_suffix('.json')
        metadata = json.loads(sidecar.read_text(encoding='utf-8'))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if metadata.get('model_sha256') != digest:
            raise ValueError('Metric model checksum mismatch: ' + str(path))
        if dataset_slug(metadata.get('dataset', '')) != args.dataset or metadata.get('data_source') != 'base_training_only':
            raise ValueError('Model must match dataset and certify base_training_only: ' + str(path))
        if name == 'safd' and metadata.get('source') != 'safd':
            raise ValueError('The primary safd model must use a SAFD cache, not an FFT/augmentation control.')
        if name == 'safd' and metadata.get('basis_mode') != 'explained':
            raise ValueError('The primary safd model must use the explained SAFD-response basis.')
        artifacts[name] = {'model': digest, 'sidecar': hashlib.sha256(sidecar.read_bytes()).hexdigest()}
    return artifacts


def build_plan(args):
    if len(args.opts) % 2 or len(set(args.opts[::2])) != len(args.opts[::2]):
        raise ValueError('--opts must contain distinct KEY VALUE pairs and be last.')
    supplied = dict(zip(args.opts[::2], args.opts[1::2]))
    if dataset_slug(supplied.get('DATASET.NAME') or config_scalar(args.data_cfg, 'NAME')) != args.dataset:
        raise ValueError('--dataset disagrees with the dataset configuration.')
    artifacts = check_models(args)
    base = {B+'PREC': 'fp32', B+'VISION_CALIBRATION': True, B+'LAMBDA_I': .1,
            B+'TAU': 16, B+'TEXT_CALIBRATION': True, B+'LAMBDA_T': .5,
            B+'USING_ENSEMBLE': True, B+'GAMMA_BASE': 1., B+'GAMMA_INC': 5.,
            B+'FREQUENCY.ENABLED': False, B+'FREQUENCY.ROUTER.ENABLED': False,
            B+'RESIDUAL.ENABLED': False, B+'RESIDUAL.BASE_ONLY': False,
            B+'UNCERTAINTY.ENABLED': False, B+'UNCERTAINTY.BASE_ONLY': False,
            B+'CONSENSUS.ENABLED': False, B+'DUAL_FREQUENCY.ENABLED': False,
            B+'DUAL_FREQUENCY.BASE_ONLY': False, B+'GUIDANCE.ENABLED': False,
            B+'GUIDANCE.BASE_ONLY': False, M+'ENABLED': True, M+'GAIN': -1.,
            M+'MODEL_PATH': args.models['safd']}
    matrix = [('baseline', {**base, M+'ENABLED': False}),
              ('zero', {**base, M+'GAIN': 0.})]
    matrix += [(name, {**base, M+'MODEL_PATH': path}) for name, path in args.models.items()]
    if set(supplied) & set(base):
        raise ValueError('Do not override matrix-managed model, gain, reference, or branch settings.')
    plan = shared.build_plan(args, matrix=matrix, launcher_path=Path(__file__))
    for run in plan:
        run['metric_artifact_hashes'] = artifacts
        run['fingerprint'] = common.sha256_json({'prior': run['fingerprint'], 'metric_artifacts': artifacts})
    return plan


def save_summary(path, args, rows, planned_runs):
    common.save_summary(path, args, rows, planned_runs)
    result = common.read_json(path)
    for row in result['runs']:
        if row['status'] not in ('completed', 'skipped'):
            continue
        metrics = common.load_completed_metrics(Path(row['output_dir']) / 'metrics.json')
        row['safd_metric'] = metrics.get('safd_metric')
        row['query_auxiliary_encodings'] = sum(s.get('query_auxiliary_encodings', 0) for s in metrics['sessions'])
        row['last_prediction_diagnostics'] = metrics['sessions'][-1].get('prediction_diagnostics')
    result['method'] = 'base_trained_frozen_safd_metric'
    result['training_cost_note'] = 'Cache extraction and base metric training are separate artifacts; add them when reporting total cost.'
    common.write_json(path, result)


def main(argv=None):
    args = parse_args(argv)
    try:
        plan = build_plan(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print('Invalid metric experiment inputs: ' + str(error), file=sys.stderr)
        return 2
    if args.summarize_only:
        rows = []
        for run in plan:
            metrics = common.completed_result(run)
            row = {key: run[key] for key in ('variant', 'seed', 'support_seed', 'output_dir', 'support_manifest', 'fingerprint')}
            row.update(status='skipped', summary=metrics['summary']) if metrics else row.update(status='missing_or_stale')
            rows.append(row)
        save_summary(Path(args.output_root) / 'summary.json', args, rows, len(plan))
        return 0
    if not args.execute:
        for run in plan:
            print(f'{run["variant"]} seed={run["seed"]}: ' + common.describe_command(run['command']))
        print(f'Dry run: {len(plan)} commands; no model evaluations started.')
        return 0
    for run in plan:
        directory = Path(run['output_dir'])
        if directory.exists() and any(directory.iterdir()) and not (directory / 'run_spec.json').is_file():
            print('Refusing unmanaged nonempty output directory: ' + str(directory), file=sys.stderr)
            return 2
    return common.execute_plan(plan, args, summary_writer=save_summary)


if __name__ == '__main__':
    raise SystemExit(main())
