"""Paired original/calibrated/fixed/learned FSCIL comparisons; dry-run by default."""
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
P = B+'COMPETITIVE_FUSION.'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=('cub200', 'cifar100', 'miniimagenet'), default='cub200')
    p.add_argument('--model', action='append', required=True, help='NAME=PATH.npz; repeat for matched controls.')
    p.add_argument('--reference-model')
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
    for item in a.model:
        if '=' not in item:
            p.error('--model must be NAME=PATH.npz')
        name, path = item.split('=', 1)
        if not re.fullmatch('[a-z][a-z0-9_]*', name) or name in ('baseline', 'zero_full', 'calibrated', 'fixed') or name in a.models:
            p.error('Use distinct names; baseline/zero_full/calibrated/fixed are reserved.')
        a.models[name] = str(Path(path).resolve())
    a.reference_model = a.reference_model or next(iter(a.models))
    if a.reference_model not in a.models:
        p.error('--reference-model must name a supplied model.')
    a.data_cfg = str(ROOT / f'configs/datasets/{a.dataset}.yaml')
    a.train_cfg = str(ROOT / 'configs/trainers/bimc_competitive_fusion.yaml')
    a.output_root = a.output_root or str(ROOT / f'outputs/competitive_fusion_{a.dataset}')
    a.suite, a.rerun = 'core', False
    return a


def check_models(args):
    metadata, artifacts = {}, {}
    for name, supplied in args.models.items():
        path = Path(supplied)
        if path.suffix != '.npz':
            raise ValueError('Fusion artifact must end in .npz.')
        sidecar = path.with_suffix('.json')
        d = json.loads(sidecar.read_text(encoding='utf-8'))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if d.get('model_sha256') != digest:
            raise ValueError('Fusion checksum mismatch: '+name)
        if d.get('format') != 'competitive_fusion_v1' or d.get('data_source') != 'base_training_only':
            raise ValueError('Use an exported base-only competitive fusion artifact.')
        if dataset_slug(d.get('dataset', '')) != args.dataset:
            raise ValueError('Fusion dataset mismatch.')
        if not isinstance(d.get('fitting_seed'), int) or not d.get('selection_protocol_sha256'):
            raise ValueError('Fusion fitting seed/protocol missing.')
        metadata[name] = d
        artifacts[name] = {'model': digest, 'sidecar': hashlib.sha256(sidecar.read_bytes()).hexdigest()}
    reference = metadata[args.reference_model]
    for name, d in metadata.items():
        for key in ('cache_sha256', 'clip_weight_sha256', 'base_sample_ids_sorted_sha256', 'reference_settings',
                    'calibration', 'fixed', 'ordinary_description_sha256', 'text_template', 'support_transform',
                    'input_provenance', 'fitting_seed', 'selection_protocol_sha256'):
            if d.get(key) != reference.get(key):
                raise ValueError(f'Models must share reference inputs/selection ({key} differs for {name}). Compare fitting seeds separately.')
    return artifacts


def build_plan(args):
    if len(args.opts) % 2 or len(set(args.opts[::2])) != len(args.opts[::2]):
        raise ValueError('--opts must contain distinct KEY VALUE pairs and be last.')
    artifacts = check_models(args)
    reference = common.read_json(Path(args.models[args.reference_model]).with_suffix('.json'))['reference_settings']
    base = {B+'PREC': 'fp32', B+'VISION_CALIBRATION': True, B+'LAMBDA_I': .1, B+'TAU': 16,
        B+'TEXT_CALIBRATION': True, B+'LAMBDA_T': .5, B+'USING_ENSEMBLE': True,
        B+'GAMMA_BASE': 1., B+'GAMMA_INC': 5., B+'FREQUENCY.ENABLED': False,
        B+'FREQUENCY.ROUTER.ENABLED': False, B+'RESIDUAL.ENABLED': False, B+'RESIDUAL.BASE_ONLY': False,
        B+'CONSENSUS.ENABLED': False, B+'UNCERTAINTY.ENABLED': False, B+'UNCERTAINTY.BASE_ONLY': False,
        B+'DUAL_FREQUENCY.ENABLED': False, B+'DUAL_FREQUENCY.BASE_ONLY': False,
        B+'GUIDANCE.ENABLED': False, B+'GUIDANCE.BASE_ONLY': False, B+'SAFD_METRIC.ENABLED': False,
        B+'PROTOTYPE_ESTIMATOR.ENABLED': False, P+'ENABLED': True,
        P+'MODEL_PATH': args.models[args.reference_model], P+'MODE': 'learned', P+'BLEND': -1.}
    for key, cfg_key in {'vision_calibration': 'VISION_CALIBRATION', 'lambda_i': 'LAMBDA_I', 'tau': 'TAU',
                         'text_calibration': 'TEXT_CALIBRATION', 'lambda_t': 'LAMBDA_T',
                         'using_ensemble': 'USING_ENSEMBLE', 'gamma_base': 'GAMMA_BASE', 'gamma_inc': 'GAMMA_INC'}.items():
        base[B+cfg_key] = reference[key]
    # The existing YACS schema declares TAU as an integer.
    if float(reference['tau']) != int(reference['tau']):
        raise ValueError('This BiMC configuration requires an integer TAU.')
    base[B+'TAU'] = int(reference['tau'])
    base.update({'DATASET.BETA': reference['beta'], 'DATASET.ENSEMBLE_ALPHA': reference['ensemble_alpha']})
    supplied = dict(zip(args.opts[::2], args.opts[1::2]))
    if set(supplied) & set(base):
        raise ValueError('Do not override matrix-managed reference/model/branch options.')
    if dataset_slug(supplied.get('DATASET.NAME', args.dataset)) != args.dataset:
        raise ValueError('--dataset and DATASET.NAME disagree.')
    matrix = [('baseline', {**base, P+'ENABLED': False}), ('zero_full', {**base, P+'MODE': 'zero'}),
              ('calibrated', {**base, P+'MODE': 'calibrated'}), ('fixed', {**base, P+'MODE': 'fixed'})]
    matrix += [(name, {**base, P+'MODEL_PATH': path}) for name, path in args.models.items()]
    plan = shared.build_plan(args, matrix=matrix, launcher_path=Path(__file__))
    for run in plan:
        run['competitive_artifact_hashes'] = artifacts
        run['fingerprint'] = common.sha256_json({'prior': run['fingerprint'], 'fusion_artifacts': artifacts})
    return plan


def save_summary(path, args, rows, planned_runs):
    common.save_summary(path, args, rows, planned_runs)
    result = common.read_json(path)
    result['method'] = 'base_trained_frozen_competitive_fusion'
    result['reference_model'] = args.reference_model
    result['cost_note'] = 'Offline cache, selection and fresh refit costs are separate from incremental evaluation.'
    for row in result['runs']:
        if row['status'] in ('completed', 'skipped'):
            metrics = common.load_completed_metrics(Path(row['output_dir'])/'metrics.json')
            row['competitive_fusion'] = metrics.get('competitive_fusion')
    for anchor in ('calibrated', 'fixed'):
        reference = {r['seed']: r for r in result['runs'] if r['variant'] == anchor and r['status'] in ('completed', 'skipped')}
        paired = {}
        for name in args.models:
            group = [r for r in result['runs'] if r['variant'] == name and r['status'] in ('completed', 'skipped')]
            paired[name] = {key: common.metric_statistics([r['summary'][key]-reference[r['seed']]['summary'][key]
                for r in group if r['seed'] in reference and r['summary'][key] is not None
                and reference[r['seed']]['summary'][key] is not None]) for key in common.METRIC_NAMES}
        result['paired_delta_vs_'+anchor] = paired
    common.write_json(path, result)


def main(argv=None):
    args = parse_args(argv)
    try:
        plan = build_plan(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print('Invalid competitive fusion inputs: '+str(exc), file=sys.stderr)
        return 2
    if args.summarize_only:
        rows = []
        for run in plan:
            metrics = common.completed_result(run)
            row = {k: run[k] for k in ('variant', 'seed', 'support_seed', 'output_dir', 'support_manifest', 'fingerprint')}
            row.update(status='skipped', summary=metrics['summary']) if metrics else row.update(status='missing_or_stale')
            rows.append(row)
        save_summary(Path(args.output_root)/'summary.json', args, rows, len(plan))
        return 0
    if not args.execute:
        for run in plan:
            print(f"{run['variant']} seed={run['seed']}: "+common.describe_command(run['command']))
        print(f'Dry run: {len(plan)} evaluations; none started.')
        return 0
    for run in plan:
        folder = Path(run['output_dir'])
        if folder.exists() and any(folder.iterdir()) and not (folder/'run_spec.json').is_file():
            print('Refusing unmanaged nonempty output: '+str(folder), file=sys.stderr)
            return 2
    return common.execute_plan(plan, args, summary_writer=save_summary)


if __name__ == '__main__':
    raise SystemExit(main())
