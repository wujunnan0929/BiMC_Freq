"""Paired G0--G4 support-only frequency guidance experiments (dry run by default)."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from tools import run_frequency_uncertainty_experiments as shared
from utils.frequency_descriptions import (
    DATASET_SPECS, config_scalar, dataset_slug, dataset_class_names,
    description_path, load_explicit_descriptions, validate_ordinary_guidance_descriptions,
)

common = shared.common
G, B = 'TRAINER.BiMC.GUIDANCE.', 'TRAINER.BiMC.'


def variants_for_suite(suite):
    base = {
        B+'PREC': 'fp32', B+'VISION_CALIBRATION': True, B+'LAMBDA_I': .1,
        B+'TAU': 16, B+'TEXT_CALIBRATION': True, B+'LAMBDA_T': .5,
        B+'GAMMA_BASE': 1., B+'GAMMA_INC': 5., B+'USING_ENSEMBLE': True,
        shared.F+'ENABLED': False, shared.F+'ROUTER.ENABLED': False,
        shared.F+'BACKEND': 'safd', shared.F+'VIEW_MODE': 'natural',
        shared.U+'ENABLED': False, B+'DUAL_FREQUENCY.ENABLED': False,
        shared.C+'ENABLED': False, shared.R+'ENABLED': False, shared.R+'BASE_ONLY': False,
        shared.U+'BASE_ONLY': False, B+'DUAL_FREQUENCY.BASE_ONLY': False,
        G+'ENABLED': True, G+'MODE': 'paired', G+'VIEW_CONTROL': 'frequency',
        G+'SEMANTIC_SOURCE': 'explicit', G+'SEMANTIC_PERMUTATION': [0,1,2],
        G+'BAND_WEIGHTS': [1.,1.,1.], G+'AUTO_CALIBRATE': True,
        G+'BASE_ONLY': False,
    }
    definitions = [
        ('G0', {G+'ENABLED': False, B+'VISION_CALIBRATION': False, B+'LAMBDA_I': 0.}),
        ('G1', {G+'ENABLED': False}),
        ('G2', {G+'MODE': 'visual'}),
        ('G3', {G+'MODE': 'marginal'}),
        ('G4', {}),
    ]
    if suite == 'all':
        definitions.extend([
            ('G4_uniform', {G+'SEMANTIC_SOURCE': 'uniform'}),
            ('G4_permuted', {G+'SEMANTIC_PERMUTATION': [2,0,1]}),
            ('G4_ordinary', {G+'SEMANTIC_SOURCE': 'ordinary'}),
            ('G4_repeat', {G+'VIEW_CONTROL': 'repeat'}),
            ('G4_augmentation', {G+'VIEW_CONTROL': 'augmentation'}),
            ('G4_low', {G+'BAND_WEIGHTS': [1.,0.,0.]}),
            ('G4_middle', {G+'BAND_WEIGHTS': [0.,1.,0.]}),
            ('G4_high', {G+'BAND_WEIGHTS': [0.,0.,1.]}),
            ('G4_fft', {shared.F+'BACKEND': 'fft'}),
        ])
    elif suite != 'core':
        raise ValueError('Unknown guidance suite: ' + suite)
    return [(name, {**base, **changes}) for name, changes in definitions]


def build_plan(args):
    matrix = variants_for_suite(args.suite)
    managed = shared.MANAGED | {key for _, overrides in matrix for key in overrides}
    if len(args.opts) % 2 or len(set(args.opts[::2])) != len(args.opts[::2]):
        raise ValueError('--opts needs distinct KEY VALUE pairs, placed last.')
    collision = set(args.opts[::2]) & managed
    if collision:
        raise ValueError('Matrix-managed keys cannot be overridden: ' + ', '.join(sorted(collision)))
    plan = shared.build_plan(args, matrix=matrix, launcher_path=Path(__file__))
    protocol_path = REPO_ROOT / 'configs/experiments/frequency_guidance_protocol.json'
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    for run in plan:
        # The shared framework also controls a disabled GDA calibration seed.
        run['command'].extend([G+'CALIBRATION_SEED', str(run['seed'] + 10101)])
        run['protocol_sha256'] = protocol_hash
        run['fingerprint'] = common.sha256_json({'prior': run['fingerprint'],
                                                'command': run['command'], 'protocol': protocol_hash})
    return plan


def validate_plan_inputs(args, plan):
    options = dict(zip(args.opts[::2], args.opts[1::2]))
    dataset = dataset_slug(options.get('DATASET.NAME') or config_scalar(args.data_cfg, 'NAME'))
    class_names = dataset_class_names(dataset)
    count = int(options.get(G+'CANDIDATES_PER_BAND', config_scalar(args.train_cfg, 'CANDIDATES_PER_BAND', '5')))
    if count < 1:
        raise ValueError('CANDIDATES_PER_BAND must be positive.')
    ordinary_path = options.get('DATASET.GPT_PATH') or config_scalar(args.data_cfg, 'GPT_PATH')
    ordinary = json.loads(Path(ordinary_path).read_text(encoding='utf-8'))
    # Every group uses the unchanged ordinary BiMC description bank.
    from utils.frequency_descriptions import normalize_name
    normalized = {normalize_name(k): v for k,v in ordinary.items()}
    for name in class_names:
        values = normalized.get(normalize_name(name))
        if not isinstance(values, list) or not values or not all(isinstance(v,str) and v.strip() for v in values):
            raise ValueError('Missing/invalid ordinary BiMC descriptions for class: ' + name)
    needs_explicit, needs_ordinary_control = False, False
    for run in plan:
        cmd = run['command']
        opts = dict(zip(cmd[cmd.index('--opts')+1::2], cmd[cmd.index('--opts')+2::2]))
        active = opts[G+'ENABLED'] == 'True' and opts[G+'MODE'] != 'visual'
        needs_explicit |= active and opts[G+'SEMANTIC_SOURCE'] == 'explicit'
        needs_ordinary_control |= active and opts[G+'SEMANTIC_SOURCE'] == 'ordinary'
    report = {'dataset': dataset, 'ordinary_classes': len(class_names)}
    if needs_ordinary_control:
        report['ordinary_control'] = validate_ordinary_guidance_descriptions(ordinary_path, class_names, count)
    if needs_explicit:
        path = description_path(dataset, options.get(shared.F+'EXPLICIT_DESCRIPTION_PATH',
                                                     config_scalar(args.train_cfg, 'EXPLICIT_DESCRIPTION_PATH', 'auto')))
        candidates = load_explicit_descriptions(path, class_names, exact=True, dataset=dataset)
        for prompts in candidates:
            for band in ('low','middle','high'):
                if len(prompts[band]) < count or len({v.strip().lower() for v in prompts[band][:count]}) != count:
                    raise ValueError(f'Guidance needs at least {count} unique descriptions per class/band: {path}')
        report['scale_text'] = {'path': str(path), 'classes': len(candidates), 'candidates_per_band': count}
    return report


def save_summary(path, args, rows, planned_runs):
    common.save_summary(path, args, rows, planned_runs)
    report = common.read_json(path)
    complete = [row for row in report['runs'] if row['status'] in ('completed','skipped')]
    extra = ('novel_to_original_base_rate', 'original_base_to_novel_rate')
    for row in complete:
        metrics = common.load_completed_metrics(Path(row['output_dir']) / 'metrics.json')
        sessions = metrics['sessions']
        row['guidance'] = {key: (metrics.get('guidance') or {}).get(key)
                           for key in ('selected_rho','protocol_sha256','selection_status','fallback_to_G1')}
        row['cost'] = {key: sum(s.get(key,0.) for s in sessions) for key in (
            'support_build_seconds','guidance_calibration_seconds','query_encoding_seconds',
            'query_classification_seconds','query_original_encodings','query_auxiliary_encodings')}
        row['cost'].update(total_wall_seconds=metrics.get('total_wall_seconds'),
                           peak_gpu_memory_bytes=metrics.get('peak_gpu_memory_bytes'),
                           retained_tensor_bytes=sessions[-1].get('retained_tensor_bytes'))
        row['clip_weight_sha256'] = metrics.get('clip_weight_sha256')
        row['summary'].update({key: sessions[-1].get(key) for key in extra})
    for name in report['aggregates']:
        group = [row for row in complete if row['variant'] == name]
        for key in extra:
            report['aggregates'][name]['metrics'][key] = common.metric_statistics([r['summary'].get(key) for r in group])
        report['aggregates'][name]['cost'] = {
            key: common.metric_statistics([r['cost'].get(key) for r in group])
            for key in ('support_build_seconds','guidance_calibration_seconds','query_encoding_seconds',
                        'query_classification_seconds','total_wall_seconds','peak_gpu_memory_bytes','retained_tensor_bytes')}
    report['paired_comparisons'] = {}
    ablations = sorted({r['variant'] for r in complete if r['variant'].startswith('G4_')})
    for ref in ('G0','G1','G2','G3',*ablations):
        references = {r['seed']: r for r in complete if r['variant'] == ref}
        paired = [r for r in complete if r['variant'] == 'G4' and r['seed'] in references]
        report['paired_comparisons']['G4_minus_'+ref] = {
            key: common.metric_statistics([r['summary'][key]-references[r['seed']]['summary'][key]
                                           for r in paired if r['summary'].get(key) is not None
                                           and references[r['seed']]['summary'].get(key) is not None])
            for key in (*common.METRIC_NAMES,*extra)}
    report['pairing_audit'] = {}
    for seed in sorted({r['seed'] for r in complete}):
        group = [r for r in complete if r['seed'] == seed]
        supports = {r['variant']: hashlib.sha256((Path(r['output_dir'])/'support.json').read_bytes()).hexdigest() for r in group}
        protocols = {r['variant']: r['guidance']['protocol_sha256'] for r in group
                     if r['guidance']['selection_status'] == 'base_class_disjoint_validation'}
        report['pairing_audit'][str(seed)] = {'support_hashes': supports,
            'support_match': len(set(supports.values())) == 1, 'calibration_protocols': protocols,
            'calibration_protocol_match': len(set(protocols.values())) == 1 if protocols else None,
            'query_auxiliary_encodings': {r['variant']:r['cost']['query_auxiliary_encodings'] for r in group}}
    report.update(method='support_only_frequency_guidance',
                  selection_policy='base_training_class_disjoint_pseudo_incremental_validation_only')
    common.write_json(path, report)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=tuple(DATASET_SPECS), default='cub200')
    parser.add_argument('--data-cfg')
    parser.add_argument('--train-cfg', default=str(REPO_ROOT/'configs/trainers/bimc_frequency_guidance.yaml'))
    parser.add_argument('--output-root')
    parser.add_argument('--support-root')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--suite', choices=('core','all'), default='core')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1,2,3])
    parser.add_argument('--variants', nargs='+')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--rerun', action='store_true')
    parser.add_argument('--keep-going', action='store_true')
    parser.add_argument('--opts', nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args(argv)
    args.data_cfg = args.data_cfg or str(REPO_ROOT/'configs/datasets'/(args.dataset+'.yaml'))
    supplied = dict(zip(args.opts[::2], args.opts[1::2]))
    if dataset_slug(supplied.get('DATASET.NAME') or config_scalar(args.data_cfg,'NAME')) != args.dataset:
        parser.error('--dataset disagrees with DATASET.NAME.')
    args.output_root = args.output_root or str(REPO_ROOT/'outputs'/('frequency_guidance_'+args.dataset))
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        plan = build_plan(args)
        if args.summarize_only:
            rows = []
            for run in plan:
                metrics = common.completed_result(run)
                row = {k:run[k] for k in ('variant','seed','support_seed','output_dir','support_manifest','fingerprint')}
                row.update(status='skipped',summary=metrics['summary']) if metrics else row.update(status='missing_or_stale')
                rows.append(row)
            save_summary(Path(args.output_root)/'summary.json',args,rows,len(plan))
            return 0
        checked = validate_plan_inputs(args, plan)
    except (ValueError, OSError, KeyError) as error:
        print('Invalid guidance experiment inputs: '+str(error), file=sys.stderr)
        return 2
    print('Guidance preflight passed:', checked, flush=True)
    if not args.execute:
        for run in plan:
            print(f'{run["variant"]} seed={run["seed"]}: '+common.describe_command(run['command']))
        print(f'Dry run: {len(plan)} commands; no experiments started.')
        return 0
    for run in plan:
        directory = Path(run['output_dir'])
        if directory.exists() and any(directory.iterdir()) and not (directory/'run_spec.json').is_file():
            print('Refusing unmanaged nonempty output directory: '+str(directory), file=sys.stderr)
            return 2
    return common.execute_plan(plan,args,summary_writer=save_summary)


if __name__ == '__main__':
    raise SystemExit(main())
