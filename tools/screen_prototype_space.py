"""Screen calibration, prototype-estimation and expert headroom from base caches."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--sequences', type=int, default=20)
    p.add_argument('--teacher-shot', type=int, default=15)
    p.add_argument('--maximum-group-drop-pp', type=float, default=.2)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--threads', type=int, default=1)
    a = p.parse_args(argv)
    import math
    if min(a.seed, a.sequences, a.teacher_shot, a.threads) < 1:
        p.error('seed/sequences/teacher-shot/threads must be positive.')
    if not math.isfinite(a.maximum_group_drop_pp) or a.maximum_group_drop_pp < 0:
        p.error('maximum-group-drop-pp must be finite and nonnegative.')
    return a


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.output_dir)
    if out.exists():
        raise ValueError('Choose a fresh output directory; existing screening results are not overwritten.')
    import torch
    import yaml
    from yacs.config import CfgNode
    from tools.cache_safd_base_features import load_cache
    from engine.prototype_evaluation import (prepare_context, validation_records, evaluate_records,
        score_record, select_calibration, identity_calibration, expert_oracle_scores,
        description_count_scores, prototype_factorial_diagnostics, expert_profile_diagnostics)
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    arrays, meta = load_cache(args.cache)
    if not meta.get('complete_base_bank') or not meta.get('has_text_features'):
        raise ValueError('Screening requires the complete original base cache and ordinary text features.')
    cfg = CfgNode(yaml.safe_load(meta['config_yaml']))
    cfg.freeze()
    cfg, protocol, _ = prepare_context(arrays, cfg, args.seed, args.sequences, args.teacher_shot)
    out.mkdir(parents=True)
    write_json(out / 'protocol.json', protocol)
    write_json(out / 'provenance.json', {'arguments': vars(args), 'cache_sha256': meta['cache_sha256'],
        'cache_metadata_payload_sha256': meta['metadata_payload_sha256'], 'runtime_torch': torch.__version__,
        'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
            ('engine/prototype_evaluation.py', 'tools/screen_prototype_space.py', 'models/bimc.py')}})
    try:
        records = validation_records(arrays, cfg, protocol, args.device)
        baseline = evaluate_records(records, lambda r: r['reference_votes'])
        print('Original BiMC:', baseline, flush=True)
        calibration, calibration_report = select_calibration(records, cfg, args.maximum_group_drop_pp)
        write_json(out / 'calibration.json', calibration_report)
        print('Calibrated reference:', calibration_report['selected_validation'], flush=True)
        group_grid = [{**identity_calibration(), 'novel_bias': b} for b in (0., -.2, -.1, -.05, .05, .1, .2)]
        _, group_report = select_calibration(records, cfg, args.maximum_group_drop_pp, grid=group_grid)
        write_json(out / 'group_bias_control.json', group_report)
        teacher = evaluate_records(records, lambda r: score_record(
            r, cfg, image_prototypes=r['teacher_prototypes'], calibration=calibration))
        expert = evaluate_records(records, lambda r: expert_oracle_scores(r, cfg))
        prototype_controls = prototype_factorial_diagnostics(records, cfg, calibration)
        expert_profiles = expert_profile_diagnostics(records, cfg, calibration)
        write_json(out / 'prototype_controls.json', prototype_controls)
        write_json(out / 'expert_profiles.json', expert_profiles)
        print('Prototype controls:', {name: row['metrics'] for name, row in
                                     prototype_controls['variants'].items()}, flush=True)
        print('Individual experts:', {name: row['metrics'] for name, row in
                                     expert_profiles['experts'].items()}, flush=True)
        counts = [int((arrays['description_targets'] == c).sum()) for c in meta['base_class_ids']]
        equal_count = evaluate_records(records, lambda r: description_count_scores(r, cfg, min(counts)))
        equal_count_calibrated = evaluate_records(records, lambda r: description_count_scores(
            r, cfg, min(counts), calibration))
        result = {'status': 'completed', 'report_version': 2, 'dataset': meta['dataset'],
            'screening_seed': args.seed,
            'scope': 'base_training_class_disjoint_screening_only', 'benchmark_test_evaluation': False,
            'new_CLIP_or_SAFD_encodings': 0, 'baseline': baseline,
            'calibrated_reference': calibration_report['selected_validation'],
            'selected_calibration': calibration, 'calibration_candidate_count': calibration_report['candidate_count'],
            'group_bias_only': group_report['selected_validation'],
            'teacher_prototype_diagnostic': {'metrics': teacher, 'uses_extra_labeled_teacher_images': True,
                'deployable': False, 'selection_candidate': False, 'strict_mathematical_upper_bound': False,
                'teacher_variant': 'teacher_raw', 'visual_guidance_applied_to_teacher': False,
                'reference_visual_guidance_enabled': bool(cfg.TRAINER.BiMC.VISION_CALIBRATION),
                'comparison_changes_bank_and_guidance': bool(cfg.TRAINER.BiMC.VISION_CALIBRATION)},
            'expert_union_diagnostic': {'metrics': expert, 'uses_query_labels_to_choose_expert': True,
                'experts': ['original_BiMC', 'fused_prototype', 'covariance', 'description_max'],
                'deployable': False, 'selection_candidate': False},
            'prototype_factorial_diagnostics': prototype_controls,
            'expert_profile_diagnostics': expert_profiles,
            'description_counts': {'min': min(counts), 'max': max(counts)},
            'equal_description_count_control': {'per_class_count': min(counts),
                'uncalibrated_metrics': equal_count, 'calibrated_metrics': equal_count_calibrated,
                'description_proto_unchanged': True, 'selection_candidate': False,
                'rule': 'evenly_spaced_in_existing_text_order; no_query_or_image_grounding'},
            'teacher_shot': args.teacher_shot, 'protocol_sha256': protocol['sha256'],
            'wall_seconds': time.perf_counter() - started,
            'limitations': ['Validation stages share images and classes and are not independent statistical repeats.',
                'Calibration is selected on this bank; its selected metrics are development scores.',
                'Teacher/expert diagnostics use privileged information and cannot be reported as deployable results.']}
        write_json(out / 'summary.json', result)
        print('Saved screening:', out / 'summary.json', flush=True)
    except Exception as exc:
        write_json(out / 'failure.json', {'status': 'failed', 'error': str(exc)})
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
