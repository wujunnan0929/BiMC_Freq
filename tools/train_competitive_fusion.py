"""Select fixed expert weights and fit a base-only frozen competition module."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', required=True)
    p.add_argument('--output', required=True, help='Fresh model.npz; JSON, training and screening reports also saved.')
    p.add_argument('--method', choices=('group', 'joint'), default='joint')
    p.add_argument('--no-support-quality', action='store_true')
    p.add_argument('--auxiliary', choices=('none', 'safd', 'fft', 'augmentation'), default='none')
    p.add_argument('--aux-cache')
    p.add_argument('--anchor-model', help='Reuse the fixed/calibrated anchor from the identical cache and class split.')
    p.add_argument('--auxiliary-probability', type=float, default=.25)
    p.add_argument('--seed', type=int, default=2)
    p.add_argument('--sequences', type=int, default=20)
    p.add_argument('--teacher-shot', type=int, default=15)
    p.add_argument('--query', type=int, default=5)
    p.add_argument('--old-shot', type=int, default=20)
    p.add_argument('--old-ways', type=int, nargs='+', default=[10, 20, 30])
    p.add_argument('--novel-ways', type=int, nargs='+', default=[5, 10, 20])
    p.add_argument('--novel-shots', type=int, nargs='+', default=[2, 5])
    p.add_argument('--episode-pool', type=int, default=100)
    p.add_argument('--steps', type=int, default=300, help='0 selects fixed weights and exports an exact zero learned module.')
    p.add_argument('--validate-every', type=int, default=50)
    p.add_argument('--early-checkpoints', type=int, nargs='+', default=[1, 5, 10, 20])
    p.add_argument('--hidden', type=int, default=16)
    p.add_argument('--group-limit', type=float, default=2.)
    p.add_argument('--mix-limit', type=float, default=.5)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--weight-decay', type=float, default=.0001)
    p.add_argument('--group-weight', type=float, default=.5)
    p.add_argument('--protection-weight', type=float, default=.5)
    p.add_argument('--distillation-weight', type=float, default=.1)
    p.add_argument('--blend-grid', type=float, nargs='+', default=[0., .25, .5, 1.])
    p.add_argument('--weight-grid', type=float, nargs='+', default=[0., .25, .5, .75, .8, 1.])
    p.add_argument('--bias-grid', type=float, nargs='+', default=[0., -.2, -.1, .1, .2])
    p.add_argument('--maximum-group-drop-pp', type=float, default=.2)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--threads', type=int, default=1)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    path = Path(args.output)
    if path.suffix != '.npz' or any(p.exists() for p in (path, path.with_suffix('.json'),
            path.with_suffix('.training.json'), path.with_suffix('.screening.json'))):
        raise ValueError('Choose a fresh .npz output; fitted artifacts are never overwritten.')
    if args.threads < 1:
        raise ValueError('threads must be positive.')
    import torch
    import yaml
    from yacs.config import CfgNode
    from tools.cache_safd_base_features import load_cache
    from engine.competitive_training import FusionOptions, fit_competition, export_fitted_fusion, validate_options
    options = FusionOptions(**{k: getattr(args, k) for k in FusionOptions.__dataclass_fields__ if hasattr(args, k)},
                            use_support_quality=not args.no_support_quality)
    validate_options(options)
    torch.set_num_threads(args.threads)
    arrays, metadata = load_cache(args.cache)
    auxiliary, aux_metadata = load_cache(args.aux_cache) if args.aux_cache else (None, None)
    cfg = CfgNode(yaml.safe_load(metadata['config_yaml'])); cfg.freeze()
    anchor = None
    if args.anchor_model:
        from engine.competitive_runtime import FORMAT, validate_metadata
        anchor_path = Path(args.anchor_model)
        anchor = json.loads(anchor_path.with_suffix('.json').read_text(encoding='utf-8'))
        validate_metadata(anchor)
        if anchor.get('format') != FORMAT or anchor.get('model_sha256') != hashlib.sha256(anchor_path.read_bytes()).hexdigest():
            raise ValueError('Fixed anchor format/checksum mismatch.')
    model, report, cfg = fit_competition(arrays, metadata, cfg, options, args.device, auxiliary, aux_metadata, anchor)
    saved = export_fitted_fusion(path, model, report, cfg, metadata, args.cache, arrays['sample_id'],
        args.aux_cache or (args.cache if args.auxiliary != 'none' else None))
    screening = {'status': 'completed', 'dataset': metadata['dataset'], 'screening_seed': args.seed,
        'benchmark_test_evaluation': False, 'new_CLIP_or_SAFD_encodings': 0,
        'protocol_sha256': report['selection_protocol']['sha256'],
        **{k: report[k] for k in ('reference', 'calibrated_reference', 'fixed_reference', 'calibration', 'fixed',
                                   'selected_validation', 'selected_step', 'blend', 'group_diagnostics')},
        'fixed_candidate_count': report['fixed_report']['candidate_count'],
        'reused_fixed_anchor': args.anchor_model,
        'learned_H_delta_vs_fixed_pp': report['selected_validation']['harmonic_accuracy']-report['fixed_reference']['harmonic_accuracy'],
        'uses_diagnostic_teachers_for_selection': False}
    path.with_suffix('.screening.json').write_text(json.dumps(screening, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print('Frozen competitive fusion:', path, 'step=', saved['selected_step'], 'blend=', saved['blend'],
          'fixed=', saved['fixed'], flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
