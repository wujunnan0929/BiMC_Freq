"""Fit a base-only support-conditioned prototype estimator from an existing cache."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', required=True)
    p.add_argument('--output', required=True, help='Fresh .npz; JSON sidecar and training report are also exported.')
    p.add_argument('--method', choices=('center', 'boundary'), default='boundary')
    p.add_argument('--auxiliary', choices=('none', 'safd', 'fft', 'augmentation'), default='none')
    p.add_argument('--aux-cache', help='Row-aligned source cache; SAFD can reuse --cache when its source is safd.')
    p.add_argument('--auxiliary-probability', type=float, default=.25)
    p.add_argument('--seed', type=int, default=2)
    p.add_argument('--sequences', type=int, default=20)
    p.add_argument('--teacher-shot', type=int, default=15)
    p.add_argument('--query', type=int, default=5)
    p.add_argument('--old-shot', type=int, default=20)
    p.add_argument('--old-ways', type=int, nargs='+', default=[10, 20, 30])
    p.add_argument('--novel-ways', type=int, nargs='+', default=[5, 10])
    p.add_argument('--novel-shots', type=int, nargs='+', default=[2, 5])
    p.add_argument('--episode-pool', type=int, default=100)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--validate-every', type=int, default=50)
    p.add_argument('--early-checkpoints', type=int, nargs='+', default=[1, 5, 10, 20])
    p.add_argument('--hidden', type=int, default=32)
    p.add_argument('--max-delta', type=float, default=.15)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--weight-decay', type=float, default=.0001)
    p.add_argument('--center-weight', type=float, default=1.)
    p.add_argument('--margin-weight', type=float, default=1.)
    p.add_argument('--anchor-weight', type=float, default=.5)
    p.add_argument('--blend-grid', type=float, nargs='+', default=[0., .25, .5, 1.])
    p.add_argument('--maximum-group-drop-pp', type=float, default=.2)
    p.add_argument('--no-calibration', action='store_true')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--threads', type=int, default=1)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    path = Path(args.output)
    if path.suffix != '.npz' or any(p.exists() for p in (path, path.with_suffix('.json'), path.with_suffix('.training.json'))):
        raise ValueError('Choose a fresh .npz output; existing fitted artifacts are never overwritten.')
    if args.threads < 1:
        raise ValueError('threads must be positive.')
    import torch
    import yaml
    from yacs.config import CfgNode
    from tools.cache_safd_base_features import load_cache
    from engine.prototype_estimator_training import EstimatorOptions, fit_estimator, export_fitted_estimator, validate_options
    options = EstimatorOptions(**{k: getattr(args, k) for k in EstimatorOptions.__dataclass_fields__ if hasattr(args, k)},
                               calibrate=not args.no_calibration)
    validate_options(options)
    torch.set_num_threads(args.threads)
    arrays, metadata = load_cache(args.cache)
    auxiliary, aux_metadata = load_cache(args.aux_cache) if args.aux_cache else (None, None)
    cfg = CfgNode(yaml.safe_load(metadata['config_yaml']))
    cfg.freeze()
    model, report, cfg = fit_estimator(arrays, metadata, cfg, options, args.device, auxiliary, aux_metadata)
    saved = export_fitted_estimator(args.output, model, report, cfg, metadata, args.cache, arrays['sample_id'],
                                   args.aux_cache or (args.cache if args.auxiliary != 'none' else None))
    print('Frozen support estimator:', args.output, 'step=', saved['selected_step'], 'blend=', saved['blend'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
