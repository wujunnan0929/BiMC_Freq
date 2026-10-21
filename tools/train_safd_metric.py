"""Fit a small SAFD-informed metric from a frozen base feature cache (no CLIP load)."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', required=True)
    p.add_argument('--output', required=True, help='Fresh .npz output; JSON sidecar and training report are also written.')
    p.add_argument('--basis', choices=('explained', 'original_within'), default='explained')
    p.add_argument('--method', choices=('learned', 'closed_form'), default='learned')
    p.add_argument('--strength-grid', type=float, nargs='+', default=[0., .1, 1., 10.],
                   help='Closed-form control only; must include zero.')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--validate-every', type=int, default=100)
    p.add_argument('--episode-pool', type=int, default=100)
    p.add_argument('--sequences', type=int, default=20)
    p.add_argument('--rank', type=int, default=64)
    p.add_argument('--ridge', type=float, default=.1)
    p.add_argument('--lr', type=float, default=.01)
    p.add_argument('--weight-decay', type=float, default=.001)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--old-way', type=int, default=10)
    p.add_argument('--novel-way', type=int, default=5)
    p.add_argument('--old-shot', type=int, default=10)
    p.add_argument('--query', type=int, default=5)
    p.add_argument('--maximum-group-drop-pp', type=float, default=.2)
    p.add_argument('--device', default='cpu', help='cpu or cuda; model weights are never loaded.')
    p.add_argument('--allow-smoke-cache', action='store_true', help='Allow incomplete feature banks for wiring tests, not publication runs.')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output)
    if output.suffix != '.npz':
        raise ValueError('Metric output must end in .npz.')
    if any(p.exists() for p in (output, output.with_suffix('.json'), output.with_suffix('.training.json'))):
        raise ValueError('Refusing to overwrite an existing fitted metric; choose a new --output.')
    import yaml
    from yacs.config import CfgNode
    from tools.cache_safd_base_features import load_cache
    from engine.safd_metric_training import TrainingOptions, fit_metric, export_metric
    arrays, metadata = load_cache(args.cache)
    if not metadata.get('complete_base_bank') and not args.allow_smoke_cache:
        raise ValueError('Production metric fitting requires the complete original base bank. Use --allow-smoke-cache only for wiring tests.')
    required = ('text_features', 'description_features', 'description_targets', 'description_proto')
    if not all(name in arrays for name in required):
        raise ValueError('Metric training needs a cache with ordinary BiMC text features.')
    cfg = CfgNode(yaml.safe_load(metadata['config_yaml']))
    cfg.freeze()
    options = TrainingOptions(**{key: getattr(args, key) for key in (
        'steps', 'validate_every', 'episode_pool', 'rank', 'ridge', 'lr', 'weight_decay', 'seed',
        'sequences', 'old_way', 'novel_way', 'old_shot', 'query', 'maximum_group_drop_pp', 'basis',
        'method', 'strength_grid')})
    adapter, report, cfg = fit_metric(arrays, cfg, options, args.device)
    saved = export_metric(args.output, adapter, report, cfg, metadata, args.cache, arrays['sample_id'])
    print('Frozen metric:', args.output, 'selected_step=', saved['selected_step'], 'gain=', saved['gain'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
