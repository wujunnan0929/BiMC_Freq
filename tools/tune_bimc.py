"""Select BiMC parameters on base-only validation episodes from a CLIP bank.

python tools/tune_bimc.py --dataset cub200 --bank banks/cub200_vitb16 --dry-run
python tools/tune_bimc.py --dataset cub200 --bank banks/cub200_vitb16 --shot 1

The search never evaluates the benchmark test set. The exported best.yaml is
a manual profile accepted by main.py --retune_cfg for a later fixed evaluation.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import setup_cfg
from utils.bimc_retune import check_output_directory, prepare_retune_config


# Public config names are shared with the manual entry, avoiding a second set
# of lambda/beta conventions. The cached pipeline has been used for parity in
# the existing fdrfreq experiments.
PARAMETERS = {
    'DATASET.BETA': 'beta',
    'TRAINER.BiMC.LAMBDA_T': 'lambda_t',
    'TRAINER.BiMC.LAMBDA_I': 'holistic.lam',
    'TRAINER.BiMC.TAU': 'holistic.tau',
    'DATASET.ENSEMBLE_ALPHA': 'ensemble.alpha',
    'TRAINER.BiMC.GAMMA_BASE': 'ensemble.gamma_base',
    'TRAINER.BiMC.GAMMA_INC': 'ensemble.gamma_inc',
}


def get_value(cfg, path):
    for key in path.split('.'):
        cfg = cfg[key]
    return cfg


def set_value(cfg, path, value):
    keys = path.split('.')
    for key in keys[:-1]:
        cfg = cfg[key]
    cfg[keys[-1]] = value


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(name + ' must be a positive integer.')


def load_search(filename):
    import yaml
    with open(filename, encoding='utf-8') as handle:
        search = yaml.safe_load(handle)
    if not isinstance(search, dict):
        raise ValueError('Search YAML must be a mapping.')
    expected = {'objective', 'split_mode', 'split_seeds', 'sequences', 'grid'}
    if set(search) - expected:
        raise ValueError('Unknown search settings: ' + ', '.join(sorted(set(search) - expected)))
    if search.get('objective') not in ('hm', 'last', 'avg'):
        raise ValueError('objective must be hm, last or avg.')
    if search.get('split_mode') not in ('contiguous', 'random'):
        raise ValueError('split_mode must be contiguous or random.')
    seeds = search.get('split_seeds')
    if (not isinstance(seeds, list) or not seeds
            or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('split_seeds must be a nonempty list of distinct nonnegative integers.')
    _positive_integer('sequences', search.get('sequences'))
    grid = search.get('grid')
    if not isinstance(grid, dict):
        raise ValueError('grid must be a mapping (use {} to validate a fixed configuration).')
    for key, values in grid.items():
        if key not in PARAMETERS:
            raise ValueError('Unsupported search parameter: ' + str(key))
        if not isinstance(values, list) or not values:
            raise ValueError(key + ' requires a nonempty list of candidate values.')
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) for v in values):
            raise ValueError(key + ' candidates must be finite numbers.')
        if len(set(values)) != len(values):
            raise ValueError(key + ' contains duplicate candidates.')
    return search


def pipeline_config(cfg):
    b = cfg.TRAINER.BiMC
    return {
        'classifier': 'bimc', 'beta': cfg.DATASET.BETA,
        'lambda_t': b.LAMBDA_T if b.TEXT_CALIBRATION else 0.0,
        'holistic': {'donor': 'visual',
                     'lam': b.LAMBDA_I if b.VISION_CALIBRATION else 0.0,
                     'tau': b.TAU},
        'ensemble': {'alpha': cfg.DATASET.ENSEMBLE_ALPHA,
                     'gamma_base': b.GAMMA_BASE, 'gamma_inc': b.GAMMA_INC}
                    if b.USING_ENSEMBLE else None,
        'segments': None, 'anchor': None, 'ula': None, 'novel_bias': 0.0,
    }


def validate_grid(cfg, grid):
    # Validate each scalar through the same range checks as the manual entry.
    # Also enforce YACS types so the exported profile can be merged back in.
    for key, values in grid.items():
        if key.startswith('TRAINER.BiMC.LAMBDA_T') and not cfg.TRAINER.BiMC.TEXT_CALIBRATION:
            raise ValueError('Enable TEXT_CALIBRATION before searching LAMBDA_T.')
        if key in ('TRAINER.BiMC.LAMBDA_I', 'TRAINER.BiMC.TAU') and not cfg.TRAINER.BiMC.VISION_CALIBRATION:
            raise ValueError('Enable VISION_CALIBRATION before searching LAMBDA_I/TAU.')
        if key in ('DATASET.ENSEMBLE_ALPHA', 'TRAINER.BiMC.GAMMA_BASE', 'TRAINER.BiMC.GAMMA_INC'):
            if not cfg.TRAINER.BiMC.USING_ENSEMBLE:
                raise ValueError('Enable USING_ENSEMBLE before searching ensemble parameters.')
        for value in values:
            trial = cfg.clone()
            trial.defrost()
            trial.merge_from_list([key, repr(value)])
            prepare_retune_config(trial)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', choices=('cub200', 'cifar100', 'miniimagenet'), required=True)
    parser.add_argument('--bank', required=True, help='Existing fdrfreq feature-bank directory.')
    parser.add_argument('--data-cfg', help='Defaults to configs/datasets/DATASET.yaml.')
    parser.add_argument('--train-cfg', default=str(ROOT / 'configs/trainers/bimc_ensemble.yaml'))
    parser.add_argument('--retune-cfg', help='Initial fixed parameters; defaults to configs/retune/DATASET.yaml.')
    parser.add_argument('--search-cfg', default=str(ROOT / 'configs/retune/search.yaml'))
    parser.add_argument('--shot', type=int, help='Overrides NUM_INC_SHOT before --opts.')
    parser.add_argument('--out', help='Search output directory; defaults to a configuration-specific path.')
    parser.add_argument('--dry-run', action='store_true', help='Preview without loading the bank or creating files.')
    parser.add_argument('--opts', nargs=argparse.REMAINDER, default=[], help='Fixed YACS KEY VALUE overrides; place last.')
    return parser.parse_args(argv)


def build_plan(args):
    # --opts for a searched parameter would be ambiguous. Require a singleton
    # list in the search YAML to fix it, or remove that key from the grid.
    if len(args.opts) % 2 or len(set(args.opts[::2])) != len(args.opts[::2]):
        raise ValueError('--opts requires distinct KEY VALUE pairs.')
    search = load_search(args.search_cfg)
    collision = set(search['grid']) & set(args.opts[::2])
    if collision:
        raise ValueError('Keys occur in both grid and --opts: ' + ', '.join(sorted(collision))
                         + '. Remove them from the grid to fix them with --opts.')
    if 'OUTPUT_DIR' in args.opts[::2]:
        raise ValueError('Use --out for the search directory; best.yaml gets a new evaluation directory.')
    opts = ['TRAINER.BiMC.PREC', 'fp32']
    if args.shot is not None:
        _positive_integer('--shot', args.shot)
        opts += ['DATASET.NUM_INC_SHOT', str(args.shot)]
    opts += args.opts
    cfg = setup_cfg(
        args.data_cfg or str(ROOT / f'configs/datasets/{args.dataset}.yaml'), args.train_cfg, opts,
        retune_cfg_file=args.retune_cfg or str(ROOT / f'configs/retune/{args.dataset}.yaml'),
    )
    if cfg.TRAINER.BiMC.PREC != 'fp32':
        raise ValueError('Cached auto-tuning uses fp32 CLIP features; set TRAINER.BiMC.PREC fp32.')
    validate_grid(cfg, search['grid'])
    from fdrfreq.select import expand_grid
    entry = {'name': 'BiMC_auto', 'base': pipeline_config(cfg),
             'grid': {PARAMETERS[key]: values for key, values in search['grid'].items()},
             'fallback': {}, 'include_fallback': True}
    candidates, _ = expand_grid(entry)
    spec = {'dataset': args.dataset, 'bank_path': str(Path(args.bank).resolve()),
            'fixed_config': cfg.dump(), 'search': search}
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
    out = Path(args.out) if args.out else ROOT / 'outputs/retune/search' / args.dataset / f'shot{cfg.DATASET.NUM_INC_SHOT}_{digest}'
    return {'cfg': cfg, 'search': search, 'entry': entry, 'spec': spec, 'out': out,
            'candidates': len(candidates), 'episodes': len(search['split_seeds']) * search['sequences']}


def validate_bank(bank, cfg):
    def canonical(name):
        return ''.join(c for c in name.lower() if c.isalnum())

    if canonical(bank.meta['dataset']) != canonical(cfg.DATASET.NAME):
        raise ValueError('Feature bank dataset does not match the retune configuration.')
    if bank.meta.get('backbone') != cfg.MODEL.BACKBONE.NAME:
        raise ValueError('Feature bank backbone does not match MODEL.BACKBONE.NAME.')
    if bank.num_init != cfg.DATASET.NUM_INIT_CLS or bank.num_inc != cfg.DATASET.NUM_INC_CLS:
        raise ValueError('Feature bank class/session split does not match the dataset configuration.')
    if bank.num_classes != cfg.DATASET.NUM_CLASSES:
        raise ValueError('Feature bank class count does not match DATASET.NUM_CLASSES.')
    if bank.meta.get('smoke_subset'):
        raise ValueError('Subset/smoke feature banks cannot be used for benchmark parameter selection.')
    expected = bank.meta.get('gpt_sha256')
    if expected:
        path = Path(cfg.DATASET.GPT_PATH)
        if not path.is_file():
            raise ValueError('Description file is needed to verify bank identity: ' + str(path))
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Feature bank descriptions do not match DATASET.GPT_PATH.')


def best_profile(cfg, selected):
    """Export every resolved setting so later trainer defaults cannot override it."""
    result = cfg.clone()
    result.defrost()
    for public, cached in PARAMETERS.items():
        if cached.startswith('ensemble.') and selected['ensemble'] is None:
            continue
        value = get_value(selected, cached)
        # A disabled calibration's stored strength is irrelevant. Preserve the
        # explicit manual value rather than reinterpreting it as a tuned value.
        if cached == 'lambda_t' and not result.TRAINER.BiMC.TEXT_CALIBRATION:
            continue
        if cached.startswith('holistic.') and not result.TRAINER.BiMC.VISION_CALIBRATION:
            continue
        set_value(result, public, value)
    # The search folder must never become the benchmark evaluation folder.
    result.OUTPUT_DIR = ''
    result.freeze()
    return result


def _write_json(path, value):
    # Selection reports contain numpy scalars but not feature matrices.
    import numpy as np

    def convert(item):
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError('Cannot serialize ' + type(item).__name__)

    path.write_text(json.dumps(value, default=convert, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def run_search(plan, bank):
    from fdrfreq import select as sel

    cfg, search, out = plan['cfg'], plan['search'], plan['out']
    validate_bank(bank, cfg)
    check_output_directory(out)
    # Preflight the protocol before producing outputs. Every episode uses only
    # original base-class training images, with disjoint support/query rows.
    episodes = sel.dev_episodes(bank, search['split_seeds'], search['sequences'],
                               cfg.DATASET.NUM_INC_SHOT, search['split_mode'])
    for episode in episodes:
        if episode.query_split != 'train' or any(int(c) >= bank.num_init for group in episode.classes for c in group):
            raise ValueError('Automatic tuning accepts base-training-only episodes.')
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / 'search_spec.json', plan['spec'])

    def progress(done, total):
        print(f'Validation episodes: {done}/{total}', flush=True)

    report = sel.select(bank, plan['entry'], episodes, objective=search['objective'],
                        tolerance=100.0, progress=progress)
    selected = report['selected_config']
    # Add the public keys and initial/manual comparison to the usual fdrfreq
    # report; retain its lock format for later cached evaluate-once if desired.
    profile = best_profile(cfg, selected)
    report['selected_public_params'] = {k: get_value(profile, k) for k in PARAMETERS}
    report['search_settings'] = copy.deepcopy(search)
    report['test_evaluated'] = False
    _write_json(out / 'search_report.json', report)
    _write_json(out / 'locked_BiMC_auto.json', sel.lock(report))
    (out / 'best.yaml').write_text(
        '# Selected on base-training validation episodes; benchmark test set was not evaluated.\n'
        + profile.dump(), encoding='utf-8')
    print('Selected parameters:', json.dumps(report['selected_public_params'], ensure_ascii=False))
    print('Saved:', out / 'best.yaml')
    return report


def main(argv=None):
    args = parse_args(argv)
    try:
        plan = build_plan(args)
        print(f"Grid: {plan['candidates']} combinations + initial configuration; "
              f"{plan['episodes']} validation episodes; objective={plan['search']['objective']}", flush=True)
        print('Search output:', plan['out'])
        if args.dry_run:
            print('Dry run: bank not loaded; no search or benchmark evaluation performed.')
            return 0
        if not (Path(args.bank) / 'meta.json').is_file():
            raise ValueError('Feature bank not found: ' + args.bank
                             + '. Create it with python -m fdrfreq.extract first (see docs/bimc_retune.md).')
        from fdrfreq.bank import load_bank
        run_search(plan, load_bank(args.bank))
    except (ValueError, KeyError, OSError, TypeError) as error:
        print('BiMC automatic tuning failed: ' + str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
