"""Base-only hyper-parameter selection and configuration locking.

A matrix entry looks like::

    {"name": "K1", "base": {...pipeline config...},
     "grid": {"segments.alpha": [0.0, 0.1, 0.2], "segments.q": [37, 50]},
     "fallback": {"segments.alpha": 0.0},
     "tolerance": 0.2}

Every grid point is scored on the same pseudo-incremental episodes (split
seeds x sequences, built only from base-session training images). The
selected point maximises mean pseudo last-session accuracy subject to its
last-session base and novel accuracies not dropping by more than
``tolerance`` points below the fallback point (the configuration with the
new component switched off). Ties keep the earlier grid point, so list grid
values from weakest to strongest.
"""

import copy
import hashlib
import itertools
import json
import time

import numpy as np

from .bank import bank_fingerprint
from .pipeline import EpisodeRunner, deep_merge
from .protocol import pseudo_episodes, check_episode


def set_path(cfg, path, value):
    cfg = copy.deepcopy(cfg)
    node = cfg
    keys = path.split('.')
    for k in keys[:-1]:
        if node.get(k) is None:
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value
    return cfg


def expand_grid(entry):
    base = entry.get('base', {})
    grid = entry.get('grid', {})
    paths = list(grid)
    out = []
    for values in itertools.product(*[grid[p] for p in paths]) if paths else [()]:
        cfg = base
        for p, v in zip(paths, values):
            cfg = set_path(cfg, p, v)
        out.append((dict(zip(paths, values)), cfg))
    fb = entry.get('fallback')
    fallback = None
    if fb is not None:
        cfg = base
        for p, v in fb.items():
            cfg = set_path(cfg, p, v)
        fallback = cfg
    return out, fallback


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def dev_episodes(bank, split_seeds=(2, 3, 4), n_sequences=10, shot=None, split_mode='random'):
    eps = []
    for s in split_seeds:
        for e in pseudo_episodes(bank, s, n_sequences=n_sequences, shot=shot, split_mode=split_mode):
            check_episode(e, bank)
            eps.append(e)
    return eps


def evaluate_candidates(bank, episodes, configs):
    """Return per-config arrays [n_episodes, 3] of (last, last_base, last_novel) + avg."""
    out = [[] for _ in configs]
    for e in episodes:
        runner = EpisodeRunner(bank, e)
        for i, cfg in enumerate(configs):
            r = runner.run(cfg)['summary']
            out[i].append((r['last'], r['last_base'], r['last_novel'], r['avg'], r['last_hm']))
    return [np.asarray(o, dtype=np.float64) for o in out]


def select(bank, entry, episodes, progress=None, tolerance=None, objective='last'):
    t0 = time.time()
    cands, fallback = expand_grid(entry)
    configs = [c for _, c in cands] + ([fallback] if fallback is not None else [])
    scores = evaluate_candidates(bank, episodes, configs)
    tol = float(tolerance if tolerance is not None else entry.get('tolerance', 0.2))
    split_ids = np.array([e.meta['split_seed'] for e in episodes])
    rows = []
    fb = scores[-1].mean(0) if fallback is not None else None
    for i, (params, cfg) in enumerate(cands):
        m = scores[i].mean(0)
        per_split = [float(scores[i][split_ids == s, 0].mean()) for s in np.unique(split_ids)]
        ok = True
        if fb is not None:
            ok = (m[1] >= fb[1] - tol) and (m[2] >= fb[2] - tol)
        rows.append({'params': params, 'last': m[0], 'last_base': m[1], 'last_novel': m[2],
                     'avg': m[3], 'last_hm': m[4], 'per_split_last': per_split,
                     'feasible': bool(ok), 'hash': config_hash(cfg)})
    feasible = [i for i, r in enumerate(rows) if r['feasible']]
    if not feasible:
        best_cfg, best = fallback, None
    else:
        key = {'last': 'last', 'hm': 'last_hm', 'avg': 'avg'}[objective]
        best = max(feasible, key=lambda i: (round(rows[i][key], 9), -i))
        best_cfg = cands[best][1]
    report = {
        'name': entry['name'], 'selected_params': rows[best]['params'] if best is not None else 'fallback',
        'selected_config': best_cfg, 'selected_hash': config_hash(best_cfg),
        'fallback_config': fallback,
        'fallback_metrics': None if fb is None else {'last': fb[0], 'last_base': fb[1],
                                                     'last_novel': fb[2], 'avg': fb[3]},
        'candidates': rows, 'tolerance': tol, 'objective': objective,
        'dev_protocol': {'split_seeds': sorted(set(split_ids.tolist())),
                         'split_mode': episodes[0].meta.get('split_mode', 'random'),
                         'n_episodes': len(episodes),
                         'shot': episodes[0].meta['shot'], 'way': episodes[0].meta['way']},
        'bank': bank_fingerprint(bank), 'dataset': bank.meta.get('dataset'),
        'seconds': time.time() - t0,
    }
    return report


def lock(report):
    cfg = report['selected_config']
    return {'name': report['name'], 'config': cfg, 'config_hash': config_hash(cfg),
            'selected_params': report['selected_params'], 'dev_protocol': report['dev_protocol'],
            'bank': report['bank'], 'dataset': report['dataset'],
            'locked_at': time.strftime('%Y-%m-%d %H:%M:%S')}


def chain(entry, locked):
    """Use a locked configuration as the base of the next selection stage."""
    e = copy.deepcopy(entry)
    e['base'] = deep_merge(locked['config'], entry.get('base', {}))
    return e
