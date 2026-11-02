"""Paper tables from ``fdrfreq.run evaluate`` outputs.

    python -m fdrfreq.report --dir DIR --anchor A2 [--format md|latex] [--shot 5]

Main table: Avg / Last / Base / Novel / HM / PD (mean +- std over support seeds).
Paired table vs the anchor: delta Last with a 95% paired bootstrap CI (images
resampled within seed), number of seeds with a positive delta, fixed/broken
counts and the number of seeds with exact McNemar p < 0.05.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .metrics import paired_bootstrap, mcnemar_exact, fixed_broken


def _load(dirpath, shot=None):
    runs = {}
    for f in sorted(Path(dirpath).glob('eval_*_shot*.json')):
        r = json.loads(f.read_text(encoding='utf-8'))
        if shot is not None and int(r['shot']) != int(shot):
            continue
        pred = np.load(f.with_name(f.name.replace('eval_', 'pred_').replace('.json', '.npz')))
        runs[r['name']] = (r, pred)
    return runs


def _fmt(m, s, latex):
    return f'{m:.2f}$\\pm${s:.2f}' if latex else f'{m:.2f}±{s:.2f}'


def build_report(dirpath, anchor=None, fmt='md', shot=None):
    runs = _load(dirpath, shot)
    if not runs:
        return 'no evaluation files found'
    latex = fmt == 'latex'
    keys = ('avg', 'last', 'last_base', 'last_novel', 'last_hm', 'pd')
    head = ['Method', 'Avg', 'Last', 'Base', 'Novel', 'HM', 'PD']
    lines = []
    sep = ' & ' if latex else ' | '
    end = ' \\\\' if latex else ''
    lines.append(sep.join(head) + end)
    if not latex:
        lines.append('|'.join(['---'] * len(head)))
    for name, (r, _) in runs.items():
        cells = [name] + [_fmt(r['mean'][k], r['std'][k], latex) for k in keys]
        lines.append(sep.join(cells) + end)
    out = ['Main results (mean ± std over support seeds)', *lines, '']
    if anchor and anchor in runs:
        ra, pa = runs[anchor]
        seeds = list(ra['seeds'])
        head = ['Method', 'ΔLast vs ' + anchor, '95% CI', 'seeds>0', 'fixed', 'broken', 'McNemar p<.05']
        lines = [sep.join(head) + end]
        if not latex:
            lines.append('|'.join(['---'] * len(head)))
        for name, (r, p) in runs.items():
            if name == anchor:
                continue
            ca, cb, fx, br, sig = [], [], 0, 0, 0
            for s in seeds:
                ya, yb = pa[f'labels_seed{s}'], p[f'labels_seed{s}']
                if not np.array_equal(ya, yb):
                    raise ValueError('query order differs between runs')
                a_ok, b_ok = pa[f'pred_seed{s}'] == ya, p[f'pred_seed{s}'] == yb
                ca.append(a_ok)
                cb.append(b_ok)
                fb = fixed_broken(pa[f'pred_seed{s}'], p[f'pred_seed{s}'], ya)
                fx += fb['fixed']
                br += fb['broken']
                sig += int(mcnemar_exact(a_ok, b_ok)['p'] < 0.05)
            bs = paired_bootstrap(ca, cb)
            lines.append(sep.join([name, f"{bs['delta']:+.2f}", f"[{bs['ci_low']:+.2f}, {bs['ci_high']:+.2f}]",
                                   f"{bs['seeds_positive']}/{bs['n_seeds']}", str(fx), str(br),
                                   f'{sig}/{len(seeds)}']) + end)
        out += [f'Paired comparison with {anchor} (last session)', *lines, '']
    sess = {}
    for name, (r, _) in runs.items():
        sess[name] = np.mean([v['summary']['sessions'] for v in r['seeds'].values()], axis=0)
    n = max(len(v) for v in sess.values())
    head = ['Method'] + [f'S{i}' for i in range(n)]
    lines = [sep.join(head) + end]
    if not latex:
        lines.append('|'.join(['---'] * len(head)))
    for name, v in sess.items():
        lines.append(sep.join([name] + [f'{x:.2f}' for x in v]) + end)
    out += ['Session-wise accuracy (mean over seeds)', *lines]
    return '\n'.join(out)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--dir', required=True)
    p.add_argument('--anchor')
    p.add_argument('--format', default='md', choices=('md', 'latex'))
    p.add_argument('--shot', type=int)
    a = p.parse_args(argv)
    print(build_report(a.dir, a.anchor, a.format, a.shot))


if __name__ == '__main__':
    main()
