"""Paper tables from ``fdrfreq.run evaluate`` outputs.

    python -m fdrfreq.report --dir DIR --anchor A2 [--format md|latex] [--shot 5] [--pairs A:B ...]

Main table: Avg / Last / Base / Novel / HM / PD (mean +- std over support seeds).
Paired table vs the anchor: delta Last with a 95% paired bootstrap CI (images
resampled within seed), number of seeds with a positive delta, fixed/broken
counts and the number of seeds with exact McNemar p < 0.05. ``--pairs A:B`` adds the same
statistics for B - A on any listed pair (pairs with a missing run are skipped and named).

    python -m fdrfreq.report --dir DIR --shot 1 --pairs A:B ... --hier-json pairs_shot1.json
writes, for each pair, both intervals (images within seed; seeds then images) with the
seed-level mean +- sd and a bootstrap p-value, and

    python -m fdrfreq.report --holm FILE [FILE ...] --family A:B [A:B ...]
applies Holm's correction across every listed setting and pair of the family.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .metrics import paired_bootstrap, mcnemar_exact, fixed_broken, hierarchical_bootstrap, holm


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


def _paired_rows(runs, pairs, sep, end):
    rows, missing = [], []
    for a, b in pairs:
        if a not in runs or b not in runs:
            missing.append(f'{a}:{b}')
            continue
        (ra, pa), (_, p) = runs[a], runs[b]
        ca, cb, fx, br, sig = [], [], 0, 0, 0
        for s in ra['seeds']:
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
        rows.append(sep.join([f'{b} - {a}', f"{bs['delta']:+.2f}", f"[{bs['ci_low']:+.2f}, {bs['ci_high']:+.2f}]",
                              f"{bs['seeds_positive']}/{bs['n_seeds']}", str(fx), str(br),
                              f"{sig}/{len(ra['seeds'])}"]) + end)
    return rows, missing


def pair_stats(dirpath, pairs, shot=None):
    """Per-pair statistics for JSON output (both bootstrap schemes)."""
    runs = _load(dirpath, shot)
    out = []
    for a, b in pairs:
        if a not in runs or b not in runs:
            out.append({'pair': f'{a}:{b}', 'missing': True})
            continue
        (ra, pa), (_, p) = runs[a], runs[b]
        ca, cb, sig = [], [], 0
        for s_ in ra['seeds']:
            ya, yb = pa[f'labels_seed{s_}'], p[f'labels_seed{s_}']
            if not np.array_equal(ya, yb):
                raise ValueError('query order differs between runs')
            ca.append(pa[f'pred_seed{s_}'] == ya)
            cb.append(p[f'pred_seed{s_}'] == yb)
            sig += int(mcnemar_exact(ca[-1], cb[-1])['p'] < 0.05)
        w = paired_bootstrap(ca, cb)
        h = hierarchical_bootstrap(ca, cb)
        out.append({'pair': f'{a}:{b}', 'a': a, 'b': b, 'shot': int(ra['shot']), 'dataset': ra.get('dataset'),
                    'delta': w['delta'], 'ci_within': [w['ci_low'], w['ci_high']],
                    'ci_hier': [h['ci_low'], h['ci_high']], 'seed_mean': h['seed_mean'],
                    'seed_sd': h['seed_sd'], 'p_boot': h['p_boot'],
                    'seeds_positive': w['seeds_positive'], 'n_seeds': w['n_seeds'],
                    'mcnemar_seeds': sig})
    return out


def holm_table(files, family):
    """Holm-adjusted p over all (setting, pair) members of the family found in the files."""
    rows, absent = [], []
    for f in files:
        recs = json.loads(Path(f).read_text(encoding='utf-8'))
        have = {r['pair'] for r in recs if not r.get('missing')}
        where = next((f"{r.get('dataset')} {r.get('shot')}-shot" for r in recs), str(f))
        absent += [f'{where} {pair}' for pair in family if pair not in have]
        rows += [r for r in recs if not r.get('missing') and r['pair'] in family]
    if not rows:
        return 'no family member found'
    adj = holm([r['p_boot'] for r in rows])
    lines = ['Setting | Pair (B - A) | Delta | 95% CI (seeds+images) | seed mean ± sd | p (boot) | p (Holm)',
             '---|---|---|---|---|---|---']
    for r, q in zip(rows, adj):
        lines.append(f"{r.get('dataset')} {r['shot']}-shot | {r['b']} - {r['a']} | {r['delta']:+.2f} | "
                     f"[{r['ci_hier'][0]:+.2f}, {r['ci_hier'][1]:+.2f}] | {r['seed_mean']:+.2f} ± {r['seed_sd']:.2f} | "
                     f"{r['p_boot']:.4f} | {q:.4f}")
    lines.append(f'family size m = {len(rows)}')
    if absent:
        lines.append('not found, so not counted in m: ' + '; '.join(absent))
    return '\n'.join(lines)


def build_report(dirpath, anchor=None, fmt='md', shot=None, pairs=None):
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
    if pairs:
        head = ['Pair (B - A)', 'ΔLast', '95% CI', 'seeds>0', 'fixed', 'broken', 'McNemar p<.05']
        lines = [sep.join(head) + end]
        if not latex:
            lines.append('|'.join(['---'] * len(head)))
        rows, missing = _paired_rows(runs, pairs, sep, end)
        out += ['Pre-registered paired comparisons (last session)', *lines, *rows]
        if missing:
            out.append(f"skipped (run missing): {', '.join(missing)}")
        out.append('')
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
    p.add_argument('--dir')
    p.add_argument('--anchor')
    p.add_argument('--format', default='md', choices=('md', 'latex'))
    p.add_argument('--shot', type=int)
    p.add_argument('--pairs', nargs='+', default=None, help='A:B pairs, reported as B - A')
    p.add_argument('--hier-json', default=None, help='write per-pair statistics (both bootstraps) to this file')
    p.add_argument('--holm', nargs='+', default=None, help='pair-statistics JSON files to correct jointly')
    p.add_argument('--family', nargs='+', default=None, help='A:B pairs forming the Holm family')
    a = p.parse_args(argv)
    if a.holm:
        print(holm_table(a.holm, set(a.family or [])))
        return
    if not a.dir:
        p.error('--dir is required unless --holm is given')
    pairs = [tuple(x.split(':', 1)) for x in a.pairs] if a.pairs else None
    if a.hier_json and pairs:
        Path(a.hier_json).write_text(json.dumps(pair_stats(a.dir, pairs, a.shot), indent=1), encoding='utf-8')
    print(build_report(a.dir, a.anchor, a.format, a.shot, pairs))


if __name__ == '__main__':
    main()
