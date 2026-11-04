"""Error types of the locked test predictions (no new test evaluation).

Reads the ``eval_*_shot*.json`` / ``pred_*_shot*.npz`` files that ``run evaluate`` wrote
during the single test run and breaks every last-session error into

    base->base  base query, predicted as another base class
    base->new   base query, predicted as a new class
    new->base   new-class query, predicted as a base class
    new->new    new-class query, predicted as another new class
                (split into the same incremental session / another session)

It also reports, for each method against an anchor (BiMC_ens by default), which error types
the method fixes and which it introduces (paired over the same queries), and the base / new
accuracy of every session from the stored per-session metrics.

    python -m fdrfreq.errtypes --dataset cub200
    python -m fdrfreq.errtypes --dataset cub200 --methods BiMC_ens A2 M4a --anchor BiMC_ens

Writes <out>/errtypes.md and <out>/errtypes.json (out = outputs/fdrfreq/<dataset>_<tag>).
"""

import argparse
import json
from pathlib import Path

import numpy as np

CATS = ('correct', 'base->base', 'base->new', 'new->base', 'new->new')
DEFAULT_METHODS = ('Proto', 'TEEN', 'P4a_Proto', 'BiMC', 'P4a_BiMC', 'BiMC_ens', 'LA', 'ULA', 'A2',
                   'M4a', 'M4a_soft', 'M4a_random', 'M4a_freq', 'M4a_text')


def categorize(pred, labels, num_init):
    """Index into CATS for every query."""
    pred, labels = np.asarray(pred), np.asarray(labels)
    lb, pb = labels < num_init, pred < num_init
    cat = np.zeros(len(labels), dtype=np.int64)
    wrong = pred != labels
    cat[wrong & lb & pb] = 1
    cat[wrong & lb & ~pb] = 2
    cat[wrong & ~lb & pb] = 3
    cat[wrong & ~lb & ~pb] = 4
    return cat


def session_of(cls, num_init, num_inc):
    cls = np.asarray(cls)
    return np.where(cls < num_init, 0, 1 + (cls - num_init) // num_inc)


def setting_report(d, num_init, num_inc, methods, anchor):
    d = Path(d)
    runs = {}
    for f in sorted(d.glob('eval_*_shot*.json')):
        r = json.loads(f.read_text(encoding='utf-8'))
        if r['name'] in methods or r['name'] == anchor:
            runs[r['name']] = (r, np.load(f.with_name(f.name.replace('eval_', 'pred_').replace('.json', '.npz'))))
    if not runs:
        return None
    names = [m for m in methods if m in runs]
    seeds = list(next(iter(runs.values()))[0]['seeds'])
    rep = {'dir': str(d), 'seeds': seeds, 'num_init': num_init, 'num_inc': num_inc, 'methods': {}}
    cats = {}
    for name in names:
        r, p = runs[name]
        acc = {k: [] for k in ('base_acc', 'new_acc', 'base->base', 'base->new', 'new->base', 'new->new',
                               'new->new same session', 'new->new other session', 'n_base', 'n_new')}
        cats[name] = {}
        for s in seeds:
            y, yh = p[f'labels_seed{s}'], p[f'pred_seed{s}']
            c = categorize(yh, y, num_init)
            cats[name][s] = (c, y)
            nb, nn = int((y < num_init).sum()), int((y >= num_init).sum())
            acc['n_base'].append(nb)
            acc['n_new'].append(nn)
            acc['base_acc'].append(100.0 * (c[y < num_init] == 0).mean())
            acc['new_acc'].append(100.0 * (c[y >= num_init] == 0).mean())
            acc['base->base'].append(100.0 * (c == 1).sum() / nb)
            acc['base->new'].append(100.0 * (c == 2).sum() / nb)
            acc['new->base'].append(100.0 * (c == 3).sum() / nn)
            acc['new->new'].append(100.0 * (c == 4).sum() / nn)
            same = (c == 4) & (session_of(yh, num_init, num_inc) == session_of(y, num_init, num_inc))
            acc['new->new same session'].append(100.0 * same.sum() / nn)
            acc['new->new other session'].append(100.0 * ((c == 4) & ~same).sum() / nn)
        sess = [v['per_session'] for v in r['seeds'].values()]
        rep['methods'][name] = {
            'rates': {k: float(np.mean(v)) for k, v in acc.items()},
            'session_base': [float(np.nanmean([s[t]['base'] for s in sess])) for t in range(len(sess[0]))],
            'session_new': [float(np.nanmean([s[t]['novel'] for s in sess])) if t > 0 else None
                            for t in range(len(sess[0]))],
            'session_acc': [float(np.mean([s[t]['acc'] for s in sess])) for t in range(len(sess[0]))],
        }
    # paired transitions against the anchor: rows = anchor category, columns = method category
    if anchor in cats:
        for name in names:
            if name == anchor:
                continue
            M = np.zeros((len(CATS), len(CATS)))
            for s in seeds:
                (ca, ya), (cm, ym) = cats[anchor][s], cats[name][s]
                if not np.array_equal(ya, ym):
                    raise ValueError('query order differs between runs')
                np.add.at(M, (ca, cm), 1)
            M /= len(seeds)
            rep['methods'][name]['vs_anchor'] = {
                'anchor': anchor, 'matrix': M.tolist(),
                'fixed_by_type': {CATS[i]: float(M[i, 0]) for i in range(1, len(CATS))},
                'broken_by_type': {CATS[j]: float(M[0, j]) for j in range(1, len(CATS))},
            }
    return rep


def to_markdown(dataset, reps):
    out = [f'# Error types, {dataset} (last session, mean over support seeds)', '']
    for shot, rep in reps.items():
        if rep is None:
            out += [f'## {shot}-shot: no evaluation files found', '']
            continue
        out += [f'## {shot}-shot ({rep["dir"]}, seeds {" ".join(rep["seeds"])})', '',
                'Rates: base errors in % of base queries, new errors in % of new queries.', '',
                'Method | base acc | new acc | base->base | base->new | new->base | new->new | (same session) | (other session)',
                '---|---|---|---|---|---|---|---|---']
        for name, m in rep['methods'].items():
            r = m['rates']
            out.append(f"{name} | {r['base_acc']:.2f} | {r['new_acc']:.2f} | {r['base->base']:.2f} | "
                       f"{r['base->new']:.2f} | {r['new->base']:.2f} | {r['new->new']:.2f} | "
                       f"{r['new->new same session']:.2f} | {r['new->new other session']:.2f}")
        anchor = next((m['vs_anchor']['anchor'] for m in rep['methods'].values() if 'vs_anchor' in m), None)
        if anchor:
            out += ['', f'Paired against {anchor} (queries per seed): fixed = {anchor} wrong -> method right, '
                        f'by {anchor}\'s error type; broken = {anchor} right -> method wrong, by the method\'s error type.', '',
                    'Method | fixed b->b | fixed b->n | fixed n->b | fixed n->n | broken b->b | broken b->n | broken n->b | broken n->n | net',
                    '---|---|---|---|---|---|---|---|---|---']
            for name, m in rep['methods'].items():
                if 'vs_anchor' not in m:
                    continue
                fx, br = m['vs_anchor']['fixed_by_type'], m['vs_anchor']['broken_by_type']
                net = sum(fx.values()) - sum(br.values())
                out.append(f'{name} | ' + ' | '.join(f'{fx[c]:.1f}' for c in CATS[1:]) + ' | '
                           + ' | '.join(f'{br[c]:.1f}' for c in CATS[1:]) + f' | {net:+.1f}')
        out += ['', 'Per-session accuracy (all / base / new), mean over seeds', '']
        for name, m in rep['methods'].items():
            out.append(f"{name}: all " + ' '.join(f'{x:.2f}' for x in m['session_acc']))
            out.append(f"{' ' * len(name)}  base " + ' '.join(f'{x:.2f}' for x in m['session_base']))
            out.append(f"{' ' * len(name)}  new  " + ' '.join('  -  ' if x is None else f'{x:.2f}'
                                                         for x in m['session_new']))
        out.append('')
    return '\n'.join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--tag', default='vitb16')
    ap.add_argument('--shots', type=int, nargs='+', default=[1, 5])
    ap.add_argument('--methods', nargs='+', default=list(DEFAULT_METHODS))
    ap.add_argument('--anchor', default='BiMC_ens')
    ap.add_argument('--bank', default=None, help='default: banks/<dataset>_<tag>')
    ap.add_argument('--out', default=None, help='default: outputs/fdrfreq/<dataset>_<tag>')
    a = ap.parse_args(argv)
    bank = Path(a.bank or f'banks/{a.dataset}_{a.tag}')
    meta = json.loads((bank / 'meta.json').read_text(encoding='utf-8'))
    num_init, num_inc = int(meta['num_init']), int(meta['num_inc'])
    out = Path(a.out or f'outputs/fdrfreq/{a.dataset}_{a.tag}')
    reps = {s: setting_report(out / f'shot{s}_hm', num_init, num_inc, a.methods, a.anchor) for s in a.shots}
    (out / 'errtypes.json').write_text(json.dumps({'dataset': a.dataset, 'settings': reps}, indent=1),
                                       encoding='utf-8')
    md = to_markdown(a.dataset, reps)
    (out / 'errtypes.md').write_text(md + '\n', encoding='utf-8')
    print(md)


if __name__ == '__main__':
    main()
