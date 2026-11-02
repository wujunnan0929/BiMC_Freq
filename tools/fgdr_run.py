"""FGDR experiment CLI (CPU, NumPy only; runs on cached features).

Typical order (see fgdr/README.md):
  parity          BiMC / BiMC-dagger on the benchmark (fixed reference, no tuning)
  diagnose-masks  G1' gate: class specificity and shot stability of frequency masks (base only)
  diagnose-views  D1/D2: per-view accuracy and complementarity on pseudo episodes (base only)
  dev             grid evaluation on base-only pseudo episodes + per-family selection draft
  lock            freeze chosen configs into a checksummed selection file
  final           one evaluation of the locked configs on the benchmark (all seeds)
  diagnose-prototypes  novel-prototype quality vs full-data class means (training images only)
  compare         paired statistics between two methods of a final run
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fgdr import cache as fc
from fgdr.bimc_ref import unit
from fgdr.grounding import (KeywordLibrary, class_signatures, class_specificity, perturbation_deltas,
                            ladder_deltas, random_jaccard, shot_stability, topq_masks)
from fgdr.metrics import (bootstrap_diff, mcnemar_exact, oracle_gain, paired_counts, seed_summary)
from fgdr.pipeline import (EpisodeCache, aggregate, expand, normalize_spec, required_views, run_episode,
                           select)
from fgdr.protocol import VIEW_KEYS, benchmark_episode, pseudo_episodes

REFERENCES = [
    {'name': 'BiMC', 'bimc': {'ensemble': False}},
    {'name': 'BiMC_dagger', 'bimc': {'ensemble': True}},
]


def _dump(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding='utf-8')


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


def _keywords(path):
    return {'keywords': KeywordLibrary.load(path)} if path else {}


def _defaults(meta):
    return meta.get('bimc_defaults', {})


def _views_for(specs):
    return tuple(v for v in VIEW_KEYS if v in required_views(specs))


def _make_lock(meta, selected, extra=None):
    lock = {'status': 'locked', 'cache_meta_sha256': meta['meta_sha256'], 'dataset': meta.get('dataset'),
            'selected': selected, 'locked_at': datetime.now(timezone.utc).isoformat()}
    lock.update(extra or {})
    lock['lock_sha256'] = fc.canonical_sha(lock)
    return lock


# --------------------------------------------------------------------------- parity / final


def run_final(cache, specs, lock_path, shot, seeds, out, keywords=None, note=None):
    views = _views_for(specs)
    test, meta, lock = fc.load_test(cache, lock_path, views=views)
    train, _ = fc.load_train(cache, views=views)
    resources = _keywords(keywords)
    results = {s['name']: {} for s in specs}
    preds = {}
    for seed in seeds:
        ep = benchmark_episode(train, test, meta, shot, seed, views=views)
        shared = EpisodeCache(ep)
        for spec in specs:
            start = time.perf_counter()
            r = run_episode(ep, spec, _defaults(meta), resources, keep_predictions=True, cache=shared)
            r['seconds'] = time.perf_counter() - start
            preds[f'{spec["name"]}::seed{seed}'] = r.pop('last_predictions')
            labels = r.pop('last_labels')
            r.pop('last_rows')
            results[spec['name']][seed] = r
            print(f'seed {seed} {spec["name"]}: last {r["summary"]["last"]:.3f} avg {r["summary"]["avg"]:.3f}',
                  flush=True)
        preds[f'__labels__::seed{seed}'] = labels
    reference = specs[0]['name']
    table = {}
    for name, per_seed in results.items():
        row = {key: seed_summary([per_seed[s]['summary'][key] for s in seeds])
               for key in ('avg', 'last', 'pd', 'last_base', 'last_novel', 'last_h')
               if all(per_seed[s]['summary'].get(key) is not None for s in seeds)}
        row['sessions_mean'] = np.mean([per_seed[s]['summary']['sessions'] for s in seeds], axis=0).tolist()
        if name != reference:
            paired = []
            for s in seeds:
                a, b = preds[f'{reference}::seed{s}'], preds[f'{name}::seed{s}']
                lab = preds[f'__labels__::seed{s}']
                counts = paired_counts(a, b, lab)
                counts['mcnemar_p'] = mcnemar_exact(counts['C'], counts['D'])
                counts.update(bootstrap_diff(a, b, lab, reps=1000, seed=s))
                paired.append(counts)
            diffs = [p['diff'] for p in paired]
            row['paired_vs_reference'] = {'per_seed': paired, 'mean_diff': float(np.mean(diffs)),
                                          'positive_seeds': int(sum(d > 0 for d in diffs))}
        table[name] = row
    payload = {'cache': str(cache), 'lock_sha256': lock['lock_sha256'], 'shot': shot, 'seeds': list(seeds),
               'reference': reference, 'specs': specs, 'table': table, 'per_seed': results, 'note': note}
    out = Path(out)
    _dump(out / 'results.json', payload)
    np.savez(out / 'predictions.npz', **preds)
    (out / 'table.md').write_text(markdown_table(table, reference), encoding='utf-8')
    print((out / 'table.md').read_text(encoding='utf-8'))
    return payload


def markdown_table(table, reference):
    lines = ['| Method | Sessions (mean) | Avg | Last | PD | Base | Novel | H | ΔLast vs ref (+seeds) |',
             '|---|---|---|---|---|---|---|---|---|']
    for name, row in table.items():
        f = lambda k: (f"{row[k]['mean']:.2f}±{row[k]['std']:.2f}" if k in row else '—')
        sessions = ' '.join(f'{v:.1f}' for v in row['sessions_mean'])
        delta = '—'
        if 'paired_vs_reference' in row:
            p = row['paired_vs_reference']
            delta = f"{p['mean_diff']:+.2f} ({p['positive_seeds']}/{len(p['per_seed'])})"
        lines.append(f'| {name} | {sessions} | {f("avg")} | {f("last")} | {f("pd")} | {f("last_base")} | '
                     f'{f("last_novel")} | {f("last_h")} | {delta} |')
    return '\n'.join(lines) + f'\n\nReference: {reference}\n'


def cmd_parity(a):
    meta = fc.load_meta(a.cache)
    out = Path(a.out)
    lock = _make_lock(meta, {'references': REFERENCES}, {'purpose': 'fixed_reference_parity'})
    _dump(out / 'reference_lock.json', lock)
    payload = run_final(a.cache, REFERENCES, out / 'reference_lock.json', a.shot, a.seeds, out,
                        note='Fixed references only; no tuned configuration was evaluated.')
    if a.expect is not None:
        got = payload['table']['BiMC_dagger']['last']['mean']
        print(f'Parity: BiMC_dagger last {got:.3f} vs engine {a.expect:.3f} (diff {got - a.expect:+.3f})')
        if abs(got - a.expect) > a.tolerance:
            print('PARITY FAILED: fix the cache/reference before any method comparison.', file=sys.stderr)
            return 3
    return 0


def cmd_final(a):
    lock = fc.load_lock(a.lock)
    specs = list(REFERENCES)
    names = {s['name'] for s in specs}
    for spec in lock['selected'].get('methods', []):
        if spec['name'] not in names:
            specs.append(spec)
            names.add(spec['name'])
    for spec in lock['selected'].get('baselines', []):
        if spec['name'] not in names:
            specs.append(spec)
            names.add(spec['name'])
    if a.reference:
        specs.sort(key=lambda s: s['name'] != a.reference)
    run_final(a.cache, specs, a.lock, a.shot, a.seeds, a.out, a.keywords)
    return 0


# --------------------------------------------------------------------------- dev / lock


def cmd_dev(a):
    config = json.loads(Path(a.config).read_text(encoding='utf-8'))
    reference = config.get('reference', 'BiMC_dagger')
    base_specs = list(REFERENCES) + config.get('baselines', []) + config['methods']
    configs = []
    for spec in base_specs:
        configs.extend(expand(spec))
    names = [c['name'] for c in configs]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate config names after grid expansion')
    resources = _keywords(a.keywords)
    views = _views_for(configs)
    train, meta = fc.load_train(a.cache, views=views)
    overrides = {k: getattr(a, k) for k in ('n_fit', 'way', 'n_sessions', 'query_per_class')}
    per_config = {c['name']: [] for c in configs}
    start = time.perf_counter()
    count = 0
    for ep in pseudo_episodes(train, meta, shot=a.shot, split_seeds=a.split_seeds,
                              sequences=a.sequences, views=views, **overrides):
        shared = EpisodeCache(ep)
        for spec in configs:
            per_config[spec['name']].append(run_episode(ep, spec, _defaults(meta), resources, cache=shared))
        count += 1
        print(f'episode {count} done ({time.perf_counter() - start:.1f}s)', flush=True)
    rows = {}
    for spec in configs:
        res = per_config[spec['name']]
        rows[spec['name']] = {'family': spec.get('family', spec['name']), 'spec': spec, 'mean': aggregate(res),
                              'episode_last': [r['summary']['last'] for r in res],
                              'episode_novel_to_base': [r['summary']['errors']['novel_to_base_pct'] for r in res]}
    ref_last = np.asarray(rows[reference]['episode_last'])
    for row in rows.values():
        diff = np.asarray(row['episode_last']) - ref_last
        row['paired_last_diff'] = {'mean': float(diff.mean()), 'sd': float(diff.std(ddof=1)) if len(diff) > 1 else 0.,
                                   'positive': int((diff > 0).sum()), 'negative': int((diff < 0).sum())}
    families = []
    for spec in config['methods']:
        family = spec.get('family', spec['name'])
        if family not in families:
            families.append(family)
    draft = {'cache': str(a.cache), 'cache_meta_sha256': meta['meta_sha256'], 'reference': reference,
             'protocol': {'shot': a.shot, 'split_seeds': a.split_seeds, 'sequences': a.sequences, **overrides},
             'guards': {'max_base_drop': a.max_base_drop, 'max_novel_drop': a.max_novel_drop},
             'reference_mean': rows[reference]['mean'], 'families': {}}
    for family in families:
        best = select(rows, family, reference, a.max_base_drop, a.max_novel_drop)
        draft['families'][family] = None if best is None else {
            'name': best, 'spec': rows[best]['spec'], 'mean': rows[best]['mean'],
            'paired_last_diff': rows[best]['paired_last_diff']}
    out = Path(a.out)
    _dump(out / 'dev_rows.json', rows)
    _dump(out / 'selection_draft.json', draft)
    print(json.dumps({f: (v['name'], round(v['paired_last_diff']['mean'], 3)) if v else None
                      for f, v in draft['families'].items()}, indent=2, ensure_ascii=False))
    return 0


def cmd_lock(a):
    draft = json.loads(Path(a.draft).read_text(encoding='utf-8'))
    meta = fc.load_meta(draft['cache'] if a.cache is None else a.cache)
    if meta['meta_sha256'] != draft['cache_meta_sha256']:
        raise ValueError('Draft was produced on a different cache')
    methods = []
    for family in a.families:
        entry = draft['families'].get(family)
        if entry is None:
            print(f'Family {family} has no feasible configuration; not locked.', file=sys.stderr)
            continue
        spec = normalize_spec(entry['spec'])
        spec['name'] = family
        methods.append(spec)
    baselines = []
    if a.baselines_config:
        baselines = json.loads(Path(a.baselines_config).read_text(encoding='utf-8')).get('baselines', [])
    lock = _make_lock(meta, {'methods': methods, 'baselines': baselines},
                      {'draft_protocol': draft['protocol'], 'draft_guards': draft['guards']})
    _dump(a.out, lock)
    print('Locked', [m['name'] for m in methods], '->', a.out)
    return 0


# --------------------------------------------------------------------------- diagnostics


def cmd_diagnose_masks(a):
    train, meta = fc.load_train(a.cache, views=('blur', 'aug'))
    base = np.asarray(meta['sessions'][0])
    rows = np.isin(train['labels'], base)
    labels = train['labels'][rows]
    feats = train['final'][rows]
    report = {'q': a.q, 'shots': a.shots, 'random_jaccard': {q: random_jaccard(q, feats.shape[1]) for q in a.q}}
    for view in ('blur', 'aug'):
        if view not in train:
            continue
        deltas = (ladder_deltas(feats, train['blur'][rows]) if view == 'blur'
                  else perturbation_deltas(feats, train['aug'][rows]))
        sig = class_signatures(deltas, labels, base)
        ref = sig.mean(0)
        entry = {}
        for q in a.q:
            masks = topq_masks(sig - ref[None], q)
            entry[f'q{q}'] = {'class_specificity': class_specificity(masks),
                              'shot_stability': {f'shot{s}': shot_stability(deltas, labels, base, ref, q, s, a.draws)
                                                 for s in a.shots}}
        report[view] = entry
    if a.keywords:
        lib = KeywordLibrary.load(a.keywords)
        for coord in lib.masks:
            kb = lib.library(base, coord)
            per_class = []
            for c in base:
                kws = [k for k in lib.keywords_of(c) if k in kb]
                m = np.zeros(feats.shape[1], bool)
                for k in kws:
                    m |= kb[k]
                per_class.append(m)
            report[f'keyword_union_{coord}'] = class_specificity(np.stack(per_class)[:, None, :])
    verdict = {}
    for view in ('blur', 'aug'):
        if view in report:
            for q in a.q:
                e = report[view][f'q{q}']
                stable = e['shot_stability'].get('shot5', [0])
                verdict[f'{view}_q{q}'] = bool(max(e['class_specificity']) < 0.5 and max(stable) >= 0.3)
    report['gate_G1prime_pass'] = verdict
    _dump(Path(a.out) / 'mask_diagnostics.json', report)
    print(json.dumps(report, indent=2, default=_json_default)[:4000])
    return 0


def _proto_predict(support_feats, support_labels, classes, query_feats):
    protos = unit(np.stack([support_feats[support_labels == c].mean(0) for c in classes]))
    return unit(query_feats) @ protos.T


def cmd_diagnose_views(a):
    train, meta = fc.load_train(a.cache, views=('blur', 'aug'))
    views = tuple(v for v in ('blur', 'aug') if v in train)
    stats = {}
    count = 0
    overrides = {k: getattr(a, k) for k in ('n_fit', 'way', 'n_sessions', 'query_per_class')}
    for ep in pseudo_episodes(train, meta, shot=a.shot, split_seeds=a.split_seeds, sequences=a.sequences,
                              views=views, **overrides):
        seen = np.concatenate(ep.sessions)
        sup_f = np.concatenate([b.feats for b in ep.support])
        sup_l = np.concatenate([b.labels for b in ep.support])
        ref_scores = _proto_predict(sup_f, sup_l, seen, ep.query.feats)
        ref_pred = ref_scores.argmax(1)
        bimc = run_episode(ep, REFERENCES[1], _defaults(meta), keep_predictions=True)['last_predictions']
        for view in views:
            sup_v = np.concatenate([b.views[view] for b in ep.support])
            for k in range(sup_v.shape[1]):
                view_scores = _proto_predict(sup_v[:, k], sup_l, seen, ep.query.views[view][:, k])
                pred = view_scores.argmax(1)
                fused = (ref_scores + view_scores).argmax(1)
                key = f'{view}{k}'
                s = stats.setdefault(key, [])
                og = oracle_gain(ref_pred, pred, ep.query.labels)
                og['sum_fusion_gain'] = float(np.mean(fused == ep.query.labels) * 100 - og['ref_acc'])
                og['vs_bimc_dagger'] = oracle_gain(bimc, pred, ep.query.labels)['oracle_gain']
                s.append(og)
        count += 1
    report = {k: {m: float(np.nanmean([x[m] for x in v])) for m in v[0]} for k, v in stats.items()}
    report['episodes'] = count
    _dump(Path(a.out) / 'view_diagnostics.json', report)
    print(json.dumps(report, indent=2))
    return 0


def cmd_diagnose_prototypes(a):
    """Novel-prototype quality against full-data (oracle) class means; training images only."""
    lock = fc.load_lock(a.lock)
    specs = list(REFERENCES) + lock['selected'].get('methods', []) + lock['selected'].get('baselines', [])
    resources = _keywords(a.keywords)
    views = _views_for(specs)
    train, meta = fc.load_train(a.cache, views=views)
    order = np.concatenate([np.asarray(s) for s in meta['sessions']])
    oracle = unit(np.stack([train['final'][train['labels'] == c].mean(0) for c in order]))
    report = {}
    for seed in a.seeds:
        ep = benchmark_episode(train, None, meta, a.shot, seed, views=views)
        novel = np.arange(ep.num_base, ep.num_classes)
        for spec in specs:
            r = run_episode(ep, spec, _defaults(meta), resources, return_prototypes=True)
            protos = r['prototypes']
            cos = np.sum(protos[novel] * oracle[novel], axis=1)
            nearest = (protos[novel] @ oracle.T).argmax(1)
            to_base = float(np.mean(nearest < ep.num_base) * 100)
            entry = report.setdefault(spec['name'], {'cos_to_oracle': [], 'self_nearest_pct': [], 'nearest_is_base_pct': []})
            entry['cos_to_oracle'].append(float(cos.mean()))
            entry['self_nearest_pct'].append(float(np.mean(nearest == novel) * 100))
            entry['nearest_is_base_pct'].append(to_base)
    summary = {name: {k: seed_summary(v) for k, v in entry.items()} for name, entry in report.items()}
    _dump(Path(a.out) / 'prototype_diagnostics.json', summary)
    for name, entry in summary.items():
        print(f"{name:14s} cos {entry['cos_to_oracle']['mean']:.4f}  self-NN {entry['self_nearest_pct']['mean']:.1f}%  "
              f"NN-in-base {entry['nearest_is_base_pct']['mean']:.1f}%")
    return 0


def cmd_compare(a):
    payload = json.loads(Path(a.results).read_text(encoding='utf-8'))
    preds = np.load(Path(a.results).parent / 'predictions.npz')
    out = {}
    for seed in payload['seeds']:
        lab = preds[f'__labels__::seed{seed}']
        pa, pb = preds[f'{a.a}::seed{seed}'], preds[f'{a.b}::seed{seed}']
        c = paired_counts(pa, pb, lab)
        c['mcnemar_p'] = mcnemar_exact(c['C'], c['D'])
        c.update(bootstrap_diff(pa, pb, lab, reps=2000, seed=seed))
        out[seed] = c
    print(json.dumps(out, indent=2))
    return 0


# --------------------------------------------------------------------------- CLI


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)

    s = sub.add_parser('parity'); s.add_argument('--cache', required=True); s.add_argument('--out', required=True)
    s.add_argument('--shot', type=int, default=5); s.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    s.add_argument('--expect', type=float); s.add_argument('--tolerance', type=float, default=0.15)

    s = sub.add_parser('dev'); s.add_argument('--cache', required=True); s.add_argument('--config', required=True)
    s.add_argument('--out', required=True); s.add_argument('--keywords'); s.add_argument('--shot', type=int, default=5)
    s.add_argument('--split-seeds', type=int, nargs='+', default=[2, 3, 4]); s.add_argument('--sequences', type=int, default=20)
    s.add_argument('--n-fit', type=int); s.add_argument('--way', type=int); s.add_argument('--n-sessions', type=int)
    s.add_argument('--query-per-class', type=int)
    s.add_argument('--max-base-drop', type=float, default=0.2); s.add_argument('--max-novel-drop', type=float, default=0.2)

    s = sub.add_parser('lock'); s.add_argument('--draft', required=True); s.add_argument('--families', nargs='+', required=True)
    s.add_argument('--out', required=True); s.add_argument('--cache'); s.add_argument('--baselines-config')

    s = sub.add_parser('final'); s.add_argument('--cache', required=True); s.add_argument('--lock', required=True)
    s.add_argument('--out', required=True); s.add_argument('--keywords'); s.add_argument('--shot', type=int, default=5)
    s.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5]); s.add_argument('--reference', default='BiMC_dagger')

    s = sub.add_parser('diagnose-masks'); s.add_argument('--cache', required=True); s.add_argument('--out', required=True)
    s.add_argument('--q', type=int, nargs='+', default=[37, 50]); s.add_argument('--shots', type=int, nargs='+', default=[1, 5])
    s.add_argument('--draws', type=int, default=10); s.add_argument('--keywords')

    s = sub.add_parser('diagnose-views'); s.add_argument('--cache', required=True); s.add_argument('--out', required=True)
    s.add_argument('--shot', type=int, default=5); s.add_argument('--split-seeds', type=int, nargs='+', default=[2, 3, 4])
    s.add_argument('--sequences', type=int, default=5)
    s.add_argument('--n-fit', type=int); s.add_argument('--way', type=int); s.add_argument('--n-sessions', type=int)
    s.add_argument('--query-per-class', type=int)

    s = sub.add_parser('diagnose-prototypes'); s.add_argument('--cache', required=True); s.add_argument('--lock', required=True)
    s.add_argument('--out', required=True); s.add_argument('--keywords'); s.add_argument('--shot', type=int, default=5)
    s.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])

    s = sub.add_parser('compare'); s.add_argument('--results', required=True); s.add_argument('--a', required=True)
    s.add_argument('--b', required=True)
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    handler = {'parity': cmd_parity, 'dev': cmd_dev, 'lock': cmd_lock, 'final': cmd_final,
               'diagnose-masks': cmd_diagnose_masks, 'diagnose-views': cmd_diagnose_views,
               'diagnose-prototypes': cmd_diagnose_prototypes,
               'compare': cmd_compare}[a.command]
    return handler(a)


if __name__ == '__main__':
    raise SystemExit(main())
