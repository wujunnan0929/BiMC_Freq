"""Command line entry point.

    python -m fdrfreq.run smoke    --out outputs/fdrfreq_smoke
    python -m fdrfreq.run parity   --bank BANK --seeds 1 2 3 4 5
    python -m fdrfreq.run diagnose --bank BANK --out DIR
    python -m fdrfreq.run select   --bank BANK --matrix fdrfreq/configs/cub200.json \
                                   --entries A0 A1 A2 --out DIR
    python -m fdrfreq.run evaluate --bank BANK --locked DIR/locked_A2.json ... --out DIR

``select`` never reads test images. ``evaluate`` only accepts locked files
produced by ``select`` and refuses a locked file selected on another bank.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .bank import load_bank, bank_fingerprint
from .protocol import benchmark_episode, check_episode
from .pipeline import EpisodeRunner
from . import select as sel


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), indent=1, ensure_ascii=False), encoding='utf-8')
    return path


def load_matrix(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    return {e['name']: e for e in data['entries']}, data


# ----------------------------------------------------------------- commands
def cmd_parity(a):
    bank = load_bank(a.bank)
    out = {}
    for seed in a.seeds:
        ep = benchmark_episode(bank, seed, shot=a.shot)
        check_episode(ep, bank)
        r = EpisodeRunner(bank, ep)
        out[seed] = {'support_source': ep.meta['support_source']}
        for name, cfg in (('BiMC', {}), ('BiMC_ens', {'ensemble': {}})):
            s = r.run(cfg)['summary']
            out[seed][name] = s
            print(f"seed {seed} {name:9s} avg {s['avg']:.2f} last {s['last']:.2f} "
                  f"sessions {' '.join(f'{v:.2f}' for v in s['sessions'])}")
    if a.out:
        write_json(Path(a.out) / f'parity_shot{a.shot or bank.inc_shot}.json', out)


def cmd_select(a):
    bank = load_bank(a.bank)
    entries, _ = load_matrix(a.matrix)
    names = a.entries or list(entries)
    episodes = sel.dev_episodes(bank, a.split_seeds, a.sequences, a.shot)
    print(f'{len(episodes)} pseudo-incremental episodes (base training images only)')
    out = Path(a.out)
    for name in names:
        entry = entries[name]
        if 'chain' in entry:
            locked = json.loads((out / f"locked_{entry['chain']}.json").read_text(encoding='utf-8'))
            entry = sel.chain(entry, locked)
        rep = sel.select(bank, entry, episodes)
        lk = sel.lock(rep)
        write_json(out / f'select_{name}.json', rep)
        write_json(out / f'locked_{name}.json', lk)
        fb = rep['fallback_metrics']
        best = max(rep['candidates'], key=lambda r: r['last']) if rep['candidates'] else None
        msg = f"[{name}] selected {rep['selected_params']}"
        if best is not None:
            chosen = next((c for c in rep['candidates'] if c['hash'] == rep['selected_hash']), None)
            if chosen:
                msg += f" pseudo-last {chosen['last']:.2f} (base {chosen['last_base']:.2f}, novel {chosen['last_novel']:.2f})"
        if fb:
            msg += f" | fallback {fb['last']:.2f}"
        print(msg, f"[{rep['seconds']:.1f}s]")


def cmd_evaluate(a):
    bank = load_bank(a.bank)
    fp = bank_fingerprint(bank)
    out = Path(a.out)
    for path in a.locked:
        lk = json.loads(Path(path).read_text(encoding='utf-8'))
        if lk.get('bank') != fp and not a.allow_other_bank:
            raise SystemExit(f'{path} was selected on bank {lk.get("bank")}, not {fp}.')
        if sel.config_hash(lk['config']) != lk['config_hash']:
            raise SystemExit(f'{path} was edited after locking (hash mismatch).')
        res = {'name': lk['name'], 'config_hash': lk['config_hash'], 'bank': fp,
               'dataset': bank.meta.get('dataset'), 'shot': a.shot or bank.inc_shot, 'seeds': {}}
        preds = {}
        for seed in a.seeds:
            ep = benchmark_episode(bank, seed, shot=a.shot)
            check_episode(ep, bank)
            t0 = time.time()
            r = EpisodeRunner(bank, ep).run(lk['config'])
            res['seeds'][str(seed)] = {'summary': r['summary'], 'per_session': r['per_session'],
                                       'support_source': ep.meta['support_source'],
                                       'seconds': time.time() - t0}
            preds[f'pred_seed{seed}'] = r['last_pred']
            preds[f'labels_seed{seed}'] = r['last_labels']
        S = [res['seeds'][str(s)]['summary'] for s in a.seeds]
        res['mean'] = {k: float(np.mean([s[k] for s in S])) for k in ('avg', 'last', 'last_base', 'last_novel', 'last_hm', 'pd')}
        res['std'] = {k: float(np.std([s[k] for s in S])) for k in ('avg', 'last', 'last_base', 'last_novel', 'last_hm', 'pd')}
        shot = a.shot or bank.inc_shot
        write_json(out / f"eval_{lk['name']}_shot{shot}.json", res)
        np.savez(out / f"pred_{lk['name']}_shot{shot}.npz", **preds)
        m, s = res['mean'], res['std']
        print(f"[{lk['name']}] avg {m['avg']:.2f}±{s['avg']:.2f} last {m['last']:.2f}±{s['last']:.2f} "
              f"base {m['last_base']:.2f} novel {m['last_novel']:.2f} hm {m['last_hm']:.2f}")


def cmd_diagnose(a):
    from .diagnose import run_diagnostics
    bank = load_bank(a.bank)
    rep = run_diagnostics(bank, split_seeds=a.split_seeds, n_sequences=a.sequences, shot=a.shot,
                          q=a.q, sigmas=a.sigmas)
    write_json(Path(a.out) / f'diagnostics_shot{a.shot or bank.inc_shot}.json', rep)
    from .diagnose import print_summary
    print_summary(rep)


def _dev_metrics(out, name):
    f = Path(out) / f'select_{name}.json'
    if not f.exists():
        return None
    rep = json.loads(f.read_text(encoding='utf-8'))
    for c in rep['candidates']:
        if c['hash'] == rep['selected_hash']:
            return c
    fb = rep.get('fallback_metrics')
    if fb:
        return {'last': fb['last'], 'last_base': fb['last_base'], 'last_novel': fb['last_novel'],
                'per_split_last': None, 'params': 'fallback'}
    return None


def gate_report(out):
    """Development-only go/no-go verdicts (no test image is involved)."""
    m = {n: _dev_metrics(out, n) for n in ('BiMC_ens', 'A1', 'A2', 'K0', 'K1', 'K2_random', 'Q1',
                                           'Q2_classperm', 'Q3_random', 'Q4', 'Q4_shuffled', 'F1',
                                           'F3_aug', 'F4_noise', 'F5_randband', 'F6_blurdiff', 'R1')}
    diag = None
    for f in Path(out).glob('diagnostics_shot*.json'):
        diag = json.loads(f.read_text(encoding='utf-8'))

    def d(a, b, key='last'):
        if m.get(a) is None or m.get(b) is None:
            return None
        return m[a][key] - m[b][key]

    def splits_pos(a, b):
        pa, pb = (m.get(a) or {}).get('per_split_last'), (m.get(b) or {}).get('per_split_last')
        if not pa or not pb:
            return None
        return sum(x > y for x, y in zip(pa, pb)), len(pa)

    rows = []

    def add(gate, ok, detail):
        rows.append((gate, 'PASS' if ok else ('n/a' if ok is None else 'FAIL'), detail))

    v = d('A2', 'BiMC_ens')
    add('G0 anchor (A2 > BiMC-dagger on dev)', None if v is None else v > 0.5,
        f'dLast {v:+.2f}' if v is not None else 'missing')
    v1, v2 = d('K1', 'K0'), d('K1', 'K2_random')
    ok = None if v1 is None or v2 is None else (v1 >= 0.2 and v2 >= 0.1)
    add("G0' FDR control (K1-K0 >= +0.2, K1-K2 >= +0.1)", ok,
        f'K1-K0 {v1:+.2f}, K1-K2 {v2:+.2f}, splits {splits_pos("K1", "K0")}' if ok is not None else 'missing')
    if diag and 'freq_mask_class_jaccard_per_band' in diag:
        jac = max(diag['freq_mask_class_jaccard_per_band'])
        stab = min(diag['freq_mask_shot_stability_per_band'])
        add("G1' frequency masks (class Jaccard < 0.5, shot stability >= 0.3)", jac < 0.5 and stab >= 0.3,
            f'max Jaccard {jac:.3f} (random {diag["random_mask_jaccard"]:.3f}), min stability {stab:.3f}')
    if diag and 'D3_gda_last' in diag:
        D3 = diag['D3_gda_last']
        dct = {k: v for k, v in D3.items() if k.startswith('dct_') and k.endswith('radial')}
        ctrl = max([D3[k]['delta_vs_cls'] for k in D3 if k.startswith(('aug_', 'noise')) or k.endswith('random')]
                   or [0.0])
        best = max(dct.items(), key=lambda kv: kv[1]['delta_vs_cls']) if dct else None
        if best:
            k, b = best
            ok = b['delta_vs_cls'] >= 0.2 and b['splits_positive'] >= 2 and b['delta_vs_cls'] - ctrl >= 0.1
            add('G1 information bound (stand-alone joint GDA)', ok,
                f'{k} {b["delta_vs_cls"]:+.2f} (splits+ {b["splits_positive"]}), best control {ctrl:+.2f}')
    for meth, fb, ctrls in (('Q1', 'K0', ('Q2_classperm', 'Q3_random')),
                            ('Q4', 'K1', ('Q4_shuffled',)),
                            ('F1', 'A2', ('F3_aug', 'F4_noise', 'F5_randband'))):
        v = d(meth, fb)
        cs = [d(meth, c) for c in ctrls]
        if v is None or any(c is None for c in cs):
            add(f'G2 {meth}', None, 'missing')
            continue
        ok = v >= 0.3 and min(cs) >= 0.2
        add(f'G2 {meth} (vs {fb} >= +0.3, vs controls >= +0.2)', ok,
            f'vs {fb} {v:+.2f}; ' + ', '.join(f'vs {c} {x:+.2f}' for c, x in zip(ctrls, cs)))
    lines = ['Development gates (pseudo-incremental, base training images only)']
    lines += [f'  [{s:4s}] {g}: {det}' for g, s, det in rows]
    return '\n'.join(lines)


def cmd_gates(a):
    text = gate_report(a.dir)
    print(text)
    Path(a.dir, 'gates.txt').write_text(text, encoding='utf-8')


def cmd_smoke(a):
    from .synthetic import make_bank
    from .bank import save_bank
    out = Path(a.out)
    b = make_bank()
    bank_dir = out / 'bank'
    save_bank(bank_dir, b._arrays, b.meta, overwrite=True)
    matrix = Path(__file__).with_name('configs') / 'synthetic.json'
    common = ['--bank', str(bank_dir), '--out', str(out)]
    main(['select', *common, '--matrix', str(matrix), '--split-seeds', '2', '3', '--sequences', '3'])
    entries, _ = load_matrix(matrix)
    main(['evaluate', *common, '--seeds', '1', '2', '3', '--locked',
          *[str(out / f'locked_{n}.json') for n in entries]])
    main(['diagnose', *common, '--split-seeds', '2', '--sequences', '2', '--q', '8'])
    main(['gates', '--dir', str(out)])
    from .report import build_report
    print(build_report(out, anchor='A2', fmt='md'))


def main(argv=None):
    p = argparse.ArgumentParser(prog='fdrfreq')
    sub = p.add_subparsers(dest='cmd', required=True)

    def common(sp, out_required=True):
        sp.add_argument('--bank', required=True)
        sp.add_argument('--out', required=out_required)
        sp.add_argument('--shot', type=int, default=None)

    sp = sub.add_parser('parity')
    common(sp, out_required=False)
    sp.add_argument('--seeds', type=int, nargs='+', default=[1])
    sp.set_defaults(func=cmd_parity)

    sp = sub.add_parser('select')
    common(sp)
    sp.add_argument('--matrix', required=True)
    sp.add_argument('--entries', nargs='*')
    sp.add_argument('--split-seeds', type=int, nargs='+', default=[2, 3, 4])
    sp.add_argument('--sequences', type=int, default=10)
    sp.set_defaults(func=cmd_select)

    sp = sub.add_parser('evaluate')
    common(sp)
    sp.add_argument('--locked', nargs='+', required=True)
    sp.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    sp.add_argument('--allow-other-bank', action='store_true')
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser('diagnose')
    common(sp)
    sp.add_argument('--split-seeds', type=int, nargs='+', default=[2, 3, 4])
    sp.add_argument('--sequences', type=int, default=5)
    sp.add_argument('--q', type=int, default=50)
    sp.add_argument('--sigmas', type=float, nargs='+', default=[1, 2, 4])
    sp.set_defaults(func=cmd_diagnose)

    sp = sub.add_parser('gates')
    sp.add_argument('--dir', required=True)
    sp.set_defaults(func=cmd_gates)

    sp = sub.add_parser('smoke')
    sp.add_argument('--out', default='outputs/fdrfreq_smoke')
    sp.set_defaults(func=cmd_smoke)

    a = p.parse_args(argv)
    a.func(a)
    return 0


if __name__ == '__main__':
    sys.exit(main())
