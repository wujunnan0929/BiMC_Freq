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
    episodes = sel.dev_episodes(bank, a.split_seeds, a.sequences, a.shot, a.split_mode)
    print(f'{len(episodes)} pseudo-incremental episodes (base training images only)')
    out = Path(a.out)
    for name in names:
        entry = entries[name]
        if 'chain' in entry:
            locked = json.loads((out / f"locked_{entry['chain']}.json").read_text(encoding='utf-8'))
            entry = sel.chain(entry, locked)
        rep = sel.select(bank, entry, episodes, tolerance=a.tolerance, objective=a.objective)
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
                if chosen.get('channels'):
                    msg += f" ch/class {chosen['channels']:.0f}"
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
                          q=a.q, sigmas=a.sigmas, split_mode=a.split_mode)
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
                'last_hm': fb.get('last_hm', float('nan')), 'channels': 0.0,
                'per_split_last': None, 'params': 'fallback'}
    return None


def gate_report(out):
    """Development-only go/no-go verdicts (no test image is involved)."""
    m = {n: _dev_metrics(out, n) for n in ('BiMC_ens', 'A1', 'A2', 'K0', 'K1', 'K2_random', 'Q1',
                                           'Q6_union', 'Q6_random', 'Q6_var', 'Q6_mag', 'Q6_fisher',
                                           'Q7_eb', 'Q7_uniform', 'Q7_shuffled', 'Q7e_eb',
                                           'M1', 'M1_lowvar', 'M1_mag', 'M1_fisher', 'M1_freq', 'M1_ebratio',
                                           'M1_random', 'M1_noholistic', 'M1_global', 'M1_visualsim',
                                           'M2_ebmask', 'M3', 'M3_lowvar', 'M3_mag', 'M3_fisher', 'M3_freq',
                                           'M3_ebratio', 'M3_random', 'M3_global', 'M3_visualsim', 'M3_holistic',
                                           'Proto', 'BiMC', 'P_Proto', 'P_BiMC', 'Ours_ULA_M3',
                                           'M4', 'M4_lowvar', 'M4_mag', 'M4_fisher', 'M4_freq', 'M4_ebratio',
                                           'M4_random', 'M4_segment', 'M4_visualsim', 'M4_holistic', 'M4_soft',
                                           'M4_ebmask', 'P4_Proto', 'P4_BiMC', 'Ours_ULA_M4',
                                           'M4a', 'M4a_lowvar', 'M4a_mag', 'M4a_fisher', 'M4a_freq', 'M4a_ebratio',
                                           'M4a_random', 'M4a_text', 'M4a_segment', 'M4a_holistic', 'M4a_soft',
                                           'M4a_ebmask', 'P4a_Proto', 'P4a_BiMC', 'Ours_ULA_M4a',
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
                            ('Q6_union', 'K0', ('Q6_random', 'Q6_var', 'Q6_mag', 'Q6_fisher')),
                            ('Q7_eb', 'K0', ('Q7_uniform', 'Q7_shuffled', 'Q6_var')),
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
    # G3: variance-guided segment recomposition against matched-budget channel sets
    v = d('M1', 'K0')
    hard = ('M1_random', 'M1_lowvar', 'M1_fisher')
    soft = ('M1_mag', 'M1_freq', 'M1_ebratio', 'Q6_var')
    cs = [d('M1', c) for c in hard]
    if v is None or any(c is None for c in cs):
        add('G3 M1', None, 'missing')
    else:
        info = [f'vs {c} {d("M1", c):+.2f}' for c in soft if d('M1', c) is not None]
        add('G3 M1 (vs K0 >= +0.3, vs random/lowvar/fisher >= +0.2)', v >= 0.3 and min(cs) >= 0.2,
            f'vs K0 {v:+.2f} (HM {d("M1", "K0", "last_hm"):+.2f}, splits {splits_pos("M1", "K0")}); '
            + ', '.join(f'vs {c} {x:+.2f}' for c, x in zip(hard, cs))
            + ('; info ' + ', '.join(info) if info else ''))
    comp = [(c, d('M1', c)) for c in ('M1_noholistic', 'M1_global', 'M1_visualsim')]
    if all(x is not None for _, x in comp):
        add('G3c M1 components (each removal costs >= +0.1)', min(x for _, x in comp) >= 0.1,
            ', '.join(f'M1-{c} {x:+.2f}' for c, x in comp))
    v = d('M2_ebmask', 'M1')
    if v is not None:
        ch = [(m.get(n) or {}).get('channels') for n in ('M1', 'M2_ebmask')]
        chs = '' if None in ch else f' (channels/class M1 {ch[0]:.0f}, M2 {ch[1]:.0f})'
        add('G4 shot-adaptive EB rule (M2 >= M1 - 0.1: prefer M2, no k)', v >= -0.1,
            f'M2-M1 {v:+.2f}, M2-K0 {d("M2_ebmask", "K0"):+.2f}' + chs)
    v = d('Q7e_eb', 'Q7_eb')
    if v is not None:
        add('G4b soft EB, measured K-shot variance (info)', None if d('Q7e_eb', 'M1') is None else d('Q7e_eb', 'M1') >= -0.1,
            f'Q7e-Q7 {v:+.2f}, Q7e-K0 {d("Q7e_eb", "K0"):+.2f}, Q7e-M1 {d("Q7e_eb", "M1"):+.2f}'
            if d('Q7e_eb', 'M1') is not None else f'Q7e-Q7 {v:+.2f}')
    # G5: final candidate M3 (sec 12.8) on top of the anchor, no holistic step
    v = d('M3', 'A2')
    hard = ('M3_random', 'M3_lowvar', 'M3_fisher')
    cs = [d('M3', c) for c in hard]
    if v is None or any(c is None for c in cs):
        add('G5 M3', None, 'missing')
    else:
        info = [f'vs {c} {d("M3", c):+.2f}' for c in ('M3_mag', 'M3_freq', 'M3_ebratio', 'M1', 'Q7e_eb')
                if d('M3', c) is not None]
        add('G5 M3 (vs A2 >= +0.3, vs random/lowvar/fisher >= +0.2)', v >= 0.3 and min(cs) >= 0.2,
            f'vs A2 {v:+.2f} (HM {d("M3", "A2", "last_hm"):+.2f}, splits {splits_pos("M3", "A2")}); '
            + ', '.join(f'vs {c} {x:+.2f}' for c, x in zip(hard, cs)) + ('; info ' + ', '.join(info) if info else ''))
    for gate, a_, b_ in (('G5c keep segment-local donors (M3 - M3_global >= +0.1)', 'M3', 'M3_global'),
                         ('G5c keep text donor similarity (M3 - M3_visualsim >= +0.1)', 'M3', 'M3_visualsim'),
                         ('G5c add holistic back (M3_holistic - M3 >= +0.1)', 'M3_holistic', 'M3')):
        v = d(a_, b_)
        if v is not None:
            add(gate, v >= 0.1, f'{a_}-{b_} {v:+.2f} (HM {d(a_, b_, "last_hm"):+.2f})')
    pl = [(x, y) for x, y in (('P_Proto', 'Proto'), ('P_BiMC', 'BiMC'), ('Ours_ULA_M3', 'M3')) if d(x, y) is not None]
    if pl:
        add('G5p plug-in / ULA on top (info)', None, ', '.join(f'{x}-{y} {d(x, y):+.2f}' for x, y in pl))
    # G6: final method M4 (sec 12.9)
    v = d('M4', 'A2')
    hard = ('M4_random', 'M4_lowvar', 'M4_fisher')
    cs = [d('M4', c) for c in hard]
    if v is None or any(c is None for c in cs):
        add('G6 M4', None, 'missing')
    else:
        info = [f'vs {c} {d("M4", c):+.2f}' for c in ('M4_mag', 'M4_freq', 'M4_ebratio', 'M3', 'M1')
                if d('M4', c) is not None]
        add('G6 M4 (vs A2 >= +0.3, vs random/lowvar/fisher >= +0.2)', v >= 0.3 and min(cs) >= 0.2,
            f'{m["M4"].get("params")}; vs A2 {v:+.2f} (HM {d("M4", "A2", "last_hm"):+.2f}, splits {splits_pos("M4", "A2")}); '
            + ', '.join(f'vs {c} {x:+.2f}' for c, x in zip(hard, cs)) + ('; info ' + ', '.join(info) if info else ''))
    for gate, a_, b_ in (('G6c replace beats shrink-all (M4 - M4_soft >= +0.1)', 'M4', 'M4_soft'),
                         ('G6c keep text donor similarity (M4 - M4_visualsim >= +0.1)', 'M4', 'M4_visualsim'),
                         ('G6c switch to segment-ranked donors (M4_segment - M4 >= +0.1)', 'M4_segment', 'M4'),
                         ('G6c add holistic back (M4_holistic - M4 >= +0.1)', 'M4_holistic', 'M4')):
        v = d(a_, b_)
        if v is not None:
            add(gate, v >= 0.1, f'{a_}-{b_} {v:+.2f} (HM {d(a_, b_, "last_hm"):+.2f})')
    v = d('M4_ebmask', 'M4')
    if v is not None:
        add('G6e EB channel rule (M4_ebmask >= M4 - 0.1; adopt only if the same c wins at every shot)', v >= -0.1,
            f'M4_ebmask-M4 {v:+.2f}, selected {m["M4_ebmask"].get("params")}, channels/class '
            f'{m["M4_ebmask"].get("channels", 0):.0f} vs {m["M4"].get("channels", 0):.0f}')
    pl = [(x, y) for x, y in (('P4_Proto', 'Proto'), ('P4_BiMC', 'BiMC'), ('Ours_ULA_M4', 'M4')) if d(x, y) is not None]
    if pl:
        add('G6p plug-in / ULA on top (info)', None, ', '.join(f'{x}-{y} {d(x, y):+.2f}' for x, y in pl))
    # G7: frozen final method M4a (sec 12.11). Everything below is reported, nothing changes the method.
    v = d('M4a', 'A2')
    hard = ('M4a_random', 'M4a_lowvar', 'M4a_fisher')
    cs = [d('M4a', c) for c in hard]
    if v is None or any(c is None for c in cs):
        add('G7 M4a', None, 'missing')
    else:
        info = [f'vs {c} {d("M4a", c):+.2f}' for c in ('M4a_mag', 'M4a_freq', 'M4a_ebratio', 'M4', 'M3')
                if d('M4a', c) is not None]
        add('G7 M4a (vs A2 >= +0.3, vs random/lowvar/fisher >= +0.2)', v >= 0.3 and min(cs) >= 0.2,
            f'{m["M4a"].get("params")}; vs A2 {v:+.2f} (HM {d("M4a", "A2", "last_hm"):+.2f}, splits {splits_pos("M4a", "A2")}); '
            + ', '.join(f'vs {c} {x:+.2f}' for c, x in zip(hard, cs)) + ('; info ' + ', '.join(info) if info else ''))
    for gate, a_, b_ in (('G7c claim: replace beats shrink-all (M4a - M4a_soft >= +0.1)', 'M4a', 'M4a_soft'),
                         ('G7c info: text donor similarity (M4a_text - M4a)', 'M4a_text', 'M4a'),
                         ('G7c info: segment-ranked donors (M4a_segment - M4a)', 'M4a_segment', 'M4a'),
                         ('G7c info: holistic step on top (M4a_holistic - M4a)', 'M4a_holistic', 'M4a')):
        v = d(a_, b_)
        if v is not None:
            add(gate, v >= 0.1 if gate.startswith('G7c claim') else None,
                f'{a_}-{b_} {v:+.2f} (HM {d(a_, b_, "last_hm"):+.2f})')
    v = d('M4a_ebmask', 'M4a')
    if v is not None:
        add('G7e EB channel rule (analysis only)', None,
            f'M4a_ebmask-M4a {v:+.2f}, selected {m["M4a_ebmask"].get("params")}, channels/class '
            f'{m["M4a_ebmask"].get("channels", 0):.0f} vs {m["M4a"].get("channels", 0):.0f}')
    pl = [(x, y) for x, y in (('P4a_Proto', 'Proto'), ('P4a_BiMC', 'BiMC'), ('Ours_ULA_M4a', 'M4a'))
          if d(x, y) is not None]
    if pl:
        add('G7p plug-in / ULA on top (info)', None, ', '.join(f'{x}-{y} {d(x, y):+.2f}' for x, y in pl))
    lines = ['Development gates (pseudo-incremental, base training images only)']
    lines += [f'  [{s:4s}] {g}: {det}' for g, s, det in rows]
    return '\n'.join(lines)


def cmd_summary(a):
    """Selected dev metrics of each entry (what used to be pasted as a Python snippet)."""
    lines = []
    for d in a.dirs:
        lines.append(f'==== {d}')
        names = a.entries or sorted(f.stem[len('select_'):] for f in Path(d).glob('select_*.json'))
        for n in names:
            f = Path(d) / f'select_{n}.json'
            if not f.exists():
                lines.append(f'{n:14s} missing')
                continue
            r = json.loads(f.read_text(encoding='utf-8'))
            c = next((c for c in r['candidates'] if c['hash'] == r['selected_hash']), None)
            if c is None:
                fb = r.get('fallback_metrics') or {}
                lines.append(f"{n:14s} last {fb.get('last', float('nan')):.2f} HM {fb.get('last_hm', float('nan')):.2f} fallback")
                continue
            lines.append(f"{n:14s} last {c['last']:.2f} B {c['last_base']:.2f} N {c['last_novel']:.2f} "
                         f"HM {c['last_hm']:.2f} ch {c.get('channels', 0):.0f} "
                         f"splits {[round(x, 2) for x in c['per_split_last']]} {r['selected_params']}")
    text = '\n'.join(lines)
    print(text)
    if a.out:
        Path(a.out).write_text(text + '\n', encoding='utf-8')


def cmd_gates(a):
    text = gate_report(a.dir)
    print(text)
    Path(a.dir, 'gates.txt').write_text(text, encoding='utf-8')


PROBE_CONFIGS = {
    'BiMC': {},
    'BiMC_ens': {'ensemble': {}},
    'GDA_only': {'anchor': {'omega': 1.0, 'fusion': 'gda_only', 'temperature': 1.0}},
    'GDA_w0.05_T1': {'ensemble': {}, 'anchor': {'omega': 0.05, 'temperature': 1.0}},
    'GDA_w0.1_T1': {'ensemble': {}, 'anchor': {'omega': 0.1, 'temperature': 1.0}},
    'GDA_w0.2_T0.3': {'ensemble': {}, 'anchor': {'omega': 0.2, 'temperature': 0.3}},
    'K0_lam0.2': {'ensemble': {}, 'holistic': {'donor': 'text', 'lam': 0.2, 'tau': 16.0}},
}


def cmd_probe(a):
    """Protocol fidelity: do fixed reference configs move base/novel the same way on
    pseudo episodes (both split modes) as on the benchmark? No selection happens here."""
    from .metrics import error_types
    bank = load_bank(a.bank)
    out = {}
    sources = [('random', None), ('contiguous', None)] + ([('benchmark', None)] if a.with_test else [])
    for mode, _ in sources:
        if mode == 'benchmark':
            eps = [benchmark_episode(bank, s, shot=a.shot) for s in (1, 2, 3)]
        else:
            eps = sel.dev_episodes(bank, a.split_seeds, a.sequences, a.shot, mode)
        rows = {}
        runners = [EpisodeRunner(bank, e) for e in eps]
        for name, cfg in PROBE_CONFIGS.items():
            S, E = [], []
            for r, e in zip(runners, eps):
                res = r.run(cfg)
                S.append(res['summary'])
                E.append(error_types(res['last_pred'], res['last_labels'], e.classes[0]))
            rows[name] = {k: float(np.mean([s[k] for s in S])) for k in ('last', 'last_base', 'last_novel')}
            rows[name].update({k: float(np.mean([x[k] for x in E])) for k in E[0]})
        out[mode] = rows
        print(f'== {mode} ({len(eps)} episodes)')
        ref = rows['BiMC_ens']
        for name, v in rows.items():
            print(f"  {name:14s} last {v['last']:6.2f} ({v['last'] - ref['last']:+.2f})  "
                  f"base {v['last_base']:6.2f} ({v['last_base'] - ref['last_base']:+.2f})  "
                  f"novel {v['last_novel']:6.2f} ({v['last_novel'] - ref['last_novel']:+.2f})  "
                  f"n->b {v['novel_to_base']:5.1f} b->n {v['base_to_novel']:5.1f}")
    write_json(Path(a.out) / f'probe_shot{a.shot or bank.inc_shot}.json', out)


def cmd_frontier(a):
    """Base/novel trade-off of segment-wise recomposition per decomposition axis.

    Every axis is swept over the same strengths (and channel budgets q) on top of
    a locked configuration (usually K0), on base-only pseudo episodes. Compare
    axes at matched base cost or matched channel coverage, not at one alpha.
    """
    from .pipeline import deep_merge
    bank = load_bank(a.bank)
    locked = json.loads(Path(a.base_locked).read_text(encoding='utf-8'))['config']
    if a.alpha_from_locked:
        a.alphas = [float((locked.get('segments') or {}).get('alpha', 1.0))]
    # the reference is always the locked config itself; the override only changes what is swept
    base = deep_merge(locked, json.loads(Path(a.override).read_text(encoding='utf-8'))) if a.override else locked
    eps = sel.dev_episodes(bank, a.split_seeds, a.sequences, a.shot, a.split_mode)
    runners = [EpisodeRunner(bank, e) for e in eps]
    ref = [r.run(locked)['summary'] for r in runners]
    rb = {k: float(np.mean([x[k] for x in ref])) for k in ('last', 'last_base', 'last_novel', 'last_hm')}
    print(f"reference ({len(eps)} episodes): last {rb['last']:.2f} base {rb['last_base']:.2f} "
          f"novel {rb['last_novel']:.2f} hm {rb['last_hm']:.2f}")
    rows = []
    for axis in a.axes:
        prior = axis.startswith('prior:')
        if prior:
            qs = a.k
        else:
            qs = a.q if (axis.startswith('freq') or axis.endswith('_union')) else [a.q[0]]
        for q in qs:
            for alpha in a.alphas:
                if prior:
                    seg = {'type': 'prior', 'rank': axis.split(':', 1)[1], 'k': q,
                           'alpha': alpha, 'c': q, 's2': 'empirical'}
                else:
                    seg = {'type': axis, 'q': q, 'alpha': alpha, 'sim': 'text',
                           'sigmas': [1, 2, 4], 'bands': [0, 1, 2]}
                # r / tau / sim / scope are inherited from the locked config unless given
                if a.r is not None:
                    seg['r'] = float(a.r) if a.r <= 1 else int(a.r)
                if a.tau is not None:
                    seg['tau'] = a.tau
                cfg = deep_merge(base, {'segments': seg})
                S, cov = [], []
                for r in runners:
                    res = r.run(cfg)
                    S.append(res['summary'])
                    if res['segment_stats']:
                        cov.append(res['segment_stats']['channels_touched'])
                m = {k: float(np.mean([x[k] for x in S])) for k in ('last', 'last_base', 'last_novel', 'last_hm')}
                row = {'axis': axis, 'q': q, 'alpha': alpha, **m,
                       'd_last': m['last'] - rb['last'], 'd_base': m['last_base'] - rb['last_base'],
                       'd_novel': m['last_novel'] - rb['last_novel'],
                       'channels': float(np.mean(cov)) if cov else 0.0}
                rows.append(row)
                print(f"{axis:16s} q{q:<5} a{alpha:<5} ch {row['channels']:6.1f}  dLast {row['d_last']:+.2f} "
                      f"dBase {row['d_base']:+.2f} dNovel {row['d_novel']:+.2f} HM {m['last_hm']:.2f}")
    write_json(Path(a.out) / f'frontier_shot{a.shot or bank.inc_shot}.json', {'reference': rb, 'rows': rows})


def cmd_errconc(a):
    from .diagnose import error_concentration
    bank = load_bank(a.bank)
    out = {}
    for shot in a.shots:
        res = error_concentration(bank, a.split_seeds, a.sequences, shot, a.q, a.sigmas, a.split_mode)
        out[str(shot)] = res
        print(f'== {shot}-shot (pseudo-novel prototypes, q={a.q} per band)')
        for k, v in res.items():
            if k.startswith('_'):
                continue
            print(f"  {k:22s} coverage {100 * v['coverage']:5.1f}%  error captured {100 * v['capture']:5.1f}%  "
                  f"concentration x{v['concentration']:.2f}")
        print('  overlap (Jaccard, same size): ' + ', '.join(f'{k} {v:.2f}' for k, v in res['_overlap_jaccard'].items()))
        cur = res['_capture_curve']
        fr = list(next(iter(cur.values())))
        print('  error captured vs channel budget: ' + ' '.join(f'{float(f):>6.2f}' for f in fr))
        for n, row in cur.items():
            print(f'    {n:8s}' + ' '.join(f'{100 * row[f]:5.1f}%' for f in fr))
    write_json(Path(a.out) / 'error_concentration.json', out)


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
    sp.add_argument('--split-mode', default='random', choices=('random', 'contiguous'))
    sp.add_argument('--tolerance', type=float, help='override every entry tolerance (pp)')
    sp.add_argument('--objective', default='last', choices=('last', 'hm', 'avg'),
                    help='pseudo last-session metric maximised by the selection')
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
    sp.add_argument('--split-mode', default='random', choices=('random', 'contiguous'))
    sp.set_defaults(func=cmd_diagnose)

    sp = sub.add_parser('probe')
    common(sp)
    sp.add_argument('--split-seeds', type=int, nargs='+', default=[2, 3, 4])
    sp.add_argument('--sequences', type=int, default=5)
    sp.add_argument('--with-test', action='store_true',
                    help='also score the same fixed reference configs on the benchmark (seeds 1-3)')
    sp.set_defaults(func=cmd_probe)

    sp = sub.add_parser('frontier')
    common(sp)
    sp.add_argument('--base-locked', required=True, help='locked_*.json to start from (e.g. locked_K0.json)')
    sp.add_argument('--axes', nargs='+', default=['freq_union', 'var_union', 'mag_union', 'fisher_union',
                                                  'random_union', 'freq', 'freq_classperm', 'freq_common', 'freq_random', 'keyword',
                                                  'keyword_fdr', 'keyword_random'])
    sp.add_argument('--alphas', type=float, nargs='+', default=[0.05, 0.1, 0.2, 0.3, 0.5])
    sp.add_argument('--q', type=int, nargs='+', default=[50, 100, 150])
    sp.add_argument('--k', type=float, nargs='+', default=[0.05, 0.1, 0.2, 0.3, 0.4],
                    help="channel budget (fraction of d) for 'prior:<rank>' axes; for prior:ebmask it is c")
    sp.add_argument('--r', type=float, default=None, help='donors (<=1: fraction of base classes); default: from the locked config')
    sp.add_argument('--tau', type=float, default=None, help='default: from the locked config')
    sp.add_argument('--override', default=None,
                    help='JSON file deep-merged on top of the locked config (e.g. when the locked config is a fallback)')
    sp.add_argument('--alpha-from-locked', action='store_true',
                    help="sweep only the locked segments' alpha (1.0 if the locked config has no segment step)")
    sp.add_argument('--split-seeds', type=int, nargs='+', default=[2, 3, 4])
    sp.add_argument('--sequences', type=int, default=10)
    sp.add_argument('--split-mode', default='contiguous', choices=('random', 'contiguous'))
    sp.set_defaults(func=cmd_frontier)

    sp = sub.add_parser('errconc')
    sp.add_argument('--bank', required=True)
    sp.add_argument('--out', required=True)
    sp.add_argument('--shots', type=int, nargs='+', default=[1, 5])
    sp.add_argument('--q', type=int, default=50)
    sp.add_argument('--sigmas', type=float, nargs='+', default=[1, 2, 4])
    sp.add_argument('--split-seeds', type=int, nargs='+', default=[2, 3, 4])
    sp.add_argument('--sequences', type=int, default=5)
    sp.add_argument('--split-mode', default='contiguous', choices=('random', 'contiguous'))
    sp.set_defaults(func=cmd_errconc)

    sp = sub.add_parser('summary')
    sp.add_argument('--dirs', nargs='+', required=True)
    sp.add_argument('--entries', nargs='*')
    sp.add_argument('--out')
    sp.set_defaults(func=cmd_summary)

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
