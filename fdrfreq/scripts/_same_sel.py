"""PAPER_PLAN 13.9 R4 check: do two selections lock the same parameters with the same
development metrics? python _same_sel.py select_A.json select_B.json"""
import json
import sys


def chosen(path):
    rep = json.load(open(path, encoding='utf-8'))
    if rep['selected_params'] == 'fallback':
        return 'fallback', rep['fallback_metrics'], None
    c = next(c for c in rep['candidates'] if c['hash'] == rep['selected_hash'])
    return c['params'], {k: c[k] for k in ('last', 'last_base', 'last_novel', 'last_hm')}, c['per_split_last']


a, b = chosen(sys.argv[1]), chosen(sys.argv[2])
same = a[0] == b[0] and all(abs(a[1][k] - b[1][k]) < 1e-9 for k in a[1]) and a[2] == b[2]
print(('R4 check: identical ' if same else 'R4 check: DIFFERENT ') + f'{sys.argv[1]} {a[0]} | {sys.argv[2]} {b[0]}')
