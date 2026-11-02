"""Session metrics, error decomposition and paired statistics."""

from math import comb

import numpy as np


def harmonic(a, b):
    return 0.0 if a + b == 0 else 2 * a * b / (a + b)


def session_metrics(preds, labels, num_base, session_index):
    preds, labels = np.asarray(preds), np.asarray(labels)
    acc = float(np.mean(preds == labels) * 100) if len(labels) else float('nan')
    base = labels < num_base
    out = {'session': session_index, 'acc': acc, 'n': int(len(labels)),
           'base_acc': float(np.mean(preds[base] == labels[base]) * 100) if base.any() else float('nan')}
    if session_index > 0 and (~base).any():
        out['novel_acc'] = float(np.mean(preds[~base] == labels[~base]) * 100)
        out['h'] = harmonic(out['base_acc'], out['novel_acc'])
    return out


def summarize(sessions):
    accs = [s['acc'] for s in sessions]
    last = sessions[-1]
    out = {'avg': float(np.mean(accs)), 'last': float(accs[-1]), 'first': float(accs[0]),
           'pd': float(accs[0] - accs[-1]), 'sessions': [float(a) for a in accs],
           'last_base': last.get('base_acc'), 'last_novel': last.get('novel_acc'), 'last_h': last.get('h')}
    return out


def error_decomposition(preds, labels, num_base):
    """Fractions of errors at the final session by source/target group."""
    preds, labels = np.asarray(preds), np.asarray(labels)
    wrong = preds != labels
    base_true, base_pred = labels < num_base, preds < num_base
    novel_err = wrong & ~base_true
    base_err = wrong & base_true
    def frac(mask, total):
        return float(mask.sum() / max(total.sum(), 1) * 100)
    return {'novel_errors': int(novel_err.sum()), 'base_errors': int(base_err.sum()),
            'novel_to_base_pct': frac(novel_err & base_pred, novel_err),
            'novel_to_novel_pct': frac(novel_err & ~base_pred, novel_err),
            'base_to_novel_pct': frac(base_err & ~base_pred, base_err),
            'base_to_base_pct': frac(base_err & base_pred, base_err)}


def paired_counts(pred_a, pred_b, labels):
    """C: a wrong -> b right (corrected); D: a right -> b wrong (damaged); W: both wrong, changed."""
    a_ok, b_ok = pred_a == labels, pred_b == labels
    return {'C': int(np.sum(~a_ok & b_ok)), 'D': int(np.sum(a_ok & ~b_ok)),
            'W': int(np.sum(~a_ok & ~b_ok & (pred_a != pred_b))), 'n': int(len(labels))}


def mcnemar_exact(c, d):
    n = c + d
    if n == 0:
        return 1.0
    k = min(c, d)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return float(min(1.0, 2 * p))


def bootstrap_diff(pred_a, pred_b, labels, reps=2000, seed=0):
    rng = np.random.RandomState(seed)
    a = (pred_a == labels).astype(np.float64)
    b = (pred_b == labels).astype(np.float64)
    n = len(labels)
    diffs = np.empty(reps)
    for r in range(reps):
        idx = rng.randint(0, n, n)
        diffs[r] = (b[idx].mean() - a[idx].mean()) * 100
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {'diff': float((b.mean() - a.mean()) * 100), 'ci95': [float(lo), float(hi)]}


def oracle_gain(pred_ref, pred_view, labels):
    ref_ok, view_ok = pred_ref == labels, pred_view == labels
    n11 = np.sum(ref_ok & view_ok); n00 = np.sum(~ref_ok & ~view_ok)
    n10 = np.sum(ref_ok & ~view_ok); n01 = np.sum(~ref_ok & view_ok)
    denom = n11 * n00 + n01 * n10
    q = float((n11 * n00 - n01 * n10) / denom) if denom else float('nan')
    return {'oracle_gain': float(np.mean(ref_ok | view_ok) * 100 - np.mean(ref_ok) * 100),
            'p_view_right_given_ref_wrong': float(n01 / max(n01 + n00, 1) * 100),
            'double_fault': float(n00 / len(labels) * 100), 'q_statistic': q,
            'view_acc': float(np.mean(view_ok) * 100), 'ref_acc': float(np.mean(ref_ok) * 100)}


def seed_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {'mean': float(values.mean()), 'std': float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            'values': values.tolist()}
