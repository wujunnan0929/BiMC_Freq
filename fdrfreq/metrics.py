"""FSCIL metrics and paired statistics used in every table of the paper."""

import math

import numpy as np


def session_metrics(pred, labels, base_classes):
    pred, labels = np.asarray(pred), np.asarray(labels)
    correct = pred == labels
    is_base = np.isin(labels, base_classes)
    acc = 100.0 * correct.mean() if len(correct) else float('nan')
    base = 100.0 * correct[is_base].mean() if is_base.any() else float('nan')
    novel = 100.0 * correct[~is_base].mean() if (~is_base).any() else float('nan')
    hm = (2 * base * novel / (base + novel)) if (is_base.any() and (~is_base).any()
                                                 and base + novel > 0) else float('nan')
    return {'acc': acc, 'base': base, 'novel': novel, 'hm': hm}


def summarize_sessions(per_session):
    accs = [s['acc'] for s in per_session]
    last = per_session[-1]
    return {'avg': float(np.mean(accs)), 'last': float(accs[-1]), 'first': float(accs[0]),
            'pd': float(accs[0] - accs[-1]), 'last_base': last['base'],
            'last_novel': last['novel'], 'last_hm': last['hm'],
            'sessions': [float(a) for a in accs]}


def error_types(pred, labels, base_classes):
    pred, labels = np.asarray(pred), np.asarray(labels)
    wrong = pred != labels
    lb, pb = np.isin(labels, base_classes), np.isin(pred, base_classes)
    n_nov, n_base = max((~lb).sum(), 1), max(lb.sum(), 1)
    return {'novel_to_base': 100.0 * (wrong & ~lb & pb).sum() / n_nov,
            'novel_to_novel': 100.0 * (wrong & ~lb & ~pb).sum() / n_nov,
            'base_to_novel': 100.0 * (wrong & lb & ~pb).sum() / n_base,
            'base_to_base': 100.0 * (wrong & lb & pb).sum() / n_base}


def mcnemar_exact(a_correct, b_correct):
    """Two-sided exact McNemar test on paired correctness vectors."""
    a, b = np.asarray(a_correct, bool), np.asarray(b_correct, bool)
    n01 = int((a & ~b).sum())  # A right, B wrong
    n10 = int((~a & b).sum())  # A wrong, B right
    n = n01 + n10
    if n == 0:
        return {'a_only': 0, 'b_only': 0, 'p': 1.0}
    k = min(n01, n10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return {'a_only': n01, 'b_only': n10, 'p': float(min(1.0, 2 * tail))}


def fixed_broken(anchor_pred, method_pred, labels):
    """C/D/W counts: fixed (anchor wrong -> method right), broken, wrong-changed."""
    a, m, y = map(np.asarray, (anchor_pred, method_pred, labels))
    ar, mr = a == y, m == y
    return {'fixed': int((~ar & mr).sum()), 'broken': int((ar & ~mr).sum()),
            'wrong_changed': int((~ar & ~mr & (a != m)).sum()), 'n': int(len(y))}


def complementarity(a_correct, b_correct):
    a, b = np.asarray(a_correct, bool), np.asarray(b_correct, bool)
    n = len(a)
    n11, n00 = (a & b).sum(), (~a & ~b).sum()
    n10, n01 = (a & ~b).sum(), (~a & b).sum()
    q_den = n11 * n00 + n01 * n10
    return {'acc_a': 100.0 * a.mean(), 'acc_b': 100.0 * b.mean(),
            'oracle_gain': 100.0 * ((a | b).mean() - a.mean()),
            'p_b_given_not_a': 100.0 * (n01 / max((~a).sum(), 1)),
            'double_fault': 100.0 * n00 / n,
            'agreement': 100.0 * (n11 + n00) / n,
            'q_statistic': float((n11 * n00 - n01 * n10) / q_den) if q_den else float('nan')}


def paired_bootstrap(correct_a, correct_b, reps=2000, seed=0):
    """CI of mean_seed(acc_b - acc_a); images resampled within each seed.

    correct_a / correct_b: lists (one entry per seed) of aligned bool vectors.
    """
    r = np.random.RandomState(seed)
    diffs = []
    per_seed = [np.asarray(b, float) - np.asarray(a, float) for a, b in zip(correct_a, correct_b)]
    point = 100.0 * float(np.mean([d.mean() for d in per_seed]))
    for _ in range(reps):
        vals = []
        for d in per_seed:
            idx = r.randint(0, len(d), len(d))
            vals.append(d[idx].mean())
        diffs.append(np.mean(vals))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {'delta': point, 'ci_low': 100.0 * float(lo), 'ci_high': 100.0 * float(hi),
            'seeds_positive': int(sum(d.mean() > 0 for d in per_seed)), 'n_seeds': len(per_seed)}


def hierarchical_bootstrap(correct_a, correct_b, reps=4000, seed=0):
    """Two-level bootstrap of mean_seed(acc_b - acc_a): resample the support seeds, then the
    test images within each drawn seed, so the interval includes the variance of the support
    draw. Per-image differences take values in {-1, 0, 1}, so each image resample is one
    multinomial draw over those three outcomes. Also returns the seed-level mean and standard
    deviation of the paired differences and a two-sided bootstrap p-value for 'no difference'."""
    r = np.random.RandomState(seed)
    per_seed = [np.asarray(b, float) - np.asarray(a, float) for a, b in zip(correct_a, correct_b)]
    S = len(per_seed)
    counts = [np.array([(d < 0).sum(), (d == 0).sum(), (d > 0).sum()]) for d in per_seed]
    n = [int(c.sum()) for c in counts]
    probs = [c / max(c.sum(), 1) for c in counts]
    means = np.array([d.mean() for d in per_seed])
    boots = np.empty(reps)
    for i in range(reps):
        pick = r.randint(0, S, S)
        vals = []
        for s in pick:
            m = r.multinomial(n[s], probs[s])
            vals.append((m[2] - m[0]) / max(n[s], 1))
        boots[i] = np.mean(vals)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2.0 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {'delta': 100.0 * float(means.mean()), 'ci_low': 100.0 * float(lo), 'ci_high': 100.0 * float(hi),
            'seed_mean': 100.0 * float(means.mean()),
            'seed_sd': 100.0 * float(means.std(ddof=1)) if S > 1 else float('nan'),
            'p_boot': float(min(1.0, max(p, 1.0 / reps))), 'n_seeds': S}


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(pvals, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj.tolist()


def topk_recall(scores, labels_idx, k):
    top = np.argsort(-scores, axis=1)[:, :k]
    return 100.0 * float((top == np.asarray(labels_idx)[:, None]).any(1).mean())
