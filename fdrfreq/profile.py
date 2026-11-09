"""Frequency-free analyses for the Neurocomputing revision (PAPER_PLAN sec 13, E8 / E9 / E11).

error_profile      Where does a K-shot prototype's error live? Capture, coverage and
                   concentration of class-agnostic channel sets at fixed budgets
                   (no blur views needed), a capture-vs-budget curve up to every
                   channel, the rank agreement between the base pooled variance and
                   the pseudo-novel classes' own variance, and between the Fisher and
                   the variance rankings. Development episodes only.
variance_transfer  The same agreement on the real benchmark split: pooled within-class
                   variance of the base classes against that of the new classes, both
                   from training images (analysis only; never used for selection).
budget_prediction  Corollary 1 in numbers: k*(K) = |{j: var_j / (K g^2) > t^2}| / d from
                   base statistics, with t^2 the donor error (constant or per channel).
dcr_timing         Wall-clock cost of the recomposition step per new class.
"""

import time

import numpy as np

from . import keywords as kw
from .linalg import l2n, softmax, within_class_centered
from .recompose import n_donors
from .select import dev_episodes

CURVE = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def spearman(a, b):
    """Spearman rank correlation (average ranks for ties)."""
    def rank(x):
        x = np.asarray(x, dtype=np.float64)
        order = np.argsort(x, kind='mergesort')
        r = np.empty(len(x))
        r[order] = np.arange(len(x), dtype=np.float64)
        # average ties
        _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, weights=r)
        return (sums / cnt)[inv]
    ra, rb = rank(a), rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float('nan')


def _channel_scores(r, X, y, lambda_t=0.5):
    """Base-only channel scores of one episode (larger = earlier in the ranking)."""
    Xc, _ = within_class_centered(X, y, r.base_cls)
    wvar = (Xc ** 2).mean(0)
    scores = {'var': wvar, 'lowvar': -wvar, 'mag': np.abs(X).mean(0),
              'fisher': r.raw_mean[:r.nb].var(0) / np.maximum(wvar, 1e-12)}
    r._cur_lam_t = lambda_t
    scores['ape'] = r._ape_criterion(r.text_cal(lambda_t))
    return scores


def error_profile(bank, split_seeds=(2, 3, 4), n_sequences=5, shot=None, split_mode='contiguous',
                  budgets=(0.1, 0.25), curve=CURVE, exclude_support=True, keyword_m=8, q=50):
    from .pipeline import EpisodeRunner
    eps = dev_episodes(bank, split_seeds, n_sequences, shot, split_mode)
    F, y = bank.feat('train', 'orig').astype(np.float64), bank.labels('train')
    D = bank.dim
    sets, curv, n_ce = {}, {}, 0
    rho_nov, rho_cls, rho_fisher, jac_top10 = [], [], [], []
    for e in eps:
        r = EpisodeRunner(bank, e)
        X0, y0 = r.sup_feat[0], r.sup_lab[0]
        sc = _channel_scores(r, X0, y0)
        support_rows = np.unique(np.concatenate([np.asarray(s) for s in e.support[1:]])) if len(e.support) > 1 \
            else np.array([], dtype=int)
        lib = kw.build_library(bank, r.base_cls, q=q) if bank.has_keywords() else None
        rho_fisher.append(spearman(sc['fisher'], sc['var']))
        nov_var = []
        for j in range(r.nb, len(r.all_cls)):
            c = int(r.all_cls[j])
            rows = np.flatnonzero(y == c)
            if exclude_support:
                rows = np.setdiff1d(rows, support_rows)
            full = l2n(F[rows].mean(0))
            err = (r.raw_proto[j] - full) ** 2
            tot = max(err.sum(), 1e-18)
            cv = F[rows].var(0)
            nov_var.append(cv)
            rho_cls.append(spearman(cv, sc['var']))
            n_ce += 1
            for b in budgets:
                kk = max(1, int(round(b * D)))
                masks = {n: kw.topk_signed(sc[n], kk) for n in sc}
                rnd = np.zeros(D, bool)
                rnd[np.random.RandomState(c * 131 + 7 + kk).choice(D, kk, replace=False)] = True
                masks['random'] = rnd
                masks['oracle'] = kw.topk_signed(err, kk)
                for n, g in masks.items():
                    cap = err[g].sum() / tot
                    sets.setdefault(f'{n}@{b:g}', []).append((cap, g.mean(), cap / max(g.mean(), 1e-12)))
            if lib is not None:
                words = kw.class_keywords(bank, c, lib, m=keyword_m)
                if words:
                    g = np.logical_or.reduce([lib.masks[w] for w in words])
                    cap = err[g].sum() / tot
                    sets.setdefault('keyword', []).append((cap, g.mean(), cap / max(g.mean(), 1e-12)))
            for frac in curve:
                kk = max(1, int(round(frac * D)))
                for n in sc:
                    curv.setdefault(n, {}).setdefault(frac, []).append(err[kw.topk_signed(sc[n], kk)].sum() / tot)
                curv.setdefault('random', {}).setdefault(frac, []).append(kk / D)
                curv.setdefault('oracle', {}).setdefault(frac, []).append(np.sort(err)[::-1][:kk].sum() / tot)
        nv = np.mean(nov_var, axis=0)
        rho_nov.append(spearman(nv, sc['var']))
        k10 = max(1, int(round(0.1 * D)))
        a_, b_ = kw.topk_signed(nv, k10), kw.topk_signed(sc['var'], k10)
        jac_top10.append(np.logical_and(a_, b_).sum() / np.logical_or(a_, b_).sum())
    out = {'n_class_episodes': int(n_ce), 'n_episodes': len(eps), 'exclude_support': bool(exclude_support),
           'sets': {}, 'curve': {n: {f'{f:g}': float(np.mean(v)) for f, v in d.items()} for n, d in curv.items()},
           'agreement': {'spearman_var_base_vs_novel_pooled': float(np.mean(rho_nov)),
                         'spearman_var_base_vs_each_novel_class': float(np.mean(rho_cls)),
                         'jaccard_top10_base_vs_novel_pooled': float(np.mean(jac_top10)),
                         'spearman_fisher_vs_var': float(np.mean(rho_fisher))}}
    for n, v in sets.items():
        v = np.asarray(v)
        out['sets'][n] = {'capture': float(v[:, 0].mean()), 'coverage': float(v[:, 1].mean()),
                          'concentration': float(v[:, 2].mean()), 'n': int(len(v))}
    return out


def _pooled_var(X, y, classes):
    Xc, _ = within_class_centered(X, y, classes)
    return (Xc ** 2).sum(0) / max(len(Xc) - len(classes), 1)


def variance_transfer(bank):
    """Base pooled within-class variance vs the new classes' (benchmark split, training images)."""
    X, y = bank.feat('train', 'orig').astype(np.float64), bank.labels('train')
    base = np.arange(bank.num_init)
    new = np.arange(bank.num_init, bank.num_classes)
    vb = _pooled_var(X[np.isin(y, base)], y[np.isin(y, base)], base)
    vn = _pooled_var(X[np.isin(y, new)], y[np.isin(y, new)], new)
    per_class = [spearman(X[y == c].var(0), vb) for c in new]
    k10 = max(1, int(round(0.1 * bank.dim)))
    a_, b_ = kw.topk_signed(vb, k10), kw.topk_signed(vn, k10)
    return {'spearman_base_vs_new_pooled': spearman(vb, vn),
            'spearman_base_vs_each_new_class_mean': float(np.mean(per_class)),
            'spearman_base_vs_each_new_class_min': float(np.min(per_class)),
            'jaccard_top10': float(np.logical_and(a_, b_).sum() / np.logical_or(a_, b_).sum()),
            'n_new_classes': int(len(new))}


def budget_prediction(bank, shots=(1, 2, 3, 5, 10), r=1.0, tau=22.0):
    """Corollary 1 with base statistics of the benchmark's base session (visual donors).

    var_j: pooled within-class variance of the unit image features; g2: mean squared
    norm of the base class means; t2_j: leave-one-out donor error of the base classes,
    corrected for the base prototypes' own sampling noise (as in EpisodeRunner._eb_stats).
    k*_const(K) uses one t2 (the channel mean, the corollary's assumption);
    k*_chan(K) uses t2_j per channel.
    """
    X, y = bank.feat('train', 'orig').astype(np.float64), bank.labels('train')
    base = np.arange(bank.num_init)
    keep = np.isin(y, base)
    X, y = X[keep], y[keep]
    var = _pooled_var(X, y, base)
    means = np.stack([X[y == c].mean(0) for c in base])
    counts = np.array([(y == c).sum() for c in base], dtype=np.float64)
    g2 = float((np.linalg.norm(means, axis=1) ** 2).mean())
    P = l2n(means)
    S = P @ P.T
    np.fill_diagonal(S, -np.inf)
    nb = len(base)
    rr = min(n_donors(r, nb), nb - 1)
    Dn = np.zeros_like(P)
    for b in range(nb):
        top = np.argpartition(-S[b], rr - 1)[:rr]
        Dn[b] = l2n(softmax(float(tau) * S[b, top]) @ P[top])
    E2 = ((P - Dn) ** 2).mean(0)
    t2 = np.maximum(E2 - var / (counts.mean() * g2), 1e-3 * E2.mean())
    t2c = float(t2.mean())
    pred = {}
    for K in shots:
        s2 = var / (float(K) * g2)
        pred[str(K)] = {'k_const': float((s2 > t2c).mean()), 'k_channel': float((s2 > t2).mean())}
    return {'g2': g2, 't2_mean': t2c, 'var_median': float(np.median(var)), 'r': r, 'tau': tau,
            'prediction': pred}


def dcr_timing(bank, config, shot=None, repeats=5):
    """Seconds per new class for the recomposition step: prototype construction with and
    without it on one development episode (the per-class cost does not depend on the split)."""
    from .pipeline import EpisodeRunner, resolve_config
    import copy
    ep = dev_episodes(bank, (2,), 1, shot, 'contiguous')[0]
    off = copy.deepcopy(config)
    off['segments'] = None
    times = {}
    for name, cfg in (('with', config), ('without', off)):
        best = float('inf')
        c = resolve_config(cfg, bank)
        for _ in range(repeats):
            r = EpisodeRunner(bank, ep)
            t0 = time.perf_counter()
            r.vis_cal(c)
            best = min(best, time.perf_counter() - t0)
        times[name] = best
    n_new = int(sum(len(c) for c in ep.classes[1:]))
    return {'seconds_with': times['with'], 'seconds_without': times['without'], 'n_new_classes': n_new,
            'ms_per_new_class': 1000.0 * max(times['with'] - times['without'], 0.0) / max(n_new, 1)}
