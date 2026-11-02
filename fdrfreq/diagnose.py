"""Base-only diagnostics that decide whether a frequency method is worth building.

All quantities use pseudo-incremental episodes (base training images only).

G1' mask diagnostics
    class specificity: mean pairwise Jaccard of per-class frequency masks
        (random q-subsets give q/(2d-q); FDR keyword library for reference)
    shot stability: Jaccard(mask from K shots, mask from all images)
D0  top-k recall of the BiMC-dagger reference at the last pseudo session
D1/D2 representation complementarity: nearest-class-mean accuracy of every
    candidate representation and its oracle gain / conditional accuracy /
    double fault / Q statistic against the original CLS representation
D3  information upper bound: GDA on [CLS; extra block] vs GDA on CLS alone
    (both stand-alone, no fusion), per extra block, with noise and
    augmentation controls
"""

import numpy as np

from . import keywords as kw
from .dctband import BlockBuilder
from .freqmask import (class_delta_means, class_masks, class_specificity, random_jaccard,
                       shot_stability, ladder_views)
from .linalg import l2n
from .metrics import complementarity, topk_recall, error_types
from .pipeline import EpisodeRunner
from .select import dev_episodes


def _ncm(train_feat, train_lab, classes, query_feat):
    M = l2n(np.stack([train_feat[train_lab == c].mean(0) for c in classes]))
    return np.asarray(classes)[np.argmax(l2n(query_feat) @ M.T, axis=1)]


def representations(bank, sigmas):
    reps = [('view', v) for v in bank.views('train') if v != 'orig']
    for s in sigmas:
        if f'blur{float(s):g}' in bank.views('train'):
            reps.append(('blurdiff', s))
    for L in bank.dct_layers():
        for kind in ('radial', 'random'):
            if bank.has(f'train_dct_L{L}_{kind}'):
                for band in range(3):
                    reps.append(('dct', (L, kind, band)))
    return reps


def _rep_transform(bank, rep, fit_rows, fit_lab, base_cls):
    """Return a function (split, rows) -> features for one representation."""
    typ, arg = rep
    if typ == 'view':
        return lambda split, rows: bank.feat(split, arg)[rows]
    if typ == 'blurdiff':
        return lambda split, rows: (bank.feat(split, 'orig')[rows]
                                    - bank.feat(split, f'blur{float(arg):g}')[rows])
    L, kind, band = arg
    b = BlockBuilder({'type': 'dct', 'layer': L, 'kind': kind, 'band': band, 'k': 64,
                      'residualize': False}).fit(bank, fit_rows, fit_lab, base_cls)
    return lambda split, rows: b.transform(bank, split, rows)


def run_diagnostics(bank, split_seeds=(2, 3, 4), n_sequences=5, shot=None, q=50,
                    sigmas=(1, 2, 4), split_mode='random'):
    eps = dev_episodes(bank, split_seeds, n_sequences, shot, split_mode)
    ytr = bank.labels('train')
    rep = {'dataset': bank.meta.get('dataset'), 'n_episodes': len(eps), 'q': q,
           'sigmas': list(sigmas), 'dim': bank.dim}
    # ---------------------------------------------------- G1' mask diagnostics
    have_ladder = all(v in bank.views('train') for v in ladder_views(sigmas))
    if have_ladder:
        spec, stab, kwspec = [], [], []
        for s in split_seeds:
            e = next(x for x in eps if x.meta['split_seed'] == s)
            A = e.classes[0]
            fit = e.support[0]
            dm = class_delta_means(bank, 'train', fit, ytr[fit], A, sigmas)
            common = dm.mean(0)
            masks = class_masks(dm, common, q)
            spec.append(class_specificity(masks))
            rows_by_class = [fit[ytr[fit] == c] for c in A]
            stab.append(shot_stability(bank, rows_by_class, sigmas, q, e.meta['shot'], common,
                                       reps=3, seed=s))
            if bank.has_keywords():
                lib = kw.build_library(bank, A, q=q)
                if len(lib.masks) > 1:
                    M = np.stack(list(lib.masks.values()))[:, None, :]
                    kwspec.append(class_specificity(M)[0])
        rep['freq_mask_class_jaccard_per_band'] = np.mean(spec, 0).tolist()
        rep['freq_mask_shot_stability_per_band'] = np.mean(stab, 0).tolist()
        rep['random_mask_jaccard'] = random_jaccard(q, bank.dim)
        rep['keyword_library_jaccard'] = float(np.mean(kwspec)) if kwspec else None
    # ------------------------------------------------------- D0 / D1 / D2
    reps = representations(bank, sigmas)
    comp = {f'{t}:{a}': [] for t, a in reps}
    d0 = {'top1': [], 'top2': [], 'top5': []}
    errs = []
    for e in eps:
        r = EpisodeRunner(bank, e)
        out = r.run({'ensemble': {}}, keep_scores=True)
        lab = out['last_labels']
        idx = np.array([r.col[int(c)] for c in lab])
        for k in (1, 2, 5):
            d0[f'top{k}'].append(topk_recall(out['last_scores'], idx, k))
        errs.append(error_types(out['last_pred'], lab, e.classes[0]))
        sup = np.concatenate(e.support)
        seen = e.seen(len(e.classes) - 1)
        qrows = e.queries[-1]
        orig_pred = _ncm(bank.feat('train', 'orig')[sup], ytr[sup], seen, bank.feat('train', 'orig')[qrows])
        orig_ok = orig_pred == ytr[qrows]
        for rp in reps:
            f = _rep_transform(bank, rp, e.support[0], ytr[e.support[0]], e.classes[0])
            Xs, Xq = f('train', sup), f('train', qrows)
            ok = _ncm(Xs, ytr[sup], seen, Xq) == ytr[qrows]
            comp[f'{rp[0]}:{rp[1]}'].append(complementarity(orig_ok, ok))
    rep['D0_bimc_ens_topk_recall'] = {k: float(np.mean(v)) for k, v in d0.items()}
    rep['D0_error_types'] = {k: float(np.mean([x[k] for x in errs])) for k in errs[0]}
    rep['D2_complementarity'] = {k: {m: float(np.nanmean([x[m] for x in v])) for m in v[0]}
                                 for k, v in comp.items() if v}
    # ------------------------------------------------------------------ D3
    blocks = {'cls_only': []}
    for L in bank.dct_layers():
        blocks[f'dct_L{L}_radial'] = [{'type': 'dct', 'layer': L, 'kind': 'radial', 'bands': [0, 1, 2], 'k': 64}]
        if bank.has(f'train_dct_L{L}_random'):
            blocks[f'dct_L{L}_random'] = [{'type': 'dct', 'layer': L, 'kind': 'random', 'bands': [0, 1, 2], 'k': 64}]
    for s in sigmas:
        if f'blur{float(s):g}' in bank.views('train'):
            blocks[f'blurdiff{s:g}'] = [{'type': 'blurdiff', 'sigma': s, 'k': 64}]
    for v in ('flip', 'gray'):
        if v in bank.views('train'):
            blocks[f'aug_{v}'] = [{'type': 'view', 'view': v, 'k': 64}]
    blocks['noise'] = [{'type': 'noise', 'k': 192}]
    d3 = {k: [] for k in blocks}
    for e in eps:
        r = EpisodeRunner(bank, e)
        for k, bl in blocks.items():
            cfg = {'anchor': {'blocks': bl, 'omega': 1.0, 'fusion': 'gda_only', 'temperature': 1.0}}
            d3[k].append(r.run(cfg)['summary']['last'])
    base = np.asarray(d3['cls_only'])
    split_ids = np.array([e.meta['split_seed'] for e in eps])
    rep['D3_gda_last'] = {}
    for k, v in d3.items():
        v = np.asarray(v)
        per_split = [float((v - base)[split_ids == s].mean()) for s in np.unique(split_ids)]
        rep['D3_gda_last'][k] = {'last': float(v.mean()), 'delta_vs_cls': float((v - base).mean()),
                                 'delta_per_split': per_split,
                                 'splits_positive': int(sum(x > 0 for x in per_split))}
    return rep


def print_summary(rep):
    print(f"Diagnostics on {rep['dataset']} ({rep['n_episodes']} pseudo episodes)")
    if 'freq_mask_class_jaccard_per_band' in rep:
        print('  freq mask class Jaccard per band:', [round(x, 3) for x in rep['freq_mask_class_jaccard_per_band']],
              '| random', round(rep['random_mask_jaccard'], 3), '| keyword lib', rep['keyword_library_jaccard'])
        print('  freq mask shot stability per band:', [round(x, 3) for x in rep['freq_mask_shot_stability_per_band']])
    print('  D0 BiMC-dagger top-k recall:', {k: round(v, 2) for k, v in rep['D0_bimc_ens_topk_recall'].items()})
    print('  D0 error types (%):', {k: round(v, 2) for k, v in rep['D0_error_types'].items()})
    print('  D2 representation vs CLS (NCM):')
    for k, v in rep['D2_complementarity'].items():
        print(f"    {k:28s} acc {v['acc_b']:.2f} (cls {v['acc_a']:.2f}) oracle +{v['oracle_gain']:.2f} "
              f"P(right|cls wrong) {v['p_b_given_not_a']:.2f} Q {v['q_statistic']:.3f}")
    print('  D3 stand-alone GDA last-session acc (delta vs CLS):')
    for k, v in rep['D3_gda_last'].items():
        print(f"    {k:22s} {v['last']:.2f} ({v['delta_vs_cls']:+.2f}; splits positive {v['splits_positive']})")


def error_concentration(bank, split_seeds=(2, 3, 4), n_sequences=5, shot=None, q=50,
                        sigmas=(1, 2, 4), split_mode='contiguous', m=8):
    """Where does a K-shot prototype's error live?

    For every pseudo-novel class, err_j = (p_K[j] - p_all[j])^2, with p_K the unit
    K-shot prototype and p_all the unit mean of all its training images. For a
    channel mask g: capture = sum_g err / sum err, coverage = |g| / d,
    concentration = capture / coverage (1 = no better than random channels).
    A mask that concentrates the error is a mask worth recomposing.
    """
    from .pipeline import EpisodeRunner
    eps = dev_episodes(bank, split_seeds, n_sequences, shot, split_mode)
    F, y = bank.feat('train', 'orig'), bank.labels('train')
    acc = {}
    for e in eps:
        r = EpisodeRunner(bank, e)
        deltas, common = r.freq_deltas(sigmas)
        lib = kw.build_library(bank, r.base_cls, q=q) if bank.has_keywords() else None
        for j in range(r.nb, len(r.all_cls)):
            c = int(r.all_cls[j])
            oracle = l2n(F[y == c].mean(0))
            err = (r.raw_proto[j] - oracle) ** 2
            masks = {}
            masks['freq_own'] = np.logical_or.reduce([kw.topq_mask(deltas[j, b] - common[b], q) for b in range(len(sigmas))])
            donors = [np.random.RandomState(c * 31 + b).randint(r.nb) for b in range(len(sigmas))]
            masks['freq_classperm'] = np.logical_or.reduce([kw.topq_mask(deltas[d, b] - common[b], q) for b, d in enumerate(donors)])
            masks['freq_base_common'] = kw.topq_mask(np.abs(deltas[:r.nb]).mean((0, 1)), int(masks['freq_own'].sum()))
            rr = np.random.RandomState(c * 131 + 7)
            rnd = np.zeros(bank.dim, bool)
            rnd[rr.choice(bank.dim, int(masks['freq_own'].sum()), replace=False)] = True
            masks['random_same_size'] = rnd
            if lib is not None:
                words = kw.class_keywords(bank, c, lib, m=m)
                if words:
                    masks['keyword'] = np.logical_or.reduce([lib.masks[w] for w in words])
            for k, g in masks.items():
                cov = g.mean()
                cap = err[g].sum() / max(err.sum(), 1e-18)
                acc.setdefault(k, []).append((cap, cov, cap / max(cov, 1e-12)))
    out = {}
    for k, v in acc.items():
        v = np.asarray(v)
        out[k] = {'capture': float(v[:, 0].mean()), 'coverage': float(v[:, 1].mean()),
                  'concentration': float(v[:, 2].mean()), 'n': int(len(v))}
    return out
