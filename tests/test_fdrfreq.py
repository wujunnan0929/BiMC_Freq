"""Tests for the cache-based training-free calibration package (numpy only)."""

import json
import math

import numpy as np
import pytest

from fdrfreq import keywords as kw
from fdrfreq.dctband import dct_matrix, radial_bands, random_bands, band_energies
from fdrfreq.freqmask import class_specificity, random_jaccard, class_delta_means, class_masks
from fdrfreq.gda import GDA
from fdrfreq.linalg import ledoit_wolf_shrinkage, softmax, l2n
from fdrfreq.metrics import (mcnemar_exact, paired_bootstrap, fixed_broken, session_metrics,
                             complementarity)
from fdrfreq.pipeline import EpisodeRunner
from fdrfreq.protocol import (benchmark_episode, pseudo_episodes, check_episode,
                              replica_support_indices, task_classes)
from fdrfreq import select as sel
from fdrfreq.synthetic import make_bank


@pytest.fixture(scope='module')
def bank():
    return make_bank()


@pytest.fixture(scope='module')
def bench(bank):
    return benchmark_episode(bank, 1)


# ----------------------------------------------------------------- protocol
def test_replica_sampler_is_deterministic_and_complete():
    labels = np.repeat(np.arange(10), 8)
    tasks = task_classes(10, 4, 2)
    a = replica_support_indices(labels, tasks, -1, 5, seed=3)
    b = replica_support_indices(labels, tasks, -1, 5, seed=3)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert len(a[0]) == 4 * 8 and all(len(x) == 2 * 5 for x in a[1:])
    full = replica_support_indices(labels, tasks, 8, 5, seed=3)  # shot == #candidates permutes
    assert sorted(full[0].tolist()) == sorted(a[0].tolist())
    assert not np.array_equal(full[1], a[1])  # base permutation consumed the rng first


def test_protocols_are_leak_free(bank, bench):
    assert check_episode(bench, bank)
    assert bench.meta['support_source'] == 'stored_engine_manifest'
    eps = pseudo_episodes(bank, 2, n_sequences=3)
    for e in eps:
        assert check_episode(e, bank)
        assert e.query_split == 'train'
        A, V = set(e.meta['A']), set(e.meta['V'])
        assert not A & V and A | V == set(range(bank.num_init))
        assert set(e.classes[0]) == A
        assert set(np.concatenate(e.classes[1:])) == V
    # the fit/query split of A is fixed per split seed
    assert np.array_equal(eps[0].support[0], eps[1].support[0])


# --------------------------------------------------------- exact fallbacks
def _pred(runner, cfg):
    return runner.run(cfg)['last_pred']


def test_zero_strength_is_exactly_off(bank, bench):
    r = EpisodeRunner(bank, bench)
    ens = {'ensemble': {}}
    ref = _pred(r, ens)
    assert np.array_equal(ref, _pred(r, {**ens, 'anchor': {'omega': 0.0}}))
    assert np.array_equal(ref, _pred(r, {**ens, 'novel_bias': 0.0}))
    seg0 = {**ens, 'segments': {'type': 'keyword', 'alpha': 0.0, 'q': 10, 'r': 5}}
    assert np.array_equal(ref, _pred(r, seg0))
    seg0f = {**ens, 'segments': {'type': 'freq', 'alpha': 0.0, 'q': 8, 'r': 5}}
    assert np.array_equal(ref, _pred(r, seg0f))
    raw = _pred(r, {**ens, 'holistic': None})
    assert np.array_equal(raw, _pred(r, {**ens, 'holistic': {'donor': 'text', 'lam': 0.0, 'tau': 16}}))
    blk = [{'type': 'dct', 'layer': 6, 'bands': [0, 1, 2], 'k': 8}]
    a = {**ens, 'anchor': {'omega': 0.3, 'blocks': blk}}
    eb0 = {**ens, 'anchor': {'omega': 0.3, 'blocks': blk, 'eb': {'lam': 0.0}}}
    assert np.array_equal(_pred(r, a), _pred(r, eb0))


# -------------------------------------------------- BiMC reference parity
def _naive_bimc(bank, ep, beta, lam_t, lam_i, tau, alpha=None, gb=1.0, gi=5.0):
    F = bank.feat('train', 'orig').astype(np.float64)
    ytr = bank.labels('train')
    Fq = bank.feat('test', 'orig').astype(np.float64)
    yq = bank.labels('test')
    text = bank.text_prompt.astype(np.float64)
    dproto = bank.desc_proto.astype(np.float64)
    dfeat, dtgt = bank.desc_feat.astype(np.float64), bank.desc_target
    protos, covs, weights, cls_all = [], [], [], []
    base = None
    accs = []
    for t, (cls, sup) in enumerate(zip(ep.classes, ep.support)):
        X, y = F[sup], ytr[sup]
        P = l2n(np.stack([X[y == c].mean(0) for c in cls]))
        if t == 0:
            base = P
        else:
            w = softmax(tau * P @ base.T, axis=1)
            d = l2n(w @ base)
            P = l2n((1 - lam_i) * P + lam_i * d)
        protos.append(P)
        cov = np.cov(X, rowvar=False)
        cov = cov + (gb if t == 0 else gi) * np.mean(np.diag(cov)) * np.eye(cov.shape[0])
        covs.append(cov)
        weights.append(len(cls))
        cls_all.extend(cls.tolist())
        V = np.concatenate(protos)
        C = np.array(cls_all)
        q = np.isin(yq, C)
        x = Fq[q]
        fused = l2n(beta * ((1 - lam_t) * text[C] + lam_t * dproto[C]) + (1 - beta) * V)
        p = softmax(x @ fused.T, axis=1)
        if alpha is not None:
            Pinv = np.linalg.pinv(sum(wi * ci for wi, ci in zip(weights, covs)) / sum(weights))
            maha = np.stack([((x - v) @ Pinv * (x - v)).sum(1) for v in V], 1)
            pc = softmax(-maha / 512, axis=1)
            knn = np.stack([(x @ dfeat[dtgt == c].T).max(1) for c in C], 1)
            pk = softmax(knn, axis=1)
            nb = len(ep.classes[0])
            p = np.concatenate([alpha * p[:, :nb] + (1 - alpha) * pc[:, :nb],
                                alpha * p[:, nb:] + (1 - alpha) * pk[:, nb:]], 1)
        accs.append(100 * np.mean(C[p.argmax(1)] == yq[q]))
    return accs


def test_bimc_matches_naive_reimplementation(bank, bench):
    r = EpisodeRunner(bank, bench)
    got = r.run({})['summary']['sessions']
    want = _naive_bimc(bank, bench, beta=0.5, lam_t=0.5, lam_i=0.1, tau=16)
    assert np.allclose(got, want)
    got = r.run({'ensemble': {}})['summary']['sessions']
    want = _naive_bimc(bank, bench, 0.5, 0.5, 0.1, 16, alpha=0.7)
    assert np.allclose(got, want)


# ---------------------------------------------------------------------- GDA
def test_kappa_term_is_the_expected_plugin_correction(bank):
    r = np.random.RandomState(0)
    X = r.randn(300, 6)
    y = np.repeat(np.arange(3), 100)
    X[y == 1] += 2.0
    g = GDA(shrink=0.0).fit_base(X, y, [0, 1, 2])
    g.add_classes(np.stack([X[y == c].mean(0) for c in range(3)]), [100, 5, 1])
    s0, s1 = g.scores(X[:4], kappa=0.0), g.scores(X[:4], kappa=1.0)
    diff = (s1 - s0)[0] * g.dim
    assert np.allclose(diff, g.trPW / (2 * np.array([100, 5, 1])))
    assert diff[2] > diff[1] > diff[0]


def test_ledoit_wolf_behaves():
    r = np.random.RandomState(0)
    iso = r.randn(200, 20)
    assert ledoit_wolf_shrinkage(iso - iso.mean(0)) > 0.5
    aniso = r.randn(5000, 20) * np.linspace(0.1, 5, 20)
    assert ledoit_wolf_shrinkage(aniso - aniso.mean(0)) < 0.1


# ---------------------------------------------------------------------- DCT
def test_dct_bands():
    n = 14
    D = dct_matrix(n)
    assert np.allclose(D @ D.T, np.eye(n), atol=1e-10)
    bands = radial_bands(n)
    union = np.logical_or.reduce(bands)
    assert not union[0, 0] and union.sum() == n * n - 1
    assert sum(b.sum() for b in bands) == n * n - 1
    rb = random_bands(n, seed=1)
    assert [b.sum() for b in rb] == [b.sum() for b in bands]
    assert np.logical_or.reduce(rb).sum() == n * n - 1 and not np.logical_or.reduce(rb)[0, 0]
    const = np.ones((2, n * n, 3))
    e = band_energies(const, D, bands)
    assert np.allclose(e, 0, atol=1e-10)
    r = np.random.RandomState(0)
    tok = r.randn(1, n * n, 4)
    e = band_energies(tok, D, bands)
    C = np.einsum('uh,bhwc,vw->buvc', D, tok.reshape(1, n, n, 4), D)
    total = (C ** 2).sum((1, 2)) - C[:, 0, 0] ** 2
    recon = sum((e[:, i] ** 2) * bands[i].sum() for i in range(3))
    assert np.allclose(recon, total)


# ----------------------------------------------------------------- keywords
def test_candidate_words_exclude_class_name_and_stopwords():
    words = dict(kw.candidate_words(['The Red-winged Blackbird has a red shoulder patch and a black body.',
                                     'Its shoulder patch is bright red.'], 'Red winged Blackbird'))
    assert 'shoulder' in words and words['shoulder'] == 2
    assert 'blackbird' not in words and 'the' not in words and 'red' not in words


def test_library_union_df_and_random_twin(bank):
    base = np.arange(bank.num_init)
    lib = kw.build_library(bank, base, q=10, min_df=2, max_df=0.5)
    pc, pw = bank.array('kw_pair_class'), bank.array('kw_pair_word')
    for w, m in lib.masks.items():
        users = np.unique(pc[(pw == w) & np.isin(pc, base)])
        assert 2 <= len(users) <= 0.5 * len(base)
        assert m.sum() >= 10
    twin = lib.random_twin(seed=3)
    assert {w: m.sum() for w, m in twin.masks.items()} == {w: m.sum() for w, m in lib.masks.items()}
    assert all(np.array_equal(twin.masks[w], lib.random_twin(seed=3).masks[w]) for w in twin.masks)
    words = kw.class_keywords(bank, bank.num_init + 1, lib, m=4)
    assert len(words) <= 4 and all(w in lib.masks for w in words)


# ---------------------------------------------------------------- freqmask
def test_mask_statistics(bank):
    m = np.zeros((3, 1, 20), dtype=bool)
    m[:, 0, :5] = True
    assert class_specificity(m) == [1.0]
    m[1, 0] = False
    m[1, 0, 10:15] = True
    m[2, 0] = False
    m[2, 0, 15:20] = True
    assert class_specificity(m) == [0.0]
    r = np.random.RandomState(0)
    sims = []
    for _ in range(2000):
        a, b = np.zeros(100, bool), np.zeros(100, bool)
        a[r.choice(100, 20, replace=False)] = True
        b[r.choice(100, 20, replace=False)] = True
        sims.append((a & b).sum() / (a | b).sum())
    assert abs(np.mean(sims) - random_jaccard(20, 100)) < 0.01


def test_frequency_masks_find_detail_channels():
    b = make_bank(seed=1)
    y = b.labels('train')
    rows = np.flatnonzero(y < b.num_init)
    classes = np.arange(b.num_init)
    dm = class_delta_means(b, 'train', rows, y[rows], classes, [1, 2, 4])
    masks = class_masks(dm, dm.mean(0), q=6)
    assert np.mean(class_specificity(masks)) < 0.5


# ------------------------------------------------------------------ metrics
def test_metrics():
    a = np.array([1, 1, 0, 0, 1, 0], bool)
    b = np.array([1, 0, 1, 1, 1, 0], bool)
    m = mcnemar_exact(a, b)
    assert (m['a_only'], m['b_only']) == (1, 2) and math.isclose(m['p'], 1.0)
    assert mcnemar_exact(a, a)['p'] == 1.0
    bs = paired_bootstrap([a], [b], reps=200)
    assert math.isclose(bs['delta'], 100 * (b.mean() - a.mean()))
    y = np.array([0, 1, 2, 3])
    fb = fixed_broken(np.array([0, 9, 2, 9]), np.array([0, 1, 9, 8]), y)
    assert fb == {'fixed': 1, 'broken': 1, 'wrong_changed': 1, 'n': 4}
    s = session_metrics(np.array([0, 1, 5, 5]), np.array([0, 1, 5, 6]), [0, 1])
    assert s['base'] == 100 and s['novel'] == 50 and math.isclose(s['hm'], 2 * 100 * 50 / 150)
    c = complementarity(a, b)
    assert math.isclose(c['oracle_gain'], 100 * ((a | b).mean() - a.mean()))


# ---------------------------------------------------------------- selection
def test_selection_respects_tolerance_and_ties(bank):
    eps = sel.dev_episodes(bank, (2,), 2)
    entry = {'name': 'LA', 'base': {'ensemble': {}},
             'grid': {'novel_bias': [0.0, 50.0]}, 'fallback': {'novel_bias': 0.0}, 'tolerance': 0.2}
    rep = sel.select(bank, entry, eps)
    assert rep['selected_params'] == {'novel_bias': 0.0}
    tie = {'name': 'tie', 'base': {'ensemble': {}}, 'grid': {'novel_bias': [0.0, 0.0]},
           'fallback': {'novel_bias': 0.0}}
    rep = sel.select(bank, tie, eps)
    assert rep['candidates'][0]['hash'] == rep['selected_hash']


def test_selection_never_reads_test_images():
    b = make_bank()
    for k in [k for k in b._arrays if k.startswith('test_')]:
        del b._arrays[k]
    eps = sel.dev_episodes(b, (2,), 2)
    entry = {'name': 'A1', 'base': {'ensemble': {}, 'anchor': {'blocks': [{'type': 'dct', 'layer': 6, 'bands': [0, 1], 'k': 4}]}},
             'grid': {'anchor.omega': [0.0, 0.2]}, 'fallback': {'anchor.omega': 0.0}}
    sel.select(b, entry, eps)
    with pytest.raises(KeyError):
        b.labels('test')


# --------------------------------------------------------------------- CLI
def test_cli_smoke_and_lock_integrity(tmp_path):
    from fdrfreq.run import main
    main(['smoke', '--out', str(tmp_path)])
    assert (tmp_path / 'eval_A2_shot5.json').exists()
    res = json.loads((tmp_path / 'eval_A2_shot5.json').read_text())
    assert set(res['seeds']) == {'1', '2', '3'}
    from fdrfreq.report import build_report
    rep = build_report(tmp_path, anchor='A2', pairs=[('A1', 'A2'), ('A2', 'not_run')])
    assert 'A2 - A1' in rep and 'skipped (run missing): A2:not_run' in rep
    lk = json.loads((tmp_path / 'locked_A2.json').read_text())
    lk['config']['anchor']['omega'] = 0.99
    bad = tmp_path / 'locked_tampered.json'
    bad.write_text(json.dumps(lk))
    with pytest.raises(SystemExit):
        main(['evaluate', '--bank', str(tmp_path / 'bank'), '--out', str(tmp_path), '--locked', str(bad)])


def test_predictive_mode_matches_closed_form():
    r = np.random.RandomState(1)
    X = r.randn(200, 5)
    y = np.repeat(np.arange(2), 100)
    g = GDA(shrink=0.0).fit_base(X, y, [0, 1])
    M = np.stack([X[y == c].mean(0) for c in range(2)])
    g.add_classes(M, [100, 3])
    q = r.randn(4, 5)
    got = g.scores(q, kappa=1.0, mode='predictive', normalize_by_dim=False)
    want = np.stack([-0.5 * (((q - M[c]) @ g.P) * (q - M[c])).sum(1) / (1 + 1 / n)
                     - 0.5 * 5 * np.log(1 + 1 / n) for c, n in enumerate([100, 3])], 1)
    assert np.allclose(got, want)
    lda = g.scores(q, kappa=0.0, mode='predictive', normalize_by_dim=False)
    assert np.array_equal(lda.argmax(1), g.scores(q, kappa=0.0, normalize_by_dim=False).argmax(1))


def test_freq_common_segments_run_and_fall_back(bank, bench):
    r = EpisodeRunner(bank, bench)
    ens = {'ensemble': {}}
    ref = r.run(ens)['last_pred']
    off = r.run({**ens, 'segments': {'type': 'freq_common', 'alpha': 0.0, 'q': 8, 'r': 5}})['last_pred']
    assert np.array_equal(ref, off)
    on = r.run({**ens, 'segments': {'type': 'freq_common', 'alpha': 0.5, 'q': 8, 'r': 5}})
    assert on['segment_stats']['segments_per_class'] == 3


def test_channel_prior_controls_have_matched_size(bank, bench):
    r = EpisodeRunner(bank, bench)
    base = {'ensemble': {}}
    sizes = {}
    for t in ('freq_union', 'var_union', 'mag_union', 'fisher_union', 'random_union'):
        res = r.run({**base, 'segments': {'type': t, 'alpha': 0.5, 'q': 6, 'r': 5}})
        sizes[t] = res['segment_stats']['channels_touched']
        assert res['segment_stats']['segments_per_class'] == 1
        off = r.run({**base, 'segments': {'type': t, 'alpha': 0.0, 'q': 6, 'r': 5}})['last_pred']
        assert np.array_equal(off, r.run(base)['last_pred'])
    assert len(set(sizes.values())) == 1


def test_ula_matches_algorithm_1():
    from fdrfreq.pipeline import ula_adjust
    r = np.random.RandomState(0)
    z = r.randn(200, 12) * 0.05
    nb, tau, lam, T, a, tth = 8, 0.04, 0.4, 0.25, 8.0, 0.06
    got = ula_adjust(z, nb, tau, lam, T, a, tth)
    want = z.copy()
    for n in range(len(z)):
        order = np.argsort(-z[n])
        c1, c2 = order[0], order[1]
        m = z[n, c1] - z[n, c2]
        if c1 >= nb or c2 < nb or m >= tau:
            continue
        h = min(max((tau - m) / tau, 0), 1)
        p = np.exp(z[n, :nb] / T - (z[n, :nb] / T).max())
        p /= p.sum()
        H = -(p * np.log(p)).sum() / np.log(nb)
        d = lam * h / (1 + np.exp(-a * (H - tth)))
        want[n, c1] -= d
        want[n, c2] += d
    assert np.allclose(got, want)
    assert np.array_equal(ula_adjust(z, nb, tau, 0.0, T, a, tth), z)


def test_eb_recomposition_is_shot_aware_and_falls_back(bank, bench):
    r = EpisodeRunner(bank, bench)
    raw = {'ensemble': {}, 'holistic': None}
    ref = r.run(raw)['last_pred']
    off = r.run({**raw, 'segments': {'type': 'eb', 'alpha': 0.0, 'r': 5}})['last_pred']
    assert np.array_equal(ref, off)
    on = r.run({**raw, 'segments': {'type': 'eb', 'alpha': 1.0, 'r': 5}})
    uni = r.run({**raw, 'segments': {'type': 'eb_uniform', 'alpha': 1.0, 'r': 5}})
    assert on['segment_stats']['channels_touched'] > 0
    assert np.isclose(on['segment_stats']['channels_touched'], uni['segment_stats']['channels_touched'])
    one = EpisodeRunner(bank, benchmark_episode(bank, 1, shot=1))
    s1 = one.run({**raw, 'segments': {'type': 'eb', 'alpha': 1.0, 'r': 5}})['segment_stats']['channels_touched']
    assert s1 > on['segment_stats']['channels_touched']   # fewer shots -> borrow more


def test_prior_channel_sets_matched_size_and_direction(bank, bench):
    r = EpisodeRunner(bank, bench)
    r.run({'ensemble': {}})
    from fdrfreq.pipeline import SEGMENT_DEFAULTS, deep_merge
    base_vis = r.raw_proto[:r.nb]
    tc = l2n(r.text_cal(0.5))
    r._cur_lam_t = 0.5
    k = 0.2
    want = int(round(k * bank.dim))
    masks = {}
    for rank in ('var', 'lowvar', 'mag', 'fisher', 'random', 'freq', 'ebratio'):
        s = deep_merge(SEGMENT_DEFAULTS, {'type': 'prior', 'rank': rank, 'k': k, 'r': 5})
        masks[rank] = r._prior_mask(s, 5, tc, base_vis)
        assert masks[rank].dtype == bool and masks[rank].sum() == want, rank
    assert not (masks['var'] & masks['lowvar']).any()
    # the prior 'var' rank with the union size reproduces the earlier var_union control
    s_u = deep_merge(SEGMENT_DEFAULTS, {'type': 'var_union', 'q': 6, 'r': 5})
    deltas, common = r.freq_deltas(s_u['sigmas'])
    U = r._channel_prior(s_u, 'var_union', deltas, common)
    s_p = deep_merge(SEGMENT_DEFAULTS, {'type': 'prior', 'rank': 'var', 'k': int(U.sum()), 'r': 5})
    assert np.array_equal(U, r._prior_mask(s_p, 5, tc, base_vis))


def test_prior_segments_fall_back_and_scope_matters(bank, bench):
    r = EpisodeRunner(bank, bench)
    k0 = {'ensemble': {}, 'holistic': {'donor': 'text', 'lam': 0.3, 'tau': 16.0}}
    ref = r.run(k0)['last_pred']
    seg = {'type': 'prior', 'rank': 'var', 'k': 0.2, 'r': 5}
    off = r.run({**k0, 'segments': {**seg, 'alpha': 0.0}})['last_pred']
    assert np.array_equal(ref, off)
    on = r.run({**k0, 'segments': {**seg, 'alpha': 1.0}})
    assert on['segment_stats']['channels_touched'] == int(round(0.2 * bank.dim))
    P_seg = r.vis_cal(r.run({**k0, 'segments': {**seg, 'alpha': 1.0}})['config'])
    P_glb = r.vis_cal(r.run({**k0, 'segments': {**seg, 'alpha': 1.0, 'scope': 'global'}})['config'])
    nov = slice(r.nb, None)
    assert np.allclose(np.linalg.norm(P_glb[nov], axis=1), 1.0)
    assert not np.allclose(P_seg[nov], P_glb[nov])


def test_empirical_shot_variance_and_ebmask_shrink_with_shots(bank):
    from fdrfreq.pipeline import SEGMENT_DEFAULTS, deep_merge
    one = EpisodeRunner(bank, benchmark_episode(bank, 1, shot=1))
    s = deep_merge(SEGMENT_DEFAULTS, {'type': 'prior', 'rank': 'ebmask', 's2': 'empirical', 'c': 1.0, 'r': 5})
    one._cur_lam_t = 0.5
    tc = l2n(one.text_cal(0.5))
    sig2, g2, t2, _ = one._eb_stats(s, tc, one.raw_proto[:one.nb])
    v1 = one._shot_var(1, 'empirical', 8, sig2, g2)
    v5 = one._shot_var(5, 'empirical', 8, sig2, g2)
    assert v1.sum() > 2.0 * v5.sum()
    assert np.array_equal(v1, one._shot_var(1, 'empirical', 8, sig2, g2))   # deterministic
    m1 = one._prior_mask(s, 1, tc, one.raw_proto[:one.nb])
    m5 = one._prior_mask(s, 5, tc, one.raw_proto[:one.nb])
    assert m1.sum() >= m5.sum()


def test_donor_fraction_and_full_replacement(bank, bench):
    from fdrfreq.recompose import n_donors
    assert n_donors(1.0, 50) == 50 and n_donors(0.5, 50) == 25 and n_donors(29, 50) == 29
    assert n_donors(1, 50) == 1 and n_donors(0.25, 3) == 1 and n_donors(80, 50) == 50
    r = EpisodeRunner(bank, bench)
    cfg = {'ensemble': {}, 'holistic': None,
           'segments': {'type': 'prior', 'rank': 'var', 'k': 1.0, 'alpha': 1.0, 'scope': 'global',
                        'sim': 'text', 'r': 1.0, 'tau': 22.0}}
    P = r.vis_cal(r.run(cfg)['config'])
    tc = l2n(r.text_cal(0.5))
    base_vis = l2n(r.raw_proto[:r.nb])
    for j in range(r.nb, len(r.all_cls)):
        w = softmax(22.0 * (base_vis.shape[0] and (tc[:r.nb] @ tc[j]) /
                            (np.linalg.norm(tc[:r.nb], axis=1) * np.linalg.norm(tc[j]))))
        assert np.allclose(P[j], l2n(w @ base_vis), atol=1e-10)


def test_visual_donor_full_replacement_is_exact(bank, bench):
    r = EpisodeRunner(bank, bench)
    plain = r.vis_cal(r.run({'ensemble': {}, 'holistic': None})['config'])
    cfg = {'ensemble': {}, 'holistic': None,
           'segments': {'type': 'prior', 'rank': 'var', 'k': 1.0, 'alpha': 1.0, 'scope': 'global',
                        'sim': 'visual', 'r': 1.0, 'tau': 16.0}}
    P = r.vis_cal(r.run(cfg)['config'])
    base_vis = l2n(r.raw_proto[:r.nb])
    for j in range(r.nb, len(r.all_cls)):
        w = softmax(16.0 * (base_vis @ l2n(plain[j])))
        assert np.allclose(P[j], l2n(w @ base_vis), atol=1e-10)


def test_m4a_family_is_well_formed_and_frontier_reference_is_the_locked_config(bank, tmp_path):
    from pathlib import Path
    from fdrfreq.run import main
    from fdrfreq.bank import save_bank
    root = Path(__file__).resolve().parents[1] / 'fdrfreq' / 'configs'
    entries = json.loads((root / 'paper.json').read_text())['entries']
    seen = set()
    for e in entries:
        if 'chain' in e:
            assert e['chain'] in seen, e['name']
        seen.add(e['name'])
    E = {e['name']: e for e in entries}
    assert E['M4a']['base']['segments']['sim'] == 'visual'
    assert E['M4a']['grid']['segments.alpha'] == [0.7, 1.0]
    assert E['M4a_text']['base']['segments']['sim'] == 'text'
    assert E['M4a_soft']['base']['holistic']['donor'] == 'visual'
    for n in ('P4a_Proto', 'P4a_BiMC'):
        assert E[n]['grid']['segments.alpha'] == [0.7, 1.0] and E[n]['base']['segments']['sim'] == 'visual'
    save_bank(tmp_path / 'bank', bank._arrays, bank.meta, overwrite=True)
    common = ['--bank', str(tmp_path / 'bank'), '--out', str(tmp_path)]
    main(['select', *common, '--matrix', str(root / 'paper.json'), '--split-seeds', '2', '--sequences', '1',
          '--split-mode', 'contiguous', '--entries', 'BiMC_ens', 'A1', 'A2'])
    lk = tmp_path / 'locked_A2.json'
    main(['frontier', *common, '--split-seeds', '2', '--sequences', '1', '--split-mode', 'contiguous',
          '--base-locked', str(lk),
          '--alpha-from-locked', '--override', str(root / 'frontier_m4a_fallback.json'),
          '--axes', 'prior:var', '--k', '0.2'])
    fr = json.loads(next(tmp_path.glob('frontier_shot*.json')).read_text())
    sel_a2 = json.loads((tmp_path / 'select_A2.json').read_text())
    ref_last = next((c['last'] for c in sel_a2['candidates'] if c['hash'] == sel_a2['selected_hash']),
                    (sel_a2.get('fallback_metrics') or {}).get('last'))
    assert [r['alpha'] for r in fr['rows']] == [1.0]
    assert math.isclose(fr['reference']['last'], ref_last, abs_tol=1e-6)
    from fdrfreq.run import gate_report
    assert 'G7 M4a' in gate_report(tmp_path)
