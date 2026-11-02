import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fgdr import bimc_ref as B
from fgdr import cache as fc
from fgdr.anchor import BlockTransform, ShotAwareGDA, mix_votes, pooled_within_covariance
from fgdr.grounding import (KeywordLibrary, class_specificity, jaccard, keyword_random_mask,
                            ladder_deltas, random_jaccard, random_masks, shot_stability, topq_masks)
from fgdr.keywords import extract_keywords, stem
from fgdr.metrics import mcnemar_exact, paired_counts, oracle_gain
from fgdr.protocol import Bank, benchmark_episode, pseudo_episode, replica_support_indices
from fgdr.recompose import donor_weights, holistic, segment_update
from fgdr_synth import synthetic_arrays, write_cache


def test_replica_support_matches_datamanager_logic():
    targets = np.repeat(np.arange(6), 8)
    sessions = [np.arange(0, 2), np.arange(2, 4), np.arange(4, 6)]
    draw = replica_support_indices(targets, sessions, -1, 3, seed=5)
    rng = np.random.RandomState(5)
    expected = [np.flatnonzero(targets == 0), np.flatnonzero(targets == 1)]
    assert np.array_equal(draw[0], np.concatenate(expected))
    second = [rng.choice(np.flatnonzero(targets == c), 3, replace=False) for c in (2, 3)]
    assert np.array_equal(draw[1], np.concatenate(second))
    assert len(draw[2]) == 6


def test_holistic_visual_equals_bimc_soft_calibration():
    rng = np.random.RandomState(0)
    base, cur = B.unit(rng.randn(7, 16)), B.unit(rng.randn(3, 16))
    ref = B.soft_calibration(base, cur, 0.1, 16.0)
    sim = B.unit(cur) @ B.unit(base).T
    assert np.allclose(holistic(cur, base, sim, 0.1, 16.0), ref)


def test_donor_weights_topr_and_fdr_softmax():
    sim = np.array([[0.9, 0.1, 0.5, 0.2]])
    w = donor_weights(sim, 10.0, topr=2)
    assert np.count_nonzero(w) == 2 and np.isclose(w.sum(), 1)
    wf = donor_weights(sim, 10.0, topr=2, fdr_softmax=True)
    assert np.all(wf > 0) and np.isclose(wf.sum(), 1)


def test_segment_update_zero_alpha_is_identity_and_only_touches_mask():
    rng = np.random.RandomState(1)
    base, proto = B.unit(rng.randn(5, 12)), B.unit(rng.randn(12))
    mask = np.zeros(12, bool); mask[:4] = True
    same, applied = segment_update(proto, base, [(mask, 0.0)], proto, base, 16.0)
    assert applied == 0 and np.allclose(same, proto)
    moved, applied = segment_update(proto, base, [(mask, 0.5)], proto, base, 16.0)
    assert applied == 1 and np.allclose(moved[~mask], proto[~mask]) and not np.allclose(moved[mask], proto[mask])


def test_bimc_votes_match_manual_formula():
    rng = np.random.RandomState(2)
    q, img = B.unit(rng.randn(4, 8)), B.unit(rng.randn(3, 8))
    text, dproto = B.unit(rng.randn(3, 8)), B.unit(rng.randn(3, 8))
    desc, targets = B.unit(rng.randn(6, 8)), np.array([0, 0, 1, 1, 2, 2])
    cov = B.session_covariance(B.unit(rng.randn(20, 8)), 1.0)
    prec = np.linalg.inv(cov)
    p = B.resolve_params({}, {'beta': 0.3, 'ensemble_alpha': 0.8})
    votes = B.bimc_votes(q, img, text, dproto, desc, targets, prec, 2, p)
    fused = B.unit(0.3 * (0.5 * text + 0.5 * dproto) + 0.7 * img)
    pf = B.softmax(q @ fused.T)
    d2 = np.array([[(x - m) @ prec @ (x - m) for m in img] for x in q])
    pc = B.softmax(-d2 / 512)
    knn = np.stack([np.max(q @ desc[targets == c].T, axis=1) for c in range(3)], 1)
    pk = B.softmax(knn)
    manual = np.concatenate([0.8 * pf[:, :2] + 0.2 * pc[:, :2], 0.8 * pf[:, 2:] + 0.2 * pk[:, 2:]], 1)
    assert np.allclose(votes, manual)


def test_torch_cov_convention():
    x = np.random.RandomState(3).randn(10, 4)
    cov = B.session_covariance(x, 0.0)
    centered = x - x.mean(0)
    assert np.allclose(cov, centered.T @ centered / 9)


def test_gda_predictive_kappa_zero_equals_plugin_and_mixing_identity():
    rng = np.random.RandomState(4)
    z = rng.randn(60, 6); labels = np.repeat(np.arange(3), 20)
    g0 = ShotAwareGDA(0.1, 0.0, 'predictive').fit(z, labels, range(3))
    g1 = ShotAwareGDA(0.1, 0.0, 'plugin').fit(z, labels, range(3))
    means, counts = rng.randn(3, 6), np.array([20, 20, 5])
    assert np.allclose(g0.logits(z, means, counts), g1.logits(z, means, counts))
    votes = np.abs(rng.rand(60, 3))
    assert mix_votes(votes, rng.randn(60, 3), 0.0, 1.0) is votes
    g2 = ShotAwareGDA(0.1, 1.0, 'predictive').fit(z, labels, range(3))
    diff = g2.logits(z, means, counts) - g1.logits(z, means, counts)
    assert not np.allclose(diff[:, 2], diff[:, 0])


def test_pooled_within_covariance_ignores_between_class_scatter():
    rng = np.random.RandomState(5)
    noise = rng.randn(40, 3) * 0.1
    labels = np.repeat([0, 1], 20)
    shifted = noise + np.where(labels[:, None] == 1, 10.0, 0.0)
    assert np.allclose(pooled_within_covariance(noise, labels, [0, 1]),
                       pooled_within_covariance(shifted, labels, [0, 1]))


def test_block_transform_residualises_against_f():
    train, _, _ = synthetic_arrays()
    rows = np.arange(len(train['labels']))
    bank = Bank(train['final'], train['labels'], rows, 'train',
                {'dct': train['dct'].astype(np.float32), 'blur': train['blur'].astype(np.float32)})
    t = BlockTransform([{'kind': 'dct', 'layer': 0, 'band': 1, 'dim': 4},
                        {'kind': 'blur', 'index': 0, 'dim': 4}, {'kind': 'noise', 'dim': 3}]).fit(bank)
    z = t.transform(bank)
    assert z.shape == (len(rows), 32 + 11)
    assert np.allclose(z[:, :32], train['final'])
    z2 = t.transform(bank)
    assert np.allclose(z, z2)


def test_mask_tools():
    sig = np.array([[[0.1, -0.9, 0.3, 0.0]]])
    m = topq_masks(sig, 2)
    assert m[0, 0].tolist() == [False, True, True, False]
    rm = random_masks((40, 1, 200), 20, seed=0)
    spec = class_specificity(rm)[0]
    assert abs(spec - random_jaccard(20, 200)) < 0.03
    a = keyword_random_mask(3, 50, 10); b = keyword_random_mask(3, 50, 10)
    assert np.array_equal(a, b) and a.sum() == 10
    assert jaccard(a, a) == 1.0


def test_ladder_and_shot_stability_run():
    train, _, _ = synthetic_arrays()
    d = ladder_deltas(train['final'], train['blur'].astype(np.float32))
    assert d.shape == train['blur'].shape
    ref = d.mean(0)
    vals = shot_stability(d, train['labels'], range(5), ref, q=8, shot=5, draws=3)
    assert len(vals) == 3 and all(0 <= v <= 1 for v in vals)


def test_keyword_library_union():
    masks = {'final': np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], bool)}
    lib = KeywordLibrary([0, 1, 2], [7, 7, 8], masks, {'5': [7, 8]}, ['a'] * 9)
    union = lib.library([0, 1], 'final')
    assert union[7].tolist() == [True, True, False, False] and 8 not in union
    assert lib.keywords_of(5) == [7, 8]


def test_keyword_extraction_excludes_names_and_generic_words():
    descs = [['A red cardinal has a crest and a red bill.', 'crest on head'],
             ['The blue jay has a crest and blue wings.'],
             ['The sparrow has streaked brown wings.'],
             ['A gull has grey wings and webbed feet.']]
    names = ['Northern Cardinal', 'Blue Jay', 'House Sparrow', 'Herring Gull']
    vocab, kw, surface = extract_keywords(descs, names, [0, 1, 2, 3], m=4, min_classes=2, max_class_frac=0.75)
    assert 'cardinal' not in vocab and 'crest' in vocab and stem('wings') in vocab
    assert all(vocab[i] != 'blue' for i in kw[1])
    assert set(surface[0]) == set(kw[0])


def test_mcnemar_and_paired_counts():
    labels = np.array([0, 1, 2, 3])
    c = paired_counts(np.array([0, 0, 0, 0]), np.array([0, 1, 0, 3]), labels)
    assert c['C'] == 2 and c['D'] == 0
    assert mcnemar_exact(5, 5) == 1.0 and mcnemar_exact(10, 0) < 0.01
    og = oracle_gain(np.array([0, 0, 2, 3]), np.array([0, 1, 0, 3]), labels)
    assert og['oracle_gain'] == 25.0


def test_cache_roundtrip_and_test_lock(tmp_path):
    stem = tmp_path / 'syn'
    train, test, meta = write_cache(stem)
    arrays, meta2 = fc.load_train(stem)
    assert np.allclose(arrays['final'], train['final']) and meta2['dataset'] == 'synthetic'
    bad = tmp_path / 'lock.json'
    bad.write_text(json.dumps({'status': 'draft'}))
    with pytest.raises(ValueError):
        fc.load_test(stem, bad)
    lock = {'status': 'locked', 'cache_meta_sha256': meta2['meta_sha256'], 'selected': {}}
    lock['lock_sha256'] = fc.canonical_sha(lock)
    good = tmp_path / 'good.json'
    good.write_text(json.dumps(lock))
    arrays_t, _, _ = fc.load_test(stem, good)
    assert len(arrays_t['labels']) == len(test['labels'])


def test_pseudo_episode_is_base_only_and_disjoint():
    train, _, meta = synthetic_arrays(num_classes=30, num_init=20)
    ep = pseudo_episode(train, meta, n_fit=10, way=5, n_sessions=2, shot=5, split_seed=2, sequence_seed=0)
    used = np.concatenate([b.rows for b in ep.support])
    assert np.all(np.isin(train['labels'][used], np.arange(20)))
    assert np.all(np.isin(train['labels'][ep.query.rows], np.arange(20)))
    assert not set(used.tolist()) & set(ep.query.rows.tolist())
    assert ep.num_base == 10 and ep.num_classes == 20


def test_benchmark_episode_layout():
    train, test, meta = synthetic_arrays()
    ep = benchmark_episode(train, test, meta, 5, 1)
    assert ep.num_base == 10 and len(ep.sessions) == 5
    assert all(len(b.labels) == 25 for b in ep.support[1:])
    assert np.array_equal(ep.class_ids, np.arange(30))
