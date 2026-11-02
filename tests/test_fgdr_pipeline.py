import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fgdr.grounding import KeywordLibrary
from fgdr.pipeline import aggregate, expand, run_episode, select
from fgdr.protocol import benchmark_episode, pseudo_episode
from fgdr_synth import synthetic_arrays


def _episode(shot=5):
    train, test, meta = synthetic_arrays()
    return benchmark_episode(train, test, meta, shot, 1), meta


def _keywords(num_classes=30, dim=32, seed=0):
    rng = np.random.RandomState(seed)
    vocab = [f'kw{i}' for i in range(12)]
    pair_class, pair_kw, masks, class_kw = [], [], [], {}
    for c in range(num_classes):
        kws = sorted(rng.choice(12, 3, replace=False).tolist())
        class_kw[str(c)] = kws
        for k in kws:
            m = np.zeros(dim, bool); m[rng.choice(dim, 6, replace=False)] = True
            pair_class.append(c); pair_kw.append(k); masks.append(m)
    masks = np.stack(masks)
    return KeywordLibrary(pair_class, pair_kw, {'final': masks, 'fdr': masks[::-1].copy()}, class_kw, vocab)


def _last(result):
    return result['summary']['last']


def test_bimc_reference_equals_visual_holistic_and_identity_fallbacks():
    ep, meta = _episode()
    d = meta['bimc_defaults']
    ref = run_episode(ep, {'name': 'bimc'}, d, keep_predictions=True)
    hol = run_episode(ep, {'name': 'h', 'calibration': {'type': 'holistic', 'weight_source': 'visual'}}, d,
                      keep_predictions=True)
    assert np.array_equal(ref['last_predictions'], hol['last_predictions'])
    lib = _keywords()
    seg0 = run_episode(ep, {'name': 's', 'calibration': {'type': 'segments', 'weight_source': 'visual',
                                                         'segments': {'source': 'keyword', 'alpha': 0.0}}},
                       d, {'keywords': lib}, keep_predictions=True)
    assert np.array_equal(ref['last_predictions'], seg0['last_predictions'])
    anc0 = run_episode(ep, {'name': 'a', 'anchor': {'enabled': True, 'alpha': 0.0}}, d, keep_predictions=True)
    assert np.array_equal(ref['last_predictions'], anc0['last_predictions'])
    assert len(ref['sessions']) == 5 and ref['sessions'][-1]['n'] == len(ep.query.labels)


def test_all_method_families_run():
    ep, meta = _episode()
    d = meta['bimc_defaults']
    res = {'keywords': _keywords()}
    specs = [
        {'name': 'K0', 'calibration': {'type': 'holistic', 'weight_source': 'text', 'lambda': 0.2, 'tau': 22}},
        {'name': 'K1', 'calibration': {'type': 'segments', 'weight_source': 'text', 'lambda': 0.2,
                                       'segments': {'source': 'keyword', 'alpha': 0.3, 'topr': 5}}},
        {'name': 'K1fdr', 'calibration': {'type': 'segments', 'weight_source': 'text', 'lambda': 0.2,
                                          'segments': {'source': 'keyword', 'coord': 'fdr', 'alpha': 0.3,
                                                       'topr': 5, 'fdr_softmax': True}}},
        {'name': 'K2', 'calibration': {'type': 'segments', 'segments': {'source': 'keyword_random', 'alpha': 0.3}}},
        {'name': 'Q1', 'calibration': {'type': 'segments', 'weight_source': 'text', 'lambda': 0.2,
                                       'segments': {'source': 'frequency', 'q': 8, 'alpha': 0.3, 'topr': 5}}},
        {'name': 'Q2', 'calibration': {'type': 'segments', 'segments': {'source': 'frequency_class_shuffled', 'q': 8}}},
        {'name': 'Q3', 'calibration': {'type': 'segments', 'segments': {'source': 'frequency_random', 'q': 8}}},
        {'name': 'Qp', 'calibration': {'type': 'segments', 'segments': {'source': 'perturbation', 'q': 8,
                                                                         'bands': [0, 1]}}},
        {'name': 'Q4', 'calibration': {'type': 'segments', 'segments': {'source': 'keyword', 'alpha': 0.3,
                                                                         'alpha_from_frequency': True}}},
        {'name': 'A2', 'anchor': {'enabled': True, 'alpha': 0.2, 'kappa': 1.0, 'mode': 'predictive'}},
        {'name': 'F1', 'anchor': {'enabled': True, 'alpha': 0.2,
                                  'blocks': [{'kind': 'dct', 'layer': 1, 'band': 2, 'dim': 4},
                                             {'kind': 'noise', 'dim': 4}]}},
        {'name': 'cal', 'anchor': {'enabled': True, 'alpha': 0.2, 'novel_mean': 'calibrated'}},
    ]
    for spec in specs:
        out = run_episode(ep, spec, d, res)
        assert 0 <= _last(out) <= 100, spec['name']
        assert out['summary']['errors']['novel_errors'] >= 0


def test_one_shot_runs():
    ep, meta = _episode(shot=1)
    out = run_episode(ep, {'name': 'a', 'anchor': {'enabled': True, 'alpha': 0.2, 'kappa': 1.0,
                                                    'mode': 'predictive'}}, meta['bimc_defaults'])
    assert len(out['sessions']) == 5


def test_expand_and_select_prefer_mild_on_ties():
    spec = {'name': 'K0', 'calibration': {'type': 'holistic', 'weight_source': 'text'},
            'grid': {'calibration.lambda': [0.0, 0.1, 0.2], 'calibration.tau': [16, 22]}}
    configs = expand(spec)
    assert len(configs) == 6 and configs[0]['calibration']['lambda'] == 0.0
    assert all(c['family'] == 'K0' for c in configs)
    rows = {'ref': {'family': 'ref', 'mean': {'last': 50, 'last_base': 60, 'last_novel': 40}}}
    for i, c in enumerate(configs):
        rows[c['name']] = {'family': 'K0', 'mean': {'last': 51 if i in (2, 4) else 50,
                                                    'last_base': 60 if i != 2 else 59, 'last_novel': 40}}
    assert select(rows, 'K0', 'ref', max_base_drop=0.2) == configs[4]['name']
    assert select(rows, 'K0', 'ref', max_base_drop=2.0) == configs[2]['name']


def test_pseudo_episode_runs_full_pipeline():
    train, _, meta = synthetic_arrays(num_classes=30, num_init=20)
    ep = pseudo_episode(train, meta, n_fit=10, way=5, n_sessions=2, shot=5, split_seed=2, sequence_seed=1)
    out = run_episode(ep, {'name': 'Q1', 'calibration': {'type': 'segments', 'weight_source': 'text',
                                                          'segments': {'source': 'frequency', 'q': 8}}},
                      meta['bimc_defaults'])
    agg = aggregate([out, out])
    assert agg['last'] == out['summary']['last']


def test_cached_votes_match_reference_and_shared_cache_is_neutral():
    from fgdr import bimc_ref as B
    from fgdr.anchor import ShotAwareGDA
    from fgdr.pipeline import EpisodeCache, _anchor_logits
    rng = np.random.RandomState(7)
    q, img = B.unit(rng.randn(9, 8)), B.unit(rng.randn(4, 8))
    text, dproto = B.unit(rng.randn(4, 8)), B.unit(rng.randn(4, 8))
    desc, targets = B.unit(rng.randn(8, 8)), np.repeat(np.arange(4), 2)
    prec = np.linalg.inv(B.session_covariance(B.unit(rng.randn(30, 8)), 1.0))
    p = B.resolve_params({}, {'beta': 0.65, 'ensemble_alpha': 0.6})
    qp = q @ prec
    cached = B.bimc_votes_cached(q, img, text, dproto, B.softmax(B.knn_logits(q, desc, targets, 4)), qp,
                                 np.einsum('nd,nd->n', qp, q), prec, 2, p)
    assert np.allclose(cached, B.bimc_votes(q, img, text, dproto, desc, targets, prec, 2, p))
    z = rng.randn(40, 5); labels = np.repeat(np.arange(4), 10)
    gda = ShotAwareGDA(0.1, 1.0, 'predictive').fit(z, labels, range(4))
    zq = rng.randn(6, 5); means = rng.randn(4, 5); counts = np.array([10, 10, 5, 1])
    zqp = zq @ gda.precision
    terms = {'precision': gda.precision, 'zqp': zqp, 'zqpz': np.einsum('nd,nd->n', zqp, zq)}
    anc = {'mode': 'predictive', 'kappa': 1.0}
    assert np.allclose(_anchor_logits(terms, np.ones(6, bool), means, counts, anc), gda.logits(zq, means, counts))
    ep, meta = _episode()
    d = meta['bimc_defaults']
    shared = EpisodeCache(ep)
    specs = [{'name': 'a', 'anchor': {'enabled': True, 'alpha': 0.2, 'kappa': 1.0, 'mode': 'predictive'}},
             {'name': 'k', 'calibration': {'type': 'holistic', 'weight_source': 'text', 'lambda': 0.3}},
             {'name': 'g', 'bimc': {'gamma_inc': 2.0}}]
    for spec in specs:
        fresh = run_episode(ep, spec, d, keep_predictions=True)
        reused = run_episode(ep, spec, d, keep_predictions=True, cache=shared)
        assert np.array_equal(fresh['last_predictions'], reused['last_predictions'])
