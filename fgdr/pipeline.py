"""Method specification, episode runner, grid expansion and base-only selection.

A method spec is a JSON-like dict::

    {"name": "K0", "family": "K0",
     "bimc": {...BiMC overrides...},
     "calibration": {"type": "bimc" | "none" | "holistic" | "segments", ...},
     "anchor": {"enabled": true, ...},
     "grid": {"calibration.lambda": [0.0, 0.1, 0.2], ...}}

Every strength in a grid should list its zero/identity value first: ties are
broken by grid order, so the selection falls back to the simpler model.
"""

import copy
import itertools
import json

import numpy as np

from fgdr import bimc_ref as B
from fgdr.anchor import BlockTransform, ShotAwareGDA, mix_votes
from fgdr.grounding import (class_signatures, keyword_random_mask, random_masks, topq_masks,
                            view_deltas)
from fgdr.metrics import error_decomposition, session_metrics, summarize
from fgdr.recompose import cosine, frequency_alpha_scales, holistic, segment_update

CAL_DEFAULTS = {'type': 'bimc', 'weight_source': 'visual', 'lambda': None, 'tau': None,
                'topr': None, 'fdr_softmax': False, 'segments': None}
SEG_DEFAULTS = {'source': 'keyword', 'coord': 'final', 'q': 50, 'bands': [0, 1, 2], 'alpha': 0.2,
                'alpha_per_band': None, 'tau': 16.0, 'topr': 29, 'sim_source': 'text',
                'fdr_softmax': False, 'alpha_from_frequency': False, 'alpha_shuffle': False,
                'alpha_band': 0, 'view': 'blur', 'seed': 0}
ANCHOR_DEFAULTS = {'enabled': False, 'alpha': 0.0, 'temperature': 0.3, 'ridge': 0.1, 'kappa': 0.0,
                   'mode': 'plugin', 'novel_mean': 'raw', 'blocks': [], 'residual_ridge': 1.0}


def _merge(defaults, value):
    out = copy.deepcopy(defaults)
    out.update(value or {})
    return out


def normalize_spec(spec):
    spec = copy.deepcopy(spec)
    spec.setdefault('name', 'unnamed')
    spec.setdefault('family', spec['name'].split('|')[0])
    spec['bimc'] = dict(spec.get('bimc') or {})
    spec['calibration'] = _merge(CAL_DEFAULTS, spec.get('calibration'))
    if spec['calibration']['type'] == 'segments':
        spec['calibration']['segments'] = _merge(SEG_DEFAULTS, spec['calibration'].get('segments'))
    spec['anchor'] = _merge(ANCHOR_DEFAULTS, spec.get('anchor'))
    spec.pop('grid', None)
    return spec


def _set_path(spec, path, value):
    node = spec
    keys = path.split('.')
    for key in keys[:-1]:
        if node.get(key) is None:
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def expand(spec):
    grid = spec.get('grid') or {}
    if not grid:
        s = copy.deepcopy(spec)
        s.pop('grid', None)
        s.setdefault('family', s['name'])
        return [s]
    keys = list(grid)
    out = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        s = copy.deepcopy(spec)
        s.pop('grid', None)
        s.setdefault('family', spec['name'])
        for key, value in zip(keys, combo):
            _set_path(s, key, value)
        s['name'] = spec['name'] + '|' + ','.join(f'{k}={json.dumps(v)}' for k, v in zip(keys, combo))
        out.append(s)
    return out


# --------------------------------------------------------------------------- context


def _needs_views(spec):
    cal, anc = spec['calibration'], spec['anchor']
    views = set()
    seg = cal.get('segments') or {}
    if cal['type'] == 'segments':
        if seg['source'].startswith('frequency'):
            views.add(seg['view'])
        if seg['source'] == 'perturbation':
            views.add('aug')
        if seg.get('alpha_from_frequency'):
            views.add('blur')
    if anc['enabled']:
        for block in anc['blocks']:
            if block['kind'] != 'noise':
                views.add(block['kind'])
    return views


def _base_context(ep, bank, spec, resources, base_classes):
    ctx = {'ref': {}}
    cal = spec['calibration']
    if cal['type'] == 'segments':
        seg = cal['segments']
        views = set()
        if seg['source'] in ('frequency', 'frequency_random', 'frequency_class_shuffled'):
            views.add(seg['view'])
        if seg['source'] == 'perturbation':
            views.add('aug')
        if seg['alpha_from_frequency']:
            views.add('blur')
        for view in views:
            if view not in bank.views:
                raise ValueError(f'Segment source needs cached {view} views')
            deltas = view_deltas(bank, view)
            ctx['ref'][view] = class_signatures(deltas, bank.labels, base_classes).mean(0)
        if seg['source'].startswith('keyword'):
            library = (resources or {}).get('keywords')
            if library is None:
                raise ValueError('Keyword segments need a keyword library (tools/fgdr_keywords.py)')
            ctx['keyword_library'] = library.library(ep.class_ids[base_classes], seg['coord'])
            ctx['keyword_source'] = library
    return ctx


def _segments_for(classes, bank, ep, ctx, seg, protos):
    d = protos.shape[1]
    source = seg['source']
    out = []
    if source in ('keyword', 'keyword_random'):
        lib = ctx['keyword_library']
        source_lib = ctx['keyword_source']
        sigs = None
        if seg['alpha_from_frequency']:
            deltas = view_deltas(bank, 'blur')
            sigs = class_signatures(deltas, bank.labels, classes) - ctx['ref']['blur'][None]
        rng = np.random.RandomState(seg['seed'] + 17 * int(classes[0]))
        for i, c in enumerate(classes):
            kws = [k for k in source_lib.keywords_of(ep.class_ids[c]) if k in lib]
            if source == 'keyword':
                masks = [lib[k] for k in kws]
            else:
                masks = [keyword_random_mask(k, d, int(lib[k].sum()), seg['seed']) for k in kws]
            alphas = np.full(len(masks), float(seg['alpha']))
            if sigs is not None and masks:
                scales = frequency_alpha_scales(sigs[i, seg['alpha_band']], protos[i], masks)
                if seg['alpha_shuffle']:
                    scales = rng.permutation(scales)
                alphas = alphas * scales
            out.append(list(zip(masks, alphas)))
        return out
    view = 'aug' if source == 'perturbation' else seg['view']
    deltas = view_deltas(bank, view)
    sig = class_signatures(deltas, bank.labels, classes) - ctx['ref'][view][None]
    masks = topq_masks(sig, seg['q'])
    if source == 'frequency_random':
        masks = random_masks(masks.shape, seg['q'], seg['seed'] + int(np.sum(ep.class_ids[classes])))
    elif source == 'frequency_class_shuffled':
        masks = np.roll(masks, 1, axis=0)
    per_band = seg['alpha_per_band']
    for i in range(len(classes)):
        segs = []
        for j, band in enumerate(seg['bands']):
            alpha = float(per_band[j]) if per_band is not None else float(seg['alpha'])
            segs.append((masks[i, band], alpha))
        out.append(segs)
    return out


def _calibrate(protos, classes, bank, ep, base_protos, tvec, base_classes, params, cal, ctx):
    kind = cal['type']
    if kind == 'none':
        return protos
    if kind == 'bimc':
        if not params['vision_calibration']:
            return protos
        return B.soft_calibration(base_protos, protos, params['lambda_i'], params['tau'])
    lam = params['lambda_i'] if cal['lambda'] is None else float(cal['lambda'])
    tau = params['tau'] if cal['tau'] is None else float(cal['tau'])
    if cal['weight_source'] == 'visual':
        sim = cosine(protos, base_protos)
    elif cal['weight_source'] == 'text':
        sim = cosine(tvec[classes], tvec[base_classes])
    else:
        raise ValueError('weight_source must be visual or text')
    out = holistic(protos, base_protos, sim, lam, tau, cal['topr'], cal['fdr_softmax'])
    if kind == 'holistic':
        return out
    if kind != 'segments':
        raise ValueError('unknown calibration type ' + kind)
    seg = cal['segments']
    segments = _segments_for(classes, bank, ep, ctx, seg, protos)
    updated = []
    for i, c in enumerate(classes):
        if seg['sim_source'] == 'text':
            cur_vec, base_vecs = tvec[c], tvec[base_classes]
        else:
            cur_vec, base_vecs = protos[i], base_protos
        value, _ = segment_update(out[i], base_protos, segments[i], cur_vec, base_vecs,
                                  seg['tau'], seg['topr'], seg['fdr_softmax'])
        updated.append(B.unit(value))
    return np.stack(updated)


# --------------------------------------------------------------------------- runner


class EpisodeCache:
    """Config-independent per-episode terms shared by every config of a sweep."""

    def __init__(self, ep):
        self.ep = ep
        self.bimc = {}
        self.anchor = {}

    def session_terms(self, t, gamma_base, gamma_inc):
        key = (t, float(gamma_base), float(gamma_inc))
        if key not in self.bimc:
            ep = self.ep
            covs = [B.session_covariance(ep.support[s].feats, gamma_base if s == 0 else gamma_inc)
                    for s in range(t + 1)]
            weights = [len(ep.sessions[s]) for s in range(t + 1)]
            precision = B.inverse(B.merge_covariances(covs, weights))
            seen = int(ep.sessions[t][-1]) + 1
            mask = ep.query.labels < seen
            q = B.unit(ep.query.feats[mask])
            qp = q @ precision
            keep = ep.desc_targets < seen
            knn = B.softmax(B.knn_logits(q, ep.desc[keep], ep.desc_targets[keep], seen))
            self.bimc[key] = {'mask': mask, 'precision': precision, 'qp': qp,
                              'qpq': np.einsum('nd,nd->n', qp, q), 'knn_prob': knn}
        return self.bimc[key]

    def anchor_terms(self, anc):
        key = json.dumps({'ridge': anc['ridge'], 'blocks': anc['blocks'], 'rr': anc['residual_ridge']},
                         sort_keys=True)
        if key not in self.anchor:
            ep = self.ep
            base = ep.support[0]
            transform = BlockTransform(anc['blocks']).fit(base, ridge=anc['residual_ridge'])
            zb = transform.transform(base)
            gda = ShotAwareGDA(anc['ridge']).fit(zb, base.labels, ep.sessions[0])
            zq = transform.transform(ep.query)
            zqp = zq @ gda.precision
            supports = [zb] + [transform.transform(ep.support[s]) for s in range(1, len(ep.sessions))]
            self.anchor[key] = {'precision': gda.precision, 'zq': zq, 'zqp': zqp,
                                'zqpz': np.einsum('nd,nd->n', zqp, zq), 'supports': supports}
        return self.anchor[key]


def _anchor_logits(terms, mask, means, counts, anc):
    prec = terms['precision']
    d2 = (terms['zqpz'][mask][:, None] - 2.0 * terms['zqp'][mask] @ means.T
          + np.einsum('cd,cd->c', means @ prec, means)[None, :])
    if anc['mode'] == 'plugin' or anc['kappa'] == 0:
        return -0.5 * d2
    inflate = 1.0 + anc['kappa'] / np.asarray(counts, dtype=np.float64)
    return -0.5 * d2 / inflate[None] - 0.5 * prec.shape[0] * np.log(inflate)[None]


def run_episode(ep, spec, meta_defaults, resources=None, keep_predictions=False, cache=None,
                return_prototypes=False):
    spec = normalize_spec(spec)
    params = B.resolve_params(spec['bimc'], meta_defaults)
    cache = cache if cache is not None and cache.ep is ep else EpisodeCache(ep)
    cal, anc = spec['calibration'], spec['anchor']
    c_total, d = ep.num_classes, ep.text.shape[1]
    lambda_t = params['lambda_t'] if params['text_calibration'] else 0.0
    tvec = B.unit(B.text_mix(ep.text, ep.desc_proto, lambda_t))
    img = np.zeros((c_total, d))
    counts = np.zeros(c_total, dtype=np.int64)
    sessions = []
    base_classes = ep.sessions[0]
    use_anchor = anc['enabled'] and anc['alpha'] > 0
    aterms = cache.anchor_terms(anc) if use_anchor else None
    z_means = None
    ctx = {}
    base_protos = None
    preds = labels = None
    for t, classes in enumerate(ep.sessions):
        bank = ep.support[t]
        raw, n = B.class_means(bank.feats, bank.labels, classes)
        counts[classes] = n
        protos = B.unit(raw)
        if t == 0:
            base_protos = protos
            img[classes] = protos
            ctx = _base_context(ep, bank, spec, resources, base_classes)
            if use_anchor:
                zs = aterms['supports'][0]
                z_means = np.zeros((c_total, zs.shape[1]))
                z_means[classes] = B.class_means(zs, bank.labels, classes)[0]
        else:
            img[classes] = _calibrate(protos, classes, bank, ep, base_protos, tvec, base_classes,
                                      params, cal, ctx)
            if use_anchor:
                means = B.class_means(aterms['supports'][t], bank.labels, classes)[0]
                if anc['novel_mean'] == 'calibrated':
                    means[:, :d] = img[classes] * np.linalg.norm(raw, axis=1, keepdims=True)
                z_means[classes] = means
        seen = int(classes[-1]) + 1
        terms = cache.session_terms(t, params['gamma_base'], params['gamma_inc'])
        mask = terms['mask']
        if not mask.any():
            continue
        votes = B.bimc_votes_cached(ep.query.feats[mask], img[:seen], ep.text[:seen], ep.desc_proto[:seen],
                                    terms['knn_prob'], terms['qp'], terms['qpq'], terms['precision'],
                                    ep.num_base, params)
        if use_anchor:
            logits = _anchor_logits(aterms, mask, z_means[:seen], counts[:seen], anc)
            votes = mix_votes(votes, logits, anc['alpha'], anc['temperature'])
        preds, labels = votes.argmax(1), ep.query.labels[mask]
        sessions.append(session_metrics(preds, labels, ep.num_base, t))
    result = {'name': spec['name'], 'family': spec['family'], 'sessions': sessions,
              'summary': summarize(sessions) if sessions else {}, 'info': ep.info}
    if preds is not None:
        result['summary']['errors'] = error_decomposition(preds, labels, ep.num_base)
    if return_prototypes:
        result['prototypes'] = img
    if keep_predictions:
        result['last_predictions'] = preds
        result['last_labels'] = labels
        result['last_rows'] = ep.query.rows[ep.query.labels < ep.num_classes]
    return result


def required_views(specs):
    views = set()
    for spec in specs:
        views |= _needs_views(normalize_spec(spec))
    return views


# --------------------------------------------------------------------------- selection

SUMMARY_KEYS = ('avg', 'last', 'last_base', 'last_novel', 'last_h')


def aggregate(results):
    out = {}
    for key in SUMMARY_KEYS:
        values = [r['summary'][key] for r in results if r['summary'].get(key) is not None]
        out[key] = float(np.mean(values)) if values else None
    return out


def select(rows, family, reference, max_base_drop=0.2, max_novel_drop=0.2, objective='last'):
    """Best config of a family on pseudo episodes, subject to base/novel guards.

    rows: ordered {config name: {'family', 'spec', 'mean'}}. Ties go to the earlier
    (milder) grid entry. Returns None if no config satisfies the guards.
    """
    ref = rows[reference]['mean']
    best, best_value = None, -np.inf
    for name, row in rows.items():
        if row['family'] != family:
            continue
        mean = row['mean']
        if mean['last_base'] < ref['last_base'] - max_base_drop:
            continue
        if ref['last_novel'] is not None and mean['last_novel'] < ref['last_novel'] - max_novel_drop:
            continue
        value = round(mean[objective], 6)
        if value > best_value:
            best, best_value = name, value
    return best
