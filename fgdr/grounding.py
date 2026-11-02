"""Channel segments: frequency-grounded masks, keyword library, mask diagnostics.

Frequency grounding (visual, class-specific, computable from few shots):
  ladder delta  D_k(x) = unit(f(G_{s_{k-1}} x)) - unit(f(G_{s_k} x)),  s_0 = 0
  signature     S_{c,k} = mean_{x in c} D_k(x) - mean_{b in base} S_{b,k}
  mask          g_{c,k} = TopQ(|S_{c,k}|)
The same construction on non-frequency perturbations (``aug`` views, delta
unit(f(x)) - unit(f(aug_i x))) gives the matched control.
"""

import json
from pathlib import Path

import numpy as np

from fgdr.bimc_ref import unit


def ladder_deltas(orig, blur):
    """[n, S, D] successive differences along the blur ladder (orig first)."""
    orig = unit(orig)[:, None, :]
    blur = unit(blur)
    prev = np.concatenate([orig, blur[:, :-1]], axis=1)
    return prev - blur


def perturbation_deltas(orig, views):
    return unit(orig)[:, None, :] - unit(views)


def view_deltas(bank, view):
    if view == 'blur':
        return ladder_deltas(bank.feats, bank.views['blur'])
    if view == 'aug':
        return perturbation_deltas(bank.feats, bank.views['aug'])
    raise ValueError('view must be blur or aug')


def class_signatures(deltas, labels, classes):
    return np.stack([deltas[labels == c].mean(0) for c in classes])


def topq_masks(signatures, q):
    """Boolean masks of the q largest |values| along the last axis."""
    sig = np.abs(np.asarray(signatures))
    d = sig.shape[-1]
    q = int(min(q, d))
    idx = np.argpartition(-sig, q - 1, axis=-1)[..., :q]
    masks = np.zeros(sig.shape, dtype=bool)
    np.put_along_axis(masks, idx, True, axis=-1)
    return masks


def jaccard(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = np.logical_or(a, b).sum(-1)
    inter = np.logical_and(a, b).sum(-1)
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


def random_jaccard(q, d):
    return q / (2 * d - q)


def class_specificity(masks):
    """Mean pairwise Jaccard across classes, per band. masks [C, S, D]."""
    c = masks.shape[0]
    out = []
    for k in range(masks.shape[1]):
        m = masks[:, k].astype(np.float64)
        inter = m @ m.T
        sizes = m.sum(1)
        union = sizes[:, None] + sizes[None, :] - inter
        iu = np.triu_indices(c, 1)
        out.append(float(np.mean(inter[iu] / np.maximum(union[iu], 1))))
    return out


def shot_stability(deltas, labels, classes, reference, q, shot, draws=10, seed=0):
    """Mean Jaccard between masks from `shot` random images and from all images."""
    rng = np.random.RandomState(seed)
    full = topq_masks(class_signatures(deltas, labels, classes) - reference[None], q)
    scores = np.zeros((len(classes), deltas.shape[1]))
    for i, c in enumerate(classes):
        rows = np.flatnonzero(labels == c)
        if len(rows) <= shot:
            scores[i] = np.nan
            continue
        vals = []
        for _ in range(draws):
            pick = rng.choice(rows, size=shot, replace=False)
            sig = deltas[pick].mean(0) - reference
            vals.append(jaccard(topq_masks(sig, q), full[i]))
        scores[i] = np.mean(vals, axis=0)
    return np.nanmean(scores, axis=0).tolist()


def random_masks(shape, q, seed):
    rng = np.random.RandomState(seed)
    scores = rng.rand(*shape)
    return topq_masks(scores, q)


def keyword_random_mask(keyword_index, d, size, seed=0):
    """Fixed random mask per keyword (identical for every class), FDR's random-segment control."""
    rng = np.random.RandomState((int(seed) * 1000003 + int(keyword_index)) % (2 ** 32))
    mask = np.zeros(d, dtype=bool)
    mask[rng.choice(d, size=int(size), replace=False)] = True
    return mask


class KeywordLibrary:
    """FDR keyword->channel library built from per-(class, keyword) grounding masks.

    The library for a set of base classes is the union, per keyword, of the
    masks grounded in those classes' descriptions (FDR: ``main_knowledge_base |= mask``).
    """

    def __init__(self, pair_class, pair_keyword, masks, class_keywords, vocab, info=None):
        self.pair_class = np.asarray(pair_class, dtype=np.int64)
        self.pair_keyword = np.asarray(pair_keyword, dtype=np.int64)
        self.masks = {k: np.asarray(v, dtype=bool) for k, v in masks.items()}
        self.class_keywords = {int(k): [int(i) for i in v] for k, v in class_keywords.items()}
        self.vocab = list(vocab)
        self.info = info or {}
        self._cache = {}

    @classmethod
    def load(cls, stem):
        stem = str(stem)
        for suffix in ('.npz', '.json'):
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
        meta = json.loads(Path(stem + '.json').read_text(encoding='utf-8'))
        with np.load(stem + '.npz', allow_pickle=False) as saved:
            masks = {k[len('mask_'):]: saved[k] for k in saved.files if k.startswith('mask_')}
            pair_class, pair_keyword = saved['pair_class'], saved['pair_keyword']
        return cls(pair_class, pair_keyword, masks, meta['class_keywords'], meta['vocab'], meta)

    def library(self, base_class_ids, coord='final'):
        key = (coord, tuple(sorted(int(c) for c in base_class_ids)))
        if key not in self._cache:
            masks = self.masks[coord]
            keep = np.isin(self.pair_class, np.asarray(base_class_ids))
            lib = {}
            for kw, mask in zip(self.pair_keyword[keep], masks[keep]):
                kw = int(kw)
                lib[kw] = lib[kw] | mask if kw in lib else mask.copy()
            self._cache[key] = lib
        return self._cache[key]

    def keywords_of(self, class_id):
        return self.class_keywords.get(int(class_id), [])
