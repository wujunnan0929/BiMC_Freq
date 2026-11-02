"""LLM-free attribute keywords and the FDR keyword-to-channel library.

Extraction time (``candidate_words``): every class gets its most frequent
content words from its own BiMC descriptions, excluding stop words, generic
words and the class-name tokens. ``fdrfreq.extract`` grounds each
(class, word) pair in the frozen text encoder and stores a [D] score.

Run time (``build_library`` / ``class_keywords``): only the *base classes of
the current protocol* (pseudo-base A in development, the real base classes in
evaluation) define the vocabulary and the shared keyword masks. A word is a
transferable attribute if it occurs in >= min_df and <= max_df * |base|
base classes. As in FDR, a keyword's mask is the union of the top-q channels
over all base classes that use it. A novel class keeps its top-m library
words by tf * idf (idf from base classes only, so no future-class text is
used to decide anything about the base library).
"""

import math
import re

import numpy as np

STOP = set('''a an the and or of to in on at by for with from as is are was were be been being
this that these those it its it's their there they them he she his her which who whom whose what
when where while than then also very more most much many some any each every other such only
own same so not no nor but if into over under about above below between through during before
after up down out off again further once can will just should would could may might must shall
do does did done having have has had typically usually often generally commonly mostly mainly
always sometimes may appear appears appearing look looks looking like seen see known called
photo image picture species bird birds animal animals object objects thing things type types kind
kinds part parts one two three four several various different distinctive characteristic
characteristics feature features featuring identify identified identifying recognize
recognized recognizing overall relatively slightly somewhat fairly quite rather
small large medium sized size shape shaped color colored colour coloured'''.split())


def tokenize_words(text):
    return re.findall(r"[a-z][a-z\-]+[a-z]", text.lower())


def name_tokens(name):
    words = set(tokenize_words(name.replace('_', ' ')))
    plural = {w + 's' for w in words} | {w + 'es' for w in words}
    return words | plural


def candidate_words(sentences, class_name, max_words=40, min_len=3):
    """Return [(word, tf)] for one class, most frequent first (ties alphabetical)."""
    banned = STOP | name_tokens(class_name)
    counts = {}
    for s in sentences:
        for w in tokenize_words(s):
            if len(w) < min_len or w in banned:
                continue
            counts[w] = counts.get(w, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:max_words]


def topq_mask(score, q):
    score = np.abs(np.asarray(score, dtype=np.float64))
    q = int(min(q, score.shape[-1]))
    idx = np.argpartition(-score, q - 1)[:q]
    m = np.zeros(score.shape[-1], dtype=bool)
    m[idx] = True
    return m


class KeywordLibrary:
    def __init__(self, masks, df, n_base, coord, q):
        self.masks, self.df, self.n_base = masks, df, n_base
        self.coord, self.q = coord, q

    def idf(self, word):
        return math.log(self.n_base / max(self.df.get(word, 0), 1))

    def random_twin(self, seed=0):
        """Same vocabulary and mask sizes, channels drawn at random (FDR ablation)."""
        out = {}
        for w, m in self.masks.items():
            r = np.random.RandomState((int(seed) * 1000003 + int(w) * 7919) % (2 ** 31 - 1))
            k = int(m.sum())
            idx = r.choice(m.shape[0], size=k, replace=False)
            mm = np.zeros_like(m)
            mm[idx] = True
            out[w] = mm
        return KeywordLibrary(out, self.df, self.n_base, self.coord + '_random', self.q)


def build_library(bank, base_classes, q=50, coord='final', min_df=2, max_df=0.3):
    if not bank.has_keywords():
        raise ValueError('Bank has no keyword grounding; re-run extraction with --keywords.')
    key = 'kw_score_' + coord
    if not bank.has(key):
        raise ValueError(f'Bank has no {key}.')
    pc = np.asarray(bank.array('kw_pair_class'))
    pw = np.asarray(bank.array('kw_pair_word'))
    score = bank.array(key)
    base_classes = np.asarray(base_classes)
    sel = np.flatnonzero(np.isin(pc, base_classes))
    words, df_counts = np.unique(pw[sel], return_counts=True)
    df = dict(zip(words.tolist(), df_counts.tolist()))
    upper = max_df * len(base_classes)
    ok = {w for w, n in df.items() if min_df <= n <= upper}
    masks = {}
    for i in sel:
        w = int(pw[i])
        if w not in ok:
            continue
        m = topq_mask(score[i], q)
        masks[w] = masks[w] | m if w in masks else m
    return KeywordLibrary(masks, df, len(base_classes), coord, q)


def class_keywords(bank, c, library, m=8):
    pc = np.asarray(bank.array('kw_pair_class'))
    pw = np.asarray(bank.array('kw_pair_word'))
    tf = np.asarray(bank.array('kw_pair_tf'))
    rows = np.flatnonzero(pc == c)
    ranked = []
    for i in rows:
        w = int(pw[i])
        if w in library.masks:
            ranked.append((tf[i] * library.idf(w), w))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    return [w for _, w in ranked[:m]]
