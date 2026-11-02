"""Lexical attribute keywords for the FDR-style keyword library (no LLM needed).

FDR asks an LLM for m keywords per class. Here keywords are content words of the
existing BiMC descriptions, chosen by tf-idf over base classes:
  * vocabulary = stems that occur in >= min_classes and <= max_class_frac of base classes
  * class keywords = top-m vocabulary stems by tf * log(B / df) in that class's descriptions
Class-name words are excluded. Novel classes can only use vocabulary stems (as in
FDR, where keywords absent from the base knowledge base are skipped).
"""

import math
import re
from collections import Counter

STOPWORDS = set('''
a about above across after again against all almost along also although always am among an and
another any are around as at away be because been before being below between both but by can
could did do does doing down during each either else even ever every few for from further had has
have having he her here hers him his how however i if in into is it its itself just like made make
many may might more most much must my near nearly no nor not of off often on once one only onto or
other others our out over own part parts per perhaps quite rather really same seen several she
should since so some sometimes such than that the their them then there these they this those
though through throughout thus to too toward towards two under until up upon usually various very
was we were what when where whether which while who whom whose why will with within without would
you your typically generally commonly especially overall appearance appears appear looks look
looking image images photo photos picture pictures identify identified identifying distinctive
characteristic characteristics feature features featuring known species type kind called usually
something including include includes found visible noticeable typical typically often slightly
'''.split())

GENERIC = {'bird', 'birds', 'animal', 'object', 'thing', 'things', 'body', 'color', 'colour',
           'colors', 'colours', 'colored', 'coloured', 'shape', 'shaped', 'size', 'sized', 'pattern',
           'patterns', 'area', 'areas', 'side', 'sides', 'top', 'bottom', 'front', 'back'}

WORD = re.compile(r"[a-z]+")


def stem(word):
    if len(word) > 4 and word.endswith('ies'):
        return word[:-3] + 'y'
    if len(word) > 4 and word.endswith('es') and word[-3] in 'sxz':
        return word[:-2]
    if len(word) > 3 and word.endswith('s') and not word.endswith('ss'):
        return word[:-1]
    return word


def name_words(class_name):
    return {stem(w) for w in WORD.findall(class_name.lower().replace('_', ' ').replace('-', ' '))}


def content_tokens(text, excluded):
    out = []
    for word in WORD.findall(text.lower()):
        if len(word) < 3 or word in STOPWORDS or word in GENERIC:
            continue
        s = stem(word)
        if s in excluded or s in GENERIC:
            continue
        out.append((s, word))
    return out


def extract_keywords(descriptions, class_names, base_class_ids, m=8, min_classes=2,
                     max_class_frac=0.5):
    """descriptions: list (per class id) of lists of strings.

    Returns vocab (list of stems), class_keywords {class_id: [vocab idx]}, and
    surface {class_id: {vocab idx: (sentence, surface word)}} for grounding.
    """
    counts, surfaces = {}, {}
    for cid, (texts, name) in enumerate(zip(descriptions, class_names)):
        excluded = name_words(name)
        counter, surface = Counter(), {}
        for sentence in texts:
            for s, w in content_tokens(sentence, excluded):
                counter[s] += 1
                surface.setdefault(s, (sentence, w))
        counts[cid], surfaces[cid] = counter, surface
    base = list(base_class_ids)
    df = Counter()
    for cid in base:
        df.update(set(counts[cid]))
    limit = max_class_frac * len(base)
    vocab = sorted(s for s, n in df.items() if min_classes <= n <= limit)
    index = {s: i for i, s in enumerate(vocab)}
    class_keywords, surface_out = {}, {}
    for cid, counter in counts.items():
        scored = [(c * math.log(len(base) / df[s]), s) for s, c in counter.items() if s in index]
        scored.sort(key=lambda t: (-t[0], t[1]))
        chosen = [s for _, s in scored[:m]]
        class_keywords[cid] = [index[s] for s in chosen]
        surface_out[cid] = {index[s]: surfaces[cid][s] for s in chosen}
    return vocab, class_keywords, surface_out
