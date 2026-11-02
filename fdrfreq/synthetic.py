"""Small synthetic bank with CLIP-like structure, for tests and CPU smoke runs.

Built so that every component has something to find: text is a noisy view of
the visual class mean (modality alignment), blur views remove a sparse
class-specific "detail" vector (frequency grounding has signal), one patch
DCT band is class-informative, and keywords have word-specific channels.
"""

import numpy as np

from .bank import Bank, l2n
from .protocol import task_classes, replica_support_indices


def make_bank(num_classes=30, num_init=18, num_inc=3, dim=48, n_train=24, n_test=12,
              inc_shot=5, sigmas=(1, 2, 4), dct_layers=(6,), width=16, seed=0):
    r = np.random.RandomState(seed)
    mu = l2n(r.randn(num_classes, dim))
    detail = np.zeros((num_classes, dim))
    for c in range(num_classes):
        idx = r.choice(dim, 6, replace=False)
        detail[c, idx] = r.randn(6)
    arrays = {}

    def split(n_per):
        labels = np.repeat(np.arange(num_classes), n_per)
        perm = r.permutation(len(labels))
        labels = labels[perm]
        base = mu[labels] + 0.55 * r.randn(len(labels), dim) / np.sqrt(dim) * 4
        feats = {'orig': l2n(base + 0.5 * detail[labels])}
        for s in sigmas:
            feats[f'blur{float(s):g}'] = l2n(base + 0.5 * detail[labels] * np.exp(-s / 2.0)
                                             + 0.02 * r.randn(len(labels), dim))
        feats['flip'] = l2n(base + 0.5 * detail[labels] + 0.05 * r.randn(len(labels), dim))
        feats['gray'] = l2n(base + 0.3 * detail[labels] + 0.05 * r.randn(len(labels), dim))
        dct = {}
        for L in dct_layers:
            e = np.abs(r.randn(len(labels), 3, width)) * 0.3
            proj = np.random.RandomState(L).randn(num_classes, width)
            e[:, 1] += np.abs(proj[labels]) * 0.5
            dct[L] = e.astype(np.float16)
        return labels, feats, dct

    ytr, ftr, dtr = split(n_train)
    yte, fte, dte = split(n_test)
    arrays['train_labels'], arrays['test_labels'] = ytr.astype(np.int64), yte.astype(np.int64)
    for v in ftr:
        arrays['train_feat_' + v] = ftr[v].astype(np.float32)
        arrays['test_feat_' + v] = fte[v].astype(np.float32)
    for L in dct_layers:
        for kind in ('radial', 'random'):
            arrays[f'train_dct_L{L}_{kind}'] = dtr[L] if kind == 'radial' else np.abs(
                r.randn(*dtr[L].shape)).astype(np.float16)
            arrays[f'test_dct_L{L}_{kind}'] = dte[L] if kind == 'radial' else np.abs(
                r.randn(*dte[L].shape)).astype(np.float16)
    arrays['text_prompt'] = l2n(mu + 0.6 * r.randn(num_classes, dim) / np.sqrt(dim) * 3).astype(np.float32)
    desc, tgt = [], []
    for c in range(num_classes):
        for _ in range(4):
            desc.append(l2n(arrays['text_prompt'][c] + 0.3 * r.randn(dim) / np.sqrt(dim) * 3))
            tgt.append(c)
    desc = np.asarray(desc, dtype=np.float32)
    tgt = np.asarray(tgt, dtype=np.int64)
    arrays['desc_feat'], arrays['desc_target'] = desc, tgt
    arrays['desc_proto'] = l2n(np.stack([desc[tgt == c].mean(0) for c in range(num_classes)])).astype(np.float32)
    vocab = [f'word{i}' for i in range(24)]
    word_ch = [r.choice(dim, 10, replace=False) for _ in vocab]
    pc, pw, ptf, sc = [], [], [], []
    for c in range(num_classes):
        for w in r.choice(len(vocab), 6, replace=False):
            pc.append(c)
            pw.append(int(w))
            ptf.append(int(r.randint(1, 5)))
            s = np.abs(r.randn(dim)) * 0.1
            s[word_ch[w]] += 1.0
            sc.append(s)
    arrays['kw_pair_class'] = np.asarray(pc, np.int64)
    arrays['kw_pair_word'] = np.asarray(pw, np.int64)
    arrays['kw_pair_tf'] = np.asarray(ptf, np.int64)
    arrays['kw_score_final'] = np.asarray(sc, np.float16)
    arrays['kw_score_fdr'] = np.asarray(np.roll(sc, 3, axis=1), np.float16)
    tasks = task_classes(num_classes, num_init, num_inc)
    for s in (1, 2, 3):
        sup = replica_support_indices(ytr, tasks, -1, inc_shot, s)
        arrays[f'support_seed{s}'] = np.concatenate(sup)
        arrays[f'support_seed{s}_offsets'] = np.cumsum([0] + [len(x) for x in sup]).astype(np.int64)
    meta = {'dataset': 'synthetic', 'backbone': 'none', 'class_names': [f'class {i}' for i in range(num_classes)],
            'num_init': num_init, 'num_inc': num_inc, 'base_shot': -1, 'inc_shot': inc_shot,
            'beta': 0.5, 'ensemble_alpha': 0.7, 'kw_vocab': vocab, 'sigmas': list(sigmas),
            'dct_layers': list(dct_layers)}
    return Bank(arrays, meta)
