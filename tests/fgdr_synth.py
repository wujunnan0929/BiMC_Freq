"""Synthetic FGDR caches for tests (no CLIP, no images)."""

import numpy as np

from fgdr import cache as fc
from fgdr.protocol import make_sessions, replica_support_indices, support_key


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def synthetic_arrays(num_classes=30, num_init=10, num_inc=5, dim=32, train_per_class=12,
                     test_per_class=6, blur_levels=3, aug_levels=2, layers=2, width=16, seed=0):
    rng = np.random.RandomState(seed)
    means = unit(rng.randn(num_classes, dim))
    fine = unit(rng.randn(num_classes, dim))          # class-specific "fine detail" direction

    def images(count):
        labels = np.repeat(np.arange(num_classes), count)
        coarse = means[labels] + 0.35 * rng.randn(len(labels), dim)
        detail = 0.4 * fine[labels]
        final = unit(coarse + detail)
        blur = np.stack([unit(coarse + detail * (1 - (k + 1) / (blur_levels + 1))
                              + 0.02 * rng.randn(len(labels), dim)) for k in range(blur_levels)], 1)
        aug = np.stack([unit(final + 0.15 * rng.randn(len(labels), dim)) for _ in range(aug_levels)], 1)
        dct = np.abs(rng.randn(len(labels), layers, 3, width)) + 0.1 * np.abs(np.resize(fine[labels], (len(labels), width))[:, None, None, :])
        return {'final': final.astype(np.float32), 'labels': labels.astype(np.int64),
                'blur': blur.astype(np.float16), 'aug': aug.astype(np.float16),
                'dct': dct.astype(np.float16), 'dct_rand': rng.permutation(dct.reshape(-1)).reshape(dct.shape).astype(np.float16)}

    train = images(train_per_class)
    test = images(test_per_class)
    train['sample_id'] = np.arange(len(train['labels']), dtype=np.int64)
    test['sample_id'] = np.arange(len(test['labels']), dtype=np.int64)
    train['text'] = unit(means + 0.3 * rng.randn(num_classes, dim)).astype(np.float32)
    desc_targets = np.repeat(np.arange(num_classes), 4)
    train['desc'] = unit(means[desc_targets] + 0.5 * rng.randn(len(desc_targets), dim)).astype(np.float32)
    train['desc_targets'] = desc_targets.astype(np.int64)
    train['desc_proto'] = unit(np.stack([train['desc'][desc_targets == c].mean(0)
                                         for c in range(num_classes)])).astype(np.float32)
    sessions = make_sessions(num_classes, num_init, num_inc)
    supports = {}
    for shot in (1, 5):
        for s in (1, 2):
            draw = replica_support_indices(train['labels'], sessions, -1, shot, s)
            supports[support_key(shot, s)] = [d.tolist() for d in draw]
    meta = {'dataset': 'synthetic', 'class_names': [f'class {i}' for i in range(num_classes)],
            'sessions': [s.tolist() for s in sessions], 'supports': supports,
            'bimc_defaults': {'beta': 0.3, 'ensemble_alpha': 0.8, 'lambda_t': 0.5, 'lambda_i': 0.1,
                              'tau': 16.0, 'gamma_base': 1.0, 'gamma_inc': 5.0, 'ensemble': True,
                              'cov_temperature': 512.0},
            'blur_sigmas': list(range(1, blur_levels + 1)), 'aug_names': [f'aug{i}' for i in range(aug_levels)],
            'dct_layers': list(range(layers))}
    return train, test, meta


def write_cache(stem, **kwargs):
    train, test, meta = synthetic_arrays(**kwargs)
    meta['sha256'] = {'train': fc.save_split(stem, 'train', train), 'test': fc.save_split(stem, 'test', test)}
    fc.save_meta(stem, meta)
    return train, test, fc.load_meta(stem)
