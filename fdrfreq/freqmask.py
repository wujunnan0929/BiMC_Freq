"""Visual frequency grounding: which embedding channels a class's fine detail drives.

For a Gaussian ladder sigma_0 = 0 < sigma_1 < ... < sigma_K (pixels), band k is
the feature-space difference of Gaussians
    Delta_k(x) = n(f(G_{sigma_{k-1}} x)) - n(f(G_{sigma_k} x)),   k = 1..K
(k = 1 is the finest detail). Only blurred natural images are encoded, never
high-pass images, so every view stays inside CLIP's training distribution.

A class's band-k mask is the top-q channels of |mean_x Delta_k(x) - common_k|,
where common_k is the average over base classes (the blur direction every
image shares). Novel classes compute their masks from their own supports.
"""

import numpy as np

from .keywords import topq_mask


def ladder_views(sigmas):
    return ['orig'] + [f'blur{float(s):g}' for s in sigmas]


def delta_features(bank, split, rows, sigmas):
    views = [bank.feat(split, v)[rows].astype(np.float64) for v in ladder_views(sigmas)]
    return np.stack([views[k - 1] - views[k] for k in range(1, len(views))], axis=1)


def class_delta_means(bank, split, rows, labels, classes, sigmas):
    D = delta_features(bank, split, rows, sigmas)
    return np.stack([D[labels == c].mean(0) for c in classes])  # [C,K,D]


def class_masks(delta_means, common, q):
    C, K, _ = delta_means.shape
    out = np.zeros(delta_means.shape, dtype=bool)
    for c in range(C):
        for k in range(K):
            out[c, k] = topq_mask(delta_means[c, k] - common[k], q)
    return out


def jaccard(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 1.0


def class_specificity(masks):
    """Mean pairwise Jaccard between classes, per band (low = class-specific)."""
    C, K, _ = masks.shape
    out = []
    for k in range(K):
        m = masks[:, k].astype(np.float64)
        inter = m @ m.T
        sizes = m.sum(1)
        union = sizes[:, None] + sizes[None] - inter
        iu = np.triu_indices(C, 1)
        out.append(float(np.mean(inter[iu] / np.maximum(union[iu], 1))))
    return out


def random_jaccard(q, d):
    """Expected Jaccard of two random q-subsets of d channels."""
    inter = q * q / d
    return float(inter / (2 * q - inter))


def shot_stability(bank, rows_by_class, sigmas, q, shot, common, reps=5, seed=0):
    """Per band: Jaccard(mask from `shot` random images, mask from all images)."""
    r = np.random.RandomState(seed)
    scores = []
    for rows in rows_by_class:
        D = delta_features(bank, 'train', rows, sigmas)
        full = np.stack([topq_mask(D[:, k].mean(0) - common[k], q) for k in range(D.shape[1])])
        for _ in range(reps):
            pick = r.choice(len(rows), size=min(shot, len(rows)), replace=False)
            part = np.stack([topq_mask(D[pick, k].mean(0) - common[k], q)
                             for k in range(D.shape[1])])
            scores.append([jaccard(part[k], full[k]) for k in range(D.shape[1])])
    return np.asarray(scores).mean(0).tolist()
