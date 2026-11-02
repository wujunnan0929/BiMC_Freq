"""Small numerical helpers (numpy only)."""

import numpy as np


def softmax(z, axis=-1):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(z, axis=-1):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=axis, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))


def l2n(x, axis=-1, eps=1e-12):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), eps)


def class_means(features, labels, classes):
    return np.stack([features[labels == c].mean(0) for c in classes])


def within_class_centered(features, labels, classes):
    """Rows of features minus their class mean (only rows of ``classes``)."""
    out, counts = [], []
    for c in classes:
        x = features[labels == c]
        if len(x) == 0:
            raise ValueError(f'class {c} has no rows')
        out.append(x - x.mean(0, keepdims=True))
        counts.append(len(x))
    return np.concatenate(out).astype(np.float64), np.asarray(counts)


def ledoit_wolf_shrinkage(Xc):
    """Ledoit-Wolf shrinkage for already-centered rows (as in scikit-learn)."""
    n, p = Xc.shape
    X2 = Xc ** 2
    emp_trace = X2.sum() / n
    mu = emp_trace / p
    beta_ = float((X2.T @ X2).sum())
    S = Xc.T @ Xc
    delta_ = float((S ** 2).sum()) / n ** 2
    beta = (beta_ / n - delta_) / (p * n)
    delta = (delta_ - 2.0 * mu * emp_trace + p * mu ** 2) / p
    beta = min(beta, delta)
    return 0.0 if beta == 0 else float(beta / delta)


def pooled_within(features, labels, classes, shrink='lw', dof='n_minus_c'):
    """Pooled within-class covariance W, shrunk toward (tr W / d) I.

    Returns (W_shrunk, W_raw, delta). ``shrink`` is 'lw' (analytic) or a float.
    """
    Xc, counts = within_class_centered(features, labels, classes)
    n, d = Xc.shape
    denom = (n - len(classes)) if dof == 'n_minus_c' else n
    W = (Xc.T @ Xc) / max(denom, 1)
    delta = ledoit_wolf_shrinkage(Xc) if shrink == 'lw' else float(shrink)
    target = np.trace(W) / d
    Ws = (1.0 - delta) * W + delta * target * np.eye(d)
    return Ws, W, delta


def block_diagonal(W, sizes):
    out = np.zeros_like(W)
    s = 0
    for k in sizes:
        out[s:s + k, s:s + k] = W[s:s + k, s:s + k]
        s += k
    return out


def safe_inverse(M, ridge=0.0):
    d = M.shape[0]
    if ridge:
        M = M + ridge * np.trace(M) / d * np.eye(d)
    try:
        return np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(M)


def ridge_fit(X, Y, eta):
    """argmin_A ||Y - X A^T||^2 + eta * n * ||A||^2  (rows are samples)."""
    n, d = X.shape
    G = X.T @ X + eta * n * np.eye(d)
    return np.linalg.solve(G, X.T @ Y).T


def pca_fit(X, k):
    mu = X.mean(0)
    Xc = X - mu
    k = int(min(k, Xc.shape[1], max(Xc.shape[0] - 1, 1)))
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return mu, Vt[:k]
