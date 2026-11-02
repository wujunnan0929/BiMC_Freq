"""Shared-covariance Gaussian discriminant (LDA) for frozen CLIP features.

The anchor of the paper. Means come from each class's support rows, the
pooled within-class covariance only from the base session (frozen afterwards),
so incremental sessions add classes in O(d) without touching old statistics.

Score of class c for query x (precision P, raw pooled within W):
    s_c(x) = mu_c^T P x - 1/2 mu_c^T P mu_c + kappa * tr(P W) / (2 n_c)
The last term removes, in expectation, the plug-in penalty a class with n_c
support rows pays because its estimated mean is noisy:
    E||x - mu_hat||_P^2 = ||x - mu||_P^2 + tr(P W)/n_c.
kappa = 0 recovers ordinary plug-in LDA. ``mode='predictive'`` instead uses the
flat-prior Bayesian posterior predictive N(mu_hat, (1 + kappa/n_c) Sigma).
"""

import numpy as np

from .linalg import pooled_within, safe_inverse, block_diagonal, softmax


class GDA:
    def __init__(self, shrink='lw', ridge=0.0, block_sizes=None, block_diag=False):
        self.shrink, self.ridge = shrink, ridge
        self.block_sizes, self.block_diag = block_sizes, block_diag
        self.P = None

    def fit_base(self, features, labels, classes):
        Ws, W, delta = pooled_within(features, labels, classes, shrink=self.shrink)
        if self.block_diag and self.block_sizes:
            Ws = block_diagonal(Ws, self.block_sizes)
        self.P = safe_inverse(Ws, ridge=self.ridge)
        self.trPW = float(np.trace(self.P @ W))
        self.delta = delta
        self.dim = features.shape[1]
        self.means, self.counts = [], []
        return self

    def add_classes(self, means, counts):
        self.means.append(np.asarray(means, dtype=np.float64))
        self.counts.append(np.asarray(counts, dtype=np.float64))

    def scores(self, X, kappa=0.0, mode='debias', normalize_by_dim=True):
        """mode='debias':     linear LDA score + kappa * tr(PW) / (2 n_c)
        mode='predictive': Gaussian posterior-predictive log density with the
            mean's sampling covariance inflated by kappa / n_c (kappa = 1 is the
            flat-prior Bayesian predictive N(mu_hat, (1 + 1/n_c) Sigma)):
            -1/2 ||x - mu_c||_P^2 / (1 + kappa/n_c) - d/2 log(1 + kappa/n_c)
        Both reduce to plain LDA for kappa = 0 (up to a per-query constant).
        """
        M = np.concatenate(self.means)
        n = np.concatenate(self.counts)
        X = X.astype(np.float64)
        PM = M @ self.P
        lin = X @ PM.T - 0.5 * (PM * M).sum(1)[None]
        if mode == 'debias' or not kappa:
            s = lin + (kappa * self.trPW / (2.0 * n)[None] if kappa else 0.0)
        elif mode == 'predictive':
            xpx = ((X @ self.P) * X).sum(1)[:, None]
            maha = xpx - 2.0 * lin  # = xPx - 2 mu'Px + mu'Pmu, since lin = mu'Px - mu'Pmu/2
            infl = 1.0 + kappa / n
            s = -0.5 * maha / infl[None] - 0.5 * self.dim * np.log(infl)[None]
        else:
            raise ValueError('mode must be debias or predictive')
        if normalize_by_dim:
            s = s / self.dim
        return s

    def probs(self, X, temperature=1.0, kappa=0.0, mode='debias'):
        return softmax(self.scores(X, kappa=kappa, mode=mode) / float(temperature))
