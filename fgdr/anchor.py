"""Shot-aware pooled within-class discriminant (the statistical anchor).

Covariance: pooled within-class W over base-session features only (frozen),
precision P = (W + ridge * mean(diag W) * I)^-1  (same regulariser as
models/frequency_discriminant.precision_from_covariance).

Scores for class c with n_c support samples (mean mu_c):
  plugin      s_c = -1/2 (x - mu_c)^T P (x - mu_c)
  predictive  s_c = -1/2 (x - mu_c)^T P (x - mu_c) / (1 + kappa / n_c) - d/2 log(1 + kappa / n_c)
The predictive form is the Gaussian posterior predictive with a flat prior on
the class mean (covariance inflated by 1/n_c); kappa = 0 recovers plugin.

Mixing with BiMC votes (as in models/frequency_uncertainty.mix_uncertainty_probabilities):
  votes' = (1 - alpha) votes + alpha * mass(votes) * softmax(s / T)

Blocks (stage 1-2 information test): z = [f ; g~_1 ; ... ] where each extra block
is centred, L2-normalised, PCA-reduced on base data and ridge-residualised on f,
so a block can only contribute what f cannot linearly predict.
"""

import numpy as np

from fgdr.bimc_ref import softmax, unit


def pooled_within_covariance(feats, labels, classes):
    feats = np.asarray(feats, dtype=np.float64)
    residuals, dof = [], 0
    for c in classes:
        rows = feats[labels == c]
        if len(rows) == 0:
            raise ValueError(f'class {c} has no samples')
        residuals.append(rows - rows.mean(0))
        dof += len(rows) - 1
    if dof <= 0:
        raise ValueError('pooled covariance needs positive degrees of freedom')
    r = np.concatenate(residuals)
    w = r.T @ r / dof
    return 0.5 * (w + w.T)


def ridge_precision(cov, ridge, floor=1e-6):
    scale = max(float(np.mean(np.diag(cov))), floor)
    reg = cov + ridge * scale * np.eye(cov.shape[0])
    chol = np.linalg.cholesky(reg)
    inv_chol = np.linalg.inv(chol)
    prec = inv_chol.T @ inv_chol
    return 0.5 * (prec + prec.T)


def mix_votes(votes, logits, alpha, temperature):
    if alpha == 0:
        return votes
    mass = votes.sum(1, keepdims=True)
    return (1 - alpha) * votes + alpha * mass * softmax(logits / temperature, axis=1)


class BlockTransform:
    """Builds z = [f ; residualised extra blocks]; fitted on base-session data only."""

    def __init__(self, blocks, seed=0):
        self.blocks = list(blocks or [])
        self.seed = seed
        self.fitted = []
        self._noise = {}

    @staticmethod
    def _raw(bank, block):
        kind = block['kind']
        if kind in ('dct', 'dct_rand'):
            x = bank.views[kind][:, block['layer'], block['band'], :]
            return np.sqrt(np.maximum(np.asarray(x, np.float64), 0.0))
        if kind in ('blur', 'aug'):
            return unit(bank.views[kind][:, block['index'], :])
        if kind == 'noise':
            return None
        raise ValueError('unknown block kind ' + kind)

    def _noise_for(self, bank, dim, index):
        key = (bank.split, index)
        need = int(bank.rows.max()) + 1 if len(bank.rows) else 0
        table = self._noise.get(key)
        if table is None or len(table) < need:
            rng = np.random.RandomState(self.seed + 104729 * (index + 1) + (0 if bank.split == 'train' else 1))
            table = rng.randn(max(need, 1), dim)
            self._noise[key] = table
        return table[bank.rows]

    @staticmethod
    def _within_var(x, labels):
        """Average per-dimension pooled within-class variance."""
        total, dof = 0.0, 0
        for c in np.unique(labels):
            rows = x[labels == c]
            total += np.sum((rows - rows.mean(0)) ** 2)
            dof += len(rows) - 1
        return total / max(dof, 1) / x.shape[1]

    def fit(self, base_bank, ridge=1.0):
        """Fit PCA / residualisation on base data; scale every block so its average
        within-class variance per dimension equals that of f (uniform ridge)."""
        f = np.asarray(base_bank.feats, np.float64)
        target = self._within_var(f, base_bank.labels)
        self.fitted = []
        for i, block in enumerate(self.blocks):
            dim = int(block.get('dim', 64))
            raw = self._raw(base_bank, block)
            if raw is None:
                self.fitted.append({'kind': 'noise', 'dim': dim, 'index': i, 'scale': np.sqrt(target)})
                continue
            mean = raw.mean(0)
            x = unit(raw - mean)
            center = x.mean(0)
            xc = x - center
            _, _, vt = np.linalg.svd(xc, full_matrices=False)
            basis = vt[:dim].T
            g = xc @ basis
            gram = f.T @ f + ridge * np.eye(f.shape[1])
            coef = np.linalg.solve(gram, f.T @ g)
            resid = g - f @ coef
            scale = np.sqrt(target / max(self._within_var(resid, base_bank.labels), 1e-12))
            self.fitted.append({'kind': block['kind'], 'block': block, 'mean': mean, 'center': center,
                                'basis': basis, 'coef': coef, 'scale': scale, 'index': i})
        return self

    def transform(self, bank):
        f = np.asarray(bank.feats, np.float64)
        parts = [f]
        for item in self.fitted:
            if item['kind'] == 'noise':
                parts.append(item['scale'] * self._noise_for(bank, item['dim'], item['index']))
                continue
            raw = self._raw(bank, item['block'])
            x = unit(raw - item['mean'])
            g = (x - item['center']) @ item['basis']
            parts.append(item['scale'] * (g - f @ item['coef']))
        return np.concatenate(parts, axis=1)


class ShotAwareGDA:
    def __init__(self, ridge=0.1, kappa=0.0, mode='plugin'):
        if mode not in ('plugin', 'predictive'):
            raise ValueError('mode must be plugin or predictive')
        self.ridge, self.kappa, self.mode = float(ridge), float(kappa), mode
        self.precision = None

    def fit(self, z, labels, classes):
        self.precision = ridge_precision(pooled_within_covariance(z, labels, classes), self.ridge)
        return self

    def logits(self, zq, means, counts):
        d2 = np_mahalanobis(zq, means, self.precision)
        if self.mode == 'plugin' or self.kappa == 0:
            return -0.5 * d2
        inflate = 1.0 + self.kappa / np.asarray(counts, dtype=np.float64)
        dim = self.precision.shape[0]
        return -0.5 * d2 / inflate[None] - 0.5 * dim * np.log(inflate)[None]


def np_mahalanobis(q, m, precision):
    q = np.asarray(q, np.float64)
    m = np.asarray(m, np.float64)
    qp = q @ precision
    return (np.einsum('nd,nd->n', qp, q)[:, None] - 2 * qp @ m.T
            + np.einsum('cd,cd->c', m @ precision, m)[None])
