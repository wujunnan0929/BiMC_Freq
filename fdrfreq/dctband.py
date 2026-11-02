"""Patch-grid DCT band energies and extra feature blocks for the joint GDA.

Extraction (array-module agnostic, works for numpy and torch tensors):
    tokens [B, n*n, W] (ViT patch tokens after block l, CLS removed)
    C = D T D^T over the two spatial axes (orthonormal DCT-II)
    e_b[c] = sqrt(mean_{(u,v) in band b} C[u,v,c]^2)        -> [B, 3, W]
Radial bands on the n x n grid exclude DC (= mean pooling). The 'random'
partition assigns the same numbers of non-DC coefficients to three groups at
random; it is the frequency-ordering control (F5).

Run time (``BlockBuilder``): every extra block is fitted on base-session
training rows only: power normalisation, centering, per-sample L2, PCA to k,
ridge residualisation on the original CLIP feature (only information the
CLS feature cannot linearly predict remains), then scaling so its per-dim
within-class variance matches the CLS feature's.
"""

import numpy as np

from .linalg import pca_fit, ridge_fit, within_class_centered

RADIAL_EDGES = (3.0, 7.0)


def dct_matrix(n):
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    D = np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    D[0] *= 1 / np.sqrt(2)
    return D * np.sqrt(2.0 / n)


def radial_bands(n=14, edges=RADIAL_EDGES):
    u, v = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    rho = np.sqrt(u ** 2 + v ** 2)
    lo, hi = edges
    return [((rho > 0) & (rho <= lo)), ((rho > lo) & (rho <= hi)), (rho > hi)]


def random_bands(n=14, seed=0, edges=RADIAL_EDGES):
    sizes = [int(b.sum()) for b in radial_bands(n, edges)]
    flat = np.arange(1, n * n)  # exclude DC (index 0)
    perm = np.random.RandomState(seed).permutation(flat)
    out, s = [], 0
    for k in sizes:
        m = np.zeros(n * n, dtype=bool)
        m[perm[s:s + k]] = True
        out.append(m.reshape(n, n))
        s += k
    return out


def band_energies(tokens, D, bands):
    """Reference numpy implementation. tokens [B,n*n,W] -> [B,len(bands),W].

    ``fdrfreq.extract`` uses an equivalent torch version on the GPU and checks
    it against this function on the first batch.
    """
    tokens = np.asarray(tokens, dtype=np.float64)
    B, L, W = tokens.shape
    n = D.shape[0]
    if L != n * n:
        raise ValueError('token count must be a square grid')
    T = tokens.reshape(B, n, n, W)
    C = np.einsum('uh,bhwc,vw->buvc', D, T, D)
    E = C ** 2
    out = [np.sqrt((E * m[None, :, :, None]).sum(axis=(1, 2)) / float(m.sum())) for m in bands]
    return np.stack(out, axis=1)


# ---------------------------------------------------------------- run time
class BlockBuilder:
    """Fit/transform one extra block. spec examples:
        {'type': 'dct', 'layer': 9, 'kind': 'radial', 'band': 1, 'k': 64}
        {'type': 'view', 'view': 'flip', 'k': 64}           (augmentation control)
        {'type': 'blurdiff', 'sigma': 2, 'k': 64}            (pixel-domain DoG)
        {'type': 'noise', 'k': 64, 'seed': 0}                 (dimension control)
    """

    def __init__(self, spec, eta=1.0):
        self.spec, self.eta = dict(spec), eta

    def _raw(self, bank, split, rows):
        s = self.spec
        if s['type'] == 'dct':
            e = bank.dct(split, s['layer'], s.get('kind', 'radial'), rows)[:, int(s['band'])]
            g = np.sqrt(np.maximum(e, 0))
        elif s['type'] == 'view':
            g = bank.feat(split, s['view'])[rows]
        elif s['type'] == 'blurdiff':
            g = bank.feat(split, 'orig')[rows] - bank.feat(split, f"blur{float(s['sigma']):g}")[rows]
        elif s['type'] == 'noise':
            n_split = len(bank.labels(split))
            seed = int(s.get('seed', 0)) * 2 + (0 if split == 'train' else 1)
            full = np.random.RandomState(seed).randn(n_split, int(s['k'])).astype(np.float32)
            return full[rows].astype(np.float64)
        else:
            raise ValueError('unknown block type ' + str(s['type']))
        return np.asarray(g, dtype=np.float64)

    def fit(self, bank, rows, labels, classes):
        f = bank.feat('train', 'orig')[rows].astype(np.float64)
        g = self._raw(bank, 'train', rows)
        if self.spec['type'] != 'noise':
            self.center = g.mean(0)
            g = g - self.center
            g = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
            self.mu, self.V = pca_fit(g, self.spec.get('k', 64))
            g = (g - self.mu) @ self.V.T
        self.A = ridge_fit(f, g, self.eta) if self.spec.get('residualize', True) else None
        if self.A is not None:
            g = g - f @ self.A.T
        wf, _ = within_class_centered(f, labels, classes)
        wg, _ = within_class_centered(g, labels, classes)
        per_dim_f = (wf ** 2).sum() / wf.size
        per_dim_g = (wg ** 2).sum() / max(wg.size, 1)
        self.scale = float(np.sqrt(per_dim_f / max(per_dim_g, 1e-18)))
        self.dim = g.shape[1]
        return self

    def transform(self, bank, split, rows):
        g = self._raw(bank, split, rows)
        if self.spec['type'] != 'noise':
            g = g - self.center
            g = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
            g = (g - self.mu) @ self.V.T
        if self.A is not None:
            g = g - bank.feat(split, 'orig')[rows].astype(np.float64) @ self.A.T
        return g * self.scale


def expand_blocks(blocks):
    """Allow {'type':'dct','layer':9,'bands':[0,1,2]} shorthand."""
    out = []
    for b in blocks:
        if b.get('type') == 'dct' and 'bands' in b:
            for band in b['bands']:
                bb = {k: v for k, v in b.items() if k != 'bands'}
                bb['band'] = band
                out.append(bb)
        else:
            out.append(dict(b))
    return out
