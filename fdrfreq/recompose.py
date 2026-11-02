"""Training-free recomposition of novel visual prototypes.

Two steps, both closed form and both exactly off when their strength is 0:

1. Holistic shrinkage toward similar base prototypes (TEEN / BiMC eq. 3):
       w = softmax_b(tau * cos(src_n, src_b)),
       p_n <- n((1 - lam) p_n + lam * n(sum_b w_b p_b))
   donor='visual' is BiMC/TEEN (src = visual prototypes); donor='text' is the
   step FDR's code performs before its keyword step (src = calibrated text).

2. Segment-wise recomposition (FDR eq. 8-9, iterated over segments):
       for segment mask g:  s_b = cos(src_n[g], src_b[g]) over base b,
       w = softmax over the top-r s_b (temperature tau),
       p_n[g] <- (1 - alpha_g) p_n[g] + alpha_g * sum_b w_b p_b[g]
   Segments come from keyword grounding (text side) or from the visual
   frequency grounding of ``freqmask`` (image side).
"""

import numpy as np

from .linalg import l2n, softmax


def holistic(cur, base_vis, src_cur, src_base, lam, tau):
    if lam == 0 or len(cur) == 0:
        return l2n(cur)
    w = softmax(float(tau) * (l2n(src_cur) @ l2n(src_base).T), axis=1)
    delta = l2n(w @ l2n(base_vis))
    return l2n((1.0 - lam) * l2n(cur) + lam * delta)


def segment_step(p, segments, src_n, src_base, base_vis, r, alphas, tau, scope='segment'):
    """p: [D] unit prototype of one novel class; segments: list of bool [D].

    scope='segment' (FDR) ranks donors by similarity inside the segment;
    scope='global' ranks them once on the whole vector (ablation).
    """
    p = np.asarray(p, dtype=np.float64).copy()
    base_vis = l2n(base_vis)
    r = int(min(r, len(base_vis)))
    if scope not in ('segment', 'global'):
        raise ValueError('scope must be segment or global')
    full = (src_base @ src_n) / np.maximum(np.linalg.norm(src_base, axis=1) * np.linalg.norm(src_n), 1e-12)
    for g, a in zip(segments, alphas):
        if a == 0:
            continue
        idx = np.flatnonzero(g)
        if len(idx) == 0:
            continue
        if scope == 'global':
            sims = full
        else:
            a_vec = src_n[idx]
            B = src_base[:, idx]
            sims = (B @ a_vec) / np.maximum(np.linalg.norm(B, axis=1) * np.linalg.norm(a_vec), 1e-12)
        top = np.argpartition(-sims, r - 1)[:r]
        w = softmax(float(tau) * sims[top])
        delta = w @ base_vis[top][:, idx]
        p[idx] = (1.0 - a) * p[idx] + a * delta
    return l2n(p)


def frequency_alpha(alpha, segments, delta_fine, proto, mode, seed=0):
    """Q4: per-segment strength from how much fine detail drives the segment.

    rho_i = ||delta_fine[g_i]|| / ||proto[g_i]||, alpha_i = alpha * clip(rho_i / mean rho, 0, 2).
    mode 'freq_shuffled' permutes the same alpha_i across segments (control).
    """
    if not segments:
        return []
    rho = np.array([np.linalg.norm(delta_fine[g]) / max(np.linalg.norm(proto[g]), 1e-12)
                    if g.any() else 0.0 for g in segments])
    mean = rho[rho > 0].mean() if (rho > 0).any() else 1.0
    a = alpha * np.clip(rho / mean, 0.0, 2.0)
    if mode == 'freq_shuffled':
        a = np.random.RandomState(int(seed) % (2 ** 31 - 1)).permutation(a)
    return a.tolist()
