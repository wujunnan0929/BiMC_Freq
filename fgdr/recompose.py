"""Closed-form novel-prototype recomposition.

* ``holistic``        TEEN/BiMC calibration with a selectable similarity source.
                      visual similarity + (lambda_i, tau) reproduces BiMC exactly;
                      text similarity is the undocumented first step of FDR's code.
* ``segment_update``  FDR's segment-wise recomposition: for every channel segment g,
                      p[g] <- (1 - a) p[g] + a * sum_j w_j base_j[g], where w comes from
                      a similarity restricted to g (text or visual), optionally top-r.
"""

import numpy as np

from fgdr.bimc_ref import softmax, unit


def cosine(a, b):
    return unit(a) @ unit(b).T


def donor_weights(sim, tau, topr=None, fdr_softmax=False):
    """Row-wise donor weights over base classes.

    ``fdr_softmax`` mirrors FDR's code: non-top-r similarities are set to zero
    *before* the softmax, so they keep weight exp(0). Otherwise the softmax is
    taken over the top-r entries only.
    """
    sim = np.atleast_2d(np.asarray(sim, dtype=np.float64))
    n, b = sim.shape
    if topr is None or topr >= b:
        return softmax(tau * sim, axis=1)
    topr = int(topr)
    idx = np.argpartition(-sim, topr - 1, axis=1)[:, :topr]
    rows = np.arange(n)[:, None]
    if fdr_softmax:
        filtered = np.zeros_like(sim)
        filtered[rows, idx] = sim[rows, idx]
        return softmax(tau * filtered, axis=1)
    weights = np.zeros_like(sim)
    weights[rows, idx] = softmax(tau * sim[rows, idx], axis=1)
    return weights


def holistic(cur_protos, base_protos, sim, lam, tau, topr=None, fdr_softmax=False):
    if lam == 0:
        return unit(cur_protos)
    w = donor_weights(sim, tau, topr, fdr_softmax)
    delta = unit(w @ unit(base_protos))
    return unit((1 - lam) * unit(cur_protos) + lam * delta)


def segment_update(proto, base_protos, segments, cur_vec, base_vecs, tau, topr=None,
                   fdr_softmax=False):
    """Iterative FDR update of one prototype.

    segments: iterable of (bool mask [D], alpha). ``cur_vec``/``base_vecs`` are the
    vectors whose masked cosine chooses donors (text or visual).
    Returns the updated (unnormalised) prototype and the number of applied segments.
    """
    p = np.array(proto, dtype=np.float64, copy=True)
    base = unit(base_protos)
    base_vecs = np.asarray(base_vecs, dtype=np.float64)
    cur_vec = np.asarray(cur_vec, dtype=np.float64)
    applied = 0
    for mask, alpha in segments:
        mask = np.asarray(mask, dtype=bool)
        if alpha == 0 or not mask.any():
            continue
        sub_base = base_vecs[:, mask]
        sub_cur = cur_vec[mask]
        denom = np.linalg.norm(sub_base, axis=1) * max(np.linalg.norm(sub_cur), 1e-12)
        sim = (sub_base @ sub_cur) / np.maximum(denom, 1e-12)
        w = donor_weights(sim[None], tau, topr, fdr_softmax)[0]
        p[mask] = (1 - alpha) * p[mask] + alpha * (w @ base[:, mask])
        applied += 1
    return p, applied


def frequency_alpha_scales(signature, proto, masks, eps=1e-12):
    """Q4: relative fine-detail energy of each segment, normalised to mean one.

    rho_i = ||sig[g_i]|| / ||proto[g_i]||; scale_i = clip(rho_i / mean(rho), 0, 2).
    """
    rho = []
    for mask in masks:
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            rho.append(0.0)
            continue
        rho.append(np.linalg.norm(signature[mask]) / max(np.linalg.norm(proto[mask]), eps))
    rho = np.asarray(rho)
    mean = rho[rho > 0].mean() if np.any(rho > 0) else 1.0
    return np.clip(rho / max(mean, eps), 0.0, 2.0)
