"""BiMC / BiMC-dagger scoring on cached features (mirrors models/bimc.py exactly).

Reference (models/bimc.py, engine/engine.py):
  fused_c = beta * ((1 - lambda_t) * text_c + lambda_t * desc_proto_c) + (1 - beta) * img_c
  p_fused = softmax(<f, unit(fused_c)>)                      (no temperature)
  p_cov   = softmax(-maha(f, img_c; pinv(cov_G)) / 512)       cov_G: class-count weighted
            average of per-session total covariances + gamma * mean(diag) * I
  p_knn   = softmax(max_j <f, desc_{c,j}>)
  BiMC-dagger: base columns  a * p_fused + (1 - a) * p_cov,
               novel columns a * p_fused + (1 - a) * p_knn
  novel img_c = unit((1 - lambda_i) * raw_c + lambda_i * unit(sum_b softmax_b(tau <raw_c, base_b>) base_b))
"""

import numpy as np

DEFAULTS = dict(beta=None, lambda_t=0.5, lambda_i=0.1, tau=16.0, gamma_base=1.0, gamma_inc=5.0,
                ensemble=True, ensemble_alpha=None, vision_calibration=True, text_calibration=True,
                cov_temperature=512.0)


def unit(x, axis=-1, eps=1e-12):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), eps)


def softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def class_means(feats, labels, classes):
    feats = np.asarray(feats, dtype=np.float64)
    out, counts = [], []
    for c in classes:
        rows = feats[labels == c]
        if len(rows) == 0:
            raise ValueError(f'No support features for class {c}')
        out.append(rows.mean(0))
        counts.append(len(rows))
    return np.stack(out), np.asarray(counts, dtype=np.int64)


def soft_calibration(base_protos, cur_protos, lam, tau):
    """BiMC/TEEN visual calibration (models/bimc.py::soft_calibration)."""
    base = unit(base_protos)
    cur = unit(cur_protos)
    w = softmax(tau * cur @ base.T, axis=1)
    delta = unit(w @ base)
    return unit((1 - lam) * cur + lam * delta)


def session_covariance(feats, gamma):
    """torch.cov(features.T) + gamma * mean(diag) * I (shrink_cov with alpha2=0)."""
    feats = np.asarray(feats, dtype=np.float64)
    if len(feats) < 2:
        cov = np.zeros((feats.shape[1], feats.shape[1]))
    else:
        cov = np.cov(feats.T)
    return cov + gamma * np.mean(np.diag(cov)) * np.eye(cov.shape[0])


def merge_covariances(covs, weights):
    weights = np.asarray(weights, dtype=np.float64)
    total = sum(w * c for w, c in zip(weights, covs))
    return total / weights.sum()


def inverse(matrix):
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix)


def mahalanobis_sq(queries, means, precision):
    q = np.asarray(queries, dtype=np.float64)
    m = np.asarray(means, dtype=np.float64)
    qp = q @ precision
    quad_q = np.einsum('nd,nd->n', qp, q)
    quad_m = np.einsum('cd,cd->c', m @ precision, m)
    return quad_q[:, None] - 2.0 * qp @ m.T + quad_m[None, :]


def knn_logits(queries, desc, desc_targets, num_classes):
    sims = np.asarray(queries, dtype=np.float64) @ np.asarray(desc, dtype=np.float64).T
    out = np.full((len(queries), num_classes), -np.inf)
    for c in range(num_classes):
        cols = desc_targets == c
        if np.any(cols):
            out[:, c] = sims[:, cols].max(1)
    return out


def text_mix(text, desc_proto, lambda_t):
    return (1 - lambda_t) * np.asarray(text, np.float64) + lambda_t * np.asarray(desc_proto, np.float64)


def bimc_votes(queries, img_protos, text, desc_proto, desc, desc_targets, precision, num_base, params):
    """Positive vote matrix [Q, C_seen] for the seen classes (all inputs restricted to seen)."""
    p = params
    q = unit(queries)
    lambda_t = p['lambda_t'] if p['text_calibration'] else 0.0
    fused = unit(p['beta'] * text_mix(text, desc_proto, lambda_t) + (1 - p['beta']) * np.asarray(img_protos))
    prob_fused = softmax(q @ fused.T)
    if not p['ensemble']:
        return prob_fused
    seen = len(img_protos)
    prob_cov = softmax(-mahalanobis_sq(q, img_protos, precision) / p['cov_temperature'])
    prob_knn = softmax(knn_logits(q, desc, desc_targets, seen))
    a = p['ensemble_alpha']
    votes = np.empty_like(prob_fused)
    votes[:, :num_base] = a * prob_fused[:, :num_base] + (1 - a) * prob_cov[:, :num_base]
    votes[:, num_base:] = a * prob_fused[:, num_base:] + (1 - a) * prob_knn[:, num_base:]
    return votes


def bimc_votes_cached(queries, img_protos, text, desc_proto, knn_prob, qp, qpq, precision, num_base, params):
    """Same as :func:`bimc_votes` but reuses config-independent terms.

    knn_prob = softmax(knn logits); qp = q @ P; qpq = diag(q P q^T). These depend only
    on the session (support covariance, descriptions), never on the calibration.
    """
    p = params
    q = unit(queries)
    lambda_t = p['lambda_t'] if p['text_calibration'] else 0.0
    fused = unit(p['beta'] * text_mix(text, desc_proto, lambda_t) + (1 - p['beta']) * np.asarray(img_protos))
    prob_fused = softmax(q @ fused.T)
    if not p['ensemble']:
        return prob_fused
    m = np.asarray(img_protos, dtype=np.float64)
    d2 = qpq[:, None] - 2.0 * qp @ m.T + np.einsum('cd,cd->c', m @ precision, m)[None, :]
    prob_cov = softmax(-d2 / p['cov_temperature'])
    a = p['ensemble_alpha']
    votes = np.empty_like(prob_fused)
    votes[:, :num_base] = a * prob_fused[:, :num_base] + (1 - a) * prob_cov[:, :num_base]
    votes[:, num_base:] = a * prob_fused[:, num_base:] + (1 - a) * knn_prob[:, num_base:]
    return votes


def resolve_params(overrides, meta_defaults):
    params = dict(DEFAULTS)
    params.update({k: v for k, v in (meta_defaults or {}).items() if k in DEFAULTS})
    params.update({k: v for k, v in (overrides or {}).items() if v is not None})
    if params['beta'] is None or params['ensemble_alpha'] is None:
        raise ValueError('beta and ensemble_alpha must come from the cache metadata or the spec')
    return params
