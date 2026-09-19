"""Training-free linear discriminants over jointly whitened frequency views.

The covariance contains only pooled within-class variation. A single shared
precision accounts for feature and, in ``full`` mode, cross-view correlations.
Classes have equal priors: neither their support count nor a class-specific
covariance changes their score. No network or classifier is gradient-trained.
"""

from collections.abc import Mapping

import torch

from models.frequency_uncertainty import _features, _integer_vector, _scalar, _tensor


def _square_matrix(value, name, device=None):
    result = _tensor(value, name, device).float()
    if (result.ndim != 2 or result.shape[0] == 0
            or result.shape[0] != result.shape[1]):
        raise ValueError("{} must have nonempty square shape [P,P]".format(name))
    if not torch.isfinite(result).all():
        raise ValueError("{} exceeds float32 range".format(name))
    scale = result.abs().max().item()
    if not torch.allclose(result, result.T, rtol=1e-5, atol=scale * 1e-7):
        raise ValueError("{} must be symmetric".format(name))
    return result


def _means(stats):
    if not isinstance(stats, Mapping) or "mean" not in stats:
        raise ValueError("Statistics require mean")
    result = _tensor(stats["mean"], "mean").float()
    if result.ndim != 3 or any(size == 0 for size in result.shape):
        raise ValueError("Mean must have nonempty shape [C,B,D]")
    if not torch.isfinite(result).all():
        raise ValueError("Mean exceeds float32 range")
    # Retain the length of the sample mean, as required by the Gaussian model.
    return result


@torch.no_grad()
def pooled_full_covariance(features, labels, class_ids, structure="full"):
    """Pool normalized within-class residuals with denominator sum(n_c - 1).

    Flattening is band-major: ``[B,D] -> [B * D]``. ``block`` preserves each
    view's full feature covariance but sets all cross-view covariance to zero.
    Labels outside ``class_ids`` are excluded, including from the denominator.
    One-shot classes contribute zero residuals and no degrees of freedom.
    """
    if structure not in ("full", "block"):
        raise ValueError("structure must be full or block")
    values = _features(features)
    targets = _integer_vector(labels, "labels", values.device)
    ids = _integer_vector(class_ids, "class_ids", values.device)
    if len(targets) != len(values) or ids.unique().numel() != ids.numel():
        raise ValueError("Labels must match samples and class_ids must be unique")
    residuals, degrees = [], 0
    for class_id in ids:
        selected = values[targets == class_id]
        if len(selected) == 0:
            raise ValueError("Every requested class must have support samples")
        degrees += len(selected) - 1
        residuals.append(selected - selected.mean(dim=0))
    if degrees <= 0:
        raise ValueError("Pooled covariance requires positive within-class degrees of freedom")
    residuals = torch.cat(residuals, dim=0)
    bands, dimensions = values.shape[1:]
    if structure == "full":
        flat = residuals.flatten(1)
        covariance = flat.T @ flat / degrees
    else:
        covariance = values.new_zeros((bands * dimensions, bands * dimensions))
        for band in range(bands):
            current = residuals[:, band]
            start, stop = band * dimensions, (band + 1) * dimensions
            covariance[start:stop, start:stop] = current.T @ current / degrees
    # Symmetrizing removes harmless GEMM roundoff without changing the model.
    covariance = .5 * covariance + .5 * covariance.T
    if not torch.isfinite(covariance).all():
        raise ValueError("Pooled covariance exceeds finite float32 range")
    return covariance


@torch.no_grad()
def precision_from_covariance(covariance, ridge=0.1, var_floor=1e-6):
    """Invert ``cov + ridge * max(mean(diag(cov)), var_floor) * I``.

    Cholesky and inversion use float64 to handle singular empirical estimates
    with a positive ridge. The returned precision is float32 and is checked
    again for positive definiteness after conversion. No hidden eigenvalue
    clipping or adaptive jitter changes the requested regularizer. With zero
    ridge, a singular covariance is therefore rejected explicitly.
    """
    ridge = _scalar(ridge, "ridge")
    var_floor = _scalar(var_floor, "var_floor", strict=True)
    matrix = _square_matrix(covariance, "covariance")
    if (matrix.diagonal() < 0).any():
        raise ValueError("Covariance diagonal must be nonnegative")
    matrix = .5 * matrix.double() + .5 * matrix.T.double()
    scale = max(matrix.diagonal().mean().item(), var_floor)
    regularizer = ridge * scale
    if not torch.isfinite(torch.tensor(regularizer, dtype=torch.float64)):
        raise ValueError("Covariance regularizer must be finite")
    regularized = matrix + regularizer * torch.eye(
        matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    if not torch.isfinite(regularized).all():
        raise ValueError("Regularized covariance must be finite")
    factor, info = torch.linalg.cholesky_ex(regularized)
    if info.item() != 0:
        raise ValueError("Regularized covariance must be positive definite; use a positive ridge")
    precision = torch.cholesky_inverse(factor)
    precision = (.5 * precision + .5 * precision.T).float()
    if not torch.isfinite(precision).all():
        raise ValueError("Precision exceeds finite float32 range")
    _, info = torch.linalg.cholesky_ex(precision.double())
    if info.item() != 0:
        raise ValueError("Precision loses positive definiteness in float32; increase ridge")
    return precision


@torch.no_grad()
def prepare_discriminant(stats, precision, *, validate_precision=True):
    """Cache class weights for repeated query batches with fixed statistics.

    Returns a serializable dictionary with ``weights[P,C]``, ``bias[C]``,
    ``num_bands`` and ``feature_dim``. Rebuild it whenever means or precision
    change (for example, after appending incremental classes). Mean vectors
    are not normalized again. Count and class-specific variance are unused.
    Set validate_precision=False only for a matrix already checked by
    precision_from_covariance, to avoid repeating an O(P^3) Cholesky per episode.
    """
    mean = _means(stats)
    matrix = _square_matrix(precision, "precision", mean.device)
    classes, bands, dimensions = mean.shape
    total = bands * dimensions
    if matrix.shape != (total, total):
        raise ValueError("Precision and mean band/feature dimensions must agree")
    if not isinstance(validate_precision, bool):
        raise ValueError("validate_precision must be boolean")
    if validate_precision:
        _, info = torch.linalg.cholesky_ex(matrix.double())
        if info.item() != 0:
            raise ValueError("Precision must be positive definite")
    flat = mean.flatten(1)
    weights = (matrix @ flat.T) / total
    bias = -.5 * (flat * weights.T).sum(dim=-1)
    if not torch.isfinite(weights).all() or not torch.isfinite(bias).all():
        raise ValueError("Discriminant parameters exceed finite float32 range")
    return {"weights": weights, "bias": bias,
            "num_bands": bands, "feature_dim": dimensions}


@torch.no_grad()
def prepared_discriminant_logits(features, prepared):
    """Score queries using a ``prepare_discriminant`` result in O(N * P * C)."""
    if not isinstance(prepared, Mapping) or not all(
            key in prepared for key in ("weights", "bias", "num_bands", "feature_dim")):
        raise ValueError("Prepared discriminant requires weights, bias, and feature dimensions")
    bands, dimensions = prepared["num_bands"], prepared["feature_dim"]
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
           for value in (bands, dimensions)):
        raise ValueError("Prepared feature dimensions must be positive integers")
    weights = _tensor(prepared["weights"], "weights").float()
    bias = _tensor(prepared["bias"], "bias", weights.device).float()
    if (weights.ndim != 2 or weights.shape[0] != bands * dimensions
            or weights.shape[1] == 0 or bias.shape != (weights.shape[1],)):
        raise ValueError("Prepared weights and bias dimensions must agree")
    if not torch.isfinite(weights).all() or not torch.isfinite(bias).all():
        raise ValueError("Discriminant parameters exceed finite float32 range")
    query = _features(features, device=weights.device)
    if query.shape[1:] != (bands, dimensions):
        raise ValueError("Query and prepared band/feature dimensions must agree")
    result = query.flatten(1) @ weights + bias
    if not torch.isfinite(result).all():
        raise ValueError("Discriminant scores exceed finite float32 range")
    return result


@torch.no_grad()
def discriminant_logits(features, stats, precision, *, validate_precision=True):
    """Return ``(x @ precision @ mean.T - .5 * mean @ precision @ mean) / P``.

    This equals shared-covariance Gaussian log likelihood up to terms common
    to all classes. Equal class priors avoid biases from different support
    counts. The class-independent query quadratic and log determinant are
    omitted. Use the prepared API to reuse class weights across batches.
    """
    return prepared_discriminant_logits(features, prepare_discriminant(
        stats, precision, validate_precision=validate_precision,
    ))
