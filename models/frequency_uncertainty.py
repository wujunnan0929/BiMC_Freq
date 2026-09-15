"""Diagonal frequency likelihoods with finite-support variance correction.

These are closed-form statistics, not gradient-trained parameters. The factor
``1 + 1 / n`` is a plug-in predictive-variance approximation: it accounts for
estimating a class mean, but does not integrate covariance uncertainty and is
not an exact Bayesian posterior predictive distribution.
"""

import math
from collections.abc import Mapping

import torch
from torch.nn import functional as F


def _scalar(value, name, minimum=0.0, strict=False):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("{} must be a finite number".format(name)) from error
    if not math.isfinite(value) or (value <= minimum if strict else value < minimum):
        raise ValueError("Invalid {}".format(name))
    return value


def _tensor(value, name, device=None):
    try:
        result = torch.as_tensor(value, device=device).detach()
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("Invalid {} tensor".format(name)) from error
    if result.is_complex() or not torch.isfinite(result).all():
        raise ValueError("{} must contain finite real values".format(name))
    return result


def _integer_vector(value, name, device=None):
    result = _tensor(value, name, device)
    if result.ndim != 1 or not result.numel() or result.dtype == torch.bool:
        raise ValueError("{} must be a nonempty integer vector".format(name))
    if result.is_floating_point() and not torch.equal(result, result.round()):
        raise ValueError("{} must contain integers".format(name))
    return result.long()


def _features(value, name="features", device=None):
    result = _tensor(value, name, device).float()
    if result.ndim != 3 or any(size == 0 for size in result.shape):
        raise ValueError("{} must have nonempty shape [N,B,D]".format(name))
    if not torch.isfinite(result).all():
        raise ValueError("{} exceeds float32 range".format(name))
    # Normalize after conversion: half precision cannot represent the default
    # normalization epsilon and can underflow small vectors.
    return F.normalize(result, dim=-1)


def _statistics(stats):
    if not isinstance(stats, Mapping) or not all(
            key in stats for key in ("mean", "variance", "count")):
        raise ValueError("Statistics require mean, variance, and count")
    mean = _tensor(stats["mean"], "mean").float()
    variance = _tensor(stats["variance"], "variance", mean.device).float()
    count = _integer_vector(stats["count"], "count", mean.device)
    if mean.ndim != 3 or any(size == 0 for size in mean.shape):
        raise ValueError("Mean must have nonempty shape [C,B,D]")
    if variance.shape != mean.shape or count.shape != (mean.shape[0],):
        raise ValueError("Statistics class/band/feature shapes must agree")
    if (count < 1).any() or (variance < 0).any():
        raise ValueError("Counts must be positive and variances nonnegative")
    if not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
        raise ValueError("Statistics exceed float32 range")
    return mean, variance, count


def _prior(value, shape, device, var_floor):
    value = _tensor(value, "prior", device).float()
    if value.shape != shape or (value < 0).any() or not torch.isfinite(value).all():
        raise ValueError("Prior must be finite nonnegative [B,D]")
    return value.clamp_min(var_floor)


@torch.no_grad()
def class_statistics(features, labels, class_ids):
    """Return raw means, unbiased variances, and counts in ``class_ids`` order.

    Individual features are normalized in float32. Class means deliberately
    retain their length; re-normalizing a mean would change the Gaussian model.
    One-shot classes store zero empirical variance and use a prior at scoring.
    Labels outside ``class_ids`` are ignored, permitting an explicit subset.
    """
    values = _features(features)
    targets = _integer_vector(labels, "labels", values.device)
    ids = _integer_vector(class_ids, "class_ids", values.device)
    if len(targets) != len(values) or ids.unique().numel() != ids.numel():
        raise ValueError("Labels must match samples and class_ids must be unique")
    means, variances, counts = [], [], []
    for class_id in ids:
        selected = values[targets == class_id]
        count = len(selected)
        if count == 0:
            raise ValueError("Every requested class must have support samples")
        mean = selected.mean(dim=0)
        variance = ((selected - mean).square().sum(dim=0) / (count - 1)
                    if count > 1 else torch.zeros_like(mean))
        means.append(mean)
        variances.append(variance)
        counts.append(count)
    return {"mean": torch.stack(means), "variance": torch.stack(variances),
            "count": torch.tensor(counts, device=values.device, dtype=torch.long)}


@torch.no_grad()
def pooled_variance(stats, var_floor=1e-6):
    """Pool only within-class sums of squares, weighted by ``n_c - 1``."""
    var_floor = _scalar(var_floor, "var_floor", strict=True)
    _, variance, count = _statistics(stats)
    degrees = (count - 1).float()
    if degrees.sum() <= 0:
        raise ValueError("Pooled variance requires positive within-class degrees of freedom")
    weights = degrees / degrees.sum()
    return (variance * weights[:, None, None]).sum(dim=0).clamp_min(var_floor)


@torch.no_grad()
def predictive_variance(stats, prior, prior_strength=20.0, var_floor=1e-6,
                        covariance="shrinkage", mean_uncertainty=True):
    """Shrink diagonal variances and optionally inflate by ``1 + 1/n``.

    ``rho = prior_strength / (prior_strength + n - 1)``. One-shot classes
    always use the prior, including when prior_strength is zero. ``shared``
    uses the pooled prior for every class before optional mean inflation.
    """
    prior_strength = _scalar(prior_strength, "prior_strength")
    var_floor = _scalar(var_floor, "var_floor", strict=True)
    if covariance not in ("shrinkage", "shared"):
        raise ValueError("covariance must be shrinkage or shared")
    if not isinstance(mean_uncertainty, bool):
        raise ValueError("mean_uncertainty must be a boolean")
    mean, variance, count = _statistics(stats)
    base = _prior(prior, mean.shape[1:], mean.device, var_floor)
    if covariance == "shared":
        result = base.unsqueeze(0).expand_as(variance)
    else:
        degrees = (count - 1).float()
        # This equivalent form also handles very large finite prior strengths.
        rho = (1.0 / (1.0 + degrees / prior_strength) if prior_strength > 0
               else (degrees == 0).float())
        result = (rho[:, None, None] * base.unsqueeze(0)
                  + (1 - rho[:, None, None]) * variance)
    result = result.clamp_min(var_floor)
    if mean_uncertainty:
        result = result * (1 + count.float().reciprocal())[:, None, None]
    if not torch.isfinite(result).all() or (result <= 0).any():
        raise ValueError("Predictive variances must be positive finite float32 values")
    return result


@torch.no_grad()
def gaussian_band_logits(features, stats, prior, prior_strength=20.0,
                         var_floor=1e-6, covariance="shrinkage",
                         mean_uncertainty=True):
    """Return ``-0.5 * mean_D((q-mean)^2 / variance + log(variance))``.

    The class-independent Gaussian constant is omitted. Keeping log variance
    penalizes arbitrarily broad classes. Direct differences in bounded chunks
    avoid both a full [N,C,B,D] allocation and quadratic-expansion cancellation.
    """
    mean, _, _ = _statistics(stats)
    query = _features(features, device=mean.device)
    if query.shape[1:] != mean.shape[1:]:
        raise ValueError("Query and statistics band/feature dimensions must agree")
    variance = predictive_variance(
        stats, prior, prior_strength, var_floor, covariance, mean_uncertainty)
    log_variance = variance.log()
    output = query.new_empty((len(query), len(mean), mean.shape[1]))
    for band in range(mean.shape[1]):
        for start in range(0, len(query), 256):
            current = query[start:start + 256, band]
            for class_start in range(0, len(mean), 32):
                stop = class_start + 32
                difference = current[:, None] - mean[None, class_start:stop, band]
                terms = (difference.square() / variance[None, class_start:stop, band]
                         + log_variance[None, class_start:stop, band])
                output[start:start + 256, class_start:stop, band] = -0.5 * terms.mean(-1)
    if not torch.isfinite(output).all():
        raise ValueError("Gaussian scores exceed finite float32 range")
    return output


@torch.no_grad()
def uncertainty_logits(features, stats, prior, prior_strength=20.0,
                       var_floor=1e-6, covariance="shrinkage", mean_uncertainty=True):
    """Average Gaussian log likelihoods equally over frequency bands."""
    return gaussian_band_logits(features, stats, prior, prior_strength, var_floor,
                                covariance, mean_uncertainty).mean(dim=-1)


@torch.no_grad()
def mix_uncertainty_probabilities(reference, logits, alpha, temperature=1.0):
    """Mix likelihood probabilities while preserving each reference vote mass.

    Zero alpha returns the exact original object. Active mixing returns float32
    votes and introduces no gradients or changes to either input tensor.
    """
    alpha = _scalar(alpha, "alpha")
    temperature = _scalar(temperature, "temperature", strict=True)
    if alpha > 1 or not isinstance(reference, torch.Tensor):
        raise ValueError("alpha must lie in [0,1] and reference must be a tensor")
    votes = _tensor(reference, "reference").float()
    scores = _tensor(logits, "logits", votes.device).float()
    if (votes.ndim != 2 or any(size == 0 for size in votes.shape)
            or scores.shape != votes.shape or (votes < 0).any()):
        raise ValueError("Reference and logits must match nonempty [N,C]; votes must be nonnegative")
    if not torch.isfinite(votes).all() or not torch.isfinite(scores).all():
        raise ValueError("Reference and logits must fit float32")
    if alpha == 0:
        return reference
    scaled = scores / temperature
    mass = votes.sum(dim=-1, keepdim=True)
    if not torch.isfinite(scaled).all() or not torch.isfinite(mass).all():
        raise ValueError("Scaled logits and reference mass must be finite")
    return (1 - alpha) * votes + alpha * mass * F.softmax(scaled, dim=-1)
