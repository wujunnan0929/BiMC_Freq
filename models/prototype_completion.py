"""Low-rank empirical-Bayes prototype completion on frozen CLIP features.

Features and fixed text anchors are expected to have been encoded/normalized
by the caller. Individual feature normalization is deliberately not performed
here: Gaussian class means must retain their length. All fitting statistics
come only from the explicitly supplied fitting class IDs. The resulting model
uses joint, paired-view observation noise, and outputs only an original-image
mean, retaining its unmodelled components exactly.

This is closed-form statistical fitting, not an exact probabilistic model of
CLIP's spherical embeddings and not an absence of base-data learning.
"""

import math

import torch


def _number(value, name, *, positive=False, unit=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + ' must be a finite real scalar.')
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(name + ' must be finite and positive.')
    if unit and not 0 <= value <= 1:
        raise ValueError(name + ' must lie in [0,1].')
    return value


def _integer_vector(value, name, device):
    result = torch.as_tensor(value, device=device).detach()
    if result.ndim != 1 or not result.numel() or result.dtype == torch.bool or result.is_complex():
        raise ValueError(name + ' must be a nonempty integer vector.')
    if not torch.isfinite(result).all() or (result.is_floating_point() and not torch.equal(result, result.round())):
        raise ValueError(name + ' must contain finite integers.')
    return result.long()


def _real(value, name, device=None):
    result = torch.as_tensor(value, device=device).detach()
    if result.is_complex() or result.dtype == torch.bool or not torch.isfinite(result).all():
        raise ValueError(name + ' must contain finite real values.')
    result = result.to(dtype=torch.float64)
    if not torch.isfinite(result).all():
        raise ValueError(name + ' exceeds float64 range.')
    return result


def _inputs(features, labels, anchors, class_ids):
    x = _real(features, 'features')
    if x.ndim != 3 or any(size < 1 for size in x.shape):
        raise ValueError('features must have nonempty shape [N,V,D].')
    y = _integer_vector(labels, 'labels', x.device)
    ids = _integer_vector(class_ids, 'class_ids', x.device)
    t = _real(anchors, 'anchors', x.device)
    if len(y) != len(x) or t.shape != (len(ids), x.shape[1], x.shape[2]):
        raise ValueError('Features, labels, anchors and class IDs have incompatible shapes.')
    if ids.unique().numel() != ids.numel():
        raise ValueError('class_ids must be unique.')
    groups = [x[y == class_id] for class_id in ids]
    if any(not len(group) for group in groups):
        raise ValueError('Every requested class must contain support samples.')
    return x, t, ids, groups


def _block_diagonal(matrix, views, rank):
    index = torch.arange(views * rank, device=matrix.device) // rank
    return matrix * (index[:, None] == index[None, :])


def _regularize(matrix, shrinkage, ridge):
    """Project finite-sample estimates to PSD before isotropic shrinkage."""
    matrix = (matrix + matrix.T) * .5
    values, vectors = torch.linalg.eigh(matrix)
    matrix = (vectors * values.clamp_min(0)[None, :]) @ vectors.T
    scale = matrix.diagonal().mean().clamp_min(0)
    identity = torch.eye(len(matrix), device=matrix.device, dtype=matrix.dtype)
    result = (1 - shrinkage) * matrix + (shrinkage * scale + ridge) * identity
    return (result + result.T) * .5


@torch.no_grad()
def fit_completion(features, labels, anchors, class_ids, *, rank=16,
                   shrinkage=.2, ridge=1e-5, cross_view=True):
    """Fit a class-balanced text prior and paired-view sampling covariances.

    ``features`` is [N,V,D], with original images at view zero; ``anchors`` is
    [C,V,D] in ``class_ids`` order. Rows with other labels are ignored. At least
    two fitting classes and positive pooled within-class degrees of freedom
    are required. ``rank`` is capped at D for small synthetic feature spaces.

    The between-class residual covariance subtracts W*mean(1/n_c), correcting
    for finite base-class means before PSD projection. B and W both use the
    same ridge and shrinkage. Setting ``cross_view=False`` removes all their
    cross-view blocks, explicitly disabling transfer from auxiliary views.
    """
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError('rank must be a positive integer.')
    shrinkage = _number(shrinkage, 'shrinkage', unit=True)
    ridge = _number(ridge, 'ridge', positive=True)
    if not isinstance(cross_view, bool):
        raise ValueError('cross_view must be a bool.')
    x, text, ids, groups = _inputs(features, labels, anchors, class_ids)
    classes, views, dimension = len(ids), x.shape[1], x.shape[2]
    if classes < 2:
        raise ValueError('Completion fitting requires at least two fitting classes.')
    counts = torch.tensor([len(group) for group in groups], device=x.device, dtype=x.dtype)
    degrees = int(counts.sum().item()) - classes
    if degrees <= 0:
        raise ValueError('Completion fitting requires positive within-class degrees of freedom.')
    effective_rank = min(rank, dimension)
    means = torch.stack([group.mean(0) for group in groups])
    offset_full = (means - text).mean(0)
    class_residual = means - text - offset_full
    centred = torch.cat([group - mean for group, mean in zip(groups, means)], dim=0)

    # Each eigensystem is only D x D; the joint model is V*rank x V*rank.
    bases = []
    for view in range(views):
        residual = class_residual[:, view]
        within_view = centred[:, view]
        scatter = residual.T @ residual / (classes - 1)
        scatter = scatter + within_view.T @ within_view / degrees
        _, directions = torch.linalg.eigh((scatter + scatter.T) * .5)
        bases.append(directions[:, -effective_rank:].flip(1))
    basis = torch.stack(bases)
    project = lambda values: torch.einsum('nvd,vdr->nvr', values, basis).reshape(len(values), -1)
    projected_within = project(centred)
    within_raw = projected_within.T @ projected_within / degrees
    residual = project(class_residual)
    between_observed = residual.T @ residual / (classes - 1)
    between_raw = between_observed - within_raw * counts.reciprocal().mean()
    if not cross_view:
        within_raw = _block_diagonal(within_raw, views, effective_rank)
        between_raw = _block_diagonal(between_raw, views, effective_rank)
    within = _regularize(within_raw, shrinkage, ridge)
    between = _regularize(between_raw, shrinkage, ridge)
    # Enforce exact zero off-blocks even after numerical eigenvector rotations.
    if not cross_view:
        within = _block_diagonal(within, views, effective_rank)
        between = _block_diagonal(between, views, effective_rank)
    return {'basis': basis.detach(),
            'offset': torch.einsum('vd,vdr->vr', offset_full, basis).detach(),
            'between': between.detach(), 'within': within.detach(),
            'views': views, 'feature_dim': dimension, 'rank': effective_rank,
            'requested_rank': rank, 'shrinkage': shrinkage, 'ridge': ridge,
            'cross_view': cross_view, 'fit_class_count': classes,
            'fit_sample_count': int(counts.sum().item()),
            'fit_class_ids': ids.detach(), 'fit_class_counts': counts.long().detach(),
            'within_degrees_of_freedom': degrees}


def _state(state, views, dimension, device):
    if not isinstance(state, dict):
        raise ValueError('Completion state must be a dictionary.')
    rank = state.get('rank')
    if (isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= dimension
            or state.get('views') != views or state.get('feature_dim') != dimension):
        raise ValueError('Completion state view/feature dimensions do not match support.')
    arrays = {}
    for key, shape in (('basis', (views, dimension, rank)), ('offset', (views, rank)),
                       ('between', (views * rank, views * rank)),
                       ('within', (views * rank, views * rank))):
        if key not in state:
            raise ValueError('Completion state is missing ' + key)
        value = _real(state[key], 'state ' + key, device)
        if value.shape != shape:
            raise ValueError('Completion state shape mismatch: ' + key)
        arrays[key] = value
    identity = torch.eye(rank, device=device, dtype=torch.float64).expand(views, -1, -1)
    if not torch.allclose(arrays['basis'].transpose(1, 2) @ arrays['basis'], identity, atol=1e-5, rtol=1e-5):
        raise ValueError('Completion bases must have orthonormal columns.')
    for key in ('between', 'within'):
        value = arrays[key]
        if not torch.allclose(value, value.T, atol=1e-8, rtol=1e-5):
            raise ValueError('Completion covariance must be symmetric: ' + key)
        if torch.linalg.eigvalsh(value).min() < -1e-9:
            raise ValueError('Completion covariance must be positive semidefinite: ' + key)
    return arrays


@torch.no_grad()
def complete_prototypes(state, features, labels, anchors, class_ids):
    """Return raw completed ORIGINAL-space means [C,D], with no query use.

    Only explicitly supplied support rows determine observations and counts.
    The frozen base within-class covariance handles one-shot classes. No
    per-new-class empirical covariance, text selection or gradient fitting is
    performed. Normalize the returned means only at the classifier boundary.
    """
    dtype = torch.as_tensor(features).dtype
    output_dtype = dtype if dtype in (torch.float32, torch.float64) else torch.float32
    x, text, _, groups = _inputs(features, labels, anchors, class_ids)
    values = _state(state, x.shape[1], x.shape[2], x.device)
    basis, offset = values['basis'], values['offset']
    between, within = values['between'], values['within']
    means = torch.stack([group.mean(0) for group in groups])
    observed = torch.einsum('cvd,vdr->cvr', means, basis).reshape(len(groups), -1)
    prior = (torch.einsum('cvd,vdr->cvr', text, basis) + offset).reshape(len(groups), -1)
    completed = []
    rank = state['rank']
    for j, group in enumerate(groups):
        covariance = between + within / len(group)
        try:
            posterior = prior[j] + between @ torch.linalg.solve(covariance, observed[j] - prior[j])
        except RuntimeError as error:
            raise ValueError('Completion posterior covariance is singular or invalid.') from error
        correction = (posterior[:rank] - observed[j, :rank]) @ basis[0].T
        completed.append(means[j, 0] + correction)
    result = torch.stack(completed).to(output_dtype)
    if not torch.isfinite(result).all():
        raise ValueError('Completed prototypes must remain finite.')
    return result.detach()
