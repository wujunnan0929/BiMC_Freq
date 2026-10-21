"""Closed-form calibration from class-centred SAFD removal responses.

The module fits sufficient statistics and an optional small metric adapter. Callers must use
only fitting support images: a pseudo-query or held-out class must never enter
``fit_explained_scatter``.  All fitted tensors are detached and stored on CPU.
SAFD itself is evaluated by the caller; query scoring needs the original CLIP
embedding only.  A covariance contraction is not a no-harm guarantee.
"""

from collections.abc import Mapping
import math

import torch
from torch.nn import functional as F


def _positive_scalar(value, name, *, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + ' must be a finite real scalar.')
    value = float(value)
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(name + (' must be finite and nonnegative.' if allow_zero
                                  else ' must be finite and positive.'))
    return value


def _real(value, name, device=None):
    try:
        result = torch.as_tensor(value, device=device).detach()
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError('Invalid ' + name) from error
    if result.is_complex() or result.dtype == torch.bool or not torch.isfinite(result).all():
        raise ValueError(name + ' must contain finite real values.')
    return result


def _labels(value, length, device):
    result = _real(value, 'labels', device)
    if result.ndim != 1 or len(result) != length or result.is_floating_point():
        raise ValueError('labels must be an integer vector matching the samples.')
    return result.long()


def _normalise_rows(value, name):
    if (value.ndim < 2 or value.shape[-1] < 1
            or (value.norm(dim=-1) <= 0).any()):
        raise ValueError(name + ' must have nonzero feature vectors.')
    return F.normalize(value, dim=-1)


def _symmetric(value, name):
    matrix = _real(value, name).double()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not len(matrix):
        raise ValueError(name + ' must be a nonempty square matrix.')
    scale = float(matrix.abs().max())
    if not torch.allclose(matrix, matrix.T, rtol=1e-8, atol=max(scale, 1e-30) * 1e-10):
        raise ValueError(name + ' must be symmetric.')
    return (matrix + matrix.T) * .5


def _psd_eigh(matrix, name):
    """Drop only negative/zero eigenvalues at floating-point roundoff scale."""
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    scale = float(eigenvalues.abs().max())
    tolerance = torch.finfo(torch.float64).eps * max(1, len(matrix)) * scale * 16
    if float(eigenvalues.min()) < -tolerance:
        raise ValueError(name + ' is not positive semidefinite.')
    # Sample scatters are PSD algebraically.  Small negative eigenvalues can
    # arise from GEMM/eigh roundoff; no data-dependent jitter is added.
    eigenvalues = eigenvalues.clamp_min(0)
    return eigenvalues, eigenvectors, tolerance


@torch.no_grad()
def scale_responses(features):
    """Return amplitude-preserving ``[intermediate-coarse, original-intermediate]``.

    ``features[N,3,D]`` is ordered original, coarse, intermediate.  Each CLIP
    snapshot is unit-normalised first; the resulting ``[N,2,D]`` differences
    are *not* unit-normalised.  Repeat-original inputs produce exact zeros.
    """
    values = _real(features, 'features').double()
    if values.ndim != 3 or values.shape[1] != 3 or not len(values):
        raise ValueError('features must have nonempty shape [N,3,D].')
    values = _normalise_rows(values, 'features')
    original, coarse, intermediate = values.unbind(1)
    return torch.stack((intermediate - coarse, original - intermediate), 1)


@torch.no_grad()
def fit_explained_scatter(features, responses, labels, rank=64, ridge=.1,
                          mode='explained', class_ids=None, var_floor=1e-12):
    """Fit a regularised explained scatter or a rank-matched original control.

    ``features[N,D]`` should be unit CLIP embeddings supplied by the caller.
    They are deliberately not renormalised here: the estimator acts in the
    supplied coordinates and is invariant to per-class mean translations.
    Responses may be ``[N,P]`` or ``[N,B,D]``; their amplitude is retained.

    With degrees of freedom ``df=sum_c(n_c-1)``, both embedding and response
    vectors are centred within class.  For response-PCA directions U,

        N = C_xr U (U.T C_rr U + lambda I)^-1 U.T C_rx,
        lambda = ridge * max(trace(C_rr)/P, var_floor).

    Thus 0 <= N <= C_xx up to numerical roundoff. ``original_within`` uses
    the leading ``rank`` directions of C_xx as a cheap same-rank control;
    it needs no responses.  ``class_ids`` explicitly excludes other labels.
    Returned CPU tensors contain no individual image features or gradients.
    """
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError('rank must be a positive integer.')
    ridge = _positive_scalar(ridge, 'ridge')
    var_floor = _positive_scalar(var_floor, 'var_floor')
    if mode not in ('explained', 'original_within'):
        raise ValueError('mode must be explained or original_within.')
    x = _real(features, 'features').double()
    if x.ndim != 2 or min(x.shape) < 1:
        raise ValueError('features must have nonempty shape [N,D].')
    y = _labels(labels, len(x), x.device)
    if class_ids is None:
        ids = torch.unique(y, sorted=True)
    else:
        ids = _real(class_ids, 'class_ids', x.device)
        if ids.ndim != 1 or not len(ids) or ids.is_floating_point() or ids.unique().numel() != ids.numel():
            raise ValueError('class_ids must be distinct integers.')
        ids = ids.long()
    r = None
    if mode == 'explained':
        r = _real(responses, 'responses', x.device).double()
        if r.ndim not in (2, 3) or len(r) != len(x) or min(r.shape) < 1:
            raise ValueError('responses must have shape [N,P] or [N,B,D].')
        r = r.flatten(1)
    x_groups, r_groups, counts = [], [], []
    for cls in ids:
        mask = y == cls
        count = int(mask.sum())
        if not count:
            raise ValueError('Every fitting class must have support samples.')
        current = x[mask]
        x_groups.append(current - current.mean(0))
        counts.append(count)
        if r is not None:
            current = r[mask]
            r_groups.append(current - current.mean(0))
    degrees = sum(counts) - len(counts)
    if degrees <= 0:
        raise ValueError('Fitting needs positive within-class degrees of freedom.')
    e = torch.cat(x_groups)
    within = e.T @ e / degrees
    within = (within + within.T) * .5
    if not torch.isfinite(within).all():
        raise ValueError('Within-class scatter exceeds float64 range.')
    selected_rank, response_dim, regularizer = 0, 0, 0.
    if mode == 'original_within':
        values, vectors, tolerance = _psd_eigh(within, 'within scatter')
        keep = torch.where(values > tolerance)[0][-min(rank, degrees, x.shape[1]):]
        selected_rank = len(keep)
        scatter = (vectors[:, keep] * values[keep]) @ vectors[:, keep].T
    else:
        u = torch.cat(r_groups)
        response_dim = u.shape[1]
        response_cov = u.T @ u / degrees
        response_cov = (response_cov + response_cov.T) * .5
        cross = e.T @ u / degrees
        if not torch.isfinite(response_cov).all() or not torch.isfinite(cross).all():
            raise ValueError('Response scatter exceeds float64 range.')
        values, vectors, tolerance = _psd_eigh(response_cov, 'response scatter')
        keep = torch.where(values > tolerance)[0][-min(rank, degrees, response_dim):]
        selected_rank = len(keep)
        regularizer = ridge * max(float(response_cov.trace()) / response_dim, var_floor)
        if not math.isfinite(regularizer):
            raise ValueError('Response regularizer exceeds float64 range.')
        if selected_rank:
            cross_projected = cross @ vectors[:, keep]
            denominator = values[keep] + regularizer
            # Equivalent to the ridge solve in response-PCA coordinates.
            scatter = (cross_projected / denominator) @ cross_projected.T
        else:
            scatter = torch.zeros_like(within)
    scatter = (scatter + scatter.T) * .5
    if not torch.isfinite(scatter).all():
        raise ValueError('Explained scatter exceeds float64 range.')
    trace_within, trace_scatter = float(within.trace()), float(scatter.trace())
    return {
        'schema_version': 1, 'mode': mode,
        'scatter': scatter.cpu(), 'within_scatter': within.cpu(),
        'feature_dim': x.shape[1], 'response_dim': response_dim,
        'class_ids': ids.cpu(), 'counts': torch.tensor(counts, dtype=torch.long),
        'diagnostics': {
            'num_samples': sum(counts), 'num_classes': len(counts),
            'degrees_of_freedom': degrees, 'requested_rank': rank,
            'retained_rank': selected_rank, 'ridge': ridge,
            'ridge_regularizer': regularizer,
            'within_trace': trace_within, 'explained_trace': trace_scatter,
            'explained_trace_fraction': trace_scatter / trace_within if trace_within > 0 else 0.,
        },
    }


@torch.no_grad()
def metric_from_scatter(scatter_state, strength=1., var_floor=1e-12):
    """Return M=(I+strength*N/max(trace(C_xx)/D,var_floor))^-1 on CPU.

    Scaling uses the *original* within-class scatter, so a weak response is
    not artificially promoted to a unit-strength correction.  Zero strength
    and exactly zero scatter return an exact float64 identity matrix.
    """
    strength = _positive_scalar(strength, 'strength', allow_zero=True)
    var_floor = _positive_scalar(var_floor, 'var_floor')
    if not isinstance(scatter_state, Mapping) or not {'scatter', 'within_scatter'} <= scatter_state.keys():
        raise ValueError('Expected a fitted scatter state.')
    scatter = _symmetric(scatter_state['scatter'], 'scatter').cpu()
    within = _symmetric(scatter_state['within_scatter'], 'within scatter').cpu()
    if scatter.shape != within.shape:
        raise ValueError('Scatter and within scatter shapes must agree.')
    identity = torch.eye(len(scatter), dtype=torch.float64)
    if strength == 0 or not torch.count_nonzero(scatter):
        return identity
    _psd_eigh(within, 'within scatter')
    values, vectors, _ = _psd_eigh(scatter, 'scatter')
    scale = max(float(within.trace()) / len(within), var_floor)
    inverse = (1 + strength * (values / scale)).reciprocal()
    if not torch.isfinite(inverse).all() or (inverse <= 0).any():
        raise ValueError('Metric strength exceeds float64 range.')
    metric = (vectors * inverse) @ vectors.T
    metric = (metric + metric.T) * .5
    _, info = torch.linalg.cholesky_ex(metric)
    if int(info) != 0:
        raise ValueError('Metric loses positive definiteness; reduce strength.')
    return metric


@torch.no_grad()
def build_response_basis(scatter_state, rank=64):
    """Return positive-scatter eigenvectors, strongest first, as CPU float32.

    Column signs are fixed by their largest absolute coordinate.  Repeated
    eigenvalues do not identify a unique basis across numerical backends;
    save and freeze the returned basis instead of refitting it at inference.
    Exact zero-response states yield a valid ``[D,0]`` basis.
    """
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError('rank must be a positive integer.')
    if not isinstance(scatter_state, Mapping) or 'scatter' not in scatter_state:
        raise ValueError('Expected a fitted scatter state.')
    scatter = _symmetric(scatter_state['scatter'], 'scatter').cpu()
    values, vectors, tolerance = _psd_eigh(scatter, 'scatter')
    keep = torch.where(values > tolerance)[0].flip(0)[:rank]
    basis = vectors[:, keep]
    if len(keep):
        positions = basis.abs().argmax(0)
        signs = basis[positions, torch.arange(len(keep))].sign()
        basis = basis * signs
    return basis.float()


class SAFDMetricAdapter(torch.nn.Module):
    """Base-stage trainable diagonal metric in a frozen SAFD response basis.

    The sole parameter is ``theta[r]``. With orthonormal U and 0<kappa<1,
    M=I+U diag(kappa*tanh(theta)) U.T has eigenvalues in
    [1-kappa,1+kappa]. Both attenuation and amplification are allowed;
    response directions are not assumed to be pure nuisance. Incremental
    inference must freeze theta as well as the CLIP encoder and basis.
    """

    def __init__(self, basis, kappa=.9):
        super().__init__()
        kappa = _positive_scalar(kappa, 'kappa')
        if kappa >= 1:
            raise ValueError('kappa must be smaller than one for positive definiteness.')
        value = _real(basis, 'basis').float().clone()
        if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] > value.shape[0]:
            raise ValueError('basis must have shape [D,r] with 0<=r<=D.')
        if value.shape[1]:
            gram = value.double().T @ value.double()
            if not torch.allclose(gram, torch.eye(len(gram), device=value.device, dtype=torch.float64),
                                  rtol=1e-5, atol=1e-5):
                raise ValueError('basis columns must be orthonormal.')
        self.register_buffer('basis', value)
        self.theta = torch.nn.Parameter(value.new_zeros(value.shape[1]))
        self.kappa = kappa

    def coefficients(self):
        return self.kappa * self.theta.tanh()

    def delta_scores(self, original_features, prototypes):
        """Differentiable [N,C] metric residual; gradients enter theta only."""
        x = _real(original_features, 'original features', self.basis.device).to(self.basis.dtype)
        mu = _real(prototypes, 'prototypes', self.basis.device).to(self.basis.dtype)
        if (x.ndim != 2 or mu.ndim != 2 or x.shape[1] != self.basis.shape[0]
                or mu.shape[1] != self.basis.shape[0] or not len(mu)):
            raise ValueError('Queries/prototypes must have shapes [N,D]/[C,D] matching the basis.')
        x = _normalise_rows(x, 'original features')
        mu = _normalise_rows(mu, 'prototypes')
        query_coordinates, class_coordinates = x @ self.basis, mu @ self.basis
        coefficient = self.coefficients()
        return (2 * (query_coordinates * coefficient) @ class_coordinates.T
                - (class_coordinates.square() * coefficient).sum(-1)[None])

    def forward(self, original_features, prototypes):
        return self.delta_scores(original_features, prototypes)

    @torch.no_grad()
    def to_metric(self):
        """Export a detached metric on the adapter device for frozen inference."""
        identity = torch.eye(self.basis.shape[0], device=self.basis.device, dtype=self.basis.dtype)
        coefficient = self.coefficients()
        if not torch.count_nonzero(coefficient):
            return identity
        matrix = identity + (self.basis * coefficient) @ self.basis.T
        return ((matrix + matrix.T) * .5).detach()


@torch.no_grad()
def prepare_residual_classifier(prototypes, metric, *, validate_metric=True):
    """Compile s_M-s_I as q @ weights + bias for all currently seen classes.

    Prototypes and query vectors are unit-normalised in this scoring model.
    The state is rebuilt only when class prototypes change; no query SAFD or
    image-dependent fitting occurs.  Stored weights are detached CPU tensors.
    Set ``validate_metric=False`` only for a matrix already checked positive
    definite (or guaranteed by the bounded adapter), to avoid repeating an
    O(D^3) Cholesky per episode. Shape/finiteness/symmetry checks remain active.
    """
    mu = _real(prototypes, 'prototypes').double()
    if mu.ndim != 2 or min(mu.shape) < 1:
        raise ValueError('prototypes must have shape [C,D].')
    mu = _normalise_rows(mu, 'prototypes')
    matrix = _symmetric(metric, 'metric').to(mu.device)
    if matrix.shape != (mu.shape[1], mu.shape[1]):
        raise ValueError('Metric and prototype dimensions disagree.')
    if validate_metric:
        _, info = torch.linalg.cholesky_ex(matrix)
        if int(info) != 0:
            raise ValueError('Metric must be positive definite.')
    identity = torch.eye(mu.shape[1], dtype=torch.float64, device=mu.device)
    is_identity = torch.equal(matrix, identity)
    delta = matrix - identity
    weights = 2 * delta @ mu.T
    bias = -(mu * (mu @ delta)).sum(1)
    return {'weights': weights.cpu(), 'bias': bias.cpu(),
            'feature_dim': mu.shape[1], 'num_classes': len(mu),
            'is_identity': is_identity}


@torch.no_grad()
def apply_metric_residual(reference_votes, original_features, prepared, gain=1.):
    """Add the metric residual to log votes, preserving each row's vote mass.

    Positive output votes are float32 (float64 for float64 inputs). Row mass
    is preserved up to output roundoff. The exact original object is returned
    for zero gain or identity metric, without constructing a gradient graph.
    """
    gain = _positive_scalar(gain, 'gain', allow_zero=True)
    votes = _real(reference_votes, 'reference votes')
    if (not votes.is_floating_point() or votes.ndim != 2 or min(votes.shape) < 1
            or (votes <= 0).any()):
        raise ValueError('reference votes must be finite positive [N,C].')
    if not isinstance(prepared, Mapping) or not {'weights', 'bias', 'is_identity'} <= prepared.keys():
        raise ValueError('Expected a prepared residual classifier.')
    x = _real(original_features, 'original features', votes.device).double()
    weights = _real(prepared['weights'], 'weights', votes.device).double()
    bias = _real(prepared['bias'], 'bias', votes.device).double()
    if (x.ndim != 2 or len(x) != len(votes) or weights.shape != (x.shape[1], votes.shape[1])
            or bias.shape != (votes.shape[1],)):
        raise ValueError('Votes, queries and prepared classifier shapes disagree.')
    if gain == 0 or prepared['is_identity']:
        return reference_votes
    x = _normalise_rows(x, 'original features')
    residual = x @ weights + bias
    logits = votes.double().log() + gain * residual
    if not torch.isfinite(logits).all():
        raise ValueError('Metric residual scores exceed float64 range.')
    output = logits.softmax(-1) * votes.double().sum(-1, keepdim=True)
    dtype = torch.float64 if votes.dtype == torch.float64 else torch.float32
    output = output.to(dtype=dtype)
    # Softmax underflow must not turn a positive BiMC vote into an exact zero.
    smallest = torch.nextafter(output.new_tensor(0.), output.new_tensor(1.))
    return output.clamp_min(smallest)
