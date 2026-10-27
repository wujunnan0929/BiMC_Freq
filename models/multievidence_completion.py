"""Frozen, typed linear-Gaussian measurements for prototype completion.

Each visual block keeps its native coordinates.  Text groups are class-level
measurements, not duplicated image observations.  Fitting uses only explicit
class IDs; text masks remove unknown groups rather than treating zero as an
observed attribute.  This is empirical statistical learning, not an exact model
of CLIP features.  Only numerical statistics and class provenance are retained.
"""

import math

import torch


FORMAT = 'typed_multievidence_completion_v2'


def _real(value, name, device=None):
    tensor = torch.as_tensor(value, device=device).detach()
    if tensor.dtype == torch.bool or tensor.is_complex() or not torch.isfinite(tensor).all():
        raise ValueError(name + ' must contain finite real values.')
    return tensor.to(torch.float64)


def _ids(value, name, device=None):
    tensor = torch.as_tensor(value, device=device).detach()
    if tensor.ndim != 1 or not tensor.numel() or tensor.dtype == torch.bool or tensor.is_complex():
        raise ValueError(name + ' must be a nonempty integer vector.')
    if not torch.isfinite(tensor).all() or (tensor.is_floating_point() and not torch.equal(tensor, tensor.round())):
        raise ValueError(name + ' must contain finite integers.')
    return tensor.long()


def _positive(value, name, allow_zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (not allow_zero and value == 0)):
        raise ValueError(name + ' must be a finite ' + ('nonnegative' if allow_zero else 'positive') + ' scalar.')
    return float(value)


def _rank(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(name + ' must be a positive integer.')
    return value


def _mask(value, count, name, device):
    if value is None:
        return torch.ones(count, dtype=torch.bool, device=device)
    mask = torch.as_tensor(value, device=device).detach()
    if mask.shape != (count,) or mask.is_complex() or not torch.isfinite(mask).all():
        raise ValueError(name + ' must have finite shape [classes].')
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError(name + ' must be boolean or binary.')
    return mask.bool()


def _psd(matrix, shrinkage=0., floor=0.):
    matrix = (matrix + matrix.T) / 2
    if not len(matrix):
        return matrix
    values, vectors = torch.linalg.eigh(matrix)
    matrix = (vectors * values.clamp_min(0)[None]) @ vectors.T
    scale = matrix.diagonal().mean().clamp_min(0)
    identity = torch.eye(len(matrix), device=matrix.device, dtype=matrix.dtype)
    result = (1 - shrinkage) * matrix + (shrinkage * scale + floor) * identity
    return (result + result.T) / 2


def _measurement_covariance(matrix, blocks, shrinkage):
    """Regularize each diagonal block locally, attenuating only cross blocks.

    Adding a heterogeneous measurement cannot change an existing block's
    isotropic target. Pairwise missing-data residual covariances may be
    indefinite, so scale cross blocks down until the fixed diagonal blocks
    admit a PSD joint matrix. No global eigenvalue projection changes them.
    """
    matrix = (matrix + matrix.T) / 2
    diagonal, original_diagonal = torch.zeros_like(matrix), torch.zeros_like(matrix)
    for begin, end in blocks:
        diagonal[begin:end, begin:end] = _psd(matrix[begin:end, begin:end], shrinkage)
        original_diagonal[begin:end, begin:end] = matrix[begin:end, begin:end]
    cross = (1-shrinkage) * (matrix-original_diagonal)
    # Prediction checks PSD with an absolute tolerance. A high-variance
    # heterogeneous block must not authorize a materially negative eigenvalue
    # in another block merely by increasing a global relative tolerance.
    tolerance = min(float(diagonal.diagonal().abs().max().clamp_min(1e-12))*1e-12, 1e-10)
    def valid(scale):
        return float(torch.linalg.eigvalsh(diagonal+scale*cross).min()) >= -tolerance
    scale = 1.
    if not valid(scale):
        lower, upper = 0., 1.
        for _ in range(45):
            middle = (lower+upper)/2
            if valid(middle):
                lower = middle
            else:
                upper = middle
        scale = lower
    result = diagonal + scale*cross
    return (result+result.T)/2, scale


def _basis(scatter, rank):
    values, vectors = torch.linalg.eigh((scatter + scatter.T) / 2)
    return vectors[:, -min(rank, len(scatter)):].flip(1)


def _covariance(values):
    centered = values - values.mean(0)
    return centered.T @ centered / max(1, len(values) - 1)


def _mapping(u, observed, counts, within_u, within_yu, ridge):
    """Class-balanced ridge with a finite-class-mean errors-in-variables correction."""
    r = u.shape[1]
    if len(u) == 1:
        h = u.new_zeros(observed.shape[1], r)
        return h, observed.mean(0)
    a, b = u - u.mean(0), observed - observed.mean(0)
    inverse_count = counts.reciprocal().mean()
    latent_cov = _psd(a.T @ a / (len(u) - 1) - within_u * inverse_count)
    cross_cov = b.T @ a / (len(u) - 1) - within_yu * inverse_count
    scale = latent_cov.diagonal().mean().clamp_min(1e-10)
    system = latent_cov + ridge * scale * torch.eye(r, device=u.device, dtype=u.dtype)
    h = torch.linalg.solve(system, cross_cov.T).T
    return h, observed.mean(0) - h @ u.mean(0)


def _fit_inputs(visual_blocks, labels, text_blocks, class_ids, text_masks):
    if not isinstance(visual_blocks, dict) or 'final' not in visual_blocks:
        raise ValueError('visual_blocks must contain the final feature block.')
    if not isinstance(text_blocks, dict) or (text_masks is not None and not isinstance(text_masks, dict)):
        raise ValueError('text_blocks and text_masks must be dictionaries.')
    if any(not isinstance(k, str) or not k for k in list(visual_blocks) + list(text_blocks)):
        raise ValueError('Block names must be nonempty strings.')
    first = _real(visual_blocks['final'], 'final features')
    if first.ndim != 2 or min(first.shape) < 1:
        raise ValueError('Final features must have nonempty shape [N,D].')
    y, ids = _ids(labels, 'labels', first.device), _ids(class_ids, 'class_ids', first.device)
    if len(y) != len(first) or ids.unique().numel() != len(ids):
        raise ValueError('Labels must match rows and class_ids must be unique.')
    groups = [torch.where(y == c)[0] for c in ids]
    if any(not len(g) for g in groups):
        raise ValueError('Every fitting class must contain samples.')
    rows = torch.cat(groups)
    visual = {}
    for name in ['final'] + sorted(k for k in visual_blocks if k != 'final'):
        if not isinstance(name, str) or not name:
            raise ValueError('Block names must be nonempty strings.')
        x = _real(visual_blocks[name], 'visual ' + name, first.device)
        if x.ndim != 2 or x.shape[0] != len(y) or x.shape[1] < 1:
            raise ValueError('Visual blocks must have aligned nonempty [N,D] arrays.')
        visual[name] = x[rows]
    counts = torch.tensor([len(g) for g in groups], device=first.device, dtype=first.dtype)
    text, masks = {}, {}
    for name in sorted(text_blocks):
        if not isinstance(name, str) or not name:
            raise ValueError('Block names must be nonempty strings.')
        t = _real(text_blocks[name], 'text ' + name, first.device)
        if t.ndim != 2 or t.shape[0] != len(ids) or t.shape[1] < 1:
            raise ValueError('Text blocks must have nonempty shape [classes,D].')
        text[name] = t
        masks[name] = _mask((text_masks or {}).get(name), len(ids), 'mask ' + name, first.device)
    if set(text_masks or {}) - set(text):
        raise ValueError('Text masks name an unknown block.')
    return visual, text, masks, ids, counts


@torch.no_grad()
def fit_model(visual_blocks, labels, text_blocks, class_ids, *, target_rank=16,
              auxiliary_rank=8, text_rank=4, ridge=.1, shrinkage=.2,
              solver_ridge=1e-8, crossfit_folds=5, target_basis=None,
              text_masks=None):
    """Fit native-dimensional measurements on explicitly supplied base classes.

    Unknown text groups are excluded block-wise. Q is estimated from class-
    crossfit residuals using pairwise available classes, subtracting the paired
    finite-mean noise approximation before PSD projection. The Q final rows
    and W text rows stay exactly zero; solver jitter is recorded separately.
    """
    target_rank, auxiliary_rank, text_rank = (_rank(target_rank, 'target_rank'),
        _rank(auxiliary_rank, 'auxiliary_rank'), _rank(text_rank, 'text_rank'))
    ridge, solver_ridge = _positive(ridge, 'ridge'), _positive(solver_ridge, 'solver_ridge', True)
    shrinkage = _positive(shrinkage, 'shrinkage', True)
    if shrinkage > 1:
        raise ValueError('shrinkage must lie in [0,1].')
    if isinstance(crossfit_folds, bool) or not isinstance(crossfit_folds, int) or crossfit_folds < 2:
        raise ValueError('crossfit_folds must be at least two.')
    visual, text, masks, ids, counts = _fit_inputs(visual_blocks, labels, text_blocks, class_ids, text_masks)
    classes = len(ids)
    degrees = int(counts.sum().item()) - classes
    if classes < 2 or degrees < 1:
        raise ValueError('Fitting requires two classes and positive within-class degrees of freedom.')
    boundaries = [0] + counts.long().cumsum(0).tolist()
    means, within_rows = {}, {}
    for name, x in visual.items():
        groups = [x[boundaries[j]:boundaries[j+1]] for j in range(classes)]
        means[name] = torch.stack([g.mean(0) for g in groups])
        within_rows[name] = torch.cat([g - mean for g, mean in zip(groups, means[name])])

    visual_aliases, text_aliases, dropped = {}, {}, []
    unique_visual = {}
    for name, x in visual.items():
        duplicate = next((k for k, v in unique_visual.items() if v.shape == x.shape and torch.equal(v, x)), None)
        if duplicate is None:
            unique_visual[name] = x
        else:
            visual_aliases[name] = duplicate
    unique_text = {}
    for name, values in text.items():
        if int(masks[name].sum()) < 2:
            dropped.append(name)
            continue
        duplicate = next((k for k, v in unique_text.items() if v.shape == values.shape
                          and torch.equal(masks[k], masks[name])
                          and torch.equal(v[masks[k]], values[masks[name]])), None)
        if duplicate is None:
            unique_text[name] = values
        else:
            text_aliases[name] = duplicate

    blocks, projected_means, projected_within, dimension = [], [], [], 0
    for name in unique_visual:
        scatter = _covariance(means[name]) + within_rows[name].T @ within_rows[name] / degrees
        if name == 'final' and target_basis is not None:
            basis = _real(target_basis, 'target_basis', visual['final'].device)
            if basis.ndim != 2 or basis.shape[0] != visual['final'].shape[1] or not 1 <= basis.shape[1] <= basis.shape[0]:
                raise ValueError('target_basis must have shape [final_dimension,rank].')
            identity = torch.eye(basis.shape[1], device=basis.device, dtype=basis.dtype)
            if not torch.allclose(basis.T @ basis, identity, atol=1e-6, rtol=1e-6):
                raise ValueError('target_basis columns must be orthonormal.')
        else:
            basis = _basis(scatter, target_rank if name == 'final' else auxiliary_rank)
        r = basis.shape[1]
        blocks.append({'name': name, 'kind': 'visual', 'basis': basis.detach(),
                       'input_dimension': basis.shape[0], 'rank': r,
                       'begin': dimension, 'end': dimension+r})
        projected_means.append(means[name] @ basis)
        projected_within.append(within_rows[name] @ basis)
        dimension += r
    visual_dimension = dimension
    visual_within = torch.cat(projected_within, 1)
    within = visual_within.T @ visual_within / degrees
    available = [torch.ones(classes, dtype=torch.bool, device=ids.device) for _ in blocks]
    for name, values in unique_text.items():
        observed = values[masks[name]]
        basis = _basis(_covariance(observed), min(text_rank, max(1, len(observed)-1)))
        r = basis.shape[1]
        blocks.append({'name': name, 'kind': 'text', 'basis': basis.detach(),
                       'input_dimension': basis.shape[0], 'rank': r,
                       'begin': dimension, 'end': dimension+r})
        projected_means.append(values @ basis)
        available.append(masks[name])
        dimension += r
    observations = torch.cat(projected_means, 1)
    target_r = blocks[0]['rank']
    u = observations[:, :target_r]
    w = u.new_zeros(dimension, dimension)
    w[:visual_dimension, :visual_dimension], w_cross_scale = _measurement_covariance(
        within, [(b['begin'], b['end']) for b in blocks if b['kind']=='visual'], shrinkage)
    within_u = within[:target_r, :target_r]
    observed_target_cov = _covariance(u)
    prior_floor = float((observed_target_cov.diagonal().mean()
                         + within_u.diagonal().mean()).clamp_min(1e-12) * 1e-10)
    between = _psd(observed_target_cov - within_u * counts.reciprocal().mean(),
                   shrinkage, floor=prior_floor)
    mean = u.mean(0)
    h, bias = u.new_zeros(dimension, target_r), u.new_zeros(dimension)
    h[:target_r] = torch.eye(target_r, device=u.device, dtype=u.dtype)
    residuals = u.new_zeros(classes, dimension)
    observed_columns = torch.zeros(classes, dimension, dtype=torch.bool, device=u.device)
    noise_residual = u.new_zeros(classes, dimension, visual_dimension)
    traces = []
    for block, mask in zip(blocks, available):
        begin, end = block['begin'], block['end']
        observed_columns[:, begin:end] = mask[:, None]
        if block['name'] == 'final' and block['kind'] == 'visual':
            continue
        indices = torch.where(mask)[0]
        noise_yu = within[begin:end, :target_r] if block['kind'] == 'visual' else u.new_zeros(end-begin, target_r)
        mapping, intercept = _mapping(u[mask], observations[mask, begin:end], counts[mask], within_u, noise_yu, ridge)
        h[begin:end], bias[begin:end] = mapping, intercept
        folds = min(crossfit_folds, len(indices))
        for fold in range(folds):
            held = indices[torch.arange(len(indices), device=u.device) % folds == fold]
            train = indices[torch.arange(len(indices), device=u.device) % folds != fold]
            mapping, intercept = _mapping(u[train], observations[train, begin:end], counts[train], within_u, noise_yu, ridge)
            residuals[held, begin:end] = observations[held, begin:end] - u[held] @ mapping.T - intercept
            a = u.new_zeros(end-begin, visual_dimension)
            if block['kind'] == 'visual':
                a[:, begin:end] = torch.eye(end-begin, device=u.device, dtype=u.dtype)
            a[:, :target_r] -= mapping
            noise_residual[held, begin:end] = a
            traces.append({'block': block['name'], 'fold': fold,
                           'train_class_ids': ids[train].cpu().tolist(),
                           'heldout_class_ids': ids[held].cpu().tolist()})
    q = u.new_zeros(dimension, dimension)
    for first in blocks[1:]:
        i = slice(first['begin'], first['end'])
        for second in blocks[1:]:
            j = slice(second['begin'], second['end'])
            paired = observed_columns[:, first['begin']] & observed_columns[:, second['begin']]
            n = int(paired.sum())
            if n < 2:
                continue
            left, right = residuals[paired, i], residuals[paired, j]
            empirical = (left-left.mean(0)).T @ (right-right.mean(0)) / (n-1)
            # The crossfit maps vary by fold; subtract their average paired
            # target/measurement mean-noise contribution as an approximation.
            a, b = noise_residual[paired, i], noise_residual[paired, j]
            correction = torch.einsum('cir,rs,cjs,c->ij', a, within, b, counts[paired].reciprocal()) / n
            q[i, j] = empirical - correction
    q_cross_scale = 1.
    if dimension > target_r:
        q[target_r:, target_r:], q_cross_scale = _measurement_covariance(
            q[target_r:, target_r:],
            [(b['begin']-target_r, b['end']-target_r) for b in blocks[1:]], shrinkage)
    # Numerical solver jitter is intentionally not stored in Q or W.
    state = {'format': FORMAT, 'basis': blocks[0]['basis'], 'blocks': blocks,
             'H': h.detach(), 'bias': bias.detach(), 'mean': mean.detach(),
             'between': between.detach(), 'Q': q.detach(), 'W': w.detach(),
             'target_rank': target_r, 'target_dimension': visual['final'].shape[1],
             'visual_dimension': visual_dimension, 'observation_dimension': dimension,
             'ridge': ridge, 'shrinkage': shrinkage, 'solver_ridge': solver_ridge,
             'prior_covariance_floor': prior_floor,
             'solver_regularization': 'relative_per_measurement_diagonal',
             'covariance_regularization': 'block_local_diagonal_targets_psd_cross_attenuation',
             'W_cross_block_scale': w_cross_scale, 'Q_cross_block_scale': q_cross_scale,
             'visual_aliases': visual_aliases, 'text_aliases': text_aliases,
             'dropped_text_blocks': dropped, 'fit_class_ids': ids.detach(),
             'fit_class_counts': counts.long().detach(), 'fit_sample_count': int(counts.sum()),
             'crossfit_trace': traces, 'crossfit_folds': crossfit_folds,
             'text_available_class_counts': {k: int(v.sum()) for k,v in masks.items()},
             'finite_mean_correction': 'shared_within_covariance_errors_in_variables_and_crossfit_residual_noise_approximation',
             'new_gradient_training': False, 'retained_training_samples': 0}
    return state


def _prediction_inputs(state, visual_blocks, labels, text_blocks, class_ids, text_masks, omitted_text):
    if not isinstance(state, dict) or state.get('format') != FORMAT:
        raise ValueError('Invalid multi-evidence model format.')
    if not isinstance(visual_blocks, dict) or 'final' not in visual_blocks or not isinstance(text_blocks, dict):
        raise ValueError('Prediction needs visual_blocks including final and a text dictionary.')
    final = _real(visual_blocks['final'], 'final support')
    if final.ndim != 2 or final.shape[1] != state['target_dimension']:
        raise ValueError('Final support feature dimension does not match the model.')
    y, ids = _ids(labels, 'labels', final.device), _ids(class_ids, 'class_ids', final.device)
    if len(y) != len(final) or ids.unique().numel() != len(ids):
        raise ValueError('Prediction labels/classes are not aligned or unique.')
    groups = [torch.where(y == c)[0] for c in ids]
    if any(not len(group) for group in groups):
        raise ValueError('Every predicted class needs support samples.')
    selected_rows = torch.cat(groups)
    visuals = {k: _real(v, 'support '+k, final.device) for k,v in visual_blocks.items()}
    for key, values in visuals.items():
        if values.ndim != 2 or len(values) != len(final):
            raise ValueError('Visual support blocks must share sample rows.')
    texts, masks = {}, {}
    for key, values in text_blocks.items():
        texts[key] = _real(values, 'current text '+key, final.device)
        if texts[key].ndim != 2 or len(texts[key]) != len(ids):
            raise ValueError('Current text blocks must have [classes,D] shape.')
        masks[key] = _mask((text_masks or {}).get(key), len(ids), 'current mask '+key, final.device)
    if set(text_masks or {}) - set(texts):
        raise ValueError('Current text mask names an unknown text block.')
    if isinstance(omitted_text, str):
        raise ValueError('omitted_text must be a sequence of block names, not a string.')
    omitted = set(omitted_text or [])
    known_text = {b['name'] for b in state['blocks'] if b['kind'] == 'text'}
    known_text.update(state['text_aliases'])
    known_text.update(state['dropped_text_blocks'])
    if omitted - known_text:
        raise ValueError('omitted_text names an unknown block.')
    for alias, original in state['visual_aliases'].items():
        if (alias not in visuals or original not in visuals
                or visuals[alias].shape != visuals[original].shape
                or not torch.equal(visuals[alias][selected_rows], visuals[original][selected_rows])):
            raise ValueError('A duplicate visual block no longer matches: ' + alias)
    for alias, original in state['text_aliases'].items():
        if alias in omitted or original in omitted:
            continue
        if alias not in texts or original not in texts or not torch.equal(masks[alias], masks[original]):
            raise ValueError('A duplicate text block/mask no longer matches: ' + alias)
        if not torch.equal(texts[alias][masks[alias]], texts[original][masks[original]]):
            raise ValueError('A duplicate text block no longer matches: ' + alias)
    return final, visuals, texts, masks, ids, groups, omitted


@torch.no_grad()
def predict_model(state, visual_blocks, labels, text_blocks, class_ids, *,
                  omitted_text=None, text_masks=None):
    """Return raw final-space means without retaining support or using queries."""
    final, visuals, texts, masks, ids, groups, omitted = _prediction_inputs(
        state, visual_blocks, labels, text_blocks, class_ids, text_masks, omitted_text)
    dtype = torch.as_tensor(visual_blocks['final']).dtype
    dtype = dtype if dtype in (torch.float32, torch.float64) else torch.float32
    device = final.device
    basis = _real(state['basis'], 'state basis', device)
    arrays = {k: _real(state[k], 'state '+k, device) for k in ('H','bias','mean','between','Q','W')}
    h, bias, mean, between, q, w = (arrays[k] for k in ('H','bias','mean','between','Q','W'))
    r, d = state['target_rank'], state['observation_dimension']
    if (basis.shape != (state['target_dimension'], r) or h.shape != (d,r)
            or bias.shape != (d,) or mean.shape != (r,) or between.shape != (r,r)
            or q.shape != (d,d) or w.shape != (d,d)):
        raise ValueError('Multi-evidence state has inconsistent shapes.')
    identity = torch.eye(r, device=device, dtype=basis.dtype)
    if not torch.allclose(basis.T @ basis, identity, atol=1e-6, rtol=1e-6):
        raise ValueError('Final basis must have orthonormal columns.')
    if not torch.equal(h[:r], identity) or torch.any(bias[:r] != 0):
        raise ValueError('Final measurement must remain an identity without class-level bias.')
    for name, covariance in (('between', between), ('Q', q), ('W', w)):
        if (not torch.allclose(covariance, covariance.T, atol=1e-8, rtol=1e-5)
                or torch.linalg.eigvalsh(covariance).min() < -1e-8):
            raise ValueError('State covariance must be symmetric positive semidefinite: '+name)
    if torch.any(q[:r] != 0) or torch.any(q[:, :r] != 0):
        raise ValueError('Class-level Q must have exact zero final rows and columns.')
    if torch.any(w[state['visual_dimension']:] != 0) or torch.any(w[:,state['visual_dimension']:] != 0):
        raise ValueError('Sampling W must have exact zero text rows and columns.')
    observations, active_columns = [], []
    for block in state['blocks']:
        name, begin, end = block['name'], block['begin'], block['end']
        projection = _real(block['basis'], 'block basis '+name, device)
        if block['kind'] == 'visual':
            if name not in visuals or visuals[name].shape[1] != block['input_dimension']:
                raise ValueError('Missing or incompatible native visual block: '+name)
            observations.append(torch.stack([visuals[name][g].mean(0) for g in groups]) @ projection)
            active_columns.append(torch.ones(len(ids), end-begin, dtype=torch.bool, device=device))
        else:
            alternatives = [name] + [alias for alias, canonical in state['text_aliases'].items() if canonical == name]
            values = final.new_zeros(len(ids), block['input_dimension'])
            active = torch.zeros(len(ids), dtype=torch.bool, device=device)
            for key in alternatives:
                if key in omitted or key not in texts:
                    continue
                if texts[key].shape[1] != block['input_dimension']:
                    raise ValueError('Text feature dimension does not match: '+key)
                chosen = masks[key] & ~active
                values[chosen] = texts[key][chosen]
                active |= masks[key]
            observations.append(values @ projection)
            active_columns.append(active[:,None].expand(-1,end-begin))
    observed, active = torch.cat(observations,1), torch.cat(active_columns,1)
    results = []
    for j, group in enumerate(groups):
        selected = torch.where(active[j])[0]
        mapping = h[selected]
        covariance = q[selected][:,selected] + w[selected][:,selected] / len(group)
        system = mapping @ between @ mapping.T + covariance
        if state['solver_ridge']:
            # A very uncertain auxiliary/text measurement must not increase
            # artificial noise in the exact final measurement. Use each row's
            # own scale rather than one trace over heterogeneous measurements.
            jitter = state['solver_ridge'] * system.diagonal().clamp_min(1e-12)
            system = system + torch.diag(jitter)
        difference = observed[j,selected] - mapping @ mean - bias[selected]
        try:
            posterior = mean + between @ mapping.T @ torch.linalg.solve(system, difference)
        except RuntimeError as error:
            raise ValueError('Multi-evidence posterior system is singular; increase solver_ridge.') from error
        raw = final[group].mean(0)
        results.append(raw + (posterior - raw @ basis) @ basis.T)
    result = torch.stack(results).to(dtype)
    if not torch.isfinite(result).all():
        raise ValueError('Multi-evidence completion produced non-finite prototypes.')
    return result.detach()
