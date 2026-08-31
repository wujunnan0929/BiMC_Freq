"""Frozen-feature low-rank class corrections for few-shot incremental learning.

The reference branch supplies nonnegative votes, not logits. This module adds a
bounded correction to their logarithm. Real sessions optimize only newly added
class codes; the dictionary and historical codes are checkpointed buffers.
"""

from typing import Callable, Dict, Optional, Sequence, Union

import torch
from torch import nn
from torch.nn import functional as F


ClassIds = Union[Sequence[int], torch.Tensor]


def _integer_vector(values, device, name: str) -> torch.Tensor:
    result = torch.as_tensor(values, device=device)
    if result.ndim != 1:
        raise ValueError("{} must be one-dimensional".format(name))
    if result.is_floating_point() and (
        not torch.isfinite(result).all() or not torch.equal(result, result.round())
    ):
        raise ValueError("{} must contain integer class ids".format(name))
    return result.long()


def _local_labels(labels: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
    matches = labels[:, None].eq(class_ids[None, :])
    if not matches.any(dim=1).all():
        raise ValueError("Labels contain a class absent from the score columns")
    return matches.long().argmax(dim=1)


def _log_votes(scores, rows: int, columns: int, device, dtype) -> torch.Tensor:
    values = torch.as_tensor(scores, device=device, dtype=dtype).detach()
    if values.shape != (rows, columns):
        raise ValueError("Reference scores must have shape [{}, {}]".format(rows, columns))
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("Reference scores must be finite, nonnegative votes")
    return values.clamp_min(1e-8).log()


def _bounded_residual(features, dictionary, codes, max_delta, gain):
    raw = F.normalize(features, dim=-1) @ dictionary @ codes.t()
    return gain * max_delta * torch.tanh(raw / max_delta)


def _old_new_margin(logits, labels, new_columns, margin):
    if logits.shape[0] == 0 or new_columns.numel() == 0:
        return logits.sum() * 0.0
    correct = logits.gather(1, labels[:, None]).squeeze(1)
    strongest_new = logits.index_select(1, new_columns).max(dim=1).values
    return F.softplus(margin + strongest_new - correct).mean()


def _differentiable_sgd_step(loss, codes, learning_rate, grad_clip=0.0):
    """Keep the adaptation derivative in the outer dictionary gradient."""
    gradient = torch.autograd.grad(loss, codes, create_graph=True)[0]
    if grad_clip:
        gradient = gradient * (grad_clip / gradient.norm().clamp_min(1e-12)).clamp(max=1.0)
    return codes - learning_rate * gradient


def sample_residual_episode(
    labels: torch.Tensor,
    old_way: int,
    new_way: int,
    shot: int,
    query_shot: int,
    old_shot: int = 20,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """Sample disjoint support/query indices using an independent CPU RNG.

    Selected global class ids are ordered old first, then new. The returned
    support/query labels are *local score-column indices*. Every chosen class
    must have enough examples to serve either requested role.
    """
    if old_way < 0 or new_way < 1 or min(shot, query_shot, old_shot) < 1:
        raise ValueError("Episode ways/shots must be positive (old_way may be zero)")
    labels = _integer_vector(labels, labels.device, "labels")
    cpu_labels = labels.detach().cpu()
    classes, counts = torch.unique(cpu_labels, sorted=True, return_counts=True)
    required = max(shot, old_shot if old_way else shot) + query_shot
    eligible = classes[counts >= required]
    if eligible.numel() < old_way + new_way:
        raise ValueError("Not enough classes with disjoint support/query samples")
    if generator is None:
        generator = torch.Generator(device="cpu").manual_seed(0)
    selected = eligible[torch.randperm(eligible.numel(), generator=generator)[:old_way + new_way]]
    support_indices, query_indices, support_labels, query_labels = [], [], [], []
    for column, class_id in enumerate(selected.tolist()):
        indices = torch.where(cpu_labels == class_id)[0]
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        count = old_shot if column < old_way else shot
        support_indices.append(indices[:count])
        query_indices.append(indices[count:count + query_shot])
        support_labels.append(torch.full((count,), column, dtype=torch.long))
        query_labels.append(torch.full((query_shot,), column, dtype=torch.long))
    return {
        "selected_class_ids": selected.to(labels.device),
        "support_indices": torch.cat(support_indices).to(labels.device),
        "query_indices": torch.cat(query_indices).to(labels.device),
        "support_labels": torch.cat(support_labels).to(labels.device),
        "query_labels": torch.cat(query_labels).to(labels.device),
    }


class LowRankResidualHead(nn.Module):
    """A frozen dictionary and append-only class-code bank.

    Args:
        feature_dim: Dimension of the frozen original-view features.
        num_classes: Capacity for global class ids ``0 .. num_classes - 1``.
        rank: Shared dictionary width, at most ``feature_dim``.
        max_delta: Maximum additive correction in log-vote units.
        gain: Fixed ablation multiplier in ``[0, 1]``.

    ``state_dict`` contains the dictionary, all codes, seen mask, and correction
    scale. Instantiate the same dimensions before restoring a checkpoint.
    """

    def __init__(self, feature_dim: int, num_classes: int, rank: int,
                 max_delta: float = 0.2, gain: float = 1.0):
        super().__init__()
        if feature_dim < 1 or num_classes < 1 or not 1 <= rank <= feature_dim:
            raise ValueError("Require positive dimensions and 1 <= rank <= feature_dim")
        if not 0.0 < max_delta < float("inf") or not 0.0 <= gain <= 1.0:
            raise ValueError("max_delta must be positive/finite and gain in [0, 1]")
        self.feature_dim, self.num_classes, self.rank = feature_dim, num_classes, rank
        generator = torch.Generator(device="cpu").manual_seed(0)
        initial = torch.linalg.qr(torch.randn(feature_dim, rank, generator=generator), mode="reduced").Q
        self.register_buffer("dictionary", initial)
        self.register_buffer("codes", torch.zeros(num_classes, rank))
        self.register_buffer("seen_mask", torch.zeros(num_classes, dtype=torch.bool))
        self.register_buffer("_max_delta", torch.tensor(float(max_delta)))
        self.register_buffer("_gain", torch.tensor(float(gain)))

    @property
    def U(self):
        return self.dictionary

    @property
    def max_delta(self) -> float:
        return float(self._max_delta.item())

    @property
    def gain(self) -> float:
        return float(self._gain.item())

    def _class_ids(self, values: Optional[ClassIds]) -> torch.Tensor:
        if values is None:
            return torch.arange(self.num_classes, device=self.dictionary.device)
        ids = _integer_vector(values, self.dictionary.device, "class_ids")
        if (ids < 0).any() or (ids >= self.num_classes).any():
            raise ValueError("Class ids exceed residual-head capacity")
        if ids.unique().numel() != ids.numel():
            raise ValueError("Class ids / score columns must be unique")
        return ids

    def _features(self, values, detach: bool = False) -> torch.Tensor:
        values = torch.as_tensor(values, device=self.dictionary.device, dtype=self.dictionary.dtype)
        if values.ndim != 2 or values.shape[1] != self.feature_dim:
            raise ValueError("Features must have shape [N, feature_dim]")
        if not torch.isfinite(values).all():
            raise ValueError("Features must be finite")
        return values.detach() if detach else values

    @torch.no_grad()
    def mark_seen(self, class_ids: ClassIds):
        """Register base classes without changing their zero correction codes."""
        self.seen_mask[self._class_ids(class_ids)] = True

    def forward_residual(self, features, class_ids: Optional[ClassIds] = None):
        features = self._features(features)
        ids = self._class_ids(class_ids)
        return _bounded_residual(features, self.dictionary, self.codes[ids], self._max_delta, self._gain)

    def forward_scores(self, features, reference_scores,
                       class_ids: Optional[ClassIds] = None, temperature: float = 1.0):
        if not temperature > 0.0:
            raise ValueError("temperature must be positive")
        features = self._features(features)
        ids = self._class_ids(class_ids)
        reference = _log_votes(reference_scores, len(features), len(ids), features.device, features.dtype)
        return (reference + self.forward_residual(features, ids)) / temperature

    forward = forward_scores

    @torch.no_grad()
    def initialize_dictionary(
        self, features=None, labels=None, method: str = "random", shot: int = 5,
        repeats: int = 20, seed: int = 0, reference_shot: int = 20,
        reference_min: int = 1,
    ) -> Dict[str, Union[int, float, str]]:
        """Initialize with a local random orthobasis, residual SVD, or identity.

        SVD columns are normalized-reference minus normalized-support means.
        Support and reference are disjoint, reference size is explicitly capped
        by ``reference_shot``, and the residual matrix is NOT centered. If its
        numerical rank is smaller than the requested rank, independent random
        orthogonal directions complete the dictionary; this is logged.
        """
        if self.seen_mask.any() or self.codes.count_nonzero():
            raise ValueError("Initialize the dictionary before registering/learning classes")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        if method not in {"random", "residual_svd", "identity"}:
            raise ValueError("Unknown dictionary initialization: {}".format(method))
        residual_count, effective_rank, eligible_classes = 0, self.rank, 0
        captured_energy = 0.0
        if method == "identity":
            if self.rank != self.feature_dim:
                raise ValueError("Identity control requires rank == feature_dim")
            value = torch.eye(self.feature_dim, device=self.dictionary.device, dtype=self.dictionary.dtype)
        elif method == "random":
            value = torch.randn(self.feature_dim, self.rank, generator=generator).to(self.dictionary)
            value = torch.linalg.qr(value, mode="reduced").Q
        else:
            if min(shot, repeats, reference_shot, reference_min) < 1 or reference_min > reference_shot:
                raise ValueError("SVD shots/repeats must be positive; reference_min <= reference_shot")
            values = F.normalize(self._features(features, detach=True), dim=-1)
            targets = _integer_vector(labels, values.device, "labels")
            if len(targets) != len(values):
                raise ValueError("Feature and label lengths differ")
            targets_cpu = targets.cpu()
            residuals = []
            for class_id in torch.unique(targets_cpu).tolist():
                indices = torch.where(targets_cpu == class_id)[0]
                if indices.numel() < shot + reference_min:
                    continue
                eligible_classes += 1
                reference_count = min(reference_shot, indices.numel() - shot)
                for _ in range(repeats):
                    shuffled = indices[torch.randperm(len(indices), generator=generator)].to(values.device)
                    support = F.normalize(values[shuffled[:shot]].mean(dim=0), dim=0)
                    reference = F.normalize(values[shuffled[shot:shot + reference_count]].mean(dim=0), dim=0)
                    residuals.append(reference - support)
            if not residuals:
                raise ValueError("No base class has enough disjoint support/reference samples")
            matrix = torch.stack(residuals, dim=1)
            left, singular, _ = torch.linalg.svd(matrix, full_matrices=False)
            tolerance = max(matrix.shape) * torch.finfo(matrix.dtype).eps * singular.max()
            effective_rank = min(self.rank, int((singular > tolerance).sum().item()))
            residual_count = matrix.shape[1]
            total_energy = singular.square().sum()
            captured_energy = float((singular[:self.rank].square().sum() / total_energy.clamp_min(1e-12)).item())
            principal = left[:, :effective_rank]
            filler = torch.randn(self.feature_dim, self.rank, generator=generator).to(values)
            # QR on [principal, random] preserves the leading residual subspace.
            value = torch.linalg.qr(torch.cat([principal, filler], dim=1), mode="reduced").Q[:, :self.rank]
        self.dictionary.copy_(value)
        return {
            "method": method, "rank": self.rank, "effective_rank": effective_rank,
            "residual_count": residual_count, "eligible_classes": eligible_classes,
            "captured_energy": captured_energy, "reference_shot": reference_shot,
        }

    def fit_session(
        self, features, labels, reference_scores, seen_class_ids: ClassIds,
        new_class_ids: ClassIds, anchor_features=None, anchor_reference_scores=None,
        anchor_labels=None, steps: int = 100, lr: float = 0.05, l2: float = 0.01,
        old_margin: float = 0.0, old_weight: float = 1.0, optimizer: str = "adam",
        seed: int = 0, grad_clip: float = 5.0, temperature: float = 1.0,
    ) -> Dict[str, Union[int, float, list]]:
        """Fit only current new-class coefficients from their support examples.

        Labels are global ids. Reference columns follow ``seen_class_ids``.
        All previous ids must already be marked seen; new ids must not be seen.
        Optional anchors must be old classes and use the same all-seen score
        columns. Inputs are detached so gradients never reach their encoders.
        ``seed`` is accepted for experiment bookkeeping; optimization is full
        batch and does not consume any global RNG state.
        """
        del seed
        if steps < 0 or lr <= 0 or min(l2, old_weight, grad_clip) < 0 or temperature <= 0:
            raise ValueError("Invalid residual optimization hyperparameters")
        if optimizer.lower() not in {"adam", "sgd"}:
            raise ValueError("optimizer must be adam or sgd")
        values = self._features(features, detach=True)
        targets = _integer_vector(labels, values.device, "labels")
        seen, new = self._class_ids(seen_class_ids), self._class_ids(new_class_ids)
        if len(values) == 0 or len(values) != len(targets) or len(new) == 0:
            raise ValueError("A session requires nonempty, aligned new-class support")
        new_columns = _local_labels(new, seen)
        target_columns = _local_labels(targets, seen)
        _local_labels(targets, new)
        if not torch.equal(targets.unique().sort().values, new.sort().values):
            raise ValueError("Every new class must have support examples")
        if self.seen_mask[new].any():
            raise ValueError("A previously seen class cannot be optimized again")
        historical = torch.where(self.seen_mask)[0]
        if historical.numel() and not historical[:, None].eq(seen[None, :]).any(dim=1).all():
            raise ValueError("Score columns must include every previously seen class")
        old_columns = torch.where(~seen[:, None].eq(new[None, :]).any(dim=1))[0]
        if not self.seen_mask[seen[old_columns]].all():
            raise ValueError("Register base/previous classes with mark_seen before fitting")
        reference = _log_votes(reference_scores, len(values), len(seen), values.device, values.dtype)
        anchor_values, anchor_reference, anchor_targets = self._prepare_anchors(
            anchor_features, anchor_reference_scores, anchor_labels, seen, new
        )
        frozen_codes = self.codes[seen].detach().clone()
        frozen_codes[new_columns] = 0
        dictionary = self.dictionary.detach()
        learned = nn.Parameter(torch.zeros(len(new), self.rank, device=values.device, dtype=values.dtype))
        # index_copy produces a differentiable new tensor; no optimizer ever
        # receives the persistent dictionary or a historical class-code row.
        def objective():
            merged = frozen_codes.index_copy(0, new_columns, learned)
            logits = (reference + _bounded_residual(values, dictionary, merged, self._max_delta, self._gain)) / temperature
            ce = F.cross_entropy(logits, target_columns)
            regularizer = learned.square().sum(dim=1).mean()
            old_loss = ce.new_zeros(())
            if anchor_values is not None:
                anchor_logits = (anchor_reference + _bounded_residual(anchor_values, dictionary, merged, self._max_delta, self._gain)) / temperature
                old_loss = _old_new_margin(anchor_logits, anchor_targets, new_columns, old_margin)
            total = ce + l2 * regularizer + old_weight * old_loss
            return total, ce, old_loss, regularizer
        with torch.enable_grad():
            opt = torch.optim.Adam([learned], lr=lr) if optimizer.lower() == "adam" else torch.optim.SGD([learned], lr=lr)
            history = []
            initial = float(objective()[0].detach().item())
            actual_steps = steps if self.gain > 0 else 0
            for _ in range(actual_steps):
                opt.zero_grad(set_to_none=True)
                total, _, _, _ = objective()
                if not torch.isfinite(total):
                    raise FloatingPointError("Non-finite incremental residual objective")
                total.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_([learned], grad_clip)
                opt.step()
                history.append(float(total.detach().item()))
            total, ce, old_loss, regularizer = objective()
        if not torch.isfinite(learned).all() or not torch.isfinite(total):
            raise FloatingPointError("Non-finite incremental residual coefficients")
        with torch.no_grad():
            self.codes.index_copy_(0, new, learned.detach())
            self.seen_mask[new] = True
        return {
            "steps": actual_steps, "new_classes": len(new), "trainable_parameters": learned.numel(),
            "initial_loss": initial, "final_loss": float(total.detach().item()),
            "ce": float(ce.detach().item()), "old_loss": float(old_loss.detach().item()),
            "code_l2": float(regularizer.detach().item()), "loss_history": history,
        }

    def _prepare_anchors(self, features, scores, labels, seen, new):
        supplied = [features is not None, scores is not None, labels is not None]
        if not any(supplied):
            return None, None, None
        if not all(supplied):
            raise ValueError("Anchor features, scores and labels must be supplied together")
        values = self._features(features, detach=True)
        targets = _integer_vector(labels, values.device, "anchor_labels")
        if len(targets) != len(values):
            raise ValueError("Anchor feature and label lengths differ")
        if targets[:, None].eq(new[None, :]).any():
            raise ValueError("Safety anchors must belong to old classes")
        columns = _local_labels(targets, seen)
        reference = _log_votes(scores, len(values), len(seen), values.device, values.dtype)
        return values, reference, columns

    def refine_dictionary(
        self, base_features, base_labels, reference_callback: Optional[Callable] = None,
        episodes: int = 100, old_way: int = 5, new_way: int = 5, shot: int = 5,
        query_shot: int = 5, old_shot: int = 20, inner_steps: int = 3,
        inner_lr: float = 0.1, outer_lr: float = 0.001, l2: float = 0.01,
        old_margin: float = 0.0, old_weight: float = 1.0, orth_weight: float = 0.01,
        seed: int = 0, grad_clip: float = 5.0, temperature: float = 1.0,
        surrogate_scale: float = 10.0,
    ) -> Dict[str, Union[int, float, str, list]]:
        """Learn U using genuinely differentiable support adaptation and query loss.

        The optional callback receives ``(support_indices, query_indices,
        selected_class_ids, old_way)``. Indices address ``base_features`` and
        selected GLOBAL ids give the exact old-first/new-last column order.
        Return ``support_scores`` and ``query_scores`` as nonnegative votes;
        optionally supply ALL of ``anchor_features``, ``anchor_scores`` and
        ``anchor_labels``. Anchor labels are LOCAL score-column indices. The
        callback must build statistics from support only, never query examples.

        With no callback, normalized visual prototypes form an explicitly
        labelled surrogate reference; old support means supply safety anchors.
        Only pseudo-new support enters inner CE. Outer CE uses disjoint old+new
        query and an additional old-query/new-competition loss. The inner SGD
        updates use ``create_graph=True`` so U receives adaptation gradients.
        Persistent codes/seen_mask are never changed by this method.
        """
        if self.seen_mask.any() or self.codes.count_nonzero():
            raise ValueError("Refine the dictionary before registering/learning classes")
        if episodes < 0 or inner_steps < 1 or min(inner_lr, outer_lr, temperature, surrogate_scale) <= 0:
            raise ValueError("Invalid episodic optimization hyperparameters")
        if min(l2, old_weight, orth_weight, grad_clip) < 0:
            raise ValueError("Loss weights and gradient clipping must be nonnegative")
        values = self._features(base_features, detach=True)
        labels = _integer_vector(base_labels, values.device, "base_labels")
        if len(values) != len(labels):
            raise ValueError("Feature and label lengths differ")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        dictionary = nn.Parameter(self.dictionary.detach().clone())
        initial_dictionary = dictionary.detach().clone()
        optimizer = torch.optim.Adam([dictionary], lr=outer_lr)
        history = []
        selected_columns = torch.arange(old_way + new_way, device=values.device)
        new_columns = selected_columns[old_way:]
        with torch.enable_grad():
            for episode_index in range(episodes):
                sample = sample_residual_episode(labels, old_way, new_way, shot, query_shot, old_shot, generator)
                si, qi = sample["support_indices"], sample["query_indices"]
                sy, qy = sample["support_labels"], sample["query_labels"]
                support, query = values[si], values[qi]
                if reference_callback is None:
                    prototypes = torch.stack([support[sy == col].mean(dim=0) for col in selected_columns])
                    prototypes = F.normalize(prototypes, dim=-1)
                    def visual_votes(x):
                        return (surrogate_scale * F.normalize(x, dim=-1) @ prototypes.t()).softmax(dim=-1)
                    refs = {"support_scores": visual_votes(support), "query_scores": visual_votes(query)}
                    if old_way:
                        refs.update(anchor_features=prototypes[:old_way],
                                    anchor_scores=visual_votes(prototypes[:old_way]),
                                    anchor_labels=selected_columns[:old_way])
                else:
                    with torch.no_grad():
                        refs = reference_callback(si, qi, sample["selected_class_ids"], old_way)
                    if not isinstance(refs, dict) or not {"support_scores", "query_scores"}.issubset(refs):
                        raise ValueError("Reference callback must return support_scores and query_scores")
                support_reference = _log_votes(refs["support_scores"], len(si), len(selected_columns), values.device, values.dtype)
                query_reference = _log_votes(refs["query_scores"], len(qi), len(selected_columns), values.device, values.dtype)
                anchors, anchor_reference, anchor_targets = self._prepare_anchors(
                    refs.get("anchor_features"), refs.get("anchor_scores"), refs.get("anchor_labels"),
                    selected_columns, new_columns,
                )
                new_support = sy >= old_way
                current = torch.zeros(new_way, self.rank, device=values.device, dtype=values.dtype, requires_grad=True)
                old_codes = values.new_zeros((old_way, self.rank))
                for _ in range(inner_steps):
                    merged = torch.cat([old_codes, current], dim=0)
                    support_logits = (support_reference[new_support] + _bounded_residual(support[new_support], dictionary, merged, self._max_delta, self._gain)) / temperature
                    inner_loss = F.cross_entropy(support_logits, sy[new_support]) + l2 * current.square().sum(dim=1).mean()
                    if anchors is not None:
                        anchor_logits = (anchor_reference + _bounded_residual(anchors, dictionary, merged, self._max_delta, self._gain)) / temperature
                        inner_loss = inner_loss + old_weight * _old_new_margin(anchor_logits, anchor_targets, new_columns, old_margin)
                    current = _differentiable_sgd_step(inner_loss, current, inner_lr, grad_clip)
                merged = torch.cat([old_codes, current], dim=0)
                query_logits = (query_reference + _bounded_residual(query, dictionary, merged, self._max_delta, self._gain)) / temperature
                query_ce = F.cross_entropy(query_logits, qy)
                old_loss = _old_new_margin(query_logits[qy < old_way], qy[qy < old_way], new_columns, old_margin)
                orth = (dictionary.t() @ dictionary - torch.eye(self.rank, device=values.device, dtype=values.dtype)).square().sum() / self.rank
                outer_loss = query_ce + old_weight * old_loss + orth_weight * orth
                if not torch.isfinite(outer_loss):
                    raise FloatingPointError("Non-finite episodic dictionary objective")
                optimizer.zero_grad(set_to_none=True)
                outer_loss.backward()
                gradient_norm = float(dictionary.grad.detach().norm().item())
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_([dictionary], grad_clip)
                optimizer.step()
                if not torch.isfinite(dictionary).all():
                    raise FloatingPointError("Non-finite episodic dictionary")
                history.append({"episode": episode_index, "loss": float(outer_loss.detach().item()),
                                "query_ce": float(query_ce.detach().item()), "old_loss": float(old_loss.detach().item()),
                                "orth_loss": float(orth.detach().item()), "gradient_norm": gradient_norm})
        with torch.no_grad():
            self.dictionary.copy_(dictionary.detach())
        return {"episodes": episodes, "reference": "callback" if reference_callback else "visual_prototype_surrogate",
                "dictionary_change": float((dictionary.detach() - initial_dictionary).norm().item()),
                "loss_history": history, "inner_steps": inner_steps}


FrozenResidualHead = LowRankResidualHead
