"""Trainable routing utilities for frequency-residual FSCIL.

The router consumes label-free statistics computed from the original and
per-band logits.  It predicts one null expert and one expert per frequency
band.  The null expert is important: it gives the model an explicit way to
leave a confident spatial prediction unchanged.
"""

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalized_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    num_classes = probabilities.shape[-1]
    if num_classes <= 1:
        return probabilities.new_zeros(probabilities.shape[0])
    entropy = -torch.sum(
        probabilities * probabilities.clamp_min(1e-8).log(), dim=-1
    )
    normalizer = probabilities.new_tensor(float(num_classes)).log()
    return entropy / normalizer


def _probability_margin(probabilities: torch.Tensor) -> torch.Tensor:
    if probabilities.shape[-1] <= 1:
        return probabilities.new_ones(probabilities.shape[0])
    top_two = probabilities.topk(2, dim=-1).values
    return top_two[:, 0] - top_two[:, 1]


def build_frequency_router_features(
    original_logits: torch.Tensor,
    band_logits: torch.Tensor,
) -> torch.Tensor:
    """Build query-level, label-free routing features.

    Args:
        original_logits: Spatial logits with shape ``[N, C]``.
        band_logits: Per-band logits with shape ``[N, C, B]``.

    Returns:
        A float32 feature tensor with shape ``[N, 3 + 5 * B]``.  For three
        bands this is a compact 18-dimensional routing descriptor.
    """
    if original_logits.ndim != 2:
        raise ValueError("Original logits must have shape [N, C].")
    if band_logits.ndim != 3:
        raise ValueError("Band logits must have shape [N, C, B].")
    if band_logits.shape[:2] != original_logits.shape:
        raise ValueError("Original and band query/class dimensions must match.")
    if band_logits.shape[-1] <= 0:
        raise ValueError("At least one frequency band is required.")

    original = original_logits.float()
    bands = band_logits.float()
    original_prob = F.softmax(original, dim=-1)
    band_prob = F.softmax(bands, dim=1)

    original_entropy = _normalized_entropy(original_prob).unsqueeze(-1)
    original_margin = _probability_margin(original_prob).unsqueeze(-1)
    original_confidence = original_prob.max(dim=-1).values
    original_prediction = original_prob.argmax(dim=-1)

    band_entropies = []
    band_margins = []
    prediction_agreements = []
    confidence_deltas = []
    logit_similarities = []
    band_predictions = []
    for band_id in range(bands.shape[-1]):
        current_prob = band_prob[:, :, band_id]
        current_logits = bands[:, :, band_id]
        current_prediction = current_prob.argmax(dim=-1)
        band_predictions.append(current_prediction)
        band_entropies.append(_normalized_entropy(current_prob))
        band_margins.append(_probability_margin(current_prob))
        prediction_agreements.append(
            (current_prediction == original_prediction).float()
        )
        confidence_deltas.append(
            current_prob.max(dim=-1).values - original_confidence
        )
        logit_similarities.append(
            F.cosine_similarity(original, current_logits, dim=-1, eps=1e-8)
        )

    if len(band_predictions) == 1:
        consensus = original.new_ones(original.shape[0])
    else:
        pairwise = []
        for left in range(len(band_predictions)):
            for right in range(left + 1, len(band_predictions)):
                pairwise.append(
                    (band_predictions[left] == band_predictions[right]).float()
                )
        consensus = torch.stack(pairwise, dim=-1).mean(dim=-1)

    feature_groups = [
        original_entropy,
        original_margin,
        torch.stack(band_entropies, dim=-1),
        torch.stack(band_margins, dim=-1),
        torch.stack(prediction_agreements, dim=-1),
        torch.stack(confidence_deltas, dim=-1),
        torch.stack(logit_similarities, dim=-1),
        consensus.unsqueeze(-1),
    ]
    features = torch.cat(feature_groups, dim=-1)
    if not torch.isfinite(features).all():
        raise RuntimeError("Frequency router features contain non-finite values.")
    return features


class TrainableFrequencyRouter(nn.Module):
    """A small MLP producing null/low/middle/high expert probabilities."""

    def __init__(
        self,
        num_bands: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        null_logit_bias: float = 2.0,
    ) -> None:
        super().__init__()
        if num_bands <= 0:
            raise ValueError("The router requires at least one frequency band.")
        if hidden_dim <= 0:
            raise ValueError("Router hidden_dim must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Router dropout must lie in [0, 1).")
        self.num_bands = int(num_bands)
        self.feature_dim = 3 + 5 * self.num_bands
        self.network = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_bands + 1),
        )
        final_layer = self.network[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)
        with torch.no_grad():
            final_layer.bias[0] = float(null_logit_bias)

    def forward(
        self,
        original_logits: torch.Tensor,
        band_logits: torch.Tensor,
    ) -> torch.Tensor:
        features = build_frequency_router_features(
            original_logits, band_logits
        )
        return F.softmax(self.network(features), dim=-1)


def residual_frequency_fusion(
    original_logits: torch.Tensor,
    band_logits: torch.Tensor,
    router_probabilities: torch.Tensor,
    class_band_weights: Optional[torch.Tensor] = None,
    max_alpha: float = 1.0,
    class_alpha: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Add a query- and class-conditioned frequency residual to spatial logits.

    ``router_probabilities[:, 0]`` is the null expert.  The remaining mass
    controls the residual strength, while its relative distribution and the
    class-specific band prior decide which frequency evidence is used.
    """
    if original_logits.ndim != 2:
        raise ValueError("Original logits must have shape [N, C].")
    if band_logits.ndim != 3 or band_logits.shape[:2] != original_logits.shape:
        raise ValueError("Band logits must have shape [N, C, B].")
    num_queries, num_classes = original_logits.shape
    num_bands = band_logits.shape[-1]
    if router_probabilities.shape != (num_queries, num_bands + 1):
        raise ValueError("Router probabilities must have shape [N, B + 1].")
    if not 0.0 <= max_alpha <= 1.0:
        raise ValueError("max_alpha must lie in [0, 1].")
    if torch.any(router_probabilities < 0.0):
        raise ValueError("Router probabilities must be non-negative.")

    probability_sums = router_probabilities.sum(dim=-1)
    if not torch.allclose(
        probability_sums,
        torch.ones_like(probability_sums),
        atol=1e-4,
        rtol=1e-4,
    ):
        raise ValueError("Router probabilities must sum to one.")

    original = original_logits.float()
    bands = band_logits.float()
    router = router_probabilities.to(device=original.device, dtype=original.dtype)
    query_band_weights = router[:, 1:]
    frequency_mass = query_band_weights.sum(dim=-1, keepdim=True)

    if class_band_weights is None:
        class_weights = original.new_ones(num_classes, num_bands)
    else:
        if class_band_weights.shape != (num_classes, num_bands):
            raise ValueError("Class band weights must have shape [C, B].")
        if torch.any(class_band_weights < 0.0):
            raise ValueError("Class band weights must be non-negative.")
        class_weights = class_band_weights.to(
            device=original.device, dtype=original.dtype
        )

    joint_weights = (
        query_band_weights[:, None, :] * class_weights[None, :, :]
    )
    joint_weights = joint_weights / joint_weights.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-8)
    residuals = bands - original.unsqueeze(-1)
    residual = torch.sum(joint_weights * residuals, dim=-1)

    strength = frequency_mass * float(max_alpha)
    if class_alpha is not None:
        if class_alpha.shape != (num_classes,):
            raise ValueError("class_alpha must have shape [C].")
        if torch.any(class_alpha < 0.0) or torch.any(class_alpha > 1.0):
            raise ValueError("class_alpha values must lie in [0, 1].")
        strength = strength * class_alpha.to(
            device=original.device, dtype=original.dtype
        ).unsqueeze(0)

    return original + strength * residual


def sample_pseudo_fscil_episode(
    labels: torch.Tensor,
    class_ids: Sequence[int],
    way: int,
    shot: int,
    query: int,
    generator: Optional[torch.Generator] = None,
    old_way: int = 0,
    old_shot: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample disjoint support/query indices from base classes.

    Returns selected positions in ``class_ids``, support indices, support local
    labels, query indices, and query local labels.  Sampling happens on CPU so
    the same seeded generator works regardless of the feature device.
    """
    if labels.ndim != 1:
        raise ValueError("labels must have shape [N].")
    if way < 2 or shot <= 0 or query <= 0:
        raise ValueError("Episode way must be >= 2 and shot/query must be positive.")
    if old_way < 0 or old_way > way:
        raise ValueError("old_way must lie in [0, way].")
    if old_shot is None:
        old_shot = shot
    if old_shot <= 0:
        raise ValueError("old_shot must be positive.")

    labels_cpu = labels.detach().cpu().long()
    normalized_ids = [int(class_id) for class_id in class_ids]
    eligible_positions = []
    class_indices = {}
    maximum_shot = old_shot if old_way > 0 else shot
    required = int(max(shot, maximum_shot) + query)
    for position, class_id in enumerate(normalized_ids):
        indices = torch.nonzero(labels_cpu == class_id, as_tuple=False).flatten()
        if indices.numel() >= required:
            eligible_positions.append(position)
            class_indices[position] = indices

    if len(eligible_positions) < 2:
        raise ValueError(
            "Not enough base classes contain shot + query samples for an episode."
        )
    selected_way = min(int(way), len(eligible_positions))
    class_order = torch.randperm(
        len(eligible_positions), generator=generator
    )[:selected_way]
    selected_positions = torch.tensor(
        [eligible_positions[int(index)] for index in class_order],
        dtype=torch.long,
    )

    support_indices = []
    support_labels = []
    query_indices = []
    query_labels = []
    for local_label, position_tensor in enumerate(selected_positions):
        position = int(position_tensor)
        candidates = class_indices[position]
        class_shot = int(old_shot if local_label < old_way else shot)
        class_required = class_shot + int(query)
        order = torch.randperm(candidates.numel(), generator=generator)
        chosen = candidates[order[:class_required]]
        support_indices.append(chosen[:class_shot])
        query_indices.append(chosen[class_shot:])
        support_labels.append(
            torch.full((class_shot,), local_label, dtype=torch.long)
        )
        query_labels.append(torch.full((query,), local_label, dtype=torch.long))

    return (
        selected_positions,
        torch.cat(support_indices),
        torch.cat(support_labels),
        torch.cat(query_indices),
        torch.cat(query_labels),
    )
