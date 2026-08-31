"""Feature-only FSCIL adaptation, using the same BiMC reference at train/test.

The residual input is the original frozen CLIP feature. Frequency views, when
enabled, belong to the reference scorer. Only class means survive a session;
neither old images nor old individual image features are replayed.
"""

import math

import torch
import torch.nn.functional as F

from models.frequency import (
    calibrate_frequency_prototypes,
    compute_frequency_class_alpha,
    compute_frequency_prototypes,
    select_frequency_description_prototypes,
)
from models.incremental_residual import LowRankResidualHead, sample_residual_episode


def make_reference_scorer(model, cfg, state, num_base_classes):
    """Cache the expensive inverse once for a fixed session/episode reference."""
    inverse = torch.pinverse(state['cov_image'].float())

    @torch.no_grad()
    def score(features, frequency_features=None):
        return model.reference_scores_from_features(
            features,
            len(state['image_proto']),
            num_base_classes,
            state['image_proto'],
            state['cov_image'],
            state['description_proto'],
            state['description_features'],
            state['description_targets'],
            state['text_features'],
            cfg.DATASET.BETA,
            state.get('frequency_calibrated_proto'),
            state.get('frequency_band_weights'),
            state.get('frequency_class_alpha'),
            frequency_features,
            cov_inverse=inverse,
        ).detach()

    return score


def _group_covariance(features, gamma):
    # Same diagonal shrinkage as BiMC (alpha2 is zero there).
    if len(features) < 2:
        return torch.eye(features.shape[-1], device=features.device) * 1e-6
    covariance = torch.cov(features.float().T)
    diagonal_mean = covariance.diagonal().mean()
    return covariance + gamma * diagonal_mean * torch.eye(
        features.shape[-1], device=features.device
    )


def make_episode_reference(model, cfg, features, labels, base_state,
                           frequency_features=None):
    """Build a support-only, full BiMC reference for each pseudo episode.

    Class/text descriptions are reusable side information. All image-dependent
    quantities (including explicit description selection) use support only.
    Input indices refer to the features argument, not the full base cache.
    """
    features = features.detach().float()
    labels = labels.detach().long()
    if frequency_features is not None:
        frequency_features = frequency_features.detach().float()
    base_class_ids = [int(c) for c in base_state['class_index']]
    position_of = {class_id: i for i, class_id in enumerate(base_class_ids)}
    freq_cfg = cfg.TRAINER.BiMC.FREQUENCY

    @torch.no_grad()
    def reference(support_indices, query_indices, selected_class_ids, old_way):
        selected_ids = [int(c) for c in selected_class_ids]
        positions = torch.tensor(
            [position_of[c] for c in selected_ids], device=features.device
        )
        support = features[support_indices]
        query = features[query_indices]
        support_global_labels = labels[support_indices]
        local_labels = torch.empty_like(support_global_labels)
        raw_means = []
        for local, global_id in enumerate(selected_ids):
            mask = support_global_labels == global_id
            local_labels[mask] = local
            raw_means.append(support[mask].mean(0))
        raw_means = torch.stack(raw_means)
        prototypes = F.normalize(raw_means, dim=-1)
        if old_way and cfg.TRAINER.BiMC.VISION_CALIBRATION:
            prototypes = prototypes.clone()
            prototypes[old_way:] = model.soft_calibration(
                prototypes[:old_way], prototypes[old_way:]
            )

        description_features, description_targets = [], []
        for local, global_id in enumerate(selected_ids):
            desc = base_state['description_features'][
                base_state['description_targets'] == global_id
            ].float()
            description_features.append(desc)
            description_targets.append(torch.full(
                (len(desc),), local, device=features.device, dtype=torch.long
            ))
        covariance = features.new_zeros(features.shape[-1], features.shape[-1])
        if old_way:
            covariance += old_way * _group_covariance(
                support[local_labels < old_way], cfg.TRAINER.BiMC.GAMMA_BASE
            )
        new_way = len(selected_ids) - old_way
        covariance += new_way * _group_covariance(
            support[local_labels >= old_way], cfg.TRAINER.BiMC.GAMMA_INC
        )
        covariance /= len(selected_ids)
        state = {
            'image_proto': prototypes,
            'cov_image': covariance,
            'description_proto': base_state['description_proto'][positions].float(),
            'text_features': base_state['text_features'][positions].float(),
            'description_features': torch.cat(description_features),
            'description_targets': torch.cat(description_targets),
        }
        support_frequency = query_frequency = anchor_frequency = None
        if model.frequency_enabled:
            if frequency_features is None:
                raise ValueError('Frequency meta reference needs cached per-image views.')
            support_frequency = frequency_features[support_indices]
            query_frequency = frequency_features[query_indices]
            frequency_prototypes, uncertainty, counts = compute_frequency_prototypes(
                support_frequency, local_labels, range(len(selected_ids)),
                return_counts=True,
            )
            raw_frequency_means = torch.stack([
                support_frequency[local_labels == c].mean(0)
                for c in range(len(selected_ids))
            ])
            anchor_frequency = raw_frequency_means[:old_way]
            if freq_cfg.USE_EXPLICIT_DESCRIPTIONS:
                candidates = base_state.get('frequency_description_candidates')
                if candidates is None:
                    raise ValueError('Explicit meta grounding needs text candidates.')
                desc_proto, _, _ = select_frequency_description_prototypes(
                    candidates[positions].float(), frequency_prototypes,
                    top_k=freq_cfg.DESCRIPTION_TOPK,
                    temperature=freq_cfg.DESCRIPTION_TEMPERATURE,
                )
            else:
                desc_proto = base_state['frequency_description_proto'][positions].float()
            semantic = F.normalize(
                (1 - freq_cfg.DESCRIPTION_WEIGHT)
                * base_state['frequency_prompt_proto'][positions].float()
                + freq_cfg.DESCRIPTION_WEIGHT * desc_proto, dim=-1,
            )
            if (old_way and cfg.TRAINER.BiMC.VISION_CALIBRATION
                    and freq_cfg.NOVEL_VISION_CALIBRATION):
                frequency_prototypes = frequency_prototypes.clone()
                for band in range(frequency_prototypes.shape[1]):
                    frequency_prototypes[old_way:, band] = model.soft_calibration(
                        frequency_prototypes[:old_way, band],
                        frequency_prototypes[old_way:, band],
                    )
            calibrated, band_weights, _, alignment = calibrate_frequency_prototypes(
                visual_prototypes=frequency_prototypes,
                semantic_prototypes=semantic,
                uncertainty=uncertainty,
                semantic_weight=freq_cfg.SEMANTIC_WEIGHT,
                max_semantic_weight=freq_cfg.MAX_SEMANTIC_WEIGHT,
                uncertainty_scale=freq_cfg.UNCERTAINTY_SCALE,
                alignment_scale=freq_cfg.ALIGNMENT_SCALE,
                fusion_temperature=freq_cfg.FUSION_TEMPERATURE,
                adaptive_fusion=freq_cfg.ADAPTIVE_FUSION,
                band_prior=features.new_tensor(freq_cfg.BAND_PRIOR),
                semantic_gate_mode=freq_cfg.SEMANTIC_GATE_MODE,
            )
            if freq_cfg.RELIABILITY_ALPHA:
                class_alpha, _ = compute_frequency_class_alpha(
                    alignment, uncertainty, band_weights, counts,
                    max_alpha=freq_cfg.FREQ_ALPHA,
                    min_alpha=freq_cfg.MIN_FREQ_ALPHA,
                    uncertainty_scale=freq_cfg.RELIABILITY_UNCERTAINTY_SCALE,
                    shot_tau=freq_cfg.RELIABILITY_SHOT_TAU,
                    reliability_power=freq_cfg.RELIABILITY_POWER,
                )
            else:
                class_alpha = features.new_full((len(selected_ids),), freq_cfg.FREQ_ALPHA)
            state.update(
                frequency_calibrated_proto=calibrated,
                frequency_band_weights=band_weights,
                frequency_class_alpha=class_alpha,
            )
        scorer = make_reference_scorer(model, cfg, state, old_way)
        anchors = raw_means[:old_way]
        return {
            'support_scores': scorer(support, support_frequency),
            'query_scores': scorer(query, query_frequency),
            'anchor_features': anchors,
            'anchor_scores': (
                scorer(anchors, anchor_frequency) if old_way
                else features.new_empty(0, len(selected_ids))
            ),
            'anchor_labels': torch.arange(old_way, device=features.device),
        }

    return reference


def _episode_shape(cfg, num_classes):
    residual = cfg.TRAINER.BiMC.RESIDUAL
    way = min(residual.META_WAY, num_classes)
    if way < 2:
        raise ValueError('At least two classes are needed for pseudo FSCIL episodes.')
    old_way = min(residual.META_OLD_WAY, way - 1)
    return way, old_way


def validate_dictionary(head, model, cfg, features, labels, state,
                        frequency_features=None):
    """Evaluate adaptation on held-out base classes; never update shared U."""
    residual = cfg.TRAINER.BiMC.RESIDUAL
    class_ids = torch.unique(labels).tolist()
    way, old_way = _episode_shape(cfg, len(class_ids))
    callback = make_episode_reference(
        model, cfg, features, labels, state, frequency_features
    )
    generator = torch.Generator().manual_seed(int(cfg.SEED) + 8701)
    totals = {'reference_correct': 0, 'adapted_correct': 0, 'query_count': 0}
    for _ in range(residual.META_VAL_EPISODES):
        episode = sample_residual_episode(
            labels, old_way=old_way, new_way=way-old_way,
            shot=cfg.DATASET.NUM_INC_SHOT, query_shot=residual.META_QUERY,
            old_shot=residual.META_OLD_SHOT, generator=generator,
        )
        device = features.device
        support_idx, query_idx = episode['support_indices'], episode['query_indices']
        support_labels, query_labels = episode['support_labels'], episode['query_labels']
        selected_ids = episode['selected_class_ids']
        reference = callback(support_idx, query_idx, selected_ids, old_way)
        temporary = LowRankResidualHead(
            features.shape[-1], way, head.dictionary.shape[-1],
            max_delta=residual.MAX_DELTA, gain=residual.GAIN,
        ).to(device)
        temporary.dictionary.copy_(head.dictionary)
        temporary.mark_seen(torch.arange(old_way, device=device))
        new_mask = support_labels >= old_way
        temporary.fit_session(
            features[support_idx][new_mask], support_labels[new_mask],
            reference['support_scores'][new_mask],
            torch.arange(way, device=device), torch.arange(old_way, way, device=device),
            anchor_features=reference['anchor_features'],
            anchor_reference_scores=reference['anchor_scores'],
            anchor_labels=reference['anchor_labels'],
            steps=residual.TRAIN_STEPS, lr=residual.LR,
            optimizer=residual.OPTIMIZER,
            l2=residual.L2_WEIGHT, old_margin=residual.OLD_MARGIN,
            old_weight=residual.OLD_LOSS_WEIGHT, grad_clip=residual.GRAD_CLIP,
            temperature=residual.TEMPERATURE,
        )
        with torch.no_grad():
            votes = reference['query_scores']
            adapted = temporary.forward_scores(
                features[query_idx], votes, torch.arange(way, device=device)
            )
            totals['reference_correct'] += int((votes.argmax(1) == query_labels).sum())
            totals['adapted_correct'] += int((adapted.argmax(1) == query_labels).sum())
            totals['query_count'] += len(query_labels)
    count = totals['query_count']
    return {
        'episodes': residual.META_VAL_EPISODES,
        'class_ids': class_ids,
        'query_count': count,
        'reference_accuracy': 100 * totals['reference_correct'] / count if count else None,
        'adapted_accuracy': 100 * totals['adapted_correct'] / count if count else None,
    }


def initialize_residual(model, cfg, base_state):
    residual = cfg.TRAINER.BiMC.RESIDUAL
    features, labels = base_state['images_features'], base_state['images_targets']
    feature_dim = features.shape[-1]
    rank = feature_dim if residual.DICTIONARY == 'identity' else residual.RANK
    head = LowRankResidualHead(
        feature_dim, cfg.DATASET.NUM_CLASSES, rank,
        max_delta=residual.MAX_DELTA, gain=residual.GAIN,
    ).to(features.device)
    ids = torch.unique(labels).cpu()
    if not 0 <= residual.META_VAL_FRACTION < 1:
        raise ValueError('RESIDUAL.META_VAL_FRACTION must lie in [0, 1).')
    generator = torch.Generator().manual_seed(int(cfg.SEED) + 7101)
    order = ids[torch.randperm(len(ids), generator=generator)]
    val_count = math.ceil(len(ids) * residual.META_VAL_FRACTION)
    if residual.META_VAL_EPISODES > 0 and val_count < 2:
        raise ValueError('Base validation requires at least two held-out classes.')
    if len(ids) - val_count < 2:
        raise ValueError('Dictionary training requires at least two base training classes.')
    val_ids = order[:val_count].to(labels.device)
    train_ids = order[val_count:].to(labels.device)
    train_mask = (labels[:, None] == train_ids[None, :]).any(1)
    val_mask = (labels[:, None] == val_ids[None, :]).any(1)
    train_features, train_labels = features[train_mask], labels[train_mask]
    method = 'residual_svd' if residual.DICTIONARY == 'meta' else residual.DICTIONARY
    report = {'training_class_ids': train_ids.cpu().tolist(),
              'validation_class_ids': val_ids.cpu().tolist(),
              'effective_rank': rank, 'dictionary': residual.DICTIONARY}
    report['initialization'] = head.initialize_dictionary(
        train_features, train_labels, method=method,
        shot=cfg.DATASET.NUM_INC_SHOT, repeats=residual.SVD_REPEATS,
        reference_shot=residual.SVD_REFERENCE_SHOT, seed=int(cfg.SEED) + 7201,
    )
    frequency = base_state.get('frequency_features')
    if residual.DICTIONARY == 'meta':
        if residual.META_STEPS <= 0:
            raise ValueError('DICTIONARY=meta requires positive META_STEPS.')
        way, old_way = _episode_shape(cfg, len(train_ids))
        report['meta'] = head.refine_dictionary(
            train_features, train_labels,
            reference_callback=make_episode_reference(
                model, cfg, train_features, train_labels, base_state,
                frequency[train_mask] if frequency is not None else None,
            ),
            episodes=residual.META_STEPS,
            old_way=old_way, new_way=way - old_way,
            old_shot=residual.META_OLD_SHOT,
            shot=cfg.DATASET.NUM_INC_SHOT, query_shot=residual.META_QUERY,
            inner_steps=residual.META_INNER_STEPS, inner_lr=residual.META_INNER_LR,
            outer_lr=residual.META_LR,
            l2=residual.L2_WEIGHT, old_weight=residual.OLD_LOSS_WEIGHT,
            old_margin=residual.OLD_MARGIN, orth_weight=residual.META_ORTH_WEIGHT,
            grad_clip=residual.GRAD_CLIP, temperature=residual.TEMPERATURE,
            seed=int(cfg.SEED) + 7301,
        )
    if residual.META_VAL_EPISODES > 0:
        report['validation'] = validate_dictionary(
            head, model, cfg, features[val_mask], labels[val_mask], base_state,
            frequency[val_mask] if frequency is not None else None,
        )
    head.mark_seen(ids.to(features.device))
    head.requires_grad_(False)
    model.residual_head = head
    return report
