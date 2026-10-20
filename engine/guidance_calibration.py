"""Base-training-only selection for support-only frequency guidance.

Pseudo-base and pseudo-novel classes are disjoint. Audit images are reserved
globally before all sequences, and cannot become another sequence's queries.
Covariance is accumulated by session, exactly as in the actual Runner.
"""

import hashlib
import json
import random
import time

import torch
import torch.nn.functional as F

from engine.residual_training import _group_covariance
from models.frequency_guidance import (
    relation_distributions, settings_kwargs, update_prototypes, validate_settings,
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def synchronize(model):
    if str(model.device).startswith('cuda'):
        torch.cuda.synchronize()


def stage_sizes(cfg):
    g = cfg.TRAINER.BiMC.GUIDANCE
    if g.STAGE_SIZES:
        sizes = list(g.STAGE_SIZES)
    else:
        # Half as many classes, while preserving the number of real sessions.
        total = cfg.DATASET.NUM_CLASSES - cfg.DATASET.NUM_INIT_CLS
        inc = cfg.DATASET.NUM_INC_CLS
        if inc < 1 or total < 1 or total % inc:
            raise ValueError('Guidance pseudo stages require a positive, divisible incremental schedule.')
        sizes = [((i + 1) * inc) // 2 - (i * inc) // 2 for i in range(total // inc)]
    if not sizes or any(not isinstance(v, int) or v < 1 for v in sizes):
        raise ValueError('Pseudo stage sizes must be positive integers; supply GUIDANCE.STAGE_SIZES for 1-way schedules.')
    return sizes


def sample_protocol(cfg, labels, sample_ids):
    g = cfg.TRAINER.BiMC.GUIDANCE
    ids = sorted(int(c) for c in torch.unique(labels).cpu().tolist())
    if ids != list(range(cfg.DATASET.NUM_INIT_CLS)) or len(ids) < 4:
        raise ValueError('Guidance calibration needs the complete original base training bank (at least four classes).')
    if len(sample_ids) != len(labels) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError('Guidance base sample IDs must be unique and aligned with feature rows.')
    seed = int(g.CALIBRATION_SEED if g.CALIBRATION_SEED >= 0 else cfg.SEED + 10101)
    rng = random.Random(seed)
    shuffled = rng.sample(ids, len(ids))
    a, v = sorted(shuffled[:len(ids) // 2]), sorted(shuffled[len(ids) // 2:])
    sizes = stage_sizes(cfg)
    if sum(sizes) > len(v):
        raise ValueError('Pseudo schedule requests more novel classes than the held-out base class split.')
    indices = {c: torch.where(labels.cpu() == c)[0].tolist() for c in ids}
    audit = {}
    for c in v:
        needed = g.AUDIT_IMAGES + g.SHOT + g.QUERY
        if len(indices[c]) < needed:
            raise ValueError(f'Base validation class {c} has {len(indices[c])} images; needs {needed} distinct audit/support/query images.')
        audit[c] = rng.sample(indices[c], g.AUDIT_IMAGES)
    for c in a:
        if len(indices[c]) < g.QUERY + 2:
            raise ValueError(f'Pseudo-base class {c} lacks support after {g.QUERY} held-out queries.')
    sequences = []
    for _ in range(g.VAL_SEQUENCES):
        novel = rng.sample(v, sum(sizes))
        support, query = [], []
        for c in a:
            q = rng.sample(indices[c], g.QUERY)
            query.append(q)
            support.append([i for i in indices[c] if i not in set(q)])
        for c in novel:
            available = [i for i in indices[c] if i not in set(audit[c])]
            chosen = rng.sample(available, g.SHOT + g.QUERY)
            support.append(chosen[:g.SHOT])
            query.append(chosen[g.SHOT:])
        sequences.append({'class_ids': a + novel, 'support_indices': support,
                          'query_indices': query, 'stage_sizes': sizes})
    protocol = {'seed': seed, 'source': 'base_training_only', 'pseudo_base_class_ids': a,
                'pseudo_validation_class_ids': v, 'stage_sizes': sizes,
                'sample_ids': list(sample_ids), 'audit_indices': {str(k): value for k, value in audit.items()},
                'sequences': sequences}
    protocol['sha256'] = digest(protocol)
    return protocol


def select_candidate(candidates, maximum_drop):
    reference = next(row for row in candidates if row['rho'] == 0)
    for row in candidates:
        row['old_delta_pp'] = row['mean_base_accuracy'] - reference['mean_base_accuracy']
        row['new_delta_pp'] = row['mean_novel_accuracy'] - reference['mean_novel_accuracy']
        row['feasible'] = min(row['old_delta_pp'], row['new_delta_pp']) >= -maximum_drop - 1e-12
    feasible = [row for row in candidates if row['feasible']]
    best = max(row['mean_harmonic_accuracy'] for row in feasible)
    return min((row for row in feasible if row['mean_harmonic_accuracy'] >= best - 1e-12), key=lambda r: r['rho'])


def _means(features, groups):
    return F.normalize(torch.stack([features[rows].mean(0) for rows in groups]), dim=-1)


def _text_state(state, selected, device):
    positions = {int(c): i for i, c in enumerate(state['class_index'])}
    local = torch.tensor([positions[c] for c in selected], device=device)
    features, targets = [], []
    for i, c in enumerate(selected):
        values = state['description_features'][state['description_targets'] == c]
        features.append(values)
        targets.append(torch.full((len(values),), i, device=device, dtype=torch.long))
    return {'description_proto': state['description_proto'][local],
            'text_features': state['text_features'][local],
            'description_features': torch.cat(features), 'description_targets': torch.cat(targets)}


@torch.no_grad()
def sequence_records(model, cfg, state, sequence):
    """Yield support-only stage inputs; exposed for leakage regression tests."""
    device = state['images_features'].device
    x, f = state['images_features'], state['frequency_features']
    groups = [torch.tensor(rows, device=device) for rows in sequence['support_indices']]
    selected = sequence['class_ids']
    sizes = sequence['stage_sizes']
    old = len(selected) - sum(sizes)
    visual, bands = _means(x, groups), _means(f, groups)
    pos = {int(c): i for i, c in enumerate(state['class_index'])}
    positions = torch.tensor([pos[c] for c in selected], device=device)
    semantics = state.get('guidance_text_proto')
    semantics = semantics[positions] if semantics is not None else None
    cov_sum = old * _group_covariance(x[torch.cat(groups[:old])], cfg.TRAINER.BiMC.GAMMA_BASE)
    start = old
    for stage, size in enumerate(sizes, 1):
        end = start + size
        cov_sum = cov_sum + size * _group_covariance(x[torch.cat(groups[start:end])], cfg.TRAINER.BiMC.GAMMA_INC)
        covariance = cov_sum / end
        relations = relation_distributions(
            visual[:old], visual[start:end], bands[:old], bands[start:end],
            semantics[:old] if semantics is not None else None,
            semantics[start:end] if semantics is not None else None, **settings_kwargs(cfg),
        )
        query = torch.tensor([i for rows in sequence['query_indices'][:end] for i in rows], device=device)
        targets = torch.cat([torch.full((len(rows),), i, device=device, dtype=torch.long)
                             for i, rows in enumerate(sequence['query_indices'][:end])])
        yield {'stage': stage, 'class_ids': selected[:end], 'num_base': old,
               'visual': visual[:end], 'bands': bands[:end],
               'relations': relations, 'new_begin': start, 'new_count': size,
               'query_indices': query, 'targets': targets, 'query': x[query],
               'covariance': covariance, 'text': _text_state(state, selected[:end], device),
               'novel_count': end - old}
        start = end


@torch.no_grad()
def _audit(model, cfg, state, protocol, rho):
    """Post-selection diagnostics; these images never enter candidate scoring."""
    totals = {name: [] for name in ('raw', 'G1', 'selected')}
    margins = {name: [] for name in totals}
    for sequence in protocol['sequences']:
        old = len(protocol['pseudo_base_class_ids'])
        groups = sequence['support_indices']
        v = _means(state['images_features'], groups)
        f = _means(state['frequency_features'], groups)
        ids = sequence['class_ids']
        positions = {int(c): i for i, c in enumerate(state['class_index'])}
        text = state.get('guidance_text_proto')
        t = text[[positions[c] for c in ids]] if text is not None else None
        guided, reference, begin = [], [], old
        for size in sequence['stage_sizes']:
            end = begin + size
            relations = relation_distributions(v[:old], v[begin:end], f[:old], f[begin:end],
                                              t[:old] if t is not None else None,
                                              t[begin:end] if t is not None else None, **settings_kwargs(cfg))
            reference.append(model.soft_calibration(v[:old], v[begin:end]))
            guided.append(update_prototypes(v[:old], v[begin:end], relations, rho,
                                             cfg.TRAINER.BiMC.LAMBDA_I, model.soft_calibration))
            begin = end
        choices = {'raw': v[old:], 'G1': torch.cat(reference), 'selected': torch.cat(guided)}
        centers = _means(state['images_features'], [protocol['audit_indices'][str(c)] for c in ids[old:]])
        for name, novel in choices.items():
            totals[name].extend((1 - F.cosine_similarity(novel, centers, dim=-1)).cpu().tolist())
            bank = torch.cat((v[:old], novel))
            for j, c in enumerate(ids[old:]):
                scores = state['images_features'][protocol['audit_indices'][str(c)]] @ bank.T
                own = scores[:, old + j].clone()
                scores[:, old + j] = -torch.inf
                margins[name].extend((own - scores.max(-1).values).cpu().tolist())
    return {'source': 'globally_reserved_base_validation_audit_images_post_selection_only',
            'mean_cosine_error': {k: sum(v) / len(v) for k, v in totals.items()},
            'mean_cosine_margin': {k: sum(v) / len(v) for k, v in margins.items()},
            'class_occurrences': len(totals['raw'])}


@torch.no_grad()
def initialize_guidance(model, cfg, state):
    validate_settings(cfg)
    if model.guidance_state is not None:
        raise RuntimeError('Initialize GUIDANCE only once, during the base session.')
    if (state.get('data_source') != 'base_training_only' or state.get('task_id') != 0
            or sorted(int(c) for c in state['class_index']) != list(range(cfg.DATASET.NUM_INIT_CLS))
            or sorted(int(c) for c in torch.unique(state['images_targets'])) != list(range(cfg.DATASET.NUM_INIT_CLS))):
        raise ValueError('Guidance selection accepts the complete base training session only.')
    g = cfg.TRAINER.BiMC.GUIDANCE
    synchronize(model)
    start = time.perf_counter()
    report = {'schema_version': 1, 'method': g.MODE, 'data_source': 'base_training_only',
              'auto_calibrate': g.AUTO_CALIBRATE, 'run_seed': int(cfg.SEED),
              'semantic_source': g.SEMANTIC_SOURCE, 'view_control': g.VIEW_CONTROL,
              'selection_objective': 'mean_incremental_stage_harmonic_A_all_arrived_V',
              'maximum_mean_group_drop_pp': float(g.MAX_GROUP_DROP_PP),
              'candidate_results': [], 'query_auxiliary_encodings': 0}
    rho = float(g.RHO)
    if g.AUTO_CALIBRATE:
        protocol = sample_protocol(cfg, state['images_targets'], state['guidance_sample_ids'])
        grid = sorted(set(float(v) for v in g.RHO_GRID))
        results = {value: [] for value in grid}
        for index, sequence in enumerate(protocol['sequences']):
            # Like the real Runner, append newly adapted classes once. Past
            # novel prototypes must not be recomputed as the candidate set grows.
            prototype_banks = {}
            for rec in sequence_records(model, cfg, state, sequence):
                old, visual = rec['num_base'], rec['visual']
                inverse = torch.pinverse(rec['covariance'].float())
                for value in grid:
                    novel = update_prototypes(visual[:old], visual[rec['new_begin']:],
                                              rec['relations'], value,
                                              cfg.TRAINER.BiMC.LAMBDA_I, model.soft_calibration)
                    previous = prototype_banks.get(value, visual[:old])
                    prototype_banks[value] = torch.cat((previous, novel))
                    scores = model.reference_scores_from_features(
                        rec['query'], len(visual), old, prototype_banks[value],
                        rec['covariance'], rec['text']['description_proto'],
                        rec['text']['description_features'], rec['text']['description_targets'],
                        rec['text']['text_features'], cfg.DATASET.BETA, cov_inverse=inverse,
                    )
                    if not torch.isfinite(scores).all():
                        raise ValueError(f'Non-finite BiMC validation scores in sequence {index}, stage {rec["stage"]}.')
                    correct, mask = scores.argmax(-1) == rec['targets'], rec['targets'] < old
                    a = 100 * correct[mask].float().mean().item()
                    v = 100 * correct[~mask].float().mean().item()
                    h = 2 * a * v / (a + v) if a + v else 0.
                    results[value].append({'sequence': index, 'stage': rec['stage'],
                                           'base_accuracy': a, 'novel_accuracy': v,
                                           'harmonic_accuracy': h, 'query_occurrences': len(mask)})
            print(f'Guidance base validation sequence {index + 1}/{len(protocol["sequences"])}', flush=True)
        candidates = []
        for value, rows in results.items():
            candidates.append({'rho': value, 'mean_base_accuracy': sum(r['base_accuracy'] for r in rows) / len(rows),
                               'mean_novel_accuracy': sum(r['novel_accuracy'] for r in rows) / len(rows),
                               'mean_harmonic_accuracy': sum(r['harmonic_accuracy'] for r in rows) / len(rows),
                               'stages': rows})
        winner = select_candidate(candidates, g.MAX_GROUP_DROP_PP)
        rho = winner['rho']
        report.update(selection_status='base_class_disjoint_validation', protocol=protocol,
                      protocol_sha256=protocol['sha256'], candidate_results=candidates,
                      validation_query_occurrences=sum(r['query_occurrences'] for r in winner['stages']),
                      audit=_audit(model, cfg, state, protocol, rho))
    else:
        report.update(selection_status='manual_no_validation', protocol_sha256=None)
    model.guidance_state = {'rho': rho, 'mode': g.MODE,
                            'base_ids': [int(c) for c in state['class_index']],
                            'base_visual': state['image_proto'].detach().clone(),
                            'base_bands': state['guidance_visual_proto'].detach().clone(),
                            'base_text': (state['guidance_text_proto'].detach().clone()
                                          if state.get('guidance_text_proto') is not None else None)}
    synchronize(model)
    report.update(selected_rho=rho, fallback_to_G1=rho == 0,
                  calibration_seconds=time.perf_counter() - start,
                  refit_source='all_original_base_training_support')
    return report
