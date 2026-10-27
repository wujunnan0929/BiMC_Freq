"""Cached, class-disjoint BASE development for multi-layer/attribute completion.

Importing this module is standard-library only so planners need no ML runtime.
All encodings come from an authenticated task-zero training cache. This module
does not load benchmark images, train gradients, or export a deployable model.
"""

import csv
import hashlib
import json
import math
from pathlib import Path
import random
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
FORMAT = 'multilayer_completion_development_v1'
GROUPS = ('shape', 'part', 'color', 'pattern')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode('utf-8')).hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def experiment_variants(stage, selected_layer=None):
    """One source of truth for the pure-standard-library planner and runner."""
    def row(name, **options):
        return {'name': name, 'stage': stage, **options}
    if stage == 'repeat':
        return [row('original16', kind='legacy', mode='original', rank=16),
                row('original32', kind='legacy', mode='original', rank=32),
                row('repeat_same', kind='legacy', mode='same', rank=16),
                row('repeat_attribute', kind='legacy', mode='attribute', rank=16),
                row('single_two_text', kind='legacy', mode='pooled', rank=16),
                row('repeat_same8', kind='legacy', mode='same', rank=8),
                row('repeat_attribute8', kind='legacy', mode='attribute', rank=8),
                row('typed_single', kind='typed', text='baseline', target_rank=16),
                row('typed_repeat', kind='typed', text='baseline', target_rank=16, duplicate=True)]
    if stage == 'layers':
        return [row('typed_final', kind='typed', text='baseline', target_rank=16),
                row('typed_final24', kind='typed', text='baseline', target_rank=24),
                *[row('layer'+str(i), kind='typed', text='baseline', target_rank=16, layer=i) for i in (3, 6, 9)],
                row('score_fusion', kind='fusion', layer=selected_layer),
                row('score_fusion_final', kind='fusion', layer=None)]
    if stage == 'attributes':
        rows = [row(prefix+name, kind='typed', text=name, target_rank=16,
                    **({'layer': selected_layer} if prefix == 'middle_' else {}))
                for prefix in ('final_', 'middle_')
                for name in ('ordinary', 'attribute_mean', 'attribute_groups', 'random_groups')]
        return rows + [row('final_ordinary24', kind='typed', text='ordinary', target_rank=24),
                       row('final_attribute_groups24', kind='typed', text='attribute_groups', target_rank=24)]
    raise ValueError('Unknown development stage: ' + str(stage))


def _runtime():
    global torch, F, np
    import numpy as np
    import torch
    from torch.nn import functional as F


def _pool(values, mask):
    weight = mask.to(values.dtype)
    pooled = (values * weight[..., None]).sum(-2) / weight.sum(-1).clamp_min(1)[..., None]
    return F.normalize(pooled, dim=-1), mask.any(-1)


def semantic_blocks(arrays, cfg, mode):
    """Fixed text aggregation. T1/T2/T3 share precisely the same valid slots."""
    if mode == 'baseline':
        coefficient = cfg.TRAINER.BiMC.LAMBDA_T if cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.
        text = F.normalize((1-coefficient)*arrays['text_features']
                           + coefficient*arrays['description_proto'], dim=-1)
        return {'ordinary': text}, {'ordinary': torch.ones(len(text), dtype=torch.bool, device=text.device)}
    attributes, mask = arrays['attr_text'].clone(), arrays['attr_mask'].clone()
    ordinary, ordinary_mask = arrays['ordinary_text'], arrays['ordinary_mask']
    flat, valid = attributes.flatten(1, 2), mask.flatten(1, 2)
    # Match the actual number of available strings per class, never fill unknowns
    # with duplicate generic prompts. All attribute treatments use this same pool.
    matched = torch.minimum(valid.sum(1), ordinary_mask.sum(1))
    valid &= valid.long().cumsum(1) <= matched[:, None]
    if mode == 'ordinary':
        chosen = ordinary_mask & (ordinary_mask.long().cumsum(1) <= matched[:, None])
        text, available = _pool(ordinary, chosen)
        return {'ordinary': text}, {'ordinary': available}
    if mode == 'attribute_mean':
        text, available = _pool(flat, valid)
        return {'attributes': text}, {'attributes': available}
    if mode == 'random_groups':
        order = list(range(flat.shape[1]))
        random.Random(3101).shuffle(order)
        flat, valid = flat[:, order], valid[:, order]
        names = tuple('random_'+str(i) for i in range(4))
    elif mode == 'attribute_groups':
        names = GROUPS
    else:
        raise ValueError('Unknown semantic mode: ' + mode)
    grouped, grouped_mask = flat.reshape(len(flat), 4, 2, -1), valid.reshape(len(flat), 4, 2)
    text, available = _pool(grouped, grouped_mask)
    return {name: text[:, i] for i, name in enumerate(names)}, {name: available[:, i] for i, name in enumerate(names)}


def _configuration(metadata, fit_seed, sequences):
    from yacs.config import CfgNode
    cfg = CfgNode.load_cfg(metadata['config_yaml'])
    cfg.defrost()
    cfg.SEED = int(fit_seed)
    p = cfg.TRAINER.BiMC.PROTOTYPE_COMPLETION
    p.CALIBRATION_SEED = int(fit_seed) + 10101
    p.VAL_SEQUENCES = int(sequences)
    p.SHOT = int(cfg.DATASET.NUM_INC_SHOT)
    cfg.freeze()
    return cfg


def _scorer(row, cfg, prototypes):
    """Same algebraic BiMC score as prototype_evaluation, reusing q @ inverse."""
    from engine.prototype_evaluation import scores_from_logits
    if torch.equal(prototypes, row['reference_prototypes']):
        return row['reference_votes']
    p = prototypes.float()
    semantic = row['semantic']
    fused = cfg.DATASET.BETA * semantic + (1-cfg.DATASET.BETA) * p
    cov = (row['query_cov'] @ p.double().T
           - (p.double() * (p.double() @ row['inverse64'])).sum(1)) / 512.
    return scores_from_logits({'fused': row['query'] @ F.normalize(fused, dim=-1).T,
                               'cov': cov.float(), 'description': row['description_logits']},
                              row['num_old'], cfg)


def build_records(arrays, cfg, protocol):
    """Freeze prototypes on arrival; fit/image-dependent statistics use support only."""
    from engine.safd_metric_training import _reference_proxy, _text_subset
    from engine.residual_training import _group_covariance
    from engine.prototype_evaluation import scores_from_logits
    proxy, records = _reference_proxy(cfg), []
    x, old = arrays['final'], len(protocol['pseudo_base_class_ids'])
    plain = {k: v for k, v in arrays.items() if k in ('text_features','description_features','description_targets','description_proto')}
    for sequence, seq in enumerate(protocol['sequences']):
        ids, supports, queries = seq['class_ids'], seq['support_indices'], seq['query_indices']
        raw = torch.stack([x[rows].mean(0) for rows in supports])
        reference = F.normalize(raw, dim=-1)
        if cfg.TRAINER.BiMC.VISION_CALIBRATION:
            reference[old:] = proxy.soft_calibration(reference[:old], reference[old:])
        cov_sum = old * _group_covariance(x[[i for rows in supports[:old] for i in rows]], cfg.TRAINER.BiMC.GAMMA_BASE)
        start = old
        for stage, size in enumerate(seq['stage_sizes'], 1):
            end = start + size
            cov_sum = cov_sum + size * _group_covariance(x[[i for rows in supports[start:end] for i in rows]], cfg.TRAINER.BiMC.GAMMA_INC)
            inverse = torch.pinverse((cov_sum/end).float()).double()
            dp, df, dt, text = _text_subset(plain, ids[:end], x.device)
            qrows = [i for rows in queries[:end] for i in rows]
            q = F.normalize(x[qrows].float(), dim=-1)
            targets = torch.cat([torch.full((len(rows),), j, dtype=torch.long, device=x.device)
                                 for j, rows in enumerate(queries[:end])])
            coefficient = cfg.TRAINER.BiMC.LAMBDA_T if cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.
            semantic = (1-coefficient)*text + coefficient*dp
            similarity = q @ df.T
            description_logits = torch.stack([similarity[:, dt == j].max(1).values for j in range(end)], 1)
            row = {'sequence': sequence, 'stage': stage, 'class_ids': ids[:end], 'num_old': old,
                   'support_indices': supports[:end], 'query_indices': queries[:end],
                   'query_rows': qrows, 'query': q, 'targets': targets,
                   'raw': raw[:end], 'reference_prototypes': reference[:end],
                   'semantic': semantic, 'inverse64': inverse,
                   'query_cov': q.double() @ (inverse+inverse.T),
                   'description_logits': description_logits}
            p = reference[:end].double()
            logits = {'fused': q @ F.normalize(cfg.DATASET.BETA*semantic+(1-cfg.DATASET.BETA)*reference[:end], dim=-1).T,
                      'cov': ((row['query_cov'] @ p.T - (p * (p @ inverse)).sum(1))/512.).float(),
                      'description': description_logits}
            row['reference_votes'] = scores_from_logits(logits, old, cfg)
            records.append(row)
            start = end
    return records


def _legacy_inputs(arrays, cfg, mode):
    text, _ = semantic_blocks(arrays, cfg, 'baseline')
    ordinary = text['ordinary']
    final = arrays['final']
    if mode == 'original':
        return final[:, None], ordinary[:, None]
    if mode == 'pooled':
        return final[:, None], F.normalize(ordinary+arrays['structured_text'][:, 0], dim=-1)[:, None]
    auxiliary = ordinary if mode == 'same' else arrays['structured_text'][:, 0]
    return torch.stack((final, final), 1), torch.stack((ordinary, auxiliary), 1)


def fit_variant(definition, arrays, cfg, protocol, shared_bases):
    fit_rows, ids = protocol['fitting_indices'], protocol['pseudo_base_class_ids']
    y = arrays['labels'][fit_rows]
    if definition['kind'] == 'legacy':
        from models.prototype_completion import fit_completion
        visual, text = _legacy_inputs(arrays, cfg, definition['mode'])
        state = fit_completion(visual[fit_rows], y, text[ids], ids, rank=definition['rank'])
        return {'kind': 'legacy', 'state': state, 'visual': visual, 'text': text}
    if definition['kind'] == 'fusion':
        return {'kind': 'fusion', 'layer': definition['layer']}
    from models.multievidence_completion import fit_model
    mode = definition['text']
    texts, masks = semantic_blocks(arrays, cfg, mode)
    visual = {'final': arrays['final']}
    if definition.get('layer') is not None:
        key = 'layer_'+str(definition['layer'])
        visual[key] = arrays[key]
    if definition.get('duplicate'):
        visual['final_duplicate'] = arrays['final']
        texts = {**texts, 'ordinary_duplicate': texts['ordinary']}
        masks = {**masks, 'ordinary_duplicate': masks['ordinary']}
    rank = int(definition['target_rank'])
    state = fit_model({k: v[fit_rows] for k,v in visual.items()}, y,
                      {k:v[ids] for k,v in texts.items()}, ids,
                      text_masks={k:v[ids] for k,v in masks.items()},
                      target_rank=rank, target_basis=shared_bases.get(rank),
                      auxiliary_rank=8, text_rank=4 if mode in ('attribute_groups','random_groups') else 16)
    return {'kind':'typed','state':state,'visual':visual,'text':texts,'masks':masks}


def sequence_completions(fitted, arrays, sequence, omitted_text=None):
    ids, supports = sequence['class_ids'], sequence['support_indices']
    old = len(ids)-sum(sequence['stage_sizes'])
    ids = ids[old:]
    rows = [i for group in supports[old:] for i in group]
    if fitted['kind'] == 'legacy':
        from models.prototype_completion import complete_prototypes
        return complete_prototypes(fitted['state'], fitted['visual'][rows], arrays['labels'][rows], fitted['text'][ids], ids)
    from models.multievidence_completion import predict_model
    return predict_model(fitted['state'], {k:v[rows] for k,v in fitted['visual'].items()},
                         arrays['labels'][rows], {k:v[ids] for k,v in fitted['text'].items()}, ids,
                         text_masks={k:v[ids] for k,v in fitted['masks'].items()}, omitted_text=omitted_text)


def _metrics(scores, row):
    prediction, target = scores.argmax(1), row['targets']
    reference = row['reference_votes'].argmax(1)
    correct, old = prediction == target, target < row['num_old']
    ref_correct = reference == target
    a = 100*correct[old].float().mean().item()
    n = 100*correct[~old].float().mean().item()
    rival = scores.clone()
    own = scores.gather(1, target[:,None]).squeeze(1)
    rival.scatter_(1, target[:,None], -torch.inf)
    return {'sequence': row['sequence'], 'stage': row['stage'],
            'accuracy': 100*correct.float().mean().item(),
            'base_accuracy': a, 'novel_accuracy': n,
            'harmonic_accuracy': 2*a*n/(a+n) if a+n else 0.,
            'corrected': int((correct & ~ref_correct).sum()),
            'damaged': int((~correct & ref_correct).sum()),
            'changed': int((prediction != reference).sum()),
            'query_occurrences': len(target),
            'old_to_novel_errors': int(((prediction >= row['num_old']) & old).sum()),
            'novel_to_old_errors': int(((prediction < row['num_old']) & ~old).sum()),
            'old_within_errors': int(((prediction < row['num_old']) & old & ~correct).sum()),
            'novel_within_errors': int(((prediction >= row['num_old']) & ~old & ~correct).sum()),
            'mean_true_class_margin': float((own-rival.max(1).values).mean()),
            'exact_reference_scores': torch.equal(scores, row['reference_votes'])}


def _aggregate(rows):
    result = {key: sum(r[key] for r in rows)/len(rows)
              for key in ('accuracy','base_accuracy','novel_accuracy','harmonic_accuracy','mean_true_class_margin')}
    result.update({key: sum(r[key] for r in rows) for key in (
        'corrected','damaged','changed','query_occurrences','old_to_novel_errors',
        'novel_to_old_errors','old_within_errors','novel_within_errors')})
    result['stages'] = len(rows)
    result['exact_reference_scores'] = all(r['exact_reference_scores'] for r in rows)
    return result


def _evaluate(fitted, definition, arrays, cfg, protocol, records, grid, omitted_text=None):
    from engine.completion_calibration import blend_completion
    totals, prediction_banks = {v:[] for v in grid}, {v:[] for v in grid}
    completions = {}
    if fitted['kind'] != 'fusion':
        completions = {i:sequence_completions(fitted, arrays, seq, omitted_text)
                       for i,seq in enumerate(protocol['sequences'])}
    identities = {'sample_id':[], 'target':[], 'reference_prediction':[], 'sequence':[], 'stage':[]}
    for row in records:
        old, end = row['num_old'], len(row['class_ids'])
        classes = torch.tensor(row['class_ids'], device=row['query'].device)
        reference = row['reference_prototypes']
        if fitted['kind'] == 'fusion':
            key = 'final' if fitted['layer'] is None else 'layer_'+str(fitted['layer'])
            values = arrays[key]
            proto = F.normalize(torch.stack([values[rows].mean(0) for rows in row['support_indices']]), dim=-1)
            native = F.softmax(16*(F.normalize(values[row['query_rows']], dim=-1) @ proto.T), dim=1)
            reference_mass = row['reference_votes'].sum(1, keepdim=True)
        else:
            posterior = completions[row['sequence']][:end-old]
        for strength in grid:
            if strength == 0:
                scores = row['reference_votes']
            elif fitted['kind'] == 'fusion':
                scores = (1-strength)*row['reference_votes'] + strength*reference_mass*native
            else:
                bank = torch.cat((reference[:old], blend_completion(
                    reference[old:], row['raw'][old:], posterior, strength)))
                scores = _scorer(row, cfg, bank)
            measured=_metrics(scores, row)
            complete=((arrays['attr_mask'].all(2).all(1)) & arrays['ordinary_mask'].all(1))[classes[row['targets']]]
            measured['complete_attribute_query_occurrences']=int(complete.sum())
            measured['complete_attribute_correct']=int(((scores.argmax(1)==row['targets']) & complete).sum())
            measured['incomplete_attribute_query_occurrences']=int((~complete).sum())
            measured['incomplete_attribute_correct']=int(((scores.argmax(1)==row['targets']) & ~complete).sum())
            totals[strength].append(measured)
            prediction_banks[strength].append(classes[scores.argmax(1)].detach().cpu().numpy())
        for key, values in (('sample_id', arrays['sample_id'][row['query_rows']]),
                            ('target', classes[row['targets']]),
                            ('reference_prediction', classes[row['reference_votes'].argmax(1)])):
            identities[key].append(values.detach().cpu().numpy())
        identities['sequence'].append(np.full(len(row['targets']),row['sequence'],dtype=np.int64))
        identities['stage'].append(np.full(len(row['targets']),row['stage'],dtype=np.int64))
    candidates = [{'blend': value, **_aggregate(totals[value])} for value in grid]
    for candidate,value in zip(candidates,grid):
        for subset in ('complete_attribute','incomplete_attribute'):
            count=sum(r[subset+'_query_occurrences'] for r in totals[value])
            correct=sum(r[subset+'_correct'] for r in totals[value])
            candidate[subset+'_query_occurrences']=count
            candidate[subset+'_pooled_accuracy']=(100*correct/count if count else None)
    predictions = {k:np.concatenate(v) for k,v in identities.items()}
    predictions['blend_grid'] = np.array(grid, dtype=np.float64)
    predictions['candidate_prediction'] = np.stack([np.concatenate(prediction_banks[v]) for v in grid])
    return candidates, totals, predictions


def select_candidate(candidates, maximum_drop):
    reference = next(r for r in candidates if r['blend'] == 0)
    for row in candidates:
        for name in ('accuracy','base_accuracy','novel_accuracy','harmonic_accuracy'):
            row['delta_'+name+'_pp'] = row[name]-reference[name]
        row['feasible'] = min(row['delta_base_accuracy_pp'],row['delta_novel_accuracy_pp']) >= -maximum_drop-1e-12
    feasible = [r for r in candidates if r['feasible']]
    best = max(r['accuracy'] for r in feasible)
    return min((r for r in feasible if r['accuracy'] >= best-1e-12),key=lambda r:r['blend'])


def _tensor_scalars(value, seen=None):
    seen=set() if seen is None else seen
    if torch.is_tensor(value):
        key=(value.data_ptr(),value.numel(),str(value.dtype))
        if key in seen:
            return 0
        seen.add(key)
        return value.numel()
    if isinstance(value,dict):
        return sum(_tensor_scalars(v,seen) for v in value.values())
    if isinstance(value,(list,tuple)):
        return sum(_tensor_scalars(v,seen) for v in value)
    return 0


def _model_diagnostics(fitted, protocol):
    if fitted['kind'] == 'fusion':
        return {'kind':'class_score_fusion', 'query_uses_middle_layer':fitted['layer'] is not None,
                'new_image_forward_passes':0, 'native_layer':fitted['layer']}
    state = fitted['state']
    scalars = _tensor_scalars(state)
    out = {'kind':fitted['kind'], 'retained_statistic_scalars':scalars,
           'retained_training_samples':0, 'gradient_training_performed':False,
           'query_uses_middle_layer':False, 'fitting_indices_sha256':_digest(protocol['fitting_indices'])}
    if fitted['kind'] == 'typed':
        out.update(target_rank=state['target_rank'], observation_dimension=state['observation_dimension'],
                   visual_dimension=state['visual_dimension'],
                   blocks=[{**{k:v for k,v in block.items() if k!='basis'},
                            'basis_array_key':'block_'+str(i)+'_basis'} for i,block in enumerate(state['blocks'])],
                   visual_aliases=state['visual_aliases'],text_aliases=state['text_aliases'],
                   dropped_text_blocks=state.get('dropped_text_blocks'),
                   solver_regularization=state.get('solver_regularization'),
                   Q_final_max=float(state['Q'][:state['target_rank']].abs().max()),
                   W_text_max=(float(state['W'][state['visual_dimension']:].abs().max())
                               if state['visual_dimension'] < state['observation_dimension'] else 0.))
        for name in ('between','Q','W'):
            eigen = torch.linalg.eigvalsh(state[name])
            out[name+'_spectrum'] = {'min':float(eigen.min()),'max':float(eigen.max()),'trace':float(eigen.sum())}
        out['text_coverage_by_class'] = {key: value.detach().cpu().tolist() for key,value in fitted['masks'].items()}
        out['actual_text_rank_by_class']=[sum(block['rank'] for block in state['blocks']
            if block['kind']=='text' and bool(fitted['masks'][block['name']][i]))
            for i in range(len(next(iter(fitted['masks'].values()))))] if fitted['masks'] else []
        out['crossfit_trace']=state.get('crossfit_trace',[])
        out['finite_mean_correction']=state.get('finite_mean_correction')
        out['crossfit_preprocessing_note']='Class mapping pairs are held out; PCA and pooled noise use all fitting A rows.'
    else:
        out.update(rank=state['rank'],views=state['views'],
                   duplicate_observations_merged=False,
                   note='Legacy repeat deliberately retains old isotropic regularization as a confounding control.')
    return out


def _save_diagnostic_arrays(path, fitted):
    state = fitted.get('state',{})
    arrays = {key:value.detach().cpu().numpy() for key,value in state.items()
              if torch.is_tensor(value) and key in ('basis','H','bias','mean','between','within','offset','Q','W',
                                                     'fit_class_ids','fit_class_counts')}
    for i,block in enumerate(state.get('blocks',[])):
        arrays['block_'+str(i)+'_basis']=block['basis'].detach().cpu().numpy()
    if arrays:
        np.savez_compressed(path, **arrays)


def _teacher_diagnostics(fitted, arrays, cfg, protocol, records, strength):
    """Post-selection independent teacher images; never part of fitting or selection."""
    from engine.completion_calibration import blend_completion
    if fitted['kind']=='fusion':
        return {'available':False,'reason':'score_fusion does not construct completed prototypes'}
    details=[]
    for index,seq in enumerate(protocol['sequences']):
        row=next(r for r in reversed(records) if r['sequence']==index)
        ids,old=row['class_ids'],row['num_old']
        posterior=sequence_completions(fitted,arrays,seq)
        selected=blend_completion(row['reference_prototypes'][old:],row['raw'][old:],posterior,strength)
        banks={'raw':F.normalize(row['raw'],dim=-1),'reference':row['reference_prototypes'],
               'selected':torch.cat((row['reference_prototypes'][:old],selected))}
        for j,cls in enumerate(ids[old:],old):
            used=set(seq['support_indices'][j])|set(seq['query_indices'][j])
            available=[i for i in torch.where(arrays['labels']==cls)[0].cpu().tolist() if i not in used]
            chosen=random.Random(protocol['seed']+index*1009+int(cls)*31).sample(available,min(15,len(available)))
            if len(chosen)<2:
                continue
            teacher=F.normalize(arrays['final'][chosen].mean(0),dim=-1)
            record={'sequence':index,'class_id':int(cls),'teacher_count':len(chosen),
                    'teacher_sample_ids':arrays['sample_id'][chosen].cpu().tolist()}
            for name,bank in banks.items():
                record[name+'_cosine_error']=float(1-(bank[j]*teacher).sum())
                similarities=arrays['final'][chosen] @ bank.T
                own=similarities[:,j].clone()
                similarities[:,j]=-torch.inf
                record[name+'_cosine_margin']=float((own-similarities.max(1).values).mean())
            details.append(record)
    keys=[name+'_'+metric for name in ('raw','reference','selected') for metric in ('cosine_error','cosine_margin')]
    return {'available':bool(details),'post_selection_only':True,'selection_candidate':False,
            'strict_performance_upper_bound':False,'support_teacher_query_disjoint':True,
            'disjoint_scope':'Within each sequence; teacher bank is not globally reserved across sequences.',
            'class_occurrences':len(details),'means':{k:sum(r[k] for r in details)/len(details) for k in keys} if details else {},
            'details':details}


def _run_variant(folder, definition, arrays, cfg, protocol, records, shared_bases,
                 grid, maximum_drop, fingerprint, family_fingerprint, resume):
    folder = Path(folder)
    if folder.exists() and any(folder.iterdir()):
        state_path, spec_path = folder/'run_state.json', folder/'run_spec.json'
        if not state_path.is_file() or not spec_path.is_file():
            raise ValueError('Refusing unmanaged nonempty experiment directory: '+str(folder))
        saved = json.loads(spec_path.read_text(encoding='utf-8'))
        if saved.get('fingerprint') != fingerprint:
            raise ValueError('Experiment inputs/code changed; choose a new output root: '+str(folder))
        if not resume:
            raise ValueError('Existing run is not overwritten; use a new output root or allow resume.')
        status = json.loads(state_path.read_text(encoding='utf-8'))
        required = ('config.json','calibration.json','protocol.json','metrics.json',
                    'stage_metrics.json','predictions.npz','diagnostics.json')
        if definition['kind'] != 'fusion':
            required += ('diagnostics.npz',)
        if definition.get('text') == 'attribute_groups':
            required += ('group_diagnostics.json',)
        if status.get('status') == 'completed' and all((folder/name).is_file() for name in required):
            return json.loads((folder/'metrics.json').read_text(encoding='utf-8')), 'skipped'
        if status.get('status') not in ('running','failed','completed'):
            raise ValueError('Unrecognized saved run status; use a new output root: '+str(folder))
        # A matching managed partial run may be recomputed. Only this run's
        # named artifacts are atomically replaced; completed peers are reused.
    folder.mkdir(parents=True,exist_ok=True)
    spec = {'fingerprint':fingerprint,'family_fingerprint':family_fingerprint,
            'definition':definition,'protocol_sha256':protocol['sha256']}
    _write(folder/'run_spec.json',spec)
    _write(folder/'config.json',{'definition':definition,'blend_grid':list(grid),
                               'maximum_group_drop_pp':maximum_drop,'cfg_yaml':cfg.dump()})
    _write(folder/'protocol.json',protocol)
    _write(folder/'run_state.json',{'status':'running','fingerprint':fingerprint})
    start = time.perf_counter()
    try:
        fitted = fit_variant(definition,arrays,cfg,protocol,shared_bases)
        if fitted['kind']=='typed':
            shared_bases.setdefault(definition['target_rank'],fitted['state']['basis'])
        candidates, stages, predictions = _evaluate(fitted,definition,arrays,cfg,protocol,records,grid)
        winner = select_candidate(candidates,maximum_drop)
        strength = winner['blend']
        _write(folder/'calibration.json',{'data_source':'base_training_only','fit_seed':int(cfg.SEED),
            'selection_objective':'mean_stage_overall_accuracy','selected_blend':strength,
            'maximum_group_drop_pp':maximum_drop,'protocol_sha256':protocol['sha256'],
            'fit_class_ids':protocol['pseudo_base_class_ids'],
            'validation_class_ids':protocol['pseudo_validation_class_ids'],
            'candidate_results':candidates,'benchmark_evaluation_performed':False,
            'selection_is_development_result':True,'all_base_refit_performed':False})
        _write(folder/'stage_metrics.json',{str(k):v for k,v in stages.items()})
        diagnostics = _model_diagnostics(fitted,protocol)
        diagnostics['teacher_diagnostic']=_teacher_diagnostics(fitted,arrays,cfg,protocol,records,strength)
        _write(folder/'diagnostics.json',diagnostics)
        _save_diagnostic_arrays(folder/'diagnostics.npz',fitted)
        predictions['prediction'] = predictions['candidate_prediction'][list(grid).index(strength)]
        np.savez_compressed(folder/'predictions.npz',**predictions)
        group_diagnostics={}
        if fitted['kind']=='typed' and definition.get('text')=='attribute_groups':
            for group in GROUPS:
                reduced, _, _ = _evaluate(fitted,definition,arrays,cfg,protocol,records,(strength,),[group])
                group_diagnostics[group]={'same_frozen_model_and_blend':True,
                    'without_group':reduced[0],
                    'full_minus_without_accuracy_pp':winner['accuracy']-reduced[0]['accuracy']}
            _write(folder/'group_diagnostics.json',group_diagnostics)
        metrics={'selected_blend':strength,'summary':{k:v for k,v in winner.items() if k not in ('blend','feasible')},
                 'wall_seconds':time.perf_counter()-start,'data_source':'base_training_only',
                 'protocol_sha256':protocol['sha256'],'benchmark_evaluation_performed':False,
                 'gradient_training_performed':False,'reference_scoring':'algebraic_BiMC_same_all_candidates',
                 'diagnostics':{k:diagnostics.get(k) for k in ('kind','target_rank','observation_dimension','retained_statistic_scalars')}}
        _write(folder/'metrics.json',metrics)
        _write(folder/'run_state.json',{'status':'completed','fingerprint':fingerprint,
                                      'wall_seconds':metrics['wall_seconds']})
        return metrics,'completed'
    except Exception as error:
        _write(folder/'run_state.json',{'status':'failed','fingerprint':fingerprint,'error':str(error)})
        _write(folder/'diagnostics.json',{'error':str(error),'traceback':traceback.format_exc()})
        raise


def _source_hashes():
    names = ('engine/multilayer_completion_experiments.py','models/multievidence_completion.py',
             'models/prototype_completion.py','models/clip/model.py','models/bimc.py',
             'engine/completion_calibration.py','engine/prototype_evaluation.py',
             'engine/safd_metric_training.py','engine/residual_training.py',
             'tools/cache_multilayer_base_features.py','utils/attribute_descriptions.py',
             'tools/run_multilayer_completion_experiments.py')
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}


def _summary(root, rows, fit_seeds, stages, selection, family):
    rows=sorted(rows.values(),key=lambda r:(r['fit_seed'],('repeat','layers','attributes').index(r['stage']),r['variant']))
    expected=[{'fit_seed':seed,'stage':stage,'variant':variant['name'],
               'output_dir':f"seed_{seed}/{stage}/{variant['name']}"}
              for seed in fit_seeds for stage in stages for variant in experiment_variants(stage)]
    started={(r['fit_seed'],r['stage'],r['variant']) for r in rows}
    pending=sum((r['fit_seed'],r['stage'],r['variant']) not in started for r in expected)
    result={'schema_version':1,'method':FORMAT,'data_source':'base_training_only',
            'benchmark_evaluation_performed':False,'gradient_training_performed':False,
            'family_fingerprint':family,'fit_seeds':list(fit_seeds),'requested_stages':list(stages),
            'planned_runs':len(expected),'expected_runs':expected,'pending_runs':pending,
            'completed_runs':sum(r['status'] in ('completed','skipped') for r in rows),
            'failed_runs':sum(r['status'] not in ('completed','skipped') for r in rows),
            'runs':rows,'layer_selection':selection,
            'metric_note':'Base validation selection results, not benchmark accuracy. Stage-query observations repeat.',
            'reference_note':'Algebraic BiMC scorer; all candidates use identical reference/statistics; zero is exact.'}
    result['study_complete']=result['completed_runs']==result['planned_runs'] and result['failed_runs']==0
    aggregates={}
    for row in rows:
        if row['status'] not in ('completed','skipped'):
            continue
        key=row['stage']+'/'+row['variant']
        aggregates.setdefault(key,[]).append(row['summary'])
    result['aggregates']={key:{'n':len(values),**{m:sum(v[m] for v in values)/len(values)
        for m in ('accuracy','base_accuracy','novel_accuracy','harmonic_accuracy','delta_accuracy_pp')}}
        for key,values in aggregates.items()}
    lookup={(r['fit_seed'],r['stage'],r['variant']):r for r in rows if r['status'] in ('completed','skipped')}
    pairs=[]
    for stage,reference in (('repeat','original16'),('layers','typed_final'),('attributes','final_ordinary')):
        for name in sorted({r['variant'] for r in rows if r['stage']==stage}):
            pairs.append((stage,name,reference))
    pairs += [('repeat','typed_repeat','typed_single'),('repeat','repeat_attribute','repeat_same'),
              ('repeat','single_two_text','repeat_attribute'),('layers','score_fusion','score_fusion_final')]
    pairs += [('attributes',prefix+'attribute_groups',prefix+'random_groups') for prefix in ('final_','middle_')]
    pairs += [('attributes','middle_'+text,'final_'+text) for text in ('ordinary','attribute_mean','attribute_groups','random_groups')]
    comparisons={}
    for stage,method,reference in pairs:
        paired=[(seed,lookup[seed,stage,method]['summary'],lookup[seed,stage,reference]['summary']) for seed in fit_seeds
                if (seed,stage,method) in lookup and (seed,stage,reference) in lookup]
        if paired:
            comparisons[stage+'/'+method+'-minus-'+reference]={'n':len(paired),
                'per_fit_seed':{str(seed):{m:a[m]-b[m] for m in ('accuracy','base_accuracy','novel_accuracy','harmonic_accuracy')}
                                for seed,a,b in paired},
                'mean_delta_pp':{m:sum(a[m]-b[m] for _,a,b in paired)/len(paired)
                                 for m in ('accuracy','base_accuracy','novel_accuracy','harmonic_accuracy')}}
    result['paired_comparisons']=comparisons
    _write(Path(root)/'summary.json',result)
    with (Path(root)/'results.csv').open('w',encoding='utf-8',newline='') as handle:
        fields=['fit_seed','stage','variant','status','selected_blend','accuracy','base_accuracy',
                'novel_accuracy','harmonic_accuracy','delta_accuracy_pp','corrected','damaged']
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:row.get(k,row.get('summary',{}).get(k,'')) for k in fields})
    lines=['# Multi-layer / multi-attribute base development', '',
           '**These are base-only development/selection results, not FSCIL benchmark results.**','',
           f"Completed: {result['completed_runs']}/{result['planned_runs']}; failed: {result['failed_runs']}.",'',
           '| Stage / variant | N | Overall | Base | Novel | H | Delta overall pp |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for key,v in result['aggregates'].items():
        lines.append(f"| {key} | {v['n']} | {v['accuracy']:.4f} | {v['base_accuracy']:.4f} | "
                     f"{v['novel_accuracy']:.4f} | {v['harmonic_accuracy']:.4f} | {v['delta_accuracy_pp']:+.4f} |")
    lines+=['','Layer selection (class-disjoint BASE validation only):','',
            '```json',json.dumps(selection,ensure_ascii=False,indent=2),'```','',
            'Inspect paired same-seed differences, unknown-attribute coverage, candidate strengths, '
            'corrected/damaged observations, covariance spectra and leave-one-group-out diagnostics. '
            'Do not select methods using future benchmark labels.','',
            'Native middle-layer and text measurements have different coordinates. '
            'Q cross-fitting leaves mapping class pairs out while PCA and pooled sampling noise use the A pool; '
            'its finite-mean correction is approximate.','']
    lines+=['## Paired differences from controls','',
            '| Method minus control | N | Overall pp | Base pp | Novel pp | H pp |',
            '|---|---:|---:|---:|---:|---:|']
    for key,value in comparisons.items():
        v=value['mean_delta_pp']
        lines.append(f"| {key} | {value['n']} | {v['accuracy']:+.4f} | {v['base_accuracy']:+.4f} | "
                     f"{v['novel_accuracy']:+.4f} | {v['harmonic_accuracy']:+.4f} |")
    (Path(root)/'analysis.md').write_text('\n'.join(lines),encoding='utf-8')
    _write(Path(root)/'plan.json',{'runs':expected,'family_fingerprint':family})
    return result


def run_experiments(cache_path, output_root, *, fit_seeds=(2,3,4),
                    stages=('repeat','layers','attributes'),sequences=20,device='cpu',threads=1,
                    maximum_group_drop_pp=.2,blend_grid=(0.,.25,.5,1.),selected_layer=None,resume=True):
    """Run a resumable staged study; every image/label comes from base training."""
    if (not fit_seeds or len(set(fit_seeds))!=len(fit_seeds)
            or any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in fit_seeds)):
        raise ValueError('Use distinct positive fit seeds.')
    if (not stages or len(set(stages))!=len(stages) or set(stages)-{'repeat','layers','attributes'}):
        raise ValueError('Use distinct repeat/layers/attributes stages.')
    if (isinstance(sequences,bool) or not isinstance(sequences,int) or sequences<1
            or isinstance(threads,bool) or not isinstance(threads,int) or threads<1):
        raise ValueError('sequences and threads must be positive integers.')
    if (not blend_grid or 0 not in blend_grid or len(set(blend_grid))!=len(blend_grid)
            or any(isinstance(v,bool) or not math.isfinite(v) or not 0<=v<=1 for v in blend_grid)):
        raise ValueError('blend_grid must contain zero and distinct finite values in [0,1].')
    if not math.isfinite(maximum_group_drop_pp) or maximum_group_drop_pp<0:
        raise ValueError('maximum_group_drop_pp must be nonnegative.')
    if selected_layer not in (None,3,6,9) or device not in ('cpu','cuda'):
        raise ValueError('Unsupported selected_layer or device.')
    _runtime()
    from tools.cache_multilayer_base_features import load_cache
    from engine.completion_calibration import completion_protocol
    cached,metadata=load_cache(cache_path,require_complete=True)
    if not metadata.get('config_yaml') or metadata.get('benchmark_evaluation_performed') is not False:
        raise ValueError('Cache must record its complete base-only configuration and data source.')
    for layer in (3,6,9):
        if ('layers' in stages or ('attributes' in stages and selected_layer in (None,layer))) and 'layer_'+str(layer) not in cached:
            raise ValueError('Cache is missing native layer '+str(layer)+'. Rebuild selected-layer cache.')
    root=Path(output_root).resolve()
    source=_source_hashes()
    family=_digest({'format':FORMAT,'cache_sha256':metadata['cache_sha256'],
        'metadata_sha256':metadata.get('metadata_payload_sha256'),'fit_seeds':list(fit_seeds),
        'sequences':sequences,'device':device,'threads':threads,
        'maximum_group_drop_pp':maximum_group_drop_pp,'blend_grid':list(blend_grid),
        'selected_layer_policy':selected_layer,'source':source})
    previous={}
    if (root/'summary.json').is_file():
        previous=json.loads((root/'summary.json').read_text(encoding='utf-8'))
        if previous.get('family_fingerprint')!=family:
            raise ValueError('Cache/code/development protocol changed; choose a new output root.')
    if (root/'input_provenance.json').is_file():
        supplied=json.loads((root/'input_provenance.json').read_text(encoding='utf-8'))
        if supplied.get('family_fingerprint')!=family:
            raise ValueError('Existing input provenance differs; choose a new output root.')
    root.mkdir(parents=True,exist_ok=True)
    _write(root/'input_provenance.json',{'method':FORMAT,'family_fingerprint':family,
        'cache_path':str(Path(cache_path).resolve()),'cache_sha256':metadata['cache_sha256'],
        'cache_metadata':metadata,'source_hashes':source,'data_source':'base_training_only',
        'selected_layer_policy':selected_layer,
        'benchmark_evaluation_performed':False,'full_feature_cache_in_bundle':False})
    _write(root/'cache_metadata.json',metadata)
    _write(root/'attribute_manifest.json',metadata.get('attribute_manifest',{}))
    _write(root/'source_hashes.json',source)
    for name in source:
        destination=root/'source_snapshot'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes((ROOT/name).read_bytes())
    arrays={k:torch.as_tensor(v,device=device) for k,v in cached.items()}
    if arrays['final'].shape[0]!=len(arrays['labels']):
        raise ValueError('Cache sample arrays are not aligned.')
    rows={(r['fit_seed'],r['stage'],r['variant']):r for r in previous.get('runs',[])}
    selections=previous.get('layer_selection',{})
    all_stages=tuple(name for name in ('repeat','layers','attributes')
                     if name in set(stages)|set(previous.get('requested_stages',[])))
    _summary(root,rows,fit_seeds,all_stages,selections,family)
    grid=tuple(sorted(float(v) for v in blend_grid))
    context={}
    old_threads=torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        for seed in fit_seeds:
            cfg=_configuration(metadata,seed,sequences)
            if cfg.DATASET.NUM_INIT_CLS!=len(metadata['base_class_ids']):
                raise ValueError('Cache base classes differ from the stored dataset configuration.')
            protocol=completion_protocol(cfg,arrays['labels'],cached['sample_id'].tolist())
            records=build_records(arrays,cfg,protocol)
            context[seed]={'cfg':cfg,'protocol':protocol,'records':records,'bases':{}}

        def run_one(seed,definition):
            ctx=context[seed]
            relative=f"seed_{seed}/{definition['stage']}/{definition['name']}"
            fingerprint=_digest({'family':family,'definition':definition,'seed':seed,
                                 'protocol':ctx['protocol']['sha256']})
            record={'fit_seed':seed,'stage':definition['stage'],'variant':definition['name'],
                    'output_dir':relative,'fingerprint':fingerprint}
            try:
                metrics,status=_run_variant(root/relative,definition,arrays,ctx['cfg'],ctx['protocol'],
                    ctx['records'],ctx['bases'],grid,maximum_group_drop_pp,fingerprint,family,resume)
                record.update(status=status,summary=metrics['summary'],selected_blend=metrics['selected_blend'],
                              protocol_sha256=metrics['protocol_sha256'],wall_seconds=metrics['wall_seconds'])
                rows[seed,definition['stage'],definition['name']]=record
                _summary(root,rows,fit_seeds,all_stages,selections,family)
                print(f"{definition['stage']}/{definition['name']} fit={seed} {status}: "
                      f"overall={metrics['summary']['accuracy']:.4f} blend={metrics['selected_blend']}",flush=True)
                return record
            except Exception as error:
                record.update(status='failed',error=str(error))
                rows[seed,definition['stage'],definition['name']]=record
                _summary(root,rows,fit_seeds,all_stages,selections,family)
                raise

        if 'repeat' in stages:
            for seed in fit_seeds:
                for definition in experiment_variants('repeat'):
                    run_one(seed,definition)
        if 'layers' in stages:
            for seed in fit_seeds:
                for definition in experiment_variants('layers'):
                    if definition['kind']!='fusion':
                        run_one(seed,definition)
            gains={layer:sum(rows[seed,'layers','layer'+str(layer)]['summary']['accuracy']
                            -rows[seed,'layers','typed_final']['summary']['accuracy'] for seed in fit_seeds)/len(fit_seeds)
                   for layer in (3,6,9)}
            chosen=selected_layer if selected_layer is not None else max((3,6,9),key=lambda v:(gains[v],-v))
            status=('manual_predeclared' if selected_layer is not None else
                    'base_validation_gain' if gains[chosen]>1e-12 else 'no_gain_diagnostic_layer')
            for seed in fit_seeds:
                selections[str(seed)]={'selected_layer':chosen,'selection_status':status,
                    'mean_paired_gains_vs_typed_final_pp':gains,
                    'selection_scope':'same_global_layer_from_mean_base_validation_across_fit_seeds',
                    'fit_seeds':list(fit_seeds),'family_fingerprint':family}
            _write(root/'layer_selection.json',selections)
            for seed in fit_seeds:
                for definition in experiment_variants('layers',chosen):
                    if definition['kind']=='fusion':
                        run_one(seed,definition)
        if 'attributes' in stages:
            for seed in fit_seeds:
                if selected_layer is not None:
                    chosen=selected_layer
                    selections[str(seed)]={'selected_layer':chosen,'selection_status':'manual_predeclared',
                        'family_fingerprint':family}
                elif str(seed) in selections and selections[str(seed)].get('family_fingerprint')==family:
                    chosen=int(selections[str(seed)]['selected_layer'])
                else:
                    raise ValueError('Attributes need matching base-only layer selection; run layers first or predeclare --selected-layer.')
                for definition in experiment_variants('attributes',chosen):
                    run_one(seed,definition)
            _write(root/'layer_selection.json',selections)
        return _summary(root,rows,fit_seeds,all_stages,selections,family)
    finally:
        torch.set_num_threads(old_threads)
