"""Base-only attribute-channel transfer, with separately A-fitted group offsets.

This is a controlled FDR-inspired experiment, not a reproduction of FDR or
LGSP. Importing the planner needs only Python's standard library.
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
FORMAT = 'attribute_transfer_development_v1'
GROUPS = ('shape', 'part', 'color', 'pattern')
METHODS = ('bimc', 'single_two_text', 'whole_transfer', 'random_mask',
           'shuffled_mask', 'semantic_mask', 'semantic_reliable')


def experiment_variants():
    return [{'name': method+'__'+offset, 'method': method, 'offset_mode': offset,
             'stage': 'transfer', 'kind': 'attribute_transfer'}
            for method in METHODS for offset in ('none', 'a_inner')]


def _runtime():
    global np, torch, F, base
    import numpy as np
    import torch
    from torch.nn import functional as F
    from engine import multilayer_completion_experiments as base
    base._runtime()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode('utf-8')).hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def inner_protocol(arrays, cfg, parent, sequences):
    """Use only outer A fitting rows; reserve inner-old queries from donors."""
    classes = list(parent['pseudo_base_class_ids'])
    if len(classes) < 3:
        raise ValueError('A-only offset development needs at least three fitting classes.')
    rng = random.Random(int(parent['seed'])+900001)
    shuffled = rng.sample(classes, len(classes))
    count = max(2, len(classes)//2)
    old, novel = sorted(shuffled[:count]), sorted(shuffled[count:])
    fitting = set(parent['fitting_indices'])
    labels = arrays['labels'].detach().cpu().tolist()
    rows = {c:[i for i in sorted(fitting) if labels[i]==c] for c in classes}
    shot, query = int(cfg.DATASET.NUM_INC_SHOT), int(cfg.TRAINER.BiMC.PROTOTYPE_COMPLETION.QUERY)
    reserved = {}
    for c in old:
        if len(rows[c]) < query+2:
            raise ValueError('Insufficient A samples for inner old support/query.')
        reserved[c] = rng.sample(rows[c], query)
    for c in novel:
        if len(rows[c]) < shot+query:
            raise ValueError('Insufficient A samples for inner novel support/query.')
    # Preserve the outer arrival trajectory at the smaller inner class count.
    # CUB: outer 10 x 5-way becomes inner alternating 2/3-way, not 5 x 5-way.
    total, cumulative, previous, sizes = sum(parent['stage_sizes']), 0, 0, []
    for size in parent['stage_sizes']:
        cumulative += size
        end = cumulative*len(novel)//total
        if end > previous:
            sizes.append(end-previous)
        previous = end
    result = []
    for _ in range(sequences):
        order = rng.sample(novel, len(novel))
        supports = [[i for i in rows[c] if i not in set(reserved[c])] for c in old]
        queries = [reserved[c] for c in old]
        for c in order:
            chosen = rng.sample(rows[c], shot+query)
            supports.append(chosen[:shot])
            queries.append(chosen[shot:])
        result.append({'class_ids':old+order,'support_indices':supports,'query_indices':queries,'stage_sizes':sizes})
    protocol = {'seed':int(parent['seed'])+900001,'source':'base_training_only',
        'purpose':'A_only_offset_fitting_not_V_selection', 'parent_protocol_sha256':parent['sha256'],
        'pseudo_base_class_ids':old,'pseudo_validation_class_ids':novel,'stage_sizes':sizes,
        'sample_ids':arrays['sample_id'].detach().cpu().tolist(),'audit_indices':{},'sequences':result,
        'fitting_indices':[i for c in old for i in rows[c] if i not in set(reserved[c])],
        'reserved_pseudo_base_query_indices':{str(c):v for c,v in reserved.items()},
        'allowed_outer_fitting_indices':sorted(fitting),
        'fitting_rule':'inner_old_A_classes_only_excluding_reserved_queries'}
    protocol['sha256'] = _digest(protocol)
    return protocol


def attribute_text(arrays, grounding):
    text, available = base._pool(arrays['attr_text'], arrays['attr_mask'])
    available = available & grounding['group_available'].bool()
    return text, available


def fit_method(method, arrays, grounding, cfg, protocol, topk, temperature):
    if method == 'bimc':
        return {'kind':'identity'}
    if method == 'single_two_text':
        return base.fit_variant({'kind':'legacy','mode':'pooled','rank':16}, arrays, cfg, protocol, {})
    from models.attribute_transfer import fit_attribute_transfer
    rows, ids = protocol['fitting_indices'], protocol['pseudo_base_class_ids']
    text, available = attribute_text(arrays, grounding)
    state = fit_attribute_transfer(arrays['final'][rows], arrays['labels'][rows],
        text[ids], available[ids], ids, topk=topk, temperature=temperature)
    return {'kind':'transfer','state':state,'text':text,'available':available,
            'masks':grounding['channel_mask'].bool(),'method':method}


def sequence_completion(fitted, arrays, sequence):
    old = len(sequence['class_ids'])-sum(sequence['stage_sizes'])
    ids = sequence['class_ids'][old:]
    supports = sequence['support_indices'][old:]
    if fitted['kind'] == 'identity':
        return torch.stack([arrays['final'][r].mean(0) for r in supports]), {'kind':'identity'}
    if fitted['kind'] == 'legacy':
        return base.sequence_completions(fitted, arrays, sequence), {'kind':'legacy_pooled_two_text'}
    from models.attribute_transfer import complete_attribute_transfer
    rows = [i for group in supports for i in group]
    method = fitted['method']
    mode = {'whole_transfer':'whole','random_mask':'random','shuffled_mask':'shuffled',
            'semantic_mask':'semantic','semantic_reliable':'semantic'}[method]
    return complete_attribute_transfer(fitted['state'], arrays['final'][rows], arrays['labels'][rows],
        fitted['text'][ids], fitted['available'][ids], fitted['masks'][ids], ids,
        mode=mode, reliability=method=='semantic_reliable', return_diagnostics=True)


def _score_bank(fitted, arrays, cfg, protocol, records, grid):
    """Retain group maxima, enough to replay any old/new scalar offset exactly."""
    from engine.completion_calibration import blend_completion
    completions, traces = {}, {}
    for i, seq in enumerate(protocol['sequences']):
        completions[i], traces[str(i)] = sequence_completion(fitted, arrays, seq)
    identity = {k:[] for k in ('sample_id','target','reference_prediction','sequence','stage','is_old')}
    banks = {strength:{k:[] for k in ('best_old_class','best_novel_class','best_old_logscore','best_novel_logscore')}
             for strength in grid}
    boundaries, cursor = [], 0
    for row in records:
        old, end = row['num_old'], len(row['class_ids'])
        classes = torch.tensor(row['class_ids'], device=arrays['final'].device)
        target = classes[row['targets']]
        reference = row['reference_prototypes']
        for strength in grid:
            if strength == 0 or fitted['kind']=='identity':
                scores = row['reference_votes']
            else:
                novel = blend_completion(reference[old:], row['raw'][old:],
                                          completions[row['sequence']][:end-old], strength)
                scores = base._scorer(row, cfg, torch.cat((reference[:old],novel)))
            a, ai = scores[:,:old].max(1)
            n, ni = scores[:,old:].max(1)
            bank = banks[strength]
            bank['best_old_class'].append(classes[ai].detach().cpu().numpy())
            bank['best_novel_class'].append(classes[ni+old].detach().cpu().numpy())
            # Float64 logs preserve the exact zero-offset winner ordering.
            bank['best_old_logscore'].append(np.log(a.detach().cpu().numpy().astype(np.float64).clip(1e-300)))
            bank['best_novel_logscore'].append(np.log(n.detach().cpu().numpy().astype(np.float64).clip(1e-300)))
        for key, values in (('sample_id',arrays['sample_id'][row['query_rows']]),('target',target),
                            ('reference_prediction',classes[row['reference_votes'].argmax(1)]),
                            ('is_old',row['targets']<old)):
            identity[key].append(values.detach().cpu().numpy())
        for key in ('sequence','stage'):
            identity[key].append(np.full(len(target),row[key],dtype=np.int64))
        boundaries.append((cursor,cursor+len(target)))
        cursor += len(target)
    result = {k:np.concatenate(v) for k,v in identity.items()}
    result['boundaries'] = boundaries
    result['banks'] = {b:{k:np.concatenate(v) for k,v in bank.items()} for b,bank in banks.items()}
    zero = _predict(result['banks'][0.],0.)
    if not np.array_equal(zero,result['reference_prediction']):
        raise ValueError('Zero correction group-score reconstruction differs from reference.')
    return result, completions, traces


def _predict(bank, offset):
    return np.where(bank['best_novel_logscore']+offset > bank['best_old_logscore'],
                    bank['best_novel_class'],bank['best_old_class'])


def _measure(bundle, strength, offset):
    bank = bundle['banks'][strength]
    prediction = _predict(bank,offset)
    target, old = bundle['target'], bundle['is_old'].astype(bool)
    correct, ref_correct = prediction==target, bundle['reference_prediction']==target
    choose_new = bank['best_novel_logscore']+offset > bank['best_old_logscore']
    stage_rows = []
    for begin,end in bundle['boundaries']:
        sl = slice(begin,end)
        mask, c = old[sl], correct[sl]
        a, n = float(c[mask].mean()*100), float(c[~mask].mean()*100)
        row = {'sequence':int(bundle['sequence'][begin]),'stage':int(bundle['stage'][begin]),
            'accuracy':float(c.mean()*100),'base_accuracy':a,'novel_accuracy':n,
            'harmonic_accuracy':2*a*n/(a+n) if a+n else 0.,
            'base_within_group_accuracy':float((bank['best_old_class'][sl][mask]==target[sl][mask]).mean()*100),
            'novel_within_group_accuracy':float((bank['best_novel_class'][sl][~mask]==target[sl][~mask]).mean()*100)}
        stage_rows.append(row)
    metrics = {k:sum(row[k] for row in stage_rows)/len(stage_rows) for k in (
        'accuracy','base_accuracy','novel_accuracy','harmonic_accuracy',
        'base_within_group_accuracy','novel_within_group_accuracy')}
    metrics.update(corrected=int((correct & ~ref_correct).sum()),damaged=int((~correct & ref_correct).sum()),
        changed=int((prediction!=bundle['reference_prediction']).sum()),query_occurrences=len(target),
        old_to_novel_errors=int((old & choose_new).sum()),novel_to_old_errors=int((~old & ~choose_new).sum()),
        old_within_errors=int((old & ~choose_new & ~correct).sum()),
        novel_within_errors=int((~old & choose_new & ~correct).sum()),stages=len(stage_rows),
        exact_reference_prediction=bool(np.array_equal(prediction,bundle['reference_prediction'])))
    return metrics, stage_rows, prediction


def fit_offsets(bundle, grid, offset_grid, maximum_drop):
    reference = _measure(bundle,0.,0.)[0]
    results, offsets = [], {}
    for strength in grid:
        candidates = []
        for value in offset_grid:
            metrics = _measure(bundle,strength,value)[0]
            feasible = min(metrics['base_accuracy']-reference['base_accuracy'],
                           metrics['novel_accuracy']-reference['novel_accuracy']) >= -maximum_drop-1e-12
            candidates.append({'offset':value,'feasible':feasible,**metrics})
        eligible = [r for r in candidates if r['feasible']] or candidates
        best = max(r['accuracy'] for r in eligible)
        winner = min((r for r in eligible if r['accuracy']>=best-1e-12),key=lambda r:(abs(r['offset']),r['offset']))
        offsets[strength] = winner['offset']
        results.append({'blend':strength,'selected_offset':winner['offset'],
                        'selection_has_feasible_A_candidate':any(r['feasible'] for r in candidates),
                        'candidate_results':candidates})
    return offsets, results


def _candidate_specs(grid, offset_mode, offsets):
    result = [{'candidate_id':'identity','blend':0.,'offset':0.,'identity_fallback':True}]
    for i,strength in enumerate(grid):
        offset = offsets[strength] if offset_mode=='a_inner' else 0.
        if strength==0 and offset==0:
            continue
        result.append({'candidate_id':'candidate_'+str(i),'blend':strength,'offset':offset,'identity_fallback':False})
    return result


def _select(candidates, maximum_drop):
    reference = candidates[0]
    for candidate in candidates:
        for metric in ('accuracy','base_accuracy','novel_accuracy','harmonic_accuracy'):
            candidate['delta_'+metric+'_pp'] = candidate[metric]-reference[metric]
        candidate['feasible'] = min(candidate['delta_base_accuracy_pp'],candidate['delta_novel_accuracy_pp']) >= -maximum_drop-1e-12
    valid = [r for r in candidates if r['feasible']]
    best = max(r['accuracy'] for r in valid)
    return min((r for r in valid if r['accuracy']>=best-1e-12),
               key=lambda r:(r['blend'],abs(r['offset']),not r['identity_fallback']))


def teacher_diagnostics(arrays, protocol, records, completions, grid):
    """Run after V selection, for every blend; no teacher feedback to selection."""
    from engine.completion_calibration import blend_completion
    teachers, values = [], {b:[] for b in grid}
    for index,seq in enumerate(protocol['sequences']):
        row = next(r for r in reversed(records) if r['sequence']==index)
        old = row['num_old']
        centers, positions = [], []
        for j,cls in enumerate(row['class_ids'][old:],old):
            used = set(seq['support_indices'][j]) | set(seq['query_indices'][j])
            pool = [i for i in torch.where(arrays['labels']==cls)[0].cpu().tolist() if i not in used]
            chosen = random.Random(protocol['seed']+index*1009+cls*31).sample(pool,min(15,len(pool)))
            if len(chosen)<2:
                continue
            centers.append(F.normalize(arrays['final'][chosen].mean(0),dim=-1))
            positions.append(j)
            teachers.append({'sequence':index,'class_id':cls,'teacher_count':len(chosen),
                             'sample_ids':arrays['sample_id'][chosen].cpu().tolist()})
        if not centers:
            continue
        center = torch.stack(centers)
        raw = F.normalize(row['raw'][positions],dim=-1)
        reference = row['reference_prototypes'][positions]
        complete = F.normalize(completions[index][[j-old for j in positions]],dim=-1)
        for strength in grid:
            novel = blend_completion(row['reference_prototypes'][old:],row['raw'][old:],completions[index],strength)
            bank = torch.cat((row['reference_prototypes'][:old],novel))
            similarities = center @ bank.T
            own = similarities[torch.arange(len(center),device=center.device),positions].clone()
            similarities[torch.arange(len(center),device=center.device),positions] = -torch.inf
            candidate = bank[positions]
            for k,j in enumerate(positions):
                values[strength].append({'sequence':index,'class_id':row['class_ids'][j],
                    'raw_cosine_error':float(1-(raw[k]*center[k]).sum()),
                    'reference_cosine_error':float(1-(reference[k]*center[k]).sum()),
                    'completed_cosine_error':float(1-(complete[k]*center[k]).sum()),
                    'candidate_cosine_error':float(1-(candidate[k]*center[k]).sum()),
                    'candidate_center_margin':float(own[k]-similarities[k].max()),
                    'correction_norm':float((candidate[k]-reference[k]).norm())})
    names = ('raw_cosine_error','reference_cosine_error','completed_cosine_error',
             'candidate_cosine_error','candidate_center_margin','correction_norm')
    return {'post_selection_only':True,'used_for_selection':False,'strict_performance_upper_bound':False,
        'disjoint_scope':'Within each sequence; teachers may occur in other sequences.',
        'offset_effect':'Offsets change decisions but not prototype geometry.', 'teachers':teachers,
        'candidate_geometry':[{'blend':b,'available':bool(rows),'class_occurrences':len(rows),
            'means':{k:sum(r[k] for r in rows)/len(rows) for k in names} if rows else {},'details':rows}
            for b,rows in values.items()]}


def _source_hashes():
    names = ('engine/attribute_transfer_experiments.py','models/attribute_transfer.py',
        'utils/attribute_grounding.py','tools/cache_attribute_grounding.py',
        'tools/run_attribute_transfer_experiments.py','tools/package_attribute_transfer_results.py',
        'engine/multilayer_completion_experiments.py','models/prototype_completion.py',
        'engine/completion_calibration.py','engine/prototype_evaluation.py',
        'engine/guidance_calibration.py','engine/safd_metric_training.py','engine/residual_training.py',
        'models/bimc.py','models/clip/model.py','utils/attribute_descriptions.py',
        'tools/cache_multilayer_base_features.py','tools/cache_safd_base_features.py',
        'tools/package_multilayer_completion_results.py')
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}


def _summary(root, rows, fit_seeds, family):
    expected = [{'fit_seed':seed,'stage':'transfer','variant':definition['name'],
                 'output_dir':f"seed_{seed}/transfer/{definition['name']}"}
                for seed in fit_seeds for definition in experiment_variants()]
    ordered = sorted(rows.values(),key=lambda r:(r['fit_seed'],r['variant']))
    success = [r for r in ordered if r['status'] in ('completed','skipped')]
    started = {(r['fit_seed'],r['variant']) for r in ordered}
    pending = [r for r in expected if (r['fit_seed'],r['variant']) not in started]
    result = {'schema_version':1,'method':FORMAT,'family_fingerprint':family,
        'data_source':'base_training_only','benchmark_evaluation_performed':False,
        'gradient_training_performed':False,'fit_seeds':list(fit_seeds),'requested_stages':['transfer'],
        'planned_runs':len(expected),'expected_runs':expected,'completed_runs':len(success),
        'failed_runs':len(ordered)-len(success),'pending_runs':len(pending),
        'study_complete':len(success)==len(expected) and len(success)==len(ordered), 'runs':ordered,
        'selection_note':'A-only inner tasks fit offsets; V selects blend. Teachers are post-selection only.',
        'comparison_note':'FDR-inspired grouped-channel experiment; not an exact FDR/LGSP reproduction.',
        'metric_note':'Development selection accuracy; repeated queries are not independent observations.'}
    aggregate = {}
    for row in success:
        aggregate.setdefault(row['variant'],[]).append(row['summary'])
    keys = ('accuracy','base_accuracy','novel_accuracy','harmonic_accuracy','delta_accuracy_pp')
    result['aggregates'] = {name:{'n':len(values),**{k:sum(v[k] for v in values)/len(values) for k in keys}}
                            for name,values in aggregate.items()}
    lookup = {(r['fit_seed'],r['variant']):r for r in success}
    pairs = [(d['name'],'single_two_text__'+d['offset_mode']) for d in experiment_variants()]
    pairs += [(m+'__a_inner',m+'__none') for m in METHODS]
    pairs += [('semantic_mask__'+o,'random_mask__'+o) for o in ('none','a_inner')]
    pairs += [('semantic_mask__'+o,'shuffled_mask__'+o) for o in ('none','a_inner')]
    pairs += [('semantic_reliable__'+o,'semantic_mask__'+o) for o in ('none','a_inner')]
    result['paired_comparisons'] = {}
    for method,control in pairs:
        values = {str(seed):{k:lookup[seed,method]['summary'][k]-lookup[seed,control]['summary'][k]
                              for k in keys[:4]} for seed in fit_seeds
                  if (seed,method) in lookup and (seed,control) in lookup}
        if values:
            result['paired_comparisons'][method+'-minus-'+control] = {'n':len(values),'per_fit_seed':values,
                'mean_delta_pp':{k:sum(v[k] for v in values.values())/len(values) for k in keys[:4]}}
    _write(root/'summary.json',result)
    _write(root/'plan.json',{'runs':expected,'family_fingerprint':family})
    fields = ['fit_seed','variant','status','selected_blend','selected_offset',*keys[:-1],
              'delta_accuracy_pp','corrected','damaged']
    with (root/'results.csv').open('w',encoding='utf-8',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        for row in ordered:
            writer.writerow({k:row.get(k,row.get('summary',{}).get(k,'')) for k in fields})
    lines = ['# Attribute-channel transfer: base development','',
             '**Development selection results, not FSCIL benchmark accuracy.**','',
             f"Completed {len(success)}/{len(expected)}; pending {len(pending)}; failed {result['failed_runs']}.",'',
             '| Variant | N | Overall | Base | Novel | H | Delta pp |','|---|---:|---:|---:|---:|---:|---:|']
    for name,values in result['aggregates'].items():
        lines.append(f"| {name} | {values['n']} | {values['accuracy']:.4f} | {values['base_accuracy']:.4f} | "
                     f"{values['novel_accuracy']:.4f} | {values['harmonic_accuracy']:.4f} | {values['delta_accuracy_pp']:+.4f} |")
    lines += ['', 'Inspect all nonzero candidate teacher geometry, old/new group scores, donor traces, '
        'matched random/shuffled masks and paired offset-only controls. Zero fallback is not evidence for attributes.',
        '', 'Keyword attention-value channel grounding is a projected attention-path heuristic. '
        'The learned last MLP and residual paths are not attributed to keywords. '
        'Four pooled attribute groups and our residual recomposition differ from original FDR.']
    (root/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


def _run_variant(folder, definition, context, arrays, grounding, options, family, resume):
    folder.mkdir(parents=True,exist_ok=True)
    fingerprint = _digest({'family':family,'definition':definition,'protocol':context['protocol']['sha256'],
                           'inner_protocol':context['inner_protocol']['sha256']})
    required = ('config.json','run_spec.json','run_state.json','calibration.json','protocol.json',
        'metrics.json','stage_metrics.json','predictions.npz','diagnostics.json',
        'teacher_diagnostics.json','offset_fit.json')
    if any(folder.iterdir()):
        if not (folder/'run_spec.json').is_file() or not (folder/'run_state.json').is_file():
            raise ValueError('Refusing unmanaged nonempty run directory: '+str(folder))
        spec = json.loads((folder/'run_spec.json').read_text(encoding='utf-8'))
        if spec.get('fingerprint')!=fingerprint:
            raise ValueError('Run inputs/code changed; use a new output root.')
        if not resume:
            raise ValueError('Existing run is not overwritten; use a new output root.')
        state = json.loads((folder/'run_state.json').read_text(encoding='utf-8'))
        if state.get('status')=='completed' and all((folder/name).is_file() for name in required):
            return json.loads((folder/'metrics.json').read_text(encoding='utf-8')), 'skipped', fingerprint
        if state.get('status') not in ('completed','running','failed'):
            raise ValueError('Unsupported saved run status.')
    _write(folder/'run_spec.json',{'fingerprint':fingerprint,'family_fingerprint':family,
                                  'definition':definition,'protocol_sha256':context['protocol']['sha256']})
    _write(folder/'config.json',{'definition':definition,'settings':options,'cfg_yaml':context['cfg'].dump()})
    _write(folder/'protocol.json',context['protocol'])
    _write(folder/'run_state.json',{'status':'running','fingerprint':fingerprint})
    start = time.perf_counter()
    try:
        method = definition['method']
        grid = (0.,) if method=='bimc' else tuple(options['blend_grid'])
        if method not in context['banks']:
            fitted = fit_method(method,arrays,grounding,context['cfg'],context['protocol'],
                                options['topk'],options['temperature'])
            bank, completions, trace = _score_bank(fitted,arrays,context['cfg'],context['protocol'],context['records'],grid)
            context['banks'][method] = (bank,completions,trace,fitted)
        bank, completions, trace, fitted = context['banks'][method]
        if definition['offset_mode']=='a_inner':
            if method not in context['offsets']:
                innerfit = fit_method(method,arrays,grounding,context['cfg'],context['inner_protocol'],
                                      options['topk'],options['temperature'])
                innerbank, _, _ = _score_bank(innerfit,arrays,context['cfg'],context['inner_protocol'],context['inner_records'],grid)
                offsets, search = fit_offsets(innerbank,grid,options['offset_grid'],options['maximum_group_drop_pp'])
                context['offsets'][method] = offsets,search
            offsets, search = context['offsets'][method]
        else:
            offsets, search = {b:0. for b in grid}, []
        _write(folder/'offset_fit.json',{'data_source':'outer_A_fitting_rows_only',
            'offset_mode':definition['offset_mode'],'inner_protocol':context['inner_protocol'],
            'candidate_results':search,'uses_outer_V_images_or_labels':False,
            'offset_semantics':'Add b to the best novel log-score; old scores unchanged.'})
        specs = _candidate_specs(grid,definition['offset_mode'],offsets)
        candidates, stages, predictions = [], {}, []
        for spec in specs:
            metrics, stage, prediction = _measure(bank,spec['blend'],spec['offset'])
            candidates.append({**spec,**metrics})
            stages[spec['candidate_id']] = stage
            predictions.append(prediction)
        winner = _select(candidates,options['maximum_group_drop_pp'])
        _write(folder/'calibration.json',{'selected_candidate_id':winner['candidate_id'],
            'selected_blend':winner['blend'],'selected_offset':winner['offset'],
            'candidate_results':candidates,'selection_source':'outer_V_queries_and_reserved_outer_A_queries',
            'offset_fit_source':'A_only_inner_tasks','protocol_sha256':context['protocol']['sha256'],
            'benchmark_evaluation_performed':False,'selection_is_development_result':True})
        _write(folder/'stage_metrics.json',stages)
        # Teachers are first read after the winner is fixed, and cannot alter it.
        if method not in context['teachers']:
            context['teachers'][method] = teacher_diagnostics(arrays,context['protocol'],context['records'],completions,grid)
        _write(folder/'teacher_diagnostics.json',context['teachers'][method])
        diagnostics = {'method':method,'grounding_source':'frozen_text_attention_value_or_deletion_sidecar',
            'fit_class_ids':context['protocol']['pseudo_base_class_ids'],
            'fitting_indices_sha256':_digest(context['protocol']['fitting_indices']),
            'support_only_completion':True,'new_image_forward_passes':0,
            'retained_training_samples_in_model':0,'gradient_training_performed':False,
            'unknown_groups_are_omitted':True,'overlap_correction':'average_unique_active_group_masks_per_channel',
            'sequence_transfer_diagnostics':trace}
        _write(folder/'diagnostics.json',diagnostics)
        saved = {k:v for k,v in bank.items() if k not in ('banks','boundaries')}
        saved.update(candidate_id=np.array([r['candidate_id'] for r in specs]),
            candidate_blend=np.array([r['blend'] for r in specs],dtype=np.float64),
            candidate_offset=np.array([r['offset'] for r in specs],dtype=np.float64),
            candidate_prediction=np.stack(predictions),prediction=predictions[specs.index(next(s for s in specs if s['candidate_id']==winner['candidate_id']))])
        for key in ('best_old_class','best_novel_class','best_old_logscore','best_novel_logscore'):
            saved['candidate_'+key] = np.stack([bank['banks'][s['blend']][key] for s in specs])
        with (folder/'predictions.npz').open('wb') as handle:
            np.savez_compressed(handle,**saved)
        metrics = {'selected_candidate_id':winner['candidate_id'],'selected_blend':winner['blend'],
            'selected_offset':winner['offset'],'summary':{k:v for k,v in winner.items() if k not in ('candidate_id','blend','offset','identity_fallback','feasible')},
            'protocol_sha256':context['protocol']['sha256'],'wall_seconds':time.perf_counter()-start,
            'data_source':'base_training_only','benchmark_evaluation_performed':False,'gradient_training_performed':False}
        _write(folder/'metrics.json',metrics)
        _write(folder/'run_state.json',{'status':'completed','fingerprint':fingerprint})
        return metrics,'completed',fingerprint
    except Exception as error:
        _write(folder/'run_state.json',{'status':'failed','fingerprint':fingerprint,'error':str(error)})
        _write(folder/'diagnostics.json',{'error':str(error),'traceback':traceback.format_exc()})
        raise


def run_experiments(cache_path, grounding_path, output_root, *, fit_seeds=(2,3,4),
        sequences=20, inner_sequences=5, device='cpu', threads=1,
        blend_grid=(0.,.05,.1,.25,.5,1.), offset_grid=(-.1,-.05,-.02,0.,.02,.05,.1),
        maximum_group_drop_pp=.2, topk=5, temperature=16., resume=True):
    if not fit_seeds or len(set(fit_seeds))!=len(fit_seeds) or any(isinstance(s,bool) or not isinstance(s,int) or s<1 for s in fit_seeds):
        raise ValueError('fit_seeds must be distinct positive integers.')
    if any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in (sequences,inner_sequences,threads,topk)):
        raise ValueError('Sequence counts, threads and topk must be positive integers.')
    if (not blend_grid or 0. not in blend_grid or len(set(blend_grid))!=len(blend_grid)
        or any(isinstance(v,bool) or not math.isfinite(v) or not 0<=v<=1 for v in blend_grid)):
        raise ValueError('blend_grid must be distinct finite [0,1] values including zero.')
    if (not offset_grid or 0. not in offset_grid or len(set(offset_grid))!=len(offset_grid)
        or any(isinstance(v,bool) or not math.isfinite(v) or abs(v)>10 for v in offset_grid)):
        raise ValueError('offset_grid must be distinct finite values in [-10,10] including zero.')
    if not math.isfinite(maximum_group_drop_pp) or maximum_group_drop_pp<0 or not math.isfinite(temperature) or temperature<=0:
        raise ValueError('Invalid maximum drop or temperature.')
    if device not in ('cpu','cuda'):
        raise ValueError('device must be cpu/cuda.')
    _runtime()
    from tools.cache_multilayer_base_features import load_cache
    from tools.cache_attribute_grounding import load_grounding
    from engine.completion_calibration import completion_protocol
    cached,metadata = load_cache(cache_path,require_complete=True)
    grounded,groundmeta = load_grounding(grounding_path,source_metadata=metadata)
    source = _source_hashes()
    options = {'fit_seeds':list(fit_seeds),'sequences':sequences,'inner_sequences':inner_sequences,
        'device':device,'threads':threads,'blend_grid':sorted(float(v) for v in blend_grid),
        'offset_grid':sorted(float(v) for v in offset_grid),'maximum_group_drop_pp':maximum_group_drop_pp,
        'topk':topk,'temperature':temperature}
    grounding_sha = groundmeta.get('grounding_sha256',groundmeta.get('cache_sha256'))
    family = _digest({'format':FORMAT,'cache_sha256':metadata['cache_sha256'],
        'metadata_sha256':metadata['metadata_payload_sha256'],'grounding_sha256':grounding_sha,
        'grounding_metadata_sha256':groundmeta['metadata_payload_sha256'],'options':options,'source':source})
    root = Path(output_root).resolve()
    previous = {}
    if (root/'summary.json').is_file():
        previous = json.loads((root/'summary.json').read_text(encoding='utf-8'))
        if previous.get('family_fingerprint')!=family:
            raise ValueError('Inputs/code/protocol changed; use a new output root.')
    if (root/'input_provenance.json').is_file():
        existing = json.loads((root/'input_provenance.json').read_text(encoding='utf-8'))
        if existing.get('family_fingerprint')!=family:
            raise ValueError('Existing provenance differs; use a new output root.')
    root.mkdir(parents=True,exist_ok=True)
    _write(root/'input_provenance.json',{'method':FORMAT,'family_fingerprint':family,
        'cache_sha256':metadata['cache_sha256'],'grounding_sha256':grounding_sha,
        'cache_metadata_sha256':metadata['metadata_payload_sha256'],
        'grounding_metadata_sha256':groundmeta['metadata_payload_sha256'],
        'cache_path':str(Path(cache_path).resolve()),'grounding_path':str(Path(grounding_path).resolve()),
        'source_hashes':source,'data_source':'base_training_only','benchmark_evaluation_performed':False})
    _write(root/'cache_metadata.json',metadata)
    _write(root/'grounding_metadata.json',groundmeta)
    _write(root/'grounding_manifest.json',groundmeta.get('grounding_manifest',{}))
    _write(root/'source_hashes.json',source)
    for name in source:
        destination = root/'source_snapshot'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes((ROOT/name).read_bytes())
    arrays = {k:torch.as_tensor(v,device=device) for k,v in cached.items()}
    grounding = {k:torch.as_tensor(v,device=device) for k,v in grounded.items()}
    rows = {(r['fit_seed'],r['variant']):r for r in previous.get('runs',[])}
    _summary(root,rows,fit_seeds,family)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        for seed in fit_seeds:
            cfg = base._configuration(metadata,seed,sequences)
            protocol = completion_protocol(cfg,arrays['labels'],cached['sample_id'].tolist())
            inner = inner_protocol(arrays,cfg,protocol,inner_sequences)
            context = {'cfg':cfg,'protocol':protocol,'inner_protocol':inner,
                'records':base.build_records(arrays,cfg,protocol),'inner_records':base.build_records(arrays,cfg,inner),
                'banks':{},'offsets':{},'teachers':{}}
            for definition in experiment_variants():
                name = definition['name']
                relative = f'seed_{seed}/transfer/{name}'
                record = {'fit_seed':seed,'stage':'transfer','variant':name,'output_dir':relative}
                try:
                    metrics,status,fingerprint = _run_variant(root/relative,definition,context,
                        arrays,grounding,options,family,resume)
                    record.update(status=status,fingerprint=fingerprint,summary=metrics['summary'],
                        selected_blend=metrics['selected_blend'],selected_offset=metrics['selected_offset'],
                        protocol_sha256=metrics['protocol_sha256'],wall_seconds=metrics['wall_seconds'])
                    rows[seed,name] = record
                    _summary(root,rows,fit_seeds,family)
                    print(f"{name} fit={seed} {status}: overall={metrics['summary']['accuracy']:.4f} "
                          f"blend={metrics['selected_blend']} offset={metrics['selected_offset']}",flush=True)
                except Exception as error:
                    record.update(status='failed',error=str(error))
                    rows[seed,name] = record
                    _summary(root,rows,fit_seeds,family)
                    raise
        return _summary(root,rows,fit_seeds,family)
    finally:
        torch.set_num_threads(old_threads)
