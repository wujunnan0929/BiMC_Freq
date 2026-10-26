"""Matched-way validation and A-only feature/probe diagnostics; no benchmark IO."""
import hashlib
import json
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from engine.prototype_evaluation import evaluate_records, score_record
from engine.competitive_evaluation import fusion_record_scores, group_diagnostics
from models.competitive_fusion import evidence_features


@torch.no_grad()
def matched_records(arrays, cfg, protocol, original_records, way_pairs, seed):
    """Subset the same A/V classes, supports and reserved queries by class count.

    Old subsets are nested within a seeded class permutation; novel subsets
    retain the original arrival prefix. Support sizes and covariance arrival
    groups retain the original validation policy. No teacher features are read.
    """
    from engine.safd_metric_training import _reference_proxy, _text_subset
    from engine.residual_training import _group_covariance
    rng, proxy = random.Random(seed+77077), _reference_proxy(cfg)
    sources = {}
    for row in original_records:
        if row['sequence'] not in sources or len(row['class_ids']) > len(sources[row['sequence']]['class_ids']):
            sources[row['sequence']] = row
    pairs = sorted(set(tuple(pair) for pair in way_pairs))
    if not pairs or any(min(pair) < 1 for pair in pairs):
        raise ValueError('Matched validation needs positive observed training-way pairs.')
    result, trace = [], []
    b = cfg.TRAINER.BiMC
    for sequence, source in sorted(sources.items()):
        source_old = source['num_old']
        old_order = list(range(source_old)); rng.shuffle(old_order)
        source_sizes = protocol['sequences'][sequence]['stage_sizes']
        for old, novel in pairs:
            if old > source_old or novel > len(source['class_ids'])-source_old:
                raise ValueError('A training-way pair cannot be formed from the reserved A/V validation bank.')
            positions = old_order[:old]+list(range(source_old,source_old+novel))
            classes = [source['class_ids'][j] for j in positions]
            supports = [source['supports'][j] for j in positions]
            raw = F.normalize(torch.stack([value.mean(0) for value in supports]),dim=-1)
            proto = raw.clone()
            if b.VISION_CALIBRATION:
                proto[old:] = proxy.soft_calibration(raw[:old],raw[old:])
            sizes, remaining = [], novel
            for size in source_sizes:
                if remaining:
                    chosen = min(size,remaining); sizes.append(chosen); remaining -= chosen
            if remaining:
                raise ValueError('Original arrival groups do not cover the matched novel prefix.')
            covariance_sum = old*_group_covariance(torch.cat(supports[:old]),b.GAMMA_BASE)
            start = old
            for size in sizes:
                covariance_sum += size*_group_covariance(torch.cat(supports[start:start+size]),b.GAMMA_INC)
                start += size
            covariance = covariance_sum/(old+novel)
            inverse = torch.pinverse(covariance.float())
            query_parts = [source['query'][source['targets']==j] for j in positions]
            query = torch.cat(query_parts)
            targets = torch.cat([torch.full((len(q),),j,device=query.device,dtype=torch.long) for j,q in enumerate(query_parts)])
            support_indices = [source['support_indices'][j] for j in positions]
            query_indices = [source['query_indices'][j] for j in positions]
            teacher_indices = [protocol['audit_indices'][str(c)] for c in classes[old:]]
            for j,(s,q) in enumerate(zip(support_indices,query_indices)):
                if set(s)&set(q):
                    raise ValueError('Matched support/query indices overlap.')
                if j>=old and (set(teacher_indices[j-old])&(set(s)|set(q))):
                    raise ValueError('Matched task overlaps the reserved teacher bank.')
            global_queries = [i for indices in query_indices for i in indices]
            if not np.array_equal(np.asarray(classes)[targets.cpu().numpy()],np.asarray(arrays['labels'])[global_queries]):
                raise ValueError('Matched local/global query labels disagree.')
            dp,df,dt,text = _text_subset(arrays,classes,query.device)
            votes = proxy.reference_scores_from_features(query,old+novel,old,proto,covariance,
                dp,df,dt,text,cfg.DATASET.BETA,cov_inverse=inverse)
            lam = b.LAMBDA_T if b.TEXT_CALIBRATION else 0.
            task_id = f'seq{sequence}_old{old}_novel{novel}'
            result.append({'class_ids':classes,'num_old':old,'supports':supports,
                'reference_prototypes':proto,'raw_prototypes':raw,
                'semantic_prototypes':F.normalize((1-lam)*text+lam*dp,dim=-1),
                'query':query,'targets':targets,'covariance':covariance,'cov_inverse':inverse,
                'text_features':text,'description_proto':dp,'description_features':df,'description_targets':dt,
                'reference_votes':votes,'sequence':sequence,'stage':len(sizes),'arrival_sizes':sizes,
                'support_indices':support_indices,'query_indices':query_indices,'teacher_indices':teacher_indices,
                'query_sample_ids':np.asarray(arrays['sample_id'])[global_queries].tolist(),'matched_task_id':task_id})
            trace.append({'task_id':task_id,'sequence':sequence,'num_old':old,'class_ids':classes,
                          'arrival_sizes':sizes,'support_indices':support_indices,'query_indices':query_indices,
                          'teacher_indices':teacher_indices,'source_positions':positions})
    report = {'source_protocol_sha256':protocol['sha256'],'seed':seed,'way_pairs':[list(p) for p in pairs],
              'tasks':trace,'rule':'seeded_nested_old_subsets_and_original_novel_prefix',
              'supports_queries_and_arrival_policy_retained':True,'teacher_features_used':False}
    report['sha256'] = hashlib.sha256(json.dumps(report,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return result,report


@torch.no_grad()
def bank_baselines(records,cfg,calibration,fixed):
    return {'original':evaluate_records(records,lambda row:row['reference_votes']),
            'calibrated':evaluate_records(records,lambda row:score_record(row,cfg,calibration=calibration)),
            'fixed':evaluate_records(records,lambda row:fusion_record_scores(row,cfg,calibration,fixed))}


def bank_coverage(records):
    pairs = sorted({(row['num_old'],len(row['class_ids'])-row['num_old']) for row in records})
    totals = [sum(pair) for pair in pairs]
    return {'tasks':len(records),'old_novel_class_pairs':[list(pair) for pair in pairs],
            'total_class_range':[min(totals),max(totals)],
            'repeated_query_observations':sum(len(row['targets']) for row in records),
            'unique_query_indices':len({i for row in records for q in row['query_indices'] for i in q})}


@torch.no_grad()
def record_features(model,row):
    return evidence_features(row['fusion_probabilities'],row['num_old'],row['fusion_quality'],model.use_support_quality)


@torch.no_grad()
def feature_shift(model,records):
    x = torch.cat([model.before_standardization(record_features(model,row)) for row in records])
    z = (x-model.evidence_mean)/model.evidence_scale
    return {'observations':len(x),'mean_absolute_z':float(z.abs().mean()),
            'fraction_coordinates_outside_clip':float((z.abs()>model.standardization_clip).float().mean()),
            'per_feature_mean_z':z.mean(0).cpu().tolist(),
            'per_feature_outside_clip':(z.abs()>model.standardization_clip).float().mean(0).cpu().tolist()}


def binary_auc(scores,positive):
    """Rank AUROC with average ranks for ties; constant predictions give .5."""
    scores,positive = np.asarray(scores,dtype=float),np.asarray(positive,dtype=bool)
    n_pos,n_neg = int(positive.sum()),int((~positive).sum())
    if not n_pos or not n_neg or not np.isfinite(scores).all():
        raise ValueError('AUROC requires finite scores and both groups.')
    order = np.argsort(scores,kind='mergesort')
    ranks = np.empty(len(scores),dtype=float)
    start = 0
    while start<len(scores):
        end = start+1
        while end<len(scores) and scores[order[end]]==scores[order[start]]:
            end+=1
        ranks[order[start:end]]=(start+1+end)/2.
        start=end
    return float((ranks[positive].sum()-n_pos*(n_pos+1)/2.)/(n_pos*n_neg))


def fit_group_probe(model,records,steps=100,lr=.01):
    if steps<1 or lr<=0:
        raise ValueError('Probe budget and learning rate must be positive.')
    features,targets,weights = [],[],[]
    for row in records:
        features.append(model.input_features(record_features(model,row)))
        y = (row['targets']>=row['num_old']).float()
        targets.append(y)
        weights.append(torch.where(y.bool(),.5/y.sum(),.5/(1-y).sum())/len(records))
    x,y,w = torch.cat(features),torch.cat(targets),torch.cat(weights)
    probe = nn.Linear(x.shape[1],1).to(x.device)
    nn.init.zeros_(probe.weight); nn.init.zeros_(probe.bias)
    optimizer = torch.optim.Adam(probe.parameters(),lr=lr)
    losses=[]
    for _ in range(steps):
        loss=(F.binary_cross_entropy_with_logits(probe(x).squeeze(-1),y,reduction='none')*w).sum()
        if not torch.isfinite(loss):
            raise ValueError('Non-finite group probe loss.')
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses.append(float(loss.detach()))
    probe.eval().requires_grad_(False)
    return probe,{'loss_history':losses,'steps':steps,'learning_rate':lr,
        'threshold':0.,'threshold_rule':'fixed_logit_zero; not_validation_tuned',
        'weight':probe.weight.detach().cpu().tolist(),'bias':probe.bias.detach().cpu().tolist(),
        'fitting_source':'A_fitting_queries_only; equal_group_weight_within_each_episode'}


@torch.no_grad()
def evaluate_group_probe(model,probe,records):
    rows,scores,labels = [],[],[]
    for row in records:
        z=probe(model.input_features(record_features(model,row))).squeeze(-1).cpu().numpy()
        y=(row['targets']>=row['num_old']).cpu().numpy()
        prediction=z>=0.
        rows.append({'auc':binary_auc(z,y),'balanced_accuracy':50.*(
            np.mean(prediction[y])+np.mean(~prediction[~y]))})
        scores.append(z); labels.append(y)
    return {'macro_task_auc':float(np.mean([r['auc'] for r in rows])),
            'macro_task_balanced_accuracy':float(np.mean([r['balanced_accuracy'] for r in rows])),
            'pooled_auc':binary_auc(np.concatenate(scores),np.concatenate(labels)),
            'chance_auc':.5,'chance_balanced_accuracy':50.,'tasks':len(rows),
            'note':'Pooled AUROC can reflect task composition; prioritize macro-task AUROC.'}


@torch.no_grad()
def concise_group_errors(records,score_fn):
    report=group_diagnostics(records,score_fn)
    report['stages']=[{k:v for k,v in row.items() if k!='branch_evidence'} for row in report['stages']]
    return report
