"""One bounded A-only study: matched-way transfer, evidence scale and count priors."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',required=True)
    p.add_argument('--anchor-model',required=True,help='Previously selected fixed anchor from the identical cache/split.')
    p.add_argument('--output-dir',required=True,help='Fresh diagnostic directory; no benchmark model is produced.')
    p.add_argument('--methods',nargs='+',choices=('group','joint'),default=['group','joint'])
    p.add_argument('--representations',nargs='+',choices=('raw','standardized','prior','prior_standardized'),
                   default=['raw','standardized','prior','prior_standardized'])
    p.add_argument('--seed',type=int,default=2)
    p.add_argument('--sequences',type=int,default=20)
    p.add_argument('--teacher-shot',type=int,default=15)
    p.add_argument('--query',type=int,default=5)
    p.add_argument('--old-shot',type=int,default=20)
    p.add_argument('--old-ways',type=int,nargs='+',default=[10,20,30])
    p.add_argument('--novel-ways',type=int,nargs='+',default=[5,10,20])
    p.add_argument('--novel-shots',type=int,nargs='+',default=[2,5])
    p.add_argument('--episode-pool',type=int,default=100)
    p.add_argument('--steps',type=int,default=300)
    p.add_argument('--validate-every',type=int,default=50)
    p.add_argument('--early-checkpoints',type=int,nargs='+',default=[1,5,10,20])
    p.add_argument('--blend-grid',type=float,nargs='+',default=[0.,.25,.5,1.])
    p.add_argument('--hidden',type=int,default=16)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--maximum-group-drop-pp',type=float,default=.2)
    p.add_argument('--standardization-floor',type=float,default=1e-4)
    p.add_argument('--standardization-clip',type=float,default=5.)
    p.add_argument('--probe-steps',type=int,default=100)
    p.add_argument('--probe-lr',type=float,default=.01)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--threads',type=int,default=1)
    a=p.parse_args(argv)
    import math
    if min(a.threads,a.probe_steps)<1 or any(not math.isfinite(v) or v<=0 for v in
            (a.probe_lr,a.standardization_floor,a.standardization_clip)):
        p.error('Use positive finite normalization/probe budgets and thread count.')
    if len(set(a.methods))!=len(a.methods) or len(set(a.representations))!=len(a.representations):
        p.error('Use distinct methods and representations.')
    return a


def write_json(path,data):
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def score_deltas(metrics,reference):
    return {key+'_delta_pp':metrics[key]-reference[key] for key in
            ('base_accuracy','novel_accuracy','harmonic_accuracy')}


def run_trial(train,matched,original,cfg,calibration,fixed,baselines,options,representation,args):
    import torch
    from engine.competitive_training import optimize
    from engine.competitive_evaluation import fusion_record_scores,feasible
    from engine.prototype_evaluation import evaluate_records
    from engine.competitive_diagnostics import (record_features,feature_shift,fit_group_probe,
        evaluate_group_probe,concise_group_errors)
    from models.competitive_evidence import EvidenceFusion
    torch.manual_seed(options.seed)
    model=EvidenceFusion(options.hidden,options.method,options.use_support_quality,options.group_limit,options.mix_limit,
        representation,args.standardization_floor,args.standardization_clip).to(args.device)
    fitting_features=torch.cat([record_features(model,row) for row in train])
    model.fit_statistics(fitting_features)
    best={'step':0,'blend':0.,**baselines['matched']['fixed']}
    best_weights={key:value.detach().clone() for key,value in model.state_dict().items()}
    candidates=[dict(best,feasible=True)]
    references=[baselines['matched']['fixed']]
    def validate(step,fitted,losses):
        nonlocal best,best_weights
        with torch.no_grad():
            for blend in sorted(set(v for v in options.blend_grid if v>0)):
                metrics=evaluate_records(matched,lambda row:fusion_record_scores(row,cfg,calibration,fixed,fitted,blend))
                ok=feasible(metrics,references,options.maximum_group_drop_pp)
                candidate={'step':step,'blend':blend,**metrics,'feasible':ok,'training_episode':losses}
                candidates.append(candidate)
                if ok and metrics['harmonic_accuracy']>best['harmonic_accuracy']+1e-12:
                    best=dict(candidate)
                    best_weights={key:value.detach().clone() for key,value in fitted.state_dict().items()}
        print(f"{options.method}/{representation} step={step}: matched best H={best['harmonic_accuracy']:.4f} "
              f"selected={best['step']}/{best['blend']}",flush=True)
    history=optimize(model,train,cfg,calibration,fixed,options,options.steps,validate)
    model.load_state_dict(best_weights)
    model.eval().requires_grad_(False)
    selected={}
    errors={}
    for name,bank in (('matched',matched),('original',original)):
        score=lambda row:fusion_record_scores(row,cfg,calibration,fixed,model,best['blend'])
        metrics=evaluate_records(bank,score)
        paired=evaluate_records(bank,score,reference_score_fn=lambda row:fusion_record_scores(row,cfg,calibration,fixed))
        selected[name]={'metrics':metrics,'delta_vs_fixed_pp':score_deltas(metrics,baselines[name]['fixed']),
            'paired_vs_fixed':paired,'passes_group_guard':feasible(metrics,
                [baselines[name]['fixed']],options.maximum_group_drop_pp),
            'passes_all_reference_guards':feasible(metrics,
                [baselines[name][key] for key in ('original','calibrated','fixed')],options.maximum_group_drop_pp)}
        errors[name]=concise_group_errors(bank,score)
    probe,probe_report=fit_group_probe(model,train,args.probe_steps,args.probe_lr)
    probe_report['evaluation']={name:evaluate_group_probe(model,probe,bank) for name,bank in
                                (('fitting',train),('matched',matched),('original',original))}
    shift={name:feature_shift(model,bank) for name,bank in (('fitting',train),('matched',matched),('original',original))}
    nonzero=[row for row in candidates if row['step']>0]
    report={'method':options.method,'representation':representation,'model_config':model.diagnostic_configuration(),
        'options':asdict(options),'selected_step':best['step'],'blend':best['blend'],'selected':selected,
        'best_nonzero_matched_candidate':max(nonzero,key=lambda row:row['harmonic_accuracy']) if nonzero else None,
        'checkpoints':candidates,'training_history':history,'group_probe':probe_report,'feature_shift':shift,
        'group_errors':errors,'normalizer':{'source':'A_fitting_queries_only',
            'mean':model.evidence_mean.cpu().tolist(),'scale':model.evidence_scale.cpu().tolist()},
        'trainable_parameters':sum(value.numel() for value in model.parameters()),
        'selection_source':'matched_way_validation_only','original_scale_used_for_selection':False,
        'selection_guard_anchor':'matched_fixed; all-reference guards reported separately',
        'all_base_refit_performed':False,'benchmark_eligible':False}
    return model,report


def save_markdown(path,summary):
    lines=['# 竞争证据与任务规模诊断','',
        '本轮只用基类缓存；不运行正式测试，不导出可部署的全基类重拟合模型。',
        '各 bank 的 H 难度不同，请比较相对该 bank 固定参考的 ΔH。checkpoint/blend 只由 matched bank 选择。','',
        '| 验证 bank | 任务数 | 总类数范围 | 固定 B | 固定 N | 固定 H |',
        '|---|---:|---|---:|---:|---:|']
    for name in ('matched','original'):
        coverage=summary['coverage'][name]; baseline=summary['baselines'][name]['fixed']
        lines.append(f"| {name} | {coverage['tasks']} | {coverage['total_class_range']} | "
            f"{baseline['base_accuracy']:.4f} | {baseline['novel_accuracy']:.4f} | {baseline['harmonic_accuracy']:.4f} |")
    lines+=['',
        '| 方法 | 输入 | 选择 step/blend | matched ΔH | original ΔH | original 保护 | matched 探针 AUROC | original 探针 AUROC |',
        '|---|---|---|---:|---:|---|---:|---:|']
    for row in summary['trials']:
        selected=row['selected']; probe=row['probe_evaluation']
        lines.append(f"| {row['method']} | {row['representation']} | {row['selected_step']}/{row['blend']} | "
            f"{selected['matched']['delta_vs_fixed_pp']['harmonic_accuracy_delta_pp']:+.4f} | "
            f"{selected['original']['delta_vs_fixed_pp']['harmonic_accuracy_delta_pp']:+.4f} | "
            f"{selected['original']['passes_group_guard']} | {probe['matched']['macro_task_auc']:.4f} | "
            f"{probe['original']['macro_task_auc']:.4f} |")
    lines+=['','AUROC 为旧新组二分类的逐任务平均；随机水平 .5。它不等于多类准确率或可部署收益。',
        'matched 有增益、original 无增益：优先研究规模迁移；标准化/去类数先验改变收益时，说明证据表征值得继续核查。',
        'matched 也无额外增益时，结合探针判断当前证据信息、损失目标与可用空间，停止无界调参。',
        '若 matched 固定参考已接近满分，需要考虑性能天花板，不能单凭零增益宣判证据无效。',
        '两组共享原 A/V 类别划分和支持/查询银行，matched 只缩减候选类别，保留原支持数和协方差到达分组策略。',
        '支持数量、协方差分组等其他训练/验证差异仍存在；本轮不声称所有分布都已匹配。']
    Path(path).write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main(argv=None):
    args=parse_args(argv)
    out=Path(args.output_dir)
    if out.exists():
        raise ValueError('Choose a fresh diagnostic output directory; existing results are not overwritten.')
    import numpy as np
    import torch
    import yaml
    from yacs.config import CfgNode
    from tools.cache_safd_base_features import load_cache
    from engine.competitive_training import FusionOptions,validate_options,training_episodes
    from engine.competitive_runtime import FORMAT,validate_metadata
    from engine.safd_metric_runtime import reference_settings,sample_identity_hash
    from engine.prototype_evaluation import prepare_context,validation_records
    from engine.competitive_evaluation import prepare_records,feasible
    from engine.competitive_diagnostics import matched_records,bank_baselines,bank_coverage
    options=FusionOptions(**{key:getattr(args,key) for key in FusionOptions.__dataclass_fields__ if hasattr(args,key)})
    validate_options(options)
    torch.set_num_threads(args.threads)
    started=time.perf_counter()
    arrays,metadata=load_cache(args.cache)
    if not metadata.get('complete_base_bank') or not metadata.get('has_text_features') or metadata.get('data_source')!='base_training_only':
        raise ValueError('Use the complete original base-only cache with ordinary text features.')
    cfg=CfgNode(yaml.safe_load(metadata['config_yaml']))
    cfg.TRAINER.BiMC.GUIDANCE.QUERY=options.query
    cfg.freeze()
    cfg,protocol,pool=prepare_context(arrays,cfg,options.seed,options.sequences,options.teacher_shot)
    anchor_path=Path(args.anchor_model)
    anchor=json.loads(anchor_path.with_suffix('.json').read_text(encoding='utf-8'))
    validate_metadata(anchor)
    if anchor.get('format')!=FORMAT or anchor['model_sha256']!=hashlib.sha256(anchor_path.read_bytes()).hexdigest():
        raise ValueError('Fixed anchor format/checksum mismatch.')
    expected={'cache_sha256':metadata['cache_sha256'],'fitting_seed':options.seed,
        'selection_protocol_sha256':protocol['sha256'],'reference_settings':reference_settings(cfg),
        'clip_weight_sha256':metadata['clip_weight_sha256'],
        'base_sample_ids_sorted_sha256':sample_identity_hash(arrays['sample_id'])}
    if any(anchor.get(key)!=value for key,value in expected.items()):
        raise ValueError('Anchor cache/seed/protocol/reference differs; keep the original screening parameters.')
    calibration,fixed=anchor['calibration'],anchor['fixed']
    out.mkdir(parents=True)
    for name in ('trials','models'):
        (out/name).mkdir()
    try:
        original=prepare_records(validation_records(arrays,cfg,protocol,args.device),cfg,calibration)
        train,trace=training_episodes(arrays,cfg,pool,options,calibration,args.device)
        way_pairs={(row['num_old'],len(row['class_ids'])-row['num_old']) for row in trace}
        matched,matched_protocol=matched_records(arrays,cfg,protocol,original,way_pairs,options.seed)
        prepare_records(matched,cfg,calibration)
        baselines={name:bank_baselines(bank,cfg,calibration,fixed) for name,bank in
                   (('matched',matched),('original',original))}
        if not feasible(baselines['original']['fixed'],[baselines['original'][key] for key in ('original','calibrated')],options.maximum_group_drop_pp):
            raise ValueError('Replayed anchor violates the requested original-scale B/N guard.')
        coverage={name:bank_coverage(bank) for name,bank in (('matched',matched),('original',original))}
        write_json(out/'protocol.json',protocol)
        write_json(out/'matched_protocol.json',matched_protocol)
        write_json(out/'fitting_episodes.json',trace)
        write_json(out/'banks.json',{'coverage':coverage,'baselines':baselines})
        source_names=('models/competitive_evidence.py','engine/competitive_diagnostics.py',
            'tools/diagnose_competitive_evidence.py','engine/competitive_training.py',
            'engine/competitive_evaluation.py','engine/prototype_evaluation.py','models/competitive_fusion.py',
            'engine/safd_metric_training.py','engine/residual_training.py','models/bimc.py')
        write_json(out/'provenance.json',{'arguments':vars(args),'cache_sha256':metadata['cache_sha256'],
            'anchor_model_sha256':anchor['model_sha256'],'runtime_torch':torch.__version__,
            'anchor_sidecar_sha256':hashlib.sha256(anchor_path.with_suffix('.json').read_bytes()).hexdigest(),
            'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_names}})
        summary={'status':'running','dataset':metadata['dataset'],'seed':args.seed,
            'benchmark_test_evaluation':False,'new_CLIP_or_SAFD_encodings':0,'planned_trials':len(args.methods)*len(args.representations),
            'selection_source':'matched_way_only; original_scale_observed_after_selection',
            'normalizer_source':'A_fitting_queries_only','all_base_refit_performed':False,
            'protocol_sha256':protocol['sha256'],'matched_protocol_sha256':matched_protocol['sha256'],
            'coverage':coverage,'baselines':baselines,'calibration':calibration,'fixed':fixed,'trials':[]}
        print('Banks:',coverage,flush=True)
        for method in args.methods:
            o=replace(options,method=method,use_support_quality=method=='joint')
            for representation in args.representations:
                model,report=run_trial(train,matched,original,cfg,calibration,fixed,baselines,o,representation,args)
                name=method+'_'+representation
                write_json(out/'trials'/f'{name}.json',report)
                path=out/'models'/f'{name}.npz'
                np.savez_compressed(path,**{key:value.detach().cpu().numpy() for key,value in model.state_dict().items()})
                write_json(path.with_suffix('.json'),{'format':'competitive_evidence_diagnostic_v1',
                    'benchmark_eligible':False,'fitting_scope':'A_only_no_all_base_refit',
                    'model_config':model.diagnostic_configuration(),'statistics_fitted':True,
                    'selected_step':report['selected_step'],'blend':report['blend'],
                    'calibration':calibration,'fixed':fixed,'cache_sha256':metadata['cache_sha256'],
                    'protocol_sha256':protocol['sha256'],'matched_protocol_sha256':matched_protocol['sha256'],
                    'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
                summary['trials'].append({**{key:report[key] for key in ('method','representation','selected_step','blend','selected')},
                    'probe_evaluation':report['group_probe']['evaluation'],'trial_report':str(Path('trials')/(name+'.json'))})
                write_json(out/'summary.json',summary)
                save_markdown(out/'summary.md',summary)
                print('Finished diagnostic:',name,flush=True)
        summary.update(status='completed',wall_seconds=time.perf_counter()-started,
            limitations=['No benchmark models or all-base refit produced.',
                'Original-scale bank has already been used in prior development; it is not a fresh test set.',
                'Matched and original banks share queries/classes; no independent-repeat significance claim.',
                'Normalization and count-prior corrections are input ablations, not proven improvements.'])
        write_json(out/'summary.json',summary)
    except Exception as exc:
        write_json(out/'failure.json',{'status':'failed','error':str(exc)})
        raise
    return 0


if __name__=='__main__':
    raise SystemExit(main())
