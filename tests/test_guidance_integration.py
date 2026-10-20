"""Real Runner and PyTorch scoring, using only a mocked image/text encoder."""
import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from main import setup_cfg
from engine.engine import Runner
from engine.guidance_calibration import sample_protocol, sequence_records, select_candidate, initialize_guidance
from test_residual_integration import make_model, synthetic_state, temporary_artifact_directory

ROOT = Path(__file__).resolve().parents[1]


def configuration(directory='', rho=.5, auto=True, enabled=True, mode='paired', source='explicit', base_only=False, dataset='cub200'):
    cfg = setup_cfg(str(ROOT/f'configs/datasets/{dataset}.yaml'),str(ROOT/'configs/trainers/bimc_frequency_guidance.yaml'))
    cfg.defrost()
    cfg.SEED = 11
    cfg.OUTPUT_DIR = str(directory)
    cfg.DEVICE.DEVICE_NAME = 'cpu'
    cfg.DATASET.NUM_CLASSES, cfg.DATASET.NUM_INIT_CLS, cfg.DATASET.NUM_INC_CLS = 10,6,2
    cfg.DATASET.NUM_INC_SHOT = 2
    g = cfg.TRAINER.BiMC.GUIDANCE
    g.ENABLED, g.MODE, g.SEMANTIC_SOURCE = enabled,mode,source
    g.RHO, g.AUTO_CALIBRATE = rho,auto
    g.VAL_SEQUENCES = 2
    g.CANDIDATES_PER_BAND = 4
    g.SHOT, g.QUERY, g.AUDIT_IMAGES = 2,2,2
    g.BASE_ONLY = base_only
    cfg.freeze()
    return cfg


class Manager:
    def __init__(self,cfg,state,forbid_test=False,reverse_query=False):
        self.cfg,self.state,self.forbid_test,self.reverse_query=cfg,state,forbid_test,reverse_query
        self.support_seed=cfg.SEED
        self.template=['a photo of a {}.']
        self.class_names=[f'class_{i}' for i in range(10)]
        self.class_index_in_task=[np.arange(6),np.arange(6,8),np.arange(8,10)]
        self.num_tasks=3

    def save_support_manifest(self,path):
        Path(path).write_text(json.dumps({'seed':self.support_seed,'synthetic':True}))

    def get_dataloader(self,task_id,source,mode,accumulate_past=False):
        if accumulate_past:
            raise AssertionError('Historical image replay')
        if source=='test' and self.forbid_test:
            raise AssertionError('Base-only mode read benchmark queries')
        classes=self.class_index_in_task[task_id] if source=='train' else np.concatenate(self.class_index_in_task[:task_id+1])
        rows=[]
        for c in classes:
            indices=torch.where(self.state['images_targets']==int(c))[0]
            rows.extend((indices[:16 if task_id==0 else 2] if source=='train' else indices[-3:]).tolist())
        if source=='test' and self.reverse_query:
            rows=rows[::-1]
        # Multiple batches exercise stable sample IDs and exactly one query encoding.
        return [{'image':self.state['images_features'][rows[i:i+7]],
                 'label':self.state['images_targets'][rows[i:i+7]],
                 'global_index':torch.tensor(rows[i:i+7])} for i in range(0,len(rows),7)]


class GuidanceIntegrationTest(unittest.TestCase):
    def example(self,directory='',**kwargs):
        reverse=kwargs.pop('reverse_query',False)
        cfg=configuration(directory,**kwargs)
        full=synthetic_state(num_classes=10,samples_per_class=20)
        if kwargs.get('source') == 'ordinary':
            ordinary=full['frequency_description_candidates'].reshape(10,12,-1)
            full['description_features']=ordinary.reshape(120,-1)
            full['description_targets']=torch.arange(10).repeat_interleave(12)
            full['description_proto']=F.normalize(ordinary.mean(1),dim=-1)
        model=make_model(cfg)
        manager=Manager(cfg,full,cfg.TRAINER.BiMC.GUIDANCE.BASE_ONLY,reverse)
        def text(names,template,start):
            ids=torch.arange(start,start+len(names))
            return full['text_features'][ids],ids
        def descriptions(class_names,gpt_path,cls_begin_index):
            ids=torch.arange(cls_begin_index,cls_begin_index+len(class_names))
            mask=torch.isin(full['description_targets'],ids)
            return full['description_features'][mask],full['description_targets'][mask],full['description_proto'][ids],None
        def frequency(images,original_features=None):
            model.image_encoding_counts['auxiliary']+=3*len(images)
            return F.normalize(torch.stack([images+.4*images.roll(i,-1) for i in (1,2,3)],1),dim=-1)
        def candidates(names,path,max_candidates=None):
            start=int(str(names[0]).split('_')[-1])
            result=full['frequency_description_candidates'][start:start+len(names)]
            return result[:,:,:max_candidates] if max_candidates is not None else result
        snapshots,base=[],[]
        original=model.build_task_statistics
        def build(*args,**kw):
            state=original(*args,**kw)
            if int(state['class_index'][0])==0:
                base.append(dict(state))
            return state
        with patch('engine.engine.DatasetManager',return_value=manager),patch('engine.engine.BiMC',return_value=model):
            runner=Runner(cfg)
        save=runner._save_checkpoint
        def capture(task_id,states):
            for state in states:
                for key in ('images_features','images_targets','frequency_features','guidance_sample_ids'):
                    self.assertNotIn(key,state)
            snapshots.append({'bank':{k:v.clone() if torch.is_tensor(v) else v for k,v in (model.guidance_state or {}).items()},
                              'prototypes':[s['image_proto'].clone() for s in states]})
            save(task_id,states)
        with patch.object(model,'inference_text_feature',side_effect=text), \
             patch.object(model,'inference_all_description_feature',side_effect=descriptions), \
             patch.object(model,'inference_explicit_frequency_description_candidates',side_effect=candidates), \
             patch.object(model,'extract_frequency_img_feature',side_effect=frequency), \
             patch.object(model,'build_task_statistics',side_effect=build), \
             patch.object(runner,'_save_checkpoint',side_effect=capture), \
             contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            sessions=runner.run()
        base[0].update(data_source='base_training_only',task_id=0)
        return model,runner,sessions,snapshots,base[0]

    def test_actual_runner_calibrates_freezes_bank_and_skips_query_views(self):
        with temporary_artifact_directory() as directory:
            model,runner,sessions,snapshots,base=self.example(directory)
            self.assertEqual(len(sessions),3)
            report=runner.guidance_report
            selected=next(c for c in report['candidate_results'] if c['rho']==report['selected_rho'])
            self.assertTrue(selected['feasible'])
            for session in sessions:
                self.assertEqual(session['query_auxiliary_encodings'],0)
                self.assertEqual(session['query_original_encodings'],3*(6+2*session['session']))
            for snap in snapshots[1:]:
                for key,value in snapshots[0]['bank'].items():
                    if torch.is_tensor(value):
                        self.assertTrue(torch.equal(value,snap['bank'][key]),key)
                self.assertTrue(torch.equal(snap['prototypes'][0],snapshots[0]['prototypes'][0]))
            self.assertTrue(torch.equal(snapshots[1]['prototypes'][1],snapshots[2]['prototypes'][1]))
            checkpoint=torch.load(Path(directory)/'checkpoint.pt',weights_only=False)
            self.assertIsNotNone(checkpoint['guidance_state'])
            self.assertIsNone(checkpoint['residual_state'])
            self.assertTrue((Path(directory)/'guidance_calibration.json').is_file())
            self.assertFalse(any(p.requires_grad for p in model.parameters()))
            with self.assertRaises(RuntimeError):
                initialize_guidance(model,model.cfg,base)

    def test_zero_strength_equals_original_bimc_predictions_and_prototypes(self):
        _,_,a,sa,_=self.example(rho=0.,auto=False)
        _,_,b,sb,_=self.example(enabled=False,auto=False)
        for x,y in zip(a,b):
            self.assertEqual(x['accuracy'],y['accuracy'])
        for x,y in zip(sa[-1]['prototypes'],sb[-1]['prototypes']):
            self.assertTrue(torch.equal(x,y))

    def test_query_order_does_not_change_selection_or_guided_bank(self):
        a,ra,_,sa,_=self.example()
        b,rb,_,sb,_=self.example(reverse_query=True)
        self.assertEqual(ra.guidance_report['selected_rho'],rb.guidance_report['selected_rho'])
        self.assertEqual(ra.guidance_report['protocol_sha256'],rb.guidance_report['protocol_sha256'])
        for x,y in zip(sa[-1]['prototypes'],sb[-1]['prototypes']):
            self.assertTrue(torch.equal(x,y))

    def test_base_only_does_not_load_test_and_uniform_and_visual_need_no_text(self):
        for mode,source in [('paired','uniform'),('visual','explicit')]:
            model,runner,sessions,_,_=self.example(base_only=True,mode=mode,source=source)
            self.assertEqual(sessions,[])
            self.assertIsNone(model.guidance_state['base_text'])
            self.assertEqual(runner.guidance_report['query_auxiliary_encodings'],0)

    def test_globally_reserved_audit_and_support_query_disjointness(self):
        cfg=configuration()
        labels=torch.arange(6).repeat_interleave(20)
        protocol=sample_protocol(cfg,labels,list(range(len(labels))))
        self.assertEqual(protocol,sample_protocol(cfg,labels,list(range(len(labels)))))
        self.assertFalse(set(protocol['pseudo_base_class_ids'])&set(protocol['pseudo_validation_class_ids']))
        audit={i for rows in protocol['audit_indices'].values() for i in rows}
        for seq in protocol['sequences']:
            s={i for rows in seq['support_indices'] for i in rows}
            q={i for rows in seq['query_indices'] for i in rows}
            self.assertFalse(s&q or audit&s or audit&q)
        with self.assertRaisesRegex(ValueError,'needs'):
            sample_protocol(cfg,torch.arange(6).repeat_interleave(5),list(range(30)))

    def test_query_features_do_not_leak_into_relations_or_covariance(self):
        model,runner,_,_,state=self.example(base_only=True)
        seq=runner.guidance_report['protocol']['sequences'][0]
        first=list(sequence_records(model,model.cfg,state,seq))
        query=[i for rows in seq['query_indices'] for i in rows]
        changed={**state,'images_features':state['images_features'].clone(),
                 'frequency_features':state['frequency_features'].clone()}
        changed['images_features'][query]*=-1
        changed['frequency_features'][query]*=-1
        second=list(sequence_records(model,model.cfg,changed,seq))
        for x,y in zip(first,second):
            for key in ('visual','bands','covariance'):
                self.assertTrue(torch.equal(x[key],y[key]),key)
            for key in x['relations']:
                if x['relations'][key] is not None:
                    self.assertTrue(torch.equal(x['relations'][key],y['relations'][key]))
            self.assertEqual(x['relations']['frequency'].shape[0],x['new_count'])

    def test_candidate_guards_and_tie_break_choose_zero(self):
        candidates=[{'rho':r,'mean_base_accuracy':a,'mean_novel_accuracy':n,'mean_harmonic_accuracy':h}
                    for r,a,n,h in [(0.,80.,60.,68.),(.25,80.,60.,68.),(.5,79.,65.,71.)]]
        self.assertEqual(select_candidate(candidates,.2)['rho'],0.)
        self.assertFalse(candidates[-1]['feasible'])

    def test_ordinary_control_uses_disjoint_equal_budget_groups(self):
        model,_,sessions,_,base=self.example(source='ordinary')
        self.assertEqual(len(sessions),3)
        expected=F.normalize(base['description_features'].reshape(6,3,4,-1).mean(2),dim=-1)
        torch.testing.assert_close(base['guidance_text_proto'],expected)

    def test_actual_nominal_dataset_pseudo_schedules(self):
        from engine.guidance_calibration import stage_sizes
        for name,expected in [('cub200',[5]*10),('cifar100',[2,3]*4),('miniimagenet',[2,3]*4)]:
            cfg=setup_cfg(str(ROOT/f'configs/datasets/{name}.yaml'),str(ROOT/'configs/trainers/bimc_frequency_guidance.yaml'))
            self.assertEqual(stage_sizes(cfg),expected)

    def test_three_modes_share_calibration_protocol_on_all_datasets(self):
        for dataset in ('cub200','cifar100','miniimagenet'):
            hashes=[]
            for mode in ('visual','marginal','paired'):
                _,runner,sessions,_,_=self.example(dataset=dataset,mode=mode)
                hashes.append(runner.guidance_report['protocol_sha256'])
                self.assertTrue(all(s['query_auxiliary_encodings']==0 for s in sessions))
            self.assertEqual(len(set(hashes)),1)

    def test_calibrator_freezes_historical_pseudo_novel_prototypes(self):
        model,_,_,_,base=self.example(base_only=True)
        model.guidance_state=None
        scorer=model.reference_scores_from_features
        captured=[]
        def capture(*args,**kwargs):
            captured.append(args[3].clone())
            return scorer(*args,**kwargs)
        with patch.object(model,'reference_scores_from_features',side_effect=capture),contextlib.redirect_stdout(io.StringIO()):
            initialize_guidance(model,model.cfg,base)
        # Two sequences, two stages and the five shared rho candidates.
        self.assertEqual(len(captured),20)
        for offset in (0,10):
            for candidate in range(5):
                first=captured[offset+candidate]
                second=captured[offset+5+candidate]
                self.assertTrue(torch.equal(first,second[:len(first)]))

    def test_future_pseudo_class_features_cannot_affect_the_current_stage(self):
        model,runner,_,_,state=self.example(base_only=True)
        seq=runner.guidance_report['protocol']['sequences'][0]
        before=list(sequence_records(model,model.cfg,state,seq))[0]
        changed={**state,'images_features':state['images_features'].clone(),
                 'frequency_features':state['frequency_features'].clone()}
        future=seq['support_indices'][-1]
        changed['images_features'][future]*=-1
        changed['frequency_features'][future]*=-1
        after=list(sequence_records(model,model.cfg,changed,seq))[0]
        for key in ('visual','bands','covariance'):
            self.assertTrue(torch.equal(before[key],after[key]))
        for key,value in before['relations'].items():
            if value is not None:
                self.assertTrue(torch.equal(value,after['relations'][key]))


if __name__=='__main__':
    unittest.main()
