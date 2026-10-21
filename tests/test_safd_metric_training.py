import contextlib
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from main import setup_cfg
from engine import safd_metric_training as training
from models.safd_metric import SAFDMetricAdapter, prepare_residual_classifier, apply_metric_residual
from engine.safd_metric_runtime import load_frozen_metric, verify_base_bank, verify_reference_content
from test_residual_integration import temporary_artifact_directory
from tools import train_safd_metric as training_cli


def fixture():
    root = Path(__file__).resolve().parents[1]
    cfg = setup_cfg(str(root/'configs/datasets/cub200.yaml'),
                    str(root/'configs/trainers/bimc_frequency_guidance.yaml'))
    cfg.defrost()
    cfg.DATASET.NUM_INIT_CLS = 6
    cfg.DATASET.NUM_CLASSES = 10
    cfg.DATASET.NUM_INC_CLS = 2
    cfg.DATASET.NUM_INC_SHOT = 2
    cfg.TRAINER.BiMC.GUIDANCE.STAGE_SIZES = [1, 1]
    cfg.TRAINER.BiMC.GUIDANCE.SHOT = 2
    cfg.TRAINER.BiMC.GUIDANCE.QUERY = 2
    cfg.TRAINER.BiMC.GUIDANCE.AUDIT_IMAGES = 2
    cfg.freeze()
    rng = torch.Generator().manual_seed(37)
    labels = torch.arange(6).repeat_interleave(20)
    original = F.normalize(torch.eye(8)[labels] + .15*torch.randn(120,8,generator=rng),dim=-1)
    coarse = F.normalize(original + .2*torch.randn(120,8,generator=rng),dim=-1)
    middle = F.normalize(original + .07*torch.randn(120,8,generator=rng),dim=-1)
    text = torch.eye(8)[:6]
    arrays = {'features':torch.stack((original,coarse,middle),1).numpy(),
              'labels':labels.numpy(), 'sample_id':np.arange(120),
              'text_features':text.numpy(), 'description_proto':text.numpy(),
              'description_features':text.repeat_interleave(2,0).numpy(),
              'description_targets':np.repeat(np.arange(6),2)}
    options = training.TrainingOptions(steps=0,rank=3,sequences=2,episode_pool=2,
               old_way=1,novel_way=1,old_shot=4,query=2,validate_every=1)
    return arrays,cfg,options


class SAFDMetricTrainingTest(unittest.TestCase):
    def test_global_base_queries_and_all_validation_classes_excluded_from_fitting(self):
        arrays,cfg,options=fixture()
        labels=torch.as_tensor(arrays['labels'])
        cfg,p,pool=training.reserve_selection_split(cfg,labels,arrays['sample_id'],options)
        fit=set(p['fitting_indices'])
        reserved={i for rows in p['reserved_pseudo_base_query_indices'].values() for i in rows}
        self.assertFalse(fit&reserved)
        self.assertTrue(set(labels[list(fit)].tolist()) <= set(p['pseudo_base_class_ids']))
        self.assertFalse(set(labels[list(fit)].tolist())&set(p['pseudo_validation_class_ids']))
        for seq in p['sequences']:
            for j,cls in enumerate(p['pseudo_base_class_ids']):
                self.assertEqual(seq['query_indices'][j],p['reserved_pseudo_base_query_indices'][str(cls)])
                self.assertEqual(set(seq['support_indices'][j]),set(pool[cls]))
        x=torch.as_tensor(arrays['features'][:,0])
        _,trace=training.training_records(cfg,x,arrays,pool,options)
        for episode in trace:
            support={i for rows in episode['support_indices'] for i in rows}
            query={i for rows in episode['query_indices'] for i in rows}
            self.assertFalse(support&query)
            self.assertTrue((support|query)<=fit)

    def test_step_zero_exports_identity_and_all_base_refit_occurs_after_selection(self):
        arrays,cfg,options=fixture()
        events=[]
        original_fit,original_eval=training._fit_basis,training.evaluate
        def fit(*args,**kwargs):
            events.append(('fit',list(args[3])))
            return original_fit(*args,**kwargs)
        def evaluate(*args,**kwargs):
            events.append(('evaluate',None))
            return original_eval(*args,**kwargs)
        with patch.object(training,'_fit_basis',side_effect=fit),patch.object(training,'evaluate',side_effect=evaluate),contextlib.redirect_stdout(io.StringIO()):
            adapter,report,cfg=training.fit_metric(arrays,cfg,options)
        self.assertEqual(report['selected_step'],0)
        self.assertEqual(report['gain'],0)
        self.assertTrue(torch.equal(adapter.to_metric(),torch.eye(8)))
        fits=[i for i,e in enumerate(events) if e[0]=='fit']
        self.assertEqual(len(fits),2)
        self.assertTrue(any(e[0]=='evaluate' for e in events[fits[0]+1:fits[1]]))
        self.assertEqual(events[fits[1]][1],list(range(120)))
        self.assertEqual(set(events[fits[0]][1]),set(report['selection_protocol']['fitting_indices']))

    def test_training_updates_theta_only(self):
        adapter=SAFDMetricAdapter(torch.eye(4)[:,:2])
        q=torch.tensor([[.8,.6,0,0]],requires_grad=True)
        mu=torch.eye(4)[:2].requires_grad_()
        row={'query':q,'prototypes':mu,'log_reference':torch.zeros(1,2),
             'targets':torch.tensor([1]),'num_old':1}
        options=training.TrainingOptions(steps=3,lr=.01)
        training.train_steps(adapter,[row],options,3)
        self.assertGreater(int(torch.count_nonzero(adapter.theta)),0)
        self.assertIsNone(q.grad)
        self.assertIsNone(mu.grad)
        self.assertEqual([name for name,_ in adapter.named_parameters()],['theta'])

    def test_training_loss_weights_old_and_novel_groups_equally(self):
        adapter=SAFDMetricAdapter(torch.eye(4)[:,:2])
        row={'query':torch.eye(4)[:3], 'prototypes':torch.eye(4)[:2],
             'log_reference':torch.tensor([[3.,0.],[3.,0.],[3.,0.]]),
             'targets':torch.tensor([0,0,1]),'num_old':1}
        expected=.5*(F.cross_entropy(row['log_reference'][:2],row['targets'][:2])
                     +F.cross_entropy(row['log_reference'][2:],row['targets'][2:]))
        actual=training.train_steps(adapter,[row],training.TrainingOptions(),1)
        self.assertAlmostEqual(actual[0],float(expected),places=6)
        self.assertNotAlmostEqual(actual[0],float(F.cross_entropy(row['log_reference'],row['targets'])),places=3)
        for targets in (torch.zeros(3,dtype=torch.long),torch.ones(3,dtype=torch.long)):
            group_row={**row,'targets':targets}
            adapter=SAFDMetricAdapter(torch.eye(4)[:,:2])
            actual=training.train_steps(adapter,[group_row],training.TrainingOptions(),1)
            with self.subTest(targets=targets.tolist()):
                self.assertAlmostEqual(actual[0],float(F.cross_entropy(row['log_reference'],targets)),places=6)

    def test_full_nonzero_training_visits_checkpoints_and_changes_selection_theta(self):
        arrays,cfg,options=fixture()
        options.steps=2
        seen=[]
        original=training.train_steps
        def train(adapter,*args,**kwargs):
            result=original(adapter,*args,**kwargs)
            seen.append(adapter.theta.detach().clone())
            return result
        with patch.object(training,'train_steps',side_effect=train),contextlib.redirect_stdout(io.StringIO()):
            _,report,_=training.fit_metric(arrays,cfg,options)
        self.assertTrue(seen)
        self.assertGreater(int(torch.count_nonzero(seen[0])),0)
        self.assertEqual([r['step'] for r in report['checkpoints']],[0,1,2])
        self.assertEqual(len(report['selection_training_episodes']),2)
        self.assertGreaterEqual(report['fit_wall_seconds'],report['selection_wall_seconds'])

    def test_closed_form_control_never_optimizes_and_refits_after_selection(self):
        arrays,cfg,options=fixture()
        options.method='closed_form'
        options.strength_grid=(0.,.1,1.)
        with patch.object(training,'train_steps') as optimizer,contextlib.redirect_stdout(io.StringIO()):
            adapter,report,_=training.fit_metric(arrays,cfg,options)
        optimizer.assert_not_called()
        self.assertIsInstance(adapter,training.FrozenMetric)
        self.assertEqual(report['trainable_parameters'],0)
        self.assertEqual([r['strength'] for r in report['checkpoints']],[0.,.1,1.])
        self.assertEqual(report['final_refit_indices'],list(range(120)))
        self.assertGreater(report['phase_order'].index('refit_all_base_scatter'),report['phase_order'].index('selection_ended'))

    def test_cli_refuses_bad_output_before_loading_cache_or_training(self):
        with temporary_artifact_directory() as directory:
            path=Path(directory)/'metric.npz'
            path.write_bytes(b'existing')
            # The nonexistent cache must never be read on either failure.
            with self.assertRaisesRegex(ValueError,'overwrite'):
                training_cli.main(['--cache','nonexistent.npz','--output',str(path)])
            with self.assertRaisesRegex(ValueError,'end in .npz'):
                training_cli.main(['--cache','nonexistent.npz','--output',str(path.with_suffix('.txt'))])

    def test_validation_uses_export_metric_and_runtime_vote_scoring(self):
        adapter=SAFDMetricAdapter(torch.eye(4)[:,:2])
        with torch.no_grad():adapter.theta.copy_(torch.tensor([.12,-.23]))
        row={'query':torch.tensor([[.8,.6,0,0],[.3,.9,0,0]]),
             'prototypes':torch.eye(4)[:2], 'reference_votes':torch.tensor([[.55,.45],[.4,.6]]),
             'targets':torch.tensor([0,1]),'num_old':1}
        prepared=prepare_residual_classifier(row['prototypes'],adapter.to_metric())
        actual=apply_metric_residual(row['reference_votes'],row['query'],prepared,16)
        with patch.object(training,'apply_metric_residual',wraps=training.apply_metric_residual) as scorer:
            result=training.evaluate(adapter,[row],16)
        scorer.assert_called_once()
        self.assertAlmostEqual(result['base_accuracy'],100*float(actual.argmax(-1)[0]==0))
        self.assertAlmostEqual(result['novel_accuracy'],100*float(actual.argmax(-1)[1]==1))

    def test_export_is_accepted_by_frozen_runtime_and_refuses_overwrite(self):
        arrays,cfg,options=fixture()
        with contextlib.redirect_stdout(io.StringIO()):adapter,report,cfg=training.fit_metric(arrays,cfg,options)
        metadata={'dataset':'cub200','source':'safd','clip_weight_sha256':'test-clip',
                  'base_class_ids':list(range(6)),'complete_base_bank':True}
        manager=SimpleNamespace(template=['a photo of a {}.'],test_transform='fixed deterministic transform',
                                class_names=[f'class_{i}' for i in range(10)])
        provenance={'dataset':'cub200','train_size':len(arrays['labels']),
                    'train_targets_sha256':hashlib.sha256(arrays['labels'].tobytes()).hexdigest(),
                    'train_samples_sha256':hashlib.sha256(arrays['sample_id'].tobytes()).hexdigest()}
        manager._support_metadata=lambda:dict(provenance)
        metadata.update(ordinary_description_sha256=hashlib.sha256(Path(cfg.DATASET.GPT_PATH).read_bytes()).hexdigest(),
                        text_template=manager.template,support_transform=repr(manager.test_transform),
                        input_provenance=provenance,base_class_names=manager.class_names[:6])
        with temporary_artifact_directory() as directory:
            path=Path(directory)/'metric.npz'
            cache=Path(directory)/'cache.npz';cache.write_bytes(b'cache-fixture')
            saved=training.export_metric(path,adapter,report,cfg,metadata,cache,arrays['sample_id'])
            state=load_frozen_metric(path,cfg,'test-clip')
            verify_base_bank(state,arrays['sample_id'],8)
            verify_reference_content(state,cfg,manager)
            self.assertTrue(state['base_bank_verified'])
            self.assertTrue(state['reference_content_verified'])
            self.assertEqual(saved['gain'],0)
            self.assertEqual(saved['data_source'],'base_training_only')
            self.assertEqual(saved['precision'],'fp32')
            for key in ('ordinary_description_sha256','text_template','support_transform','input_provenance','base_class_names'):
                self.assertEqual(saved[key],metadata[key])
            self.assertEqual(json.loads(path.with_suffix('.training.json').read_text())['selected_step'],0)
            with self.assertRaisesRegex(ValueError,'overwrite'):
                training.export_metric(path,adapter,report,cfg,metadata,cache,arrays['sample_id'])
            for key in ('ordinary_description_sha256','text_template','support_transform','input_provenance','base_class_names'):
                incomplete=dict(metadata);incomplete.pop(key)
                missing_path=Path(directory)/f'missing_{key}.npz'
                with self.subTest(key=key),self.assertRaisesRegex(ValueError,key):
                    training.export_metric(missing_path,adapter,report,cfg,incomplete,cache,arrays['sample_id'])
                self.assertFalse(missing_path.exists())

    def test_unsafe_options_rejected(self):
        for name,value in [('steps',-1),('rank',0),('episode_pool',0),('lr',float('nan'))]:
            options=training.TrainingOptions();setattr(options,name,value)
            with self.assertRaises(ValueError):training.validate_options(options)


if __name__=='__main__':unittest.main()
