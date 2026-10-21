"""Provenance and real Runner checks with cached features, no CLIP download."""
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

from engine.engine import Runner
from engine import safd_metric_runtime as runtime
from test_guidance_integration import Manager, configuration
from test_residual_integration import make_model, synthetic_state, temporary_artifact_directory


CLIP_HASH = 'synthetic-frozen-clip-weight-sha256'


def config(output='', model_path='', enabled=True, gain=-1.):
    cfg = configuration(output, enabled=False, auto=False)
    cfg.defrost()
    cfg.TRAINER.BiMC.SAFD_METRIC.ENABLED = enabled
    cfg.TRAINER.BiMC.SAFD_METRIC.MODEL_PATH = str(model_path)
    cfg.TRAINER.BiMC.SAFD_METRIC.GAIN = gain
    cfg.freeze()
    return cfg


def base_ids():
    return [c*20+i for c in range(6) for i in range(16)]


def reference_manager(cfg, manager=None):
    """Real content fingerprints around a synthetic, deterministic image mapping."""
    manager = manager or SimpleNamespace(
        template=['a photo of a {}.'], class_names=[f'class_{i}' for i in range(10)])
    manager.test_transform = 'synthetic deterministic test transform'
    manager._support_metadata = lambda: {
        'dataset': cfg.DATASET.NAME, 'train_size': 200,
        'train_targets_sha256': hashlib.sha256(b'synthetic-label-order').hexdigest(),
        'train_samples_sha256': hashlib.sha256(b'synthetic-image-order').hexdigest()}
    return manager


def reference_metadata(cfg):
    manager = reference_manager(cfg)
    return {'ordinary_description_sha256': hashlib.sha256(Path(cfg.DATASET.GPT_PATH).read_bytes()).hexdigest(),
            'text_template': list(manager.template), 'support_transform': repr(manager.test_transform),
            'input_provenance': manager._support_metadata(),
            'base_class_names': manager.class_names[:cfg.DATASET.NUM_INIT_CLS],
            'precision': cfg.TRAINER.BiMC.PREC}


def save_model(path, cfg, matrix=None, gain=4., changes=None):
    path = Path(path)
    matrix = np.eye(8) if matrix is None else np.asarray(matrix)
    np.savez_compressed(path, metric=matrix)
    metadata = {
        'schema_version': 1, 'dataset': cfg.DATASET.NAME,
        'feature_dim': 8, 'gain': gain, 'clip_weight_sha256': CLIP_HASH,
        'base_class_ids': list(range(6)),
        'base_sample_ids_sorted_sha256': runtime.sample_identity_hash(base_ids()),
        'reference_settings': runtime.reference_settings(cfg),
        'model_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'data_source': 'base_training_only',
        'training_protocol': 'base_class_disjoint_validation_then_full_base_refit',
        'method': 'safd_response_metric', 'source': 'safd',
        'trainable_parameters': 7,
        **reference_metadata(cfg),
    }
    metadata.update(changes or {})
    path.with_suffix('.json').write_text(json.dumps(metadata), encoding='utf-8')
    return path


class SAFDMetricRuntimeTest(unittest.TestCase):
    def test_loader_checks_checksum_dataset_clip_classes_and_reference(self):
        with temporary_artifact_directory() as directory:
            root = Path(directory)
            cfg = config(enabled=False)
            path = root/'model.npz'
            for changes, message in (
                ({'data_source':'benchmark_test'}, 'base_training_only'),
                ({'data_source':None}, 'base_training_only'),
                ({'model_sha256':'wrong'}, 'checksum'),
                ({'dataset':'cifar100'}, 'dataset'),
                ({'clip_weight_sha256':'different'}, 'CLIP weights'),
                ({'base_class_ids':[0,1,2]}, 'base-class bank'),
                ({'reference_settings':{}}, 'reference settings'),
                ({'base_sample_ids_sorted_sha256':''}, 'base-image identity'),
                ({'precision':'fp16'}, 'precision'),
                ({'precision':None}, 'precision'),
                ({'ordinary_description_sha256':None}, 'ordinary_description'),
                ({'text_template':None}, 'text_template'),
                ({'support_transform':None}, 'support_transform'),
                ({'input_provenance':None}, 'input_provenance'),
                ({'base_class_names':None}, 'base_class_names'),
            ):
                save_model(path, cfg, changes=changes)
                with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                    runtime.load_frozen_metric(path, cfg, CLIP_HASH)
            save_model(path, cfg)
            with self.assertRaisesRegex(ValueError, 'CLIP weights'):
                runtime.load_frozen_metric(path, cfg, None)
            cfg.defrost();cfg.TRAINER.BiMC.PREC='fp16';cfg.freeze()
            with self.assertRaisesRegex(ValueError, 'fp32'):
                runtime.load_frozen_metric(path, cfg, CLIP_HASH)

    def test_reference_content_checks_actual_text_transform_mapping_and_class_order(self):
        with temporary_artifact_directory() as directory:
            root, cfg = Path(directory), config(enabled=False)
            ordinary = root/'ordinary.json'
            ordinary.write_text('{"class_0":["ordinary description"]}',encoding='utf-8')
            cfg.defrost();cfg.DATASET.GPT_PATH=str(ordinary);cfg.freeze()
            path = save_model(root/'model.npz',cfg)
            state = runtime.load_frozen_metric(path,cfg,CLIP_HASH)
            self.assertTrue(runtime.verify_reference_content(state,cfg,reference_manager(cfg)))
            self.assertTrue(runtime.metric_report(state)['reference_content_verified'])
            mutations = (
                ('text template',lambda m:setattr(m,'template',['another photo of {}'])),
                ('support transform',lambda m:setattr(m,'test_transform','changed image resize')),
                ('initial class names',lambda m:m.class_names.__setitem__(slice(0,2),list(reversed(m.class_names[:2])))),
            )
            for message,mutate in mutations:
                manager=reference_manager(cfg);mutate(manager)
                with self.subTest(message=message),self.assertRaisesRegex(ValueError,message):
                    runtime.verify_reference_content(state,cfg,manager)
                self.assertFalse(state['reference_content_verified'])
            for key,value in (('dataset','cifar100'),('train_size',201),
                              ('train_targets_sha256',hashlib.sha256(b'reassigned-labels').hexdigest()),
                              ('train_samples_sha256',hashlib.sha256(b'reordered-image-paths').hexdigest())):
                manager=reference_manager(cfg)
                current=manager._support_metadata();current[key]=value
                manager._support_metadata=lambda:current
                with self.subTest(key=key),self.assertRaisesRegex(ValueError,key):
                    runtime.verify_reference_content(state,cfg,manager)
                self.assertFalse(state['reference_content_verified'])
            runtime.verify_reference_content(state,cfg,reference_manager(cfg))
            ordinary.write_text('{"class_0":["changed ordinary description"]}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'ordinary description content'):
                runtime.verify_reference_content(state,cfg,reference_manager(cfg))
            self.assertFalse(state['reference_content_verified'])

    def test_base_bank_identity_and_dimension_are_verified_before_use(self):
        with temporary_artifact_directory() as directory:
            cfg = config(enabled=False)
            path = save_model(Path(directory)/'model.npz', cfg)
            state = runtime.load_frozen_metric(path, cfg, CLIP_HASH)
            with self.assertRaisesRegex(RuntimeError, 'Verify'):
                runtime.prepare_session_metric(state, torch.randn(6,8))
            with self.assertRaisesRegex(ValueError, 'feature dimension'):
                runtime.verify_base_bank(state, base_ids(), 7)
            with self.assertRaisesRegex(ValueError, 'different set'):
                runtime.verify_base_bank(state, base_ids()[:-1], 8)
            with self.assertRaisesRegex(ValueError, 'unique'):
                runtime.verify_base_bank(state, base_ids()+base_ids()[:1], 8)
            runtime.verify_base_bank(state, list(reversed(base_ids())), 8)
            self.assertTrue(state['base_bank_verified'])
            with self.assertRaisesRegex(RuntimeError, 'reference content'):
                runtime.prepare_session_metric(state, torch.randn(6,8))
            runtime.verify_reference_content(state, cfg, reference_manager(cfg))
            self.assertTrue(state['reference_content_verified'])
            runtime.prepare_session_metric(state, torch.randn(6,8))

    def test_loaded_identity_and_gain_zero_return_exact_reference(self):
        with temporary_artifact_directory() as directory:
            root, cfg = Path(directory), config(enabled=False)
            for matrix, gain in ((np.eye(8),4.), (np.diag([.3,1.8]+[1.]*6),0.)):
                path = save_model(root/'model.npz',cfg,matrix,gain=gain)
                state = runtime.load_frozen_metric(path,cfg,CLIP_HASH)
                runtime.verify_base_bank(state,base_ids(),8)
                runtime.verify_reference_content(state,cfg,reference_manager(cfg))
                prepared = runtime.prepare_session_metric(state,torch.randn(6,8))
                votes = torch.rand(4,6)+.1
                self.assertIs(runtime.apply_session_metric(state,prepared,torch.randn(4,8),votes),votes)

    def test_loader_rejects_non_spd_shapes_and_invalid_gains(self):
        with temporary_artifact_directory() as directory:
            root, cfg = Path(directory), config(enabled=False)
            invalid = [np.eye(7), np.diag([-1.]+[1.]*7), np.full((8,8),np.nan)]
            for matrix in invalid:
                path = save_model(root/'model.npz',cfg,matrix)
                with self.assertRaises(ValueError):
                    runtime.load_frozen_metric(path,cfg,CLIP_HASH)
            path = save_model(root/'model.npz',cfg,gain=-2.)
            with self.assertRaisesRegex(ValueError,'gain'):
                runtime.load_frozen_metric(path,cfg,CLIP_HASH)

    def run_example(self, directory, enabled=True, matrix=None, gain=4., wrong_bank=False, reverse_query=False):
        root = Path(directory)
        path = root/'model.npz'
        template = config(enabled=False)
        changes = {'base_sample_ids_sorted_sha256':'wrong-bank'} if wrong_bank else None
        save_model(path,template,matrix,gain,changes)
        cfg = config(root/'run',path,enabled=enabled)
        full = synthetic_state(num_classes=10,samples_per_class=20)
        model = make_model(cfg)
        model.clip_model.clip_weight_sha256 = CLIP_HASH
        manager = reference_manager(cfg,Manager(cfg,full,forbid_test=wrong_bank,reverse_query=reverse_query))
        def text(names,template,start):
            ids = torch.arange(start,start+len(names))
            return full['text_features'][ids],ids
        def descriptions(class_names,gpt_path,cls_begin_index):
            ids = torch.arange(cls_begin_index,cls_begin_index+len(class_names))
            mask = torch.isin(full['description_targets'],ids)
            return full['description_features'][mask],full['description_targets'][mask],full['description_proto'][ids],None
        snapshots, prepared_prototypes, scores = [], [], []
        with patch('engine.engine.DatasetManager',return_value=manager),patch('engine.engine.BiMC',return_value=model):
            runner = Runner(cfg)
        original_save = runner._save_checkpoint
        original_prepare, original_apply = runtime.prepare_session_metric, runtime.apply_session_metric
        def capture(task,states):
            snapshots.append({'metric':runner.safd_metric_state['metric'].clone() if enabled else None,
                              'gain':runner.safd_metric_state['gain'] if enabled else None,
                              'raw_means':torch.cat([s['raw_image_mean'] for s in states]).clone(),
                              'image_prototypes':torch.cat([s['image_proto'] for s in states]).clone()})
            for state in states:
                for key in ('images_features','images_targets','frequency_features'):
                    self.assertNotIn(key,state)
            original_save(task,states)
        def prepare(state,raw_means):
            prepared_prototypes.append(raw_means.clone())
            return original_prepare(state,raw_means)
        def apply(state,prepared,features,reference):
            output = original_apply(state,prepared,features,reference)
            scores.append((reference.clone(),output.clone()))
            return output
        with patch.object(model,'inference_text_feature',side_effect=text), \
             patch.object(model,'inference_all_description_feature',side_effect=descriptions), \
             patch.object(model,'extract_frequency_img_feature',side_effect=AssertionError('Query/support SAFD in frozen runtime')), \
             patch.object(runner,'_save_checkpoint',side_effect=capture), \
             patch.object(runtime,'prepare_session_metric',side_effect=prepare), \
             patch.object(runtime,'apply_session_metric',side_effect=apply), \
             contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            sessions = runner.run()
        return model,runner,sessions,snapshots,prepared_prototypes,scores

    def test_actual_runner_freezes_metric_and_corrects_all_seen_raw_prototypes(self):
        with temporary_artifact_directory() as directory:
            model,runner,sessions,snapshots,prepared,scores = self.run_example(
                directory,matrix=np.diag([.3,1.8,.7]+[1.]*5))
            self.assertEqual(len(sessions),3)
            self.assertEqual([p.shape[0] for p in prepared],[6,8,10])
            self.assertTrue(any(not torch.equal(a,b) for a,b in scores))
            for session in sessions:
                self.assertEqual(session['query_auxiliary_encodings'],0)
                self.assertEqual(session['query_original_encodings'],3*(6+2*session['session']))
                self.assertEqual(session['head_parameter_count'],7)
                self.assertEqual(session['base_fit_parameter_count'],7)
                self.assertEqual(session['incremental_trainable_parameter_count'],0)
            for i,snapshot in enumerate(snapshots):
                self.assertTrue(torch.equal(snapshot['metric'],snapshots[0]['metric']))
                self.assertEqual(snapshot['gain'],4.)
                self.assertTrue(torch.equal(prepared[i],snapshot['raw_means']))
            self.assertFalse(torch.allclose(F.normalize(snapshots[-1]['raw_means'][6:],dim=-1),
                                           snapshots[-1]['image_prototypes'][6:]))
            self.assertTrue(runner.safd_metric_state['base_bank_verified'])
            metrics=json.loads((Path(directory)/'run/metrics.json').read_text())
            self.assertEqual(metrics['status'],'completed')
            self.assertTrue(metrics['safd_metric']['base_bank_verified'])
            self.assertTrue(metrics['safd_metric']['reference_content_verified'])
            self.assertEqual(metrics['safd_metric']['query_frequency_encodings'],0)
            self.assertEqual(len(list((Path(directory)/'run').glob('predictions_session_*.npz'))),3)
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_actual_runner_zero_controls_are_exact_full_bimc(self):
        with temporary_artifact_directory() as directory:
            root=Path(directory)
            for name in ('baseline','identity','zero_gain'):
                (root/name).mkdir()
            _,_,baseline,_,_,_=self.run_example(root/'baseline',enabled=False)
            for name,matrix,gain in [('identity',np.eye(8),4.),
                                     ('zero_gain',np.diag([.3,1.8]+[1.]*6),0.)]:
                _,_,sessions,_,_,scores=self.run_example(root/name,matrix=matrix,gain=gain)
                self.assertTrue(all(torch.equal(a,b) for a,b in scores))
                self.assertTrue(all(s['reference_scores_exact_equal'] for s in sessions))
                self.assertEqual([s['accuracy'] for s in sessions],[s['accuracy'] for s in baseline])
                for i in range(3):
                    with np.load(root/'baseline/run'/f'predictions_session_{i:02d}.npz') as a, \
                         np.load(root/name/'run'/f'predictions_session_{i:02d}.npz') as b:
                        np.testing.assert_array_equal(a['sample_id'],b['sample_id'])
                        np.testing.assert_array_equal(a['prediction'],b['prediction'])

    def test_actual_runner_rejects_changed_base_images_before_test_queries(self):
        with temporary_artifact_directory() as directory:
            with self.assertRaisesRegex(ValueError,'different set of base training images'):
                self.run_example(directory,wrong_bank=True)


if __name__=='__main__':
    unittest.main()
