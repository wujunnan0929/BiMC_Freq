"""Safe artifacts and actual Runner: zero parity, frozen model and no sample replay."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from engine.engine import Runner
from engine import competitive_runtime as runtime
from engine.competitive_evaluation import identity_fixed
from engine.prototype_evaluation import identity_calibration
from engine.safd_metric_runtime import reference_settings, sample_identity_hash
from models.competitive_fusion import CompetitiveFusion
from test_guidance_integration import Manager
from test_residual_integration import make_model, synthetic_state, temporary_artifact_directory
from test_safd_metric_runtime import CLIP_HASH, base_ids, reference_manager, reference_metadata, config as base_config


def config(output='', model_path='', enabled=True, mode='learned', blend=-1.):
    cfg = base_config(output, enabled=False)
    cfg.defrost()
    p = cfg.TRAINER.BiMC.COMPETITIVE_FUSION
    p.ENABLED, p.MODEL_PATH, p.MODE, p.BLEND = enabled, str(model_path), mode, blend
    cfg.freeze()
    return cfg


def artifact(path, cfg, wrong_bank=False):
    model = CompetitiveFusion(hidden=8)
    with torch.no_grad():
        model.network[-1].bias.copy_(torch.tensor([.3, .4, -.2]))
    metadata = {'dataset': cfg.DATASET.NAME, 'data_source': 'base_training_only', 'feature_dim': 8,
        'base_class_ids': list(range(6)), 'clip_weight_sha256': CLIP_HASH,
        'base_sample_ids_sorted_sha256': sample_identity_hash([-1] if wrong_bank else base_ids()),
        'reference_settings': reference_settings(cfg), **reference_metadata(cfg),
        'model_config': model.configuration(), 'trainable_parameters': sum(p.numel() for p in model.parameters()),
        'selected_step': 2, 'fitting_seed': 2, 'selection_protocol_sha256': 'same-development-protocol',
        'blend': 1., 'calibration': identity_calibration(), 'fixed': identity_fixed(cfg)}
    runtime.export_fusion(path, model, metadata, {})
    return model


class CompetitiveRuntimeTest(unittest.TestCase):
    def run_example(self, directory, enabled=True, mode='learned', wrong_bank=False):
        root = Path(directory)
        path = root/'fusion.npz'
        cfg = config(root/'run', path, enabled, mode)
        source = artifact(path, cfg, wrong_bank)
        full = synthetic_state(num_classes=10, samples_per_class=20)
        model = make_model(cfg)
        model.clip_model.clip_weight_sha256 = CLIP_HASH
        manager = reference_manager(cfg, Manager(cfg, full, forbid_test=wrong_bank))
        def text(names, templates, start):
            ids = torch.arange(start, start+len(names))
            return full['text_features'][ids], ids
        def descriptions(class_names, gpt_path, cls_begin_index):
            ids = torch.arange(cls_begin_index, cls_begin_index+len(class_names))
            mask = torch.isin(full['description_targets'], ids)
            return full['description_features'][mask], full['description_targets'][mask], full['description_proto'][ids], None
        with patch('engine.engine.DatasetManager', return_value=manager), patch('engine.engine.BiMC', return_value=model):
            runner = Runner(cfg)
        snapshots, scores = [], []
        original_save, original_apply = runner._save_checkpoint, runtime.apply_fusion
        def capture(task, states):
            snapshots.append({'prototypes': torch.cat([s['image_proto'] for s in states]).clone(),
                'quality': torch.cat([s['fusion_quality'] for s in states]).clone() if enabled else None,
                'weights': {k:v.clone() for k,v in runner.competitive_fusion_state['model'].state_dict().items()} if enabled else {}})
            self.assertTrue(all('images_features' not in s and 'images_targets' not in s for s in states))
            original_save(task, states)
        def apply(state, cfg, merged, features, reference):
            out = original_apply(state, cfg, merged, features, reference)
            scores.append((reference.clone(), out.clone()))
            return out
        with patch.object(model, 'inference_text_feature', side_effect=text), \
             patch.object(model, 'inference_all_description_feature', side_effect=descriptions), \
             patch.object(model, 'extract_frequency_img_feature', side_effect=AssertionError('Unexpected auxiliary encoding')), \
             patch.object(runner, '_save_checkpoint', side_effect=capture), \
             patch.object(runtime, 'apply_fusion', side_effect=apply), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            sessions = runner.run()
        return runner, sessions, snapshots, scores, source

    def test_actual_runner_preserves_prototypes_and_freezes_weights(self):
        with temporary_artifact_directory() as directory:
            root = Path(directory)
            runner, sessions, snapshots, scores, source = self.run_example(root/'learned')
            _, baseline, original, _, _ = self.run_example(root/'baseline', enabled=False)
            self.assertEqual(len(sessions), 3)
            self.assertTrue(any(not torch.equal(a,b) for a,b in scores))
            for j, row in enumerate(snapshots):
                self.assertTrue(torch.equal(row['prototypes'], original[j]['prototypes']))
                if j:
                    self.assertTrue(torch.equal(row['quality'][:len(snapshots[j-1]['quality'])], snapshots[j-1]['quality']))
                for key, value in source.state_dict().items():
                    self.assertTrue(torch.equal(value, row['weights'][key]))
            for j, row in enumerate(sessions):
                self.assertEqual(row['reference_accuracy'], baseline[j]['accuracy'])
                self.assertEqual(row['query_auxiliary_encodings'], 0)
                self.assertEqual(row['incremental_trainable_parameter_count'], 0)
                self.assertIn('competition_errors', row)
            self.assertTrue(sessions[0]['reference_scores_exact_equal'])
            state = runner.competitive_fusion_state
            self.assertNotIn('query', state['_session_statistics'])
            self.assertTrue(all(not p.requires_grad for p in state['model'].parameters()))
            self.assertTrue(state['base_bank_verified'] and state['reference_content_verified'])

    def test_zero_mode_exactly_reproduces_full_reference_and_predictions(self):
        with temporary_artifact_directory() as directory:
            root = Path(directory)
            _, zero, _, scores, _ = self.run_example(root/'zero', mode='zero')
            _, baseline, _, _, _ = self.run_example(root/'baseline', enabled=False)
            self.assertTrue(all(torch.equal(a,b) for a,b in scores))
            self.assertEqual([r['accuracy'] for r in zero], [r['accuracy'] for r in baseline])
            for j in range(3):
                with np.load(root/'zero/run'/f'predictions_session_{j:02d}.npz') as a, \
                     np.load(root/'baseline/run'/f'predictions_session_{j:02d}.npz') as b:
                    np.testing.assert_array_equal(a['prediction'], b['prediction'])

    def test_mismatched_base_bank_rejected_before_test_queries(self):
        with temporary_artifact_directory() as directory, self.assertRaisesRegex(ValueError, 'different set of base training images'):
            self.run_example(directory, wrong_bank=True)

    def test_load_rejects_corruption_and_reference_mismatch(self):
        with temporary_artifact_directory() as directory:
            root = Path(directory)
            cfg = config(root/'out', root/'fusion.npz')
            artifact(root/'fusion.npz', cfg)
            original = json.loads((root/'fusion.json').read_text())
            for key, value in [('format','other'), ('model_sha256','wrong'), ('dataset','cifar100'),
                               ('clip_weight_sha256','wrong'), ('reference_settings',{}), ('fixed',{}),
                               ('model_config',{}), ('trainable_parameters',0), ('blend',float('nan'))]:
                (root/'fusion.json').write_text(json.dumps({**original, key:value}))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    runtime.load_fusion(root/'fusion.npz', cfg, CLIP_HASH)
            (root/'fusion.json').write_text(json.dumps(original))
            state = runtime.load_fusion(root/'fusion.npz', cfg, CLIP_HASH)
            self.assertTrue(all(not p.requires_grad for p in state['model'].parameters()))
            with self.assertRaisesRegex(RuntimeError, 'Verify'):
                runtime.prepare_current(state, cfg, {})
            with self.assertRaisesRegex(ValueError, 'fresh'):
                runtime.export_fusion(root/'fusion.npz', state['model'], original, {})


if __name__ == '__main__':
    unittest.main()
