"""Frozen estimator provenance and actual three-session Runner integration."""
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
from engine import prototype_estimator_runtime as runtime
from engine.safd_metric_runtime import reference_settings, sample_identity_hash
from engine.prototype_evaluation import identity_calibration
from models.support_estimator import SupportPrototypeEstimator
from test_guidance_integration import Manager, configuration
from test_residual_integration import make_model, synthetic_state, temporary_artifact_directory
from test_safd_metric_runtime import CLIP_HASH, base_ids, reference_manager, reference_metadata


def config(output='', path='', enabled=True, blend=-1., calibration=True):
    cfg = configuration(output, enabled=False, auto=False)
    cfg.defrost()
    e = cfg.TRAINER.BiMC.PROTOTYPE_ESTIMATOR
    e.ENABLED, e.MODEL_PATH, e.BLEND, e.CALIBRATION = enabled, str(path), blend, calibration
    cfg.freeze()
    return cfg


def artifact(path, cfg, active=False, blend=1., calibration=None, wrong_bank=False):
    model = SupportPrototypeEstimator(hidden=8)
    if active:
        with torch.no_grad():
            model.class_head[-1].bias.fill_(.7)
    metadata = {'dataset': cfg.DATASET.NAME, 'feature_dim': 8,
                'clip_weight_sha256': CLIP_HASH, 'base_class_ids': list(range(6)),
                'base_sample_ids_sorted_sha256': 'wrong' if wrong_bank else sample_identity_hash(base_ids()),
                'reference_settings': reference_settings(cfg), 'data_source': 'base_training_only',
                'training_protocol': 'base_only_then_frozen', 'model_config': model.configuration(),
                'trainable_parameters': sum(p.numel() for p in model.parameters()),
                'selected_step': 1 if active else 0, 'blend': blend,
                'calibration': calibration or identity_calibration(), **reference_metadata(cfg)}
    runtime.export_estimator(path, model, metadata, {'purpose': 'synthetic integration fixture'})
    return model


class PrototypeEstimatorRuntimeTest(unittest.TestCase):
    def test_export_load_checks_reference_and_finite_parameter_identity(self):
        with temporary_artifact_directory() as directory:
            cfg = config(enabled=False)
            root = Path(directory)
            model = artifact(root / 'model.npz', cfg, active=True)
            state = runtime.load_estimator(root / 'model.npz', cfg, CLIP_HASH)
            self.assertFalse(any(p.requires_grad for p in state['model'].parameters()))
            for k, v in model.state_dict().items():
                self.assertTrue(torch.equal(v, state['model'].state_dict()[k]))
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                runtime.export_estimator(root / 'model.npz', model, state['metadata'], {})
            sidecar = root / 'model.json'
            original = json.loads(sidecar.read_text())
            for key, value, message in [('format', 'other', 'format'),
                                        ('model_sha256', 'wrong', 'checksum'),
                                        ('dataset', 'cifar100', 'dataset'),
                                        ('clip_weight_sha256', 'wrong', 'CLIP weights'),
                                        ('reference_settings', {}, 'reference settings'),
                                        ('blend', float('nan'), 'blend'),
                                        ('calibration', {}, 'calibration'),
                                        ('trainable_parameters', 1, 'parameter count')]:
                sidecar.write_text(json.dumps({**original, key: value}))
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, message):
                    runtime.load_estimator(root / 'model.npz', cfg, CLIP_HASH)
            sidecar.write_text(json.dumps(original))
            with np.load(root / 'model.npz', allow_pickle=False) as saved:
                values = {k: saved[k] for k in saved.files}
            values[next(iter(values))] = np.full_like(values[next(iter(values))], np.nan)
            np.savez_compressed(root / 'model.npz', **values)
            original['model_sha256'] = hashlib.sha256((root / 'model.npz').read_bytes()).hexdigest()
            sidecar.write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, 'shape/value'):
                runtime.load_estimator(root / 'model.npz', cfg, CLIP_HASH)

    def run_example(self, directory, enabled=True, blend=1., calibration=True, wrong_bank=False, calibrated=False):
        root = Path(directory)
        path = root / 'estimator.npz'
        cfg = config(root / 'run', path, enabled, blend, calibration)
        cal = {**identity_calibration(), 'novel_bias': .1} if calibrated else identity_calibration()
        source_model = artifact(path, cfg, active=True, calibration=cal, wrong_bank=wrong_bank)
        full = synthetic_state(num_classes=10, samples_per_class=20)
        model = make_model(cfg)
        model.clip_model.clip_weight_sha256 = CLIP_HASH
        manager = reference_manager(cfg, Manager(cfg, full, forbid_test=wrong_bank))

        def text(names, templates, start):
            ids = torch.arange(start, start + len(names))
            return full['text_features'][ids], ids

        def descriptions(class_names, gpt_path, cls_begin_index):
            ids = torch.arange(cls_begin_index, cls_begin_index + len(class_names))
            mask = torch.isin(full['description_targets'], ids)
            return full['description_features'][mask], full['description_targets'][mask], full['description_proto'][ids], None

        with patch('engine.engine.DatasetManager', return_value=manager), patch('engine.engine.BiMC', return_value=model):
            runner = Runner(cfg)
        snapshots, scores = [], []
        original_save = runner._save_checkpoint
        original_apply = runtime.apply_estimator

        def capture(task, states):
            snapshots.append({'reference': torch.cat([s['image_proto'] for s in states]).clone(),
                              'estimated': torch.cat([s.get('estimated_image_proto', s['image_proto']) for s in states]).clone(),
                              'weights': {k: v.clone() for k, v in runner.prototype_estimator_state['model'].state_dict().items()} if enabled else {}})
            for row in states:
                self.assertNotIn('images_features', row)
                self.assertNotIn('images_targets', row)
            original_save(task, states)

        def apply(state, cfg, merged, features, reference):
            result = original_apply(state, cfg, merged, features, reference)
            scores.append((reference.clone(), result.clone()))
            return result

        with patch.object(model, 'inference_text_feature', side_effect=text), \
             patch.object(model, 'inference_all_description_feature', side_effect=descriptions), \
             patch.object(model, 'extract_frequency_img_feature', side_effect=AssertionError('Unexpected auxiliary image encoding')), \
             patch.object(runner, '_save_checkpoint', side_effect=capture), \
             patch.object(runtime, 'apply_estimator', side_effect=apply), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            sessions = runner.run()
        return runner, sessions, snapshots, scores, source_model

    def test_actual_runner_freezes_estimator_and_old_prototypes_preserves_reference(self):
        with temporary_artifact_directory() as directory:
            root = Path(directory)
            (root / 'baseline').mkdir(); (root / 'estimated').mkdir()
            _, baseline, baseline_snapshots, _, _ = self.run_example(root / 'baseline', enabled=False)
            runner, sessions, snapshots, scores, source = self.run_example(root / 'estimated')
            self.assertEqual(len(sessions), 3)
            self.assertEqual([len(s['estimated']) for s in snapshots], [6, 8, 10])
            self.assertTrue(torch.equal(snapshots[0]['estimated'], snapshots[0]['reference']))
            self.assertFalse(torch.equal(snapshots[-1]['estimated'][6:], snapshots[-1]['reference'][6:]))
            for j, s in enumerate(snapshots):
                self.assertTrue(torch.equal(s['reference'], baseline_snapshots[j]['reference']))
                if j:
                    self.assertTrue(torch.equal(s['estimated'][:len(snapshots[j - 1]['estimated'])], snapshots[j - 1]['estimated']))
                for key, value in source.state_dict().items():
                    self.assertTrue(torch.equal(value, s['weights'][key]))
            self.assertTrue(any(not torch.equal(a, b) for a, b in scores))
            for j, s in enumerate(sessions):
                self.assertEqual(s['query_auxiliary_encodings'], 0)
                self.assertEqual(s['reference_accuracy'], baseline[j]['accuracy'])
                self.assertEqual(s['incremental_trainable_parameter_count'], 0)
                self.assertEqual(s['head_parameter_count'], sum(p.numel() for p in source.parameters()))
            state = runner.prototype_estimator_state
            self.assertNotIn('query', state['_session_statistics'])
            self.assertTrue(state['reference_content_verified'] and state['base_bank_verified'])
            report = json.loads((root / 'estimated/run/metrics.json').read_text())
            self.assertEqual(report['prototype_estimator']['incremental_trainable_parameters'], 0)

    def test_zero_estimator_without_calibration_is_exact_full_bimc(self):
        with temporary_artifact_directory() as directory:
            root = Path(directory)
            (root / 'baseline').mkdir(); (root / 'zero').mkdir()
            _, baseline, _, _, _ = self.run_example(root / 'baseline', enabled=False)
            _, zero, _, scores, _ = self.run_example(root / 'zero', blend=0., calibration=False, calibrated=True)
            self.assertTrue(all(torch.equal(a, b) for a, b in scores))
            self.assertTrue(all(s['reference_scores_exact_equal'] for s in zero))
            self.assertEqual([s['accuracy'] for s in zero], [s['accuracy'] for s in baseline])
            for i in range(3):
                with np.load(root / 'baseline/run' / f'predictions_session_{i:02d}.npz') as a, \
                     np.load(root / 'zero/run' / f'predictions_session_{i:02d}.npz') as b:
                    np.testing.assert_array_equal(a['prediction'], b['prediction'])

    def test_calibration_only_control_does_not_change_saved_prototypes(self):
        with temporary_artifact_directory() as directory:
            _, _, snapshots, scores, _ = self.run_example(directory, blend=0., calibrated=True)
            self.assertTrue(all(torch.equal(s['reference'], s['estimated']) for s in snapshots))
            self.assertTrue(any(not torch.equal(a, b) for a, b in scores))

    def test_changed_base_bank_is_rejected_before_benchmark_queries(self):
        with temporary_artifact_directory() as directory, self.assertRaisesRegex(ValueError, 'different set of base training images'):
            self.run_example(directory, wrong_bank=True)


if __name__ == '__main__':
    unittest.main()
