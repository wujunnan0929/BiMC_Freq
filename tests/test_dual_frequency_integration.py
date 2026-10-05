"""Exercise calibration, the actual Runner, and memory boundaries together."""

import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from engine.dual_frequency_calibration import _records, initialize_dual_frequency
from engine.engine import Runner
from main import setup_cfg
from test_residual_integration import (
    FeatureOnlyDatasetManager, make_model, synthetic_state, temporary_artifact_directory,
)
from test_uncertainty_integration import uncertainty_config
from tools import run_safd_dual_experiments as launcher


def dual_config(directory='', strength=.6, auto=False, source='explicit', base_only=False):
    cfg = uncertainty_config(directory, control='original', auto=auto)
    cfg.defrost()
    u = cfg.TRAINER.BiMC.UNCERTAINTY
    u.COVARIANCE = 'full_shared'
    u.MEAN_UNCERTAINTY = False
    u.RIDGE_GRID = [0.1, 1.0]
    u.TEMPERATURE_GRID = [0.3]
    u.ALPHA_GRID = [0.0, 0.1]
    f = cfg.TRAINER.BiMC.FREQUENCY
    f.BACKEND = 'safd'
    f.EXPLICIT_DESCRIPTION_PATH = 'description/cub200_frequency_descriptions_deepseek.json'
    d = cfg.TRAINER.BiMC.DUAL_FREQUENCY
    d.ENABLED = True
    d.SEMANTIC_SOURCE = source
    d.AUTO_CALIBRATE = auto
    d.LAMBDA = strength
    d.FIT_EPISODES = 2
    d.BASE_ONLY = base_only
    cfg.freeze()
    return cfg


class DualFrequencyIntegrationTest(unittest.TestCase):
    def run_example(self, directory, *, strength=.6, auto=False, source='explicit', base_only=False):
        cfg = dual_config(directory, strength, auto, source, base_only)
        full = synthetic_state(num_classes=10)
        model = make_model(cfg)
        manager = FeatureOnlyDatasetManager(cfg, full)

        def text(class_names, template, cls_begin_index):
            ids = torch.arange(cls_begin_index, cls_begin_index + len(class_names))
            return full['text_features'][ids], ids

        def descriptions(class_names, gpt_path, cls_begin_index):
            ids = torch.arange(cls_begin_index, cls_begin_index + len(class_names))
            mask = torch.isin(full['description_targets'], ids)
            if source == 'ordinary':
                mask &= torch.arange(len(mask)) % 5 != 0  # Unequal counts per class.
            return (full['description_features'][mask], full['description_targets'][mask],
                    full['description_proto'][ids], None)

        def frequency(images, original_features=None):
            model.image_encoding_counts['auxiliary'] += 3 * len(images)
            return F.normalize(torch.stack([images + .5 * images.roll(i, -1) for i in (1, 2, 3)], 1), dim=-1)

        def candidates(class_names, description_path):
            ids = torch.arange(len(class_names)) + int(str(class_names[0]).split('_')[-1])
            return full['frequency_description_candidates'][ids]

        with patch('engine.engine.DatasetManager', return_value=manager), patch('engine.engine.BiMC', return_value=model):
            runner = Runner(cfg)
        snapshots, base_states = [], []
        save = runner._save_checkpoint
        build = model.build_task_statistics

        def capture_build(*args, **kwargs):
            state = build(*args, **kwargs)
            if int(state['class_index'][0]) == 0:
                base_states.append({k: v.clone() if torch.is_tensor(v) else v for k, v in state.items()})
            return state

        def capture(task_id, states):
            for state in states:
                for key in ('images_features', 'images_targets', 'frequency_features', 'uncertainty_features'):
                    self.assertNotIn(key, state)
            snapshots.append({'precision': model.uncertainty_state['precision'].clone(),
                              'lambda': model.dual_frequency_state['lambda'],
                              'scales': model.dual_frequency_state['scales'].clone(),
                              'stats': [{k: v.clone() for k, v in s['dual_statistics'].items()} for s in states]})
            save(task_id, states)

        original_loader = manager.get_dataloader

        def loader(*args, **kwargs):
            if base_only and kwargs.get('source') == 'test':
                raise AssertionError('Base-only mode must not load benchmark images')
            return original_loader(*args, **kwargs)

        with patch.object(model, 'inference_text_feature', side_effect=text), \
                patch.object(model, 'inference_all_description_feature', side_effect=descriptions), \
                patch.object(model, 'extract_frequency_img_feature', side_effect=frequency), \
                patch.object(model, 'inference_explicit_frequency_description_candidates', side_effect=candidates), \
                patch.object(model, 'inference_frequency_text_feature', side_effect=lambda names: full['frequency_prompt_proto'][:len(names)]), \
                patch.object(model, 'build_task_statistics', side_effect=capture_build), \
                patch.object(runner, '_save_checkpoint', side_effect=capture), \
                patch.object(manager, 'get_dataloader', side_effect=loader), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            sessions = runner.run()
        return model, runner, sessions, snapshots, base_states[0]

    def test_three_sessions_freeze_metric_scales_and_support_statistics(self):
        with temporary_artifact_directory() as directory:
            model, _, sessions, snapshots, _ = self.run_example(directory)
            self.assertEqual(len(sessions), 3)
            for snapshot in snapshots[1:]:
                self.assertTrue(torch.equal(snapshot['precision'], snapshots[0]['precision']))
                self.assertTrue(torch.equal(snapshot['scales'], snapshots[0]['scales']))
                self.assertEqual(snapshot['lambda'], snapshots[0]['lambda'])
                for key, value in snapshots[0]['stats'][0].items():
                    self.assertTrue(torch.equal(value, snapshot['stats'][0][key]))
            for session in sessions:
                self.assertAlmostEqual(session['accuracy'] - session['anchor_accuracy'],
                                       session['frequency_prediction_diagnostics']['all']['delta_accuracy_pp'])
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))
            checkpoint = torch.load(Path(directory) / 'checkpoint.pt', weights_only=False)
            self.assertIsNone(checkpoint['router_state'])
            self.assertIsNone(checkpoint['residual_state'])
            self.assertIsNotNone(checkpoint['dual_frequency_state'])
            with np.load(Path(directory) / 'predictions_session_02.npz') as predictions:
                self.assertIn('anchor_prediction', predictions.files)
                self.assertIn('anchor_second_candidate', predictions.files)

    def test_zero_dual_preserves_anchor_and_skips_query_auxiliary_encodings(self):
        with temporary_artifact_directory() as directory:
            model, _, sessions, _, _ = self.run_example(directory, strength=0.)
            for session in sessions:
                self.assertTrue(session['anchor_scores_exact_equal'])
                self.assertEqual(session['query_auxiliary_encodings'], 0)
                self.assertEqual(session['frequency_prediction_diagnostics']['all']['changed'], 0)
            with patch.object(model, 'extract_frequency_img_feature') as encode:
                anchor = torch.ones(2, 3)
                output, _ = model.apply_dual_frequency(torch.randn(2, 8), torch.randn(2, 8), anchor, {})
            self.assertIs(output, anchor)
            encode.assert_not_called()

    def test_auto_calibration_disjointness_guards_and_episode_support_grounding(self):
        with temporary_artifact_directory() as directory:
            model, runner, sessions, _, base = self.run_example(directory, auto=True)
            report = runner.dual_frequency_report
            self.assertEqual(report['protocol_sha256'], runner.uncertainty_report['protocol_sha256'])
            self.assertFalse(set(report['fit_class_ids']) & set(report['validation_class_ids']))
            selected = next(r for r in report['candidate_results'] if r['lambda'] == report['selected_lambda'])
            self.assertTrue(selected['selection']['feasible'])
            for row in selected['stages'][1:]:
                for name in report['guard_groups']:
                    group = row['groups'][name]
                    if group['count']:
                        self.assertGreaterEqual(group['delta_accuracy_pp'], -1e-12)
            sequence = report['validation_sequences'][0]
            first = list(_records(model, model.cfg, base, [sequence], model.uncertainty_state['precision']))
            changed = {**base, 'images_features': base['images_features'].clone(),
                       'frequency_features': base['frequency_features'].clone()}
            query = [i for group in sequence['query_indices'] for i in group]
            changed['images_features'][query] *= -1
            changed['frequency_features'][query] *= -1
            second = list(_records(model, model.cfg, changed, [sequence], model.uncertainty_state['precision']))
            for before, after in zip(first, second):
                for key in before['alignment']:
                    self.assertTrue(torch.equal(before['alignment'][key], after['alignment'][key]), key)
            self.assertEqual(len(sessions), 3)
            with self.assertRaises(RuntimeError):
                initialize_dual_frequency(model, model.cfg, base, runner.uncertainty_report)

    def test_base_only_and_semantic_controls_use_actual_runner(self):
        for source in ('ordinary', 'structured'):
            with self.subTest(source=source), temporary_artifact_directory() as directory:
                _, _, sessions, snapshots, base = self.run_example(directory, source=source, base_only=True)
                self.assertEqual(sessions, [])
                self.assertEqual(len(snapshots), 1)
                self.assertEqual(base['dual_description_candidates'].shape[1], 3)
                self.assertEqual(base['dual_description_candidates'].shape[2], 5 if source == 'ordinary' else 1)
                self.assertFalse(list(Path(directory).glob('predictions_session_*.npz')))
                self.assertEqual(json.loads((Path(directory) / 'metrics.json').read_text())['status'], 'base_validation_completed')

    def test_future_or_test_samples_cannot_initialize_dual_calibration(self):
        with temporary_artifact_directory() as directory:
            model, runner, _, _, base = self.run_example(directory, base_only=True)
            model.dual_frequency_state = None
            with self.assertRaisesRegex(ValueError, 'base training data only'):
                initialize_dual_frequency(model, model.cfg, {**base, 'data_source': 'test'},
                                          runner.uncertainty_report)
            future = {**base, 'images_targets': base['images_targets'].clone()}
            future['images_targets'][0] = 8
            with self.assertRaisesRegex(ValueError, 'base-class training'):
                initialize_dual_frequency(model, model.cfg, future, runner.uncertainty_report)
            self.assertIsNone(model.dual_frequency_state)

    def test_configs_matrix_and_invalid_stacking(self):
        plan = launcher.build_plan(launcher.parse_args(['--suite', 'all', '--seeds', '1']))
        self.assertEqual(len(plan), 19)
        self.assertEqual(len({r['support_manifest'] for r in plan}), 1)
        self.assertEqual(len({r['fingerprint'] for r in plan}), 19)
        for row in plan:
            opts = row['command'][row['command'].index('--opts') + 1:]
            cfg = setup_cfg('configs/datasets/cub200.yaml', 'configs/trainers/bimc_safd_dual.yaml', opts)
            if cfg.TRAINER.BiMC.DUAL_FREQUENCY.ENABLED:
                self.assertEqual(cfg.TRAINER.BiMC.UNCERTAINTY.VIEW_CONTROL, 'original')
                make_model(cfg)
        cfg = dual_config()
        cfg.defrost()
        cfg.TRAINER.BiMC.FREQUENCY.ROUTER.ENABLED = True
        cfg.freeze()
        with self.assertRaisesRegex(ValueError, 'DUAL_FREQUENCY requires'):
            make_model(cfg)
        with self.assertRaises(ValueError):
            launcher.build_plan(launcher.parse_args(['--opts', launcher.D+'SEMANTIC_SOURCE', 'ordinary']))


if __name__ == '__main__':
    unittest.main()
