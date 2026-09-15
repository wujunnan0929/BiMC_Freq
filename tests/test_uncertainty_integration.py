"""Exercise real statistical/Runner paths with frozen synthetic CLIP features."""

import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from engine.engine import Runner
from engine.residual_training import make_reference_scorer
from test_residual_integration import (
    FeatureOnlyDatasetManager, make_model, synthetic_state,
    temporary_artifact_directory, tiny_config,
)


def uncertainty_config(output_dir='', control='frequency', alpha=0.1, base_only=False, auto=False):
    cfg = tiny_config(output_dir=output_dir)
    cfg.defrost()
    cfg.TRAINER.BiMC.RESIDUAL.ENABLED = False
    cfg.TRAINER.BiMC.CONSENSUS.ENABLED = False
    settings = cfg.TRAINER.BiMC.UNCERTAINTY
    settings.ENABLED = True
    settings.VIEW_CONTROL = control
    settings.AUTO_CALIBRATE = auto
    settings.ALPHA = float(alpha)
    settings.BASE_ONLY = base_only
    settings.OLD_WAY = 2
    settings.NEW_WAY = 1
    settings.STAGES = 1
    settings.OLD_SHOT = 4
    settings.SHOT = 2
    settings.QUERY = 2
    settings.VAL_EPISODES = 2
    cfg.freeze()
    return cfg


class UncertaintyIntegrationTest(unittest.TestCase):
    def test_reference_unchanged_and_zero_skips_frequency_encoding(self):
        cfg = uncertainty_config(alpha=0.0)
        model, state = make_model(cfg), synthetic_state()
        cfg_off = cfg.clone()
        cfg_off.defrost()
        cfg_off.TRAINER.BiMC.UNCERTAINTY.ENABLED = False
        cfg_off.freeze()
        baseline = make_model(cfg_off)
        features = state['images_features'][::5]
        reference = make_reference_scorer(model, cfg, state, 6)(features)
        expected = make_reference_scorer(baseline, cfg_off, state, 6)(features)
        self.assertTrue(torch.equal(reference, expected))
        model.uncertainty_state = {'alpha': 0.0}
        with patch.object(model, 'extract_frequency_img_feature') as encode:
            output = model.apply_frequency_uncertainty(features, features, reference, {})
        encode.assert_not_called()
        self.assertIs(output, reference)

    def test_modes_cannot_silently_stack(self):
        for section in ('FREQUENCY', 'CONSENSUS', 'RESIDUAL'):
            cfg = uncertainty_config()
            cfg.defrost()
            getattr(cfg.TRAINER.BiMC, section).ENABLED = True
            cfg.freeze()
            with self.assertRaisesRegex(ValueError, 'UNCERTAINTY requires'):
                make_model(cfg)

    def _run(self, directory, control='frequency', alpha=0.1, base_only=False, auto=False):
        cfg = uncertainty_config(directory, control, alpha, base_only, auto)
        full = synthetic_state(num_classes=10)
        manager, model = FeatureOnlyDatasetManager(cfg, full), make_model(cfg)

        def text_features(class_names, template, cls_begin_index):
            ids = torch.arange(cls_begin_index, cls_begin_index + len(class_names))
            return full['text_features'][ids], ids

        def descriptions(class_names, gpt_path, cls_begin_index):
            ids = torch.arange(cls_begin_index, cls_begin_index + len(class_names))
            mask = torch.isin(full['description_targets'], ids)
            return (full['description_features'][mask], full['description_targets'][mask],
                    full['description_proto'][ids], None)

        def frequency(images, original_features=None):
            features = images if original_features is None else original_features
            model.image_encoding_counts['auxiliary'] += 3 * len(features)
            return F.normalize(torch.stack((features, features.roll(1, -1),
                                            features * torch.linspace(0.5, 1.5, features.shape[-1])), 1), dim=-1)

        with patch('engine.engine.DatasetManager', return_value=manager), \
                patch('engine.engine.BiMC', return_value=model):
            runner = Runner(cfg)
        snapshots = []
        save = runner._save_checkpoint

        def capture(task_id, states):
            for item in states:
                for key in ('images_features', 'images_targets', 'frequency_features',
                            'uncertainty_features', 'frequency_description_candidates'):
                    self.assertNotIn(key, item)
            snapshots.append({
                'prior': model.uncertainty_state['prior'].clone(),
                'statistics': [{k: v.clone() for k, v in item['uncertainty_statistics'].items()}
                               for item in states],
            })
            save(task_id, states)

        original_loader = manager.get_dataloader

        def load(*args, **kwargs):
            if base_only and kwargs.get('source') == 'test':
                raise AssertionError('BASE_ONLY must not construct a benchmark test loader')
            return original_loader(*args, **kwargs)

        with patch.object(model, 'inference_text_feature', side_effect=text_features), \
                patch.object(model, 'inference_all_description_feature', side_effect=descriptions), \
                patch.object(model, 'extract_frequency_img_feature', side_effect=frequency), \
                patch.object(model, 'inference_frequency_text_feature', side_effect=AssertionError('No band text')), \
                patch.object(model, 'inference_explicit_frequency_description_candidates',
                             side_effect=AssertionError('No explicit band text')), \
                patch.object(runner, '_save_checkpoint', side_effect=capture), \
                patch.object(manager, 'get_dataloader', side_effect=load), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            sessions = runner.run()
        return model, sessions, snapshots

    def test_three_sessions_freeze_prior_and_old_statistics(self):
        for control in ('frequency', 'original'):
            with self.subTest(control=control), temporary_artifact_directory() as directory:
                model, sessions, snapshots = self._run(directory, control=control)
                self.assertEqual(len(sessions), 3)
                self.assertEqual(len(snapshots), 3)
                for snapshot in snapshots[1:]:
                    self.assertTrue(torch.equal(snapshots[0]['prior'], snapshot['prior']))
                    for key, value in snapshots[0]['statistics'][0].items():
                        self.assertTrue(torch.equal(value, snapshot['statistics'][0][key]))
                for key, value in snapshots[1]['statistics'][1].items():
                    self.assertTrue(torch.equal(value, snapshots[2]['statistics'][1][key]))
                self.assertTrue(torch.equal(snapshots[0]['statistics'][0]['count'], torch.full((6,), 8)))
                self.assertTrue(torch.equal(snapshots[2]['statistics'][2]['count'], torch.full((2,), 2)))
                for session in sessions:
                    self.assertEqual(session['query_auxiliary_encodings'],
                                     0 if control == 'original' else 3 * session['query_original_encodings'])
                    self.assertAlmostEqual(session['accuracy'] - session['reference_accuracy'],
                                           session['prediction_diagnostics']['all']['delta_accuracy_pp'])
                root = Path(directory)
                report = json.loads((root / 'uncertainty_calibration.json').read_text())
                self.assertEqual(report['data_source'], 'base_training_only')
                checkpoint = torch.load(root / 'checkpoint.pt', weights_only=False)
                self.assertTrue(torch.equal(checkpoint['uncertainty_state']['prior'], snapshots[0]['prior']))
                self.assertIsNone(checkpoint['router_state'])
                self.assertIsNone(checkpoint['residual_state'])
                self.assertEqual(json.loads((root / 'metrics.json').read_text())['status'], 'completed')
                self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))

    def test_zero_runner_is_exact_reference(self):
        with temporary_artifact_directory() as directory:
            _, sessions, _ = self._run(directory, alpha=0.0)
            for session in sessions:
                self.assertTrue(session['reference_scores_exact_equal'])
                self.assertEqual(session['query_auxiliary_encodings'], 0)
                self.assertEqual(session['prediction_change_rate'], 0.0)

    def test_auto_calibration_on_actual_built_statistics(self):
        with temporary_artifact_directory() as directory:
            model, sessions, snapshots = self._run(directory, auto=True)
            report = json.loads((Path(directory) / 'uncertainty_calibration.json').read_text())
            self.assertTrue(report['auto_calibrate'])
            self.assertFalse(set(report['fit_class_ids']) & set(report['validation_class_ids']))
            self.assertEqual(report['selected_alpha'], model.uncertainty_state['alpha'])
            self.assertTrue(torch.equal(snapshots[0]['prior'], snapshots[-1]['prior']))
            self.assertEqual(len(sessions), 3)

    def test_base_only_never_reads_test_images(self):
        with temporary_artifact_directory() as directory:
            _, sessions, snapshots = self._run(directory, base_only=True)
            self.assertEqual(sessions, [])
            self.assertEqual(len(snapshots), 1)
            metrics = json.loads((Path(directory) / 'metrics.json').read_text())
            self.assertEqual(metrics['status'], 'base_validation_completed')
            self.assertIsNone(metrics['summary'])
            self.assertFalse(list(Path(directory).glob('predictions_session_*.npz')))


if __name__ == '__main__':
    unittest.main()
