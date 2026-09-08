"""Exercise actual BiMC/Runner logic with a frozen tiny encoder, no downloads."""

import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from engine.residual_training import make_reference_scorer
from test_residual_integration import (
    FeatureOnlyDatasetManager, make_model, synthetic_state,
    temporary_artifact_directory, tiny_config,
)


def consensus_config(output_dir='', control='frequency', enabled=True):
    cfg = tiny_config(output_dir=output_dir)
    cfg.defrost()
    cfg.TRAINER.BiMC.RESIDUAL.ENABLED = False
    settings = cfg.TRAINER.BiMC.CONSENSUS
    settings.ENABLED = enabled
    settings.VIEW_CONTROL = control
    settings.FIT_EPISODES = 2
    settings.VAL_EPISODES = 2
    settings.OLD_WAY = 2
    settings.NEW_WAY = 1
    settings.STAGES = 1
    settings.OLD_SHOT = 4
    settings.SHOT = 2
    settings.QUERY = 2
    settings.AUTO_CALIBRATE = False
    settings.LAMBDA = 0.03
    frequency = cfg.TRAINER.BiMC.FREQUENCY
    frequency.ENABLED = False
    frequency.USE_EXPLICIT_DESCRIPTIONS = True
    frequency.EXPLICIT_DESCRIPTION_PATH = 'synthetic-candidates.json'
    frequency.VIEW_MODE = 'natural'
    cfg.freeze()
    return cfg


class ConsensusIntegrationTest(unittest.TestCase):
    def test_candidate_reference_is_exact_legacy_bimc(self):
        state = synthetic_state()
        plain = make_model(consensus_config(enabled=False))
        model = make_model(consensus_config())
        features = state['images_features'][::5]
        expected = make_reference_scorer(plain, plain.cfg, state, 6)(features)
        actual = make_reference_scorer(model, model.cfg, state, 6)(features)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(model.frequency_enabled)
        self.assertFalse(model.frequency_fusion_enabled)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_original_control_reuses_encoding_and_augmentation_has_three_views(self):
        model = make_model(consensus_config(control='original'))
        images, features = torch.randn(2, 3, 16, 16), torch.randn(2, 8)
        with patch.object(model.clip_model, 'encode_image') as encode:
            views = model.extract_frequency_img_feature(images, original_features=features)
        encode.assert_not_called()
        self.assertTrue(torch.equal(views[:, 1], F.normalize(features, dim=-1)))
        self.assertEqual(model.image_encoding_counts['auxiliary'], 0)
        model = make_model(consensus_config(control='augmentation'))
        calls = []

        def encode(images):
            calls.append(images.clone())
            return images.flatten(1)[:, :8]

        with patch.object(model.clip_model, 'encode_image', side_effect=encode):
            views = model.extract_frequency_img_feature(images)
        self.assertEqual(views.shape, (2, 3, 8))
        self.assertEqual(len(calls), 3)
        self.assertEqual(model.image_encoding_counts['auxiliary'], 6)
        self.assertFalse(torch.equal(calls[0], calls[1]))

    def test_zero_strength_skips_all_auxiliary_encoding_exactly(self):
        model = make_model(consensus_config())
        model.consensus_state = {'lambda': 0.0}
        reference = torch.tensor([[.51, .49], [.5, .5]])
        with patch.object(model, 'extract_frequency_img_feature') as encode:
            actual, detail = model.apply_frequency_consensus(
                torch.randn(2, 8), torch.randn(2, 8), reference,
                torch.randn(2, 3, 8), torch.randn(2, 3, 8),
            )
        encode.assert_not_called()
        self.assertIs(actual, reference)
        self.assertFalse(detail['eligible'].any())
        self.assertTrue(detail['evidence'].isnan().all())

    def test_runner_three_sessions_freezes_calibration_and_releases_samples(self):
        from engine.engine import Runner

        with temporary_artifact_directory() as directory:
            cfg = consensus_config(directory)
            state = synthetic_state(num_classes=10)
            manager = FeatureOnlyDatasetManager(cfg, state)
            model = make_model(cfg)

            def text_features(names, template, start):
                ids = torch.arange(start, start + len(names))
                return state['text_features'][ids], ids

            def descriptions(class_names, gpt_path, cls_begin_index):
                ids = torch.arange(cls_begin_index, cls_begin_index + len(class_names))
                mask = torch.isin(state['description_targets'], ids)
                return (state['description_features'][mask], state['description_targets'][mask],
                        state['description_proto'][ids], None)

            def explicit(names, path):
                ids = torch.tensor([int(name.split('_')[-1]) for name in names])
                return state['frequency_description_candidates'][ids]

            def frequency(images, original_features=None):
                model.image_encoding_counts['auxiliary'] += 3 * len(images)
                return F.normalize(torch.stack((images, images.roll(1, -1),
                                                 images.roll(2, -1)), dim=1), dim=-1)

            with patch('engine.engine.DatasetManager', return_value=manager), \
                    patch('engine.engine.BiMC', return_value=model):
                runner = Runner(cfg)
            snapshots = []
            save_checkpoint = runner._save_checkpoint

            def capture(task_id, states):
                for current in states:
                    for key in ('images_features', 'images_targets', 'frequency_features',
                                'frequency_description_candidates'):
                        self.assertNotIn(key, current)
                    self.assertNotIn('frequency_calibrated_proto', current)
                    ids = torch.as_tensor(current['class_index'])
                    expected = F.normalize(
                        state['frequency_description_candidates'][ids].mean(2), dim=-1,
                    )
                    self.assertTrue(torch.allclose(current['frequency_consensus_text_proto'], expected))
                snapshots.append(json.dumps(runner.consensus_report, sort_keys=True))
                save_checkpoint(task_id, states)

            with patch.object(model, 'inference_text_feature', side_effect=text_features), \
                    patch.object(model, 'inference_all_description_feature', side_effect=descriptions), \
                    patch.object(model, 'inference_explicit_frequency_description_candidates', side_effect=explicit), \
                    patch.object(model, 'extract_frequency_img_feature', side_effect=frequency), \
                    patch.object(runner, '_save_checkpoint', side_effect=capture), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                sessions = runner.run()
            self.assertEqual(len(sessions), 3)
            self.assertEqual(snapshots[0], snapshots[2])
            for row in sessions:
                report = row['prediction_diagnostics']
                self.assertEqual(report['prediction_outside_reference_top2_count'], 0)
                self.assertAlmostEqual(row['accuracy'] - row['reference_accuracy'],
                                       report['all']['delta_accuracy_pp'])
            artifact = Path(directory)
            self.assertTrue((artifact / 'consensus_calibration.json').exists())
            self.assertTrue((artifact / 'predictions_session_02.npz').exists())
            checkpoint = torch.load(artifact / 'checkpoint.pt', weights_only=False)
            self.assertIsNotNone(checkpoint['consensus_state'])
            self.assertIsNone(checkpoint['residual_state'])
            self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))


if __name__ == '__main__':
    unittest.main()
