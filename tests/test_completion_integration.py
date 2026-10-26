"""Actual Runner integration on frozen synthetic CLIP features, without downloads."""
import contextlib
import copy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from engine.engine import Runner
from main import setup_cfg
from test_guidance_integration import Manager
from test_residual_integration import make_model, synthetic_state, temporary_artifact_directory


ROOT = Path(__file__).resolve().parents[1]


def configuration(directory='', *, enabled=True, blend=.5, auto=True, base_only=False,
                  control='frequency', semantic='structured', permutation=(0, 1, 2)):
    cfg = setup_cfg(str(ROOT / 'configs/datasets/cub200.yaml'),
                    str(ROOT / 'configs/trainers/bimc_prototype_completion.yaml'))
    cfg.defrost()
    cfg.SEED = 11
    cfg.OUTPUT_DIR = str(directory)
    cfg.DEVICE.DEVICE_NAME = 'cpu'
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.DATASET.NUM_CLASSES, cfg.DATASET.NUM_INIT_CLS, cfg.DATASET.NUM_INC_CLS = 10, 6, 2
    cfg.DATASET.NUM_INC_SHOT = 2
    c = cfg.TRAINER.BiMC.PROTOTYPE_COMPLETION
    c.ENABLED, c.BLEND, c.AUTO_CALIBRATE, c.BASE_ONLY = enabled, blend, auto, base_only
    c.VIEW_CONTROL, c.SEMANTIC_SOURCE = control, semantic
    c.SEMANTIC_PERMUTATION = list(permutation)
    c.RANK, c.VAL_SEQUENCES, c.SHOT, c.QUERY = 2, 2, 2, 2
    c.CANDIDATES_PER_BAND = 3
    cfg.freeze()
    return cfg


class CompletionIntegrationTest(unittest.TestCase):
    def example(self, directory='', reverse_query=False, **options):
        cfg = configuration(directory, **options)
        full = synthetic_state(num_classes=10, samples_per_class=20)
        model = make_model(cfg)
        manager = Manager(cfg, full, cfg.TRAINER.BiMC.PROTOTYPE_COMPLETION.BASE_ONLY, reverse_query)

        def text(names, templates, start):
            ids = torch.arange(start, start + len(names))
            return full['text_features'][ids], ids

        def descriptions(class_names, gpt_path, cls_begin_index):
            ids = torch.arange(cls_begin_index, cls_begin_index + len(class_names))
            mask = torch.isin(full['description_targets'], ids)
            return full['description_features'][mask], full['description_targets'][mask], full['description_proto'][ids], None

        def structured(names):
            start = int(str(names[0]).split('_')[-1])
            return full['frequency_prompt_proto'][start:start + len(names)]

        def candidates(names, path, max_candidates=None):
            start = int(str(names[0]).split('_')[-1])
            return full['frequency_description_candidates'][start:start + len(names), :, :max_candidates]

        def support_views(images, original_features):
            original = F.normalize(original_features.float(), dim=-1)
            control = cfg.TRAINER.BiMC.PROTOTYPE_COMPLETION.VIEW_CONTROL
            if control == 'original':
                return original[:, None]
            if control == 'repeat':
                return original[:, None].expand(-1, 2, -1).clone()
            model.image_encoding_counts['auxiliary'] += len(images)
            auxiliary = F.normalize(images + .4 * images.roll(1, -1), dim=-1)
            return torch.stack((original, auxiliary), dim=1)

        snapshots, base = [], []
        build_statistics = model.build_task_statistics

        def build(*args, **kwargs):
            current = build_statistics(*args, **kwargs)
            if int(current['class_index'][0]) == 0:
                base.append(dict(current))
            return current

        with patch('engine.engine.DatasetManager', return_value=manager), \
                patch('engine.engine.BiMC', return_value=model):
            runner = Runner(cfg)
        save = runner._save_checkpoint

        def capture(task, states):
            for state in states:
                for key in ('images_features', 'images_targets', 'completion_features', 'completion_text'):
                    self.assertNotIn(key, state)
            snapshots.append({'prototypes': [state['image_proto'].clone() for state in states],
                              'completion_state': copy.deepcopy(model.completion_state)})
            save(task, states)

        with patch.object(model, 'inference_text_feature', side_effect=text), \
                patch.object(model, 'inference_all_description_feature', side_effect=descriptions), \
                patch.object(model, 'inference_frequency_text_feature', side_effect=structured), \
                patch.object(model, 'inference_explicit_frequency_description_candidates', side_effect=candidates), \
                patch.object(model, 'extract_completion_img_feature', side_effect=support_views), \
                patch.object(model, 'build_task_statistics', side_effect=build), \
                patch.object(runner, '_save_checkpoint', side_effect=capture), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            sessions = runner.run()
        return model, runner, sessions, snapshots, base[0]

    def assert_nested_equal(self, first, second):
        if torch.is_tensor(first):
            self.assertTrue(torch.equal(first, second))
        elif isinstance(first, dict):
            self.assertEqual(set(first), set(second))
            for key in first:
                self.assert_nested_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(first, second)

    def test_actual_runner_fits_once_freezes_history_and_never_encodes_query_views(self):
        with temporary_artifact_directory() as directory:
            model, runner, sessions, snapshots, _ = self.example(directory, auto=False, blend=.5)
            self.assertEqual(len(sessions), 3)
            self.assertEqual(runner.completion_report['selected_blend'], .5)
            self.assertFalse(runner.completion_report['gradient_training_performed'])
            for stage, session in enumerate(sessions):
                self.assertEqual(session['query_auxiliary_encodings'], 0)
                self.assertEqual(session['query_original_encodings'], 3 * (6 + 2 * stage))
                count = 6 * 16 if stage == 0 else 2 * 2
                self.assertEqual(session['support_image_encodings']['original'], count)
                self.assertEqual(session['support_image_encodings']['auxiliary'], count)
                self.assertEqual(session['incremental_trainable_parameter_count'], 0)
                self.assertIn('competition_errors', session)
            for snapshot in snapshots[1:]:
                self.assert_nested_equal(snapshots[0]['completion_state'], snapshot['completion_state'])
                self.assertTrue(torch.equal(snapshots[0]['prototypes'][0], snapshot['prototypes'][0]))
            self.assertTrue(torch.equal(snapshots[1]['prototypes'][1], snapshots[2]['prototypes'][1]))
            checkpoint = torch.load(Path(directory) / 'checkpoint.pt', weights_only=False)
            self.assertIsNotNone(checkpoint['completion_state'])
            self.assertTrue((Path(directory) / 'completion_calibration.json').is_file())
            metrics = json.loads((Path(directory) / 'metrics.json').read_text())
            self.assertIn('prototype_completion', metrics)
            self.assertFalse(any(p.requires_grad for p in model.parameters()))

    def test_zero_strength_is_exact_original_bimc_in_all_sessions(self):
        with temporary_artifact_directory() as directory:
            root = Path(directory)
            _, _, a, sa, _ = self.example(root / 'zero', auto=False, blend=0.)
            _, _, b, sb, _ = self.example(root / 'baseline', enabled=False, auto=False)
            for stage, (left, right) in enumerate(zip(a, b)):
                self.assertEqual(left['accuracy'], right['accuracy'])
                self.assertEqual(left['reference_accuracy'], right['reference_accuracy'])
                with np.load(root / 'zero' / f'predictions_session_{stage:02d}.npz') as x, \
                        np.load(root / 'baseline' / f'predictions_session_{stage:02d}.npz') as y:
                    for name in ('sample_id', 'target', 'reference_prediction', 'prediction'):
                        np.testing.assert_array_equal(x[name], y[name])
            for x, y in zip(sa[-1]['prototypes'], sb[-1]['prototypes']):
                self.assertTrue(torch.equal(x, y))

    def test_query_order_does_not_affect_selection_or_prototypes(self):
        _, first, _, a, _ = self.example()
        _, second, _, b, _ = self.example(reverse_query=True)
        self.assertEqual(first.completion_report['protocol_sha256'], second.completion_report['protocol_sha256'])
        self.assertEqual(first.completion_report['selected_blend'], second.completion_report['selected_blend'])
        self.assert_nested_equal(a[-1]['completion_state'], b[-1]['completion_state'])
        for x, y in zip(a[-1]['prototypes'], b[-1]['prototypes']):
            self.assertTrue(torch.equal(x, y))

    def test_base_only_does_not_request_benchmark_queries(self):
        with temporary_artifact_directory() as directory:
            _, _, sessions, _, _ = self.example(directory, base_only=True)
            self.assertEqual(sessions, [])
            metrics = json.loads((Path(directory) / 'metrics.json').read_text())
            self.assertEqual(metrics['status'], 'base_validation_completed')

    def test_fixed_text_controls_and_permutation_are_observable(self):
        _, _, _, _, ordinary = self.example(semantic='ordinary', base_only=True, auto=False)
        expected = F.normalize(ordinary['description_features'].reshape(6, 3, -1).mean(1), dim=-1)
        torch.testing.assert_close(ordinary['completion_text'][:, 1], expected)
        _, _, _, _, paired = self.example(base_only=True, auto=False)
        _, _, _, _, wrong = self.example(base_only=True, auto=False, permutation=(2, 0, 1))
        self.assertFalse(torch.equal(paired['completion_text'][:, 1], wrong['completion_text'][:, 1]))
        self.assertTrue(torch.equal(paired['completion_text'][:, 0], wrong['completion_text'][:, 0]))

    def test_original_and_repeat_controls_do_not_encode_auxiliary_supports(self):
        for control in ('original', 'repeat'):
            with self.subTest(control=control):
                _, _, sessions, _, base = self.example(control=control, auto=False)
                self.assertEqual(base['completion_features'].shape[1], 1 if control == 'original' else 2)
                self.assertTrue(all(s['support_image_encodings']['auxiliary'] == 0 for s in sessions))

    def test_actual_support_extractor_encodes_only_selected_view(self):
        cfg = configuration(auto=False)
        model = make_model(cfg)
        images = torch.randn(4, 3, 12, 12)
        original = F.normalize(torch.randn(4, 3), dim=-1)
        calls = []

        class Views(torch.nn.Module):
            def forward(self, values):
                return torch.stack((values, .5 * values, .8 * values), 1)

        def encode(values):
            calls.append(values.clone())
            return values.mean((-1, -2))

        model.completion_views = Views()
        with patch.object(model.clip_model, 'encode_image', side_effect=encode):
            result = model.extract_completion_img_feature(images, original)
        self.assertEqual(result.shape, (4, 2, 3))
        self.assertTrue(torch.equal(result[:, 0], original))
        self.assertEqual(len(calls), 1)
        self.assertTrue(torch.equal(calls[0], .5 * images))
        self.assertEqual(model.image_encoding_counts, {'original': 0, 'auxiliary': 4})


if __name__ == '__main__':
    unittest.main()
