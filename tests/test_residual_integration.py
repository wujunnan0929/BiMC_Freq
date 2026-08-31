"""Real PyTorch integration on cached synthetic features, without CLIP downloads."""

import contextlib
import io
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from main import setup_cfg
from models.bimc import BiMC
from models.incremental_residual import LowRankResidualHead
from engine.residual_training import (
    initialize_residual,
    make_episode_reference,
    make_reference_scorer,
)


ROOT = Path(__file__).resolve().parents[1]


@contextlib.contextmanager
def temporary_artifact_directory():
    # Python 3.13's Windows 0700 tempfile directories can exclude restricted
    # runner tokens. Normal workspace inheritance also works on other systems.
    parent = ROOT / '.codex_tmp' / 'integration_tests'
    parent.mkdir(parents=True, exist_ok=True)
    directory = parent / uuid4().hex
    directory.mkdir()
    try:
        yield str(directory)
    finally:
        resolved = directory.resolve()
        if resolved.parent != parent.resolve() or resolved.name != directory.name:
            raise RuntimeError('Refusing to clean a test directory outside its artifact root')
        shutil.rmtree(resolved)


def tiny_config(frequency=False, dictionary='meta', output_dir='', gain=1.0):
    cfg = setup_cfg(
        str(ROOT / 'configs' / 'datasets' / 'cub200.yaml'),
        str(ROOT / 'configs' / 'trainers' / 'bimc_incremental_residual.yaml'),
    )
    cfg.defrost()
    cfg.SEED = 13
    cfg.OUTPUT_DIR = str(output_dir)
    cfg.DEVICE.DEVICE_NAME = 'cpu'
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.DATASET.NUM_CLASSES = 10
    cfg.DATASET.NUM_INIT_CLS = 6
    cfg.DATASET.NUM_INC_CLS = 2
    cfg.DATASET.NUM_INC_SHOT = 2
    cfg.TRAINER.BiMC.PREC = 'fp32'
    residual = cfg.TRAINER.BiMC.RESIDUAL
    residual.DICTIONARY = dictionary
    residual.RANK = 2
    residual.SVD_REPEATS = 2
    residual.SVD_REFERENCE_SHOT = 4
    residual.TRAIN_STEPS = 3
    residual.LR = 0.1
    residual.GAIN = gain
    residual.META_STEPS = 2
    residual.META_INNER_STEPS = 2
    residual.META_WAY = 4
    residual.META_OLD_WAY = 2
    residual.META_OLD_SHOT = 4
    residual.META_QUERY = 2
    residual.META_VAL_FRACTION = 1 / 3
    residual.META_VAL_EPISODES = 1
    frequency_cfg = cfg.TRAINER.BiMC.FREQUENCY
    frequency_cfg.ENABLED = frequency
    frequency_cfg.ROUTER.ENABLED = False
    frequency_cfg.USE_EXPLICIT_DESCRIPTIONS = frequency
    frequency_cfg.EXPLICIT_DESCRIPTION_PATH = 'synthetic-no-file-required.json'
    frequency_cfg.DESCRIPTION_TOPK = 2
    frequency_cfg.RELIABILITY_ALPHA = frequency
    cfg.freeze()
    return cfg


class IdentityFeatureEncoder(nn.Module):
    """The only mocked encoder consumes features, never images or tokens."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def encode_image(self, features):
        return self.scale * features


def make_model(cfg):
    with patch('models.bimc.load_clip_to_cpu', return_value=IdentityFeatureEncoder()):
        return BiMC(cfg, ['a photo of a {}.'], 'cpu')


def synthetic_state(num_classes=6, samples_per_class=12, feature_dim=8, seed=31):
    generator = torch.Generator().manual_seed(seed)

    def noise(*shape):
        return torch.randn(*shape, generator=generator)

    centers = F.normalize(noise(num_classes, feature_dim), dim=-1)
    labels = torch.arange(num_classes).repeat_interleave(samples_per_class)
    features = F.normalize(centers[labels] + 0.22 * noise(len(labels), feature_dim), dim=-1)
    raw_means = torch.stack([features[labels == c].mean(0) for c in range(num_classes)])
    descriptions = F.normalize(
        centers[:, None, :] + 0.16 * noise(num_classes, 3, feature_dim), dim=-1,
    )
    frequency = F.normalize(
        features[:, None, :] + 0.15 * noise(len(labels), 3, feature_dim), dim=-1,
    )
    raw_frequency = torch.stack([
        frequency[labels == c].mean(0) for c in range(num_classes)
    ])
    return {
        'class_index': np.arange(num_classes),
        'images_features': features,
        'images_targets': labels,
        'image_proto': F.normalize(raw_means, dim=-1),
        'raw_image_mean': raw_means,
        'cov_image': torch.eye(feature_dim) * 0.2,
        'description_proto': F.normalize(descriptions.mean(1), dim=-1),
        'description_features': descriptions.reshape(-1, feature_dim),
        'description_targets': torch.arange(num_classes).repeat_interleave(3),
        'text_features': centers,
        'text_targets': torch.arange(num_classes),
        'frequency_features': frequency,
        'frequency_prompt_proto': F.normalize(
            centers[:, None, :] + 0.25 * noise(num_classes, 3, feature_dim), dim=-1,
        ),
        'frequency_description_candidates': F.normalize(
            centers[:, None, None, :] + 0.25 * noise(num_classes, 3, 4, feature_dim), dim=-1,
        ),
        'frequency_calibrated_proto': F.normalize(raw_frequency, dim=-1),
        'frequency_band_weights': torch.ones(num_classes, 3) / 3,
        'frequency_class_alpha': torch.full((num_classes,), 0.35),
    }


class ResidualReferenceIntegrationTest(unittest.TestCase):
    def test_image_forward_and_cached_reference_match_and_zero_head_is_exact(self):
        for use_frequency in (False, True):
            with self.subTest(frequency=use_frequency):
                cfg = tiny_config(frequency=use_frequency)
                model = make_model(cfg)
                state = synthetic_state()
                features = state['images_features'][::5]
                frequency = state['frequency_features'][::5]
                scorer = make_reference_scorer(model, cfg, state, 2)
                reference = scorer(features, frequency if use_frequency else None)
                with patch.object(model, 'extract_frequency_img_feature', return_value=frequency):
                    actual = model.forward_ours(
                        features, 6, 2, state['image_proto'], state['cov_image'],
                        state['description_proto'], state['description_features'],
                        state['description_targets'], state['text_features'], cfg.DATASET.BETA,
                        state['frequency_calibrated_proto'] if use_frequency else None,
                        state['frequency_band_weights'] if use_frequency else None,
                        state['frequency_class_alpha'] if use_frequency else None,
                    )
                self.assertTrue(torch.equal(actual, reference))
                self.assertIs(model.apply_incremental_residual(features, reference), reference)
                model.residual_head = LowRankResidualHead(8, 10, 2)
                model.residual_head.mark_seen(torch.arange(6))
                self.assertIs(model.apply_incremental_residual(features, reference), reference)
                self.assertTrue(all(not p.requires_grad for p in model.clip_model.parameters()))

    def test_query_values_do_not_change_support_or_anchor_reference_with_frequency(self):
        cfg = tiny_config(frequency=True)
        model = make_model(cfg)
        state = synthetic_state()
        features, labels = state['images_features'], state['images_targets']
        frequency = state['frequency_features']
        selected = torch.tensor([4, 1, 5, 2])  # Exercise nontrivial global-to-local remapping.
        support, query = [], []
        for local, class_id in enumerate(selected):
            candidates = torch.where(labels == class_id)[0]
            support.extend(candidates[:4 if local < 2 else 2].tolist())
            query.extend(candidates[-2:].tolist())
        support, query = torch.tensor(support), torch.tensor(query)
        first = make_episode_reference(model, cfg, features, labels, state, frequency)(
            support, query, selected, 2,
        )
        changed_features = features.clone()
        changed_frequency = frequency.clone()
        changed_features[query] *= -1
        changed_frequency[query] = changed_frequency[query].flip(-1)
        second = make_episode_reference(
            model, cfg, changed_features, labels, state, changed_frequency,
        )(support, query, selected, 2)
        for key in ('support_scores', 'anchor_features', 'anchor_scores', 'anchor_labels'):
            self.assertTrue(torch.equal(first[key], second[key]), key)
        self.assertFalse(torch.allclose(first['query_scores'], second['query_scores']))
        self.assertEqual(first['support_scores'].shape, (12, 4))
        self.assertEqual(first['anchor_scores'].shape, (2, 4))
        self.assertTrue(torch.isfinite(first['query_scores']).all())

    def test_meta_initialization_uses_disjoint_validation_classes_and_freezes_reference(self):
        cfg = tiny_config(frequency=True)
        model = make_model(cfg)
        state = synthetic_state()
        snapshot = {key: value.clone() for key, value in state.items() if torch.is_tensor(value)}
        clip_snapshot = {key: value.clone() for key, value in model.clip_model.state_dict().items()}
        report = initialize_residual(model, cfg, state)
        train_ids, validation_ids = set(report['training_class_ids']), set(report['validation_class_ids'])
        self.assertFalse(train_ids.intersection(validation_ids))
        self.assertEqual(train_ids.union(validation_ids), set(range(6)))
        self.assertEqual(set(report['validation']['class_ids']), validation_ids)
        self.assertEqual(report['validation']['query_count'], 4)
        self.assertEqual(report['meta']['episodes'], 2)
        self.assertGreater(report['meta']['dictionary_change'], 0.0)
        self.assertTrue(torch.isfinite(model.residual_head.dictionary).all())
        self.assertTrue(model.residual_head.seen_mask[:6].all())
        self.assertFalse(model.residual_head.seen_mask[6:].any())
        self.assertEqual(int(model.residual_head.codes.count_nonzero()), 0)
        for key, value in snapshot.items():
            self.assertTrue(torch.equal(state[key], value), key)
        for key, value in clip_snapshot.items():
            self.assertTrue(torch.equal(model.clip_model.state_dict()[key], value), key)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))


class FeatureOnlyDatasetManager:
    def __init__(self, cfg, state):
        self.cfg = cfg
        self.state = state
        self.support_seed = cfg.SEED
        self.template = ['a photo of a {}.']
        self.class_names = [f'class_{c}' for c in range(10)]
        self.class_index_in_task = [np.arange(6), np.arange(6, 8), np.arange(8, 10)]
        self.num_tasks = 3

    def save_support_manifest(self, path):
        Path(path).write_text(json.dumps({'seed': self.support_seed, 'synthetic': True}))

    def get_dataloader(self, task_id, source, mode, accumulate_past=False):
        if accumulate_past:
            raise AssertionError('The integration runner must not revisit historical images')
        classes = (self.class_index_in_task[task_id] if source == 'train'
                   else np.concatenate(self.class_index_in_task[:task_id + 1]))
        selected = []
        for class_id in classes:
            indices = torch.where(self.state['images_targets'] == int(class_id))[0]
            if source == 'train':
                selected.extend(indices[:8 if task_id == 0 else 2].tolist())
            else:
                selected.extend(indices[-3:].tolist())
        selected = torch.tensor(selected)
        return [{
            'image': self.state['images_features'][selected],
            'label': self.state['images_targets'][selected],
        }]


class ResidualRunnerIntegrationTest(unittest.TestCase):
    def test_three_sessions_release_samples_freeze_old_codes_and_write_artifacts(self):
        from engine.engine import Runner

        for gain in (0.0, 1.0):
            with self.subTest(gain=gain), temporary_artifact_directory() as directory:
                cfg = tiny_config(dictionary='random', output_dir=directory, gain=gain)
                state = synthetic_state(num_classes=10)
                manager = FeatureOnlyDatasetManager(cfg, state)
                model = make_model(cfg)

                def text_features(class_names, template, cls_begin_index):
                    del template
                    count = len(class_names)
                    ids = torch.arange(cls_begin_index, cls_begin_index + count)
                    return state['text_features'][ids], ids

                def description_features(class_names, gpt_path, cls_begin_index):
                    del gpt_path
                    ids = torch.arange(cls_begin_index, cls_begin_index + len(class_names))
                    mask = torch.isin(state['description_targets'], ids)
                    return (state['description_features'][mask], state['description_targets'][mask],
                            state['description_proto'][ids], None)

                with patch('engine.engine.DatasetManager', return_value=manager), \
                        patch('engine.engine.BiMC', return_value=model):
                    runner = Runner(cfg)
                snapshots = []
                save_checkpoint = runner._save_checkpoint

                def capture_checkpoint(task_id, states):
                    for item in states:
                        for key in ('images_features', 'images_targets', 'frequency_features',
                                    'frequency_description_candidates'):
                            self.assertNotIn(key, item)
                    snapshots.append((model.residual_head.codes.clone(),
                                      model.residual_head.dictionary.clone()))
                    save_checkpoint(task_id, states)

                with patch.object(model, 'inference_text_feature', side_effect=text_features), \
                        patch.object(model, 'inference_all_description_feature', side_effect=description_features), \
                        patch.object(runner, '_save_checkpoint', side_effect=capture_checkpoint), \
                        contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(io.StringIO()):
                    sessions = runner.run()
                self.assertEqual(len(sessions), 3)
                self.assertEqual(len(snapshots), 3)
                self.assertTrue(torch.equal(snapshots[0][0][:6], snapshots[1][0][:6]))
                self.assertTrue(torch.equal(snapshots[1][0][:8], snapshots[2][0][:8]))
                self.assertTrue(torch.equal(snapshots[0][1], snapshots[2][1]))
                if gain:
                    self.assertGreater(int(snapshots[1][0][6:8].count_nonzero()), 0)
                else:
                    for session in sessions:
                        self.assertEqual(session['accuracy'], session['reference_accuracy'])
                        self.assertEqual(session['prediction_change_rate'], 0.0)
                payload = json.loads((Path(directory) / 'metrics.json').read_text())
                self.assertEqual(payload['status'], 'completed')
                self.assertEqual(len(payload['sessions']), 3)
                self.assertEqual(payload['summary']['final_accuracy'], sessions[-1]['accuracy'])
                checkpoint = torch.load(Path(directory) / 'checkpoint.pt', weights_only=False)
                self.assertEqual(checkpoint['session'], 2)
                self.assertEqual(len(checkpoint['statistics']), 3)
                for item in checkpoint['statistics']:
                    self.assertNotIn('images_features', item)
                    self.assertNotIn('images_targets', item)
                self.assertTrue(torch.equal(checkpoint['residual_state']['codes'], model.residual_head.codes))
                self.assertTrue((Path(directory) / 'support.json').exists())
                self.assertTrue((Path(directory) / 'config.yaml').exists())
                self.assertTrue(all(not p.requires_grad for p in model.clip_model.parameters()))


if __name__ == '__main__':
    unittest.main()
