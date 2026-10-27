"""One-pass native ViT features and authenticated base-only text-mask caches."""

import contextlib
import io
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import torch
from torch.nn import functional as F

from models.clip.model import CLIP, VisionTransformer
from tools import cache_multilayer_base_features as tool
from utils.attribute_descriptions import build_attribute_manifest
from test_attribute_descriptions import sentences


@contextlib.contextmanager
def artifact_directory():
    parent = Path(__file__).resolve().parents[1] / '.codex_tmp' / 'multilayer_cache_tests'
    parent.mkdir(parents=True, exist_ok=True)
    folder = parent / uuid4().hex
    folder.mkdir()
    try:
        yield folder
    finally:
        if folder.resolve().parent != parent.resolve():
            raise RuntimeError('Unsafe test cleanup target')
        shutil.rmtree(folder)


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.labels = np.repeat(np.arange(3), 4)
        self.images = torch.randn(12, 3, 8, 8)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        return {'image': self.images[index], 'label': int(self.labels[index]), 'global_index': index + 30}


class Manager:
    def __init__(self, cfg=None):
        self.class_index_in_task = [np.arange(3), np.arange(3, 5)]
        self.class_names = ['base0', 'base1', 'base2', 'future3', 'future4']
        self.template = ['a photo of a {}.']
        self.test_transform = 'deterministic'
        self.dataset = Dataset()
        self.calls = []
    def get_dataset(self, task, source, mode):
        self.calls.append((task, source, mode))
        if task != 0 or source != 'train':
            raise AssertionError('Cache accessed benchmark or incremental images')
        return self.dataset
    def _support_manifest(self):
        return {'version': 2, 'dataset': 'cub200', 'train_size': 12, 'seed': 1,
                'train_targets_sha256': 'targets', 'train_samples_sha256': 'samples'}


class Encoder(torch.nn.Module):
    clip_weight_sha256 = 'synthetic-clip'
    def __init__(self):
        super().__init__()
        self.calls = 0
    def encode_image_layers(self, images, layers):
        self.calls += 1
        pooled = images.mean((-1, -2)) + images.new_tensor([1., 2., 3.])
        return {'final': pooled, **{'layer_'+str(i): torch.cat((pooled, pooled+i), -1) for i in layers}}
    def encode_text(self, tokens):
        return tokens.float() + tokens.new_tensor([1., 2., 3.])


class Model(torch.nn.Module):
    def __init__(self, cfg=None, template=None, device='cpu'):
        super().__init__()
        self.device = device
        self.clip_model = Encoder()
        self.image_encoding_counts = {'original': 0, 'auxiliary': 0}
    def inference_text_feature(self, names, template, start):
        return torch.eye(3), torch.arange(3)
    def inference_all_description_feature(self, names, path, start):
        values = torch.eye(3).repeat_interleave(2, 0)
        return values, torch.arange(3).repeat_interleave(2), torch.eye(3), None
    def inference_frequency_text_feature(self, names):
        return torch.eye(3)[:, None].expand(-1, 3, -1).clone()


def collect(limit=None):
    manager = Manager()
    model = torch.nn.Module()
    model.clip_model = Encoder()
    model.image_encoding_counts = {'original': 0, 'auxiliary': 0}
    with contextlib.redirect_stdout(io.StringIO()):
        arrays, counts, total = tool.collect_base_features(manager, model, (1, 2),
            batch_size=5, workers=0, limit_per_class=limit)
    return manager, model, arrays, counts, total


def complete_arrays(arrays):
    c, d = 3, arrays['final'].shape[-1]
    eye = np.eye(c, d, dtype=np.float32)
    arrays.update(text_features=eye.copy(), description_proto=eye.copy(),
                  description_features=np.repeat(eye, 2, axis=0),
                  description_targets=np.repeat(np.arange(c, dtype=np.int64), 2),
                  structured_text=np.repeat(eye[:, None], 3, axis=1))
    manifest = build_attribute_manifest({f'base{i}': sentences(f'base{i}') for i in range(c)},
                                        [f'base{i}' for i in range(c)])
    arrays.update(tool.encode_attribute_text(manifest,
        lambda texts: np.repeat(np.array([[1., 2., 3.]], dtype=np.float32), len(texts), axis=0), d))
    return arrays, manifest


class MultilayerCacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_selected_encoder_is_exact_final_and_native_intermediates_in_one_pass(self):
        torch.manual_seed(4)
        model = VisionTransformer(8, 4, 8, 3, 2, 5).eval()
        images = torch.randn(2, 3, 8, 8)
        with torch.no_grad():
            original = model(images)
            expected = []
            hook = model.transformer.resblocks[0].register_forward_hook(
                lambda module, inputs, output: expected.append(output[0].clone()))
            result = model.forward_selected(images, (1, 2))
            hook.remove()
        self.assertTrue(torch.equal(result['final'], original))
        self.assertTrue(torch.equal(result['layer_1'], expected[0]))
        self.assertEqual(result['layer_1'].shape, (2, 8))
        self.assertEqual(result['final'].shape, (2, 5))
        for bad in [(), (0,), (4,), (1, 1), (True,)]:
            with self.assertRaises(ValueError):
                model.forward_selected(images, bad)

    def test_only_base_train_requested_one_image_forward_and_native_dimensions(self):
        manager, model, arrays, counts, total = collect()
        self.assertEqual(manager.calls, [(0, 'train', 'test')])
        self.assertEqual(model.clip_model.calls, 3)
        self.assertEqual(total, 12)
        self.assertEqual(counts['original'], 12)
        self.assertEqual(counts['auxiliary'], 0)
        self.assertEqual(arrays['final'].shape, (12, 3))
        self.assertEqual(arrays['layer_1'].shape, (12, 6))
        np.testing.assert_array_equal(arrays['sample_id'], np.arange(30, 42))
        self.assertEqual(model.image_encoding_counts, {'original': 12, 'auxiliary': 0})

    def test_missing_attributes_are_zero_masked_and_never_encoded(self):
        manifest = build_attribute_manifest({'base': ['A base has a hooked bill.']}, ['base'])
        calls = []
        def encoder(texts):
            calls.extend(texts)
            return np.ones((len(texts), 4), dtype=np.float32)
        arrays = tool.encode_attribute_text(manifest, encoder, 4)
        self.assertEqual(len(calls), 2)  # One actual phrase in each evidence bank.
        self.assertEqual(arrays['attr_mask'].sum(), 1)
        self.assertEqual(arrays['ordinary_mask'].sum(), 1)
        self.assertTrue(np.all(arrays['attr_text'][~arrays['attr_mask']] == 0))
        self.assertTrue(np.all(arrays['ordinary_text'][~arrays['ordinary_mask']] == 0))

    def test_cache_roundtrip_authenticates_shapes_masks_and_support_rows(self):
        _, _, arrays, counts, _ = collect()
        arrays, manifest = complete_arrays(arrays)
        metadata = {'format': tool.FORMAT, 'data_source': 'base_training_only',
                    'complete_base_bank': True, 'selected_layers': [1, 2],
                    'base_class_ids': [0, 1, 2], 'sample_count': 12,
                    'encoding_counts': counts, 'attribute_manifest': manifest}
        support = {'sample_ids': arrays['sample_id'].tolist(), 'targets': arrays['labels'].tolist()}
        with artifact_directory() as root:
            path = root/'cache.npz'
            tool.write_cache(path, arrays, metadata, support)
            loaded, report = tool.load_cache(path)
            np.testing.assert_array_equal(loaded['layer_1'], arrays['layer_1'])
            self.assertEqual(report['attribute_manifest'], manifest)
            with self.assertRaises(FileExistsError):
                tool.write_cache(path, arrays, metadata, support)
            with path.open('ab') as handle:
                handle.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                tool.load_cache(path)

    def test_smoke_cache_is_explicitly_rejected_by_default(self):
        _, _, arrays, counts, _ = collect(limit=2)
        arrays, manifest = complete_arrays(arrays)
        metadata = {'format': tool.FORMAT, 'data_source': 'base_training_only',
                    'complete_base_bank': False, 'selected_layers': [1, 2],
                    'base_class_ids': [0, 1, 2], 'sample_count': 6,
                    'encoding_counts': counts, 'attribute_manifest': manifest}
        support = {'sample_ids': arrays['sample_id'].tolist(), 'targets': arrays['labels'].tolist()}
        with artifact_directory() as root:
            path = root/'smoke.npz'
            tool.write_cache(path, arrays, metadata, support)
            with self.assertRaisesRegex(ValueError, 'complete base bank'):
                tool.load_cache(path)
            self.assertEqual(len(tool.load_cache(path, require_complete=False)[0]['final']), 6)

    def test_dry_run_creates_nothing_and_disables_every_optional_branch(self):
        with artifact_directory() as root:
            path = root/'absent.npz'
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(tool.main(['--data-root', 'unused', '--output', str(path), '--dry-run']), 0)
            self.assertFalse(path.exists())
            cfg = tool.build_cfg(tool.parse_args(['--data-root', 'unused', '--output', str(path)]))
            b = cfg.TRAINER.BiMC
            self.assertFalse(any([b.PROTOTYPE_COMPLETION.ENABLED, b.GUIDANCE.ENABLED,
                                  b.FREQUENCY.ENABLED, b.COMPETITIVE_FUSION.ENABLED]))

    def test_main_exports_complete_base_only_cache_with_fixed_attribute_provenance(self):
        with artifact_directory() as root:
            path = root/'native.npz'
            descriptions = root/'descriptions.json'
            descriptions.write_text(json.dumps({f'base{i}': sentences(f'base{i}') for i in range(3)}))
            args = tool.parse_args(['--data-root', 'unused', '--output', str(path), '--device', 'cpu',
                                   '--workers', '0', '--layers', '1', '2'])
            cfg = tool.build_cfg(args)
            cfg.defrost()
            cfg.DATASET.GPT_PATH = str(descriptions)
            cfg.freeze()
            with patch('datasets.data_manager.DatasetManager', Manager), \
                    patch('models.bimc.BiMC', Model), patch.object(tool, 'build_cfg', return_value=cfg), \
                    patch('models.clip.clip.tokenize', side_effect=lambda texts: torch.ones(len(texts), 3)), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(tool.main(['--data-root', 'unused', '--output', str(path), '--device', 'cpu',
                                            '--workers', '0', '--layers', '1', '2']), 0)
            arrays, meta = tool.load_cache(path)
            self.assertEqual(meta['base_class_names'], ['base0', 'base1', 'base2'])
            self.assertTrue(meta['complete_base_bank'])
            self.assertFalse(meta['benchmark_evaluation_performed'])
            self.assertEqual(meta['encoding_counts']['benchmark_query'], 0)
            self.assertEqual(meta['attribute_manifest']['coverage']['complete_attribute_classes'], 3)
            self.assertEqual(arrays['attr_mask'].sum(), 24)
            self.assertNotIn('future3', str(meta['attribute_manifest']))


if __name__ == '__main__':
    unittest.main()
