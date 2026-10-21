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

from models.safd_interventions import NestedInterventionViews
from tools import cache_safd_base_features as tool


@contextlib.contextmanager
def artifact_directory():
    parent = Path(__file__).resolve().parents[1] / '.codex_tmp' / 'safd_cache_tests'
    parent.mkdir(parents=True, exist_ok=True)
    directory = parent / uuid4().hex
    directory.mkdir()
    try:
        yield directory
    finally:
        if directory.resolve().parent != parent.resolve():
            raise RuntimeError('Unsafe temporary cleanup target')
        shutil.rmtree(directory)


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.labels = np.repeat(np.arange(3), 4)
        self.images = torch.randn(12, 3, 12, 13)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        return {'image': self.images[index], 'label': int(self.labels[index]),
                'global_index': index + 20}


class Manager:
    def __init__(self, cfg=None):
        self.class_index_in_task = [np.arange(3), np.arange(3, 5)]
        self.class_names = ['base0', 'base1', 'base2', 'future3', 'future4']
        self.template = ['a {}']
        self.test_transform = 'test-only deterministic transform'
        self.calls = []
        self.dataset = BaseDataset()
    def get_dataset(self, task, source, mode):
        self.calls.append((task, source, mode))
        if task != 0 or source != 'train':
            raise AssertionError('Cache requested benchmark or incremental samples')
        return self.dataset
    def _support_manifest(self):
        return {'version': 2, 'dataset': 'cub200', 'train_size': 12, 'seed': 1,
                'train_targets_sha256': 'targets', 'train_samples_sha256': 'samples'}


class Clip(nn.Module):
    clip_weight_sha256 = 'test-clip'
    def encode_image(self, images):
        return images.mean((-2, -1)) + images.new_tensor([1., 2., 3.])


class Model(nn.Module):
    def __init__(self, cfg=None, template=None, device='cpu'):
        super().__init__()
        self.clip_model = Clip()
        self.image_encoding_counts = {'original': 0, 'auxiliary': 0}
        self.text_names = []
    def extract_img_feature(self, images):
        self.image_encoding_counts['original'] += len(images)
        return self.clip_model.encode_image(images)
    def inference_text_feature(self, names, template, start):
        self.text_names.extend(names)
        return torch.eye(3), torch.arange(3)
    def inference_all_description_feature(self, names, path, start):
        self.text_names.extend(names)
        features = torch.eye(3).repeat_interleave(2, dim=0)
        return features, torch.arange(3).repeat_interleave(2), torch.eye(3), None


class SAFDCacheTest(unittest.TestCase):
    def collect(self, source='repeat', limit=None):
        manager, model = Manager(), Model()
        with contextlib.redirect_stdout(io.StringIO()):
            arrays, counts, total = tool.collect_base_features(manager, model,
                NestedInterventionViews(source), batch_size=5, workers=0, limit_per_class=limit)
        return manager, model, arrays, counts, total

    def test_only_base_training_requested_with_stable_ids_and_counting(self):
        manager, model, arrays, counts, total = self.collect()
        self.assertEqual(manager.calls, [(0, 'train', 'test')])
        self.assertEqual(total, 12)
        np.testing.assert_array_equal(arrays['sample_id'], np.arange(20, 32))
        self.assertEqual(arrays['features'].shape, (12, 3, 3))
        self.assertEqual(counts, {'original': 12, 'auxiliary': 0, 'benchmark_query': 0})
        self.assertEqual(model.image_encoding_counts, {'original': 12, 'auxiliary': 0})
        np.testing.assert_array_equal(arrays['features'][:, 0], arrays['features'][:, 1])

    def test_explicit_limit_and_ordinary_control_encoding_budget(self):
        manager, model, arrays, counts, total = self.collect('augmentation', limit=2)
        self.assertEqual(total, 12)
        np.testing.assert_array_equal(arrays['sample_id'], [20, 21, 24, 25, 28, 29])
        self.assertEqual(counts, {'original': 6, 'auxiliary': 12, 'benchmark_query': 0})
        self.assertEqual(model.image_encoding_counts, {'original': 6, 'auxiliary': 12})

    def test_full_text_banks_encode_only_base_names(self):
        manager, model = Manager(), Model()
        class Config:
            class DATASET:
                GPT_PATH = 'unused-mock.json'
        arrays = tool.collect_base_text(manager, model, Config())
        self.assertEqual(set(arrays), {'text_features', 'description_features', 'description_targets', 'description_proto'})
        self.assertEqual(model.text_names, ['base0', 'base1', 'base2'] * 2)
        self.assertEqual(arrays['description_features'].shape, (6, 3))
        self.assertEqual(arrays['description_targets'].dtype, np.int64)

    def test_atomic_cache_roundtrip_checksums_and_overwrite_guard(self):
        _, _, arrays, _, _ = self.collect()
        meta = {'data_source': 'base_training_only', 'base_class_ids': [0, 1, 2], 'has_text_features': False}
        with artifact_directory() as root:
            output = root / 'base.npz'
            written = tool.write_cache(output, arrays, meta, {'only': 'base'})
            loaded, metadata = tool.load_cache(output)
            np.testing.assert_array_equal(loaded['features'], arrays['features'])
            self.assertEqual(written, metadata)
            before = tool.sha256(output)
            with self.assertRaises(FileExistsError):
                tool.write_cache(output, arrays, meta, {'only': 'base'})
            self.assertEqual(tool.sha256(output), before)
            with output.open('ab') as handle:
                handle.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                tool.load_cache(output)

    def test_metadata_modification_and_missing_certificate_rejected(self):
        _, _, arrays, _, _ = self.collect()
        meta = {'data_source': 'base_training_only', 'base_class_ids': [0, 1, 2], 'has_text_features': False}
        with artifact_directory() as root:
            output = root / 'base.npz'
            tool.write_cache(output, arrays, meta, {})
            sidecar = output.with_suffix('.json')
            changed = json.loads(sidecar.read_text())
            changed['data_source'] = 'benchmark_test'
            sidecar.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'metadata'):
                tool.load_cache(output)

    def test_main_saves_base_only_sidecars_and_text_provenance(self):
        with artifact_directory() as root:
            output = root / 'base.npz'
            with patch('datasets.data_manager.DatasetManager', Manager), patch('models.bimc.BiMC', Model), \
                    contextlib.redirect_stdout(io.StringIO()):
                status = tool.main(['--dataset', 'cub200', '--data-root', 'unused', '--source', 'repeat',
                                    '--device', 'cpu', '--workers', '0', '--output', str(output),
                                    '--limit-per-class', '2'])
            self.assertEqual(status, 0)
            arrays, meta = tool.load_cache(output)
            self.assertFalse(meta['complete_base_bank'])
            self.assertEqual(meta['purpose'], 'smoke_test_subset')
            self.assertEqual(meta['encoding_counts']['benchmark_query'], 0)
            self.assertTrue(meta['has_text_features'])
            self.assertTrue(meta['ordinary_description_sha256'])
            self.assertEqual(meta['base_class_names'], ['base0', 'base1', 'base2'])
            support = json.loads(output.with_suffix('.support.json').read_text())
            self.assertNotIn('future3', str(support))
            self.assertEqual(arrays['description_features'].shape, (6, 3))

    def test_dry_run_creates_nothing_and_real_configs_disable_optional_branches(self):
        with artifact_directory() as root:
            output = root / 'unused.npz'
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(tool.main(['--data-root', 'unused', '--output', str(output), '--dry-run']), 0)
            self.assertFalse(output.exists())
            for dataset in ('cub200', 'cifar100', 'miniimagenet'):
                args = tool.parse_args(['--dataset', dataset, '--data-root', 'unused', '--output', str(output)])
                cfg = tool.build_cfg(args)
                b = cfg.TRAINER.BiMC
                self.assertFalse(b.GUIDANCE.ENABLED or b.FREQUENCY.ENABLED or b.DUAL_FREQUENCY.ENABLED
                                 or b.CONSENSUS.ENABLED or b.UNCERTAINTY.ENABLED or b.RESIDUAL.ENABLED)


if __name__ == '__main__':
    unittest.main()
