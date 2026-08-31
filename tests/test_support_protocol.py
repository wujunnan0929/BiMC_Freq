"""FSCIL support identity must not depend on transforms or adaptation RNG use."""

import contextlib
import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from uuid import uuid4

import numpy as np

from datasets.data_manager import DatasetManager


@contextlib.contextmanager
def temporary_protocol_directory():
    # Use ordinary inherited permissions; Python 3.13 Windows tempfiles can
    # create 0700 directories that restricted test-runner tokens cannot access.
    parent = Path(__file__).resolve().parents[1] / '.codex_tmp' / 'support_tests'
    parent.mkdir(parents=True, exist_ok=True)
    directory = parent / uuid4().hex
    directory.mkdir()
    try:
        yield directory
    finally:
        resolved = directory.resolve()
        if resolved.parent != parent.resolve() or resolved.name != directory.name:
            raise RuntimeError('Refusing to clean a support-test directory outside its root')
        shutil.rmtree(resolved)


def make_manager(seed=7, support_seed=-1, manifest='', base_shot=4):
    manager = DatasetManager.__new__(DatasetManager)
    manager.cfg = SimpleNamespace(
        SEED=seed,
        DATASET=SimpleNamespace(SUPPORT_SEED=support_seed, SUPPORT_MANIFEST=str(manifest)),
    )
    manager.dataset_name = 'synthetic'
    manager.class_names = [f'class_{index}' for index in range(6)]
    manager.class_index_in_task = [np.arange(0, 2), np.arange(2, 4), np.arange(4, 6)]
    manager.num_tasks = 3
    manager.num_base_shot = base_shot
    manager.num_inc_shot = 5
    manager.train_targets = np.repeat(np.arange(6), 12)
    manager.train_data = np.arange(72, dtype=np.uint8)[:, None, None, None] * np.ones(
        (72, 2, 2, 3), dtype=np.uint8,
    )
    manager.test_targets = np.repeat(np.arange(6), 3)
    manager.test_data = manager.train_data[:18].copy()
    manager.train_transform = lambda image: np.asarray(image) + 1
    manager.test_transform = lambda image: np.asarray(image) + 2
    return manager


class SupportProtocolTest(unittest.TestCase):
    def test_train_and_statistics_views_use_exactly_same_five_shot(self):
        manager = make_manager()
        train = manager.get_dataset(1, 'train', 'train')
        np.random.random(5000)  # Simulate a base router/meta-learning RNG consumer.
        statistics = manager.get_dataset(1, 'train', 'test')
        again = manager.get_dataset(1, 'train', 'train')
        np.testing.assert_array_equal(train.sample_indices, statistics.sample_indices)
        np.testing.assert_array_equal(train.sample_indices, again.sample_indices)
        self.assertEqual(len(train), 10)
        np.testing.assert_array_equal(np.bincount(train.labels)[2:4], [5, 5])
        self.assertFalse(np.array_equal(train[0]['image'], statistics[0]['image']))

    def test_independent_rng_preserves_session_order_and_global_rng_state(self):
        np.random.seed(101)
        expected_global = np.random.RandomState(101).random_sample(3)
        manager = make_manager(seed=7)
        # Accessing a late session first must still produce the sequential protocol.
        manager.get_dataset(2, 'train', 'test')
        np.testing.assert_array_equal(np.random.random(3), expected_global)
        reference_rng = np.random.RandomState(7)
        for task_id, classes in enumerate(manager.class_index_in_task):
            expected = np.concatenate([
                reference_rng.choice(np.flatnonzero(manager.train_targets == class_id),
                                     size=4 if task_id == 0 else 5, replace=False)
                for class_id in classes
            ])
            actual = manager.get_dataset(task_id, 'train', 'test').sample_indices
            np.testing.assert_array_equal(actual, expected)

    def test_support_seed_separates_support_sampling_from_model_seed(self):
        first = make_manager(seed=1, support_seed=17)
        second = make_manager(seed=99, support_seed=17)
        np.testing.assert_array_equal(
            first.get_dataset(2, 'train', 'test').sample_indices,
            second.get_dataset(2, 'train', 'test').sample_indices,
        )

    def test_accumulation_reuses_every_sessions_original_shot_count(self):
        manager = make_manager(base_shot=-1)
        original = [manager.get_dataset(task, 'train', 'test').sample_indices for task in range(3)]
        accumulated = manager.get_dataset(2, 'train', 'train', accumulated_past=True)
        np.testing.assert_array_equal(accumulated.sample_indices, np.concatenate(original))
        self.assertEqual(len(accumulated), 24 + 10 + 10)
        self.assertEqual(len(np.unique(accumulated.sample_indices)), len(accumulated))

    def test_test_data_is_cumulative_and_never_few_shot_subsampled(self):
        dataset = make_manager().get_dataset(1, 'test', 'test')
        np.testing.assert_array_equal(dataset.sample_indices, np.arange(12))
        np.testing.assert_array_equal(np.bincount(dataset.labels), [3, 3, 3, 3])

    def test_local_index_is_preserved_and_global_sample_identity_is_available(self):
        dataset = make_manager().get_dataset(1, 'train', 'test')
        sample = dataset[1]
        self.assertEqual(sample['idx'], 1)
        self.assertEqual(sample['global_index'], dataset.sample_indices[1])
        self.assertEqual(sample['sample_id'], sample['global_index'])
        self.assertEqual(sample['task_id'], 1)
        self.assertFalse(dataset.sample_indices.flags.writeable)

    def test_manifest_auto_creation_and_round_trip_across_model_seeds(self):
        with temporary_protocol_directory() as directory:
            path = Path(directory) / 'nested' / 'support.json'
            original = make_manager(seed=7, manifest=path)
            expected = original.get_dataset(2, 'train', 'test').sample_indices.copy()
            self.assertTrue(path.exists())
            restored = make_manager(seed=999, manifest=path)
            np.testing.assert_array_equal(
                restored.get_dataset(2, 'train', 'test').sample_indices, expected,
            )
            self.assertEqual(restored.support_seed, 7)
            original.save_support_manifest(path)  # Identical save is safe and idempotent.
            self.assertEqual(json.loads(path.read_text(encoding='utf-8'))['seed'], 7)

    def test_manifest_explicit_support_seed_mismatch_is_rejected(self):
        with temporary_protocol_directory() as directory:
            path = Path(directory) / 'support.json'
            make_manager(seed=7).save_support_manifest(path)
            with self.assertRaisesRegex(ValueError, 'SUPPORT_SEED'):
                make_manager(support_seed=8, manifest=path).get_dataset(0, 'train', 'test')

    def test_existing_different_protocol_is_never_overwritten(self):
        with temporary_protocol_directory() as directory:
            path = Path(directory) / 'support.json'
            make_manager(seed=7).save_support_manifest(path)
            initial_bytes = path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'Refusing to overwrite'):
                make_manager(seed=8).save_support_manifest(path)
            self.assertEqual(path.read_bytes(), initial_bytes)

    def test_manifest_rejects_corrupt_or_incompatible_data(self):
        with temporary_protocol_directory() as directory:
            path = Path(directory) / 'support.json'
            make_manager().save_support_manifest(path)
            valid = json.loads(path.read_text(encoding='utf-8'))
            mutations = {
                'dataset': lambda m: m.update(dataset='another_dataset'),
                'size': lambda m: m.update(train_size=1),
                'target_order': lambda m: m.update(train_targets_sha256='wrong'),
                'class_groups': lambda m: m['class_groups'][1].reverse(),
                'shots': lambda m: m['shots'].update(incremental=6),
                'missing_session': lambda m: m['sessions'].pop(),
                'non_integer': lambda m: m['sessions'][1]['indices'].__setitem__(0, 25.5),
                'out_of_bounds': lambda m: m['sessions'][1]['indices'].__setitem__(0, 999),
                'duplicate': lambda m: m['sessions'][1]['indices'].__setitem__(
                    0, m['sessions'][1]['indices'][1]),
                'wrong_label': lambda m: m['sessions'][1]['targets'].__setitem__(0, 0),
                'missing_support': lambda m: (
                    m['sessions'][1]['indices'].pop(), m['sessions'][1]['targets'].pop()),
            }
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    corrupt = copy.deepcopy(valid)
                    mutate(corrupt)
                    path.write_text(json.dumps(corrupt), encoding='utf-8')
                    with self.assertRaisesRegex(ValueError, 'Invalid support manifest'):
                        make_manager(manifest=path).get_dataset(1, 'train', 'test')

    def test_manifest_rejects_changed_training_labels_even_with_equal_size(self):
        with temporary_protocol_directory() as directory:
            path = Path(directory) / 'support.json'
            make_manager().save_support_manifest(path)
            modified = make_manager(manifest=path)
            modified.train_targets[[0, 24]] = modified.train_targets[[24, 0]]
            with self.assertRaisesRegex(ValueError, 'train_targets_sha256'):
                modified.get_dataset(1, 'train', 'test')

    def test_manifest_rejects_reordered_same_class_samples_with_unchanged_labels(self):
        for representation in ('image_array', 'path_array'):
            with self.subTest(representation=representation), temporary_protocol_directory() as directory:
                path = directory / 'support.json'
                original = make_manager()
                if representation == 'path_array':
                    original.train_data = np.array([
                        f'dataset/class_{index // 12}/image_{index}.jpg'
                        for index in range(len(original.train_targets))
                    ])
                original.save_support_manifest(path)
                modified = make_manager(manifest=path)
                modified.train_data = original.train_data.copy()
                modified.train_data[[0, 1]] = modified.train_data[[1, 0]]
                np.testing.assert_array_equal(modified.train_targets, original.train_targets)
                with self.assertRaisesRegex(ValueError, 'train_samples_sha256'):
                    modified.get_dataset(0, 'train', 'test')

    def test_manifest_hash_is_independent_of_array_memory_layout(self):
        with temporary_protocol_directory() as directory:
            path = directory / 'support.json'
            original = make_manager()
            original.save_support_manifest(path)
            restored = make_manager(manifest=path)
            restored.train_data = np.asfortranarray(restored.train_data)
            self.assertFalse(restored.train_data.flags.c_contiguous)
            np.testing.assert_array_equal(
                restored.get_dataset(1, 'train', 'test').sample_indices,
                original.get_dataset(1, 'train', 'test').sample_indices,
            )

    def test_manifest_rejects_changed_image_shape_even_with_identical_bytes(self):
        with temporary_protocol_directory() as directory:
            path = directory / 'support.json'
            original = make_manager()
            original.save_support_manifest(path)
            modified = make_manager(manifest=path)
            modified.train_data = modified.train_data.reshape(72, 4, 3)
            self.assertEqual(modified.train_data.tobytes(), original.train_data.tobytes())
            with self.assertRaisesRegex(ValueError, 'train_samples_sha256'):
                modified.get_dataset(0, 'train', 'test')

    def test_legacy_manifest_without_sample_identity_check_is_explicitly_rejected(self):
        with temporary_protocol_directory() as directory:
            path = directory / 'support.json'
            make_manager().save_support_manifest(path)
            legacy = json.loads(path.read_text(encoding='utf-8'))
            legacy['version'] = 1
            legacy.pop('train_samples_sha256')
            path.write_text(json.dumps(legacy), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'version 2'):
                make_manager(manifest=path).get_dataset(0, 'train', 'test')

    def test_explicit_manifest_reload_rechecks_even_previously_cached_data(self):
        with temporary_protocol_directory() as directory:
            path = directory / 'support.json'
            manager = make_manager()
            manager.save_support_manifest(path)
            manager.train_data[[0, 1]] = manager.train_data[[1, 0]]
            with self.assertRaisesRegex(ValueError, 'train_samples_sha256'):
                manager.load_support_manifest(path)


if __name__ == '__main__':
    unittest.main()
