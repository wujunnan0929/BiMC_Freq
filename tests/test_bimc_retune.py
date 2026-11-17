"""Manual parameter precedence, baseline isolation and side-effect-free preview."""

import builtins
import contextlib
import io
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

from main import main, setup_cfg
from utils.bimc_retune import check_output_directory, prepare_retune_config


ROOT = Path(__file__).resolve().parents[1]
TRAINER = str(ROOT / 'configs/trainers/bimc_ensemble.yaml')


def config(dataset='cub200', opts=None, retune=True, trainer=TRAINER):
    return setup_cfg(
        str(ROOT / f'configs/datasets/{dataset}.yaml'), trainer, opts,
        retune_cfg_file=str(ROOT / f'configs/retune/{dataset}.yaml') if retune else None,
    )


def arguments(*extra):
    return ['--data_cfg', str(ROOT / 'configs/datasets/cub200.yaml'),
            '--train_cfg', TRAINER,
            '--retune_cfg', str(ROOT / 'configs/retune/cub200.yaml'), *extra]


class ManualRetuneTest(unittest.TestCase):
    def test_profiles_start_at_published_parameters_for_each_dataset(self):
        for dataset in ('cub200', 'cifar100', 'miniimagenet'):
            with self.subTest(dataset=dataset):
                baseline, manual = config(dataset, retune=False), config(dataset)
                self.assertEqual(baseline.DATASET, manual.DATASET)
                self.assertEqual(baseline.TRAINER, manual.TRAINER)
                self.assertEqual(baseline.SEED, manual.SEED)
                self.assertTrue(manual.is_frozen())
                self.assertTrue(manual.OUTPUT_DIR)

    def test_profile_overrides_trainer_and_command_line_overrides_profile(self):
        # Ordinary BiMC trainer says False, but the manual profile chooses True.
        plain = str(ROOT / 'configs/trainers/bimc.yaml')
        self.assertTrue(config(trainer=plain).TRAINER.BiMC.USING_ENSEMBLE)
        c = config(trainer=plain, opts=[
            'DATASET.NUM_INC_SHOT', '1', 'DATASET.BETA', '0.6',
            'DATASET.ENSEMBLE_ALPHA', '0.4', 'TRAINER.BiMC.LAMBDA_T', '0.7',
            'TRAINER.BiMC.LAMBDA_I', '0.2', 'TRAINER.BiMC.TAU', '32',
            'TRAINER.BiMC.GAMMA_BASE', '2.0', 'TRAINER.BiMC.GAMMA_INC', '3.0',
            'TRAINER.BiMC.USING_ENSEMBLE', 'False',
            'SEED', '3', 'DATASET.SUPPORT_SEED', '11',
        ])
        self.assertEqual((c.DATASET.BETA, c.DATASET.ENSEMBLE_ALPHA), (0.6, 0.4))
        self.assertEqual((c.TRAINER.BiMC.LAMBDA_T, c.TRAINER.BiMC.LAMBDA_I,
                          c.TRAINER.BiMC.TAU), (0.7, 0.2, 32))
        self.assertEqual((c.SEED, c.DATASET.SUPPORT_SEED, c.DATASET.NUM_INC_SHOT), (3, 11, 1))
        self.assertFalse(c.TRAINER.BiMC.USING_ENSEMBLE)

    def test_config_hash_tracks_parameters_shots_seed_and_backbone(self):
        original = config().OUTPUT_DIR
        self.assertEqual(original, config().OUTPUT_DIR)
        for opts in (['DATASET.BETA', '0.6'], ['TRAINER.BiMC.TAU', '32'],
                     ['DATASET.NUM_INC_SHOT', '1'], ['SEED', '2'],
                     ['MODEL.BACKBONE.NAME', 'ViT-L/14']):
            self.assertNotEqual(original, config(opts=opts).OUTPUT_DIR)
        self.assertEqual(config(opts=['OUTPUT_DIR', 'my_run']).OUTPUT_DIR, 'my_run')

    def test_rejects_invalid_values_before_any_model_loading(self):
        invalid = [('DATASET.BETA', '1.1'), ('TRAINER.BiMC.LAMBDA_T', '-0.1'),
                   ('TRAINER.BiMC.LAMBDA_I', '1.2'), ('DATASET.ENSEMBLE_ALPHA', '-0.2'),
                   ('TRAINER.BiMC.TAU', '0'), ('TRAINER.BiMC.GAMMA_BASE', '-1.0'),
                   ('DATASET.NUM_INC_SHOT', '0'), ('SEED', '-1')]
        for key, value in invalid:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                config(opts=[key, value])
        c = config().clone()
        c.defrost()
        c.DATASET.BETA = float('nan')
        with self.assertRaisesRegex(ValueError, 'BETA'):
            prepare_retune_config(c)

    def test_rejects_dataset_profile_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'CUB200.*CIFAR100'):
            setup_cfg(str(ROOT / 'configs/datasets/cifar100.yaml'), TRAINER,
                      retune_cfg_file=str(ROOT / 'configs/retune/cub200.yaml'))

    def test_rejects_optional_adaptation_heads(self):
        for name in ('FREQUENCY', 'FREQUENCY.ROUTER', 'RESIDUAL', 'GUIDANCE',
                     'CONSENSUS', 'UNCERTAINTY', 'DUAL_FREQUENCY', 'SAFD_METRIC',
                     'PROTOTYPE_ESTIMATOR', 'PROTOTYPE_COMPLETION', 'COMPETITIVE_FUSION'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'optional branches'):
                config(opts=['TRAINER.BiMC.' + name + '.ENABLED', 'True'])

    def test_disabled_calibration_and_endpoint_weights_are_supported(self):
        c = config(opts=['TRAINER.BiMC.VISION_CALIBRATION', 'False',
                         'TRAINER.BiMC.TEXT_CALIBRATION', 'False',
                         'TRAINER.BiMC.LAMBDA_I', '0.0',
                         'TRAINER.BiMC.LAMBDA_T', '0.0', 'DATASET.BETA', '1.0'])
        self.assertFalse(c.TRAINER.BiMC.VISION_CALIBRATION)
        self.assertFalse(c.TRAINER.BiMC.TEXT_CALIBRATION)

    def test_rejects_leftover_flags_that_change_baseline_data_or_skip_evaluation(self):
        for flag in ('RESIDUAL.RESERVE_BASE_VALIDATION', 'RESIDUAL.BASE_ONLY',
                     'UNCERTAINTY.BASE_ONLY', 'DUAL_FREQUENCY.BASE_ONLY',
                     'GUIDANCE.BASE_ONLY', 'PROTOTYPE_COMPLETION.BASE_ONLY'):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, 'optional branches'):
                config(opts=['TRAINER.BiMC.' + flag, 'True'])

    def test_preview_does_not_import_torch_or_engine(self):
        real_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in ('torch', 'engine', 'models', 'datasets') or name == 'utils.util':
                raise AssertionError('Preview must not load ' + name)
            return real_import(name, *args, **kwargs)

        stream = io.StringIO()
        with patch('builtins.__import__', side_effect=guarded), contextlib.redirect_stdout(stream):
            main(arguments('--dry-run', '--opts', 'DATASET.BETA', '0.6'))
        self.assertIn('beta=0.6', stream.getvalue())
        self.assertIn('Dry run:', stream.getvalue())

    def test_real_entry_dispatches_resolved_config_to_existing_runner(self):
        calls = []

        class Runner:
            def __init__(self, cfg):
                calls.append(cfg)

            def run(self):
                calls.append('run')

        no_op = lambda *args: None
        modules = {
            'models.frequency_guidance': SimpleNamespace(validate_settings=no_op),
            'utils.frequency_descriptions': SimpleNamespace(validate_frequency_inputs=no_op),
            'utils.util': SimpleNamespace(set_seed=no_op, set_gpu=no_op),
            'engine.engine': SimpleNamespace(Runner=Runner),
        }
        with patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()):
            main(arguments('--opts', 'DATASET.BETA', '0.9', 'TRAINER.BiMC.TAU', '8'))
        self.assertEqual(calls[0].DATASET.BETA, 0.9)
        self.assertEqual(calls[0].TRAINER.BiMC.TAU, 8)
        self.assertTrue(calls[0].is_frozen())
        self.assertEqual(calls[1], 'run')

    def test_nonempty_run_directory_is_not_overwritten(self):
        # Test-specific root: no cleanup ever targets the user's output folders.
        root = (ROOT / '.codex_tmp').resolve()
        root.mkdir(exist_ok=True)
        # Ordinary inherited ACLs avoid Windows tempfile's private-directory ACL.
        tmp = root / ('retune_test_' + uuid4().hex)
        tmp.mkdir()
        try:
            check_output_directory(tmp)
            marker = tmp / 'metrics.json'
            marker.write_text('original', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'nonempty'):
                check_output_directory(tmp)
            self.assertEqual(marker.read_text(encoding='utf-8'), 'original')
        finally:
            resolved = tmp.resolve()
            if resolved.parent != root or not resolved.name.startswith('retune_test_'):
                raise RuntimeError('Refusing test cleanup outside its root')
            shutil.rmtree(resolved)

    def test_existing_callers_keep_legacy_output_behavior(self):
        self.assertEqual(config(retune=False).OUTPUT_DIR, '')


if __name__ == '__main__':
    unittest.main()
