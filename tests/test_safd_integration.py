import unittest
from unittest.mock import patch

import torch
from torch import nn

from main import setup_cfg
from models.bimc import BiMC
from models.frequency import RadialFrequencyDecomposer, build_frequency_decomposer
from models.safd import SAFDFrequencyDecomposer
from tools import run_safd_experiments as launcher


class TinyImageEncoder(nn.Module):
    def encode_image(self, images):
        return torch.cat((images.mean((-2, -1)), images.std((-2, -1))), -1)


class SAFDIntegrationTest(unittest.TestCase):
    def config(self, opts=None):
        cfg = setup_cfg('configs/datasets/cub200.yaml', 'configs/trainers/bimc_safd.yaml', opts)
        cfg.defrost()
        cfg.TRAINER.BiMC.FREQUENCY.SAFD.LOW_TERMS = 2
        cfg.TRAINER.BiMC.FREQUENCY.SAFD.MIDDLE_TERMS = 5
        cfg.freeze()
        return cfg

    def test_factory_routes_old_configs_and_rejects_unknown_backend(self):
        legacy = setup_cfg('configs/datasets/cub200.yaml', 'configs/trainers/bimc_frequency_v2.yaml')
        self.assertIsInstance(build_frequency_decomposer(legacy.TRAINER.BiMC.FREQUENCY), RadialFrequencyDecomposer)
        self.assertIsInstance(build_frequency_decomposer(self.config().TRAINER.BiMC.FREQUENCY), SAFDFrequencyDecomposer)
        router = setup_cfg('configs/datasets/cub200.yaml', 'configs/trainers/bimc_safd_router.yaml')
        self.assertIsInstance(build_frequency_decomposer(router.TRAINER.BiMC.FREQUENCY), SAFDFrequencyDecomposer)
        self.assertTrue(router.TRAINER.BiMC.FREQUENCY.ROUTER.ENABLED)
        self.assertFalse(router.TRAINER.BiMC.UNCERTAINTY.ENABLED)
        with self.assertRaises(ValueError):
            build_frequency_decomposer(self.config(['TRAINER.BiMC.FREQUENCY.BACKEND', 'typo']).TRAINER.BiMC.FREQUENCY)

    def test_real_bimc_extraction_uses_safd_for_all_frequency_consumers(self):
        for path in ('gda', 'fusion', 'consensus', 'router'):
            cfg = self.config()
            cfg.defrost()
            cfg.TRAINER.BiMC.UNCERTAINTY.ENABLED = path == 'gda'
            cfg.TRAINER.BiMC.FREQUENCY.ENABLED = path in ('fusion', 'router')
            cfg.TRAINER.BiMC.CONSENSUS.ENABLED = path == 'consensus'
            cfg.TRAINER.BiMC.FREQUENCY.ROUTER.ENABLED = path == 'router'
            cfg.freeze()
            with self.subTest(path=path), patch('models.bimc.load_clip_to_cpu', return_value=TinyImageEncoder()):
                model = BiMC(cfg, ['a photo of a {}.'], 'cpu')
                self.assertIsInstance(model.frequency_decomposer, SAFDFrequencyDecomposer)
                images = torch.randn(2, 3, 19, 20)
                features = model.extract_frequency_img_feature(images)
                self.assertEqual(features.shape, (2, 3, 6))
                self.assertTrue(torch.isfinite(features).all())

    def test_matrix_pairs_backends_and_supports_and_all_commands_merge(self):
        plan = launcher.build_plan(launcher.parse_args(['--suite', 'all', '--seeds', '1']))
        self.assertEqual(len(plan), 8)
        self.assertEqual(len({row['support_manifest'] for row in plan}), 1)
        self.assertEqual(len({row['fingerprint'] for row in plan}), 8)
        configs = {}
        for row in plan:
            command = row['command']
            configs[row['variant']] = self.config(command[command.index('--opts') + 1:])
        self.assertEqual(configs['fft_gda'].TRAINER.BiMC.FREQUENCY.BACKEND, 'fft')
        self.assertEqual(configs['safd_gda'].TRAINER.BiMC.FREQUENCY.BACKEND, 'safd')
        for name in ('fft_gda', 'safd_gda'):
            self.assertTrue(configs[name].TRAINER.BiMC.UNCERTAINTY.AUTO_CALIBRATE)
        self.assertFalse(configs['zero'].TRAINER.BiMC.UNCERTAINTY.AUTO_CALIBRATE)
        self.assertEqual(configs['zero'].TRAINER.BiMC.UNCERTAINTY.ALPHA, 0.0)

    def test_matrix_rejects_backend_overrides(self):
        with self.assertRaises(ValueError):
            launcher.build_plan(launcher.parse_args(['--opts', 'TRAINER.BiMC.FREQUENCY.BACKEND', 'fft']))


if __name__ == '__main__':
    unittest.main()
