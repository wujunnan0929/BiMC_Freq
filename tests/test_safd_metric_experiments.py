import contextlib
import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from test_residual_integration import temporary_artifact_directory
from tools import run_safd_metric_experiments as tool
from main import setup_cfg


class SAFDMetricExperimentTest(unittest.TestCase):
    def model_file(self, directory, source='safd'):
        path = Path(directory) / 'model.npz'
        path.write_bytes(b'planning fixture only; runtime loads real NPZ')
        path.with_suffix('.json').write_text(json.dumps({
            'dataset': 'cub200', 'data_source': 'base_training_only', 'source': source,
            'basis_mode': 'explained',
            'model_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}))
        return path

    def test_plans_share_support_and_zero_has_exact_gain(self):
        with temporary_artifact_directory() as directory:
            path = self.model_file(directory)
            args = tool.parse_args(['--model', 'safd='+str(path), '--seeds', '1', '2'])
            plan = tool.build_plan(args)
            self.assertEqual(len(plan), 6)
            for seed in (1, 2):
                self.assertEqual(len({r['support_manifest'] for r in plan if r['seed'] == seed}), 1)
            for run in plan:
                options = run['command'][run['command'].index('--opts')+1:]
                cfg = setup_cfg(args.data_cfg, args.train_cfg, options)
                self.assertTrue(cfg.TRAINER.BiMC.USING_ENSEMBLE)
                self.assertFalse(cfg.TRAINER.BiMC.GUIDANCE.ENABLED)
                self.assertFalse(cfg.TRAINER.BiMC.FREQUENCY.ENABLED)
                if run['variant'] == 'zero':
                    self.assertEqual(cfg.TRAINER.BiMC.SAFD_METRIC.GAIN, 0.)
                if run['variant'] == 'baseline':
                    self.assertFalse(cfg.TRAINER.BiMC.SAFD_METRIC.ENABLED)

    def test_default_plan_does_not_execute_or_create_output(self):
        with temporary_artifact_directory() as directory:
            path = self.model_file(directory)
            output = Path(directory) / 'unused'
            with contextlib.redirect_stdout(io.StringIO()), patch.object(tool.common, 'execute_plan') as execute:
                code = tool.main(['--model', 'safd='+str(path), '--output-root', str(output), '--seeds', '1'])
            self.assertEqual(code, 0)
            execute.assert_not_called()
            self.assertFalse(output.exists())

    def test_model_hash_source_and_matrix_overrides_are_checked(self):
        with temporary_artifact_directory() as directory:
            path = self.model_file(directory, source='fft')
            with self.assertRaisesRegex(ValueError, 'SAFD cache'):
                tool.build_plan(tool.parse_args(['--model', 'safd='+str(path)]))
            path = self.model_file(directory)
            sidecar = path.with_suffix('.json')
            metadata = json.loads(sidecar.read_text())
            metadata['basis_mode'] = 'original_within'
            sidecar.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'SAFD-response basis'):
                tool.build_plan(tool.parse_args(['--model', 'safd='+str(path)]))
            path = self.model_file(directory)
            with self.assertRaisesRegex(ValueError, 'matrix-managed'):
                tool.build_plan(tool.parse_args(['--model', 'safd='+str(path), '--opts', tool.M+'GAIN', '20']))
            path.write_bytes(b'changed artifact')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                tool.build_plan(tool.parse_args(['--model', 'safd='+str(path)]))


if __name__ == '__main__':
    unittest.main()
