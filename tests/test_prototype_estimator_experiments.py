import contextlib
import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from main import setup_cfg
from test_residual_integration import temporary_artifact_directory
from tools import run_prototype_estimator_experiments as tool


class PrototypeEstimatorExperimentTest(unittest.TestCase):
    def artifact(self, directory, name='boundary', **changes):
        path = Path(directory) / (name + '.npz')
        path.write_bytes(b'planning fixture; the runtime validates real NPZ arrays')
        d = {'format': 'support_prototype_estimator_v1', 'dataset': 'cub200',
             'data_source': 'base_training_only', 'cache_sha256': 'same-cache',
             'fitting_seed': 2, 'selection_protocol_sha256': 'same-class-query-split',
             'calibration': {'fused_temperature': 1., 'cov_temperature': 1.,
                             'description_temperature': 1., 'novel_bias': 0.},
             'model_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), **changes}
        path.with_suffix('.json').write_text(json.dumps(d))
        return path

    def test_paired_plan_has_distinct_full_zero_and_calibrated_controls(self):
        with temporary_artifact_directory() as directory:
            boundary = self.artifact(directory)
            center = self.artifact(directory, 'center')
            args = tool.parse_args(['--model', 'boundary='+str(boundary), '--model', 'center='+str(center), '--seeds', '1', '2'])
            plan = tool.build_plan(args)
            self.assertEqual(len(plan), 10)
            for seed in (1, 2):
                self.assertEqual(len({r['support_manifest'] for r in plan if r['seed'] == seed}), 1)
            for row in plan:
                opts = row['command'][row['command'].index('--opts')+1:]
                cfg = setup_cfg(args.data_cfg, args.train_cfg, opts)
                p = cfg.TRAINER.BiMC.PROTOTYPE_ESTIMATOR
                self.assertFalse(cfg.TRAINER.BiMC.SAFD_METRIC.ENABLED)
                if row['variant'] == 'baseline':
                    self.assertFalse(p.ENABLED)
                elif row['variant'] == 'zero_full':
                    self.assertEqual(p.BLEND, 0.)
                    self.assertFalse(p.CALIBRATION)
                elif row['variant'] == 'calibrated':
                    self.assertEqual(p.BLEND, 0.)
                    self.assertTrue(p.CALIBRATION)

    def test_artifact_and_calibration_mismatch_fail_before_running(self):
        with temporary_artifact_directory() as directory:
            a = self.artifact(directory)
            b = self.artifact(directory, 'center', cache_sha256='different')
            with self.assertRaisesRegex(ValueError, 'reference inputs'):
                tool.build_plan(tool.parse_args(['--model', 'boundary='+str(a), '--model', 'center='+str(b)]))
            b = self.artifact(directory, 'center', fitting_seed=3)
            with self.assertRaisesRegex(ValueError, 'fitting_seed'):
                tool.build_plan(tool.parse_args(['--model', 'boundary='+str(a), '--model', 'center='+str(b)]))
            a.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                tool.build_plan(tool.parse_args(['--model', 'boundary='+str(a)]))

    def test_default_is_dry_run_and_managed_model_flags_are_protected(self):
        with temporary_artifact_directory() as directory:
            path = self.artifact(directory)
            out = Path(directory) / 'uncreated'
            with contextlib.redirect_stdout(io.StringIO()), patch.object(tool.common, 'execute_plan') as execute:
                self.assertEqual(tool.main(['--model', 'boundary='+str(path), '--seeds', '1', '--output-root', str(out)]), 0)
            execute.assert_not_called()
            self.assertFalse(out.exists())
            with self.assertRaisesRegex(ValueError, 'matrix-managed'):
                tool.build_plan(tool.parse_args(['--model', 'boundary='+str(path), '--opts', tool.P+'BLEND', '1.']))


if __name__ == '__main__':
    unittest.main()
