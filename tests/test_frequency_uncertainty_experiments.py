"""The matrix must preserve paired supports and keep calibration audit honest."""

import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch
from uuid import uuid4

from tools import run_frequency_uncertainty_experiments as tool


class UncertaintyExperimentTest(unittest.TestCase):
    def test_core_and_all_matrix_have_expected_independent_controls(self):
        core = dict(tool.variants_for_suite('core'))
        self.assertEqual(set(core), {'baseline', 'zero', 'original_shared', 'original_shrinkage',
                                     'frequency_shared', 'frequency_shrinkage'})
        self.assertFalse(core['baseline'][tool.U + 'ENABLED'])
        self.assertFalse(core['zero'][tool.U + 'AUTO_CALIBRATE'])
        self.assertEqual(core['zero'][tool.U + 'ALPHA'], 0.0)
        for name, values in core.items():
            self.assertFalse(values[tool.F + 'ENABLED'])
            self.assertFalse(values[tool.C + 'ENABLED'])
            self.assertFalse(values[tool.R + 'ENABLED'])
            if name.startswith('original'):
                self.assertEqual(values[tool.U + 'VIEW_CONTROL'], 'original')
        router = dict(tool.variants_for_suite('all'))['router']
        self.assertTrue(router[tool.F + 'ROUTER.ENABLED'])
        self.assertFalse(router[tool.U + 'ENABLED'])

    def test_same_seed_shares_support_and_calibration_seed(self):
        args = tool.parse_args(['--seeds', '1', '2', '--opts', 'DEVICE.GPU_ID', '0'])
        plan = tool.build_plan(args)
        self.assertEqual(len(plan), 12)
        for seed in (1, 2):
            group = [row for row in plan if row['seed'] == seed]
            self.assertEqual(len({row['support_manifest'] for row in group}), 1)
            self.assertEqual(len({row['output_dir'] for row in group}), 6)
            for row in group:
                command = row['command']
                values = command[command.index('--opts') + 1:]
                opts = dict(zip(values[::2], values[1::2]))
                self.assertEqual(opts['DATASET.SUPPORT_SEED'], str(seed))
                self.assertEqual(opts[tool.U + 'CALIBRATION_SEED'], str(seed + 10101))
                self.assertEqual(opts['DEVICE.GPU_ID'], "'0'")
        self.assertNotEqual(plan[0]['support_manifest'], plan[6]['support_manifest'])

    def test_dry_run_creates_nothing_and_does_not_execute(self):
        output = tool.REPO_ROOT / '.codex_tmp' / ('unused_uncertainty_plan_' + uuid4().hex)
        with patch.object(tool.common, 'execute_plan') as execute, contextlib.redirect_stdout(io.StringIO()):
            code = tool.main(['--output-root', str(output), '--seeds', '1', '--dry-run'])
        self.assertEqual(code, 0)
        execute.assert_not_called()
        self.assertFalse(output.exists())

    def test_managed_protocol_override_invalid_variants_and_seeds_fail(self):
        for argv in (['--opts', tool.U + 'BASE_ONLY', 'True'],
                     ['--opts', 'SEED', '2'], ['--seeds', '0'], ['--seeds', '1', '1'],
                     ['--variants', 'router'], ['--variants', 'baseline', 'baseline'],
                     ['--opts', 'DATASET.ROOT']):
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                tool.build_plan(tool.parse_args(argv))

    def test_manual_zero_protocol_is_not_compared_with_auto_episodes(self):
        def row(name, automatic, protocol):
            return {'variant': name, 'seed': 1, 'status': 'completed',
                    'output_dir': str(tool.REPO_ROOT / '.codex_tmp' / 'no_such_audit_output'),
                    'uncertainty_calibration': {
                        'data_source': 'base_training_only', 'seed': 10102, 'run_seed': 1,
                        'auto_calibrate': automatic, 'protocol_sha256': protocol,
                        'fit_class_ids': [0, 1], 'validation_class_ids': [2, 3] if automatic else [],
                    }}
        rows = [row('zero', False, 'manual'), row('original_shrinkage', True, 'paired'),
                row('frequency_shrinkage', True, 'paired')]
        report = tool.audit_pairing(rows)['1']
        self.assertTrue(report['calibration_protocol_match'])
        self.assertNotIn('zero', report['calibration_protocols'])
        self.assertEqual(report['calibration_errors'], {})
        rows[2]['uncertainty_calibration'].pop('protocol_sha256')
        report = tool.audit_pairing(rows)['1']
        self.assertFalse(report['calibration_protocol_match'])
        self.assertIn('frequency_shrinkage', report['calibration_errors'])


if __name__ == '__main__':
    unittest.main()
