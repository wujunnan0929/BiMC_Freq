"""The GDA matrix preserves paired supports and immutable experiment protocols."""

import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch
from uuid import uuid4

from tools import run_frequency_discriminant_experiments as tool
from tools import run_frequency_uncertainty_experiments as legacy


class DiscriminantExperimentTest(unittest.TestCase):
    def args(self, argv=()):
        # The plan tests need readable configuration bytes, not a model runtime.
        return tool.parse_args(['--train-cfg', str(tool.REPO_ROOT /
                                'configs/trainers/bimc_frequency_uncertainty.yaml'), *argv])

    def options(self, row):
        command = row['command']
        values = command[command.index('--opts') + 1:]
        return dict(zip(values[::2], values[1::2]))

    def test_default_configuration_is_separate_and_execution_is_opt_in(self):
        args = tool.parse_args([])
        self.assertEqual(Path(args.train_cfg).name, 'bimc_frequency_discriminant.yaml')
        self.assertEqual(Path(args.output_root).name, 'frequency_discriminant_cub')
        self.assertEqual(args.seeds, [1, 2, 3])
        self.assertFalse(args.execute)

    def test_core_matrix_has_legacy_and_covariance_controls(self):
        core = dict(tool.variants_for_suite('core'))
        self.assertEqual(list(core), ['baseline', 'original_shared', 'frequency_shrinkage',
                                     'original_gda', 'frequency_gda_block', 'frequency_gda_joint'])
        self.assertFalse(core['baseline'][tool.U + 'ENABLED'])
        expected = {
            'original_shared': ('original', 'shared', True),
            'frequency_shrinkage': ('frequency', 'shrinkage', True),
            'original_gda': ('original', 'full_shared', False),
            'frequency_gda_block': ('joint', 'block_shared', False),
            'frequency_gda_joint': ('joint', 'full_shared', False),
        }
        for name, (view, covariance, uncertainty) in expected.items():
            with self.subTest(variant=name):
                values = core[name]
                self.assertEqual(values[tool.U + 'VIEW_CONTROL'], view)
                self.assertEqual(values[tool.U + 'COVARIANCE'], covariance)
                self.assertEqual(values[tool.U + 'MEAN_UNCERTAINTY'], uncertainty)
                self.assertTrue(values[tool.U + 'AUTO_CALIBRATE'])
        for values in core.values():
            self.assertEqual(values[tool.U + 'RIDGE'], 0.1)
            for name in (tool.F + 'ENABLED', tool.F + 'ROUTER.ENABLED',
                         tool.C + 'ENABLED', tool.R + 'ENABLED'):
                self.assertFalse(values[name])
        extra = dict(tool.variants_for_suite('all'))
        self.assertEqual(len(extra), 8)
        self.assertEqual(set(extra) - set(core), {'original_diagonal', 'original_repeat_gda'})
        self.assertFalse(extra['original_diagonal'][tool.U + 'MEAN_UNCERTAINTY'])
        self.assertEqual(extra['original_diagonal'][tool.U + 'COVARIANCE'], 'shared')
        self.assertEqual(extra['original_repeat_gda'][tool.U + 'VIEW_CONTROL'], 'repeat')
        self.assertEqual(extra['original_repeat_gda'][tool.U + 'COVARIANCE'], 'full_shared')
        self.assertFalse(extra['original_repeat_gda'][tool.U + 'MEAN_UNCERTAINTY'])

    def test_paired_support_and_calibration_seeds_and_old_support_reuse(self):
        support_root = tool.REPO_ROOT / 'outputs/frequency_uncertainty_cub/support'
        args = self.args(['--support-root', str(support_root), '--opts', 'DEVICE.GPU_ID', '0'])
        plan = tool.build_plan(args)
        self.assertEqual(len(plan), 18)
        legacy_plan = legacy.build_plan(args)
        for seed in args.seeds:
            group = [row for row in plan if row['seed'] == seed]
            self.assertEqual(len({row['support_manifest'] for row in group}), 1)
            self.assertEqual(len({row['output_dir'] for row in group}), 6)
            self.assertEqual(group[0]['support_manifest'], next(
                row['support_manifest'] for row in legacy_plan if row['seed'] == seed))
            for row in group:
                options = self.options(row)
                self.assertEqual(options['DATASET.SUPPORT_SEED'], str(seed))
                self.assertEqual(options[tool.U + 'CALIBRATION_SEED'], str(seed + 10101))
                self.assertEqual(options['DEVICE.GPU_ID'], "'0'")

    def test_legacy_matrix_and_launcher_identity_are_not_mutated(self):
        args = self.args(['--seeds', '1'])
        matrix_before = legacy.variants_for_suite('all')
        legacy_before = legacy.build_plan(args)
        plan = tool.build_plan(args)
        self.assertEqual(legacy.variants_for_suite('all'), matrix_before)
        self.assertEqual(legacy.build_plan(args), legacy_before)
        self.assertNotEqual(plan[0]['source_hash'], legacy_before[0]['source_hash'])
        with patch.object(legacy, 'build_plan', wraps=legacy.build_plan) as builder:
            tool.build_plan(args)
        self.assertEqual(builder.call_args.kwargs['launcher_path'], Path(tool.__file__))

    def test_protocol_overrides_invalid_variants_and_seeds_fail(self):
        bad = [
            ['--opts', tool.U + 'MEAN_UNCERTAINTY', 'True'],
            ['--opts', tool.U + 'RIDGE', '0.01'],
            ['--opts', tool.U + 'COVARIANCE', 'shared'],
            ['--opts', tool.U + 'BASE_ONLY', 'True'],
            ['--opts', 'SEED', '2'], ['--seeds', '0'], ['--seeds', '1', '1'],
            ['--variants', 'router'], ['--variants', 'baseline', 'baseline'],
            ['--opts', 'DATASET.ROOT'],
        ]
        for argv in bad:
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                tool.build_plan(self.args(argv))
        with self.assertRaises(ValueError):
            tool.variants_for_suite('unknown')

    def test_dry_run_does_not_create_outputs_or_launch_processes(self):
        output = tool.REPO_ROOT / '.codex_tmp' / ('unused_gda_plan_' + uuid4().hex)
        args = self.args(['--output-root', str(output), '--seeds', '1'])
        with patch.object(tool, 'parse_args', return_value=args), \
                patch.object(tool.common, 'execute_plan') as execute, \
                contextlib.redirect_stdout(io.StringIO()):
            code = tool.main([])
        self.assertEqual(code, 0)
        execute.assert_not_called()
        self.assertFalse(output.exists())

    def test_summary_distinguishes_completed_and_stale_and_does_not_execute(self):
        args = self.args(['--seeds', '1', '--variants', 'baseline', 'original_gda', '--summarize-only'])
        with patch.object(tool, 'parse_args', return_value=args), \
                patch.object(tool.common, 'completed_result', side_effect=[{'summary': {}}, None]), \
                patch.object(tool, 'save_summary') as save, \
                patch.object(tool.common, 'execute_plan') as execute, \
                contextlib.redirect_stdout(io.StringIO()):
            code = tool.main([])
        self.assertEqual(code, 0)
        execute.assert_not_called()
        rows = save.call_args.args[2]
        self.assertEqual([row['status'] for row in rows], ['skipped', 'missing_or_stale'])
        self.assertEqual(save.call_args.args[3], 2)

    def test_execute_refuses_unmanaged_nonempty_output_directory(self):
        args = self.args(['--seeds', '1', '--variants', 'original_gda', '--execute'])
        plan = tool.build_plan(args)
        with patch.object(tool, 'parse_args', return_value=args), \
                patch.object(tool, 'build_plan', return_value=plan), \
                patch.object(Path, 'exists', return_value=True), \
                patch.object(Path, 'iterdir', return_value=iter([Path('unrelated.txt')])), \
                patch.object(Path, 'is_file', return_value=False), \
                patch.object(tool.common, 'execute_plan') as execute, \
                contextlib.redirect_stderr(io.StringIO()):
            code = tool.main([])
        self.assertEqual(code, 2)
        execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
