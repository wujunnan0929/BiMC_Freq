"""Completion plans must preserve paired data and remain safe to inspect offline."""

import builtins
import contextlib
import io
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

from tools import run_prototype_completion_experiments as tool


def arguments(*extra):
    # Planning tests do not depend on the concurrently authored new YAML.
    return tool.parse_args(['--train-cfg', str(tool.ROOT/'configs/trainers/bimc.yaml'), *extra])


def options(row):
    command = row['command']
    values = command[command.index('--opts')+1:]
    return dict(zip(values[::2], values[1::2]))


@contextlib.contextmanager
def artifact_directory():
    # Python 3.13 tempfile's private Windows ACL can prevent child executables
    # and this test interpreter from reopening files. Use a normal workspace dir.
    parent = (tool.ROOT/'.codex_tmp').resolve()
    directory = parent/('completion_planner_'+uuid4().hex)
    directory.mkdir(parents=True, exist_ok=False)
    try:
        yield directory
    finally:
        resolved = directory.resolve()
        if resolved.parent != parent or not resolved.name.startswith('completion_planner_'):
            raise RuntimeError('Refusing cleanup outside the completion test directory.')
        shutil.rmtree(resolved)


class CompletionExperimentTest(unittest.TestCase):
    def test_controls_isolate_views_semantics_and_cross_view_covariance(self):
        matrix = dict(tool.variants_for_suite())
        self.assertEqual(set(matrix), {'baseline', 'zero', 'original', 'frequency',
            'ordinary_text', 'permuted_text', 'augmentation', 'repeat', 'block'})
        self.assertFalse(matrix['baseline'][tool.P+'ENABLED'])
        self.assertFalse(matrix['zero'][tool.P+'AUTO_CALIBRATE'])
        self.assertEqual(matrix['zero'][tool.P+'BLEND'], 0.)
        self.assertEqual(matrix['original'][tool.P+'VIEW_CONTROL'], 'original')
        self.assertEqual(matrix['augmentation'][tool.P+'VIEW_CONTROL'], 'augmentation')
        self.assertEqual(matrix['repeat'][tool.P+'VIEW_CONTROL'], 'repeat')
        self.assertEqual(matrix['ordinary_text'][tool.P+'SEMANTIC_SOURCE'], 'ordinary')
        self.assertEqual(matrix['permuted_text'][tool.P+'SEMANTIC_PERMUTATION'], [2, 0, 1])
        self.assertFalse(matrix['block'][tool.P+'CROSS_VIEW'])
        self.assertEqual(matrix['ordinary_text'][tool.P+'CANDIDATES_PER_BAND'], 1)
        self.assertEqual(dict(tool.variants_for_suite(semantic_source='explicit'))[
            'ordinary_text'][tool.P+'CANDIDATES_PER_BAND'], 5)
        for row in matrix.values():
            self.assertEqual(row[tool.F+'BACKEND'], 'fft')
            for key in ('FREQUENCY.ENABLED', 'FREQUENCY.ROUTER.ENABLED', 'GUIDANCE.ENABLED',
                        'CONSENSUS.ENABLED', 'UNCERTAINTY.ENABLED', 'DUAL_FREQUENCY.ENABLED',
                        'RESIDUAL.ENABLED', 'SAFD_METRIC.ENABLED', 'PROTOTYPE_ESTIMATOR.ENABLED',
                        'COMPETITIVE_FUSION.ENABLED'):
                self.assertFalse(row[tool.B+key])

    def test_seed_pairing_calibration_identity_and_gpu_argument_quoting(self):
        plan = tool.build_plan(arguments('--seeds', '1', '2', '--opts', 'DEVICE.GPU_ID', '0'))
        self.assertEqual(len(plan), 18)
        for seed in (1, 2):
            group = [row for row in plan if row['seed'] == seed]
            self.assertEqual(len({row['support_manifest'] for row in group}), 1)
            self.assertEqual(len({row['fingerprint'] for row in group}), 9)
            for row in group:
                opts = options(row)
                self.assertEqual(opts[tool.P+'CALIBRATION_SEED'], str(seed+10101))
                self.assertEqual(opts['DATASET.SUPPORT_SEED'], str(seed))
                self.assertEqual(opts['DEVICE.GPU_ID'], "'0'")
        self.assertNotEqual(plan[0]['support_manifest'], plan[9]['support_manifest'])

    def test_dry_plan_writes_nothing_and_imports_no_numerical_libraries(self):
        original_import = builtins.__import__

        def restricted_import(name, *args, **kwargs):
            if name.split('.')[0] in ('torch', 'numpy', 'torchvision'):
                raise AssertionError('Planning must not import ' + name)
            return original_import(name, *args, **kwargs)

        with artifact_directory() as directory:
            output = Path(directory)/'not_created'
            args = ['--train-cfg', str(tool.ROOT/'configs/trainers/bimc.yaml'),
                    '--output-root', str(output), '--seeds', '1', '--variants', 'baseline', 'frequency']
            with patch.object(tool.common, 'execute_plan') as execute, \
                    patch('builtins.__import__', side_effect=restricted_import), \
                    contextlib.redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(tool.main(args), 0)
            self.assertIn('Dry run: 2 evaluations', stdout.getvalue())
            execute.assert_not_called()
            self.assertFalse(output.exists())

    def test_rejects_unpaired_or_ambiguous_managed_overrides(self):
        invalid = (['--opts', tool.P+'BASE_ONLY', 'True'],
                   ['--opts', tool.P+'SEMANTIC_SOURCE', 'ordinary'],
                   ['--opts', tool.P+'CALIBRATION_SEED', '22'],
                   ['--opts', 'SEED', '2'], ['--seeds', '0'], ['--seeds', '1', '1'],
                   ['--variants', 'unknown'], ['--variants', 'zero', 'zero'],
                   ['--opts', 'DEVICE.GPU_ID'],
                   ['--opts', 'DATALOADER.NUM_WORKERS', '0', 'DATALOADER.NUM_WORKERS', '1'])
        for argv in invalid:
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                tool.build_plan(arguments(*argv))

    def test_explicit_semantics_are_fingerprinted_and_ordinary_control_stays_ordinary(self):
        with artifact_directory() as directory:
            path = Path(directory)/'descriptions.json'
            path.write_text('{}', encoding='utf-8')  # Schema is checked by runtime, not the planner.
            args = arguments('--semantic-source', 'explicit', '--description-path', str(path), '--seeds', '1')
            plan = tool.build_plan(args)
            by_name = {row['variant']: row for row in plan}
            self.assertEqual(options(by_name['frequency'])[tool.P+'SEMANTIC_SOURCE'], 'explicit')
            self.assertEqual(options(by_name['ordinary_text'])[tool.P+'SEMANTIC_SOURCE'], 'ordinary')
            first = by_name['frequency']['fingerprint']
            path.write_text('{"changed": true}', encoding='utf-8')
            second = next(r for r in tool.build_plan(args) if r['variant'] == 'frequency')['fingerprint']
            self.assertNotEqual(first, second)
        with self.assertRaisesRegex(ValueError, 'requires --description-path'):
            tool.build_plan(arguments('--semantic-source', 'explicit'))

    def test_manual_zero_does_not_falsely_fail_automatic_protocol_pairing(self):
        def row(name, automatic, protocol):
            details = {'source': 'base_training_only', 'seed': 10102,
                'pseudo_base_class_ids': [0, 1], 'pseudo_validation_class_ids': [2],
                'sample_ids': list(range(8)), 'fitting_indices': [0, 1, 2, 3],
                'reserved_pseudo_base_query_indices': {'0': [4], '1': [5]},
                'sequences': [{'class_ids': [0, 1, 2], 'support_indices': [[0, 1], [2, 3], [6]],
                               'query_indices': [[4], [5], [7]]}]}
            checksum = tool.common.sha256_json(details)
            details['sha256'] = checksum
            return {'variant': name, 'seed': 1, 'status': 'completed',
                'output_dir': str(tool.ROOT/'.codex_tmp'/'no_completion_audit_fixture'),
                'completion_calibration': {'data_source': 'base_training_only',
                    'seed': 10102, 'run_seed': 1, 'auto_calibrate': automatic,
                    'protocol_sha256': checksum if automatic else protocol,
                    'protocol': details if automatic else None, 'fit_class_ids': [0, 1],
                    'validation_class_ids': [2] if automatic else []}}
        rows = [row('zero', False, None), row('original', True, 'paired'),
                row('frequency', True, 'paired')]
        report = tool.audit_pairing(rows)['1']
        self.assertTrue(report['calibration_protocol_match'])
        self.assertEqual(report['calibration_errors'], {})
        rows[2]['completion_calibration']['validation_class_ids'] = [1, 2]
        report = tool.audit_pairing(rows)['1']
        self.assertIn('frequency', report['calibration_errors'])
        rows[2]['completion_calibration']['validation_class_ids'] = [2]
        rows[2]['completion_calibration']['protocol']['sequences'][0]['query_indices'][0] = [0]
        rows[2]['query_auxiliary_encodings'] = 1
        errors = tool.audit_pairing(rows)['1']['calibration_errors']['frequency']
        self.assertTrue(any('checksum' in error for error in errors))
        self.assertTrue(any('support/query' in error for error in errors))
        self.assertTrue(any('Query auxiliary' in error for error in errors))


if __name__ == '__main__':
    unittest.main()
