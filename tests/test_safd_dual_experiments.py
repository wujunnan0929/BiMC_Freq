import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from test_residual_integration import temporary_artifact_directory
from tools import run_safd_dual_experiments as tool


class SAFDDualExperimentTest(unittest.TestCase):
    def test_default_plan_is_read_only_and_preserves_paired_seed_protocols(self):
        with temporary_artifact_directory() as directory:
            output = Path(directory) / 'unused'
            args = tool.parse_args(['--output-root', str(output)])
            plan = tool.build_plan(args)
            self.assertEqual(len(plan), 24)
            for seed in args.seeds:
                runs = [r for r in plan if r['seed'] == seed]
                self.assertEqual(len({r['support_manifest'] for r in runs}), 1)
                for run in runs:
                    values = run['command'][run['command'].index('--opts') + 1:]
                    opts = dict(zip(values[::2], values[1::2]))
                    self.assertEqual(opts[tool.U+'CALIBRATION_SEED'], str(seed + 10101))
            with patch.object(tool.common, 'execute_plan') as execute, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(tool.main(['--output-root', str(output), '--dry-run']), 0)
            execute.assert_not_called()
            self.assertFalse(output.exists())

    def test_summary_audits_anchor_predictions_zero_identity_and_total_time(self):
        with temporary_artifact_directory() as directory:
            root = Path(directory)
            summary = {key: 70. for key in tool.common.METRIC_NAMES}
            anchor_report = {'data_source': 'base_training_only', 'seed': 10102, 'run_seed': 1,
                             'auto_calibrate': True, 'protocol_sha256': 'paired',
                             'fit_class_ids': [0, 1], 'validation_class_ids': [2, 3]}
            rows = []
            for name in ('baseline', 'original_gda', 'zero_dual', 'fft_dual', 'safd_dual'):
                folder = root / name
                folder.mkdir()
                (folder / 'support.json').write_text('{"same":true}')
                metrics = {'status': 'completed', 'summary': summary,
                           'sessions': [{'accuracy': 70., 'eval_seconds': 2., 'fit_seconds': 1.,
                                         'anchor_accuracy': 70., 'reference_scores_exact_equal': name == 'baseline'},
                                        {'accuracy': 70., 'eval_seconds': 3., 'fit_seconds': 4.,
                                         'anchor_accuracy': 70., 'reference_scores_exact_equal': name == 'baseline'}]}
                (folder / 'metrics.json').write_text(json.dumps(metrics))
                if name != 'baseline':
                    (folder / 'uncertainty_calibration.json').write_text(json.dumps(anchor_report))
                is_dual = name.endswith('dual')
                if is_dual:
                    (folder / 'dual_frequency_calibration.json').write_text(json.dumps({
                        'auto_calibrate': name != 'zero_dual', 'anchor_protocol_sha256': 'paired',
                        'protocol_sha256': 'paired', 'selected_lambda': 0. if name == 'zero_dual' else .1}))
                for session in range(2):
                    payload = {'sample_id': np.array([5, 9]), 'target': np.array([1, 0]),
                               'reference_prediction': np.array([0, 0]),
                               'prediction': np.array([0, 0]) if name == 'baseline' else np.array([1, 0])}
                    if is_dual:
                        payload['anchor_prediction'] = np.array([1, 0])
                    np.savez_compressed(folder / f'predictions_session_{session:02d}.npz', **payload)
                rows.append({'variant': name, 'seed': 1, 'support_seed': 1, 'status': 'completed',
                             'output_dir': str(folder), 'summary': summary})
            args = tool.parse_args(['--seeds', '1'])
            path = root / 'summary.json'
            tool.save_summary(path, args, rows, len(rows))
            report = json.loads(path.read_text())
            checks = report['dual_pairing_audit']['1']
            for name in ('zero_dual', 'fft_dual', 'safd_dual'):
                self.assertTrue(checks[name]['same_anchor_protocol'])
                self.assertTrue(checks[name]['anchor_comparison']['complete_session_set'])
                self.assertEqual(checks[name]['anchor_comparison']['prediction_mismatches'], 0)
            self.assertEqual(checks['zero_dual']['zero_output_comparison']['prediction_mismatches'], 0)
            self.assertIn('fft_dual', report['paired_delta_vs_reference'])
            for row in report['runs']:
                self.assertEqual(row['eval_seconds_total'], 5.)
                self.assertEqual(row['fit_seconds_total'], 5.)
            # Corrupt a zero output while leaving its anchor intact; audit must fail.
            folder = root / 'zero_dual'
            np.savez_compressed(folder / 'predictions_session_01.npz', sample_id=[5, 9], target=[1, 0],
                                reference_prediction=[0, 0], prediction=[0, 0], anchor_prediction=[1, 0])
            tool.save_summary(path, args, rows, len(rows))
            changed = json.loads(path.read_text())['dual_pairing_audit']['1']['zero_dual']
            self.assertEqual(changed['zero_output_comparison']['prediction_mismatches'], 1)
            self.assertEqual(changed['anchor_comparison']['prediction_mismatches'], 0)


if __name__ == '__main__':
    unittest.main()
