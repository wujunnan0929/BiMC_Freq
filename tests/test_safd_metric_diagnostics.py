"""Diagnostic score parity and exact, base-only replay checks."""
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from engine import safd_metric_training as training
from models.safd_metric import prepare_residual_classifier, apply_metric_residual
from tools import diagnose_safd_metric as diagnostic
from test_safd_metric_training import fixture
from test_residual_integration import temporary_artifact_directory


class SAFDMetricDiagnosticTest(unittest.TestCase):
    def row(self):
        return {'query': torch.eye(2), 'prototypes': torch.eye(2),
                'reference_votes': torch.tensor([[.8, .2], [.2, .8]]),
                'targets': torch.tensor([0, 1]), 'num_old': 1}

    def test_sweep_reproduces_runtime_and_detects_valid_spd_reversal(self):
        row = self.row()
        summaries, stages, probe = diagnostic.score_sweep([row], .5 * torch.eye(2), [0., .1, 16.])
        self.assertEqual([r['harmonic_accuracy'] for r in summaries], [100., 100., 0.])
        self.assertEqual(summaries[-1]['damaged'], 2)
        self.assertEqual(summaries[-1]['corrected'], 0)
        self.assertEqual(summaries[-1]['fraction_reference_pair_reversed'], 1.)
        self.assertEqual(summaries[0]['max_abs_scaled_residual'], 0)
        self.assertTrue(all(r['runtime_score_parity_checked'] for r in summaries))
        prepared = prepare_residual_classifier(row['prototypes'], .5 * torch.eye(2))
        for s in summaries:
            actual = apply_metric_residual(row['reference_votes'], row['query'], prepared, s['gain'])
            self.assertEqual(s['correct'], int((actual.argmax(1) == row['targets']).sum()))
        self.assertEqual(set(stages), {'0.0', '0.1', '16.0'})
        self.assertEqual(probe['query'].shape, (2, 2))

    def test_random_scores_and_identity_use_actual_runtime_dtype_and_ties(self):
        rng = torch.Generator().manual_seed(13)
        row = {'query': torch.randn(9, 5, generator=rng),
               'prototypes': torch.randn(3, 5, generator=rng),
               'reference_votes': torch.rand(9, 3, generator=rng) + .1,
               'targets': torch.arange(9) % 3, 'num_old': 1}
        for dtype in (torch.float32, torch.float64):
            row['reference_votes'] = row['reference_votes'].to(dtype)
            for m in (torch.eye(5), torch.diag(torch.tensor([.2, .7, 1., 1.3, 1.8]))):
                summary, _, _ = diagnostic.score_sweep([row], m, [0., .03, 1., 16.])
                p = prepare_residual_classifier(row['prototypes'], m)
                for s in summary:
                    y = apply_metric_residual(row['reference_votes'], row['query'], p, s['gain']).argmax(1)
                    self.assertEqual(s['correct'], int((y == row['targets']).sum()))

    def test_score_probe_includes_old_and_novel_queries_in_a_large_stage(self):
        row = {'query': torch.eye(2).repeat_interleave(25, 0),
               'prototypes': torch.eye(2),
               'reference_votes': torch.tensor([[.8, .2]]).repeat(50, 1),
               'targets': torch.arange(2).repeat_interleave(25), 'num_old': 1}
        _, _, probe = diagnostic.score_sweep([row], torch.eye(2), [0.])
        self.assertEqual(probe['query'].shape, (32, 2))
        self.assertEqual(int((probe['targets_local'] == 0).sum()), 16)
        self.assertEqual(int((probe['targets_local'] == 1).sum()), 16)

    def test_closed_form_replay_matches_recorded_selection_and_only_uses_A(self):
        arrays, cfg, options = fixture()
        options.method = 'closed_form'
        options.strength_grid = (0., .1)
        with contextlib.redirect_stdout(io.StringIO()):
            _, report, _ = training.fit_metric(arrays, cfg, options)
        args = SimpleNamespace(mode='closed_form', gains=[0., .1, 16.], strengths=[0., .1],
                               checkpoints=[0], device='cpu', save_score_probe=True, reference_tolerance_pp=1e-5)
        with temporary_artifact_directory() as directory, contextlib.redirect_stdout(io.StringIO()):
            out = Path(directory)
            result = diagnostic.diagnose(arrays, {'complete_base_bank': True}, report, cfg, args, out)
            self.assertEqual(result['status'], 'completed')
            self.assertTrue(result['all_compared_checkpoints_within_tolerance'])
            self.assertEqual(len(result['candidates']), 6)
            self.assertFalse(result['benchmark_test_evaluation'])
            self.assertFalse(result['frozen_model_exported'])
            self.assertTrue((out / 'score_probe.npz').is_file())
            checks = json.loads((out / 'checks.json').read_text())
            self.assertEqual(checks['fitting_scatter_diagnostics']['num_samples'],
                             len(report['selection_protocol']['fitting_indices']))

    def test_learned_replay_preserves_one_optimizer_trajectory(self):
        arrays, cfg, options = fixture()
        options.steps = 2
        with contextlib.redirect_stdout(io.StringIO()):
            _, report, _ = training.fit_metric(arrays, cfg, options)
        args = SimpleNamespace(mode='learned', gains=[0., .1, 16.], strengths=[0.],
                               checkpoints=[0, 1, 2], device='cpu', save_score_probe=False, reference_tolerance_pp=1e-5)
        with temporary_artifact_directory() as directory, contextlib.redirect_stdout(io.StringIO()):
            result = diagnostic.diagnose(arrays, {}, report, cfg, args, Path(directory))
        self.assertTrue(result['checks']['training_episode_replay_exact'])
        self.assertTrue(result['all_compared_checkpoints_within_tolerance'])
        self.assertEqual([r['step'] for r in result['training_curve']], [0, 1, 2])
        self.assertGreater(result['training_curve'][-1]['theta_l2'], 0)
        self.assertEqual(len(result['candidates']), 9)

    def test_wrong_row_order_is_rejected_before_replaying_scores(self):
        arrays, cfg, options = fixture()
        with contextlib.redirect_stdout(io.StringIO()):
            _, report, _ = training.fit_metric(arrays, cfg, options)
        arrays = dict(arrays)
        arrays['sample_id'] = arrays['sample_id'][::-1].copy()
        with temporary_artifact_directory() as directory, self.assertRaisesRegex(ValueError, 'row order'):
            diagnostic.diagnose(arrays, {}, report, cfg, SimpleNamespace(), Path(directory))

    def test_cli_rejects_missing_zero_control(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            diagnostic.parse_args(['--cache', 'x', '--training-report', 'r', '--output-dir', 'out', '--gains', '1', '16'])


if __name__ == '__main__':
    unittest.main()
