"""Calibration parity, split isolation and privileged-diagnostic boundaries."""
import contextlib
import io
from pathlib import Path
import unittest

import torch
from torch.nn import functional as F

from engine import prototype_evaluation as evaluation
from engine.safd_metric_training import _reference_proxy
from test_safd_metric_training import fixture
from test_residual_integration import temporary_artifact_directory
from tools import screen_prototype_space as cli


class PrototypeScreeningTest(unittest.TestCase):
    def records(self):
        arrays, cfg, _ = fixture()
        cfg, protocol, pool = evaluation.prepare_context(arrays, cfg, seed=3, sequences=2, teacher_shot=4)
        return arrays, cfg, protocol, pool, evaluation.validation_records(arrays, cfg, protocol)

    def test_teacher_support_query_and_fitting_class_isolation(self):
        arrays, cfg, protocol, pool, records = self.records()
        fit = set(protocol['fitting_indices'])
        reserved = {i for rows in protocol['reserved_pseudo_base_query_indices'].values() for i in rows}
        self.assertFalse(fit & reserved)
        self.assertEqual(len(fit), 3 * 18)
        for row in records:
            self.assertEqual(len(row['arrival_sizes']), row['stage'])
            self.assertEqual(len(row['supports']), len(row['class_ids']))
            for j, c in enumerate(row['class_ids'][row['num_old']:], row['num_old']):
                teacher = set(protocol['audit_indices'][str(c)])
                self.assertFalse(teacher & set(row['support_indices'][j]))
                self.assertFalse(teacher & set(row['query_indices'][j]))
                self.assertFalse(teacher & fit)
            query = torch.tensor([i for rows in row['query_indices'] for i in rows])
            actual = torch.tensor(row['class_ids'])[row['targets']]
            self.assertTrue(torch.equal(actual, torch.as_tensor(arrays['labels'])[query]))

    def test_identity_and_equal_prototypes_return_exact_legacy_object(self):
        _, cfg, _, _, records = self.records()
        row = records[0]
        self.assertIs(evaluation.score_record(row, cfg), row['reference_votes'])
        self.assertIs(evaluation.score_record(row, cfg, row['reference_prototypes'].clone(),
                                            evaluation.identity_calibration()), row['reference_votes'])
        runtime = dict(row)
        runtime['image_proto'] = runtime.pop('reference_prototypes')
        runtime['cov_image'] = runtime.pop('covariance')
        self.assertIs(evaluation.score_record(runtime, cfg, runtime['image_proto']), row['reference_votes'])

    def test_changed_prototype_scores_match_legacy_bimc_with_same_statistics(self):
        _, cfg, _, _, records = self.records()
        proxy = _reference_proxy(cfg)
        for row in records:
            proto = row['reference_prototypes'].clone()
            proto[-1] = F.normalize(proto[-1] + .12 * torch.arange(proto.shape[1]), dim=-1)
            expected = proxy.reference_scores_from_features(row['query'], len(proto), row['num_old'],
                proto, row['covariance'], row['description_proto'], row['description_features'],
                row['description_targets'], row['text_features'], cfg.DATASET.BETA,
                cov_inverse=row['cov_inverse'])
            actual = evaluation.score_record(row, cfg, proto)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
            self.assertTrue(torch.isfinite(actual).all())
            self.assertTrue((actual > 0).all())

    def test_grid_selection_keeps_identity_guard_and_excludes_oracles(self):
        _, cfg, _, _, records = self.records()
        grid = [{**evaluation.identity_calibration(), 'novel_bias': v} for v in (0., -.1, .1)]
        calibration, report = evaluation.select_calibration(records, cfg, maximum_drop=0., grid=grid)
        self.assertEqual(report['candidate_count'], 3)
        self.assertFalse(report['oracle_candidates_allowed'])
        self.assertTrue(all(report['selected_validation'][k] >= report['reference'][k] - 1e-12
                            for k in ('base_accuracy', 'novel_accuracy')))
        self.assertIn(calibration, grid)
        self.assertEqual(report['candidates'][0]['calibration'], evaluation.identity_calibration())
        baseline = evaluation.evaluate_records(records, lambda r: r['reference_votes'])
        self.assertEqual(baseline['changed_observations'], 0)

    def test_description_count_control_uses_only_a_fixed_text_subset(self):
        _, cfg, _, _, records = self.records()
        row = records[0]
        self.assertIs(evaluation.description_count_scores(row, cfg, 2), row['reference_votes'])
        selected = torch.cat([torch.where(row['description_targets'] == c)[0][:1]
                              for c in range(len(row['reference_prototypes']))])
        expected = _reference_proxy(cfg).reference_scores_from_features(
            row['query'], len(row['reference_prototypes']), row['num_old'], row['reference_prototypes'],
            row['covariance'], row['description_proto'], row['description_features'][selected],
            row['description_targets'][selected], row['text_features'], cfg.DATASET.BETA,
            cov_inverse=row['cov_inverse'])
        torch.testing.assert_close(evaluation.description_count_scores(row, cfg, 1), expected,
                                   rtol=1e-5, atol=2e-6)
        with self.assertRaises(ValueError):
            evaluation.description_count_scores(row, cfg, 3)

    def test_oracle_is_label_using_diagnostic_and_teacher_does_not_change_reference(self):
        _, cfg, _, _, records = self.records()
        snapshots = [r['reference_votes'].clone() for r in records]
        baseline = evaluation.evaluate_records(records, lambda r: r['reference_votes'])
        oracle = evaluation.evaluate_records(records, lambda r: evaluation.expert_oracle_scores(r, cfg))
        self.assertGreaterEqual(oracle['base_accuracy'], baseline['base_accuracy'])
        self.assertGreaterEqual(oracle['novel_accuracy'], baseline['novel_accuracy'])
        evaluation.evaluate_records(records, lambda r: evaluation.score_record(r, cfg, r['teacher_prototypes']))
        self.assertTrue(all(torch.equal(r['reference_votes'], s) for r, s in zip(records, snapshots)))

    def test_cli_refuses_existing_output_before_loading_nonexistent_cache(self):
        with temporary_artifact_directory() as directory:
            with self.assertRaisesRegex(ValueError, 'fresh output'):
                cli.main(['--cache', 'nonexistent.npz', '--output-dir', directory])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.parse_args(['--cache', 'x', '--output-dir', 'out', '--teacher-shot', '0'])


if __name__ == '__main__':
    unittest.main()
