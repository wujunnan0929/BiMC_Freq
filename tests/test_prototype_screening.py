"""Calibration parity, split isolation and privileged-diagnostic boundaries."""
import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch

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

    def test_factorial_controls_apply_same_guidance_and_preserve_old_prototypes(self):
        _, cfg, _, _, records = self.records()
        row = records[0]
        snapshots = {key: row[key].clone() for key in ('raw_prototypes', 'reference_prototypes',
                                                      'teacher_prototypes', 'reference_votes')}
        calibration = {**evaluation.identity_calibration(), 'fused_temperature': .1}
        expected = row['teacher_prototypes'].clone()
        old = row['num_old']
        expected[old:] = _reference_proxy(cfg).soft_calibration(
            row['reference_prototypes'][:old], expected[old:])
        torch.testing.assert_close(evaluation.prototype_control_scores(row, cfg, 'teacher_guided', calibration),
                                   evaluation.score_record(row, cfg, expected, calibration))
        torch.testing.assert_close(expected[:old], row['reference_prototypes'][:old], rtol=0, atol=0)
        self.assertIs(evaluation.prototype_control_scores(row, cfg, 'support_guided'), row['reference_votes'])
        report = evaluation.prototype_factorial_diagnostics(records, cfg, calibration)
        self.assertEqual(report['fixed_score_calibration'], calibration)
        self.assertEqual(report['support_counts'], [2])
        self.assertEqual(report['teacher_counts'], [4])
        self.assertEqual(report['variants']['support_guided']['metrics']['changed_observations'], 0)
        self.assertFalse(report['variants']['teacher_guided']['deployable'])
        self.assertFalse(any(v['selection_candidate'] for v in report['variants'].values()))
        self.assertEqual(set(report['contrasts']), {'disable_guidance_support', 'disable_guidance_teacher',
                                                  'teacher_vs_support_guided', 'teacher_vs_support_raw'})
        self.assertTrue(all(torch.equal(row[key], value) for key, value in snapshots.items()))
        with self.assertRaisesRegex(ValueError, 'Unknown prototype control'):
            evaluation.prototype_control_scores(row, cfg, 'unknown')

    def test_factorial_guided_and_raw_agree_when_visual_guidance_is_disabled(self):
        arrays, cfg, protocol, _, _ = self.records()
        cfg = cfg.clone()
        cfg.defrost()
        cfg.TRAINER.BiMC.VISION_CALIBRATION = False
        cfg.freeze()
        row = evaluation.validation_records(arrays, cfg, protocol)[0]
        for source in ('support', 'teacher'):
            torch.testing.assert_close(evaluation.prototype_control_scores(row, cfg, source + '_guided'),
                                       evaluation.prototype_control_scores(row, cfg, source + '_raw'),
                                       rtol=0, atol=0)

    def test_expert_profiles_separate_single_predictors_from_query_label_union(self):
        _, cfg, _, _, _ = self.records()
        target = torch.arange(4)
        def scores(correct_indices):
            prediction = (target + 1) % 4
            prediction[correct_indices] = target[correct_indices]
            return F.one_hot(prediction, 4).float()
        original, calibrated = scores([0, 3]), scores([0, 1, 3])
        row = {'targets': target, 'num_old': 2, 'reference_votes': original}
        logits = {'fused': scores([0, 2]), 'cov': scores([1, 2]), 'description': scores([1, 3])}
        with patch.object(evaluation, 'branch_logits', return_value=logits), \
             patch.object(evaluation, 'score_record', return_value=calibrated):
            report = evaluation.expert_profile_diagnostics([row], cfg)
        self.assertEqual(report['experts']['original_BiMC']['metrics']['damaged_observations'], 1)
        self.assertEqual(report['experts']['calibrated_BiMC']['metrics']['changed_observations'], 0)
        self.assertEqual(report['best_fixed_expert_by_development_H']['name'], 'calibrated_BiMC')
        union = report['label_oracle_union']
        self.assertEqual(union['metrics']['harmonic_accuracy'], 100.)
        self.assertEqual(union['metrics']['corrected_observations'], 1)
        self.assertAlmostEqual(union['H_gap_vs_best_fixed_expert_pp'], 100. - 200. / 3.)
        pair = next(p for p in report['pairwise_correctness']
                    if p['left'] == 'original_BiMC' and p['right'] == 'fused_prototype')
        self.assertEqual(pair['groups']['all'], {'observations': 4, 'both_correct': 1,
                         'left_only_correct': 1, 'right_only_correct': 1, 'both_wrong': 1})
        self.assertEqual(pair['groups']['novel']['left_only_correct'], 1)
        self.assertEqual(pair['groups']['novel']['right_only_correct'], 1)
        self.assertFalse(report['selection_candidate'])

    def test_explicit_metric_anchor_keeps_legacy_default_counts(self):
        target = torch.arange(4)
        wrong = F.one_hot((target + 1) % 4, 4).float()
        correct = F.one_hot(target, 4).float()
        records = [{'targets': target, 'num_old': 2, 'reference_votes': wrong}]
        legacy = evaluation.evaluate_records(records, lambda row: correct)
        paired = evaluation.evaluate_records(records, lambda row: correct, lambda row: correct)
        self.assertEqual(legacy['corrected_observations'], 4)
        self.assertEqual(paired['corrected_observations'], 0)
        self.assertEqual(paired['changed_observations'], 0)
        self.assertEqual(legacy['harmonic_accuracy'], paired['harmonic_accuracy'])

    def test_cli_refuses_existing_output_before_loading_nonexistent_cache(self):
        with temporary_artifact_directory() as directory:
            with self.assertRaisesRegex(ValueError, 'fresh output'):
                cli.main(['--cache', 'nonexistent.npz', '--output-dir', directory])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.parse_args(['--cache', 'x', '--output-dir', 'out', '--teacher-shot', '0'])


if __name__ == '__main__':
    unittest.main()
