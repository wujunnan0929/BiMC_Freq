import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from engine import prototype_estimator_training as training
from engine.prototype_evaluation import prepare_context
from models.support_estimator import SupportPrototypeEstimator
from test_safd_metric_training import fixture
from test_residual_integration import temporary_artifact_directory


def settings():
    return training.EstimatorOptions(seed=1, sequences=2, teacher_shot=2, query=2, old_shot=4,
        old_ways=(1,), novel_ways=(1,), novel_shots=(2,), episode_pool=2,
        steps=2, validate_every=1, early_checkpoints=(1,), hidden=8,
        blend_grid=(0., .5), calibrate=False)


class PrototypeEstimatorTrainingTest(unittest.TestCase):
    def test_fitting_episodes_exclude_validation_and_separate_support_teacher_query(self):
        arrays, cfg, _ = fixture()
        o = settings()
        cfg, protocol, pool = prepare_context(arrays, cfg, o.seed, o.sequences, o.teacher_shot)
        records, trace = training.training_episodes(arrays, cfg, pool, o)
        fit = set(protocol['fitting_indices'])
        reserved = {i for rows in protocol['reserved_pseudo_base_query_indices'].values() for i in rows}
        for ep in trace:
            self.assertTrue(set(ep['class_ids']) <= set(protocol['pseudo_base_class_ids']))
            for s, t, q in zip(ep['support_indices'], ep['teacher_indices'], ep['query_indices']):
                self.assertFalse(set(s) & set(t) or set(s) & set(q) or set(t) & set(q))
                self.assertTrue((set(s) | set(t) | set(q)) <= fit)
                self.assertFalse((set(s) | set(t) | set(q)) & reserved)
        self.assertEqual(len(records), 2)

    def test_auxiliary_has_identical_support_teacher_query_trace(self):
        arrays, cfg, _ = fixture()
        o = settings()
        cfg, _, pool = prepare_context(arrays, cfg, o.seed, o.sequences, o.teacher_shot)
        _, original = training.training_episodes(arrays, cfg, pool, o)
        o.auxiliary, o.auxiliary_probability = 'safd', 1.
        records, perturbed = training.training_episodes(arrays, cfg, pool, o, auxiliary_features=arrays['features'])
        for a, b in zip(original, perturbed):
            self.assertEqual({k: v for k, v in a.items() if k != 'auxiliary_edits'},
                             {k: v for k, v in b.items() if k != 'auxiliary_edits'})
            self.assertEqual(len(b['auxiliary_edits']), 1)
        self.assertTrue(all(torch.isfinite(r['query']).all() for r in records))

    def test_way_sampling_does_not_collapse_novel_ways_when_old_choice_is_too_large(self):
        arrays, cfg, _ = fixture()
        o = settings(); o.old_ways = (2, 5); o.novel_ways = (2, 3); o.episode_pool = 12
        pool = {c: np.flatnonzero(arrays['labels'] == c).tolist() for c in range(6)}
        _, trace = training.training_episodes(arrays, cfg, pool, o)
        self.assertTrue(all(ep['num_old'] == 2 for ep in trace))
        self.assertTrue(all(len(ep['class_ids']) - ep['num_old'] in (2, 3) for ep in trace))

    def test_optimizer_updates_shared_parameters_and_keeps_feature_inputs_frozen(self):
        arrays, cfg, _ = fixture()
        o = settings()
        cfg, _, pool = prepare_context(arrays, cfg, o.seed, o.sequences, o.teacher_shot)
        records, _ = training.training_episodes(arrays, cfg, pool, o)
        model = SupportPrototypeEstimator(hidden=8)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        history = training.optimize(model, records, o, 2)
        self.assertEqual(len(history), 2)
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in model.state_dict().items()))
        self.assertTrue(all(x.grad is None for row in records for x in row['supports']))
        self.assertTrue(all(np.isfinite(row['loss']) for row in history))

    def test_zero_step_selects_reference_and_refits_only_after_selection(self):
        arrays, cfg, _ = fixture()
        o = settings(); o.steps = 0
        with contextlib.redirect_stdout(io.StringIO()):
            model, report, _ = training.fit_estimator(arrays,
                {'complete_base_bank': True, 'has_text_features': True}, cfg, o)
        self.assertEqual(report['selected_step'], 0)
        self.assertEqual(report['blend'], 0)
        self.assertEqual(report['final_refit_training_episodes'], [])
        self.assertEqual(report['reference'], report['calibrated_reference'])
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_selected_duration_is_refit_fresh_on_all_base_only_after_selection(self):
        arrays, cfg, _ = fixture()
        arrays = dict(arrays)
        arrays['features'] = torch.nn.functional.normalize(
            torch.randn(120, 3, 8, generator=torch.Generator().manual_seed(123)), dim=-1).numpy()
        o = settings()
        real_evaluate = training.evaluate_estimator
        calls = []
        real_episodes = training.training_episodes
        def episodes(*args, **kwargs):
            calls.append(('pool', set(args[2])))
            return real_episodes(*args, **kwargs)
        def improved(*args, **kwargs):
            metrics = real_evaluate(*args, **kwargs)
            # Force duration selection to exercise the full-base phase; this
            # synthetic improvement is not a measured performance claim.
            metrics.update(base_accuracy=100., novel_accuracy=100., harmonic_accuracy=100.)
            return metrics
        with patch.object(training, 'training_episodes', side_effect=episodes), \
             patch.object(training, 'evaluate_estimator', side_effect=improved), \
             contextlib.redirect_stdout(io.StringIO()):
            _, report, _ = training.fit_estimator(arrays,
                {'complete_base_bank': True, 'has_text_features': True}, cfg, o)
        self.assertEqual(report['selected_step'], 1)
        self.assertEqual(len(calls[0][1]), 3)
        self.assertEqual(calls[-1][1], set(range(6)))
        self.assertTrue(report['final_refit_training_episodes'])
        self.assertEqual(len(report['final_training_history']), 1)
        self.assertLess(report['phase_order'].index('selection_ended'),
                        report['phase_order'].index('fresh_all_base_fixed_step_refit'))

    def test_auxiliary_row_and_weight_identity_are_enforced(self):
        arrays, _, _ = fixture()
        meta = {'source': 'safd', 'dataset': 'cub200', 'clip_weight_sha256': 'weightA'}
        wrong = dict(meta, clip_weight_sha256='weightB')
        with self.assertRaisesRegex(ValueError, 'provenance'):
            training.check_auxiliary(arrays, meta, arrays, wrong, 'safd')
        changed = dict(arrays, sample_id=arrays['sample_id'][::-1].copy())
        with self.assertRaisesRegex(ValueError, 'row order'):
            training.check_auxiliary(arrays, meta, changed, meta, 'safd')

    def test_query_option_controls_both_training_and_reserved_validation_queries(self):
        arrays, cfg, _ = fixture()
        o = settings(); o.steps = 0; o.query = 3
        with contextlib.redirect_stdout(io.StringIO()):
            _, report, _ = training.fit_estimator(arrays,
                {'complete_base_bank': True, 'has_text_features': True}, cfg, o)
        self.assertTrue(all(len(q) == 3 for q in report['selection_protocol']['reserved_pseudo_base_query_indices'].values()))
        self.assertTrue(all(len(q) == 3 for ep in report['selection_training_episodes'] for q in ep['query_indices']))

    def test_validation_teacher_features_cannot_change_selection(self):
        arrays, cfg, _ = fixture()
        o = settings()
        _, protocol, _ = prepare_context(arrays, cfg, o.seed, o.sequences, o.teacher_shot)
        teacher_rows = [i for rows in protocol['audit_indices'].values() for i in rows]
        altered = dict(arrays)
        altered['features'] = arrays['features'].copy()
        altered['features'][teacher_rows] = torch.nn.functional.normalize(
            torch.randn(len(teacher_rows), 3, 8, generator=torch.Generator().manual_seed(77)), dim=-1).numpy()
        meta = {'complete_base_bank': True, 'has_text_features': True}
        with contextlib.redirect_stdout(io.StringIO()):
            _, a, _ = training.fit_estimator(arrays, meta, cfg, o)
            _, b, _ = training.fit_estimator(altered, meta, cfg, o)
        self.assertEqual(a['checkpoints'], b['checkpoints'])
        self.assertEqual(a['selected_validation'], b['selected_validation'])


if __name__ == '__main__':
    unittest.main()
