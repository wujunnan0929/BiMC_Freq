"""Selection isolation and support-only statistical-completion regressions."""

import contextlib
import copy
import io
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from engine import completion_calibration as calibration
from test_completion_integration import configuration
from test_residual_integration import make_model, synthetic_state


def base_fixture():
    cfg = configuration(auto=True, base_only=True)
    state = synthetic_state(num_classes=6, samples_per_class=20)
    state['completion_features'] = torch.stack((state['images_features'], F.normalize(
        state['images_features'] + .35 * state['images_features'].roll(1, -1), dim=-1)), dim=1)
    state['completion_text'] = torch.stack((state['text_features'], state['frequency_prompt_proto'][:, 0]), dim=1)
    state.update(data_source='base_training_only', task_id=0)
    model = make_model(cfg)
    model._last_support_ids = list(range(len(state['images_targets'])))
    protocol = calibration.completion_protocol(cfg, state['images_targets'], model._last_support_ids)
    return cfg, model, state, protocol


class CompletionCalibrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def assert_statistics_equal(self, first, second):
        self.assertEqual(set(first), set(second))
        for key in first:
            with self.subTest(key=key):
                if torch.is_tensor(first[key]):
                    self.assertTrue(torch.equal(first[key], second[key]))
                else:
                    self.assertEqual(first[key], second[key])

    def test_protocol_is_deterministic_and_globally_excludes_a_queries(self):
        cfg, model, current, protocol = base_fixture()
        self.assertEqual(protocol, calibration.completion_protocol(cfg, current['images_targets'], model._last_support_ids))
        a, v = protocol['pseudo_base_class_ids'], protocol['pseudo_validation_class_ids']
        self.assertFalse(set(a) & set(v))
        fit = set(protocol['fitting_indices'])
        reserved = {i for rows in protocol['reserved_pseudo_base_query_indices'].values() for i in rows}
        self.assertFalse(fit & reserved)
        self.assertEqual(set(current['images_targets'][list(fit)].tolist()), set(a))
        for seq in protocol['sequences']:
            support = {i for rows in seq['support_indices'] for i in rows}
            query = {i for rows in seq['query_indices'] for i in rows}
            self.assertFalse(support & query)
            self.assertEqual({i for rows in seq['query_indices'][:len(a)] for i in rows}, reserved)
            self.assertEqual({i for rows in seq['support_indices'][:len(a)] for i in rows}, fit)

    def test_changing_v_and_reserved_a_queries_cannot_change_fitting_statistics(self):
        cfg, _, current, protocol = base_fixture()
        a = protocol['pseudo_base_class_ids']
        first = calibration._fit(cfg, current, protocol['fitting_indices'], a)
        changed = copy.deepcopy(current)
        fitting = set(protocol['fitting_indices'])
        excluded = [i for i in range(len(current['images_targets'])) if i not in fitting]
        changed['images_features'][excluded] *= -10
        changed['completion_features'][excluded] *= -10
        changed['completion_text'][protocol['pseudo_validation_class_ids']] *= -10
        second = calibration._fit(cfg, changed, protocol['fitting_indices'], a)
        self.assert_statistics_equal(first, second)

    def test_validation_queries_cannot_change_candidates_or_covariance(self):
        cfg, model, current, protocol = base_fixture()
        fitted = calibration._fit(cfg, current, protocol['fitting_indices'], protocol['pseudo_base_class_ids'])
        sequence = protocol['sequences'][0]
        first = list(calibration.completion_sequence_records(model, cfg, current, sequence, fitted))
        changed = copy.deepcopy(current)
        query_rows = [i for rows in sequence['query_indices'] for i in rows]
        changed['images_features'][query_rows] *= -1
        changed['completion_features'][query_rows] *= -1
        second = list(calibration.completion_sequence_records(model, cfg, changed, sequence, fitted))
        for before, after in zip(first, second):
            for key in ('reference', 'raw', 'completed', 'covariance'):
                self.assertTrue(torch.equal(before[key], after[key]), key)
            self.assertFalse(torch.equal(before['query'], after['query']))

    def test_future_class_support_does_not_change_earlier_stage_candidates(self):
        cfg, model, current, protocol = base_fixture()
        fitted = calibration._fit(cfg, current, protocol['fitting_indices'], protocol['pseudo_base_class_ids'])
        sequence = protocol['sequences'][0]
        before = list(calibration.completion_sequence_records(model, cfg, current, sequence, fitted))[0]
        changed = copy.deepcopy(current)
        future = sequence['support_indices'][-1]
        changed['images_features'][future] *= -1
        changed['completion_features'][future] *= -1
        after = list(calibration.completion_sequence_records(model, cfg, changed, sequence, fitted))[0]
        for key in ('reference', 'raw', 'completed', 'covariance', 'query'):
            self.assertTrue(torch.equal(before[key], after[key]), key)

    def test_history_and_zero_path_are_exact_for_every_candidate(self):
        cfg, model, current, protocol = base_fixture()
        fitted = calibration._fit(cfg, current, protocol['fitting_indices'], protocol['pseudo_base_class_ids'])
        records = list(calibration.completion_sequence_records(model, cfg, current, protocol['sequences'][0], fitted))
        old = records[0]['num_base']
        banks = {}
        for record in records:
            for value in (0., .25, 1.):
                novel = calibration.blend_completion(record['reference'][old:], record['raw'][old:], record['completed'], value)
                bank = torch.cat((record['reference'][:old], novel))
                if value in banks:
                    self.assertTrue(torch.equal(banks[value], bank[:len(banks[value])]))
                banks[value] = bank
                if value == 0:
                    self.assertTrue(torch.equal(bank, record['reference']))
        reference = records[0]['reference']
        self.assertIs(calibration.blend_completion(reference, reference, reference, 0.), reference)

    def test_selection_uses_overall_accuracy_then_smallest_blend_with_group_guards(self):
        rows = [{'blend': value, 'mean_base_accuracy': base, 'mean_novel_accuracy': novel,
                 'mean_accuracy': overall, 'mean_harmonic_accuracy': harmonic}
                for value, base, novel, overall, harmonic in
                [(0., 80., 60., 72., 68.), (.25, 79.9, 61., 73., 69.),
                 (.5, 80., 60., 72.8, 75.), (1., 79., 70., 80., 78.)]]
        self.assertEqual(calibration.select_completion_candidate(rows, .2)['blend'], .25)
        self.assertFalse(rows[-1]['feasible'])
        ties = [copy.deepcopy(rows[0]), {**rows[0], 'blend': .25}]
        self.assertEqual(calibration.select_completion_candidate(ties, .2)['blend'], 0.)

    def test_all_base_refit_happens_after_selection_and_zero_skips_refit(self):
        for selected in (.5, 0.):
            with self.subTest(selected=selected):
                cfg, model, current, protocol = base_fixture()
                calls = []
                fit = calibration._fit

                def record_fit(*args, **kwargs):
                    fitted = fit(*args, **kwargs)
                    calls.append(('fit', fitted['fit_class_ids'].tolist(), fitted['fit_sample_count']))
                    return fitted

                def select(model_arg, cfg_arg, current_arg, protocol_arg, fitted):
                    calls.append(('select', fitted['fit_class_ids'].tolist(), fitted['fit_sample_count']))
                    return {'blend': selected, 'stages': []}, []

                with patch.object(calibration, '_fit', side_effect=record_fit), \
                        patch.object(calibration, '_select', side_effect=select), contextlib.redirect_stdout(io.StringIO()):
                    report = calibration.initialize_completion(model, cfg, current)
                self.assertEqual([row[0] for row in calls], ['fit', 'select', 'fit'] if selected else ['fit', 'select'])
                self.assertEqual(calls[0][1], protocol['pseudo_base_class_ids'])
                self.assertEqual(calls[0][2], len(protocol['fitting_indices']))
                self.assertEqual(report['selected_blend'], selected)
                if selected:
                    self.assertEqual(calls[-1][1], list(range(6)))
                    self.assertEqual(calls[-1][2], len(current['images_targets']))
                    self.assertEqual(report['refit_class_ids'], list(range(6)))
                else:
                    self.assertIsNone(model.completion_state['statistics'])
                    self.assertEqual(report['refit_source'], 'skipped_zero_blend')
                with self.assertRaisesRegex(RuntimeError, 'only once'):
                    calibration.initialize_completion(model, cfg, current)


if __name__ == '__main__':
    unittest.main()
