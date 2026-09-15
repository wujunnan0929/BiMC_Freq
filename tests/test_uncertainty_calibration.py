"""Check class-disjoint selection, support-only estimates, and frozen refitting."""

import copy
import json
import unittest
from unittest.mock import patch

import torch

from engine.uncertainty_calibration import initialize_uncertainty, _sequence_records
from models.frequency_uncertainty import class_statistics, pooled_variance
from test_residual_integration import make_model, synthetic_state
from test_uncertainty_integration import uncertainty_config


def calibration_state(classes=6, control='frequency'):
    state = synthetic_state(num_classes=classes)
    state['uncertainty_features'] = (state['images_features'].unsqueeze(1).clone()
                                     if control == 'original' else state['frequency_features'].clone())
    # The auxiliary classifier never requires frequency descriptions.
    for key in list(state):
        if key.startswith('frequency_'):
            state.pop(key)
    return state


class UncertaintyCalibrationTest(unittest.TestCase):
    def test_class_split_prior_fit_and_refit_are_reproducible(self):
        cfg, state = uncertainty_config(auto=True), calibration_state()
        model = make_model(cfg)
        before = {k: v.clone() for k, v in state.items() if torch.is_tensor(v)}
        report = initialize_uncertainty(model, cfg, state)
        fit, validation = set(report['fit_class_ids']), set(report['validation_class_ids'])
        self.assertFalse(fit & validation)
        self.assertEqual(fit | validation, set(range(6)))
        for sequence in report['validation_sequences']:
            self.assertTrue(set(sequence['class_ids']) <= validation)
            support = {index for group in sequence['support_indices'] for index in group}
            query = {index for group in sequence['query_indices'] for index in group}
            self.assertFalse(support & query)
        expected_fit = pooled_variance(class_statistics(state['uncertainty_features'],
                                                        state['images_targets'], sorted(fit)))
        self.assertTrue(torch.allclose(torch.tensor(report['selection_prior']), expected_fit))
        expected_all = pooled_variance(class_statistics(state['uncertainty_features'],
                                                        state['images_targets'], list(range(6))))
        self.assertTrue(torch.equal(model.uncertainty_state['prior'], expected_all))
        self.assertEqual(report, initialize_uncertainty(make_model(cfg), cfg, state))
        json.dumps(report, allow_nan=False)
        for key, value in before.items():
            self.assertTrue(torch.equal(value, state[key]), key)
        with self.assertRaisesRegex(RuntimeError, 'once'):
            initialize_uncertainty(model, cfg, state)

    def test_query_images_do_not_affect_support_statistics_or_reference(self):
        cfg, state = uncertainty_config(auto=True), calibration_state()
        model = make_model(cfg)
        sequence = {
            'class_ids': [0, 1, 2],
            'support_indices': [[0, 1, 2, 3], [12, 13, 14, 15], [24, 25]],
            'query_indices': [[4, 5], [16, 17], [26, 27]],
        }
        first = list(_sequence_records(model, cfg, state, sequence))
        changed = copy.deepcopy(state)
        queries = [i for row in sequence['query_indices'] for i in row]
        changed['images_features'][queries] *= -1
        changed['uncertainty_features'][queries] *= -1
        second = list(_sequence_records(model, cfg, changed, sequence))
        for original, perturbed in zip(first, second):
            for key, value in original['stats'].items():
                self.assertTrue(torch.equal(value, perturbed['stats'][key]))
            for key, value in original['reference_state'].items():
                self.assertTrue(torch.equal(value, perturbed['reference_state'][key]))

    def test_tied_candidates_choose_exact_zero_once_and_counts_balance(self):
        cfg = uncertainty_config(auto=True)
        with patch('engine.uncertainty_calibration.mix_uncertainty_probabilities',
                   side_effect=lambda reference, logits, alpha, temperature: reference):
            report = initialize_uncertainty(make_model(cfg), cfg, calibration_state())
        self.assertEqual(report['selected_alpha'], 0.0)
        self.assertEqual(sum(c['alpha'] == 0 for c in report['candidate_results']), 1)
        for candidate in report['candidate_results']:
            self.assertEqual(candidate['corrected'], 0)
            self.assertEqual(candidate['damaged'], 0)
            for stage in candidate['stages']:
                groups = stage['groups']
                self.assertEqual(groups['all']['count'], groups['old']['count'] + groups['new']['count'])
                self.assertEqual(groups['new']['count'], groups['current_new']['count']
                                 + groups['historical_incremental']['count'])
            expected = sum((s['groups']['old']['accuracy'] + s['groups']['new']['accuracy']) / 2
                           for s in candidate['stages'][1:]) / (len(candidate['stages']) - 1)
            self.assertAlmostEqual(candidate['balanced_objective'], expected)

    def test_manual_mode_needs_no_pseudo_episode_capacity(self):
        cfg = uncertainty_config(auto=False, alpha=0.2)
        with patch('engine.uncertainty_calibration.sample_sequence', side_effect=AssertionError('No episodes')):
            report = initialize_uncertainty(make_model(cfg), cfg, calibration_state(classes=2))
        self.assertEqual(report['selection_status'], 'manual_unvalidated')
        self.assertEqual(report['selected_alpha'], 0.2)
        self.assertEqual(report['candidate_results'], [])
        self.assertEqual(report['validation_sequences'], [])

    def test_shared_covariance_deduplicates_irrelevant_strengths(self):
        cfg = uncertainty_config(auto=True, control='original')
        cfg.defrost()
        cfg.TRAINER.BiMC.UNCERTAINTY.COVARIANCE = 'shared'
        cfg.freeze()
        report = initialize_uncertainty(make_model(cfg), cfg, calibration_state(control='original'))
        self.assertEqual(len(report['candidate_results']), 1 + 3 * 3)

    def test_future_test_data_wrong_bands_and_invalid_grids_are_rejected(self):
        cfg = uncertainty_config(auto=True)
        state = calibration_state()
        state['images_targets'][0] = cfg.DATASET.NUM_INIT_CLS
        with self.assertRaisesRegex(ValueError, 'base-class'):
            initialize_uncertainty(make_model(cfg), cfg, state)
        state = calibration_state()
        state['data_source'] = 'test'
        with self.assertRaisesRegex(ValueError, 'base training'):
            initialize_uncertainty(make_model(cfg), cfg, state)
        with self.assertRaisesRegex(ValueError, 'band count'):
            initialize_uncertainty(make_model(cfg), cfg, calibration_state(control='original'))
        for name, value in (('ALPHA_GRID', [0.1]), ('TEMPERATURE_GRID', [0.0]),
                            ('PRIOR_GRID', [-1.0]), ('FIT_FRACTION', 1.0)):
            bad = cfg.clone()
            bad.defrost()
            setattr(bad.TRAINER.BiMC.UNCERTAINTY, name, value)
            bad.freeze()
            with self.assertRaises(ValueError):
                initialize_uncertainty(make_model(bad), bad, calibration_state())


if __name__ == '__main__':
    unittest.main()
