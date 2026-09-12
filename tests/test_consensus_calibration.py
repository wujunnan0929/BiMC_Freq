import copy
import json
import unittest

import torch
from torch.nn import functional as F

from engine.consensus_calibration import (
    build_support_reference, initialize_consensus, sample_sequence,
)
from test_consensus_integration import consensus_config
from test_residual_integration import make_model, synthetic_state
from tools.analyze_consensus_calibration import inspect_report


def calibration_state():
    state = synthetic_state()
    state['frequency_consensus_text_proto'] = F.normalize(
        state['frequency_description_candidates'].mean(2), dim=-1,
    )
    return state


class ConsensusCalibrationTest(unittest.TestCase):
    def test_selection_only_changes_lambda_with_identical_scales_and_sequences(self):
        reports = []
        for objective, guard in (('micro_all', -1.), ('micro_incremental', -1.),
                                 ('balanced_incremental', -1.), ('balanced_incremental', 0.)):
            cfg = consensus_config()
            cfg.defrost()
            cfg.TRAINER.BiMC.CONSENSUS.AUTO_CALIBRATE = True
            cfg.TRAINER.BiMC.CONSENSUS.OBJECTIVE = objective
            cfg.TRAINER.BiMC.CONSENSUS.MAX_GROUP_DROP_PP = guard
            cfg.freeze()
            report = initialize_consensus(make_model(cfg), cfg, calibration_state())
            self.assertEqual(report['selected_lambda'], report['balanced_safe_lambda'] if guard == 0
                             else report['selection_audit'][objective])
            json.dumps(report, allow_nan=False)
            inspect_report(report)
            reports.append(report)
        for report in reports[1:]:
            self.assertEqual(report['scales'], reports[0]['scales'])
            self.assertEqual(report['protocol_sha256'], reports[0]['protocol_sha256'])
        legacy = reports[0]
        self.assertEqual(legacy['selected_lambda'], min(legacy['candidate_results'], key=lambda row:
                         (-row['correct'], row['lambda']))['lambda'])
        for candidate in legacy['candidate_results']:
            self.assertEqual(candidate['correct'], sum(s['groups']['all']['correct']
                                                       for s in candidate['stages']))
            self.assertEqual(candidate['old_damaged'], sum(s['groups']['old']['damaged']
                                                           for s in candidate['stages']))
            for stage in candidate['stages']:
                groups = stage['groups']
                self.assertEqual(groups['all']['count'], groups['old']['count']+groups['new']['count'])
                self.assertEqual(groups['new']['count'], groups['historical_incremental']['count']
                                 + groups['current_new']['count'])
                for group in groups.values():
                    self.assertEqual(group['correct'] - group['reference_correct'],
                                     group['corrected'] - group['damaged'])
                    self.assertEqual(group['changed'], group['corrected'] + group['damaged']
                                     + group['wrong_to_wrong'])

    def test_disjoint_classes_samples_and_initialization_only_once(self):
        cfg = consensus_config()
        model, state = make_model(cfg), calibration_state()
        snapshot = {key: value.clone() for key, value in state.items() if torch.is_tensor(value)}
        report = initialize_consensus(model, cfg, state)
        fit_ids, val_ids = set(report['fit_class_ids']), set(report['validation_class_ids'])
        self.assertFalse(fit_ids & val_ids)
        self.assertEqual(fit_ids | val_ids, set(range(6)))
        for key, allowed in (('fit_sequences', fit_ids), ('validation_sequences', val_ids)):
            for sequence in report[key]:
                self.assertTrue(set(sequence['class_ids']) <= allowed)
                support = {index for row in sequence['support_indices'] for index in row}
                query = {index for row in sequence['query_indices'] for index in row}
                self.assertFalse(support & query)
        for key, value in snapshot.items():
            self.assertTrue(torch.equal(state[key], value), key)
        with self.assertRaises(RuntimeError):
            initialize_consensus(model, cfg, state)

    def test_query_changes_cannot_change_support_reference(self):
        cfg, state = consensus_config(), calibration_state()
        model = make_model(cfg)
        settings = cfg.TRAINER.BiMC.CONSENSUS
        sequence = sample_sequence(state['images_targets'], list(range(6)), settings,
                                   torch.Generator().manual_seed(7))
        support = [index for row in sequence['support_indices'] for index in row]
        query = [index for row in sequence['query_indices'] for index in row]
        before = build_support_reference(model, cfg, state, support, sequence['class_ids'], 2, 1)
        changed = copy.deepcopy(state)
        changed['images_features'][query] *= -1
        changed['frequency_features'][query] *= -1
        after = build_support_reference(model, cfg, changed, support, sequence['class_ids'], 2, 1)
        self.assertTrue(torch.equal(before[1], after[1]))
        for key, value in before[3].items():
            self.assertTrue(torch.equal(value, after[3][key]), key)

    def test_no_evidence_selects_zero_and_is_reproducible(self):
        cfg = consensus_config()
        cfg.defrost()
        cfg.TRAINER.BiMC.CONSENSUS.AUTO_CALIBRATE = True
        cfg.freeze()
        state = calibration_state()
        state['frequency_features'].zero_()
        first = initialize_consensus(make_model(cfg), cfg, state)
        second = initialize_consensus(make_model(cfg), cfg, state)
        self.assertEqual(first, second)
        self.assertEqual(first['selected_lambda'], 0.)
        self.assertFalse(any(value for row in first['active_sources'] for value in row))

    def test_future_classes_are_rejected(self):
        cfg = consensus_config()
        state = calibration_state()
        state['images_targets'][0] = 8
        with self.assertRaisesRegex(ValueError, 'base-class'):
            initialize_consensus(make_model(cfg), cfg, state)


if __name__ == '__main__':
    unittest.main()
