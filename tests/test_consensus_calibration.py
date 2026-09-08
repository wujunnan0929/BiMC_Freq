import copy
import unittest

import torch
from torch.nn import functional as F

from engine.consensus_calibration import (
    build_support_reference, initialize_consensus, sample_sequence,
)
from test_consensus_integration import consensus_config
from test_residual_integration import make_model, synthetic_state


def calibration_state():
    state = synthetic_state()
    state['frequency_consensus_text_proto'] = F.normalize(
        state['frequency_description_candidates'].mean(2), dim=-1,
    )
    return state


class ConsensusCalibrationTest(unittest.TestCase):
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
