import unittest

import numpy as np

from utils.consensus_metrics import prediction_diagnostics


class ConsensusMetricsTest(unittest.TestCase):
    def test_correction_damage_identity_and_disjoint_groups(self):
        reference = np.array([[.5, .4, .1, .0], [.4, .5, .1, .0],
                              [.1, .2, .4, .3], [.1, .4, .2, .3]])
        output = np.array([[.4, .5, .1, .0], [.5, .4, .1, .0],
                           [.1, .2, .3, .4], [.1, .3, .2, .4]])
        target = np.array([1, 1, 3, 2])
        report = prediction_diagnostics(
            reference, output, target, 2, [np.array([0, 1]), np.array([2]), np.array([3])],
            eligible=np.ones(4, dtype=bool), encoded=np.array([True, True, True, False]),
        )
        all_rows = report['all']
        self.assertEqual((all_rows['corrected'], all_rows['damaged'],
                          all_rows['wrong_to_wrong']), (2, 1, 1))
        self.assertEqual(all_rows['delta_accuracy_pp'], 25.)
        self.assertEqual(all_rows['accuracy'] - all_rows['reference_accuracy'], 25.)
        self.assertEqual(all_rows['top2_recall'], 75.)
        self.assertEqual(report['prediction_outside_reference_top2_count'], 0)
        self.assertEqual([report[k]['count'] for k in
                          ('base', 'historical_incremental', 'current_new')], [2, 1, 1])
        self.assertEqual(report['damaged_prediction_destinations']['base']['base'], 1)
        for source, destinations in report['damaged_prediction_destinations'].items():
            self.assertEqual(sum(destinations.values()), report[source]['damaged'])

    def test_incremental_damage_destination_distinguishes_base_and_other_incremental(self):
        reference = np.array([[.1, .2, .5, .3], [.1, .4, .2, .5]])
        output = np.array([[.1, .2, .3, .5], [.1, .5, .2, .4]])
        report = prediction_diagnostics(reference, output, np.array([2, 3]), 2,
                                        [np.array([0, 1]), np.array([2]), np.array([3])])
        flow = report['damaged_prediction_destinations']
        self.assertEqual(flow['historical_incremental']['current_new'], 1)
        self.assertEqual(flow['current_new']['base'], 1)
        self.assertEqual(report['historical_incremental_to_base_output_rate'], 0.)
        self.assertEqual(report['current_new_to_base_output_rate'], 100.)

    def test_empty_incremental_groups_and_ties(self):
        scores = np.array([[.5, .5, .1], [.4, .3, .3]])
        report = prediction_diagnostics(scores, scores, np.array([1, 2]), 0, [np.arange(3)])
        self.assertEqual(report['all']['top2_recall'], 50.)
        self.assertEqual(report['all']['changed'], 0)
        self.assertIsNone(report['current_new']['accuracy'])
        self.assertEqual(report['current_new']['count'], 0)


if __name__ == '__main__':
    unittest.main()
