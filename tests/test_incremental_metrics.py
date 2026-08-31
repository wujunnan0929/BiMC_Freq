import json
import unittest

import numpy as np

from utils.incremental_metrics import (
    add_forgetting,
    compute_forgetting,
    compute_session_metrics,
    summarize_sessions,
)


GROUPS = [[0, 1], [2, 3], [4, 5]]


def scores_from_predictions(predictions, num_classes):
    scores = np.full((len(predictions), num_classes), -2.0)
    scores[np.arange(len(predictions)), predictions] = 2.0
    return scores


class IncrementalSessionMetricsTest(unittest.TestCase):
    def test_base_session_uses_json_null_for_incremental_metrics(self):
        targets = np.array([0, 0, 0, 1, 1, 1])
        predictions = np.array([0, 0, 1, 1, 0, 0])
        result = compute_session_metrics(
            scores_from_predictions(predictions, 2), targets, 0, GROUPS,
        )
        self.assertEqual(result['session'], 0)
        self.assertAlmostEqual(result['accuracy'], 50.0)
        self.assertEqual(result['task_acc'], [50.0])
        self.assertAlmostEqual(result['class_macro_accuracy'], 50.0)
        for key in ('novel_accuracy', 'harmonic_accuracy',
                    'current_novel_accuracy', 'old_to_new_rate',
                    'inc_session_mean_accuracy'):
            self.assertIsNone(result[key])
        self.assertIsInstance(json.dumps(result), str)

    def test_one_increment_session_metrics_have_expected_denominators(self):
        targets = np.repeat(np.arange(4), 2)
        predictions = np.array([0, 2, 1, 1, 2, 2, 0, 3])
        result = compute_session_metrics(
            scores_from_predictions(predictions, 4), targets, 1, GROUPS,
        )
        self.assertEqual(result['accuracy'], 75.0)
        self.assertEqual(result['base_accuracy'], 75.0)
        self.assertEqual(result['novel_accuracy'], 75.0)
        self.assertEqual(result['harmonic_accuracy'], 75.0)
        self.assertEqual(result['current_novel_accuracy'], 75.0)
        # One out of all four historical samples is assigned to the current classes.
        self.assertEqual(result['old_to_new_rate'], 25.0)
        self.assertEqual(result['task_acc'], [75.0, 75.0])
        self.assertEqual(result['class_macro_accuracy'], 75.0)
        self.assertEqual(result['inc_session_mean_accuracy'], 75.0)

    def test_novel_is_micro_over_all_incremental_samples(self):
        targets = np.array(
            [0, 0, 1, 1] + [2] * 6 + [3] * 2 + [4] * 2 + [5] * 2,
        )
        predictions = np.array(
            [0, 4, 1, 0] + [2] * 6 + [5] * 2 + [3] * 2 + [5, 4],
        )
        result = compute_session_metrics(
            scores_from_predictions(predictions, 6), targets, 2, GROUPS,
        )
        self.assertEqual(result['accuracy'], 56.25)
        self.assertEqual(result['base_accuracy'], 50.0)
        self.assertAlmostEqual(result['novel_accuracy'], 7 / 12 * 100.0)
        self.assertAlmostEqual(
            result['harmonic_accuracy'],
            2 * 50.0 * (7 / 12 * 100.0) / (50.0 + 7 / 12 * 100.0),
        )
        self.assertEqual(result['current_novel_accuracy'], 25.0)
        self.assertEqual(result['old_to_new_rate'], 25.0)
        self.assertEqual(result['task_acc'], [50.0, 75.0, 25.0])
        self.assertEqual(result['inc_session_mean_accuracy'], 50.0)
        self.assertAlmostEqual(result['class_macro_accuracy'], 250.0 / 6)

    def test_task_accuracy_keeps_full_precision(self):
        targets = np.array([0, 0, 0, 1, 1, 1])
        predictions = np.array([0, 0, 1, 1, 1, 0])
        result = compute_session_metrics(
            scores_from_predictions(predictions, 2), targets, 0, GROUPS,
        )
        self.assertEqual(result['task_acc'][0], 4 / 6 * 100.0)

    def test_zero_base_and_novel_accuracy_has_zero_harmonic(self):
        targets = np.repeat(np.arange(4), 2)
        predictions = np.array([1, 1, 0, 0, 3, 3, 2, 2])
        result = compute_session_metrics(
            scores_from_predictions(predictions, 4), targets, 1, GROUPS,
        )
        self.assertEqual(result['base_accuracy'], 0.0)
        self.assertEqual(result['novel_accuracy'], 0.0)
        self.assertEqual(result['harmonic_accuracy'], 0.0)

    def test_invalid_shapes_indices_empty_values_and_groups_are_rejected(self):
        valid_scores = np.eye(4)
        valid_targets = np.arange(4)
        invalid_calls = {
            'score rank': (valid_scores[0], valid_targets, 1, GROUPS),
            'target rank': (valid_scores, valid_targets[:, None], 1, GROUPS),
            'length': (valid_scores[:3], valid_targets, 1, GROUPS),
            'empty': (np.empty((0, 4)), np.empty(0, dtype=int), 1, GROUPS),
            'column count': (np.eye(5)[:4], valid_targets, 1, GROUPS),
            'nan': (np.where(np.eye(4) == 1, np.nan, 0), valid_targets, 1, GROUPS),
            'inf': (np.where(np.eye(4) == 1, np.inf, 0), valid_targets, 1, GROUPS),
            'float target': (valid_scores, valid_targets.astype(float), 1, GROUPS),
            'unseen target': (valid_scores, np.array([0, 1, 2, 4]), 1, GROUPS),
            'bad task': (valid_scores, valid_targets, 3, GROUPS),
            'bool task': (valid_scores, valid_targets, True, GROUPS),
            'duplicate group': (valid_scores, valid_targets, 1,
                                [[0, 1], [1, 2], [3, 4]]),
            'class index gap': (valid_scores, valid_targets, 1,
                                [[0, 2], [3, 4], [5, 6]]),
        }
        for name, arguments in invalid_calls.items():
            with self.subTest(name=name), self.assertRaises((TypeError, ValueError)):
                compute_session_metrics(*arguments)

    def test_missing_seen_class_data_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'no samples'):
            compute_session_metrics(
                scores_from_predictions([0, 1, 2], 4), np.array([0, 1, 2]),
                1, GROUPS,
            )


class ForgettingAndSummaryTest(unittest.TestCase):
    def setUp(self):
        self.sessions = [
            {
                'session': 0, 'accuracy': 80.0, 'base_accuracy': 80.0,
                'novel_accuracy': None, 'harmonic_accuracy': None,
                'old_to_new_rate': None, 'task_acc': [80.0],
            },
            {
                'session': 1, 'accuracy': 82.0, 'base_accuracy': 90.0,
                'novel_accuracy': 70.0, 'harmonic_accuracy': 78.75,
                'old_to_new_rate': 10.0, 'task_acc': [90.0, 70.0],
            },
            {
                'session': 2, 'accuracy': 75.0, 'base_accuracy': 75.0,
                'novel_accuracy': 68.0, 'harmonic_accuracy': 71.33,
                'old_to_new_rate': 12.5, 'task_acc': [75.0, 65.0, 85.0],
            },
        ]

    def test_forgetting_uses_each_task_best_prior_accuracy(self):
        # task 0: max(80, 90) - 75 = 15; task 1: 70 - 65 = 5.
        self.assertEqual(
            compute_forgetting(self.sessions[-1], self.sessions[:-1]), 10.0,
        )
        enriched = add_forgetting(self.sessions[-1], self.sessions[:-1])
        self.assertEqual(enriched['forgetting'], 10.0)
        self.assertNotIn('forgetting', self.sessions[-1])

    def test_forgetting_can_be_negative_when_old_tasks_improve(self):
        self.assertEqual(
            compute_forgetting([95.0, 75.0, 85.0], self.sessions[:-1]), -5.0,
        )

    def test_base_session_forgetting_is_null(self):
        self.assertIsNone(compute_forgetting([80.0], []))
        self.assertIsNone(add_forgetting(self.sessions[0], [])['forgetting'])

    def test_summary_uses_final_metrics_and_average_all_session_accuracy(self):
        summary = summarize_sessions(self.sessions)
        self.assertEqual(summary, {
            'final_accuracy': 75.0,
            'average_accuracy': 79.0,
            'base_accuracy': 75.0,
            'novel_accuracy': 68.0,
            'harmonic_accuracy': 71.33,
            'old_to_new_rate': 12.5,
            'forgetting': 10.0,
        })
        self.assertIsInstance(json.dumps(summary), str)

    def test_single_base_session_summary_has_null_incremental_fields(self):
        summary = summarize_sessions(self.sessions[:1])
        self.assertIsNone(summary['novel_accuracy'])
        self.assertIsNone(summary['harmonic_accuracy'])
        self.assertIsNone(summary['old_to_new_rate'])
        self.assertIsNone(summary['forgetting'])

    def test_summary_and_forgetting_validate_history_and_nonfinite_metrics(self):
        with self.assertRaisesRegex(ValueError, 'history'):
            compute_forgetting([70.0, 80.0], [])
        with self.assertRaisesRegex(ValueError, 'entries'):
            compute_forgetting([70.0, np.nan], self.sessions[:1])
        broken = [dict(row) for row in self.sessions]
        broken[2]['accuracy'] = np.inf
        with self.assertRaisesRegex(ValueError, 'finite'):
            summarize_sessions(broken)
        broken = [dict(row) for row in self.sessions]
        broken[1]['session'] = 3
        with self.assertRaisesRegex(ValueError, 'ordered'):
            summarize_sessions(broken)
        with self.assertRaisesRegex(ValueError, 'non-empty'):
            summarize_sessions([])


if __name__ == '__main__':
    unittest.main()
