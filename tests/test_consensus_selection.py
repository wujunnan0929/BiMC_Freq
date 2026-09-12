import copy
import unittest

from utils.consensus_selection import score_candidate, select_lambda, summarize_counts
from tools.analyze_consensus_calibration import inspect_report


def group(n, reference, corrected=0, damaged=0):
    return summarize_counts(dict(count=n, reference_correct=reference,
                                 correct=reference+corrected-damaged,
                                 corrected=corrected, damaged=damaged,
                                 eligible=corrected+damaged, top2_hits=n,
                                 changed=corrected+damaged, wrong_to_wrong=0))


def stage(index, old, new):
    counts = {key: old[key] + new[key] for key in
              ('count', 'reference_correct', 'correct', 'corrected', 'damaged',
               'eligible', 'top2_hits', 'changed', 'wrong_to_wrong')}
    return {'stage': index, 'groups': {'old': old, 'new': new, 'all': summarize_counts(counts)}}


class SelectionTest(unittest.TestCase):
    def candidates(self):
        # Legacy selects a base-only gain that costs accuracy at increment stage 1.
        return [
            {'lambda': 0., 'stages': [stage(0, group(100, 70), group(0, 0)),
                                      stage(1, group(100, 70), group(10, 7))]},
            {'lambda': .1, 'stages': [stage(0, group(100, 70, 10), group(0, 0)),
                                      stage(1, group(100, 70, damaged=1), group(10, 7))]},
            {'lambda': .03, 'stages': [stage(0, group(100, 70), group(0, 0)),
                                       stage(1, group(100, 70, damaged=2), group(10, 7, 1))]},
        ]

    def test_stage0_exclusion_and_group_balancing_change_choice(self):
        candidates = self.candidates()
        self.assertEqual(select_lambda(candidates, 'micro_all'), .1)
        self.assertEqual(select_lambda(candidates, 'micro_incremental'), 0.)
        self.assertEqual(select_lambda(candidates, 'balanced_incremental'), .03)
        self.assertEqual(select_lambda(candidates, 'balanced_incremental', 0.), 0.)
        self.assertEqual(select_lambda(candidates, 'balanced_incremental', 2.), .03)

    def test_legacy_matches_pooled_correct_and_smallest_tie(self):
        candidates = self.candidates()
        expected = min(candidates, key=lambda c: (
            -sum(s['groups']['all']['correct'] for s in c['stages']), c['lambda']))['lambda']
        self.assertEqual(select_lambda(candidates, 'micro_all'), expected)
        tied = copy.deepcopy(candidates[0])
        tied['lambda'] = .001
        self.assertEqual(select_lambda([tied, candidates[0]], 'micro_all'), 0.)

    def test_balanced_weights_stages_equally_and_guard_is_per_stage(self):
        stages = [stage(1, group(100, 70, 2), group(10, 7)),
                  stage(2, group(100, 70, damaged=1), group(100, 70, 10))]
        score = score_candidate(stages, 'balanced_incremental', 0.)
        self.assertAlmostEqual(score['score'], ((72+70)/2 + (69+80)/2)/2)
        self.assertFalse(score['feasible'])  # pooled old gain must not hide stage 2 loss
        self.assertEqual(score['constraint_violations'][0]['stage'], 2)

    def test_bad_objective_empty_group_and_invalid_guard_rejected(self):
        for objective, guard in (('test_accuracy', -1.), ('micro_all', -2.),
                                 ('micro_all', float('nan'))):
            with self.assertRaises(ValueError):
                score_candidate(self.candidates()[0]['stages'], objective, guard)
        with self.assertRaises(ValueError):
            score_candidate([stage(1, group(2, 1), group(0, 0))], 'balanced_incremental')

    def test_legacy_diagnostic_uses_saved_query_counts_not_default_assumptions(self):
        report = {'data_source': 'base_training_only', 'schema_version': 1, 'seed': 9102,
                  'selected_lambda': .1, 'candidate_class_counts': [2, 3],
                  'validation_sequences': [{'query_indices': [[1, 2], [3, 4], [5, 6]]}],
                  'validation_query_occurrences': 10,
                  'candidate_results': [{'lambda': .1, 'accuracy': 70, 'corrected': 2, 'damaged': 1,
                                         'old_corrected': 2, 'old_damaged': 0,
                                         'new_corrected': 0, 'new_damaged': 1}]}
        result = inspect_report(report)
        self.assertEqual(result['old_query_fraction_all_stages'], .8)
        self.assertEqual(result['candidate_results'][0]['new_delta_pp'], -50.)
        self.assertIsNone(result['selection_audit'])
        report['validation_query_occurrences'] = 11
        with self.assertRaises(ValueError):
            inspect_report(report)


if __name__ == '__main__':
    unittest.main()
