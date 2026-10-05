import unittest

import torch
from torch.nn import functional as F

from models.dual_frequency import (
    align_semantics, apply_evidence, frequency_evidence, pair_margins, remove_global, support_statistics,
)


class DualFrequencyTest(unittest.TestCase):
    def example(self):
        generator = torch.Generator().manual_seed(71)
        global_features = F.normalize(torch.randn(3, 12, generator=generator), dim=-1)
        bands = F.normalize(torch.randn(3, 3, 12, generator=generator), dim=-1)
        means, _ = remove_global(bands, global_features)
        stats = {'mean': means * .9, 'count': torch.tensor([20, 5, 5])}
        candidates = F.normalize(.8 * global_features[:, None, None] + .6 * means[:, :, None], dim=-1)
        aligned = align_semantics(stats, candidates, global_features)
        return global_features, bands, stats, candidates, aligned

    def test_projection_is_orthogonal_and_identical_view_has_zero_evidence(self):
        original, bands, _, _, _ = self.example()
        residual, _ = remove_global(bands, original)
        self.assertLess(float((residual * original[:, None]).sum(-1).abs().max()), 1e-6)
        repeated, _ = remove_global(original[:, None].expand(-1, 3, -1), original)
        self.assertEqual(int(repeated.count_nonzero()), 0)
        ordinary, energy = remove_global(original[:, None].expand(-1, 3, -1), original, False)
        self.assertTrue(torch.allclose(ordinary, original[:, None].expand(-1, 3, -1)))
        self.assertTrue(torch.allclose(energy, torch.ones_like(energy)))

    def test_alignment_selects_discriminative_candidate_and_attenuates_few_shot(self):
        original, _, stats, candidates, _ = self.example()
        wrong = candidates.roll(1, 0)
        options = torch.cat((wrong, candidates), 2)
        aligned = align_semantics(stats, options, original, top_k=1)
        self.assertTrue((aligned['selected_indices'] == 1).all())
        self.assertTrue((aligned['class_margin'] > 0).all())
        self.assertTrue((aligned['reliability'] >= 0).all())
        self.assertTrue((aligned['reliability'] <= 1).all())
        fewer = align_semantics({**stats, 'count': torch.ones(3)}, options, original, top_k=1)
        self.assertTrue((fewer['reliability'] < aligned['reliability']).all())

    def test_consensus_disagreement_and_weak_reliability_cannot_be_amplified(self):
        visual = torch.ones(4, 3)
        scales = torch.ones(2, 3)
        self.assertTrue(torch.equal(frequency_evidence(visual, -visual, visual, scales), torch.zeros(4)))
        strong = frequency_evidence(visual, visual, visual, scales)
        weak = frequency_evidence(visual, visual, visual * .2, scales)
        self.assertTrue(torch.allclose(weak, .2 * strong))
        self.assertTrue(torch.equal(frequency_evidence(visual, visual, visual, scales * 0), torch.zeros(4)))

    def test_zero_identity_top_two_bound_and_probability_mass(self):
        original, bands, _, _, aligned = self.example()
        scores = torch.tensor([[.36, .35, .29], [.98, .01, .01]])
        zero, _ = apply_evidence(scores, original[:2], bands[:2], aligned, torch.ones(2, 3), 0.)
        self.assertIs(zero, scores)
        # Deliberately make the ambiguous pair's second class the visual/text target.
        query = aligned['visual'][1:2].expand(2, -1, -1)
        output, detail = apply_evidence(scores, original[:2], query, aligned,
                                        torch.ones(2, 3) * .05, 1., residualize=False)
        self.assertEqual(int(output[0].argmax()), 1)
        self.assertTrue(torch.equal(output[1], scores[1]))
        self.assertTrue(torch.equal(output[:, 2], scores[:, 2]))
        self.assertTrue(torch.allclose(output.sum(1), scores.sum(1)))
        self.assertTrue((detail['evidence'].abs() <= 1).all())

    def test_query_batch_composition_does_not_change_scores_or_support(self):
        original, bands, stats, _, aligned = self.example()
        snapshots = {k: v.clone() for k, v in aligned.items()}
        scores = torch.tensor([[.36, .35, .29], [.35, .36, .29], [.4, .3, .3]])
        all_output, _ = apply_evidence(scores, original, bands, aligned, torch.ones(2, 3), .6)
        each = torch.cat([apply_evidence(scores[i:i+1], original[i:i+1], bands[i:i+1], aligned,
                                         torch.ones(2, 3), .6)[0] for i in range(3)])
        self.assertTrue(torch.allclose(all_output, each, atol=1e-7))
        for key, value in snapshots.items():
            self.assertTrue(torch.equal(value, aligned[key]))
        self.assertEqual(stats['count'].tolist(), [20, 5, 5])

    def test_visual_only_evidence_does_not_depend_on_text_candidates(self):
        original, bands, stats, candidates, aligned = self.example()
        different = align_semantics(stats, -candidates, original)
        scores = torch.tensor([[.36, .35, .29], [.35, .36, .29], [.4, .3, .3]])
        first = pair_margins(scores, original, bands, aligned, mode='visual')
        second = pair_margins(scores, original, bands, different, mode='visual')
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[2], second[2]))
        self.assertTrue(torch.equal(frequency_evidence(*first, torch.ones(2, 3), 'visual'),
                                    frequency_evidence(*second, torch.ones(2, 3), 'visual')))

    def test_one_shot_zero_residual_statistics_remain_finite(self):
        original, _, _, _, _ = self.example()
        stats = support_statistics(original, original[:, None].expand(-1, 3, -1),
                                   torch.arange(3), [2, 0, 1])
        self.assertTrue(torch.equal(stats['count'], torch.ones(3, dtype=torch.long)))
        self.assertEqual(int(stats['mean'].count_nonzero()), 0)
        self.assertEqual(int(stats['variance'].count_nonzero()), 0)

    def test_invalid_settings_are_rejected(self):
        original, bands, stats, candidates, _ = self.example()
        with self.assertRaises(ValueError):
            align_semantics(stats, candidates, original, permutation=[0, 0, 2])
        with self.assertRaises(ValueError):
            align_semantics(stats, candidates, original, temperature=0.)
        with self.assertRaises(ValueError):
            remove_global(bands * float('nan'), original)
        with self.assertRaises(ValueError):
            frequency_evidence(torch.ones(1, 3), torch.ones(1, 3), torch.ones(1, 3),
                               torch.ones(2, 3), bands=[0., 0., 0.])


if __name__ == '__main__':
    unittest.main()
