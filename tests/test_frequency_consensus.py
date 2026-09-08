import unittest

import torch

from models.frequency_consensus import (
    bounded_evidence, fit_margin_scales, pair_candidates, pairwise_margins,
    rerank_frequency_consensus, rerank_with_evidence,
)


class FrequencyConsensusTest(unittest.TestCase):
    def test_zero_and_nonflips_are_exact_original_votes(self):
        votes = torch.tensor([[.4, .35, .25], [.5, .5, .1]], dtype=torch.float16)
        for strength, evidence in ((0., torch.tensor([-1., -1.])),
                                   (.1, torch.tensor([1., 1.]))):
            output, _ = rerank_with_evidence(votes, evidence, strength)
            self.assertIs(output, votes)

    def test_pair_mass_unchanged_third_class_cannot_win_and_bound(self):
        generator = torch.Generator().manual_seed(48)
        votes = torch.rand(256, 20, generator=generator) + .01
        evidence = -torch.ones(256)
        output, detail = rerank_with_evidence(votes, evidence, .2)
        winner = output.argmax(1)
        self.assertTrue(((winner == detail['first']) | (winner == detail['second'])).all())
        self.assertTrue(torch.allclose(output.sum(1), votes.sum(1)))
        self.assertTrue(torch.equal(output[detail['log_ratio'] >= .2], votes[detail['log_ratio'] >= .2]))
        self.assertTrue(detail['changed'].any())

    def test_ties_and_tiny_negative_margin_preserve_intended_second_winner(self):
        votes = torch.tensor([[.5, .5, .5], [.3, .7, .7]])
        first, second, _ = pair_candidates(votes)
        self.assertEqual(first.tolist(), [0, 1])
        self.assertEqual(second.tolist(), [1, 2])
        output, _ = rerank_with_evidence(votes, -torch.ones(2), 1e-9)
        self.assertEqual(output.argmax(1).tolist(), [1, 2])

    def test_disagreement_is_zero_and_one_band_stays_divided_by_three(self):
        visual = torch.tensor([[2., 1., -1.]])
        semantic = torch.tensor([[1., -2., 1.]])
        evidence = bounded_evidence(visual, semantic, torch.ones(2, 3))
        self.assertAlmostEqual(float(evidence), float(torch.tanh(torch.tensor(1.)) / 3), places=6)
        self.assertTrue((evidence.abs() <= 1).all())

    def test_degenerate_scales_disable_evidence(self):
        zeros = torch.zeros(4, 3)
        scales = fit_margin_scales(zeros, zeros)
        self.assertEqual(int(scales.count_nonzero()), 0)
        for mode in ('visual', 'semantic', 'average', 'consensus'):
            result = bounded_evidence(torch.ones(4, 3), -torch.ones(4, 3), scales, mode)
            self.assertEqual(int(result.count_nonzero()), 0)

    def test_semantic_permutation_changes_only_semantic_margins(self):
        votes = torch.tensor([[.6, .4]])
        query = torch.eye(3).unsqueeze(0)
        visual = torch.stack((torch.eye(3), -torch.eye(3)))
        dv, dt = pairwise_margins(votes, query, visual, visual)
        dv2, dt2 = pairwise_margins(votes, query, visual, visual, [1, 2, 0])
        self.assertTrue(torch.equal(dv, dv2))
        self.assertFalse(torch.equal(dt, dt2))
        self.assertEqual(int(dt2.count_nonzero()), 0)

    def test_reject_logits_invalid_scales_and_permutations(self):
        with self.assertRaises(ValueError):
            pair_candidates(torch.tensor([[0., -1.]]))
        with self.assertRaises(ValueError):
            bounded_evidence(torch.ones(1, 3), torch.ones(1, 3), -torch.ones(2, 3))
        with self.assertRaises(ValueError):
            rerank_frequency_consensus(torch.ones(1, 2), torch.ones(1, 3, 4),
                                       torch.ones(2, 3, 4), torch.ones(2, 3, 4),
                                       torch.ones(2, 3), .1, permutation=[0, 0, 1])


if __name__ == '__main__':
    unittest.main()
