"""Competition invariants: exact anchors, conditional mixtures and legal inputs."""
import unittest
import torch
from torch.nn import functional as F

from models.competitive_fusion import (CompetitiveFusion, branch_probabilities, fixed_scores,
    competitive_scores, evidence_features, support_quality, FEATURE_DIM)
from engine.prototype_evaluation import identity_calibration


class CompetitiveFusionTest(unittest.TestCase):
    def example(self):
        torch.manual_seed(27)
        p = branch_probabilities({name: torch.randn(12, 6) for name in ('fused', 'cov', 'description')}, identity_calibration())
        quality = torch.rand(6, 3)
        fixed = {'alpha_old': .8, 'alpha_novel': .5, 'novel_bias_delta': .1}
        return p, quality, fixed, identity_calibration()

    def test_zero_blend_and_zero_network_return_exact_anchor_object(self):
        p, q, fixed, cal = self.example()
        model = CompetitiveFusion()
        anchor = fixed_scores(p, 3, fixed, cal)
        with torch.no_grad():
            self.assertIs(competitive_scores(model, p, 3, q, fixed, cal, 0., anchor), anchor)
            self.assertIs(competitive_scores(model, p, 3, q, fixed, cal, 1., anchor), anchor)
        with torch.no_grad():
            model.network[-1].bias.fill_(.5)
        self.assertIs(competitive_scores(model, p, 3, q, fixed, cal, 0., anchor), anchor)

    def test_live_gradient_at_zero_and_no_input_gradient(self):
        p, q, fixed, cal = self.example()
        q.requires_grad_(True)
        model = CompetitiveFusion()
        out = competitive_scores(model, p, 3, q, fixed, cal)
        loss = F.nll_loss((out/out.sum(-1, keepdim=True)).log(), torch.arange(12)%6)
        loss.backward()
        self.assertGreater(float(model.network[-1].weight.grad.abs().sum()), 0.)
        self.assertIsNone(q.grad)

    def test_group_only_preserves_within_group_conditionals(self):
        p, q, fixed, cal = self.example()
        model = CompetitiveFusion(method='group')
        with torch.no_grad():
            model.network[-1].bias.copy_(torch.tensor([.6, .4, -.8]))
            anchor = fixed_scores(p, 3, fixed, cal)
            actual = competitive_scores(model, p, 3, q, fixed, cal)
        for sl in (slice(None, 3), slice(3, None)):
            torch.testing.assert_close(actual[:, sl]/actual[:, sl].sum(-1, keepdim=True),
                                       anchor[:, sl]/anchor[:, sl].sum(-1, keepdim=True))
        torch.testing.assert_close(actual[:, :3], anchor[:, :3])
        self.assertTrue(bool((actual[:, 3:] > anchor[:, 3:]).all()))

    def test_within_group_updates_preserve_group_masses_without_odds_change(self):
        p, q, fixed, cal = self.example()
        model = CompetitiveFusion()
        with torch.no_grad():
            model.network[-1].bias.copy_(torch.tensor([0., .8, -.8]))
            anchor = fixed_scores(p, 3, fixed, cal)
            actual = competitive_scores(model, p, 3, q, fixed, cal)
        self.assertFalse(torch.equal(actual, anchor))
        for sl in (slice(None, 3), slice(3, None)):
            torch.testing.assert_close(actual[:, sl].sum(-1), anchor[:, sl].sum(-1))

    def test_class_permutation_equivariance_and_quality_ablation(self):
        p, q, fixed, cal = self.example()
        model = CompetitiveFusion()
        with torch.no_grad():
            model.network[-1].weight.normal_(std=.1)
            actual = competitive_scores(model, p, 3, q, fixed, cal)
            order = torch.tensor([2, 0, 1, 4, 5, 3])
            permuted = competitive_scores(model, {k: v[:, order] for k, v in p.items()}, 3, q[order], fixed, cal)
        torch.testing.assert_close(actual[:, order], permuted)
        a = evidence_features(p, 3, q, use_support_quality=False)
        b = evidence_features(p, 3, q*10, use_support_quality=False)
        self.assertEqual(a.shape, (12, FEATURE_DIM))
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(a[:, -6:], torch.zeros_like(a[:, -6:])))

    def test_quality_is_support_order_and_feature_rotation_invariant(self):
        torch.manual_seed(19)
        supports = [F.normalize(torch.randn(n, 8), dim=-1) for n in (2, 5)]
        proto = F.normalize(torch.randn(2, 8), dim=-1)
        semantic = F.normalize(torch.randn(2, 8), dim=-1)
        rotate = torch.linalg.qr(torch.randn(8, 8)).Q
        a = support_quality(supports, proto, semantic)
        b = support_quality([v.flip(0)@rotate for v in supports], proto@rotate, semantic@rotate)
        torch.testing.assert_close(a, b)


if __name__ == '__main__':
    unittest.main()
