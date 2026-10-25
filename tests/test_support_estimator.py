"""Geometry, gradient isolation and joint-arrival symmetry for the estimator."""
import unittest

import torch
from torch.nn import functional as F

from models.support_estimator import SupportPrototypeEstimator, correct_current


class SupportPrototypeEstimatorTest(unittest.TestCase):
    def sample(self, dimension=9, count=5):
        rng = torch.Generator().manual_seed(39 + dimension)
        support = F.normalize(torch.randn(count, dimension, generator=rng), dim=-1)
        reference = F.normalize(support.mean(0) + .1 * torch.randn(dimension, generator=rng), dim=-1)
        text = F.normalize(torch.randn(dimension, generator=rng), dim=-1)
        competitors = F.normalize(torch.randn(4, dimension, generator=rng), dim=-1)
        return support, reference, text, competitors

    def nonzero_model(self, **kwargs):
        torch.manual_seed(67)
        model = SupportPrototypeEstimator(**kwargs)
        with torch.no_grad():
            model.sample_head[-1].weight.fill_(.7)
            model.class_head[-1].weight.fill_(.4)
            model.class_head[-1].bias.fill_(.5)
        return model

    def test_initial_zero_delta_has_live_gradients_and_no_feature_gradients(self):
        support, ref, text, competitors = [v.requires_grad_() for v in self.sample()]
        model = SupportPrototypeEstimator()
        output, details = model(support, ref, text, competitors, return_details=True)
        self.assertTrue(torch.equal(details['delta'], torch.zeros_like(details['delta'])))
        torch.testing.assert_close(output, ref.detach(), rtol=1e-6, atol=1e-7)
        loss = 1 - output @ text.detach()
        loss.backward()
        self.assertGreater(float(model.class_head[-1].weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.sample_head[-1].weight.grad.abs().sum()), 0.)
        for feature in (support, ref, text, competitors):
            self.assertIsNone(feature.grad)

    def test_support_and_competitor_permutations_preserve_nonzero_estimate(self):
        support, ref, text, competitors = self.sample()
        model = self.nonzero_model()
        actual = model(support, ref, text, competitors)
        permuted = model(support[torch.tensor([3, 0, 4, 1, 2])], ref, text, competitors.flip(0))
        torch.testing.assert_close(permuted, actual, rtol=1e-6, atol=1e-7)
        self.assertGreater(float((actual - ref).norm()), 1e-4)

    def test_dimensions_shots_norm_and_movement_bound(self):
        model = self.nonzero_model(max_delta=.12)
        for dimension, count in ((3, 1), (9, 5), (23, 12)):
            with self.subTest(dimension=dimension, count=count):
                values = self.sample(dimension, count)
                output, detail = model(*values, return_details=True)
                self.assertEqual(output.shape, (dimension,))
                self.assertTrue(torch.isfinite(output).all())
                self.assertAlmostEqual(float(output.norm()), 1., places=6)
                self.assertLessEqual(float(detail['delta'].norm()), .1200001)
                # Geometric worst-case bound after normalizing unit ref+delta.
                cosine = (output @ values[1]).clamp(-1, 1)
                self.assertLessEqual(float(torch.acos(cosine)), float(torch.asin(torch.tensor(.12))) + 1e-5)

    def test_disabled_competitors_cannot_change_estimate(self):
        model = self.nonzero_model(use_competitors=False)
        support, ref, text, competitors = self.sample()
        a = model(support, ref, text, competitors)
        b = model(support, ref, text, -competitors)
        c = model(support, ref, text, None)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(a, c))

    def test_common_feature_rotation_rotates_output_without_changing_rule(self):
        support, ref, text, competitors = self.sample()
        model = self.nonzero_model()
        rotation = torch.linalg.qr(torch.randn(9, 9, generator=torch.Generator().manual_seed(19)))[0]
        expected = model(support, ref, text, competitors) @ rotation
        actual = model(support @ rotation, ref @ rotation, text @ rotation, competitors @ rotation)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=2e-7)

    def test_joint_arrival_class_permutation_and_no_mutation(self):
        model = self.nonzero_model()
        values = [self.sample(9, count) for count in (3, 5, 7)]
        supports = [v[0] for v in values]
        ref = torch.stack([v[1] for v in values])
        text = torch.stack([v[2] for v in values])
        past = self.sample()[3]
        copies = [v.clone() for v in supports + [ref, text, past]]
        output = correct_current(model, supports, ref, text, past, blend=.7)
        order = torch.tensor([2, 0, 1])
        permutation = correct_current(model, [supports[i] for i in order], ref[order], text[order], past, blend=.7)
        torch.testing.assert_close(permutation, output[order], rtol=1e-6, atol=1e-7)
        for actual, copy in zip(supports + [ref, text, past], copies):
            self.assertTrue(torch.equal(actual, copy))

    def test_zero_blend_returns_original_object_without_support_inspection(self):
        model = self.nonzero_model()
        reference = torch.eye(4)[:2]
        self.assertIs(correct_current(model, None, reference, None, blend=0), reference)

    def test_invalid_values_shapes_and_configurations_rejected(self):
        for kwargs in ({'hidden': 0}, {'max_delta': 0}, {'max_delta': 1},
                       {'max_delta': float('nan')}, {'use_competitors': 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SupportPrototypeEstimator(**kwargs)
        model = SupportPrototypeEstimator()
        support, ref, text, competitors = self.sample()
        for values in ((support[:0], ref, text, competitors),
                       (support, torch.zeros_like(ref), text, competitors),
                       (support, ref, text[:-1], competitors),
                       (support, ref, text, competitors[:, :-1]),
                       (support * float('nan'), ref, text, competitors)):
            with self.assertRaises(ValueError):
                model(*values)
        with self.assertRaises(ValueError):
            correct_current(model, [support], ref[None], text[None], blend=1.1)


if __name__ == '__main__':
    unittest.main()
