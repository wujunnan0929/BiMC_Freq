"""Analytic and fitting-isolation checks for joint prototype completion."""

import copy
import unittest

import torch

from models.prototype_completion import fit_completion, complete_prototypes


def scalar_state(views=1, dimension=1, between=None, within=None):
    basis = torch.zeros(views, dimension, 1, dtype=torch.float64)
    basis[:, 0, 0] = 1
    return {'basis': basis, 'offset': torch.zeros(views, 1, dtype=torch.float64),
            'between': torch.eye(views, dtype=torch.float64) if between is None else torch.tensor(between, dtype=torch.float64),
            'within': torch.eye(views, dtype=torch.float64) if within is None else torch.tensor(within, dtype=torch.float64),
            'views': views, 'feature_dim': dimension, 'rank': 1}


def fitting_data(views=2):
    generator = torch.Generator().manual_seed(29)
    anchors = torch.randn(6, views, 4, generator=generator, dtype=torch.float64)
    errors = torch.randn(6, 1, 4, generator=generator, dtype=torch.float64) * .4
    noise = torch.randn(6, 8, 1, 4, generator=generator, dtype=torch.float64) * .2
    independent = torch.randn(6, 8, views, 4, generator=generator, dtype=torch.float64) * .03
    x = anchors[:, None] + errors[:, None] + noise + independent
    return x.flatten(0, 1), torch.arange(6).repeat_interleave(8), anchors


class PrototypeCompletionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_scalar_posterior_and_unmodelled_original_components(self):
        state = scalar_state(dimension=2, between=[[.25]], within=[[1.]])
        x = torch.tensor([[[1., 9.]]], requires_grad=True)
        anchors = torch.tensor([[[0., 7.]]], requires_grad=True)
        result = complete_prototypes(state, x, [5], anchors, [5])
        torch.testing.assert_close(result, torch.tensor([[.2, 9.]]))
        self.assertFalse(result.requires_grad)
        # Four observations reduce sampling variance from 1 to 1/4.
        many = complete_prototypes(state, x.detach().repeat(4, 1, 1), [5] * 4, anchors, [5])
        torch.testing.assert_close(many, torch.tensor([[.5, 9.]]))

    def test_correlated_observation_noise_has_analytic_cross_view_effect(self):
        state = scalar_state(2, between=[[1., 0.], [0., 1.]], within=[[1., .8], [.8, 1.]])
        x = torch.tensor([[[1.], [2.]]], dtype=torch.float64)
        anchor = torch.zeros(1, 2, 1, dtype=torch.float64)
        result = complete_prototypes(state, x, [2], anchor, [2])
        # (I + W)^-1 [1,2] has first element (2 - .8*2)/(4-.8^2).
        self.assertAlmostEqual(result.item(), .4 / 3.36, places=12)
        changed = complete_prototypes(state, torch.tensor([[[1.], [-2.]]]), [2], anchor, [2])
        self.assertGreater(changed.item(), result.item())
        independent = scalar_state(2)
        torch.testing.assert_close(complete_prototypes(independent, x, [2], anchor, [2]), torch.tensor([[.5]], dtype=torch.float64))

    def test_fitting_corrects_finite_class_mean_noise_and_balances_classes(self):
        x = torch.tensor([[[0.]], [[2.]], [[4.]], [[6.]]], dtype=torch.float64)
        anchors = torch.zeros(2, 1, 1, dtype=torch.float64)
        model = fit_completion(x, [4, 4, 9, 9], anchors, [4, 9], rank=1, shrinkage=0, ridge=1e-8)
        # Means 1,5; residual covariance 8. Pooled W=2 and mean(1/n)=1/2.
        self.assertAlmostEqual(model['offset'].abs().item(), 3., places=12)
        self.assertAlmostEqual(model['within'].item(), 2. + 1e-8, places=10)
        self.assertAlmostEqual(model['between'].item(), 7. + 1e-8, places=10)
        unequal = torch.tensor([[[0.]], [[2.]], [[4.]], [[5.]], [[6.]]])
        balanced = fit_completion(unequal, [4, 4, 9, 9, 9], anchors, [4, 9], rank=1)
        self.assertAlmostEqual(balanced['offset'].abs().item(), 3., places=12)

    def test_unrequested_class_rows_do_not_enter_basis_or_statistics(self):
        x, labels, anchors = fitting_data()
        state = fit_completion(x, labels, anchors, list(range(6)), rank=3)
        extra = torch.full((10, 2, 4), 10000., dtype=x.dtype)
        other = fit_completion(torch.cat((x, extra)), torch.cat((labels, torch.full((10,), 99))), anchors, list(range(6)), rank=3)
        for key in state:
            with self.subTest(key=key):
                if isinstance(state[key], torch.Tensor):
                    torch.testing.assert_close(state[key], other[key], atol=0, rtol=0)
                else:
                    self.assertEqual(state[key], other[key])

    def test_cross_view_control_zeroes_blocks_and_removes_auxiliary_effect(self):
        x, labels, anchors = fitting_data()
        model = fit_completion(x, labels, anchors, list(range(6)), rank=3, cross_view=False)
        for key in ('between', 'within'):
            self.assertEqual(torch.count_nonzero(model[key][:3, 3:]).item(), 0)
            self.assertEqual(torch.count_nonzero(model[key][3:, :3]).item(), 0)
        support = x[:2].clone()
        before = complete_prototypes(model, support, [8, 8], anchors[:1], [8])
        support[:, 1] += 20
        after = complete_prototypes(model, support, [8, 8], anchors[:1], [8])
        torch.testing.assert_close(before, after, atol=0, rtol=0)

    def test_duplicate_views_do_not_create_independent_sampling_evidence(self):
        x, labels, anchors = fitting_data(views=1)
        single = fit_completion(x, labels, anchors, list(range(6)), rank=4, shrinkage=0, ridge=1e-10)
        repeated = fit_completion(x.repeat(1, 2, 1), labels, anchors.repeat(1, 2, 1), list(range(6)), rank=4, shrinkage=0, ridge=1e-10)
        a = complete_prototypes(single, x[:3], [20] * 3, anchors[:1], [20])
        b = complete_prototypes(repeated, x[:3].repeat(1, 2, 1), [20] * 3, anchors[:1].repeat(1, 2, 1), [20])
        torch.testing.assert_close(a, b, atol=2e-8, rtol=2e-8)
        # Sampling covariance explicitly records the positive cross-view terms.
        torch.testing.assert_close(repeated['within'][:4, 4:], repeated['within'][:4, :4] - 1e-10 * torch.eye(4, dtype=torch.float64), atol=1e-12, rtol=1e-10)

    def test_one_shot_predictions_are_finite_and_leave_frozen_model_unchanged(self):
        x, labels, anchors = fitting_data()
        state = fit_completion(x.requires_grad_(), labels, anchors.requires_grad_(), list(range(6)), rank=2)
        original = copy.deepcopy(state)
        a = complete_prototypes(state, x[:1], [44], anchors[:1], [44])
        self.assertTrue(torch.isfinite(a).all())
        self.assertFalse(a.requires_grad)
        for key in state:
            if isinstance(state[key], torch.Tensor):
                self.assertFalse(state[key].requires_grad)
                torch.testing.assert_close(state[key], original[key], atol=0, rtol=0)

    def test_support_and_requested_class_order_do_not_change_estimates(self):
        x, labels, anchors = fitting_data()
        state = fit_completion(x, labels, anchors, list(range(6)), rank=3)
        selected = torch.tensor([0, 2, 8, 10])
        support, target = x[selected], labels[selected]
        expected = complete_prototypes(state, support, target, anchors[:2], [0, 1])
        shuffled = complete_prototypes(state, support.flip(0), target.flip(0), anchors[:2].flip(0), [1, 0])
        torch.testing.assert_close(expected, shuffled.flip(0))

    def test_invalid_fitting_inputs_and_options_are_rejected(self):
        x, labels, anchors = fitting_data()
        for options in ({'rank': 0}, {'rank': True}, {'shrinkage': 1.1}, {'ridge': 0}, {'cross_view': 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                fit_completion(x, labels, anchors, list(range(6)), **options)
        with self.assertRaisesRegex(ValueError, 'two fitting classes'):
            fit_completion(x, labels, anchors[:1], [0])
        with self.assertRaisesRegex(ValueError, 'degrees of freedom'):
            fit_completion(x[::8], labels[::8], anchors, list(range(6)))
        with self.assertRaisesRegex(ValueError, 'unique'):
            fit_completion(x, labels, anchors[:2], [0, 0])
        with self.assertRaisesRegex(ValueError, 'support samples'):
            fit_completion(x, labels, anchors[:2], [0, 100])
        corrupt = x.clone()
        corrupt[0, 0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            fit_completion(corrupt, labels, anchors, list(range(6)))

    def test_prediction_rejects_dimension_and_corrupt_covariance_mismatch(self):
        state = scalar_state()
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            complete_prototypes(state, torch.ones(1, 2, 1), [0], torch.ones(1, 2, 1), [0])
        state['within'][0, 0] = -1
        with self.assertRaisesRegex(ValueError, 'semidefinite'):
            complete_prototypes(state, torch.ones(1, 1, 1), [0], torch.ones(1, 1, 1), [0])


if __name__ == '__main__':
    unittest.main()
