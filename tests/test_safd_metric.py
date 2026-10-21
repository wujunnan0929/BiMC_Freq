"""Core mathematical and regression checks; no CLIP weights or image data."""
import unittest

import torch
from torch.nn import functional as F

from models.safd_metric import (
    SAFDMetricAdapter, apply_metric_residual, build_response_basis,
    fit_explained_scatter, metric_from_scatter,
    prepare_residual_classifier, scale_responses,
)


class SAFDMetricTest(unittest.TestCase):
    def samples(self):
        rng = torch.Generator().manual_seed(71)
        labels = torch.arange(4).repeat_interleave(12)
        x = torch.randn(48, 7, generator=rng, dtype=torch.float64)
        response = torch.cat((x[:, :3], torch.randn(48, 2, generator=rng, dtype=torch.float64)), 1)
        return x, response, labels

    def test_scale_responses_keep_amplitude_and_repeat_is_exact_zero(self):
        values = torch.tensor([[[1., 0.], [0., 1.], [1., 1.]]], requires_grad=True)
        response = scale_responses(values)
        z = F.normalize(values.detach().double(), dim=-1)
        self.assertTrue(torch.equal(response[:, 0], z[:, 2] - z[:, 1]))
        self.assertTrue(torch.equal(response[:, 1], z[:, 0] - z[:, 2]))
        self.assertFalse(response.requires_grad)
        repeat = values.detach()[:, :1].expand(-1, 3, -1)
        self.assertTrue(torch.equal(scale_responses(repeat), torch.zeros(1, 2, 2, dtype=torch.float64)))

    def test_explained_scatter_psd_loewner_and_metric_contraction(self):
        x, response, labels = self.samples()
        state = fit_explained_scatter(x.requires_grad_(), response.requires_grad_(), labels, rank=3)
        n, c = state['scatter'], state['within_scatter']
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(n).min()), -1e-12)
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(c-n).min()), -1e-12)
        self.assertFalse(n.requires_grad)
        self.assertEqual(n.device.type, 'cpu')
        m = metric_from_scatter(state, strength=2.)
        eigenvalues = torch.linalg.eigvalsh(m)
        self.assertGreater(float(eigenvalues.min()), 0.)
        self.assertLessEqual(float(eigenvalues.max()), 1+1e-12)
        self.assertEqual(state['diagnostics']['retained_rank'], 3)

    def test_per_class_mean_shifts_do_not_change_estimator(self):
        x, response, labels = self.samples()
        shifts_x = torch.tensor([[3., 1., -1., 2., 0., 4., 9.], [-2., 1., 0., 3., 6., -1., 4.],
                                 [1., 7., 2., 3., 9., 0., -2.], [-3., 2., 1., -5., 3., 1., 0.]])
        shifts_r = torch.arange(20).reshape(4, 5).double()
        a = fit_explained_scatter(x, response, labels, rank=4)
        b = fit_explained_scatter(x+shifts_x[labels], response+shifts_r[labels], labels, rank=4)
        torch.testing.assert_close(a['scatter'], b['scatter'], rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(a['within_scatter'], b['within_scatter'], rtol=1e-11, atol=1e-11)

    def test_paired_permutation_invariant_response_shuffle_changes_scatter(self):
        x, response, labels = self.samples()
        order = torch.randperm(len(x), generator=torch.Generator().manual_seed(9))
        a = fit_explained_scatter(x, response, labels, rank=4)
        b = fit_explained_scatter(x[order], response[order], labels[order], rank=4)
        c = fit_explained_scatter(x, response[order], labels, rank=4)
        torch.testing.assert_close(a['scatter'], b['scatter'], rtol=1e-11, atol=1e-11)
        self.assertGreater(float((a['scatter']-c['scatter']).abs().max()), .05)

    def test_zero_responses_identity_metric_and_exact_reference_object(self):
        x, response, labels = self.samples()
        state = fit_explained_scatter(x, torch.zeros_like(response), labels)
        metric = metric_from_scatter(state, strength=100.)
        self.assertTrue(torch.equal(metric, torch.eye(7, dtype=torch.float64)))
        prepared = prepare_residual_classifier(x[:4], metric)
        votes = torch.rand(5, 4)+.1
        self.assertIs(apply_metric_residual(votes, x[:5], prepared, gain=3.), votes)
        active = prepare_residual_classifier(x[:4], metric_from_scatter(
            fit_explained_scatter(x, response, labels), strength=1.))
        self.assertIs(apply_metric_residual(votes, x[:5], active, gain=0.), votes)
        zero = metric_from_scatter(fit_explained_scatter(x, response, labels), strength=0.)
        self.assertTrue(torch.equal(zero, metric))

    def test_compiled_formula_positive_mass_and_no_input_mutation(self):
        x, response, labels = self.samples()
        metric = metric_from_scatter(fit_explained_scatter(x, response, labels))
        prototypes = x[:4].clone().requires_grad_()
        query = x[4:9].clone().requires_grad_()
        prepared = prepare_residual_classifier(prototypes, metric)
        mu, q = F.normalize(prototypes.detach(), dim=-1), F.normalize(query.detach(), dim=-1)
        delta = metric-torch.eye(7, dtype=torch.float64)
        expected = 2*q@delta@mu.T - (mu*(mu@delta)).sum(1)
        actual = q@prepared['weights']+prepared['bias']
        torch.testing.assert_close(actual, expected)
        votes = torch.tensor([[.4, .1, .3, .2]]).repeat(5, 1)
        original = votes.clone()
        output = apply_metric_residual(votes, query, prepared, gain=4.)
        self.assertTrue((output > 0).all())
        self.assertFalse(output.requires_grad)
        self.assertTrue(torch.equal(votes, original))
        torch.testing.assert_close(output.sum(1), votes.sum(1))
        expected = (votes.double().log()+4*expected).softmax(1)*votes.sum(1, keepdim=True)
        torch.testing.assert_close(output.double(), expected, rtol=1e-6, atol=1e-7)

    def test_original_within_is_rank_matched_and_class_names_are_not_priors(self):
        x, response, labels = self.samples()
        a = fit_explained_scatter(x, None, labels, rank=2, mode='original_within')
        b = fit_explained_scatter(x, None, labels*11+7, rank=2, mode='original_within')
        self.assertEqual(a['diagnostics']['retained_rank'], 2)
        self.assertLessEqual(int(torch.linalg.matrix_rank(a['scatter'], atol=1e-10)), 2)
        torch.testing.assert_close(a['scatter'], b['scatter'])
        # Class exclusions must remove all statistics from an excluded class.
        subset = fit_explained_scatter(x, response, labels, class_ids=[0, 2], rank=2)
        mask = (labels==0)|(labels==2)
        direct = fit_explained_scatter(x[mask], response[mask], labels[mask], rank=2)
        torch.testing.assert_close(subset['scatter'], direct['scatter'])

    def test_small_response_not_promoted_to_unit_strength_and_invalid_inputs(self):
        x, response, labels = self.samples()
        # With the fixed ridge floor and original-scatter scale, tiny responses
        # approach identity instead of being normalised into full directions.
        weak = fit_explained_scatter(x, response*1e-10, labels)
        metric = metric_from_scatter(weak)
        self.assertLess(float((metric-torch.eye(7)).abs().max()), 1e-6)
        for kwargs in ({'rank':0}, {'ridge':0}, {'mode':'unknown'}):
            with self.assertRaises(ValueError):
                fit_explained_scatter(x, response, labels, **kwargs)
        with self.assertRaisesRegex(ValueError, 'degrees of freedom'):
            fit_explained_scatter(x, response, torch.arange(len(x)))
        with self.assertRaises(ValueError):
            metric_from_scatter({'scatter': -torch.eye(2), 'within_scatter':torch.eye(2)})

    def test_adapter_initial_identity_has_nonzero_theta_gradient_only(self):
        x, response, labels = self.samples()
        state = fit_explained_scatter(x, response, labels, rank=3)
        basis = build_response_basis(state, rank=3)
        adapter = SAFDMetricAdapter(basis)
        self.assertEqual([name for name, _ in adapter.named_parameters()], ['theta'])
        q = x[:5].float().requires_grad_()
        mu = x[5:9].float().requires_grad_()
        scores = adapter.delta_scores(q, mu)
        self.assertTrue(torch.equal(scores, torch.zeros_like(scores)))
        self.assertTrue(torch.equal(adapter.to_metric(), torch.eye(7)))
        (scores * torch.arange(scores.numel()).reshape_as(scores)).sum().backward()
        self.assertIsNotNone(adapter.theta.grad)
        self.assertGreater(float(adapter.theta.grad.abs().sum()), 0.)
        self.assertIsNone(q.grad)
        self.assertIsNone(mu.grad)
        self.assertFalse(adapter.basis.requires_grad)

    def test_adapter_spd_bounds_compiled_scores_and_query_order(self):
        x, response, labels = self.samples()
        adapter = SAFDMetricAdapter(build_response_basis(fit_explained_scatter(x, response, labels), rank=3))
        with torch.no_grad():
            adapter.theta.copy_(torch.tensor([-20., .3, 20.]))
        metric = adapter.to_metric()
        eigenvalues = torch.linalg.eigvalsh(metric.double())
        self.assertGreaterEqual(float(eigenvalues.min()), .1-1e-6)
        self.assertLessEqual(float(eigenvalues.max()), 1.9+1e-6)
        q, mu = x[:5].float(), x[5:9].float()
        expected = adapter.delta_scores(q, mu).detach().double()
        compiled = prepare_residual_classifier(mu, metric)
        actual = F.normalize(q.double(), dim=-1)@compiled['weights']+compiled['bias']
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        order = torch.tensor([4, 2, 0, 1, 3])
        torch.testing.assert_close(adapter(q[order], mu), adapter(q, mu)[order])

    def test_zero_response_adapter_and_invalid_basis(self):
        x, response, labels = self.samples()
        basis = build_response_basis(fit_explained_scatter(x, torch.zeros_like(response), labels))
        self.assertEqual(tuple(basis.shape), (7, 0))
        adapter = SAFDMetricAdapter(basis)
        self.assertTrue(torch.equal(adapter(x[:5], x[5:9]), torch.zeros(5, 4)))
        self.assertTrue(torch.equal(adapter.to_metric(), torch.eye(7)))
        with self.assertRaises(ValueError):
            SAFDMetricAdapter(torch.ones(7, 2))
        with self.assertRaises(ValueError):
            SAFDMetricAdapter(torch.eye(2), kappa=1.)

    def test_prevalidated_metric_skips_only_expensive_spd_check(self):
        x, response, labels = self.samples()
        metric = metric_from_scatter(fit_explained_scatter(x, response, labels))
        checked = prepare_residual_classifier(x[:4], metric)
        fast = prepare_residual_classifier(x[:4], metric, validate_metric=False)
        self.assertTrue(torch.equal(checked['weights'], fast['weights']))
        self.assertTrue(torch.equal(checked['bias'], fast['bias']))
        with self.assertRaisesRegex(ValueError, 'positive definite'):
            prepare_residual_classifier(x[:4], -torch.eye(7))
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            prepare_residual_classifier(x[:4], torch.eye(6), validate_metric=False)


if __name__ == '__main__':
    unittest.main()
