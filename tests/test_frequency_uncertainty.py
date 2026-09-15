"""CPU tests for finite-support frequency statistics and predictive scores."""

import unittest

import torch
from torch.nn import functional as F

from models.frequency_uncertainty import (
    class_statistics,
    gaussian_band_logits,
    mix_uncertainty_probabilities,
    pooled_variance,
    predictive_variance,
    uncertainty_logits,
)


def _stats(counts=(2, 5), bands=1, dimensions=2):
    return {"mean": torch.zeros(len(counts), bands, dimensions),
            "variance": torch.ones(len(counts), bands, dimensions),
            "count": torch.tensor(counts)}


class FrequencyUncertaintyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_statistics_normalize_samples_keep_raw_mean_and_class_order(self):
        features = torch.tensor([[[3., 0.]], [[0., 5.]], [[-2., 0.]]], requires_grad=True)
        stats = class_statistics(features, torch.tensor([4, 4, 9]), [9, 4])
        torch.testing.assert_close(stats['mean'], torch.tensor([[[-1., 0.]], [[.5, .5]]]))
        torch.testing.assert_close(stats['variance'], torch.tensor([[[0., 0.]], [[.5, .5]]]))
        torch.testing.assert_close(stats['count'], torch.tensor([1, 2]))
        self.assertFalse(stats['mean'].requires_grad)
        self.assertFalse(stats['variance'].requires_grad)

    def test_one_shot_uses_prior_even_with_zero_prior_strength(self):
        stats = class_statistics(torch.tensor([[[1., 0.]]]), [7], [7])
        prior = torch.tensor([[.25, .5]])
        for strength in (0., 20.):
            actual = predictive_variance(stats, prior, prior_strength=strength)
            torch.testing.assert_close(actual, 2 * prior.unsqueeze(0))
        plain = predictive_variance(stats, prior, prior_strength=0, mean_uncertainty=False)
        torch.testing.assert_close(plain, prior.unsqueeze(0))

    def test_unbiased_shrinkage_and_mean_uncertainty_have_analytic_values(self):
        stats = _stats((2, 5))
        stats['variance'] = torch.tensor([[[2., 4.]], [[6., 8.]]])
        prior = torch.tensor([[10., 20.]])
        result = predictive_variance(stats, prior, prior_strength=3, mean_uncertainty=False)
        expected = torch.stack((.75 * prior + .25 * stats['variance'][0],
                                3 / 7 * prior + 4 / 7 * stats['variance'][1]))
        torch.testing.assert_close(result, expected)
        inflated = predictive_variance(stats, prior, prior_strength=3)
        torch.testing.assert_close(inflated, expected * torch.tensor([1.5, 1.2])[:, None, None])
        unshrunk = predictive_variance(stats, prior, prior_strength=0, mean_uncertainty=False)
        torch.testing.assert_close(unshrunk, stats['variance'])

    def test_shared_variance_is_class_independent_before_count_correction(self):
        stats = _stats((1, 3))
        stats['variance'][1] *= 100
        prior = torch.tensor([[.2, .4]], requires_grad=True)
        shared = predictive_variance(stats, prior, covariance='shared', mean_uncertainty=False)
        torch.testing.assert_close(shared, prior.detach().expand(2, 1, 2))
        corrected = predictive_variance(stats, prior, covariance='shared')
        torch.testing.assert_close(corrected, shared * torch.tensor([2., 4 / 3])[:, None, None])
        self.assertFalse(shared.requires_grad)
        self.assertFalse(corrected.requires_grad)

    def test_pooling_weights_within_class_degrees_of_freedom(self):
        stats = _stats((1, 3, 5))
        stats['variance'] = torch.tensor([[[1000., 1000.]], [[2., 4.]], [[5., 7.]]])
        result = pooled_variance(stats)
        torch.testing.assert_close(result, torch.tensor([[4., 6.]]))

    def test_pooling_contains_no_between_class_variance(self):
        features = torch.tensor([[[1., 0.]], [[1., 0.]], [[-1., 0.]], [[-1., 0.]]])
        stats = class_statistics(features, [0, 0, 1, 1], [0, 1])
        result = pooled_variance(stats, var_floor=1e-5)
        torch.testing.assert_close(result, torch.full((1, 2), 1e-5))
        self.assertGreater(features[:, 0, 0].var().item(), 1.)
        with self.assertRaises(ValueError):
            pooled_variance(class_statistics(features[[0, 2]], [0, 1], [0, 1]))

    def test_gaussian_logits_match_analytic_diagonal_density(self):
        stats = _stats()
        stats['mean'] = torch.tensor([[[1., 0.]], [[0., 1.]]])
        stats['variance'] = torch.tensor([[[.2, .5]], [[.3, .7]]])
        query = torch.tensor([[[2., 0.]], [[0., 3.]]], requires_grad=True)
        expected = torch.empty(2, 2, 1)
        normalized = F.normalize(query.detach(), dim=-1)
        for row in range(2):
            for column in range(2):
                variance = stats['variance'][column, 0]
                delta = normalized[row, 0] - stats['mean'][column, 0]
                expected[row, column, 0] = -.5 * (delta.square() / variance + variance.log()).mean()
        actual = gaussian_band_logits(query, stats, torch.ones(1, 2),
                                      prior_strength=0, mean_uncertainty=False)
        torch.testing.assert_close(actual, expected)
        self.assertFalse(actual.requires_grad)

    def test_log_determinant_penalizes_broad_class_at_same_mean(self):
        stats = _stats()
        stats['mean'][:, 0, 0] = 1
        stats['variance'][0] = .1
        query = torch.tensor([[[1., 0.]]])
        actual = uncertainty_logits(query, stats, torch.ones(1, 2), prior_strength=0,
                                    mean_uncertainty=False)
        self.assertGreater(actual[0, 0].item(), actual[0, 1].item())
        torch.testing.assert_close(actual[0], torch.tensor([-.5 * torch.tensor(.1).log(), 0.]))

    def test_class_and_band_permutations_are_equivariant(self):
        generator = torch.Generator().manual_seed(12)
        support = torch.randn(12, 3, 7, generator=generator)
        query = torch.randn(4, 3, 7, generator=generator)
        labels = torch.arange(3).repeat_interleave(4)
        stats = class_statistics(support, labels, [0, 1, 2])
        prior = pooled_variance(stats)
        actual = gaussian_band_logits(query, stats, prior)
        class_order = [2, 0, 1]
        band_order = [1, 2, 0]
        permuted = {key: value[class_order] for key, value in stats.items()}
        permuted['mean'] = permuted['mean'][:, band_order]
        permuted['variance'] = permuted['variance'][:, band_order]
        reordered = gaussian_band_logits(query[:, band_order], permuted, prior[band_order])
        torch.testing.assert_close(reordered, actual[:, class_order][:, :, band_order])
        logits = uncertainty_logits(query, stats, prior)
        torch.testing.assert_close(logits, actual.mean(-1))
        torch.testing.assert_close(uncertainty_logits(query[:, band_order], permuted, prior[band_order]),
                                   logits[:, class_order])

    def test_chunk_boundaries_match_direct_computation(self):
        generator = torch.Generator().manual_seed(23)
        query = torch.randn(257, 2, 3, generator=generator)
        stats = _stats(tuple([3] * 33), bands=2, dimensions=3)
        stats['mean'] = torch.rand(33, 2, 3, generator=generator)
        prior = torch.full((2, 3), .2)
        variance = predictive_variance(stats, prior)
        difference = F.normalize(query, dim=-1)[:, None] - stats['mean'][None]
        expected = -.5 * (difference.square() / variance[None] + variance.log()[None]).mean(-1)
        torch.testing.assert_close(gaussian_band_logits(query, stats, prior), expected)

    def test_half_precision_inputs_compute_in_float32(self):
        features = torch.tensor([[[60000., 30000.]], [[0., 60000.]],
                                 [[.00001, 0.]], [[0., 0.]]], dtype=torch.float16)
        labels = [0, 0, 1, 1]
        half_stats = class_statistics(features, labels, [0, 1])
        float_stats = class_statistics(features.float(), labels, [0, 1])
        for key in ('mean', 'variance'):
            self.assertEqual(half_stats[key].dtype, torch.float32)
            torch.testing.assert_close(half_stats[key], float_stats[key])
        prior = pooled_variance(half_stats)
        result = uncertainty_logits(features, half_stats, prior)
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(torch.isfinite(result).all())
        torch.testing.assert_close(result, uncertainty_logits(features.float(), float_stats, prior))

    def test_mixing_zero_is_same_object_and_active_mix_preserves_mass(self):
        reference = torch.tensor([[.2, .4, .1], [1., 2., 3.], [0., 0., 0.]], dtype=torch.float16)
        logits = torch.tensor([[2., -1., 0.], [.3, 1., -.1], [0., 1., 2.]], requires_grad=True)
        original = reference.clone()
        self.assertIs(mix_uncertainty_probabilities(reference, logits, alpha=0), reference)
        result = mix_uncertainty_probabilities(reference, logits, alpha=.3, temperature=.7)
        mass = reference.float().sum(-1, keepdim=True)
        expected = .7 * reference.float() + .3 * mass * torch.softmax(logits.detach() / .7, -1)
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(result.sum(-1), mass.squeeze(-1))
        self.assertFalse(result.requires_grad)
        torch.testing.assert_close(reference, original)
        full = mix_uncertainty_probabilities(reference, logits, alpha=1)
        torch.testing.assert_close(full, mass * logits.detach().softmax(-1))

    def test_invalid_features_labels_and_class_ids_are_rejected(self):
        valid = torch.ones(2, 1, 2)
        bad_arguments = [
            (torch.ones(2, 2), [0, 0], [0]),
            (torch.empty(0, 1, 2), [], [0]),
            (torch.full_like(valid, float('nan')), [0, 0], [0]),
            (valid, [0], [0]), (valid, [0., .5], [0]),
            (valid, [0, 0], [0, 0]), (valid, [0, 0], [1]),
            (valid, [0, 0], []),
        ]
        for arguments in bad_arguments:
            with self.subTest(arguments=str(arguments)), self.assertRaises(ValueError):
                class_statistics(*arguments)

    def test_invalid_statistics_and_variance_parameters_are_rejected(self):
        stats = _stats()
        prior = torch.ones(1, 2)
        for updates in ({'count': torch.tensor([0, 2])},
                        {'count': torch.tensor([2., 2.5])},
                        {'variance': -torch.ones(2, 1, 2)},
                        {'mean': torch.ones(2, 2)},
                        {'variance': torch.ones(2, 1, 3)},
                        {'mean': torch.full((2, 1, 2), float('inf'))}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                predictive_variance({**stats, **updates}, prior)
        for kwargs in ({'prior_strength': -1}, {'prior_strength': float('nan')},
                       {'var_floor': 0}, {'var_floor': float('inf')},
                       {'covariance': 'full'}, {'mean_uncertainty': 'yes'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                predictive_variance(stats, prior, **kwargs)
        for invalid_prior in (torch.ones(3, 2), -prior, torch.full_like(prior, float('nan'))):
            with self.assertRaises(ValueError):
                predictive_variance(stats, invalid_prior)
        with self.assertRaises(ValueError):
            pooled_variance({})
        with self.assertRaises(ValueError):
            gaussian_band_logits(torch.ones(2, 3, 2), stats, prior)

    def test_invalid_mixing_parameters_are_rejected(self):
        reference = torch.ones(2, 3)
        scores = torch.zeros_like(reference)
        for alpha in (-.1, 1.1, float('nan'), float('inf')):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                mix_uncertainty_probabilities(reference, scores, alpha)
        for temperature in (0, -1, float('nan'), float('inf')):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                mix_uncertainty_probabilities(reference, scores, .5, temperature)
        for votes, logits in ((-reference, scores), (reference, scores[:, :2]),
                              (reference, scores + float('nan')),
                              (reference + float('inf'), scores)):
            with self.assertRaises(ValueError):
                mix_uncertainty_probabilities(votes, logits, .5)


if __name__ == '__main__':
    unittest.main()
