"""Independent numerical checks for frozen shared-covariance discriminants."""

import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from models.frequency_discriminant import (
    discriminant_logits,
    pooled_full_covariance,
    precision_from_covariance,
    prepare_discriminant,
    prepared_discriminant_logits,
)
from models.frequency_uncertainty import class_statistics


class FrequencyDiscriminantTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        generator = torch.Generator().manual_seed(42)
        self.support = torch.randn(12, 2, 3, generator=generator)
        self.query = torch.randn(5, 2, 3, generator=generator)
        self.labels = torch.tensor([7] * 3 + [2] * 4 + [9] * 5)
        self.ids = [9, 7, 2]
        self.stats = class_statistics(self.support, self.labels, self.ids)

    def test_covariance_matches_independent_per_class_outer_products(self):
        normalized = F.normalize(self.support, dim=-1).flatten(1).double()
        expected = torch.zeros(6, 6, dtype=torch.float64)
        for label in self.ids:
            group = normalized[self.labels == label]
            for sample in group:
                difference = sample - group.mean(0)
                expected += torch.outer(difference, difference)
        expected /= len(self.labels) - len(self.ids)
        actual = pooled_full_covariance(self.support, self.labels, self.ids)
        torch.testing.assert_close(actual.double(), expected, atol=5e-8, rtol=2e-6)

    def test_covariance_excludes_unrequested_classes_and_oneshot_degrees(self):
        subset = pooled_full_covariance(self.support, self.labels, [7, 2])
        kept = self.labels != 9
        torch.testing.assert_close(subset, pooled_full_covariance(
            self.support[kept], self.labels[kept], [7, 2]))
        augmented = torch.cat((self.support, torch.ones(1, 2, 3)), dim=0)
        labels = torch.cat((self.labels, torch.tensor([19])))
        torch.testing.assert_close(pooled_full_covariance(augmented, labels, self.ids + [19]),
                                   pooled_full_covariance(self.support, self.labels, self.ids))

    def test_covariance_contains_no_between_class_scatter(self):
        support = torch.tensor([[[1., 0.]], [[1., 0.]], [[-1., 0.]], [[-1., 0.]]])
        covariance = pooled_full_covariance(support, [0, 0, 1, 1], [0, 1])
        torch.testing.assert_close(covariance, torch.zeros(2, 2))
        self.assertGreater(support[:, 0, 0].var().item(), 1.)
        precision = precision_from_covariance(covariance, ridge=.25, var_floor=1e-4)
        torch.testing.assert_close(precision, torch.eye(2) * 40000)

    def test_block_retains_within_view_correlations_but_discards_cross_view(self):
        correlated = self.support[:, :1].repeat(1, 2, 1)
        full = pooled_full_covariance(correlated, self.labels, self.ids, "full")
        block = pooled_full_covariance(correlated, self.labels, self.ids, "block")
        torch.testing.assert_close(block[:3, :3], full[:3, :3])
        torch.testing.assert_close(block[3:, 3:], full[3:, 3:])
        torch.testing.assert_close(block[:3, 3:], torch.zeros(3, 3))
        self.assertGreater(full[:3, 3:].abs().sum().item(), .1)
        self.assertGreater(block[:3, :3].triu(1).abs().sum().item(), .01)

    def test_ridge_precision_matches_direct_double_inverse(self):
        covariance = pooled_full_covariance(self.support, self.labels, self.ids)
        for ridge in (.01, .1, 1.):
            expected = torch.linalg.inv(covariance.double()
                                        + ridge * covariance.double().diagonal().mean() * torch.eye(6))
            actual = precision_from_covariance(covariance, ridge=ridge)
            torch.testing.assert_close(actual.double(), expected, rtol=2e-6, atol=2e-6)
            self.assertGreater(torch.linalg.eigvalsh(actual.double()).min().item(), 0.)
        torch.testing.assert_close(precision_from_covariance(torch.eye(3), ridge=0), torch.eye(3))

    def test_logits_equal_direct_mahalanobis_up_to_query_only_offset(self):
        covariance = pooled_full_covariance(self.support, self.labels, self.ids)
        precision = precision_from_covariance(covariance)
        query = F.normalize(self.query, dim=-1).flatten(1).double()
        means = self.stats["mean"].flatten(1).double()
        expected = torch.empty(5, 3, dtype=torch.float64)
        for row, sample in enumerate(query):
            for column, mean in enumerate(means):
                delta = sample - mean
                expected[row, column] = -.5 * delta @ precision.double() @ delta / 6
        actual = discriminant_logits(self.query, self.stats, precision).double()
        offset = .5 * torch.einsum("ni,ij,nj->n", query, precision.double(), query) / 6
        torch.testing.assert_close(actual, expected + offset[:, None], atol=5e-7, rtol=5e-6)
        torch.testing.assert_close(actual[:, 1:] - actual[:, :1],
                                   expected[:, 1:] - expected[:, :1], atol=5e-7, rtol=5e-6)

    def test_means_are_not_renormalized_and_counts_do_not_bias_classes(self):
        means = torch.tensor([[[.5, 0.]], [[1., 0.]]])
        stats = {"mean": means, "variance": torch.tensor([[[.1, .5]], [[100., 300.]]]),
                 "count": torch.tensor([1, 500])}
        actual = discriminant_logits(torch.tensor([[[1., 0.]]]), stats, torch.eye(2))
        torch.testing.assert_close(actual, torch.tensor([[.1875, .25]]))
        # Irrelevant statistics may differ without affecting the shared model.
        changed = {"mean": means, "variance": stats["variance"].flip(0),
                   "count": stats["count"].flip(0)}
        torch.testing.assert_close(actual, discriminant_logits(
            torch.tensor([[[1., 0.]]]), changed, torch.eye(2)))

    def test_duplicate_views_and_singular_covariance_remain_finite_with_ridge(self):
        support = self.support[:, :1].repeat(1, 3, 1)
        query = self.query[:, :1].repeat(1, 3, 1)
        covariance = pooled_full_covariance(support, self.labels, self.ids)
        self.assertLess(torch.linalg.matrix_rank(covariance).item(), 9)
        precision = precision_from_covariance(covariance, ridge=.01)
        logits = discriminant_logits(query, class_statistics(support, self.labels, self.ids), precision)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertGreater(torch.linalg.eigvalsh(precision.double()).min().item(), 0.)
        with self.assertRaises(ValueError):
            precision_from_covariance(torch.zeros(3, 3), ridge=0)

    def test_joint_precision_changes_correlated_view_predictions_scores(self):
        full = precision_from_covariance(pooled_full_covariance(self.support, self.labels, self.ids))
        block = precision_from_covariance(pooled_full_covariance(
            self.support, self.labels, self.ids, structure="block"))
        full_scores = discriminant_logits(self.query, self.stats, full)
        block_scores = discriminant_logits(self.query, self.stats, block)
        self.assertFalse(torch.allclose(full_scores, block_scores, atol=1e-3, rtol=1e-3))

    def test_class_and_view_permutations_are_equivariant(self):
        covariance = pooled_full_covariance(self.support, self.labels, self.ids)
        precision = precision_from_covariance(covariance)
        actual = discriminant_logits(self.query, self.stats, precision)
        class_order = [1, 2, 0]
        permuted_stats = {key: value[class_order] for key, value in self.stats.items()}
        torch.testing.assert_close(discriminant_logits(self.query, permuted_stats, precision),
                                   actual[:, class_order])
        indices = torch.tensor([3, 4, 5, 0, 1, 2])
        swapped = pooled_full_covariance(self.support.flip(1), self.labels, list(reversed(self.ids)))
        torch.testing.assert_close(swapped, covariance[indices][:, indices])
        band_stats = {"mean": self.stats["mean"].flip(1)}
        torch.testing.assert_close(discriminant_logits(self.query.flip(1), band_stats,
                                                       precision_from_covariance(swapped)), actual)

    def test_prepared_scores_match_direct_scores_across_query_batches(self):
        precision = precision_from_covariance(pooled_full_covariance(self.support, self.labels, self.ids))
        prepared = prepare_discriminant(self.stats, precision)
        self.assertEqual(prepared["weights"].shape, (6, 3))
        self.assertEqual(prepared["bias"].shape, (3,))
        self.assertEqual(prepared["num_bands"], 2)
        self.assertEqual(prepared["feature_dim"], 3)
        chunks = [prepared_discriminant_logits(self.query[:2], prepared),
                  prepared_discriminant_logits(self.query[2:], prepared)]
        torch.testing.assert_close(torch.cat(chunks), discriminant_logits(self.query, self.stats, precision))

    def test_prevalidated_precision_avoids_repeated_factorization(self):
        precision = precision_from_covariance(pooled_full_covariance(self.support, self.labels, self.ids))
        expected = discriminant_logits(self.query, self.stats, precision)
        with patch('torch.linalg.cholesky_ex', side_effect=AssertionError('Already validated')):
            actual = discriminant_logits(self.query, self.stats, precision, validate_precision=False)
        torch.testing.assert_close(actual, expected)
        # The fast path retains finite/shape/symmetry checks.
        with self.assertRaises(ValueError):
            prepare_discriminant(self.stats, torch.full_like(precision, float('nan')),
                                 validate_precision=False)

    def test_half_inputs_compute_float32_without_mutation_or_gradients(self):
        support = self.support.half().requires_grad_()
        query = self.query.half().requires_grad_()
        original = support.detach().clone()
        covariance = pooled_full_covariance(support, self.labels, self.ids)
        covariance.requires_grad_()
        precision = precision_from_covariance(covariance)
        precision.requires_grad_()
        stats = class_statistics(support, self.labels, self.ids)
        stats["mean"].requires_grad_()
        mean_before = stats["mean"].detach().clone()
        precision_before = precision.detach().clone()
        prepared = prepare_discriminant(stats, precision)
        actual = prepared_discriminant_logits(query, prepared)
        for value in (prepared["weights"], prepared["bias"], actual):
            self.assertFalse(value.requires_grad)
            self.assertEqual(value.dtype, torch.float32)
        torch.testing.assert_close(support, original)
        torch.testing.assert_close(stats["mean"], mean_before)
        torch.testing.assert_close(precision, precision_before)
        torch.testing.assert_close(actual, discriminant_logits(query.float(), stats, precision))

    def test_invalid_support_and_class_protocol_are_rejected(self):
        valid = torch.ones(2, 1, 2)
        cases = [(valid, [0], [0]), (valid, [0, 0], [1]), (valid, [0, 0], [0, 0]),
                 (valid, [0, 1], [0, 1]), (valid, [0., .5], [0]),
                 (torch.ones(2, 2), [0, 0], [0]),
                 (torch.full_like(valid, float("nan")), [0, 0], [0])]
        for arguments in cases:
            with self.subTest(arguments=str(arguments)), self.assertRaises(ValueError):
                pooled_full_covariance(*arguments)
        with self.assertRaises(ValueError):
            pooled_full_covariance(valid, [0, 0], [0], structure="diagonal")

    def test_invalid_covariance_and_regularization_are_rejected(self):
        for covariance in (torch.ones(2, 3), torch.empty(0, 0), -torch.eye(2),
                           torch.tensor([[1., 2.], [0., 1.]]),
                           torch.tensor([[1., 2.], [2., 1.]]),
                           torch.full((2, 2), float("inf")),
                           torch.eye(2, dtype=torch.float64) * 1e100):
            with self.subTest(covariance=covariance), self.assertRaises(ValueError):
                precision_from_covariance(covariance)
        for kwargs in ({"ridge": -1}, {"ridge": float("nan")}, {"ridge": float("inf")},
                       {"var_floor": 0}, {"var_floor": float("inf")},
                       {"ridge": 1e308, "var_floor": 1e308}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                precision_from_covariance(torch.eye(2), **kwargs)
        with self.assertRaises(ValueError):
            precision_from_covariance(torch.zeros(2, 2), ridge=1e-100, var_floor=1e-100)

    def test_invalid_mean_precision_query_and_prepared_shapes_are_rejected(self):
        precision = torch.eye(6)
        for stats in ({}, {"mean": torch.ones(3, 6)}, {"mean": torch.empty(0, 2, 3)},
                      {"mean": torch.full((3, 2, 3), float("nan"))}):
            with self.assertRaises(ValueError):
                prepare_discriminant(stats, precision)
        for matrix in (torch.eye(3), -precision, torch.ones(6, 6)):
            with self.assertRaises(ValueError):
                prepare_discriminant(self.stats, matrix)
        prepared = prepare_discriminant(self.stats, precision)
        for query in (torch.ones(3, 1, 6), torch.ones(2, 6), torch.full_like(self.query, float("inf"))):
            with self.assertRaises(ValueError):
                prepared_discriminant_logits(query, prepared)
        for updates in ({"num_bands": True}, {"feature_dim": 0}, {"weights": torch.ones(3, 3)},
                        {"bias": torch.ones(4)}, {"weights": torch.full((6, 3), float("nan"))}):
            with self.assertRaises(ValueError):
                prepared_discriminant_logits(self.query, {**prepared, **updates})
        with self.assertRaises(ValueError):
            prepared_discriminant_logits(self.query, {})


if __name__ == "__main__":
    unittest.main()
