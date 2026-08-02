import unittest

import torch

from utils.frequency_analysis import (
    CLIP_MEAN,
    CLIP_STD,
    FourierBandStop,
    FrequencyContributionAccumulator,
    band_energy_fractions,
    classification_diagnostics,
    coordinate_ascent_class_gate,
    equal_energy_band_edges,
    exact_mcnemar_pvalue,
    paired_accuracy_statistics,
    pearson_correlation,
    per_class_accuracies,
    residual_full_weights,
    semantic_margins,
    spearman_correlation,
    summarize_frequency_records,
)


class FourierBandStopTest(unittest.TestCase):
    def test_constant_image_is_unchanged_when_dc_is_preserved(self):
        rgb = torch.full((2, 3, 16, 16), 0.4)
        mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
        normalized = (rgb - mean) / std
        transform = FourierBandStop(
            ['low', 'mid', 'high'], [0.0, 0.15, 0.35, 1.0]
        )

        for band_index in range(3):
            result = transform.remove(normalized, band_index)
            self.assertTrue(torch.allclose(result, normalized, atol=1e-5))

    def test_remove_all_returns_one_image_batch_per_band(self):
        normalized = torch.randn(2, 3, 16, 16)
        transform = FourierBandStop(
            ['low', 'mid', 'high'], [0.0, 0.15, 0.35, 1.0], fft_device='cpu'
        )
        results = transform.remove_all(normalized)
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertEqual(result.shape, normalized.shape)
            self.assertEqual(result.device, normalized.device)
            self.assertTrue(torch.isfinite(result).all())

    def test_selected_band_matches_remove_all(self):
        normalized = torch.randn(2, 3, 16, 16)
        transform = FourierBandStop(
            ['low', 'mid', 'high'], [0.0, 0.15, 0.35, 1.0], fft_device='cpu'
        )
        all_results = transform.remove_all(normalized)
        selected = transform.remove(normalized, 2)
        self.assertTrue(torch.allclose(selected, all_results[2], atol=1e-6))

    def test_radial_masks_partition_all_non_dc_frequencies(self):
        transform = FourierBandStop(
            ['low', 'mid', 'high'], [0.0, 0.15, 0.35, 1.0]
        )
        masks = transform._masks(16, 16, torch.device('cpu'))
        coverage = torch.stack(masks).sum(dim=0)
        self.assertEqual(float(coverage[0, 0, 0, 0]), 0.0)
        non_dc = coverage.flatten()[1:]
        self.assertTrue(torch.all(non_dc == 1))

    def test_equal_energy_edges_balance_a_power_spectrum(self):
        power = torch.ones(64, 64, dtype=torch.float64)
        power[0, 0] = 0.0
        edges = equal_energy_band_edges(power, num_bands=3)
        fractions = band_energy_fractions(power, edges)
        self.assertEqual(edges[0], 0.0)
        self.assertEqual(edges[-1], 1.0)
        self.assertTrue(all(left < right for left, right in zip(edges[:-1], edges[1:])))
        for fraction in fractions:
            self.assertAlmostEqual(fraction, 1.0 / 3.0, delta=0.03)

    def test_updating_edges_invalidates_cached_masks(self):
        transform = FourierBandStop(
            ['low', 'mid', 'high'], [0.0, 0.15, 0.35, 1.0]
        )
        original_masks = transform._masks(16, 16, torch.device('cpu'))
        transform.set_band_edges([0.0, 0.1, 0.2, 1.0])
        updated_masks = transform._masks(16, 16, torch.device('cpu'))
        self.assertIsNot(original_masks, updated_masks)


class FrequencyStatisticsTest(unittest.TestCase):
    def test_semantic_margin(self):
        image_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        text_features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        labels = torch.tensor([0, 1])
        margins = semantic_margins(image_features, text_features, labels)
        self.assertTrue(torch.allclose(margins, torch.tensor([1.0, 1.0])))

    def test_correlations_handle_monotonic_values_and_ties(self):
        x = torch.tensor([1.0, 2.0, 2.0, 4.0])
        y = torch.tensor([2.0, 4.0, 4.0, 8.0])
        self.assertAlmostEqual(pearson_correlation(x, y), 1.0, places=6)
        self.assertAlmostEqual(spearman_correlation(x, y), 1.0, places=6)
        self.assertAlmostEqual(spearman_correlation(x, -y), -1.0, places=6)

    def test_class_records_and_summary(self):
        accumulator = FrequencyContributionAccumulator(
            ['low', 'mid', 'high'], weight_temperature=0.1
        )
        accumulator.update(
            torch.tensor([0, 0, 1, 1]),
            {
                'low': torch.tensor([0.4, 0.2, 0.0, 0.0]),
                'mid': torch.tensor([0.1, 0.1, 0.3, 0.3]),
                'high': torch.tensor([0.0, 0.0, 0.1, 0.1]),
            },
        )
        records = accumulator.records(['zero', 'one'], task_id=0)
        self.assertEqual(records[0]['dominant_band'], 'low')
        self.assertEqual(records[1]['dominant_band'], 'mid')
        summary = summarize_frequency_records(records, ['low', 'mid', 'high'])
        self.assertEqual(summary['num_classes'], 2)
        self.assertEqual(summary['dominant_band_counts']['low'], 1)
        self.assertEqual(summary['dominant_band_counts']['mid'], 1)
        self.assertGreater(summary['mean_pairwise_frequency_weight_l1'], 0.0)
        self.assertGreater(summary['class_effect_eta_squared']['low'], 0.0)
        self.assertGreater(summary['class_effect_eta_squared']['mid'], 0.0)

    def test_exact_mcnemar_pvalue(self):
        self.assertEqual(exact_mcnemar_pvalue(0, 0), 1.0)
        self.assertAlmostEqual(exact_mcnemar_pvalue(0, 3), 0.25, places=7)
        self.assertEqual(exact_mcnemar_pvalue(1, 1), 1.0)

    def test_paired_accuracy_statistics(self):
        reference = torch.tensor([True, True, False, False])
        candidate = torch.tensor([True, False, True, False])
        result = paired_accuracy_statistics(
            reference, candidate, bootstrap_samples=100, seed=7
        )
        self.assertEqual(result['both_correct'], 1)
        self.assertEqual(result['reference_only_correct'], 1)
        self.assertEqual(result['candidate_only_correct'], 1)
        self.assertEqual(result['both_wrong'], 1)
        self.assertEqual(result['net_correct'], 0)
        self.assertAlmostEqual(result['accuracy_delta'], 0.0)
        self.assertAlmostEqual(result['mcnemar_exact_p'], 1.0)
        self.assertEqual(len(result['bootstrap_accuracy_delta_ci']), 2)

    def test_coordinate_ascent_class_gate_improves_accuracy(self):
        labels = torch.tensor([0, 1, 0, 1])
        full_logits = torch.tensor([
            [2.0, 1.0],
            [2.0, 1.0],
            [2.0, 1.0],
            [2.0, 1.0],
        ])
        removed_logits = torch.tensor([
            [2.0, 0.5],
            [0.0, 2.0],
            [2.0, 0.5],
            [0.0, 2.0],
        ])
        result = coordinate_ascent_class_gate(
            full_logits,
            removed_logits,
            labels,
            initial_gate=torch.ones(2),
            max_passes=3,
        )
        self.assertEqual(result['correct'], 4)
        self.assertAlmostEqual(result['accuracy'], 1.0)
        self.assertGreaterEqual(result['accepted_flips'], 1)
        self.assertEqual(result['correct_trajectory'][0], 2)

    def test_residual_full_weights_preserve_a_full_logit_floor(self):
        weights = residual_full_weights(torch.tensor([1.0, 0.0, 0.25]), 0.2)
        self.assertTrue(
            torch.allclose(weights, torch.tensor([1.0, 0.8, 0.85]))
        )
        self.assertGreaterEqual(float(weights.min()), 0.8)
        with self.assertRaises(ValueError):
            residual_full_weights(torch.tensor([1.0, 0.0]), 1.1)

    def test_classification_diagnostics_include_fscil_class_changes(self):
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        reference = torch.tensor([0, 1, 1, 0, 2, 0])
        candidate = torch.tensor([0, 0, 1, 1, 0, 0])
        class_accuracy, counts = per_class_accuracies(candidate, labels, 3)
        self.assertTrue(torch.equal(counts, torch.tensor([2, 2, 2])))
        self.assertTrue(
            torch.allclose(class_accuracy, torch.tensor([1.0, 1.0, 0.0]))
        )

        result = classification_diagnostics(
            candidate,
            labels,
            num_classes=3,
            num_base_classes=2,
            reference_predictions=reference,
        )
        self.assertAlmostEqual(result['micro_accuracy'], 4.0 / 6.0)
        self.assertAlmostEqual(result['macro_accuracy'], 2.0 / 3.0)
        self.assertAlmostEqual(result['base_micro_accuracy'], 1.0)
        self.assertAlmostEqual(result['novel_micro_accuracy'], 0.0)
        self.assertEqual(result['zero_accuracy_classes'], 1)
        changes = result['class_change_vs_reference']
        self.assertEqual(changes['improved_classes'], 2)
        self.assertEqual(changes['degraded_classes'], 1)
        self.assertEqual(changes['zeroed_classes'], 1)


if __name__ == '__main__':
    unittest.main()
