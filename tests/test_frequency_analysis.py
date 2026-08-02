import unittest

import torch

from utils.frequency_analysis import (
    CLIP_MEAN,
    CLIP_STD,
    FourierBandStop,
    FrequencyContributionAccumulator,
    band_energy_fractions,
    equal_energy_band_edges,
    pearson_correlation,
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


if __name__ == '__main__':
    unittest.main()
