import unittest

import torch
import torch.nn.functional as F

from models.frequency import (
    RadialFrequencyDecomposer,
    calibrate_frequency_prototypes,
    compute_frequency_logits,
    compute_frequency_prototypes,
    route_description_embeddings,
)


class FrequencyModuleTest(unittest.TestCase):
    def test_frequency_components_reconstruct_pixels(self):
        torch.manual_seed(1)
        pixels = torch.rand(2, 3, 16, 16)
        decomposer = RadialFrequencyDecomposer(0.2, 0.55)

        components = decomposer.split_pixels(pixels)

        self.assertEqual(components.shape, (2, 3, 3, 16, 16))
        self.assertTrue(
            torch.allclose(components.sum(dim=1), pixels, atol=2e-5, rtol=1e-5)
        )

    def test_chunked_fft_matches_single_sample_fft(self):
        torch.manual_seed(11)
        pixels = torch.rand(5, 3, 16, 16)
        chunked = RadialFrequencyDecomposer(
            0.2, 0.55, fft_batch_size=3
        ).split_pixels(pixels)
        single = RadialFrequencyDecomposer(
            0.2, 0.55, fft_batch_size=1
        ).split_pixels(pixels)

        self.assertTrue(torch.allclose(chunked, single, atol=1e-6, rtol=1e-6))


    def test_frequency_clip_inputs_are_finite_and_keep_shape(self):
        torch.manual_seed(2)
        normalized_images = torch.randn(2, 3, 16, 16)
        decomposer = RadialFrequencyDecomposer(0.2, 0.55)

        band_images = decomposer(normalized_images)

        self.assertEqual(band_images.shape, (2, 3, 3, 16, 16))
        self.assertTrue(torch.isfinite(band_images).all())


    def test_frequency_prototypes_follow_explicit_class_order(self):
        features = F.normalize(
            torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[0.9, 0.1], [0.1, 0.9]],
                    [[-1.0, 0.0], [0.0, -1.0]],
                    [[-0.9, -0.1], [-0.1, -0.9]],
                ]
            ),
            dim=-1,
        )
        labels = torch.tensor([3, 3, 1, 1])

        prototypes, uncertainty = compute_frequency_prototypes(
            features, labels, class_index=[1, 3]
        )

        self.assertEqual(prototypes.shape, (2, 2, 2))
        self.assertEqual(uncertainty.shape, (2, 2))
        self.assertLess(prototypes[0, 0, 0].item(), 0)
        self.assertGreater(prototypes[1, 0, 0].item(), 0)
        self.assertTrue(torch.all(uncertainty >= -1e-6))

    def test_descriptions_are_routed_by_band_with_fallback(self):
        descriptions = [
            "a bird with a long body shape",
            "a bird with striped feather texture",
            "a generic bird description",
        ]
        embeddings = torch.eye(3)
        routed = route_description_embeddings(
            descriptions,
            embeddings,
            keyword_groups=(("shape",), ("wing",), ("texture",)),
        )

        self.assertEqual(routed.shape, (3, 3))
        self.assertTrue(torch.allclose(routed[0], torch.tensor([1.0, 0.0, 0.0])))
        self.assertTrue(torch.allclose(routed[2], torch.tensor([0.0, 1.0, 0.0])))
        expected_fallback = F.normalize(torch.ones(3), dim=0)
        self.assertTrue(torch.allclose(routed[1], expected_fallback))


    def test_semantic_calibration_is_agreement_gated_and_normalized(self):
        visual = F.normalize(torch.randn(2, 3, 4), dim=-1)
        semantic = visual.clone()
        semantic[:, 1] = -visual[:, 1]
        uncertainty = torch.tensor([[0.1, 0.2, 0.5], [0.4, 0.1, 0.2]])

        calibrated, band_weights, gates, alignment = calibrate_frequency_prototypes(
            visual,
            semantic,
            uncertainty,
            semantic_weight=0.25,
            max_semantic_weight=0.65,
            uncertainty_scale=2.0,
            alignment_scale=4.0,
            fusion_temperature=1.0,
            adaptive_fusion=True,
            band_prior=torch.ones(3),
        )

        self.assertTrue(
            torch.allclose(calibrated.norm(dim=-1), torch.ones(2, 3), atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(band_weights.sum(dim=1), torch.ones(2), atol=1e-6)
        )
        self.assertTrue(torch.allclose(gates[:, 1], torch.zeros(2), atol=1e-6))
        self.assertTrue(torch.all(alignment[:, 1] < -0.999))


    def test_frequency_logits_apply_class_specific_band_weights(self):
        queries = F.normalize(
            torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]), dim=-1
        )
        prototypes = F.normalize(
            torch.tensor(
                [
                    [[1.0, 0.0], [1.0, 0.0]],
                    [[0.0, 1.0], [0.0, 1.0]],
                ]
            ),
            dim=-1,
        )
        weights = torch.tensor([[0.8, 0.2], [0.8, 0.2]])

        logits = compute_frequency_logits(queries, prototypes, weights)

        self.assertEqual(logits.shape, (1, 2))
        self.assertTrue(
            torch.allclose(logits, torch.tensor([[0.8, 0.2]]), atol=1e-6)
        )

    def test_end_to_end_frequency_tensor_pipeline(self):
        torch.manual_seed(7)
        decomposer = RadialFrequencyDecomposer(0.2, 0.55)
        support_images = torch.randn(6, 3, 16, 16)
        query_images = torch.randn(2, 3, 16, 16)
        labels = torch.tensor([0, 0, 0, 1, 1, 1])

        def dummy_encode(band_images):
            means = band_images.mean(dim=(-2, -1))
            deviations = band_images.std(dim=(-2, -1))
            return F.normalize(torch.cat((means, deviations), dim=-1), dim=-1)

        support_bands = decomposer(support_images)
        query_bands = decomposer(query_images)
        support_features = torch.stack(
            [dummy_encode(support_bands[:, band]) for band in range(3)], dim=1
        )
        query_features = torch.stack(
            [dummy_encode(query_bands[:, band]) for band in range(3)], dim=1
        )
        visual, uncertainty = compute_frequency_prototypes(
            support_features, labels, class_index=[0, 1]
        )
        semantic = F.normalize(torch.randn_like(visual), dim=-1)
        calibrated, weights, _, _ = calibrate_frequency_prototypes(
            visual,
            semantic,
            uncertainty,
            semantic_weight=0.25,
            max_semantic_weight=0.65,
            uncertainty_scale=2.0,
            alignment_scale=4.0,
            fusion_temperature=1.0,
            adaptive_fusion=True,
        )
        logits = compute_frequency_logits(query_features, calibrated, weights)
        probabilities = torch.softmax(logits, dim=-1)

        self.assertEqual(logits.shape, (2, 2))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(
            torch.allclose(probabilities.sum(dim=-1), torch.ones(2), atol=1e-6)
        )


if __name__ == "__main__":
    unittest.main()
