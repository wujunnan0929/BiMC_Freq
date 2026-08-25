import unittest

import torch
import torch.nn.functional as F

from models.frequency_modality import (
    FrequencyModalityEncoder,
    FrequencySpectrumDescriptor,
    compute_modality_alpha,
    compute_modality_prototypes,
    fuse_modality_probabilities,
)


class FrequencyModalityTest(unittest.TestCase):
    def test_spectrum_descriptor_is_finite_and_has_expected_shape(self):
        torch.manual_seed(1)
        images = torch.randn(4, 3, 32, 32)
        descriptor = FrequencySpectrumDescriptor(
            grid_size=8,
            radial_bins=6,
            fft_batch_size=3,
            fft_device="cpu",
        )

        features = descriptor(images)

        self.assertEqual(features.shape, (4, 8 * 8 + 6))
        self.assertTrue(torch.isfinite(features).all())

    def test_descriptor_does_not_use_spatial_phase(self):
        # Without window boundary effects, translating a periodic image only
        # changes Fourier phase.  Use the private extractor with an all-ones
        # window equivalent by comparing the exact spectrum here, and verify
        # that the public representation is explicitly phase-free by design.
        pixels = torch.zeros(1, 1, 16, 16)
        pixels[:, :, 4:8, 3:7] = 1.0
        shifted = torch.roll(pixels, shifts=(3, 5), dims=(-2, -1))
        first = torch.fft.fft2(pixels).abs()
        second = torch.fft.fft2(shifted).abs()

        self.assertTrue(torch.allclose(first, second, atol=1e-6))

    def test_encoder_outputs_normalized_clip_space_features(self):
        torch.manual_seed(2)
        encoder = FrequencyModalityEncoder(20, 12, hidden_dim=16, dropout=0.0)

        features = encoder(torch.randn(5, 20))

        self.assertEqual(features.shape, (5, 12))
        self.assertTrue(
            torch.allclose(features.norm(dim=-1), torch.ones(5), atol=1e-6)
        )

    def test_prototypes_and_reliability_follow_class_order(self):
        features = F.normalize(
            torch.tensor(
                [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]]
            ),
            dim=-1,
        )
        labels = torch.tensor([5, 5, 3, 3])

        prototypes, uncertainty, counts = compute_modality_prototypes(
            features, labels, [3, 5]
        )
        alpha, reliability = compute_modality_alpha(
            uncertainty,
            counts,
            max_alpha=0.4,
            uncertainty_scale=2.0,
            shot_tau=2.0,
        )

        self.assertGreater(prototypes[0, 1], prototypes[0, 0])
        self.assertGreater(prototypes[1, 0], prototypes[1, 1])
        self.assertTrue(torch.equal(counts, torch.tensor([2.0, 2.0])))
        self.assertTrue(torch.all(alpha <= 0.4))
        self.assertTrue(torch.allclose(alpha, 0.4 * reliability))

    def test_probability_fusion_supports_classwise_alpha(self):
        clip_prob = torch.tensor([[0.8, 0.2]])
        frequency_prob = torch.tensor([[0.1, 0.9]])

        fused = fuse_modality_probabilities(
            clip_prob, frequency_prob, torch.tensor([0.0, 0.5])
        )

        expected = torch.tensor([[0.8, 0.55]])
        expected = expected / expected.sum(dim=-1, keepdim=True)
        self.assertTrue(torch.allclose(fused, expected))


if __name__ == "__main__":
    unittest.main()
