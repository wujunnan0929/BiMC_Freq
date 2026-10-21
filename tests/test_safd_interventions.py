import unittest
from unittest.mock import patch

import torch

from models.safd_interventions import (
    NestedInterventionViews, counterfactual_responses, encode_intervention_features,
)


class SAFDInterventionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(518)

    def make(self, source='safd', **kwargs):
        return NestedInterventionViews(source, low_terms=2, middle_terms=5,
                                       angles=8, max_signals=24, **kwargs)

    def images(self, module, n=3):
        pixels = .15 + .7 * torch.rand(n, 3, 19, 20)
        return (pixels - module.mean) / module.std

    def test_safd_nested_views_match_components_without_blend_or_enhancement(self):
        module = self.make()
        images = self.images(module)
        with patch('torch.fft.fft2', side_effect=AssertionError('SAFD used FFT')):
            views = module(images)
        pixels = images * module.std + module.mean
        parts = module.decomposer.split_pixels(pixels)
        expected = torch.stack((parts[:, 0], parts[:, 0] + parts[:, 1]), 1).clamp(0, 1)
        recovered = views[:, 1:] * module.std[:, None] + module.mean[:, None]
        self.assertTrue(torch.equal(views[:, 0], images))
        torch.testing.assert_close(recovered, expected, atol=2e-6, rtol=1e-6)
        self.assertTrue((module.pixel_rmse(views) >= 0).all())
        self.assertEqual(tuple(views.shape), (3, 3, 3, 19, 20))

    def test_safd_sample_permutation_and_batching_do_not_mix_images(self):
        module = self.make(batch_size=3)
        images = self.images(module)
        full = module(images)
        alone = torch.cat([module(images[i:i+1]) for i in range(3)])
        permutation = module(images[[2, 0, 1]])
        torch.testing.assert_close(full, alone, atol=2e-5, rtol=1e-4)
        torch.testing.assert_close(full[[2, 0, 1]], permutation, atol=2e-5, rtol=1e-4)

    def test_fft_control_constructs_nested_components(self):
        module = self.make('fft', fft_device='cpu')
        images = self.images(module)
        parts = module.decomposer.split_pixels(images * module.std + module.mean)
        views = module(images)
        recovered = views[:, 1:] * module.std[:, None] + module.mean[:, None]
        expected = torch.stack((parts[:, 0], parts[:, 0] + parts[:, 1]), 1).clamp(0, 1)
        torch.testing.assert_close(recovered, expected, atol=2e-6, rtol=1e-6)

    def test_repeat_zero_response_and_no_auxiliary_encoding(self):
        module = self.make('repeat')
        images = self.images(module)
        calls = []
        def encode(value):
            calls.append(len(value))
            return value.mean((-2, -1)) + torch.tensor([1., 2., 3.])
        features, rms = encode_intervention_features(images, module, encode,
            lambda _: self.fail('Repeat control encoded auxiliaries'))
        self.assertEqual(calls, [3])
        self.assertTrue(torch.equal(features[:, 0], features[:, 1]))
        self.assertTrue(torch.equal(features[:, 0], features[:, 2]))
        self.assertEqual(counterfactual_responses(features).abs().max().item(), 0)
        self.assertEqual(rms.abs().max().item(), 0)

    def test_signed_responses_telescope_and_keep_amplitudes(self):
        features = torch.randn(4, 3, 7)
        response = counterfactual_responses(features)
        torch.testing.assert_close(response[:, 0], features[:, 2] - features[:, 1])
        torch.testing.assert_close(response.sum(1), features[:, 0] - features[:, 1])
        torch.testing.assert_close(counterfactual_responses(2 * features), 2 * response)

    def test_ordinary_views_finite_and_encoding_is_gradient_free(self):
        module = self.make('augmentation')
        images = self.images(module).requires_grad_(True)
        weight = torch.randn(3, 6, requires_grad=True)
        calls = []
        def encode(value):
            calls.append(len(value))
            return value.mean((-2, -1)) @ weight
        features, rms = encode_intervention_features(images, module, encode)
        self.assertEqual(calls, [3, 3, 3])
        self.assertFalse(features.requires_grad)
        self.assertFalse(rms.requires_grad)
        self.assertEqual(tuple(rms.shape), (3, 2))
        torch.testing.assert_close(features.norm(dim=-1), torch.ones(3, 3))
        self.assertIn('compute', NestedInterventionViews.__doc__)

    def test_invalid_shapes_nonfinite_values_and_zero_encoder_are_rejected(self):
        module = self.make('repeat')
        for images in (torch.ones(2, 4, 10, 10), torch.full((2, 3, 10, 10), float('nan'))):
            with self.assertRaises(ValueError):
                module(images)
        with self.assertRaisesRegex(ValueError, 'nonzero'):
            encode_intervention_features(torch.zeros(1, 3, 10, 10), module,
                                         lambda _: torch.zeros(1, 4))


if __name__ == '__main__':
    unittest.main()
