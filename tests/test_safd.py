import unittest
from unittest.mock import patch

import torch

from models.safd import SAFDFrequencyDecomposer


class SAFDTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(127)

    def make(self, **kwargs):
        options = dict(low_terms=2, middle_terms=5, angles=8, max_signals=32)
        options.update(kwargs)
        return SAFDFrequencyDecomposer(**options)

    def test_components_reconstruct_and_preserve_color_on_rectangular_input(self):
        pixels = torch.rand(3, 3, 17, 22)
        components = self.make().split_pixels(pixels)
        self.assertEqual(components.shape, (3, 3, 3, 17, 22))
        torch.testing.assert_close(components.sum(1), pixels, atol=2e-6, rtol=1e-6)
        torch.testing.assert_close(components[:, 0].mean((-2, -1)),
                                   pixels.mean((-2, -1)), atol=2e-6, rtol=1e-6)
        self.assertLess(components[:, 1:].mean((-2, -1)).abs().max().item(), 2e-6)

    def test_nested_orthogonal_projections_reduce_error(self):
        pixels = torch.rand(2, 3, 19, 20)
        parts = self.make().split_pixels(pixels)
        coarse_error = (pixels - parts[:, 0]).square().sum()
        middle_error = parts[:, 2].square().sum()
        self.assertLess(middle_error, coarse_error)
        for i, j in ((0, 1), (0, 2), (1, 2)):
            self.assertLess((parts[:, i] * parts[:, j]).sum().abs().item(), 2e-4)

    def test_no_fft_calls_and_constant_has_no_detail(self):
        pixels = torch.tensor([0.2, 0.5, 0.8]).view(1, 3, 1, 1).expand(2, 3, 18, 18)
        with patch('torch.fft.fft2', side_effect=AssertionError('FFT called')), \
                patch('torch.fft.fft', side_effect=AssertionError('FFT called')), \
                patch('torch.fft.ifft', side_effect=AssertionError('FFT called')):
            parts = self.make().split_pixels(pixels)
        torch.testing.assert_close(parts[:, 0], pixels, atol=2e-6, rtol=1e-6)
        self.assertLess(parts[:, 1:].abs().max().item(), 2e-6)

    def test_chunking_permutation_and_unrelated_images_do_not_change_views(self):
        pixels = torch.rand(3, 3, 19, 20)
        batched = self.make(batch_size=3).split_pixels(pixels)
        alone = self.make(batch_size=1).split_pixels(pixels)
        permuted = self.make(batch_size=2).split_pixels(pixels[[2, 0, 1]])
        torch.testing.assert_close(batched, alone, atol=2e-5, rtol=1e-4)
        torch.testing.assert_close(batched[[2, 0, 1]], permuted, atol=2e-5, rtol=1e-4)

    def test_hilbert_operator_matches_analytic_sinusoid_even_and_odd(self):
        for length in (17, 18):
            module = self.make()
            _, _, _, hilbert = module._grid(length, torch.device('cpu'))
            theta = torch.arange(length) * (2 * torch.pi / length)
            torch.testing.assert_close(torch.cos(3 * theta) @ hilbert,
                                       torch.sin(3 * theta), atol=3e-6, rtol=1e-5)
            torch.testing.assert_close(torch.ones(length) @ hilbert,
                                       torch.zeros(length), atol=1e-6, rtol=0)

    def test_adaptive_poles_capture_localized_rational_signal(self):
        # A single nonzero Szego pole has a broad ordinary Fourier spectrum.
        # Its real TM span should capture more energy than two fixed harmonics.
        length = 128
        theta = torch.arange(length) * (2 * torch.pi / length)
        z = torch.polar(torch.ones(length), theta)
        signal = (z * (1 - 0.8 ** 2) ** 0.5 / (1 - 0.8 * z)).real
        signals = signal[None, None].expand(1, 4, -1)
        adaptive = self.make(low_terms=1, middle_terms=2)._axis_basis(signals)[1]
        fixed = self.make(low_terms=1, middle_terms=2, radii=(0.0,))._axis_basis(signals)[1]
        def error(basis):
            return (signals - signals @ basis @ basis.transpose(1, 2)).square().sum()
        self.assertLess(error(adaptive), 0.05 * error(fixed))

    def test_natural_views_dtype_range_blend_and_detail_cap(self):
        module = self.make(high_enhance=10, max_detail_rms=0.01)
        pixels = 0.2 + 0.6 * torch.rand(2, 3, 19, 20)
        normalized = ((pixels - module.mean) / module.std).half()
        views = module(normalized)
        restored = views.float() * module.std[:, None] + module.mean[:, None]
        actual = normalized.float() * module.std + module.mean
        parts = module.split_pixels(actual)
        self.assertEqual(views.dtype, torch.float16)
        self.assertTrue(torch.isfinite(views).all())
        self.assertGreaterEqual(restored.min().item(), -0.001)
        self.assertLessEqual(restored.max().item(), 1.001)
        torch.testing.assert_close(restored[:, 0], torch.lerp(parts[:, 0], actual, .25).clamp(0, 1),
                                   atol=4e-4, rtol=1e-3)
        rms = (restored[:, 2] - actual).square().mean((-2, -1)).sqrt()
        self.assertLessEqual(rms.max().item(), 0.0102)

    def test_original_bypasses_decomposition_exactly(self):
        module = self.make(view_mode='original')
        images = torch.randn(2, 3, 8, 8).half()
        with patch.object(module, 'split_pixels', side_effect=AssertionError('Decomposed')):
            self.assertTrue(torch.equal(module(images), images[:, None].expand(-1, 3, -1, -1, -1)))

    def test_tiny_images_empty_batches_and_cache_device_moves(self):
        for h, w in ((2, 3), (3, 2), (4, 4)):
            module = self.make()
            pixels = torch.rand(1, 3, h, w)
            torch.testing.assert_close(module.split_pixels(pixels).sum(1), pixels)
            self.assertTrue(module._grids)
            module.to('cpu')
            self.assertFalse(module._grids)
        self.assertEqual(self.make().split_pixels(torch.empty(0, 3, 8, 8)).shape, (0, 3, 3, 8, 8))

    def test_invalid_settings_and_inputs_fail(self):
        for settings in (dict(low_terms=5), dict(angles=0), dict(batch_size=0),
                         dict(radii=(1.,)), dict(radii=(.3, .3)),
                         dict(original_blend=(1.1, 0)), dict(high_enhance=float('nan'))):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.make(**settings)
        for pixels in (torch.rand(2, 1, 8, 8), torch.rand(2, 3, 1, 8),
                       torch.full((1, 3, 8, 8), float('nan'))):
            with self.assertRaises(ValueError):
                self.make().split_pixels(pixels)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_autocast_and_cpu_agree(self):
        pixels = torch.rand(2, 3, 19, 20)
        cpu = self.make().split_pixels(pixels)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            gpu = self.make().cuda().split_pixels(pixels.cuda()).cpu()
        self.assertEqual(gpu.dtype, torch.float32)
        torch.testing.assert_close(cpu, gpu, atol=3e-5, rtol=2e-4)


if __name__ == '__main__':
    unittest.main()
