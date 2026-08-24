import unittest

import torch
import torch.nn.functional as F

from models.frequency_router import (
    TrainableFrequencyRouter,
    build_frequency_router_features,
    residual_frequency_fusion,
    sample_pseudo_fscil_episode,
)


class FrequencyRouterTest(unittest.TestCase):
    def test_router_features_are_finite_and_have_expected_shape(self):
        torch.manual_seed(1)
        original = torch.randn(7, 5)
        bands = torch.randn(7, 5, 3)

        features = build_frequency_router_features(original, bands)

        self.assertEqual(features.shape, (7, 18))
        self.assertEqual(features.dtype, torch.float32)
        self.assertTrue(torch.isfinite(features).all())

    def test_null_router_preserves_original_logits(self):
        original = torch.tensor([[0.2, 0.8], [1.0, -1.0]])
        bands = torch.randn(2, 2, 3)
        router = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(2, -1)

        mixed = residual_frequency_fusion(
            original, bands, router, max_alpha=1.0
        )

        self.assertTrue(torch.allclose(mixed, original))

    def test_router_and_class_priors_select_band_per_class(self):
        original = torch.zeros(1, 2)
        bands = torch.tensor(
            [[[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]]]
        )
        router = torch.tensor([[0.0, 1.0 / 3, 1.0 / 3, 1.0 / 3]])
        class_weights = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )

        mixed = residual_frequency_fusion(
            original,
            bands,
            router,
            class_band_weights=class_weights,
            max_alpha=1.0,
        )

        self.assertTrue(
            torch.allclose(mixed, torch.tensor([[1.0, 200.0]]), atol=1e-6)
        )

    def test_pseudo_episode_has_old_and_novel_support_sizes(self):
        labels = torch.arange(4).repeat_interleave(10)
        generator = torch.Generator().manual_seed(3)

        (
            selected,
            support_indices,
            support_labels,
            query_indices,
            query_labels,
        ) = sample_pseudo_fscil_episode(
            labels,
            class_ids=[0, 1, 2, 3],
            way=4,
            shot=2,
            query=3,
            old_way=2,
            old_shot=4,
            generator=generator,
        )

        self.assertEqual(selected.numel(), 4)
        self.assertEqual(support_indices.numel(), 12)
        self.assertEqual(query_indices.numel(), 12)
        self.assertEqual(int((support_labels < 2).sum()), 8)
        self.assertEqual(int((support_labels >= 2).sum()), 4)
        self.assertTrue(
            set(support_indices.tolist()).isdisjoint(query_indices.tolist())
        )
        self.assertTrue(torch.equal(torch.bincount(query_labels), torch.full((4,), 3)))

    def test_router_can_learn_a_consistently_helpful_band(self):
        original = torch.tensor([[0.0, 2.0], [2.0, 0.0]]).repeat(8, 1)
        helpful = torch.tensor([[3.0, 0.0], [0.0, 3.0]]).repeat(8, 1)
        bands = torch.stack((helpful, original, original), dim=-1)
        targets = torch.tensor([0, 1]).repeat(8)
        router = TrainableFrequencyRouter(
            num_bands=3,
            hidden_dim=16,
            dropout=0.0,
            null_logit_bias=0.0,
        )
        optimizer = torch.optim.Adam(router.parameters(), lr=0.05)

        with torch.no_grad():
            initial_helpful_weight = router(original, bands)[:, 1].mean().item()

        for _ in range(40):
            probabilities = router(original, bands)
            mixed = residual_frequency_fusion(
                original, bands, probabilities, max_alpha=1.0
            )
            route_target = torch.zeros_like(probabilities)
            route_target[:, 1] = 1.0
            loss = F.cross_entropy(mixed, targets) + F.kl_div(
                probabilities.clamp_min(1e-8).log(),
                route_target,
                reduction="batchmean",
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            final_probabilities = router(original, bands)
            final_mixed = residual_frequency_fusion(
                original, bands, final_probabilities, max_alpha=1.0
            )

        self.assertGreater(
            final_probabilities[:, 1].mean().item(),
            initial_helpful_weight + 0.5,
        )
        self.assertTrue(torch.equal(final_mixed.argmax(dim=-1), targets))


if __name__ == "__main__":
    unittest.main()
