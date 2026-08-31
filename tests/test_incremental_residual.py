"""CPU regression tests for frozen low-rank FSCIL adaptation.

Run: python -m unittest discover -s tests -p test_incremental_residual.py
No CLIP checkpoint, image dataset, GPU, or network access is needed.
"""

import copy
import unittest

import torch
from torch.nn import functional as F

from models.incremental_residual import (
    LowRankResidualHead,
    _differentiable_sgd_step,
    sample_residual_episode,
)


def _toy_base():
    generator = torch.Generator().manual_seed(19)
    centers = torch.eye(4)
    features = torch.cat([
        centers[c] + 0.15 * torch.randn(12, 4, generator=generator)
        for c in range(4)
    ])
    return features, torch.arange(4).repeat_interleave(12)


class IncrementalResidualTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_zero_codes_preserve_reference_ranking_and_temperature(self):
        head = LowRankResidualHead(4, 6, 2)
        features = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]])
        votes = torch.tensor([[0.02, 0.9, 0.08], [0.0, 0.4, 0.6]])
        output = head.forward_scores(features, votes, [5, 1, 3], temperature=0.5)
        self.assertTrue(torch.equal(output.argmax(1), votes.argmax(1)))
        self.assertTrue(torch.equal(output, votes.clamp_min(1e-8).log() / 0.5))
        self.assertEqual(head.forward_residual(features, [5, 1, 3]).count_nonzero().item(), 0)

    def test_zero_gain_and_hard_residual_bound(self):
        features = torch.eye(3)
        active = LowRankResidualHead(3, 3, 3, max_delta=0.07, gain=0.8)
        active.initialize_dictionary(method="identity")
        active.codes.copy_(torch.eye(3) * 1e6)
        delta = active.forward_residual(features)
        self.assertTrue(torch.isfinite(delta).all())
        self.assertLessEqual(delta.abs().max().item(), 0.070001)
        self.assertTrue(torch.allclose(delta.diag(), torch.full((3,), 0.056)))
        disabled = LowRankResidualHead(3, 3, 3, gain=0)
        disabled.codes.fill_(1e6)
        self.assertEqual(disabled.forward_residual(features).count_nonzero().item(), 0)

    def test_only_current_codes_change_and_encoder_gradients_stay_absent(self):
        head = LowRankResidualHead(4, 7, 4, max_delta=2.)
        head.initialize_dictionary(method="identity")
        head.codes[4] = torch.tensor([0.1, -0.2, 0.3, 0.4])
        head.mark_seen([4])
        original = copy.deepcopy(head.state_dict())
        features = torch.tensor([[1., 0., 0., 0.], [1., 0.1, 0., 0.]], requires_grad=True)
        votes = torch.tensor([[0.7, 0.3], [0.7, 0.3]], requires_grad=True)
        anchors = torch.tensor([[-1., 0., 0., 0.]], requires_grad=True)
        anchor_votes = torch.tensor([[0.8, 0.2]], requires_grad=True)
        report = head.fit_session(
            features, torch.tensor([1, 1]), votes, [4, 1], [1],
            anchor_features=anchors, anchor_reference_scores=anchor_votes,
            anchor_labels=torch.tensor([4]), steps=35, lr=0.1,
            l2=0.001, old_weight=0.3,
        )
        self.assertLess(report["final_loss"], report["initial_loss"])
        self.assertGreater(head.codes[1].norm().item(), 0.)
        self.assertTrue(torch.equal(head.dictionary, original["dictionary"]))
        untouched = torch.tensor([0, 2, 3, 4, 5, 6])
        self.assertTrue(torch.equal(head.codes[untouched], original["codes"][untouched]))
        self.assertTrue(torch.equal(head.seen_mask, torch.tensor([False, True, False, False, True, False, False])))
        self.assertEqual(list(head.parameters()), [])
        for tensor in [features, votes, anchors, anchor_votes, head.dictionary, head.codes]:
            self.assertIsNone(tensor.grad)
        # A second session cannot touch either the base row or first new row.
        previous = head.codes.clone()
        head.fit_session(torch.eye(4)[2:3], torch.tensor([6]),
                         torch.tensor([[0.3, 0.3, 0.4]]), [1, 4, 6], [6], steps=4)
        self.assertTrue(torch.equal(head.codes[[1, 4]], previous[[1, 4]]))

    def test_class_mapping_and_anchor_validation_fail_before_mutation(self):
        head = LowRankResidualHead(4, 5, 2)
        features = torch.eye(4)[:1]
        scores = torch.tensor([[0.6, 0.4]])
        with self.assertRaisesRegex(ValueError, "Register"):
            head.fit_session(features, [2], scores, [0, 2], [2])
        head.mark_seen([0])
        with self.assertRaisesRegex(ValueError, "absent"):
            head.fit_session(features, [3], scores, [0, 2], [2])
        with self.assertRaisesRegex(ValueError, "unique"):
            head.fit_session(features, [2], scores, [2, 2], [2])
        with self.assertRaisesRegex(ValueError, "old classes"):
            head.fit_session(features, [2], scores, [0, 2], [2],
                             anchor_features=features, anchor_reference_scores=scores,
                             anchor_labels=[2])
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            head.fit_session(features, [2], -scores, [0, 2], [2])
        self.assertEqual(head.codes.count_nonzero().item(), 0)
        self.assertEqual(head.seen_mask.sum().item(), 1)

    def test_episode_samples_are_disjoint_reproducible_and_local(self):
        labels = torch.tensor([2, 6, 9, 12]).repeat_interleave(15)
        state = torch.random.get_rng_state().clone()
        first = sample_residual_episode(labels, 2, 2, 3, 4, old_shot=6,
                                        generator=torch.Generator().manual_seed(3))
        second = sample_residual_episode(labels, 2, 2, 3, 4, old_shot=6,
                                         generator=torch.Generator().manual_seed(3))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        for key in first:
            self.assertTrue(torch.equal(first[key], second[key]))
        si, qi = first["support_indices"], first["query_indices"]
        self.assertEqual(set(si.tolist()) & set(qi.tolist()), set())
        self.assertEqual(si.unique().numel(), 18)
        self.assertEqual(qi.unique().numel(), 16)
        self.assertTrue(torch.equal(labels[si], first["selected_class_ids"][first["support_labels"]]))
        self.assertTrue(torch.equal(labels[qi], first["selected_class_ids"][first["query_labels"]]))
        self.assertEqual(torch.bincount(first["support_labels"]).tolist(), [6, 6, 3, 3])
        with self.assertRaisesRegex(ValueError, "Not enough"):
            sample_residual_episode(labels, 2, 2, 5, 10, old_shot=6)

    def test_svd_is_finite_orthogonal_reproducible_and_reports_degeneracy(self):
        features, labels = _toy_base()
        state = torch.random.get_rng_state().clone()
        first, second = LowRankResidualHead(4, 4, 2), LowRankResidualHead(4, 4, 2)
        result = first.initialize_dictionary(features, labels, "residual_svd", shot=3,
                                             reference_shot=5, repeats=6, seed=7)
        second.initialize_dictionary(features, labels, "residual_svd", shot=3,
                                     reference_shot=5, repeats=6, seed=7)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertTrue(torch.allclose(first.dictionary.t() @ first.dictionary, torch.eye(2), atol=1e-5))
        self.assertTrue(torch.equal(first.dictionary, second.dictionary))
        self.assertEqual(result["residual_count"], 24)
        self.assertEqual(result["effective_rank"], 2)
        self.assertGreater(result["captured_energy"], 0)
        degenerate = LowRankResidualHead(4, 4, 3)
        result = degenerate.initialize_dictionary(torch.ones_like(features), labels,
                                                  "residual_svd", shot=3, repeats=2)
        self.assertEqual(result["effective_rank"], 0)
        self.assertTrue(torch.isfinite(degenerate.dictionary).all())
        self.assertTrue(torch.allclose(degenerate.dictionary.t() @ degenerate.dictionary, torch.eye(3), atol=1e-5))

    def test_meta_updates_dictionary_via_adaptation_without_updating_codes(self):
        features, labels = _toy_base()
        features.requires_grad_(True)
        head = LowRankResidualHead(4, 4, 2, max_delta=1.)
        state = torch.random.get_rng_state().clone()
        invocations = []
        callback_votes = []

        def callback(si, qi, selected, old_way):
            self.assertFalse(set(si.tolist()) & set(qi.tolist()))
            self.assertEqual(old_way, 2)
            invocations.append(selected.clone())
            # A fixed weak reference ensures dictionary improvements cannot be
            # explained by a trainable teacher or an orthogonality penalty.
            support_votes = torch.ones(len(si), len(selected), requires_grad=True)
            query_votes = torch.ones(len(qi), len(selected), requires_grad=True)
            callback_votes.extend([support_votes, query_votes])
            return {"support_scores": support_votes, "query_scores": query_votes}

        result = head.refine_dictionary(
            features, labels, callback, episodes=4, old_way=2, new_way=2,
            shot=3, old_shot=4, query_shot=3, inner_steps=2,
            inner_lr=0.4, outer_lr=0.01, old_weight=0., orth_weight=0., seed=12,
        )
        self.assertEqual(len(invocations), 4)
        self.assertEqual(result["reference"], "callback")
        self.assertGreater(result["dictionary_change"], 1e-6)
        self.assertGreater(result["loss_history"][0]["gradient_norm"], 0.)
        self.assertEqual(head.codes.count_nonzero().item(), 0)
        self.assertEqual(head.seen_mask.count_nonzero().item(), 0)
        self.assertFalse(head.dictionary.requires_grad)
        self.assertIsNone(features.grad)
        for votes in callback_votes:
            self.assertIsNone(votes.grad)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_differentiable_inner_gradient_matches_finite_difference(self):
        # Detect removal of create_graph=True: that drops the derivative of the
        # learned code with respect to the shared dictionary.
        dtype = torch.float64
        support = F.normalize(torch.tensor([[1., 0.2], [0.8, -0.1]], dtype=dtype), dim=-1)
        query = F.normalize(torch.tensor([[0.9, 0.3], [0.2, 1.]], dtype=dtype), dim=-1)
        labels = torch.tensor([1, 0])

        def adapt_and_loss(dictionary, differentiable):
            code = torch.zeros(1, 1, dtype=dtype, requires_grad=True)
            merged = torch.cat([torch.zeros_like(code), code], dim=0)
            inner = F.cross_entropy(torch.tanh(support @ dictionary @ merged.t()), torch.ones(2, dtype=torch.long))
            if differentiable:
                code = _differentiable_sgd_step(inner, code, 0.4)
            else:
                code = code - 0.4 * torch.autograd.grad(inner, code)[0]
            merged = torch.cat([torch.zeros_like(code), code], dim=0)
            return F.cross_entropy(torch.tanh(query @ dictionary @ merged.t()), labels)

        dictionary = torch.tensor([[0.7], [0.4]], dtype=dtype, requires_grad=True)
        exact = torch.autograd.grad(adapt_and_loss(dictionary, True), dictionary)[0]
        truncated = torch.autograd.grad(adapt_and_loss(dictionary, False), dictionary)[0]
        epsilon = 1e-5
        numeric = torch.zeros_like(dictionary)
        for index in range(2):
            plus, minus = dictionary.detach().clone(), dictionary.detach().clone()
            plus[index] += epsilon
            minus[index] -= epsilon
            numeric[index] = (adapt_and_loss(plus, False).detach() - adapt_and_loss(minus, False).detach()) / (2 * epsilon)
        self.assertTrue(torch.allclose(exact, numeric, atol=1e-7, rtol=1e-5))
        self.assertFalse(torch.allclose(truncated, numeric, atol=1e-5, rtol=1e-3))

    def test_checkpoint_restores_predictions_and_seen_state(self):
        head = LowRankResidualHead(4, 7, 2, max_delta=0.7, gain=0.3)
        head.codes[6] = torch.tensor([0.2, -0.1])
        head.mark_seen([0, 6])
        restored = LowRankResidualHead(4, 7, 2, max_delta=0.1, gain=1.)
        restored.load_state_dict(copy.deepcopy(head.state_dict()))
        features, _ = _toy_base()
        self.assertTrue(torch.equal(head.forward_residual(features, [6, 0]), restored.forward_residual(features, [6, 0])))
        self.assertTrue(torch.equal(head.seen_mask, restored.seen_mask))
        self.assertEqual(head.max_delta, restored.max_delta)
        self.assertEqual(head.gain, restored.gain)


if __name__ == "__main__":
    unittest.main()
