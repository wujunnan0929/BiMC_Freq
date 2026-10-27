"""Typed native-space completion, class isolation and missing-evidence tests."""

import copy
import unittest

import torch

from models.multievidence_completion import fit_model, predict_model


def fixture(classes=8, shots=10):
    generator = torch.Generator().manual_seed(312)
    centers = torch.randn(classes, 5, generator=generator)
    targets = torch.arange(classes).repeat_interleave(shots)
    image = centers[targets] + .25 * torch.randn(classes*shots, 5, generator=generator)
    auxiliary = image @ torch.randn(5, 7, generator=generator) + .15 * torch.randn(classes*shots, 7, generator=generator)
    text = centers @ torch.randn(5, 6, generator=generator) + .2 * torch.randn(classes, 6, generator=generator)
    return {'final': image, 'layer_3': auxiliary}, targets, {'shape': text}, list(range(classes))


def model(visual, labels, text, ids, **extra):
    return fit_model(visual, labels, text, ids, target_rank=3, auxiliary_rank=2,
                     text_rank=2, **extra)


class MultiEvidenceCompletionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_native_dimensions_and_structural_noise_zeros(self):
        visual, labels, text, ids = fixture()
        state = model(visual, labels, text, ids)
        self.assertEqual(state['basis'].shape, (5, 3))
        self.assertEqual([b['input_dimension'] for b in state['blocks']], [5, 7, 6])
        self.assertEqual(state['H'].shape, (7, 3))
        torch.testing.assert_close(state['H'][:3], torch.eye(3, dtype=torch.float64), atol=0, rtol=0)
        self.assertFalse(torch.any(state['bias'][:3]))
        self.assertFalse(torch.any(state['Q'][:3]))
        self.assertFalse(torch.any(state['Q'][:, :3]))
        self.assertFalse(torch.any(state['W'][5:]))
        self.assertFalse(torch.any(state['W'][:, 5:]))
        self.assertGreater(float(state['W'][:3, 3:5].abs().sum()), 0)
        for name in ('between', 'Q', 'W'):
            self.assertGreaterEqual(float(torch.linalg.eigvalsh(state[name]).min()), -1e-10)
        output = predict_model(state, visual, labels, text, ids)
        self.assertEqual(output.shape, (8, 5))
        self.assertTrue(torch.isfinite(output).all())

    def test_crossfit_holds_each_available_class_out_of_its_mapping(self):
        visual, labels, text, ids = fixture()
        mask = torch.tensor([True, False, True, True, False, True, True, True])
        state = model(visual, labels, text, ids, text_masks={'shape': mask}, crossfit_folds=3)
        seen = {}
        for record in state['crossfit_trace']:
            self.assertFalse(set(record['train_class_ids']) & set(record['heldout_class_ids']))
            seen.setdefault(record['block'], []).extend(record['heldout_class_ids'])
        self.assertEqual(sorted(seen['layer_3']), ids)
        self.assertEqual(sorted(seen['shape']), torch.where(mask)[0].tolist())
        self.assertEqual(state['text_available_class_counts']['shape'], int(mask.sum()))

    def test_unrequested_class_images_and_text_do_not_affect_fit(self):
        visual, labels, text, ids = fixture()
        fitting = ids[:5]
        first = model(visual, labels, {'shape': text['shape'][:5]}, fitting)
        changed = {name: values.clone() for name, values in visual.items()}
        for values in changed.values():
            values[labels >= 5] *= -100
        second = model(changed, labels, {'shape': text['shape'][:5]}, fitting)
        for name in ('basis', 'H', 'bias', 'mean', 'between', 'Q', 'W'):
            self.assertTrue(torch.equal(first[name], second[name]), name)

    def test_exact_duplicates_are_canonicalized_without_extra_evidence(self):
        visual, labels, text, ids = fixture()
        first = model(visual, labels, text, ids)
        doubled_visual = {**visual, 'final_copy': visual['final'].clone()}
        doubled_text = {**text, 'shape_copy': text['shape'].clone()}
        second = model(doubled_visual, labels, doubled_text, ids)
        self.assertEqual(second['visual_aliases'], {'final_copy': 'final'})
        self.assertEqual(second['text_aliases'], {'shape_copy': 'shape'})
        self.assertEqual(first['observation_dimension'], second['observation_dimension'])
        self.assertEqual(first['fit_sample_count'], second['fit_sample_count'])
        a = predict_model(first, visual, labels, text, ids)
        b = predict_model(second, doubled_visual, labels, doubled_text, ids)
        self.assertTrue(torch.equal(a, b))
        wrong = {**doubled_visual, 'final_copy': doubled_visual['final_copy'] + .01}
        with self.assertRaisesRegex(ValueError, 'duplicate visual'):
            predict_model(second, wrong, labels, doubled_text, ids)

    def test_unknown_text_is_removed_per_class_not_observed_as_zero(self):
        visual, labels, text, ids = fixture()
        masks = {'shape': torch.tensor([True, False, True, True, False, True, True, True])}
        state = model(visual, labels, text, ids, text_masks=masks)
        current = {'shape': text['shape'].clone()}
        current['shape'][1] = 1e6
        current_mask = {'shape': torch.tensor([True, False, True, True, True, True, True, True])}
        result = predict_model(state, visual, labels, current, ids, text_masks=current_mask)
        removed = predict_model(state, visual, labels, current, ids, omitted_text=['shape'])
        self.assertTrue(torch.equal(result[1], removed[1]))
        self.assertFalse(torch.equal(result[0], removed[0]))
        dropped = model(visual, labels, text, ids, text_masks={'shape': torch.arange(8) == 0})
        self.assertEqual(dropped['dropped_text_blocks'], ['shape'])
        self.assertFalse(any(b['kind'] == 'text' for b in dropped['blocks']))

    def test_omitting_all_text_recovers_same_visual_only_model(self):
        visual, labels, text, ids = fixture()
        visual = {'final': visual['final']}
        first = model(visual, labels, text, ids)
        second = model(visual, labels, {}, ids)
        a = predict_model(first, visual, labels, text, ids, omitted_text=['shape'])
        b = predict_model(second, visual, labels, {}, ids)
        self.assertTrue(torch.equal(a, b))

    def test_more_support_approaches_raw_final_mean_without_static_final_noise(self):
        visual, labels, text, ids = fixture()
        state = model(visual, labels, text, ids)
        center = torch.tensor([[1.5, -.7, .4, .2, .9]])
        auxiliary = torch.ones(1, 7)
        semantic = {'shape': -torch.ones(1, 6)}
        small = predict_model(state, {'final': center, 'layer_3': auxiliary}, torch.tensor([99]), semantic, [99])
        large = predict_model(state, {'final': center.repeat(10000, 1), 'layer_3': auxiliary.repeat(10000, 1)},
                              torch.full((10000,), 99), semantic, [99])
        self.assertLess(float((large-center).norm()), float((small-center).norm()))
        self.assertLess(float((large-center).norm()), .001)

    def test_huge_text_error_removes_its_influence(self):
        visual, labels, text, ids = fixture()
        visual = {'final': visual['final']}
        state = model(visual, labels, text, ids)
        weakened = copy.deepcopy(state)
        weakened['Q'][3:,3:] += 1e8 * torch.eye(state['observation_dimension']-3, dtype=torch.float64)
        target = predict_model(state, visual, labels, text, ids, omitted_text=['shape'])
        observed = predict_model(weakened, visual, labels, text, ids)
        torch.testing.assert_close(observed, target, atol=1e-5, rtol=1e-5)

    def test_one_shot_is_finite_and_model_and_autograd_stay_frozen(self):
        visual, labels, text, ids = fixture()
        state = model(visual, labels, text, ids)
        before = {k:v.clone() for k,v in state.items() if torch.is_tensor(v)}
        rows = torch.arange(0, len(labels), 10)
        supports = {k:v[rows].clone().requires_grad_(True) for k,v in visual.items()}
        output = predict_model(state, supports, labels[rows], text, ids)
        self.assertFalse(output.requires_grad)
        self.assertTrue(torch.isfinite(output).all())
        for key, values in before.items():
            self.assertTrue(torch.equal(state[key], values), key)
            self.assertFalse(state[key].requires_grad)
        self.assertEqual(state['retained_training_samples'], 0)

    def test_target_basis_is_shared_across_semantic_variants(self):
        visual, labels, text, ids = fixture()
        first = model(visual, labels, text, ids)
        changed = {'color': text['shape'].roll(1, 0)}
        second = model(visual, labels, changed, ids, target_basis=first['basis'])
        self.assertTrue(torch.equal(first['basis'], second['basis']))
        self.assertTrue(torch.equal(first['between'], second['between']))

    def test_requested_class_order_and_support_order_are_invariant(self):
        visual, labels, text, ids = fixture()
        state = model(visual, labels, text, ids)
        reference = predict_model(state, visual, labels, text, ids)
        reverse = torch.arange(len(labels)-1, -1, -1)
        output = predict_model(state, {k:v[reverse] for k,v in visual.items()}, labels[reverse],
                               {'shape': text['shape'].flip(0)}, ids[::-1])
        torch.testing.assert_close(output.flip(0), reference, atol=1e-6, rtol=1e-6)

    def test_invalid_inputs_and_structural_covariance_corruption_are_rejected(self):
        visual, labels, text, ids = fixture()
        for option in ({'ridge': 0}, {'shrinkage': 2}, {'solver_ridge': -1}, {'crossfit_folds': 1}):
            with self.subTest(option=option), self.assertRaises(ValueError):
                model(visual, labels, text, ids, **option)
        with self.assertRaisesRegex(ValueError, 'Text blocks'):
            model(visual, labels, {'shape': text['shape'][:2]}, ids)
        with self.assertRaisesRegex(ValueError, 'positive within-class'):
            rows = torch.arange(0, len(labels), 10)
            model({k:v[rows] for k,v in visual.items()}, labels[rows], text, ids)
        state = model(visual, labels, text, ids)
        state['Q'][0,0] = 1e-6
        with self.assertRaisesRegex(ValueError, 'exact zero final'):
            predict_model(state, visual, labels, text, ids)


if __name__ == '__main__':
    unittest.main()
