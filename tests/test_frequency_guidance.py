import unittest
from unittest.mock import Mock

import torch
import torch.nn.functional as F

from models.frequency_guidance import relation_distributions, update_prototypes


class RelationsTest(unittest.TestCase):
    def setUp(self):
        gen = torch.Generator().manual_seed(207)
        self.base = F.normalize(torch.randn(7, 12, generator=gen), dim=-1)
        self.novel = F.normalize(torch.randn(4, 12, generator=gen), dim=-1)
        self.bands = F.normalize(torch.randn(7, 3, 12, generator=gen), dim=-1)
        self.new_bands = F.normalize(torch.randn(4, 3, 12, generator=gen), dim=-1)
        self.text = F.normalize(torch.randn(7, 3, 12, generator=gen), dim=-1)
        self.new_text = F.normalize(torch.randn(4, 3, 12, generator=gen), dim=-1)

    def relations(self, **kwargs):
        return relation_distributions(self.base, self.novel, self.bands, self.new_bands,
                                      self.text, self.new_text, **kwargs)

    def test_all_relations_are_finite_normalized_and_training_free(self):
        for mode in ('visual','marginal','paired'):
            rel = self.relations(mode=mode, tau=1000, tau_text=1000)
            for value in rel.values():
                if value is not None:
                    self.assertTrue(torch.isfinite(value).all())
                    torch.testing.assert_close(value.sum(-1), torch.ones_like(value.sum(-1)))
                    self.assertFalse(value.requires_grad)

    def test_pairing_matters_but_marginal_control_is_permutation_invariant(self):
        for mode in ('paired','marginal'):
            before = self.relations(mode=mode)['frequency']
            after = self.relations(mode=mode, permutation=(2,0,1))['frequency']
            if mode == 'paired':
                self.assertGreater(float((before-after).abs().max()), .01)
            else:
                torch.testing.assert_close(before, after)

    def test_eta_zero_equals_visual_and_uniform_text_is_temperature_control(self):
        a = self.relations(mode='paired', eta=0)['frequency']
        b = self.relations(mode='visual')['frequency']
        torch.testing.assert_close(a,b)
        uniform = self.relations(uniform_semantics=True)['frequency']
        expected = (8 * torch.einsum('csd,bsd->csb',self.new_bands,self.bands)).softmax(-1).mean(1)
        torch.testing.assert_close(uniform,expected)

    def test_zero_rho_calls_original_function_and_zero_shift_preserves_raw(self):
        ref = Mock(return_value=self.novel)
        rel = self.relations()
        result = update_prototypes(self.base,self.novel,rel,0,reference=ref)
        self.assertIs(result,self.novel)
        ref.assert_called_once_with(self.base,self.novel)
        self.assertIs(update_prototypes(self.base,self.novel,rel,1,shift=0),self.novel)

    def test_selected_weights_and_guided_vectors_are_normalized(self):
        out = update_prototypes(self.base,self.novel,self.relations(),.75)
        torch.testing.assert_close(out.norm(dim=-1),torch.ones(4))
        self.assertFalse(out.requires_grad)

    def test_single_band_really_ignores_other_bands(self):
        before = self.relations(band_weights=(1,0,0))['frequency']
        self.new_bands[:,1:] *= -1
        self.new_text[:,1:] *= -1
        after = self.relations(band_weights=(1,0,0))['frequency']
        torch.testing.assert_close(before,after)


if __name__ == '__main__':
    unittest.main()
