"""A-only feature normalization, count correction, exact controls and AUROC ties."""
import json
from pathlib import Path
import unittest

import torch

from engine.competitive_diagnostics import binary_auc
from engine.competitive_runtime import load_fusion
from engine.prototype_evaluation import identity_calibration
from models.competitive_evidence import EvidenceFusion,remove_count_prior
from models.competitive_fusion import CompetitiveFusion,evidence_features,competitive_scores,fixed_scores
from test_residual_integration import temporary_artifact_directory
from test_competitive_runtime import config


class CompetitiveEvidenceTest(unittest.TestCase):
    def features(self):
        torch.manual_seed(93)
        p={name:torch.randn(12,6).softmax(-1) for name in ('fused','cov','description')}
        q=torch.rand(6,3)
        return p,q,evidence_features(p,3,q)

    def test_raw_model_is_exact_legacy_control(self):
        _,_,x=self.features()
        legacy=CompetitiveFusion()
        with torch.no_grad():
            legacy.network[-1].weight.normal_(std=.1)
        diagnostic=EvidenceFusion(representation='raw')
        diagnostic.network.load_state_dict(legacy.network.state_dict())
        diagnostic.fit_statistics(x)
        self.assertTrue(torch.equal(legacy(x),diagnostic(x)))

    def test_standardizer_fits_A_only_and_never_updates_on_validation(self):
        _,_,x=self.features()
        x[:,5]=3.
        x.requires_grad_(True)
        model=EvidenceFusion(representation='standardized')
        with self.assertRaisesRegex(RuntimeError,'A-only'):
            model(x)
        model.fit_statistics(x)
        torch.testing.assert_close(model.evidence_mean,x.detach().mean(0))
        self.assertEqual(float(model.evidence_scale[5]),1.)
        mean,scale=model.evidence_mean.clone(),model.evidence_scale.clone()
        extreme=model.input_features(x.detach()+100.)
        self.assertTrue(bool((extreme.abs()<=5.).all()))
        self.assertTrue(torch.equal(mean,model.evidence_mean) and torch.equal(scale,model.evidence_scale))
        model(x).sum().backward()
        self.assertIsNone(x.grad)

    def test_uniform_evidence_count_prior_correction_is_size_invariant(self):
        features=[]
        for old,novel in ((10,5),(20,10)):
            p={name:torch.ones(4,old+novel)/(old+novel) for name in ('fused','cov','description')}
            x=evidence_features(p,old,torch.zeros(old+novel,3),use_support_quality=False)
            features.append(remove_count_prior(x))
        torch.testing.assert_close(features[0],features[1],atol=1e-6,rtol=1e-6)
        self.assertTrue(torch.equal(features[0][:,27:30],torch.zeros(4,3)))

    def test_zero_blend_preserves_fixed_scores_for_every_representation(self):
        p,q,x=self.features()
        fixed={'alpha_old':.8,'alpha_novel':.75,'novel_bias_delta':0.}
        cal=identity_calibration(); anchor=fixed_scores(p,3,fixed,cal)
        for name in ('raw','standardized','prior','prior_standardized'):
            model=EvidenceFusion(representation=name); model.fit_statistics(x)
            with torch.no_grad():
                model.network[-1].bias.fill_(.5)
            self.assertIs(competitive_scores(model,p,3,q,fixed,cal,0.,anchor),anchor)

    def test_standardized_predictions_are_query_batch_independent(self):
        _,_,x=self.features()
        model=EvidenceFusion(representation='prior_standardized'); model.fit_statistics(x)
        with torch.no_grad():
            model.network[-1].weight.normal_(std=.1)
            full=model(x)
            split=torch.cat((model(x[:5]),model(x[5:])))
        torch.testing.assert_close(full,split)

    def test_auc_handles_perfect_inverse_and_constant_predictions(self):
        self.assertEqual(binary_auc([0,1,2,3],[0,0,1,1]),1.)
        self.assertEqual(binary_auc([3,2,1,0],[0,0,1,1]),0.)
        self.assertEqual(binary_auc([1,1,1,1],[0,0,1,1]),.5)

    def test_diagnostic_snapshot_cannot_load_as_benchmark_model(self):
        with temporary_artifact_directory() as directory:
            path=Path(directory)/'diagnostic.npz'
            path.with_suffix('.json').write_text(json.dumps({'format':'competitive_evidence_diagnostic_v1'}))
            with self.assertRaisesRegex(ValueError,'format'):
                load_fusion(path,config(), 'any-clip-hash')


if __name__=='__main__':
    unittest.main()
