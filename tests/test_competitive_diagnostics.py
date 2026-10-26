"""Class-count controls retain the reserved class/image split and selection scope."""
import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from engine import competitive_diagnostics as diagnostic
from engine import competitive_training as training
from engine.competitive_evaluation import prepare_records,identity_fixed
from engine.prototype_evaluation import prepare_context,validation_records,identity_calibration
from test_safd_metric_training import fixture
from test_competitive_training import settings
from test_residual_integration import temporary_artifact_directory
from tools import diagnose_competitive_evidence as cli


class CompetitiveDiagnosticTest(unittest.TestCase):
    def example(self):
        arrays,cfg,_=fixture(); options=settings()
        cfg,protocol,pool=prepare_context(arrays,cfg,options.seed,options.sequences,options.teacher_shot)
        original=validation_records(arrays,cfg,protocol)
        train,trace=training.training_episodes(arrays,cfg,pool,options,identity_calibration())
        pairs={(row['num_old'],len(row['class_ids'])-row['num_old']) for row in trace}
        matched,report=diagnostic.matched_records(arrays,cfg,protocol,original,pairs,options.seed)
        prepare_records(original,cfg,identity_calibration()); prepare_records(matched,cfg,identity_calibration())
        return arrays,cfg,options,protocol,original,train,matched,report

    def test_matched_tasks_use_V_novel_classes_and_original_support_query_indices(self):
        arrays,cfg,options,protocol,original,train,matched,report=self.example()
        self.assertEqual(len(matched),2)
        original_queries={i for row in original for q in row['query_indices'] for i in q}
        for row in matched:
            self.assertEqual(row['num_old'],1)
            self.assertEqual(len(row['class_ids']),2)
            self.assertTrue(set(row['class_ids'][1:])<=set(protocol['pseudo_validation_class_ids']))
            self.assertTrue(set(row['class_ids'][:1])<=set(protocol['pseudo_base_class_ids']))
            for support,query in zip(row['support_indices'],row['query_indices']):
                self.assertFalse(set(support)&set(query))
                self.assertTrue(set(query)<=original_queries)
        repeated,again=diagnostic.matched_records(arrays,cfg,protocol,original,[(1,1)],options.seed)
        self.assertEqual(report['sha256'],again['sha256'])
        self.assertTrue(all(torch.equal(a['reference_votes'],b['reference_votes']) for a,b in zip(matched,repeated)))

    def test_full_way_reconstruction_matches_original_reference_up_to_permutation(self):
        arrays,cfg,options,protocol,original,_,_,_=self.example()
        source=max((row for row in original if row['sequence']==0),key=lambda row:len(row['class_ids']))
        old=source['num_old']; novel=len(source['class_ids'])-old
        rows,report=diagnostic.matched_records(arrays,cfg,protocol,original,[(old,novel)],options.seed)
        row=rows[0]; positions=report['tasks'][0]['source_positions']
        q=torch.cat([torch.where(source['targets']==j)[0] for j in positions])
        expected=source['reference_votes'][q][:,positions]
        torch.testing.assert_close(row['reference_votes'],expected,atol=2e-6,rtol=1e-5)

    def test_teacher_features_are_unused_and_original_bank_is_not_mutated(self):
        arrays,cfg,options,protocol,original,_,matched,_=self.example()
        snapshots=[row['reference_prototypes'].clone() for row in original]
        for row in original:
            row['teacher_prototypes'].fill_(float('nan'))
        repeated,_=diagnostic.matched_records(arrays,cfg,protocol,original,[(1,1)],options.seed)
        for a,b in zip(matched,repeated):
            torch.testing.assert_close(a['reference_votes'],b['reference_votes'])
        self.assertTrue(all(torch.equal(row['reference_prototypes'],snap) for row,snap in zip(original,snapshots)))

    def test_original_scale_is_not_scored_during_checkpoint_selection(self):
        _,cfg,options,_,original,train,matched,_=self.example()
        cal=identity_calibration(); fixed=identity_fixed(cfg)
        baselines={name:diagnostic.bank_baselines(bank,cfg,cal,fixed) for name,bank in (('matched',matched),('original',original))}
        args=SimpleNamespace(device='cpu',standardization_floor=1e-4,standardization_clip=5.,probe_steps=2,probe_lr=.01)
        from engine import prototype_evaluation as evaluation
        real_evaluate=evaluation.evaluate_records; real_optimize=training.optimize
        phase={'fitting':False}
        def optimize(*a,**kw):
            phase['fitting']=True
            try:
                return real_optimize(*a,**kw)
            finally:
                phase['fitting']=False
        def evaluate(records,*a,**kw):
            if phase['fitting']:
                self.assertIs(records,matched)
            return real_evaluate(records,*a,**kw)
        with patch.object(training,'optimize',side_effect=optimize), \
             patch.object(evaluation,'evaluate_records',side_effect=evaluate),contextlib.redirect_stdout(io.StringIO()):
            model,report=cli.run_trial(train,matched,original,cfg,cal,fixed,baselines,options,'standardized',args)
        self.assertFalse(report['original_scale_used_for_selection'])
        self.assertFalse(report['all_base_refit_performed'])
        self.assertEqual(report['normalizer']['source'],'A_fitting_queries_only')
        self.assertTrue(all(not value.requires_grad for value in model.parameters()))
        self.assertIn('macro_task_auc',report['group_probe']['evaluation']['original'])

    def test_cli_fresh_directory_check_precedes_loading_missing_inputs(self):
        with temporary_artifact_directory() as directory:
            with self.assertRaisesRegex(ValueError,'fresh'):
                cli.main(['--cache','missing.npz','--anchor-model','missing.npz','--output-dir',directory])


if __name__=='__main__':
    unittest.main()
