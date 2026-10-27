"""Actual attribute transfer study on a masked toy-CLIP grounding cache."""

import contextlib
import io
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import torch

from engine import attribute_transfer_experiments as engine
from engine.completion_calibration import completion_protocol
from test_attribute_grounding import write_synthetic_grounding
from test_multilayer_completion_experiments import write_synthetic_cache
from tools.package_attribute_transfer_results import package_results


@contextlib.contextmanager
def directory():
    parent=(Path(__file__).resolve().parents[1]/'.codex_tmp'/'attribute_engine_tests').resolve()
    parent.mkdir(parents=True,exist_ok=True)
    folder=parent/uuid4().hex
    folder.mkdir()
    try:
        yield folder
    finally:
        if folder.resolve().parent!=parent:
            raise RuntimeError('Unsafe synthetic experiment cleanup target')
        shutil.rmtree(folder)


def write_transfer_fixture(folder):
    cache,arrays,metadata=write_synthetic_cache(folder)
    grounding,grounded,groundmeta=write_synthetic_grounding(folder,metadata)
    return cache,grounding,arrays,metadata,grounded,groundmeta


def run(cache,grounding,output,**extra):
    options={'fit_seeds':(2,),'sequences':2,'inner_sequences':1,
             'blend_grid':(0.,.25,1.),'offset_grid':(-.1,0.,.1),'topk':2,'threads':1}
    options.update(extra)
    with contextlib.redirect_stdout(io.StringIO()):
        return engine.run_experiments(cache,grounding,output,**options)


class AttributeTransferExperimentsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads()
        torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_complete_matrix_has_exact_identity_replayable_offsets_and_all_blend_geometry(self):
        with directory() as folder:
            cache,grounding,arrays,_,_,_=write_transfer_fixture(folder)
            root=folder/'study'
            result=run(cache,grounding,root)
            self.assertEqual((result['planned_runs'],result['completed_runs'],result['failed_runs'],result['pending_runs']),
                             (14,14,0,0))
            self.assertTrue(result['study_complete'])
            self.assertFalse(result['benchmark_evaluation_performed'])
            self.assertEqual(len({row['protocol_sha256'] for row in result['runs']}),1)
            for row in result['runs']:
                target=root/row['output_dir']
                protocol=json.loads((target/'protocol.json').read_text())
                fitting=set(protocol['fitting_indices'])
                reserved={i for group in protocol['reserved_pseudo_base_query_indices'].values() for i in group}
                self.assertFalse(fitting & reserved)
                self.assertEqual(set(arrays['labels'][list(fitting)]),set(protocol['pseudo_base_class_ids']))
                offset=json.loads((target/'offset_fit.json').read_text())
                self.assertFalse(offset['uses_outer_V_images_or_labels'])
                inner=offset['inner_protocol']
                allowed=set(inner['allowed_outer_fitting_indices'])
                self.assertEqual(allowed,fitting)
                self.assertTrue(set(inner['fitting_indices'])<=allowed)
                self.assertTrue(set(inner['pseudo_base_class_ids'])<=set(protocol['pseudo_base_class_ids']))
                self.assertTrue(set(inner['pseudo_validation_class_ids'])<=set(protocol['pseudo_base_class_ids']))
                for sequence in inner['sequences']:
                    for support,query in zip(sequence['support_indices'],sequence['query_indices']):
                        self.assertTrue(set(support+query)<=allowed)
                        self.assertFalse(set(support)&set(query))
                with np.load(target/'predictions.npz',allow_pickle=False) as saved:
                    self.assertEqual(str(saved['candidate_id'][0]),'identity')
                    self.assertEqual(saved['candidate_blend'][0],0.)
                    self.assertEqual(saved['candidate_offset'][0],0.)
                    np.testing.assert_array_equal(saved['candidate_prediction'][0],saved['reference_prediction'])
                    replay=np.where(saved['candidate_best_novel_logscore']+saved['candidate_offset'][:,None]
                                    >saved['candidate_best_old_logscore'],
                                    saved['candidate_best_novel_class'],saved['candidate_best_old_class'])
                    np.testing.assert_array_equal(replay,saved['candidate_prediction'])
                    self.assertTrue(np.isfinite(saved['candidate_best_old_logscore']).all())
                    self.assertTrue(np.isfinite(saved['candidate_best_novel_logscore']).all())
                    self.assertEqual(saved['candidate_prediction'].shape[1],len(saved['target']))
                teacher=json.loads((target/'teacher_diagnostics.json').read_text())
                self.assertTrue(teacher['post_selection_only'])
                self.assertFalse(teacher['used_for_selection'])
                expected={0.} if row['variant'].startswith('bimc__') else {0.,.25,1.}
                self.assertEqual({c['blend'] for c in teacher['candidate_geometry']},expected)
                self.assertTrue(all(c['available'] for c in teacher['candidate_geometry']))
                for source in teacher['teachers']:
                    sequence=protocol['sequences'][source['sequence']]
                    position=sequence['class_ids'].index(source['class_id'])
                    used=sequence['support_indices'][position]+sequence['query_indices'][position]
                    self.assertFalse(set(source['sample_ids']) & set(arrays['sample_id'][used].tolist()))
            manifest=package_results(root,folder/'analysis.zip')
            self.assertTrue(manifest['complete'])
            self.assertEqual(manifest['audit']['completed_run_records'],14)
            self.assertFalse(any(r['path'].endswith('/base.npz') or r['path']=='grounding.npz' for r in manifest['files']))

    def test_outer_V_and_reserved_outer_A_queries_cannot_change_fitted_offsets(self):
        with directory() as folder:
            _,_,saved,metadata,grounded,_=write_transfer_fixture(folder)
            engine._runtime()
            arrays={k:torch.as_tensor(v) for k,v in saved.items()}
            grounding={k:torch.as_tensor(v) for k,v in grounded.items()}
            cfg=engine.base._configuration(metadata,2,2)
            outer=completion_protocol(cfg,arrays['labels'],saved['sample_id'].tolist())
            inner=engine.inner_protocol(arrays,cfg,outer,1)
            def calculate(data):
                fitted=engine.fit_method('semantic_mask',data,grounding,cfg,inner,2,16.)
                records=engine.base.build_records(data,cfg,inner)
                bank,_,_=engine._score_bank(fitted,data,cfg,inner,records,(0.,.25,1.))
                return engine.fit_offsets(bank,(0.,.25,1.),(-.1,0.,.1),.2)
            offsets,search=calculate(arrays)
            changed={k:v.clone() for k,v in arrays.items()}
            excluded=list(set(range(len(arrays['labels'])))-set(outer['fitting_indices']))
            changed['final'][excluded]*=-3
            validation=outer['pseudo_validation_class_ids']
            for key in ('attr_text','ordinary_text','text_features','description_proto','structured_text'):
                changed[key][validation]*=-2
            indices=torch.isin(changed['description_targets'],torch.as_tensor(validation))
            changed['description_features'][indices]*=-2
            other,other_search=calculate(changed)
            self.assertEqual(other,offsets)
            self.assertEqual(other_search,search)

    def test_matching_resume_skips_and_repairs_missing_teacher_and_rejects_changed_inputs(self):
        with directory() as folder:
            cache,grounding,*_=write_transfer_fixture(folder)
            root=folder/'study'
            first=run(cache,grounding,root)
            self.assertTrue(first['study_complete'])
            path=root/'seed_2/transfer/semantic_mask__none/predictions.npz'
            stamp=path.stat().st_mtime_ns
            second=run(cache,grounding,root)
            self.assertTrue(all(row['status']=='skipped' for row in second['runs']))
            self.assertEqual(path.stat().st_mtime_ns,stamp)
            missing=root/'seed_2/transfer/semantic_reliable__a_inner/teacher_diagnostics.json'
            missing.unlink()
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                package_results(root,folder/'refused.zip')
            repaired=run(cache,grounding,root)
            statuses={row['variant']:row['status'] for row in repaired['runs']}
            self.assertEqual(statuses['semantic_reliable__a_inner'],'completed')
            self.assertTrue(all(status=='skipped' for name,status in statuses.items() if name!='semantic_reliable__a_inner'))
            self.assertEqual(path.stat().st_mtime_ns,stamp)
            with self.assertRaisesRegex(ValueError,'changed'):
                run(cache,grounding,root,sequences=1)
            with self.assertRaisesRegex(ValueError,'overwritten|resume'):
                run(cache,grounding,root,resume=False)

    def test_interruption_keeps_full_plan_and_resumes_completed_peers(self):
        with directory() as folder:
            cache,grounding,*_=write_transfer_fixture(folder)
            root=folder/'interrupted'
            original=engine.fit_method
            def fail(method,*args,**kwargs):
                if method=='whole_transfer':
                    raise RuntimeError('injected attribute experiment interruption')
                return original(method,*args,**kwargs)
            with patch.object(engine,'fit_method',side_effect=fail):
                with self.assertRaisesRegex(RuntimeError,'injected'):
                    run(cache,grounding,root)
            summary=json.loads((root/'summary.json').read_text())
            self.assertEqual(summary['planned_runs'],14)
            self.assertGreater(summary['completed_runs'],0)
            self.assertEqual(summary['failed_runs'],1)
            self.assertGreater(summary['pending_runs'],0)
            self.assertFalse(summary['study_complete'])
            manifest=package_results(root,folder/'partial.zip',allow_incomplete=True)
            self.assertFalse(manifest['complete'])
            resumed=run(cache,grounding,root)
            self.assertTrue(resumed['study_complete'])
            self.assertEqual(resumed['completed_runs'],14)


if __name__=='__main__':
    unittest.main()
