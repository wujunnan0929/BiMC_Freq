"""Frequency-text grounding, three-group transfer and A-only experiment contract."""
import builtins
import contextlib
import copy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from engine import attribute_transfer_experiments as engine
from engine.completion_calibration import completion_protocol
from models.attribute_transfer import fit_attribute_transfer, complete_attribute_transfer
from models.clip import clip
from tools import cache_frequency_attribute_grounding as cache
from tools import run_attribute_transfer_experiments as cli
from tools.package_attribute_transfer_results import package_results
from utils.frequency_attributes import BANDS, PATTERNS, build_frequency_manifest
from utils.attribute_grounding import keyword_spans
from test_attribute_grounding import artifact_directory, toy_clip
from test_multilayer_completion_experiments import write_synthetic_cache
from test_attribute_transfer import fixture


def descriptions(names):
    return {name:{'low':[f'A {name} has a compact silhouette.',f'A {name} has long broad wings.'],
        'middle':[f'A {name} has a white collar and black patch.',f'A {name} has a contrasting pale band.'],
        'high':[f'A {name} has fine dark streaks.',f'A {name} has delicate feather edging.']}
        for name in names}


def frequency_fixture(folder):
    visual,arrays,source = write_synthetic_cache(folder)
    model = toy_clip(arrays['final'].shape[-1])
    payload = descriptions(source['base_class_names'])
    payload[source['base_class_names'][0]]['high'] = []
    grounded,selection,audit = cache.build(payload,source,model,clip.tokenize,clip._tokenizer,topq=3)
    path = folder/'frequency.npz'
    cache.write_grounding(path,grounded,cache.metadata_for(source,selection,audit,'synthetic-text-sha'))
    cache.load_grounding(path,source)
    return visual,path,arrays,source,grounded


class FrequencyAttributeTransferTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_static_filter_removes_flight_negation_names_and_duplicate_evidence(self):
        payload=descriptions(['Fine Streak Bird'])
        payload['Fine Streak Bird']['high'] = [
            'A Fine Streak Bird has a distinctive flight pattern.',
            'A Fine Streak Bird has no fine streaks.',
            'A Fine Streak Bird has a beak.',
            'A Fine Streak Bird has fine pale feather edges.',
            'A Fine Streak Bird has fine pale feather edges.']
        result=build_frequency_manifest(payload,['Fine Streak Bird'])
        cls=result['classes'][0]
        self.assertEqual(len(cls['groups']['high']),1)
        self.assertEqual(cls['groups']['high'][0]['text'],'A subject has fine pale feather edges.')
        self.assertEqual({r['reason'] for r in cls['rejected']}, {
            'nonvisual_or_temporal_content','negation_not_supported_by_lexical_grounding',
            'no_scale_content_keyword','duplicate_text_across_or_within_bands'})
        keys,_=keyword_spans(cls['groups']['high'][0]['text'],'Fine Streak Bird','high',clip._tokenizer,PATTERNS)
        self.assertEqual([k['keyword'] for k in keys],['fine','feather edges'])

    def test_text_vectors_pool_exact_used_sentences_and_unknowns_remain_zero(self):
        source={'base_class_names':['bird0','bird1']}
        payload=descriptions(source['base_class_names'])
        payload['bird0']['high']=[]
        payload['bird1']['high'].insert(0,'A bird1 has a fast flight pattern.')
        model=toy_clip(9)
        for method in ('attention_value','deletion'):
            arrays,selection,audit=cache.build(payload,source,model,clip.tokenize,clip._tokenizer,
                                             method=method,topq=4,batch_size=3)
            self.assertEqual(arrays['attribute_text'].shape,(2,3,9))
            self.assertFalse(arrays['group_available'][0,2])
            self.assertFalse(arrays['channel_mask'][0,2].any())
            np.testing.assert_array_equal(arrays['attribute_text'][0,2],np.zeros(9))
            for c,record in enumerate(audit['classes']):
                for g,band in enumerate(BANDS):
                    used=[s['text'] for s in record['groups'][band]['sentences'] if s['used']]
                    if not used:
                        continue
                    expected=F.normalize(F.normalize(model.encode_text(clip.tokenize(used)),dim=-1).sum(0),dim=-1)
                    np.testing.assert_allclose(arrays['attribute_text'][c,g],expected.numpy(),atol=1e-6)
                    self.assertEqual(int(arrays['channel_mask'][c,g].sum()),4)
                    self.assertTrue(all('flight pattern' not in text for text in used))
            self.assertFalse(any(p.grad is not None for p in model.parameters()))

    def test_three_groups_preserve_fit_and_future_support_isolation(self):
        x,y,text,available,masks=fixture()
        text,available,masks=text[:,:3],available[:,:3],masks[:,:3]
        ids=list(range(5))
        state=fit_attribute_transfer(x,y,text[ids],available[ids],ids,group_names=BANDS,topk=2)
        self.assertEqual(state['group_names'],list(BANDS))
        for mode in ('semantic','random','shuffled','whole'):
            first,diag=complete_attribute_transfer(state,x,y,text[5:],available[5:],masks[5:],[5,6,7],
                                                   mode=mode,return_diagnostics=True)
            changed=x.clone()
            changed[y!=5]*=-5
            second=complete_attribute_transfer(state,changed,y,text[5:6],available[5:6],masks[5:6],[5],mode=mode)
            self.assertTrue(torch.equal(first[:1],second))
            self.assertEqual([g['group'] for g in diag['classes'][0]['groups']],list(BANDS))
        unknown=complete_attribute_transfer(state,x,y,text[5:],torch.zeros_like(available[5:]),masks[5:],[5,6,7])
        self.assertTrue(torch.equal(unknown,torch.stack([x[y==c].mean(0) for c in (5,6,7)])))

    def test_sidecar_binding_and_tampering_are_rejected(self):
        with artifact_directory() as folder:
            _,path,_,source,_=frequency_fixture(folder)
            _,metadata=cache.load_grounding(path,source)
            self.assertFalse(metadata['visual_frequency_grounding_performed'])
            changed=copy.deepcopy(source)
            changed['cache_sha256']='wrong'
            with self.assertRaisesRegex(ValueError,'changed'):
                cache.load_grounding(path,changed)
            path.write_bytes(path.read_bytes()+b'corrupt')
            with self.assertRaisesRegex(ValueError,'checksum'):
                cache.load_grounding(path,source)

    def test_planner_is_stdlib_only_and_uses_fine_A_grid(self):
        original_import=builtins.__import__
        def guarded(name,*args,**kwargs):
            if name.split('.')[0] in ('torch','numpy'):
                raise AssertionError('Planner imported numerical runtime')
            return original_import(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=guarded), \
                patch('pathlib.Path.open',side_effect=AssertionError('Planner read cache')):
            args=cli.parse_args(['--cache','x.npz','--grounding','g.npz','--output-root','out',
                                 '--attribute-family','frequency'])
            plan=cli.build_plan(args)
        self.assertEqual(plan['planned_configurations'],48)
        self.assertIn(-.0005,plan['offset_grid'])
        self.assertEqual(plan['attribute_family'],'frequency')

    def test_cache_cli_loads_matching_weights_and_never_encodes_images(self):
        with artifact_directory() as folder:
            visual,arrays,source=write_synthetic_cache(folder)
            source_text=folder/'texts.json'
            source_text.write_text(json.dumps(descriptions(source['base_class_names'])),encoding='utf-8')
            model=toy_clip(arrays['final'].shape[-1])
            model.clip_weight_sha256=source['clip_weight_sha256']
            target=folder/'from_cli.npz'
            with patch('models.bimc.load_clip_to_cpu',return_value=model), \
                    patch.object(model,'encode_image',side_effect=AssertionError('Images must not be encoded')), \
                    contextlib.redirect_stdout(io.StringIO()):
                result=cache.main(['--cache',str(visual),'--descriptions',str(source_text),
                                   '--output',str(target),'--device','cpu','--topq','3'])
            self.assertEqual(result,0)
            _,metadata=cache.load_grounding(target,source)
            self.assertEqual(metadata['image_forward_count'],0)
            self.assertEqual(metadata['description_sha256'],cache.sha256(source_text))
            arguments=['--cache',str(visual),'--descriptions',str(source_text),
                       '--output',str(target),'--device','cpu','--topq','3']
            stamp={p:(p.stat().st_mtime_ns,p.read_bytes()) for p in cache.output_paths(target)}
            with patch('models.bimc.load_clip_to_cpu',side_effect=AssertionError('Resume must not load CLIP')), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(cache.main(arguments),0)
            self.assertIn('Reusing verified',output.getvalue())
            self.assertEqual(stamp,{p:(p.stat().st_mtime_ns,p.read_bytes()) for p in cache.output_paths(target)})

            # Caches created before the resume-only launcher change remain valid.
            saved,metadata=cache.load_grounding(target,source)
            metadata['source_files_sha256']['tools/cache_frequency_attribute_grounding.py']='previous-launcher-sha'
            cache.write_grounding(target,saved,metadata,overwrite=True)
            stamp={p:(p.stat().st_mtime_ns,p.read_bytes()) for p in cache.output_paths(target)}
            with patch('models.bimc.load_clip_to_cpu',side_effect=AssertionError('Resume must not load CLIP')), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cache.main(arguments),0)
            for extra,reason in ((['--topq','4'],'topq'),(['--method','deletion'],'method'),
                                 (['--sentences-per-band','1'],'frequency_manifest')):
                with patch('models.bimc.load_clip_to_cpu',side_effect=AssertionError('Mismatch must not load CLIP')), \
                        contextlib.redirect_stderr(io.StringIO()) as error:
                    self.assertEqual(cache.main(arguments+extra),2)
                self.assertIn(reason,error.getvalue())
            self.assertEqual(stamp,{p:(p.stat().st_mtime_ns,p.read_bytes()) for p in cache.output_paths(target)})

            # Even unchanged selected sentences require the same source-file SHA.
            text_bytes=source_text.read_bytes()
            source_text.write_bytes(text_bytes+b'\n')
            with contextlib.redirect_stderr(io.StringIO()) as error:
                self.assertEqual(cache.main(arguments),2)
            self.assertIn('description_sha256',error.getvalue())
            source_text.write_bytes(text_bytes)
            original_sha=cache.sha256
            def changed_source(path):
                return 'changed-extractor' if Path(path)==cache.ROOT/'utils/frequency_attributes.py' else original_sha(path)
            with patch.object(cache,'sha256',side_effect=changed_source), \
                    contextlib.redirect_stderr(io.StringIO()) as error:
                self.assertEqual(cache.main(arguments),2)
            self.assertIn('extraction source utils/frequency_attributes.py',error.getvalue())
            target.write_bytes(target.read_bytes()+b'corrupt')
            with contextlib.redirect_stderr(io.StringIO()) as error:
                self.assertEqual(cache.main(arguments),2)
            self.assertIn('checksum',error.getvalue())

    def test_partial_or_temporary_outputs_report_files_without_overwriting(self):
        with artifact_directory() as folder:
            target=folder/'partial.npz'
            self.assertFalse(cache.existing_output(target))
            target.write_bytes(b'partial')
            with self.assertRaisesRegex(ValueError,'Incomplete.*partial.json'):
                cache.existing_output(target)
            self.assertEqual(target.read_bytes(),b'partial')
            temporary=target.with_suffix('.npz.tmp')
            temporary.write_bytes(b'writing')
            with self.assertRaisesRegex(ValueError,'Temporary.*partial.npz.tmp'):
                cache.existing_output(target)
            self.assertEqual(temporary.read_bytes(),b'writing')

    def test_real_16_run_matrix_packages_inner_replay_and_resumes(self):
        with artifact_directory() as folder:
            visual,grounding,*_=frequency_fixture(folder)
            root=folder/'study'
            options=dict(attribute_family='frequency',fit_seeds=(2,),sequences=2,inner_sequences=1,
                         blend_grid=(0.,.05,1.),offset_grid=(-.002,0.,.002),threads=1,topk=2)
            with contextlib.redirect_stdout(io.StringIO()):
                result=engine.run_experiments(visual,grounding,root,**options)
            self.assertEqual((result['planned_runs'],result['completed_runs']),(16,16))
            self.assertTrue(result['study_complete'])
            self.assertEqual(result['group_order'],list(BANDS))
            for row in result['runs']:
                run=root/row['output_dir']
                with np.load(run/'predictions.npz',allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['candidate_prediction'][0],saved['reference_prediction'])
                if row['variant'].endswith('__a_inner'):
                    offsets=json.loads((run/'offset_fit.json').read_text())
                    inner=offsets['inner_protocol']
                    with np.load(run/'inner_scores.npz',allow_pickle=False) as saved:
                        allowed=set(np.asarray(inner['sample_ids'])[inner['allowed_outer_fitting_indices']].tolist())
                        self.assertTrue(set(saved['sample_id'].tolist())<=allowed)
                        for i,search in enumerate(offsets['candidate_results']):
                            for candidate in search['candidate_results']:
                                prediction=np.where(saved['best_novel_logscore'][i]+candidate['offset']
                                    >saved['best_old_logscore'][i],saved['best_novel_class'][i],saved['best_old_class'][i])
                                accuracy=np.mean([(prediction[a:b]==saved['target'][a:b]).mean()*100
                                                  for a,b in saved['boundaries']])
                                self.assertAlmostEqual(accuracy,candidate['accuracy'],places=10)
            manifest=package_results(root,folder/'analysis.zip')
            self.assertTrue(manifest['complete'])
            self.assertEqual(sum(f['path'].endswith('/inner_scores.npz') for f in manifest['files']),8)
            with zipfile.ZipFile(folder/'analysis.zip') as bundle:
                self.assertIsNone(bundle.testzip())
            with contextlib.redirect_stdout(io.StringIO()):
                resumed=engine.run_experiments(visual,grounding,root,**options)
            self.assertTrue(all(r['status']=='skipped' for r in resumed['runs']))
            missing=root/'seed_2/transfer/semantic_mask__a_inner/inner_scores.npz'
            missing.unlink()
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                package_results(root,folder/'refused.zip')
            with contextlib.redirect_stdout(io.StringIO()):
                repaired=engine.run_experiments(visual,grounding,root,**options)
            self.assertEqual(sum(r['status']=='completed' for r in repaired['runs']),1)

    def test_frequency_A_fit_cannot_use_V_text_or_reserved_A_query_images(self):
        with artifact_directory() as folder:
            _,_,saved,source,grounded=frequency_fixture(folder)
            engine._runtime()
            arrays={k:torch.as_tensor(v) for k,v in saved.items()}
            grounding={k:torch.as_tensor(v) for k,v in grounded.items()}
            cfg=engine.base._configuration(source,2,2)
            outer=completion_protocol(cfg,arrays['labels'],saved['sample_id'].tolist())
            inner=engine.inner_protocol(arrays,cfg,outer,1)
            def fitted_scores(data,ground):
                fitted=engine.fit_method('semantic_mask',data,ground,cfg,inner,2,16.)
                bank,_,_=engine._score_bank(fitted,data,cfg,inner,engine.base.build_records(data,cfg,inner),(0.,.05,1.))
                return engine.fit_offsets(bank,(0.,.05,1.),(-.002,0.,.002),.2)
            first=fitted_scores(arrays,grounding)
            changed={k:v.clone() for k,v in arrays.items()}
            excluded=list(set(range(len(arrays['labels'])))-set(outer['fitting_indices']))
            changed['final'][excluded]*=-3
            other={k:v.clone() for k,v in grounding.items()}
            other['attribute_text'][outer['pseudo_validation_class_ids']]*=-2
            other['channel_mask'][outer['pseudo_validation_class_ids']]=False
            self.assertEqual(first,fitted_scores(changed,other))


if __name__=='__main__':
    unittest.main()
