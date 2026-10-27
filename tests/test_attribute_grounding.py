"""Real frozen-text extraction and authenticated channel-mask provenance."""

import builtins
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

from models.clip import clip
from models.clip.model import CLIP
from tools import cache_attribute_grounding as cache
from utils.attribute_descriptions import GROUPS, build_attribute_manifest
from utils.attribute_grounding import build_grounding, keyword_spans, text_token_contributions
from test_attribute_descriptions import sentences


@contextlib.contextmanager
def artifact_directory():
    parent=Path(__file__).resolve().parents[1]/'.codex_tmp'/'attribute_grounding_tests'
    parent.mkdir(parents=True,exist_ok=True)
    directory=parent/uuid4().hex
    directory.mkdir()
    try:
        yield directory
    finally:
        if directory.resolve().parent != parent.resolve():
            raise RuntimeError('Unsafe grounding test cleanup target')
        shutil.rmtree(directory)


def toy_clip(dimension=8):
    with torch.random.fork_rng():
        torch.manual_seed(41)
        model=CLIP(embed_dim=dimension,image_resolution=8,vision_layers=1,vision_width=64,
                   vision_patch_size=4,context_length=77,vocab_size=49408,
                   transformer_width=8,transformer_heads=2,transformer_layers=2)
    return model.float().eval().requires_grad_(False)


def metadata_for(source,audit):
    return {'schema_version':1,'format':cache.FORMAT,'data_source':'base_training_only',
            'source_cache_sha256':source['cache_sha256'],
            'source_metadata_sha256':source['metadata_payload_sha256'],
            'clip_weight_sha256':source['clip_weight_sha256'],
            'base_class_ids':source['base_class_ids'],'base_class_names':source['base_class_names'],
            'group_order':list(GROUPS),'method':audit['method'],'topq':audit['topq'],
            'no_image_encoding':True,'new_gradient_training_performed':False,'grounding_manifest':audit,
            'synthetic_fixture':True}


def write_synthetic_grounding(directory,source_metadata):
    """Public integration fixture: real toy-CLIP grounding of source sentences.

    The source bank itself is a synthetic test fixture. This output must never
    be described as a learned/real-data CLIP performance result.
    """
    dimension=source_metadata.get('feature_dim') or source_metadata['array_shapes']['final'][-1]
    arrays,audit=build_grounding(source_metadata['attribute_manifest'],toy_clip(dimension),
                                clip.tokenize,clip._tokenizer,topq=min(3,dimension),batch_size=128)
    path=Path(directory)/'grounding.npz'
    cache.write_grounding(path,arrays,metadata_for(source_metadata,audit))
    certified,metadata=cache.load_grounding(path,source_metadata)
    return path,certified,metadata


class AttributeGroundingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads()
        torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def manifest(self):
        return build_attribute_manifest({'base0':sentences('base0'),'base1':sentences('base1')},['base0','base1'])

    def test_real_attention_path_has_final_output_coordinates_and_preserves_default_encoder(self):
        model=toy_clip(11)
        tokens=clip.tokenize(['A base0 has a slender body.','A base1 has blue wings.'])
        expected=model.encode_text(tokens)
        encoded,contribution=text_token_contributions(model,tokens)
        self.assertTrue(torch.equal(encoded,expected))
        self.assertEqual(contribution.shape,(2,77,11))
        self.assertTrue(torch.isfinite(contribution).all())
        self.assertTrue(torch.equal(model.encode_text(tokens),expected))
        self.assertFalse(any(p.grad is not None for p in model.parameters()))
        block=model.transformer.resblocks[-1]
        self.assertEqual(len(block._forward_hooks),0)
        self.assertEqual(len(block.ln_1._forward_hooks),0)

    def test_species_name_is_excluded_and_actual_keyword_bpe_positions_are_recorded(self):
        text='A Red spotted Bird has a red head and striped wings.'
        keywords,rejected=keyword_spans(text,'Red spotted Bird','color',clip._tokenizer)
        self.assertEqual([k['keyword'] for k in keywords],['red'])
        self.assertEqual(rejected[0]['reason'],'class_name')
        ids=clip.tokenize(text)[0]
        for keyword in keywords:
            self.assertEqual(ids[slice(*keyword['token_span'])].tolist(),keyword['token_ids'])
            self.assertEqual(text[slice(*keyword['char_span'])],keyword['keyword'])
        pattern,rejected=keyword_spans(text,'Red spotted Bird','pattern',clip._tokenizer)
        self.assertEqual([k['keyword'] for k in pattern],['striped'])
        self.assertTrue(any(r['reason']=='class_name' for r in rejected))

    def test_attention_masks_are_deterministic_fixed_topq_and_batch_size_independent(self):
        model=toy_clip(12)
        first,audit=build_grounding(self.manifest(),model,clip.tokenize,clip._tokenizer,topq=4,batch_size=1)
        second,_=build_grounding(self.manifest(),model,clip.tokenize,clip._tokenizer,topq=4,batch_size=16)
        self.assertTrue(np.all(first['group_available']))
        np.testing.assert_array_equal(first['channel_mask'].sum(-1),np.full((2,4),4))
        np.testing.assert_array_equal(first['channel_mask'],second['channel_mask'])
        np.testing.assert_allclose(first['channel_score'],second['channel_score'],atol=2e-7)
        self.assertIn('not exact FDR',audit['grounding_note'])
        self.assertEqual(audit['group_available_counts'],dict.fromkeys(GROUPS,2))

    def test_unknown_and_groups_without_actual_content_keywords_are_not_filled(self):
        manifest=self.manifest()
        manifest['classes'][0]['groups']['shape']=[]
        manifest['classes'][0]['groups']['color']=[{'text':'A base0 has a hooked bill.'}]
        arrays,audit=build_grounding(manifest,toy_clip(),clip.tokenize,clip._tokenizer,topq=3)
        self.assertFalse(arrays['group_available'][0,0])
        self.assertFalse(arrays['group_available'][0,2])
        self.assertTrue(np.all(arrays['channel_score'][0,[0,2]]==0))
        self.assertFalse(arrays['channel_mask'][0,[0,2]].any())
        self.assertEqual(audit['classes'][0]['groups']['shape']['reason'],'unknown_attribute')
        self.assertEqual(audit['classes'][0]['groups']['color']['reason'],'no_groundable_keywords')

    def test_deletion_is_final_feature_change_and_retains_deleted_text_provenance(self):
        model=toy_clip(8)
        arrays,audit=build_grounding(self.manifest(),model,clip.tokenize,clip._tokenizer,topq=3,method='deletion')
        self.assertTrue(np.all(arrays['group_available']))
        detail=audit['classes'][0]['groups']['color']['sentences'][0]
        self.assertIn('base0',detail['deleted_text'])
        self.assertNotIn('blue',detail['deleted_text'])
        original=torch.nn.functional.normalize(model.encode_text(clip.tokenize(detail['text'])).float(),dim=-1)
        deleted=torch.nn.functional.normalize(model.encode_text(clip.tokenize(detail['deleted_text'])).float(),dim=-1)
        self.assertGreater(float((original-deleted).abs().sum()),0)

    def test_sidecar_roundtrip_binds_source_identity_and_rejects_tampering(self):
        from test_multilayer_completion_experiments import write_synthetic_cache
        with artifact_directory() as directory:
            _,_,source=write_synthetic_cache(directory)
            path,arrays,metadata=write_synthetic_grounding(directory,source)
            self.assertEqual(metadata['grounding_sha256'],metadata['cache_sha256'])
            self.assertEqual(metadata['source_cache_sha256'],source['cache_sha256'])
            self.assertNotEqual(metadata['source_cache_sha256'],metadata['grounding_sha256'])
            self.assertEqual(arrays['channel_score'].shape,(6,4,8))
            altered={**source,'cache_sha256':'different'}
            with self.assertRaisesRegex(ValueError,'source cache'):
                cache.load_grounding(path,altered)
            with self.assertRaises(FileExistsError):
                cache.write_grounding(path,arrays,metadata)
            with path.open('ab') as handle:
                handle.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'checksum'):
                cache.load_grounding(path,source)

    def test_incorrect_masks_or_unproven_keywords_cannot_be_certified(self):
        from test_multilayer_completion_experiments import write_synthetic_cache
        with artifact_directory() as directory:
            _,_,source=write_synthetic_cache(directory)
            _,arrays,metadata=write_synthetic_grounding(directory,source)
            bad={k:v.copy() for k,v in arrays.items()}
            bad['channel_mask'][0,1]=False
            with self.assertRaisesRegex(ValueError,'topq'):
                cache.write_grounding(directory/'bad.npz',bad,metadata)
            metadata=json.loads(json.dumps(metadata))
            metadata['grounding_manifest']['classes'][0]['groups']['part']['sentences']=[]
            with self.assertRaisesRegex(ValueError,'no used content keyword'):
                cache.write_grounding(directory/'missing.npz',arrays,metadata)

    def test_dry_run_is_standard_library_only_and_does_not_create_outputs(self):
        original=builtins.__import__
        def guarded(name,*args,**kwargs):
            if name.split('.')[0] in ('torch','numpy','models','yacs'):
                raise AssertionError('Dry run imported ML runtime: '+name)
            return original(name,*args,**kwargs)
        with artifact_directory() as directory:
            with patch('builtins.__import__',side_effect=guarded),contextlib.redirect_stdout(io.StringIO()) as out:
                result=cache.main(['--cache',str(directory/'not_loaded.npz'),
                                   '--output',str(directory/'not_created.npz'),'--dry-run'])
            self.assertEqual(result,0)
            self.assertIn('no_image_encoding',out.getvalue())
            self.assertFalse((directory/'not_created.npz').exists())

    def test_actual_cli_reuses_source_and_never_constructs_dataset_or_encodes_images(self):
        from test_multilayer_completion_experiments import write_synthetic_cache
        with artifact_directory() as directory:
            path,_,source=write_synthetic_cache(directory)
            model=toy_clip(8)
            model.clip_weight_sha256=source['clip_weight_sha256']
            with patch('models.bimc.load_clip_to_cpu',return_value=model), \
                    patch('datasets.data_manager.DatasetManager',side_effect=AssertionError('Dataset access')), \
                    patch.object(model,'encode_image',side_effect=AssertionError('Image encoding')), \
                    contextlib.redirect_stdout(io.StringIO()):
                result=cache.main(['--cache',str(path),'--output',str(directory/'cli.npz'),
                                   '--device','cpu','--topq','3'])
                self.assertEqual(result,0)
                model.clip_weight_sha256='incorrect-weights'
                with contextlib.redirect_stderr(io.StringIO()):
                    result=cache.main(['--cache',str(path),'--output',str(directory/'mismatch.npz'),
                                       '--device','cpu','--topq','3'])
                self.assertEqual(result,2)
            _,metadata=cache.load_grounding(directory/'cli.npz',source)
            self.assertEqual(metadata['image_forward_count'],0)
            self.assertFalse((directory/'mismatch.npz').exists())

    def test_invalid_arguments_and_noncausal_attention_are_rejected(self):
        for kwargs in ({'topq':0},{'batch_size':0},{'method':'invented'}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                build_grounding(self.manifest(),toy_clip(),clip.tokenize,clip._tokenizer,**kwargs)
        model=toy_clip()
        model.transformer.resblocks[-1].attn_mask=torch.zeros(77,77)
        with self.assertRaisesRegex(ValueError,'causal'):
            text_token_contributions(model,clip.tokenize('A base0 has a red head.'))
        self.assertEqual(len(model.transformer.resblocks[-1]._forward_hooks),0)


if __name__=='__main__':
    unittest.main()
