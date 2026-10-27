"""Result bundle authentication does not load numerical or executable payloads."""

import builtins
import contextlib
import hashlib
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4
import zipfile

from tools import package_attribute_transfer_results as bundle

ROOT = Path(__file__).resolve().parents[1]


@contextlib.contextmanager
def directory():
    parent = (ROOT/'.codex_tmp').resolve()
    folder = parent/('attribute_package_'+uuid4().hex)
    folder.mkdir(parents=True)
    try:
        yield folder
    finally:
        if folder.resolve().parent != parent:
            raise RuntimeError('Unsafe fixture cleanup target')
        shutil.rmtree(folder)


def save(path,value):
    path.write_text(json.dumps(value),encoding='utf-8')


def fixture(root):
    root.mkdir()
    folder = root/'seed_2/transfer/semantic_mask__none'
    folder.mkdir(parents=True)
    protocol = {'source':'base_training_only','fitting_indices':[0,1]}
    encoded = json.dumps(protocol,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()
    protocol['sha256'] = hashlib.sha256(encoded).hexdigest()
    for name in bundle.RUN_REQUIRED:
        if name.endswith('.json'):
            save(folder/name,{'fingerprint':'run','family_fingerprint':'study','status':'completed'})
        else:
            (folder/name).write_bytes(b'fixture-no-array-loading')
    save(folder/'protocol.json',protocol)
    row = {'fit_seed':2,'stage':'transfer','variant':'semantic_mask__none',
           'output_dir':folder.relative_to(root).as_posix(),'fingerprint':'run',
           'protocol_sha256':protocol['sha256'],'status':'completed'}
    expected = [{k:row[k] for k in ('fit_seed','stage','variant','output_dir')}]
    summary = {'method':'attribute_transfer_development_v1','family_fingerprint':'study',
               'data_source':'base_training_only','benchmark_evaluation_performed':False,
               'fit_seeds':[2], 'planned_runs':1,'completed_runs':1,'failed_runs':0,'pending_runs':0,
               'study_complete':True,'expected_runs':expected,'runs':[row]}
    save(root/'summary.json',summary)
    save(root/'plan.json',{'runs':expected,'family_fingerprint':'study'})
    save(root/'input_provenance.json',{'family_fingerprint':'study','cache_sha256':'visual',
                                      'grounding_sha256':'grounding'})
    cache={'cache_sha256':'visual','metadata_payload_sha256':'metadata','base_class_ids':[0,1],
           'clip_weight_sha256':'weights'}
    save(root/'cache_metadata.json',cache)
    grounding={'grounding_sha256':'grounding','source_cache_sha256':'visual',
               'source_metadata_sha256':'metadata','base_class_ids':[0,1],
               'clip_weight_sha256':'weights','grounding_manifest':{'classes':[]}}
    save(root/'grounding_metadata.json',grounding)
    save(root/'grounding_manifest.json',grounding['grounding_manifest'])
    source=root/'source_snapshot/models/core.py'
    source.parent.mkdir(parents=True)
    source.write_text('raise RuntimeError("Source snapshots must not execute")\n',encoding='utf-8')
    save(root/'source_hashes.json',{'models/core.py':hashlib.sha256(source.read_bytes()).hexdigest()})
    return summary,folder


class AttributeTransferPackageTest(unittest.TestCase):
    def test_complete_package_verifies_files_and_includes_all_candidate_diagnostics(self):
        with directory() as parent:
            root=parent/'study'
            fixture(root)
            for name in ('base.npz','grounding.npz','checkpoint.pt','api_key.json','config.local.json'):
                (root/name).write_bytes(b'excluded')
            archive=parent/'analysis.zip'
            original_import=builtins.__import__
            def guard(name,*args,**kwargs):
                if name.split('.')[0] in ('numpy','torch'):
                    raise AssertionError('Packager imported numerical library')
                return original_import(name,*args,**kwargs)
            with patch('builtins.__import__',side_effect=guard):
                manifest=bundle.package_results(root,archive)
            self.assertTrue(manifest['complete'])
            self.assertEqual(manifest['audit']['errors'],[])
            with zipfile.ZipFile(archive) as z:
                for item in manifest['files']:
                    value=z.read(item['path'])
                    self.assertEqual(len(value),item['bytes'])
                    self.assertEqual(hashlib.sha256(value).hexdigest(),item['sha256'])
                names=z.namelist()
                self.assertIn('seed_2/transfer/semantic_mask__none/teacher_diagnostics.json',names)
                self.assertIn('seed_2/transfer/semantic_mask__none/offset_fit.json',names)
                self.assertIn('grounding_manifest.json',names)
                self.assertNotIn('grounding.npz',names)
            with self.assertRaises(FileExistsError):
                bundle.package_results(root,archive)

    def test_missing_all_blend_teacher_diagnostic_refuses_complete_package(self):
        with directory() as parent:
            root=parent/'study'
            _,folder=fixture(root)
            (folder/'teacher_diagnostics.json').unlink()
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                bundle.package_results(root,parent/'refused.zip')
            self.assertFalse((parent/'refused.zip').exists())
            manifest=bundle.package_results(root,parent/'partial.zip',allow_incomplete=True)
            self.assertFalse(manifest['complete'])
            self.assertIn(folder.relative_to(root).as_posix()+'/teacher_diagnostics.json',
                          manifest['audit']['missing_artifacts'])

    def test_cache_and_grounding_identity_mismatch_is_rejected(self):
        for key in ('source_cache_sha256','grounding_sha256','source_metadata_sha256','clip_weight_sha256','base_class_ids'):
            with self.subTest(key=key),directory() as parent:
                root=parent/'study'
                fixture(root)
                path=root/'grounding_metadata.json'
                data=json.loads(path.read_text())
                data[key]='mismatched'
                save(path,data)
                with self.assertRaisesRegex(ValueError,'Incomplete'):
                    bundle.build_manifest(root)

    def test_unstarted_planned_run_and_failed_status_are_not_certified_complete(self):
        with directory() as parent:
            root=parent/'study'
            summary,_=fixture(root)
            expected={'fit_seed':3,'stage':'transfer','variant':'semantic_mask__none',
                      'output_dir':'seed_3/transfer/semantic_mask__none'}
            summary['expected_runs'].append(expected)
            summary.update(planned_runs=2,pending_runs=1,study_complete=False)
            save(root/'summary.json',summary)
            save(root/'plan.json',{'runs':summary['expected_runs'],'family_fingerprint':'study'})
            with self.assertRaisesRegex(ValueError,'Unstarted'):
                bundle.build_manifest(root)
            manifest=bundle.build_manifest(root,allow_incomplete=True)
            self.assertEqual(manifest['audit']['pending_run_records'],1)
            self.assertFalse(manifest['complete'])

    def test_absent_identity_on_both_inputs_does_not_count_as_a_matching_identity(self):
        with directory() as parent:
            root=parent/'study'
            fixture(root)
            for name in ('cache_metadata.json','grounding_metadata.json'):
                path=root/name
                data=json.loads(path.read_text())
                data.pop('base_class_ids',None)
                data.pop('metadata_payload_sha256',None)
                data.pop('source_metadata_sha256',None)
                save(path,data)
            save(root/'source_hashes.json',{})
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                bundle.build_manifest(root)

    def test_escaping_run_and_snapshot_references_are_never_read(self):
        with directory() as parent:
            root=parent/'study'
            summary,_=fixture(root)
            summary['runs'][0]['output_dir']='../outside'
            save(root/'summary.json',summary)
            save(root/'source_hashes.json',{'../../outside/secret.py':'hash'})
            outside=parent/'outside'
            outside.mkdir()
            (outside/'secret.py').write_text('private',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                bundle.build_manifest(root)
            manifest=bundle.package_results(root,parent/'partial.zip',allow_incomplete=True)
            self.assertFalse(any('outside' in item['path'] for item in manifest['files']))

    def test_protocol_and_source_snapshot_hashes_are_checked(self):
        for name in ('source_snapshot/models/core.py','seed_2/transfer/semantic_mask__none/protocol.json'):
            with self.subTest(name=name),directory() as parent:
                root=parent/'study'
                fixture(root)
                if name.endswith('protocol.json'):
                    save(root/name,{'sha256':'changed','source':'base_training_only'})
                else:
                    (root/name).write_text('changed',encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'checksum'):
                    bundle.build_manifest(root)

    def test_missing_fingerprint_and_family_mismatch_are_rejected(self):
        with directory() as parent:
            root=parent/'study'
            _,folder=fixture(root)
            save(folder/'run_spec.json',{'family_fingerprint':'another-study'})
            with self.assertRaisesRegex(ValueError,'fingerprint'):
                bundle.build_manifest(root)

    def test_symlinks_are_excluded_without_loading_their_target(self):
        with directory() as parent:
            root=parent/'study'
            fixture(root)
            link=root/'diagnostics_extra.json'
            link.write_text('{}',encoding='utf-8')
            original=Path.is_symlink
            with patch.object(Path,'is_symlink',lambda path:True if path==link else original(path)):
                manifest=bundle.build_manifest(root)
            self.assertTrue(manifest['complete'])
            self.assertFalse(any(item['path']=='diagnostics_extra.json' for item in manifest['files']))
            self.assertIn({'path':'diagnostics_extra.json','reason':'symlink_file'},manifest['excluded'])


if __name__=='__main__':
    unittest.main()
