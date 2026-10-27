"""Analysis packages must be complete, bounded and independent of ML runtimes."""

import builtins
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4
import zipfile

from tools import package_multilayer_completion_results as bundle

ROOT = Path(__file__).resolve().parents[1]


@contextlib.contextmanager
def workspace_directory():
    parent = (ROOT/'.codex_tmp').resolve()
    folder = parent/('multilayer_package_'+uuid4().hex)
    folder.mkdir(parents=True)
    try:
        yield folder
    finally:
        resolved = folder.resolve()
        if resolved.parent != parent or not resolved.name.startswith('multilayer_package_'):
            raise RuntimeError('Refusing cleanup outside the test workspace.')
        shutil.rmtree(resolved)


def fixture(root):
    directory = root/'seed_2/repeat/original16'
    directory.mkdir(parents=True)
    for name in bundle.RUN_REQUIRED:
        if name.endswith('.json'):
            content = {'status': 'completed', 'fingerprint': 'fixture'}
            (directory/name).write_text(json.dumps(content), encoding='utf-8')
        else:
            (directory/name).write_bytes(b'compact-npz-fixture')
    summary = {'schema_version': 1, 'method': 'multilayer_multiattribute_completion',
               'data_source': 'base_training_only', 'benchmark_evaluation_performed': False,
               'planned_runs': 1, 'completed_runs': 1, 'failed_runs': 0,
               'runs': [{'stage': 'repeat', 'variant': 'original16', 'fit_seed': 2,
                         'output_dir': directory.relative_to(root).as_posix(),
                         'status': 'completed', 'fingerprint': 'fixture'}]}
    expected = [{key: summary['runs'][0][key] for key in ('stage', 'variant', 'fit_seed', 'output_dir')}]
    summary.update(expected_runs=expected, pending_runs=0, study_complete=True)
    (root/'summary.json').write_text(json.dumps(summary), encoding='utf-8')
    (root/'plan.json').write_text(json.dumps({'runs': expected}), encoding='utf-8')
    (root/'input_provenance.json').write_text(json.dumps({'cache_sha256': 'fixture'}), encoding='utf-8')
    (root/'analysis.md').write_text('Synthetic package fixture, not benchmark results.', encoding='utf-8')
    return summary, directory


class MultilayerCompletionPackageTest(unittest.TestCase):
    def test_complete_bundle_has_checksums_and_excludes_large_or_sensitive_inputs(self):
        with workspace_directory() as folder:
            root = folder/'study'
            root.mkdir()
            fixture(root)
            for name in ('features.npz', 'cache.npz', 'model.pt', 'checkpoint.pt', 'api_key.json', 'config.local.json'):
                (root/name).write_bytes(b'not-for-analysis')
            snapshot = root/'source_snapshot/models'
            snapshot.mkdir(parents=True)
            (snapshot/'core.py').write_text('pass\n', encoding='utf-8')
            archive = folder/'analysis.zip'
            manifest = bundle.package_results(root, archive)
            self.assertTrue(manifest['complete'])
            with zipfile.ZipFile(archive) as saved:
                names = set(saved.namelist())
                self.assertIn('bundle_manifest.json', names)
                self.assertIn('source_snapshot/models/core.py', names)
                for item in manifest['files']:
                    data = saved.read(item['path'])
                    self.assertEqual(hashlib.sha256(data).hexdigest(), item['sha256'])
                    self.assertEqual(len(data), item['bytes'])
                self.assertFalse(any(name in names for name in ('features.npz', 'cache.npz', 'model.pt', 'checkpoint.pt', 'api_key.json', 'config.local.json')))
            with self.assertRaises(FileExistsError):
                bundle.package_results(root, archive)

    def test_missing_artifacts_require_explicit_partial_bundle(self):
        with workspace_directory() as folder:
            root = folder/'study'
            root.mkdir()
            _, directory = fixture(root)
            (directory/'diagnostics.json').unlink()
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                bundle.package_results(root, folder/'refused.zip')
            self.assertFalse((folder/'refused.zip').exists())
            manifest = bundle.package_results(root, folder/'partial.zip', allow_incomplete=True)
            self.assertFalse(manifest['complete'])
            self.assertIn('seed_2/repeat/original16/diagnostics.json', manifest['audit']['missing_artifacts'])

    def test_escaping_run_reference_cannot_add_outside_file(self):
        with workspace_directory() as folder:
            root = folder/'study'
            root.mkdir()
            summary, _ = fixture(root)
            summary['runs'][0]['output_dir'] = '../outside'
            (root/'summary.json').write_text(json.dumps(summary), encoding='utf-8')
            outside = folder/'outside'
            outside.mkdir()
            (outside/'metrics.json').write_text('{"private":true}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'within'):
                bundle.package_results(root, folder/'refused.zip')
            manifest = bundle.package_results(root, folder/'partial.zip', allow_incomplete=True)
            self.assertFalse(manifest['complete'])
            self.assertFalse(any('outside' in item['path'] for item in manifest['files']))

    def test_fingerprint_mismatch_is_reported(self):
        with workspace_directory() as folder:
            root = folder/'study'
            root.mkdir()
            _, directory = fixture(root)
            (directory/'run_spec.json').write_text('{"fingerprint":"different"}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                bundle.build_manifest(root)

    def test_corrupt_required_json_is_not_certified_complete(self):
        with workspace_directory() as folder:
            root = folder/'study'
            root.mkdir()
            _, directory = fixture(root)
            (directory/'calibration.json').write_text('{"incomplete":', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Unreadable'):
                bundle.build_manifest(root)
            manifest = bundle.build_manifest(root, allow_incomplete=True)
            self.assertFalse(manifest['complete'])

    def test_protocol_checksum_is_checked_when_reported(self):
        with workspace_directory() as folder:
            root = folder/'study'
            root.mkdir()
            summary, directory = fixture(root)
            summary['runs'][0]['protocol_sha256'] = 'wrong'
            (root/'summary.json').write_text(json.dumps(summary), encoding='utf-8')
            (directory/'protocol.json').write_text('{"seed":2,"sha256":"wrong"}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'protocol checksum'):
                bundle.build_manifest(root)

    def test_unstarted_configurations_cannot_be_certified_complete(self):
        with workspace_directory() as folder:
            root = folder/'study'
            root.mkdir()
            summary, _ = fixture(root)
            pending = {'stage': 'layers', 'variant': 'layer3', 'fit_seed': 2,
                       'output_dir': 'seed_2/layers/layer3'}
            summary['expected_runs'].append(pending)
            summary.update(planned_runs=2, pending_runs=1, study_complete=False)
            (root/'summary.json').write_text(json.dumps(summary), encoding='utf-8')
            (root/'plan.json').write_text(json.dumps({'runs': summary['expected_runs']}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Unstarted planned'):
                bundle.build_manifest(root)
            manifest = bundle.build_manifest(root, allow_incomplete=True)
            self.assertFalse(manifest['complete'])
            self.assertEqual(manifest['audit']['pending_run_records'], 1)
            self.assertEqual(manifest['audit']['unstarted_runs'], [pending])

    def test_plan_disagreement_is_reported_even_when_counts_match(self):
        with workspace_directory() as folder:
            root = folder/'study'
            root.mkdir()
            fixture(root)
            (root/'plan.json').write_text('{"runs":[]}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'plan.json differs'):
                bundle.build_manifest(root)

    def test_symlink_files_are_never_packaged(self):
        with workspace_directory() as folder:
            root = folder/'study'
            root.mkdir()
            fixture(root)
            outside = folder/'private.json'
            outside.write_text('{"sensitive":true}', encoding='utf-8')
            try:
                (root/'diagnostics.json').symlink_to(outside)
            except OSError as error:
                self.skipTest('Symlink creation unavailable: ' + str(error))
            manifest = bundle.package_results(root, folder/'analysis.zip')
            self.assertFalse(any(item['path'] == 'diagnostics.json' for item in manifest['files']))
            self.assertIn({'path': 'diagnostics.json', 'reason': 'symlink_file'}, manifest['excluded'])

    def test_symlink_inventory_branch_does_not_read_target(self):
        # Exercise the exclusion even on Windows machines without symlink privilege.
        with workspace_directory() as folder:
            root = folder/'study'
            root.mkdir()
            fixture(root)
            optional = root/'diagnostics.json'
            optional.write_text('excluded-target', encoding='utf-8')
            original = Path.is_symlink
            def is_link(path):
                return path == optional or original(path)
            with patch.object(Path, 'is_symlink', new=is_link):
                manifest = bundle.build_manifest(root)
            self.assertTrue(manifest['complete'])
            self.assertFalse(any(item['path'] == 'diagnostics.json' for item in manifest['files']))
            self.assertIn({'path': 'diagnostics.json', 'reason': 'symlink_file'}, manifest['excluded'])

    def test_packaging_imports_no_numerical_library(self):
        original_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in ('numpy', 'torch', 'torchvision'):
                raise AssertionError('Packaging imported ' + name)
            return original_import(name, *args, **kwargs)
        with workspace_directory() as folder:
            root = folder/'study'
            root.mkdir()
            fixture(root)
            with patch('builtins.__import__', side_effect=guarded), contextlib.redirect_stdout(io.StringIO()):
                result = bundle.main(['--root', str(root), '--output', str(folder/'analysis.zip')])
            self.assertEqual(result, 0)


if __name__ == '__main__':
    unittest.main()
