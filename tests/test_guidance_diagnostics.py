import contextlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import unittest
from uuid import uuid4

import numpy as np

from tools import diagnose_frequency_guidance as tool


@contextlib.contextmanager
def artifact_directory():
    # Normal workspace ACL inheritance also supports restricted Windows runners.
    parent = Path(__file__).resolve().parents[1] / '.codex_tmp' / 'guidance_diagnostics_tests'
    parent.mkdir(parents=True, exist_ok=True)
    directory = parent / uuid4().hex
    directory.mkdir()
    try:
        yield directory
    finally:
        resolved = directory.resolve()
        if resolved.parent != parent.resolve() or resolved.name != directory.name:
            raise RuntimeError('Refusing cleanup outside test artifact root')
        shutil.rmtree(resolved)


class GuidanceDiagnosticsTest(unittest.TestCase):
    def predictions(self, root, name, ids, targets, predictions):
        path = root / name
        np.savez_compressed(path, sample_id=ids, target=targets, prediction=predictions)
        return path

    def test_reordered_ids_count_corrections_damage_and_groups(self):
        with artifact_directory() as directory:
            root = Path(directory)
            a = self.predictions(root, 'a.npz', [10, 11, 12, 13], [0, 1, 2, 3], [0, 0, 2, 1])
            b = self.predictions(root, 'b.npz', [13, 12, 11, 10], [3, 2, 1, 0], [3, 0, 1, 0])
            result = tool.compare_predictions(a, b, 2)
            self.assertTrue(result['candidate_reordered'])
            self.assertEqual(result['all']['changed'], 3)
            self.assertEqual(result['all']['corrected'], 2)
            self.assertEqual(result['all']['damaged'], 1)
            self.assertEqual(result['all']['delta_accuracy_pp'], 25)
            self.assertEqual(result['base']['delta_accuracy_pp'], 50)
            self.assertEqual(result['novel']['delta_accuracy_pp'], 0)

    def test_duplicate_ids_different_ids_and_targets_are_rejected(self):
        with artifact_directory() as directory:
            root = Path(directory)
            a = self.predictions(root, 'a.npz', [10, 11], [0, 1], [0, 1])
            for ids, targets, expected in [([10, 10], [0, 1], 'duplicate'),
                                           ([10, 12], [0, 1], 'different sample'),
                                           ([11, 10], [0, 1], 'Targets differ')]:
                b = self.predictions(root, 'b.npz', ids, targets, [0, 1])
                with self.assertRaisesRegex(ValueError, expected):
                    tool.compare_predictions(a, b, 1)

    def test_calibration_schema_and_stage_deltas(self):
        rows = []
        for rho, gain in [(0, 0), (1, .4)]:
            rows.append({'rho': rho, 'mean_base_accuracy': 80 - gain,
                         'mean_novel_accuracy': 60 + gain, 'mean_harmonic_accuracy': 70 + gain,
                         'stages': [{'sequence': 0, 'stage': 1, 'harmonic_accuracy': 70 + gain}]})
        result = tool.calibration_summary({'selected_rho': 1, 'candidate_results': rows,
                                           'selection_status': 'base_class_disjoint_validation'})
        self.assertEqual(result['selected_rho'], 1)
        self.assertAlmostEqual(result['candidates'][1]['delta_vs_rho0_pp']['mean_harmonic_accuracy'], .4)
        self.assertAlmostEqual(result['candidates'][1]['stage_harmonic_delta_vs_rho0_pp']['mean'], .4)
        rows[1]['stages'][0]['sequence'] = 1
        with self.assertRaisesRegex(ValueError, 'different sequence/stage'):
            tool.calibration_summary({'candidate_results': rows})

    def make_run(self, root, variant, seed=1):
        directory = root / 'cub200' / variant / f'seed_{seed}'
        directory.mkdir(parents=True)
        spec = {'variant': variant, 'seed': seed, 'fingerprint': 'test-' + variant,
                'command': ['main.py', '--data_cfg', 'configs/datasets/cub200.yaml']}
        (directory / 'run_spec.json').write_text(json.dumps(spec), encoding='utf-8')
        (directory / 'run_state.json').write_text(json.dumps({'status': 'completed', 'fingerprint': spec['fingerprint']}), encoding='utf-8')
        metrics = {'status': 'completed', 'dataset': 'CUB200', 'seed': seed,
                   'clip_weight_sha256': 'same-clip', 'sessions': [{'session': 0, 'accuracy': 100.}, {'session': 1, 'accuracy': 100.}]}
        (directory / 'metrics.json').write_text(json.dumps(metrics), encoding='utf-8')
        (directory / 'config.yaml').write_text('DATASET:\n  NUM_INIT_CLS: 2\n', encoding='utf-8')
        (directory / 'support.json').write_text('{"same": true}', encoding='utf-8')
        for session in (0, 1):
            self.predictions(directory, f'predictions_session_{session:02d}.npz', [0, 1], [0, 2], [0, 2])
        if variant != 'G1':
            (directory / 'guidance_calibration.json').write_text(json.dumps({
                'selected_rho': .5, 'candidate_results': [], 'selection_status': 'manual_no_validation'}), encoding='utf-8')
            for session in (0, 1):
                (directory / f'guidance_session_{session:02d}.json').write_text(json.dumps({
                    'class_ids': [2], 'diagnostics': {'movement_degrees': [1],
                                                    'guided_prototypes': [[1., 0.]]}}), encoding='utf-8')
        return directory

    def test_end_to_end_archive_excludes_checkpoint_and_unrelated_run(self):
        with artifact_directory() as directory:
            root = Path(directory)
            for variant in ('G1', 'G3', 'G4'):
                path = self.make_run(root / 'original', variant)
                (path / 'checkpoint.pt').write_bytes(b'not a trusted pickle')
                (path / 'secret.env').write_text('exclude me', encoding='utf-8')
            self.make_run(root / 'original', 'G1', seed=2)
            output, archive = root / 'diagnostic', root / 'bundle.tar.gz'
            with contextlib.redirect_stdout(io.StringIO()):
                status = tool.main(['--root', str(root / 'original'), '--case', 'cub200:1',
                                    '--output', str(output), '--archive', str(archive)])
            self.assertEqual(status, 0)
            report = json.loads((output / 'diagnostic.json').read_text(encoding='utf-8'))
            self.assertEqual(len(report['runs']), 3)
            self.assertEqual(len(report['comparisons']), 6)
            self.assertEqual(report['runs'][0]['num_base'], 2)
            with tarfile.open(archive) as packed:
                names = packed.getnames()
            self.assertIn('raw/cub200/G4/seed_1/guidance_calibration.json', names)
            self.assertFalse(any('checkpoint' in name or 'secret' in name or 'seed_2' in name for name in names))
            self.assertIn('reports/diagnosis.md', names)

    def test_summary_only_archive_is_incomplete_and_actionable(self):
        with artifact_directory() as directory:
            root = Path(directory)
            (root / 'summary_cub200.json').write_text('{}', encoding='utf-8')
            archive = root / 'bundle.tar.gz'
            with contextlib.redirect_stdout(io.StringIO()):
                status = tool.main(['--root', str(root), '--case', 'cub200:1',
                                    '--output', str(root / 'diagnostic'), '--archive', str(archive)])
            self.assertEqual(status, 1)
            report = json.loads((root / 'diagnostic/diagnostic.json').read_text(encoding='utf-8'))
            self.assertTrue(any('Summary JSON alone' in issue for issue in report['issues']))
            with tarfile.open(archive) as packed:
                self.assertIn('summaries/root_0/summary_cub200.json', packed.getnames())

    def test_mismatched_support_refuses_prediction_pair(self):
        with artifact_directory() as directory:
            root = Path(directory)
            self.make_run(root, 'G1')
            bad = self.make_run(root, 'G4')
            (bad / 'support.json').write_text('{"different": true}', encoding='utf-8')
            report, _ = tool.analyze([root], {('cub200', 1)}, ['G1', 'G4'])
            self.assertEqual(report['status'], 'incomplete')
            self.assertFalse(report['comparisons'])
            self.assertTrue(any('refusing paired analysis' in issue for issue in report['issues']))

    def test_stale_state_seed_predictions_and_protocol_are_detected(self):
        with artifact_directory() as directory:
            root = Path(directory)
            a, b = self.make_run(root, 'G3'), self.make_run(root, 'G4')
            (a / 'run_state.json').write_text('{"status":"completed","fingerprint":"stale"}', encoding='utf-8')
            metrics = json.loads((a / 'metrics.json').read_text(encoding='utf-8'))
            metrics['seed'] = 7
            metrics['sessions'][1]['accuracy'] = 0.
            (a / 'metrics.json').write_text(json.dumps(metrics), encoding='utf-8')
            for path, protocol in ((a, 'protocol-a'), (b, 'protocol-b')):
                (path / 'guidance_calibration.json').write_text(json.dumps({'protocol_sha256': protocol}), encoding='utf-8')
            report, _ = tool.analyze([root], {('cub200', 1)}, ['G3', 'G4'])
            issues = '\n'.join(report['issues'])
            self.assertIn('fingerprint differs', issues)
            self.assertIn('metrics seed differs', issues)
            self.assertIn('accuracy 100.0 disagrees', issues)
            self.assertIn('calibration-protocol mismatch', issues)
            self.assertFalse(report['comparisons'])

    def test_duplicate_identity_is_excluded_not_silently_selected(self):
        with artifact_directory() as directory:
            root = Path(directory)
            self.make_run(root / 'first', 'G1')
            self.make_run(root / 'second', 'G1')
            report, runs = tool.analyze([root], {('cub200', 1)}, ['G1'])
            self.assertFalse(runs)
            self.assertTrue(any('both excluded' in issue for issue in report['issues']))


if __name__ == '__main__':
    unittest.main()
