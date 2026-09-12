import contextlib
import io
import json
from pathlib import Path
import unittest

from main import setup_cfg
from tools import run_frequency_consensus_experiments as experiments
from test_residual_integration import temporary_artifact_directory
from test_incremental_residual_experiments import example_metrics


class ConsensusExperimentTest(unittest.TestCase):
    def test_calibration_matrix_changes_only_selection_and_has_shared_support(self):
        args = experiments.parse_args(['--suite', 'calibration', '--support-root', 'outputs/shared_support'])
        plan = experiments.build_plan(args)
        self.assertEqual(len(plan), 21)
        for seed in args.seeds:
            rows = [row for row in plan if row['seed'] == seed]
            self.assertEqual(len({row['support_manifest'] for row in rows}), 1)
            variants = {}
            for row in rows:
                command = row['command']
                cfg = setup_cfg(args.data_cfg, args.train_cfg, command[command.index('--opts')+1:])
                variants[row['variant']] = cfg.TRAINER.BiMC.CONSENSUS
            self.assertEqual(variants['consensus'].OBJECTIVE, 'micro_all')
            self.assertEqual(variants['consensus_inc_micro'].OBJECTIVE, 'micro_incremental')
            self.assertEqual(variants['consensus_balanced'].OBJECTIVE, 'balanced_incremental')
            self.assertEqual(variants['consensus_safe'].MAX_GROUP_DROP_PP, 0.)
            self.assertFalse(variants['zero'].AUTO_CALIBRATE)
            self.assertEqual(variants['zero'].LAMBDA, 0.)
        with self.assertRaises(ValueError):
            experiments.build_plan(experiments.parse_args(['--suite', 'calibration', '--fixed-lambda', '.1']))

    def test_per_seed_base_lambda_source_and_provenance_validation(self):
        with temporary_artifact_directory() as directory:
            args = experiments.parse_args(['--suite', 'calibration', '--seeds', '1', '2',
                                          '--output-root', directory])
            sources = experiments.build_plan(args)
            for row in sources:
                if row['variant'] != 'consensus_balanced':
                    continue
                output = Path(row['output_dir'])
                output.mkdir(parents=True)
                (output / 'run_spec.json').write_text(json.dumps(row), encoding='utf-8')
                (output / 'consensus_calibration.json').write_text(json.dumps({
                    'data_source': 'base_training_only', 'seed': row['seed']+9101,
                    'run_seed': row['seed'], 'auto_calibrate': True,
                    'selected_lambda': .01 if row['seed'] == 1 else 0.,
                    'selection_objective': 'balanced_incremental',
                }), encoding='utf-8')
            argv = ['--seeds', '1', '2', '--lambda-from-root', directory,
                    '--output-root', str(Path(directory) / 'controls')]
            plan = experiments.build_plan(experiments.parse_args(argv))
            for row in plan:
                opts = row['command'][row['command'].index('--opts')+1:]
                values = dict(zip(opts[::2], opts[1::2]))
                self.assertEqual(values[experiments.C+'AUTO_CALIBRATE'], 'False')
                self.assertEqual(float(values[experiments.C+'LAMBDA']),
                                 .01 if row['variant'] != 'zero' and row['seed'] == 1 else 0.)
                self.assertTrue(row['lambda_provenance']['sha256'])
            extended = experiments.build_plan(experiments.parse_args(argv + ['--suite', 'all']))
            hashes = {(row['seed'], row['variant']): row['fingerprint'] for row in extended}
            self.assertTrue(all(row['fingerprint'] == hashes[row['seed'], row['variant']] for row in plan))
            with self.assertRaisesRegex(ValueError, 'override differs'):
                experiments.build_plan(experiments.parse_args(argv + ['--opts', 'DATASET.ROOT', '/different']))
            path = Path(directory) / 'cub200/consensus_balanced/seed_1/consensus_calibration.json'
            bad = json.loads(path.read_text())
            bad['data_source'] = 'test'
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, 'base-only'):
                experiments.build_plan(experiments.parse_args(argv))

    def test_full_matrix_uses_shared_support_and_valid_yaml_overrides(self):
        args = experiments.parse_args(['--suite', 'all', '--opts', 'DEVICE.GPU_ID', '0'])
        plan = experiments.build_plan(args)
        self.assertEqual(len(plan), 36)
        for seed in args.seeds:
            runs = [row for row in plan if row['seed'] == seed]
            self.assertEqual(len({row['support_manifest'] for row in runs}), 1)
        for run in plan[:12]:
            command = run['command']
            cfg = setup_cfg(args.data_cfg, args.train_cfg, command[command.index('--opts') + 1:])
            self.assertEqual(cfg.DEVICE.GPU_ID, '0')
            self.assertFalse(cfg.TRAINER.BiMC.RESIDUAL.ENABLED)
            if cfg.TRAINER.BiMC.CONSENSUS.ENABLED:
                self.assertFalse(cfg.TRAINER.BiMC.FREQUENCY.ENABLED)
                self.assertFalse(cfg.TRAINER.BiMC.FREQUENCY.ROUTER.ENABLED)
        core = experiments.build_plan(experiments.parse_args(['--suite', 'core', '--opts', 'DEVICE.GPU_ID', '0']))
        fingerprints = {(row['seed'], row['variant']): row['fingerprint'] for row in plan}
        self.assertTrue(all(row['fingerprint'] == fingerprints[row['seed'], row['variant']] for row in core))

    def test_dry_run_never_creates_outputs(self):
        with temporary_artifact_directory() as directory:
            output = Path(directory) / 'not created'
            with contextlib.redirect_stdout(io.StringIO()):
                code = experiments.main(['--dry-run', '--output-root', str(output), '--seeds', '1'])
            self.assertEqual(code, 0)
            self.assertFalse(output.exists())

    def test_fixed_lambda_keeps_zero_control_and_identical_calibration_seed(self):
        args = experiments.parse_args(['--seeds', '2', '--fixed-lambda', '.003'])
        plan = experiments.build_plan(args)
        for row in plan:
            opts = row['command'][row['command'].index('--opts')+1:]
            values = dict(zip(opts[::2], opts[1::2]))
            self.assertEqual(values[experiments.C+'CALIBRATION_SEED'], '9103')
            self.assertEqual(values[experiments.C+'AUTO_CALIBRATE'], 'False')
            self.assertEqual(values[experiments.C+'LAMBDA'], '0.0' if row['variant'] == 'zero' else '0.003')

    def test_reject_protocol_override_and_nondeterministic_seed(self):
        for argv in (['--seeds', '0'], ['--opts', 'SEED', '99'],
                     ['--opts', 'TRAINER.BiMC.CONSENSUS.MODE', 'visual'],
                     ['--fixed-lambda', '-1']):
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                experiments.build_plan(experiments.parse_args(argv))

    def test_summary_preserves_paired_differences_and_zero_sample_audit(self):
        import numpy as np
        with temporary_artifact_directory() as directory:
            root = Path(directory)
            rows = []
            for name, accuracy in (('baseline', 70.), ('zero', 70.), ('consensus', 71.)):
                output = root / name
                output.mkdir()
                payload = example_metrics(accuracy)
                if name == 'consensus':
                    payload['consensus'] = {'protocol_sha256': 'shared-protocol',
                                            'selection_objective': 'balanced_incremental',
                                            'candidate_results': [{'lambda': .03, 'stages': [{'stage': 1}]}]}
                (output / 'metrics.json').write_text(json.dumps(payload), encoding='utf-8')
                (output / 'support.json').write_text('{"same": true}', encoding='utf-8')
                np.savez_compressed(output / 'predictions_session_01.npz',
                                    sample_id=np.array([10, 20]), target=np.array([0, 1]),
                                    reference_prediction=np.array([0, 0]),
                                    prediction=np.array([0, 1] if name == 'consensus' else [0, 0]))
                rows.append({'variant': name, 'seed': 1, 'status': 'completed',
                             'output_dir': str(output), 'summary': payload['summary']})
            args = experiments.parse_args(['--seeds', '1'])
            experiments.save_summary(root / 'summary.json', args, rows, 3)
            report = json.loads((root / 'summary.json').read_text())
            self.assertEqual(report['paired_delta_vs_baseline']['consensus']['final_accuracy']['mean'], 1.)
            calibrated = next(row for row in report['runs'] if row['variant'] == 'consensus')
            self.assertEqual(calibrated['consensus_calibration']['candidate_results'][0]['stages'], [{'stage': 1}])
            self.assertIn('base_accuracy', report['paired_delta_vs_reference']['consensus']['baseline'])
            self.assertEqual(report['pairing_audit']['1']['sample_comparisons']['zero']
                             ['output_prediction_mismatches'], 0)
            self.assertEqual(report['pairing_audit']['1']['sample_comparisons']['consensus']
                             ['reference_prediction_mismatches'], 0)


if __name__ == '__main__':
    unittest.main()
