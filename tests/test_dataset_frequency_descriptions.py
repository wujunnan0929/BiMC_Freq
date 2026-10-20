"""Dataset routing, generation and fail-fast checks for the missing dual runs."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

from main import setup_cfg
from models.bimc import BiMC
from test_residual_integration import temporary_artifact_directory
from tools import generate_cub200_frequency_descriptions as generator
from tools import generate_cub200_frequency_descriptions_deepseek as provider
from tools import run_safd_dual_experiments as launcher
from utils.frequency_descriptions import (
    BANDS, DATASET_SPECS, REPO_ROOT, dataset_class_names, description_path,
    load_explicit_descriptions, normalize_name, validate_frequency_inputs,
)


def payload(name, candidates=5):
    # Parser fixtures, not LLM-generated prompts or a semantic-quality claim.
    return {band: [f'a photo of a {name} with {band} visible cue {i}' for i in range(candidates)]
            for band in BANDS}


def write_fixture(path, dataset):
    path.write_text(json.dumps({name: payload(name) for name in dataset_class_names(dataset)}), encoding='utf-8')


class DatasetFrequencyDescriptionTest(unittest.TestCase):
    def test_canonical_classes_and_default_sources_match_all_three_datasets(self):
        for dataset, spec in DATASET_SPECS.items():
            with self.subTest(dataset=dataset):
                names = dataset_class_names(dataset)
                source = generator.load_class_names(REPO_ROOT / spec['source'], dataset)
                self.assertEqual(len(names), spec['count'])
                self.assertEqual({normalize_name(n) for n in names}, {normalize_name(n) for n in source})

    def test_equal_class_counts_cannot_hide_the_wrong_dataset(self):
        with self.assertRaisesRegex(ValueError, 'do not match'):
            generator.load_class_names(REPO_ROOT / DATASET_SPECS['miniimagenet']['source'], 'cifar100')

    def test_automatic_config_paths_and_explicit_override(self):
        for dataset in DATASET_SPECS:
            cfg = setup_cfg(f'configs/datasets/{dataset}.yaml', 'configs/trainers/bimc_safd_dual.yaml')
            self.assertEqual(Path(cfg.TRAINER.BiMC.FREQUENCY.EXPLICIT_DESCRIPTION_PATH), description_path(dataset))
        cfg = setup_cfg('configs/datasets/cifar100.yaml', 'configs/trainers/bimc_safd_dual.yaml',
                        ['TRAINER.BiMC.FREQUENCY.EXPLICIT_DESCRIPTION_PATH', 'custom.json'])
        self.assertEqual(cfg.TRAINER.BiMC.FREQUENCY.EXPLICIT_DESCRIPTION_PATH, 'custom.json')

    def test_deepseek_defaults_and_messages_are_dataset_specific(self):
        for dataset in ('cifar100', 'miniimagenet'):
            with patch.dict(os.environ, {}, clear=True):
                args = provider.apply_deepseek_defaults(['--dataset=' + dataset])
            self.assertEqual(Path(args[args.index('--output') + 1]), description_path(dataset))
            prompt = generator.build_user_prompt(['apple'], 5, 22, dataset)
            self.assertIn(DATASET_SPECS[dataset]['label'], prompt)
            self.assertNotIn('Fine local properties: feather texture', prompt)
            self.assertNotIn('bird image recognition', generator.system_prompt(dataset))
            body = provider.build_deepseek_request_body('fixture', prompt, .2, 8192, 'none', True,
                                                       system_message=generator.system_prompt(dataset))
            self.assertIn('object', body['messages'][0]['content'])
        self.assertIn('32 x 32', generator.build_user_prompt(['apple'], 5, 22, 'cifar100'))

    def test_generation_checkpoint_resume_and_offline_validation(self):
        with temporary_artifact_directory() as folder:
            output = Path(folder) / 'cifar.json'
            names = generator.load_class_names(REPO_ROOT / DATASET_SPECS['cifar100']['source'], 'cifar100')[:2]
            calls = []
            def fake_call(**kwargs):
                calls.append(kwargs)
                name = names[len(calls) - 1]
                return json.dumps({name: payload(name)})
            args = ['--dataset', 'cifar100', '--output', str(output), '--limit', '2', '--batch-size', '1',
                    '--model', 'fixture', '--api-base', 'https://fixture.invalid', '--api-key-env', 'TEST_ONLY_KEY']
            with patch.dict(os.environ, {'TEST_ONLY_KEY': 'fixture'}), \
                    patch.object(generator, 'call_chat_completions', side_effect=fake_call), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(generator.main(args), 0)
            self.assertEqual(len(calls), 2)
            self.assertEqual(list(json.loads(output.read_text())), names)
            self.assertTrue(all('32 x 32' in call['user_prompt'] for call in calls))
            self.assertTrue(all('object' in call['system_message'] for call in calls))
            with patch.object(generator, 'call_chat_completions') as api, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(generator.main(args + ['--validate-only']), 0)
                self.assertEqual(generator.main(args), 0)
            api.assert_not_called()

    def test_full_dataset_preflight_and_class_normalization(self):
        for dataset in ('cifar100', 'miniimagenet'):
            with self.subTest(dataset=dataset), temporary_artifact_directory() as folder:
                path = Path(folder) / 'text.json'
                write_fixture(path, dataset)
                cfg = setup_cfg(f'configs/datasets/{dataset}.yaml', 'configs/trainers/bimc_safd_dual.yaml',
                    ['TRAINER.BiMC.FREQUENCY.EXPLICIT_DESCRIPTION_PATH', str(path)])
                report = validate_frequency_inputs(cfg)
                self.assertEqual((report['classes'], report['candidates_per_band']), (100, 5))
                names = dataset_class_names(dataset)[:3]
                aliases = [name.upper().replace(' ', '_').replace('-', '_') for name in names]
                self.assertEqual(load_explicit_descriptions(path, aliases), [payload(n) for n in names])

    def test_incomplete_bands_and_candidate_counts_are_rejected(self):
        with temporary_artifact_directory() as folder:
            path = Path(folder) / 'bad.json'
            names = ['apple', 'aquarium fish']
            invalid = {n: payload(n) for n in names}
            invalid[names[1]]['high'] = []
            path.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, 'non-empty string list'):
                load_explicit_descriptions(path, names)
            invalid[names[1]]['high'] = payload(names[1], 3)['high']
            path.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, 'same number of candidates'):
                load_explicit_descriptions(path, names)

    def test_invalid_later_class_is_detected_before_any_clip_encoding(self):
        with temporary_artifact_directory() as folder:
            path = Path(folder) / 'bad.json'
            path.write_text(json.dumps({'apple': payload('apple'), 'aquarium fish': {'low': []}}))
            class Encoder:
                device = 'cpu'
            with patch('models.bimc.clip.tokenize') as tokenize, self.assertRaises(ValueError):
                BiMC.inference_explicit_frequency_description_candidates(Encoder(), ['apple', 'aquarium fish'], path)
            tokenize.assert_not_called()

    def test_actual_candidate_encoder_handles_both_dataset_names(self):
        class FakeClip:
            def encode_text(self, tokens):
                return torch.arange(1, 9).float().expand(len(tokens), -1)
        class Encoder:
            device, clip_model = 'cpu', FakeClip()
        for dataset in ('cifar100', 'miniimagenet'):
            with temporary_artifact_directory() as folder:
                path = Path(folder) / 'full.json'
                write_fixture(path, dataset)
                names = dataset_class_names(dataset)
                with patch('models.bimc.clip.tokenize', side_effect=lambda desc: torch.zeros(len(desc), 77, dtype=torch.long)):
                    embeddings = BiMC.inference_explicit_frequency_description_candidates(Encoder(), names, path)
                self.assertEqual(tuple(embeddings.shape), (100, 3, 5, 8))
                self.assertTrue(torch.isfinite(embeddings).all())

    def test_launcher_dataset_selection_and_default_output_are_consistent(self):
        for dataset in ('cifar100', 'miniimagenet'):
            args = launcher.parse_args(['--dataset', dataset])
            self.assertEqual(Path(args.data_cfg).stem, dataset)
            self.assertEqual(Path(args.output_root).name, 'safd_dual_' + dataset)
            plan = launcher.build_plan(args)
            self.assertEqual(len(plan), 24)
            self.assertEqual(len({r['support_manifest'] for r in plan}), 3)

    def test_missing_or_wrong_text_stops_before_running_or_writing(self):
        with temporary_artifact_directory() as folder:
            output = Path(folder) / 'untouched'
            argv = ['--dataset', 'cifar100', '--execute', '--keep-going', '--output-root', str(output),
                    '--opts', launcher.F+'EXPLICIT_DESCRIPTION_PATH', str(Path(folder) / 'missing.json')]
            with patch.object(launcher.common, 'execute_plan') as run, \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(launcher.main(argv), 2)
            run.assert_not_called()
            self.assertFalse(output.exists())
            args = launcher.parse_args(['--dataset', 'miniimagenet', '--opts', launcher.F+'EXPLICIT_DESCRIPTION_PATH',
                str(REPO_ROOT / 'description/cub200_frequency_descriptions_deepseek.json')])
            with self.assertRaisesRegex(ValueError, 'house finch'):
                launcher.validate_plan_inputs(args, launcher.build_plan(args))

    def test_complete_text_allows_every_dual_variant_to_launch(self):
        for dataset in ('cifar100', 'miniimagenet'):
            with temporary_artifact_directory() as folder:
                path = Path(folder) / 'full.json'
                write_fixture(path, dataset)
                argv = ['--dataset', dataset, '--seeds', '1', '--variants', 'zero_dual', 'fft_dual', 'safd_dual',
                        '--execute', '--output-root', str(Path(folder) / 'runs'),
                        '--opts', launcher.F+'EXPLICIT_DESCRIPTION_PATH', str(path)]
                with patch.object(launcher.common, 'execute_plan', return_value=0) as run, contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(launcher.main(argv), 0)
                plan = run.call_args.args[0]
                self.assertEqual({r['variant'] for r in plan}, {'zero_dual', 'fft_dual', 'safd_dual'})

    def test_failed_subprocess_traceback_is_preserved_in_summary(self):
        with temporary_artifact_directory() as folder:
            directory = Path(folder)
            plan = [{'variant': 'fixture', 'seed': 1, 'support_seed': 1,
                     'output_dir': str(directory / 'failed'), 'support_manifest': str(directory / 'support.json'),
                     'fingerprint': 'fixture', 'cwd': str(REPO_ROOT),
                     'command': [sys.executable, '-c', 'import sys; print("ValueError: missing apple high band"); sys.exit(1)']}]
            args = launcher.parse_args(['--output-root', str(directory)])
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(launcher.common.execute_plan(plan, args), 1)
            row = json.loads((directory / 'summary.json').read_text())['runs'][0]
            self.assertIn('missing apple high band', row['error_tail'])
            self.assertTrue(Path(row['log_path']).is_file())


if __name__ == '__main__':
    unittest.main()
