"""End-to-end base-only selection and export back to the engine configuration."""

import contextlib
import io
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import yaml

from main import setup_cfg
from tools import tune_bimc as tool
from fdrfreq import select as sel
from fdrfreq.bank import save_bank
from fdrfreq.pipeline import EpisodeRunner
from fdrfreq.synthetic import make_bank


@contextlib.contextmanager
def workspace():
    parent = (tool.ROOT / '.codex_tmp').resolve()
    path = parent / ('auto_retune_' + uuid4().hex)
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        resolved = path.resolve()
        if resolved.parent != parent or not resolved.name.startswith('auto_retune_'):
            raise RuntimeError('Invalid test cleanup target')
        shutil.rmtree(resolved)


def tiny_search():
    return {'objective': 'hm', 'split_mode': 'random', 'split_seeds': [2, 3], 'sequences': 2,
            'grid': {'DATASET.BETA': [0.0, 1.0], 'TRAINER.BiMC.LAMBDA_T': [0.0, 0.5]}}


def plan(directory, search=None, opts=()):
    spec = directory / 'search.yaml'
    spec.write_text(yaml.safe_dump(search if search is not None else tiny_search()), encoding='utf-8')
    args = tool.parse_args([
        '--dataset', 'cub200', '--bank', str(directory / 'bank'),
        '--search-cfg', str(spec), '--out', str(directory / 'results'),
        '--shot', '1', '--opts',
        'DATASET.NUM_CLASSES', '12', 'DATASET.NUM_INIT_CLS', '8', 'DATASET.NUM_INC_CLS', '2',
        *opts,
    ])
    return tool.build_plan(args)


def bank():
    value = make_bank(num_classes=12, num_init=8, num_inc=2, dim=16, n_train=12, n_test=3)
    value.meta.update(dataset='cub200', backbone='ViT-B/16')
    return value


class AutomaticRetuneTest(unittest.TestCase):
    def test_default_preview_works_without_a_bank_and_writes_nothing(self):
        with workspace() as path, contextlib.redirect_stdout(io.StringIO()) as output:
            rc = tool.main(['--dataset', 'cub200', '--bank', str(path / 'missing'),
                            '--out', str(path / 'preview'), '--dry-run'])
            self.assertEqual(rc, 0)
            self.assertIn('270 combinations', output.getvalue())
            self.assertFalse((path / 'preview').exists())

    def test_search_never_accesses_test_arrays_and_exports_matching_profile(self):
        with workspace() as path:
            p, b = plan(path), bank()
            original = b.array

            def base_only(name):
                self.assertFalse(name.startswith('test_'), 'Search accessed benchmark data: ' + name)
                return original(name)

            with patch.object(b, 'array', side_effect=base_only), contextlib.redirect_stdout(io.StringIO()):
                result = tool.run_search(p, b)
            self.assertFalse(result['test_evaluated'])
            self.assertEqual(len(result['candidates']), 4)
            self.assertEqual(result['dev_protocol']['n_episodes'], 4)
            best = p['out'] / 'best.yaml'
            exported = setup_cfg(
                str(tool.ROOT / 'configs/datasets/cub200.yaml'),
                str(tool.ROOT / 'configs/trainers/bimc_ensemble.yaml'), retune_cfg_file=str(best),
            )
            self.assertEqual(tool.pipeline_config(exported), result['selected_config'])
            self.assertEqual(exported.TRAINER.BiMC.PREC, 'fp32')
            self.assertNotEqual(Path(exported.OUTPUT_DIR).resolve(), p['out'].resolve())
            locked = json.loads((p['out'] / 'locked_BiMC_auto.json').read_text(encoding='utf-8'))
            self.assertEqual(locked['config_hash'], sel.config_hash(result['selected_config']))
            report = json.loads((p['out'] / 'search_report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['search_settings'], tiny_search())
            chosen = result['fallback_metrics']['last_hm'] if result['selected_params'] == 'fallback' else next(
                row['last_hm'] for row in result['candidates'] if row['hash'] == result['selected_hash'])
            self.assertGreaterEqual(chosen + 1e-8, result['fallback_metrics']['last_hm'])

    def test_grid_search_matches_independent_candidate_evaluation(self):
        with workspace() as path, contextlib.redirect_stdout(io.StringIO()):
            p, b = plan(path), bank()
            result = tool.run_search(p, b)
            episodes = sel.dev_episodes(b, [2, 3], 2, 1, 'random')
            configs, fallback = sel.expand_grid(p['entry'])
            hm_values = []
            for _, cfg in configs:
                means = [EpisodeRunner(b, ep).run(cfg)['summary']['last_hm'] for ep in episodes]
                hm_values.append(float(np.mean(means)))
            fallback_hm = float(np.mean([
                EpisodeRunner(b, ep).run(fallback)['summary']['last_hm'] for ep in episodes]))
            selected_hm = float(np.mean([
                EpisodeRunner(b, ep).run(result['selected_config'])['summary']['last_hm'] for ep in episodes]))
            self.assertAlmostEqual(selected_hm, max([fallback_hm] + hm_values))
            self.assertEqual([round(row['last_hm'], 8) for row in result['candidates']],
                             [round(v, 8) for v in hm_values])

    def test_all_seven_parameters_map_to_pipeline_and_export(self):
        with workspace() as path:
            search = tiny_search()
            search['grid'] = {'DATASET.BETA': [0.6], 'TRAINER.BiMC.LAMBDA_T': [0.7],
                              'TRAINER.BiMC.LAMBDA_I': [0.2], 'TRAINER.BiMC.TAU': [32],
                              'DATASET.ENSEMBLE_ALPHA': [0.4],
                              'TRAINER.BiMC.GAMMA_BASE': [2.0], 'TRAINER.BiMC.GAMMA_INC': [3.0]}
            p = plan(path, search)
            candidates, _ = sel.expand_grid(p['entry'])
            exported = tool.best_profile(p['cfg'], candidates[0][1])
            for public, cached in tool.PARAMETERS.items():
                self.assertEqual(tool.get_value(exported, public), search['grid'][public][0])
                self.assertEqual(tool.get_value(candidates[0][1], cached), search['grid'][public][0])
            self.assertEqual(tool.pipeline_config(exported), candidates[0][1])

    def test_fixed_values_and_plain_bimc_are_preserved(self):
        with workspace() as path, contextlib.redirect_stdout(io.StringIO()):
            search = tiny_search()
            search['grid'] = {'DATASET.BETA': [0.7]}
            p = plan(path, search, ['TRAINER.BiMC.USING_ENSEMBLE', 'False',
                                    'TRAINER.BiMC.LAMBDA_I', '0.0'])
            self.assertEqual(p['entry']['base']['holistic']['lam'], 0.0)
            result = tool.run_search(p, bank())
            profile = tool.best_profile(p['cfg'], result['selected_config'])
            self.assertFalse(profile.TRAINER.BiMC.USING_ENSEMBLE)
            self.assertIsNone(tool.pipeline_config(profile)['ensemble'])
            self.assertEqual(profile.TRAINER.BiMC.LAMBDA_I, 0.0)

    def test_search_keys_cannot_conflict_with_manual_overrides(self):
        with workspace() as path, self.assertRaisesRegex(ValueError, 'both grid and --opts'):
            plan(path, opts=['DATASET.BETA', '0.6'])

    def test_bad_ranges_empty_grids_and_unknown_keys_fail_early(self):
        for grid in ({'TYPO': [0.5]}, {'DATASET.BETA': []}, {'DATASET.BETA': [1.2]},
                     {'DATASET.BETA': [True]}, {'DATASET.BETA': [0.5, 0.5]},
                     {'DATASET.BETA': [float('nan')]}, {'TRAINER.BiMC.TAU': [0]}):
            with self.subTest(grid=grid), workspace() as path, self.assertRaises(ValueError):
                search = tiny_search()
                search['grid'] = grid
                plan(path, search)

    def test_invalid_protocol_and_inactive_parameters_are_rejected(self):
        for override in ({'split_seeds': [2, 2]}, {'split_seeds': []}, {'sequences': 0},
                         {'objective': 'test_accuracy'}, {'split_mode': 'test'}, {'typo': 1}):
            with self.subTest(override=override), workspace() as path, self.assertRaises(ValueError):
                search = {**tiny_search(), **override}
                plan(path, search)
        with workspace() as path, self.assertRaisesRegex(ValueError, 'TEXT_CALIBRATION'):
            plan(path, opts=['TRAINER.BiMC.TEXT_CALIBRATION', 'False'])

    def test_bank_mismatch_is_rejected_before_creating_search_outputs(self):
        for key, value in [('dataset', 'cifar100'), ('backbone', 'ViT-L/14'), ('smoke_subset', 2)]:
            with self.subTest(key=key), workspace() as path:
                p, b = plan(path), bank()
                b.meta[key] = value
                with self.assertRaises(ValueError):
                    tool.run_search(p, b)
                self.assertFalse(p['out'].exists())

    def test_completed_search_cannot_be_overwritten(self):
        with workspace() as path, contextlib.redirect_stdout(io.StringIO()):
            p, b = plan(path), bank()
            tool.run_search(p, b)
            previous = (p['out'] / 'best.yaml').read_bytes()
            with self.assertRaisesRegex(ValueError, 'nonempty'):
                tool.run_search(p, b)
            self.assertEqual(previous, (p['out'] / 'best.yaml').read_bytes())

    def test_cli_runs_on_saved_bank_and_reports_missing_bank_clearly(self):
        with workspace() as path, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as errors:
            spec = path / 'grid.yaml'
            search = tiny_search()
            search.update(split_seeds=[2], sequences=1)
            spec.write_text(yaml.safe_dump(search), encoding='utf-8')
            args = ['--dataset', 'cub200', '--bank', str(path / 'bank'), '--search-cfg', str(spec),
                    '--out', str(path / 'results'), '--shot', '1', '--opts',
                    'DATASET.NUM_CLASSES', '12', 'DATASET.NUM_INIT_CLS', '8', 'DATASET.NUM_INC_CLS', '2']
            self.assertEqual(tool.main(args), 2)
            self.assertIn('Feature bank not found', errors.getvalue())
            b = bank()
            save_bank(path / 'bank', b._arrays, b.meta)
            self.assertEqual(tool.main(args), 0)
            self.assertTrue((path / 'results/best.yaml').is_file())

    def test_selection_progress_is_reported_once_per_episode(self):
        b, events = bank(), []
        episodes = sel.dev_episodes(b, [2], 2, 1, 'random')
        entry = {'name': 'fixed', 'base': {'ensemble': None}, 'grid': {}}
        sel.select(b, entry, episodes, progress=lambda done, total: events.append((done, total)))
        self.assertEqual(events, [(1, 2), (2, 2)])


if __name__ == '__main__':
    unittest.main()
