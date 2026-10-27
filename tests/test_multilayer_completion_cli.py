"""The base-only study planner must remain inspectable without numerical dependencies."""

import builtins
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType
import unittest
from unittest.mock import patch
from uuid import uuid4

from tools import run_multilayer_completion_experiments as cli


@contextlib.contextmanager
def workspace_directory():
    parent = (cli.ROOT/'.codex_tmp').resolve()
    folder = parent/('multilayer_cli_'+uuid4().hex)
    folder.mkdir(parents=True)
    try:
        yield folder
    finally:
        resolved = folder.resolve()
        if resolved.parent != parent or not resolved.name.startswith('multilayer_cli_'):
            raise RuntimeError('Refusing cleanup outside the test workspace.')
        shutil.rmtree(resolved)


def engine_fixture():
    module = ModuleType('engine.multilayer_completion_experiments')
    counts = {'repeat': 9, 'layers': 7, 'attributes': 10}
    module.experiment_variants = lambda stage, layer: [
        {'name': stage+'_'+str(i), 'stage': stage, 'selected_layer': layer}
        for i in range(counts[stage])]
    return module


class MultilayerCompletionCLITest(unittest.TestCase):
    def args(self, *extra):
        return cli.parse_args(['--cache', 'not_loaded.npz', '--output-root', 'not_written', *extra])

    def test_defaults_and_validation_do_not_require_a_cache(self):
        args = self.args()
        self.assertEqual(args.fit_seeds, [2, 3, 4])
        self.assertEqual(args.stages, ['repeat', 'layers', 'attributes'])
        self.assertFalse(args.execute)
        for values in [('--fit-seeds', '2', '2'), ('--stages', 'repeat', 'repeat'),
                       ('--sequences', '0'), ('--blend-grid', '.5', '1'),
                       ('--blend-grid', '0', 'nan'), ('--max-group-drop', '-1'),
                       ('--selected-layer', '4'), ('--allow-incomplete-package',)]:
            with self.subTest(values=values), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.args(*values)

    def test_dry_run_imports_no_numerical_library_and_writes_nothing(self):
        original_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in ('torch', 'torchvision', 'numpy'):
                raise AssertionError('Planning imported ' + name)
            return original_import(name, *args, **kwargs)
        with workspace_directory() as folder:
            out = folder/'not_created'
            with patch.dict(sys.modules, {'engine.multilayer_completion_experiments': engine_fixture()}), \
                    patch('builtins.__import__', side_effect=guarded), contextlib.redirect_stdout(io.StringIO()) as capture:
                result = cli.main(['--cache', str(folder/'absent.npz'), '--output-root', str(out)])
            self.assertEqual(result, 0)
            self.assertFalse(out.exists())
            self.assertIn('78 base-only configurations', capture.getvalue())
            self.assertEqual(list(folder.iterdir()), [])

    def test_stage_order_and_manual_layer_are_explicit_in_plan(self):
        with patch.dict(sys.modules, {'engine.multilayer_completion_experiments': engine_fixture()}):
            plan = cli.build_plan(self.args('--fit-seeds', '4', '--stages', 'attributes', 'layers', '--selected-layer', '6'))
        self.assertEqual(plan['stages'], ['layers', 'attributes'])
        self.assertEqual(plan['planned_configurations'], 17)
        self.assertEqual(plan['layer_selection'], 'manual_predeclared')
        self.assertTrue(all(row['configuration']['selected_layer'] == 6 for row in plan['runs']))
        self.assertFalse(plan['benchmark_evaluation_performed'])

    def test_real_engine_plan_does_not_import_numerical_libraries(self):
        original_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in ('torch', 'torchvision', 'numpy'):
                raise AssertionError('Actual engine planning imported ' + name)
            return original_import(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guarded):
            plan = cli.build_plan(self.args('--fit-seeds', '2'))
        self.assertEqual(plan['planned_configurations'], 26)
        self.assertEqual({row['stage'] for row in plan['runs']}, {'repeat', 'layers', 'attributes'})

    def test_execute_forwards_all_options_and_can_package(self):
        module = engine_fixture()
        seen = {}
        def run(cache, root, **kwargs):
            seen.update(cache=cache, root=root, options=kwargs)
            return {'completed_runs': 9, 'failed_runs': 0}
        module.run_experiments = run
        with workspace_directory() as folder, \
                patch.dict(sys.modules, {'engine.multilayer_completion_experiments': module}), \
                patch('tools.package_multilayer_completion_results.package_results', return_value={'complete': True, 'files': []}) as package, \
                contextlib.redirect_stdout(io.StringIO()):
            result = cli.main(['--cache', 'base.npz', '--output-root', str(folder/'run'),
                               '--fit-seeds', '5', '--stages', 'repeat', '--sequences', '3',
                               '--threads', '2', '--device', 'cpu', '--blend-grid', '0', '.5',
                               '--max-group-drop', '.1', '--no-resume', '--execute',
                               '--package', str(folder/'result.zip')])
        self.assertEqual(result, 0)
        self.assertEqual(seen['options'], {'fit_seeds': (5,), 'stages': ('repeat',), 'sequences': 3,
            'device': 'cpu', 'threads': 2, 'maximum_group_drop_pp': .1,
            'blend_grid': (0., .5), 'selected_layer': None, 'resume': False})
        package.assert_called_once()

    def test_existing_package_is_rejected_before_running_engine(self):
        module = engine_fixture()
        module.run_experiments = lambda *a, **kw: self.fail('Must refuse before running experiments.')
        with workspace_directory() as folder:
            archive = folder/'existing.zip'
            archive.write_bytes(b'preserved')
            with patch.dict(sys.modules, {'engine.multilayer_completion_experiments': module}), \
                    contextlib.redirect_stderr(io.StringIO()):
                result = cli.main(['--cache', 'cache.npz', '--output-root', str(folder/'run'),
                                   '--execute', '--package', str(archive)])
            self.assertEqual(result, 2)
            self.assertEqual(archive.read_bytes(), b'preserved')


if __name__ == '__main__':
    unittest.main()
