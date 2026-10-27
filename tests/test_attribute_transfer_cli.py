"""The attribute study planner remains usable without a numerical runtime."""

import builtins
import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType
import unittest
from unittest.mock import patch
from uuid import uuid4

from tools import run_attribute_transfer_experiments as cli

ROOT = Path(__file__).resolve().parents[1]


def fake_engine():
    module = ModuleType('engine.attribute_transfer_experiments')
    module.experiment_variants = lambda:[{'name':method+'__'+offset, 'method':method, 'offset_mode':offset}
        for method in ('bimc','single_two_text','whole_transfer','random_mask','shuffled_mask','semantic_mask','semantic_reliable')
        for offset in ('none','a_inner')]
    module.run_experiments = lambda *args,**kwargs:{'completed_runs':42,'study_complete':True,'failed_runs':0}
    return module


class AttributeTransferCLITest(unittest.TestCase):
    def arguments(self, *extra):
        return ['--cache','missing/base.npz','--grounding','missing/grounding.npz',
                '--output-root','missing/output',*extra]

    def test_default_plan_never_imports_ml_or_reads_cache(self):
        engine = fake_engine()
        original_import = builtins.__import__
        def guarded(name,*args,**kwargs):
            if name.split('.')[0] in ('numpy','torch'):
                raise AssertionError('Numerical dependency imported by planner: '+name)
            return original_import(name,*args,**kwargs)
        with patch.dict(sys.modules,{'engine.attribute_transfer_experiments':engine}), \
                patch('builtins.__import__',side_effect=guarded), \
                patch('pathlib.Path.open',side_effect=AssertionError('Planner opened a file')), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            code = cli.main(self.arguments())
        self.assertEqual(code,0)
        content = output.getvalue()
        plan = json.loads(content[:content.rfind('\nDry run:')])
        self.assertEqual(plan['planned_configurations'],42)
        self.assertEqual(len(plan['runs']),42)
        self.assertTrue(all(row['stage']=='transfer' for row in plan['runs']))
        self.assertFalse(plan['benchmark_evaluation_performed'])

    def test_execute_forwards_protocol_and_offset_settings(self):
        engine = fake_engine()
        calls = []
        def execute(*args,**kwargs):
            calls.append((args,kwargs))
            return {'completed_runs':14,'study_complete':True}
        engine.run_experiments = execute
        with patch.dict(sys.modules,{'engine.attribute_transfer_experiments':engine}), \
                contextlib.redirect_stdout(io.StringIO()):
            code = cli.main(self.arguments('--execute','--fit-seeds','2','--sequences','3',
                            '--inner-sequences','2','--topk','7','--temperature','8',
                            '--blend-grid','0','.1','--offset-grid','-.05','0','.05','--no-resume'))
        self.assertEqual(code,0)
        args,options = calls[0]
        self.assertEqual(args,('missing/base.npz','missing/grounding.npz','missing/output'))
        self.assertEqual(options['fit_seeds'],(2,))
        self.assertEqual(options['blend_grid'],(0.,.1))
        self.assertEqual(options['offset_grid'],(-.05,0.,.05))
        self.assertEqual(options['inner_sequences'],2)
        self.assertFalse(options['resume'])

    def test_execute_creates_requested_package_after_completion(self):
        engine = fake_engine()
        from tools import package_attribute_transfer_results as package
        with patch.dict(sys.modules,{'engine.attribute_transfer_experiments':engine}), \
                patch.object(package,'package_results',return_value={'complete':True,'files':[]}) as write, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(self.arguments('--execute','--package','missing/analysis.zip')),0)
        write.assert_called_once_with('missing/output','missing/analysis.zip',allow_incomplete=False)

    def test_existing_package_is_rejected_before_running(self):
        folder = ROOT/'.codex_tmp'/('attribute_cli_'+uuid4().hex)
        folder.mkdir(parents=True)
        try:
            archive = folder/'existing.zip'
            archive.write_bytes(b'keep')
            engine = fake_engine()
            with patch.dict(sys.modules,{'engine.attribute_transfer_experiments':engine}), \
                    patch.object(engine,'run_experiments',side_effect=AssertionError('Should not run')), \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(self.arguments('--execute','--package',str(archive))),2)
            self.assertEqual(archive.read_bytes(),b'keep')
        finally:
            if folder.resolve().parent != (ROOT/'.codex_tmp').resolve():
                raise RuntimeError('Unsafe cleanup target')
            shutil.rmtree(folder)

    def test_invalid_protocol_options_are_rejected(self):
        invalid = [('--fit-seeds','2','2'),('--inner-sequences','0'),('--blend-grid','.1','.5'),
                   ('--blend-grid','0','nan'),('--offset-grid','.1','.2'),
                   ('--offset-grid','0','inf'),('--offset-grid','0','11'),('--topk','0'),('--temperature','0'),
                   ('--max-group-drop','-.1'),('--allow-incomplete-package',)]
        for extra in invalid:
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    cli.parse_args(self.arguments(*extra))
                self.assertEqual(error.exception.code,2)

    def test_cli_help_does_not_require_engine_or_torch(self):
        result = subprocess.run([sys.executable,str(ROOT/'tools/run_attribute_transfer_experiments.py'),'--help'],
                                cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--grounding',result.stdout)
        self.assertIn('--inner-sequences',result.stdout)


if __name__=='__main__':
    unittest.main()
