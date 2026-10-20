import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from main import setup_cfg
from test_residual_integration import temporary_artifact_directory
from tools import run_frequency_guidance_experiments as tool


class GuidanceExperimentTest(unittest.TestCase):
    def test_paired_plans_and_actual_yacs_configs_for_three_datasets(self):
        for dataset in ('cub200','cifar100','miniimagenet'):
            args=tool.parse_args(['--dataset',dataset])
            plan=tool.build_plan(args)
            self.assertEqual(len(plan),15)
            for seed in args.seeds:
                runs=[r for r in plan if r['seed']==seed]
                self.assertEqual(len({r['support_manifest'] for r in runs}),1)
            for run in plan[:5]:
                command=run['command']
                opts=command[command.index('--opts')+1:]
                cfg=setup_cfg(args.data_cfg,args.train_cfg,opts)
                b=cfg.TRAINER.BiMC
                self.assertFalse(b.FREQUENCY.ENABLED or b.UNCERTAINTY.ENABLED or b.DUAL_FREQUENCY.ENABLED)
                self.assertTrue(b.USING_ENSEMBLE)
                self.assertEqual(b.PREC,'fp32')
                self.assertEqual(b.GUIDANCE.CALIBRATION_SEED,run['seed']+10101)
                if run['variant']=='G0':
                    self.assertFalse(b.VISION_CALIBRATION)
                    self.assertEqual(b.LAMBDA_I,0)
                if run['variant']=='G2':
                    self.assertEqual(b.GUIDANCE.MODE,'visual')

    def test_14_variants_cover_predeclared_controls(self):
        args=tool.parse_args(['--suite','all','--seeds','1'])
        plan=tool.build_plan(args)
        self.assertEqual(len(plan),14)
        self.assertIn('G4_ordinary',{r['variant'] for r in plan})
        self.assertEqual(len({r['fingerprint'] for r in plan}),14)

    def test_read_only_plan_and_fail_fast_full_text_preflight(self):
        with temporary_artifact_directory() as directory:
            output=Path(directory)/'unused'
            with contextlib.redirect_stdout(io.StringIO()),patch.object(tool.common,'execute_plan') as execute:
                self.assertEqual(tool.main(['--variants','G0','G1','--seeds','1','--output-root',str(output)]),0)
            execute.assert_not_called()
            self.assertFalse(output.exists())
            args=tool.parse_args(['--dataset','cifar100','--variants','G4','--output-root',str(output),
                                 '--opts',tool.shared.F+'EXPLICIT_DESCRIPTION_PATH',str(output/'missing.json')])
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()), \
                 patch.object(tool.common,'execute_plan') as execute:
                argv=['--dataset','cifar100','--variants','G4','--execute','--output-root',str(output),
                      '--opts',tool.shared.F+'EXPLICIT_DESCRIPTION_PATH',str(output/'missing.json')]
                self.assertEqual(tool.main(argv),2)
            execute.assert_not_called()
            self.assertFalse(output.exists())

    def test_cub_explicit_text_and_ordinary_control_pass_preflight(self):
        args=tool.parse_args(['--suite','all','--seeds','1'])
        checked=tool.validate_plan_inputs(args,tool.build_plan(args))
        self.assertEqual(checked['scale_text']['classes'],200)
        self.assertEqual(checked['ordinary_control']['unique_prompts_per_class'],15)

    def test_matrix_collision_odd_keys_and_duplicate_seeds_are_rejected(self):
        for extras in [['--opts',tool.G+'MODE','visual'],['--opts','bad'],['--seeds','1','1']]:
            with self.assertRaises(ValueError):
                tool.build_plan(tool.parse_args(extras))

    def test_mini_ordinary_budget_must_be_lowered_for_both_groups(self):
        args=tool.parse_args(['--dataset','miniimagenet','--suite','all','--variants','G4_ordinary'])
        with self.assertRaisesRegex(ValueError,'15 distinct'):
            tool.validate_plan_inputs(args,tool.build_plan(args))
        with temporary_artifact_directory() as directory:
            from utils.frequency_descriptions import dataset_class_names
            names=dataset_class_names('miniimagenet')
            path=Path(directory)/'semantic.json'
            path.write_text(json.dumps({name:{band:[f'{name} {band} description {i}' for i in range(5)]
                                                       for band in ('low','middle','high')} for name in names}))
            args=tool.parse_args(['--dataset','miniimagenet','--suite','all','--variants','G4','G4_ordinary',
                                 '--opts',tool.G+'CANDIDATES_PER_BAND','3',tool.shared.F+'EXPLICIT_DESCRIPTION_PATH',str(path)])
            report=tool.validate_plan_inputs(args,tool.build_plan(args))
            self.assertEqual(report['ordinary_control']['unique_prompts_per_class'],9)
            self.assertEqual(report['scale_text']['candidates_per_band'],3)

    def test_text_encoder_respects_reduced_budget_before_encoding(self):
        import torch
        from models.bimc import BiMC
        class Encoder:
            device='cpu'
            class clip_model:
                @staticmethod
                def encode_text(tokens):
                    return torch.ones(len(tokens),8)
        with temporary_artifact_directory() as directory:
            path=Path(directory)/'prompts.json'
            path.write_text(json.dumps({'apple':{b:[f'apple {b} detail {i}' for i in range(5)]
                                                    for b in ('low','middle','high')}}))
            with patch('models.bimc.clip.tokenize',side_effect=lambda values:torch.zeros(len(values),77,dtype=torch.long)) as tokenizer:
                out=BiMC.inference_explicit_frequency_description_candidates(Encoder(),['apple'],path,max_candidates=3)
            self.assertEqual(tuple(out.shape),(1,3,3,8))
            self.assertEqual([len(c.args[0]) for c in tokenizer.call_args_list],[3,3,3])

    def test_summary_paired_deltas_support_and_calibration_audit(self):
        with temporary_artifact_directory() as directory:
            rows=[]
            for seed in (1,2):
                for i,name in enumerate(('G0','G1','G2','G3','G4')):
                    path=Path(directory)/f'{name}_{seed}'
                    path.mkdir()
                    summary={key:60.+i+seed for key in tool.common.METRIC_NAMES}
                    metrics={'status':'completed','summary':summary,
                             'sessions':[{'accuracy':70.,'query_auxiliary_encodings':0,
                                          'novel_to_original_base_rate':12.,'original_base_to_novel_rate':2.}],
                             'guidance':({'selected_rho':.5,'protocol_sha256':'same_protocol',
                                          'selection_status':'base_class_disjoint_validation'} if i>1 else None)}
                    (path/'metrics.json').write_text(json.dumps(metrics))
                    (path/'support.json').write_text(json.dumps({'paired_seed':seed}))
                    rows.append({'variant':name,'seed':seed,'output_dir':str(path),'status':'completed','summary':summary})
            target=Path(directory)/'summary.json'
            tool.save_summary(target,tool.parse_args([]),rows,len(rows))
            report=json.loads(target.read_text())
            delta=report['paired_comparisons']['G4_minus_G1']['final_accuracy']
            self.assertEqual(delta,{'n':2,'mean':3.,'std':0.})
            self.assertTrue(report['pairing_audit']['1']['support_match'])
            self.assertTrue(report['pairing_audit']['1']['calibration_protocol_match'])


if __name__=='__main__':
    unittest.main()
