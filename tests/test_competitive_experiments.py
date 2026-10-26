"""Paired commands, artifact identity and confusion accounting."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from main import setup_cfg
from engine.safd_metric_runtime import reference_settings
from test_residual_integration import temporary_artifact_directory
from tools import run_competitive_fusion_experiments as tool
from tools import train_competitive_fusion as train_cli
from utils.competition_metrics import competition_errors


class CompetitiveExperimentTest(unittest.TestCase):
    def artifact(self, directory, name='joint', **changes):
        path = Path(directory)/(name+'.npz')
        path.write_bytes(b'planner fixture; numerical NPZ is checked by runtime')
        root = Path(__file__).resolve().parents[1]
        cfg = setup_cfg(str(root/'configs/datasets/cub200.yaml'), str(root/'configs/trainers/bimc_competitive_fusion.yaml'))
        metadata = {'format':'competitive_fusion_v1', 'dataset':'cub200', 'data_source':'base_training_only',
            'fitting_seed':2, 'selection_protocol_sha256':'same-protocol', 'cache_sha256':'same-cache',
            'calibration': {'fused_temperature':1.,'cov_temperature':1.,'description_temperature':1.,'novel_bias':0.},
            'fixed': {'alpha_old':.8,'alpha_novel':.8,'novel_bias_delta':0.},
            'reference_settings': reference_settings(cfg), 'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            **changes}
        path.with_suffix('.json').write_text(json.dumps(metadata))
        return path

    def test_paired_matrix_and_original_zero_fixed_controls(self):
        with temporary_artifact_directory() as directory:
            joint = self.artifact(directory)
            safd = self.artifact(directory, 'joint_safd')
            args = tool.parse_args(['--model','joint='+str(joint),'--model','joint_safd='+str(safd),'--seeds','1','2'])
            plan = tool.build_plan(args)
            self.assertEqual(len(plan), 12)
            for seed in (1,2):
                self.assertEqual(len({r['support_manifest'] for r in plan if r['seed']==seed}), 1)
            for row in plan:
                options = row['command'][row['command'].index('--opts')+1:]
                cfg = setup_cfg(args.data_cfg, args.train_cfg, options)
                p = cfg.TRAINER.BiMC.COMPETITIVE_FUSION
                self.assertFalse(cfg.TRAINER.BiMC.PROTOTYPE_ESTIMATOR.ENABLED)
                self.assertFalse(cfg.TRAINER.BiMC.SAFD_METRIC.ENABLED)
                if row['variant']=='baseline':
                    self.assertFalse(p.ENABLED)
                elif row['variant'] in ('zero_full','calibrated','fixed'):
                    self.assertEqual(p.MODE, {'zero_full':'zero','calibrated':'calibrated','fixed':'fixed'}[row['variant']])

    def test_mismatched_fixed_selection_or_seed_or_checksum_rejected(self):
        with temporary_artifact_directory() as directory:
            first = self.artifact(directory)
            for changes in ({'fitting_seed':3}, {'fixed':{}}, {'calibration':{}}, {'cache_sha256':'other'}):
                second = self.artifact(directory, 'other', **changes)
                with self.assertRaisesRegex(ValueError, 'share reference'):
                    tool.build_plan(tool.parse_args(['--model','joint='+str(first),'--model','other='+str(second)]))
            first.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                tool.build_plan(tool.parse_args(['--model','joint='+str(first)]))

    def test_dry_run_and_managed_options_and_fresh_training_output(self):
        with temporary_artifact_directory() as directory:
            path = self.artifact(directory)
            with patch.object(tool.common, 'execute_plan') as execute, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(tool.main(['--model','joint='+str(path),'--seeds','1']), 0)
            execute.assert_not_called()
            with self.assertRaisesRegex(ValueError, 'matrix-managed'):
                tool.build_plan(tool.parse_args(['--model','joint='+str(path),'--opts',tool.P+'BLEND','1.']))
            with self.assertRaisesRegex(ValueError, 'fresh'):
                train_cli.main(['--cache','missing.npz','--output',str(path)])

    def test_confusion_separates_group_and_within_group_errors(self):
        y = np.arange(6)
        prediction = np.array([1,3,2,0,5,5])
        result = competition_errors(np.eye(6)[prediction], y, 3)
        self.assertEqual(result['counts'], {'old_to_novel':1, 'novel_to_old':1,
            'old_within_group_errors':1, 'novel_within_group_errors':1})
        base = competition_errors(np.eye(3), np.arange(3), 3)
        self.assertIsNone(base['rates']['novel_to_old'])


if __name__ == '__main__':
    unittest.main()
