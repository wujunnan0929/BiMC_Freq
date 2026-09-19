"""Audit shared GDA through base calibration and incremental Runner boundaries."""

import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from engine.engine import Runner
from engine.uncertainty_calibration import initialize_uncertainty, _sequence_records
from test_residual_integration import make_model, temporary_artifact_directory
from test_uncertainty_calibration import calibration_state
from test_uncertainty_integration import uncertainty_config as make_uncertainty_config
import test_uncertainty_integration as uncertainty_integration


def discriminant_config(output_dir='', control='joint', alpha=0.1,
                        base_only=False, auto=False, covariance='full_shared'):
    cfg = make_uncertainty_config(output_dir, control, alpha, base_only, auto)
    cfg.defrost()
    settings = cfg.TRAINER.BiMC.UNCERTAINTY
    settings.COVARIANCE = covariance
    settings.MEAN_UNCERTAINTY = False
    settings.RIDGE = 0.1
    settings.RIDGE_GRID = [0.01, 0.1]
    settings.ALPHA_GRID = [0.0, 0.1, 0.2]
    settings.TEMPERATURE_GRID = [0.5, 1.0]
    cfg.freeze()
    return cfg


def joint_state(classes=6):
    state = calibration_state(classes=classes)
    state['uncertainty_features'] = torch.cat(
        (state['images_features'].unsqueeze(1), state['uncertainty_features']), dim=1,
    )
    return state


class DiscriminantIntegrationTest(unittest.TestCase):
    def test_calibration_excludes_validation_classes_and_query_statistics(self):
        cfg, state = discriminant_config(auto=True), joint_state()
        report = initialize_uncertainty(make_model(cfg), cfg, state)
        fit = set(report['fit_class_ids'])
        validation = set(report['validation_class_ids'])
        self.assertFalse(fit & validation)
        self.assertEqual(fit | validation, set(range(6)))
        self.assertIsInstance(report['selection_prior'], dict)
        self.assertIsInstance(report['prior'], dict)
        json.dumps(report, allow_nan=False)

        # Changing every held-out class must not change the covariance used to
        # select the hyperparameters, even though the selected setting may move.
        changed = copy.deepcopy(state)
        held_out = torch.isin(state['images_targets'], torch.tensor(sorted(validation)))
        changed['images_features'][held_out] = changed['images_features'][held_out].roll(1, -1)
        changed['uncertainty_features'][held_out] = changed['uncertainty_features'][held_out].roll(1, -1)
        changed_report = initialize_uncertainty(make_model(cfg), cfg, changed)
        self.assertEqual(report['selection_prior'], changed_report['selection_prior'])

        for sequence in report['validation_sequences']:
            self.assertTrue(set(sequence['class_ids']) <= validation)
            support = {i for row in sequence['support_indices'] for i in row}
            query = {i for row in sequence['query_indices'] for i in row}
            self.assertFalse(support & query)
        sequence = report['validation_sequences'][0]
        first = list(_sequence_records(make_model(cfg), cfg, state, sequence))
        query_indices = [i for row in sequence['query_indices'] for i in row]
        changed = copy.deepcopy(state)
        changed['images_features'][query_indices] *= -1
        changed['uncertainty_features'][query_indices] *= -1
        second = list(_sequence_records(make_model(cfg), cfg, changed, sequence))
        for original, perturbed in zip(first, second):
            for field in ('stats', 'reference_state'):
                for key, value in original[field].items():
                    self.assertTrue(torch.equal(value, perturbed[field][key]), key)

    def test_ridge_candidates_deduplicate_zero_and_manual_needs_no_episodes(self):
        cfg = discriminant_config(auto=True)
        with patch('engine.uncertainty_calibration.mix_uncertainty_probabilities',
                   side_effect=lambda reference, logits, alpha, temperature: reference):
            report = initialize_uncertainty(make_model(cfg), cfg, joint_state())
        candidates = report['candidate_results']
        self.assertEqual(sum(row['alpha'] == 0 for row in candidates), 1)
        self.assertEqual(len(candidates), 1 + 2 * 2 * 2)
        self.assertEqual(report['selected_alpha'], 0.0)
        self.assertIn('ridge', report['selected'])
        self.assertEqual({row['ridge'] for row in candidates if row['alpha'] > 0},
                         {0.01, 0.1})

        cfg = discriminant_config(alpha=0.2, auto=False)
        model = make_model(cfg)
        with patch('engine.uncertainty_calibration.sample_sequence',
                   side_effect=AssertionError('Manual GDA must not need validation episodes')):
            report = initialize_uncertainty(model, cfg, joint_state(classes=2))
        self.assertEqual(report['selection_status'], 'manual_unvalidated')
        self.assertEqual(report['selected']['ridge'], 0.1)
        self.assertEqual(model.uncertainty_state['ridge'], 0.1)
        self.assertEqual(report['candidate_results'], [])
        self.assertEqual(report['validation_sequences'], [])

    def test_shared_gda_rejects_mean_uncertainty_and_non_base_provenance(self):
        for covariance in ('full_shared', 'block_shared'):
            with self.subTest(covariance=covariance):
                cfg = discriminant_config(covariance=covariance)
                bad = cfg.clone()
                bad.defrost()
                bad.TRAINER.BiMC.UNCERTAINTY.MEAN_UNCERTAINTY = True
                bad.freeze()
                with self.assertRaisesRegex(ValueError, 'MEAN_UNCERTAINTY'):
                    initialize_uncertainty(make_model(bad), bad, joint_state())
                for marker, value in (('data_source', 'test'), ('task_id', 1)):
                    state = joint_state()
                    state[marker] = value
                    with self.assertRaises(ValueError):
                        initialize_uncertainty(make_model(cfg), cfg, state)

    def test_zero_bypasses_encoding_and_joint_query_contains_original_once(self):
        cfg, state = discriminant_config(), joint_state()
        model = make_model(cfg)
        images = state['images_features'][::5]
        views = state['uncertainty_features'][::5, 1:]
        reference = torch.ones(len(images), 6) / 6
        model.uncertainty_state = {'alpha': 0.0}
        with patch.object(model, 'extract_frequency_img_feature') as encode:
            output = model.apply_frequency_uncertainty(images, images, reference, {})
        encode.assert_not_called()
        self.assertIs(output, reference)

        precision = torch.eye(4 * images.shape[-1])
        model.uncertainty_state = {
            'alpha': 0.1, 'temperature': 1.0,
            'covariance': 'full_shared', 'precision': precision,
        }
        from models.frequency_uncertainty import class_statistics
        stats = class_statistics(state['uncertainty_features'], state['images_targets'], range(6))
        with patch.object(model, 'extract_frequency_img_feature', return_value=views) as encode, \
                patch('models.frequency_discriminant.prepared_discriminant_logits',
                      return_value=torch.zeros_like(reference)) as score:
            output = model.apply_frequency_uncertainty(images, images, reference, stats)
        encode.assert_called_once()
        self.assertIs(encode.call_args.kwargs['original_features'], images)
        used_features, prepared = score.call_args.args
        self.assertTrue(torch.equal(used_features, state['uncertainty_features'][::5]))
        self.assertEqual(prepared['num_bands'], 4)
        self.assertTrue(torch.isfinite(output).all())
        self.assertFalse(output.requires_grad)

    def test_repeat_control_skips_frequency_and_cache_updates_with_new_classes(self):
        from models.frequency_discriminant import prepare_discriminant
        from models.frequency_uncertainty import class_statistics
        cfg, state = discriminant_config(control='repeat'), joint_state()
        model = make_model(cfg)
        images = state['images_features'][::5]
        repeated = state['images_features'].unsqueeze(1).expand(-1, 4, -1)
        stats = class_statistics(repeated, state['images_targets'], range(6))
        precision = torch.eye(4 * images.shape[-1])
        model.uncertainty_state = {'alpha': 0.1, 'temperature': 1.,
                                  'covariance': 'full_shared', 'precision': precision}
        reference = torch.ones(len(images), 6) / 6
        with patch.object(model, 'extract_frequency_img_feature',
                          side_effect=AssertionError('Repeat control must not encode frequency views')), \
                patch('models.frequency_discriminant.prepare_discriminant',
                      wraps=prepare_discriminant) as prepare:
            first = model.apply_frequency_uncertainty(images, images, reference, stats)
            second = model.apply_frequency_uncertainty(images, images, reference, stats)
            torch.testing.assert_close(first, second)
            self.assertEqual(prepare.call_count, 1)
            # Appending a class must rebuild the classifier instead of returning
            # the cached six-class output from the preceding session.
            expanded = {key: torch.cat((value, value[:1])) for key, value in stats.items()}
            output = model.apply_frequency_uncertainty(images, images, torch.ones(len(images), 7), expanded)
            self.assertEqual(output.shape, (len(images), 7))
            self.assertEqual(prepare.call_count, 2)
            expanded['mean'][0].mul_(0.9)
            model.apply_frequency_uncertainty(images, images, torch.ones(len(images), 7), expanded)
            self.assertEqual(prepare.call_count, 3)

    def _run(self, directory, *, covariance='full_shared', base_only=False):
        snapshots = []
        original_save = Runner._save_checkpoint

        def capture(runner, task_id, states):
            current = runner.model.uncertainty_state
            snapshots.append({
                'prior': current['prior'].clone(),
                'precision': current['precision'].clone(),
                'settings': {key: current[key] for key in
                             ('alpha', 'temperature', 'ridge', 'covariance')},
                'statistics': [{key: value.clone()
                                for key, value in item['uncertainty_statistics'].items()}
                               for item in states],
            })
            original_save(runner, task_id, states)

        def configured(*args, **kwargs):
            return discriminant_config(*args, **kwargs, covariance=covariance)

        # Reuse the synthetic frozen encoder, in-memory data manager, and
        # forbidden historical-image/test-loader checks of the diagonal suite.
        helper = uncertainty_integration.UncertaintyIntegrationTest()
        with patch('test_uncertainty_integration.uncertainty_config', side_effect=configured), \
                patch.object(Runner, '_save_checkpoint', new=capture):
            model, sessions, _ = helper._run(
                directory, control='joint', base_only=base_only,
            )
        return model, sessions, snapshots

    def test_incremental_freezes_metric_and_saves_real_checkpoint_tensors(self):
        for covariance in ('full_shared', 'block_shared'):
            with self.subTest(covariance=covariance), temporary_artifact_directory() as directory:
                model, sessions, snapshots = self._run(directory, covariance=covariance)
                self.assertEqual(len(sessions), 3)
                self.assertEqual(len(snapshots), 3)
                for later in snapshots[1:]:
                    for field in ('prior', 'precision'):
                        self.assertTrue(torch.equal(snapshots[0][field], later[field]), field)
                    self.assertEqual(snapshots[0]['settings'], later['settings'])
                for index, earlier in enumerate(snapshots[:-1]):
                    for task_id, stats in enumerate(earlier['statistics']):
                        for key, value in stats.items():
                            self.assertTrue(torch.equal(value,
                                            snapshots[index + 1]['statistics'][task_id][key]))
                self.assertEqual(sum(len(item['count']) for item in snapshots[-1]['statistics']), 10)
                self.assertTrue(torch.equal(snapshots[-1]['statistics'][-1]['count'],
                                            torch.full((2,), 2)))
                for session in sessions:
                    self.assertEqual(session['query_auxiliary_encodings'],
                                     3 * session['query_original_encodings'])
                checkpoint = torch.load(Path(directory) / 'checkpoint.pt', weights_only=False)
                for field in ('prior', 'precision'):
                    saved = checkpoint['uncertainty_state'][field]
                    self.assertTrue(torch.is_tensor(saved))
                    self.assertTrue(torch.equal(saved, snapshots[0][field]))
                    self.assertFalse(saved.requires_grad)
                report = json.loads((Path(directory) / 'uncertainty_calibration.json').read_text())
                self.assertIsInstance(report['prior'], dict)
                self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))

    def test_base_only_never_constructs_test_loader(self):
        with temporary_artifact_directory() as directory:
            _, sessions, snapshots = self._run(directory, base_only=True)
            self.assertEqual(sessions, [])
            self.assertEqual(len(snapshots), 1)
            metrics = json.loads((Path(directory) / 'metrics.json').read_text())
            self.assertEqual(metrics['status'], 'base_validation_completed')
            self.assertFalse(list(Path(directory).glob('predictions_session_*.npz')))


if __name__ == '__main__':
    unittest.main()
