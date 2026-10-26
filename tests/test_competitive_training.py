"""Selection isolation, fixed guards, loss/refit and paired SAFD support traces."""
import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
import torch

from engine import competitive_training as training
from engine import competitive_evaluation as evaluation
from engine.prototype_evaluation import prepare_context, validation_records, identity_calibration
from test_safd_metric_training import fixture


def settings():
    return training.FusionOptions(seed=1, sequences=2, teacher_shot=2, query=2, old_shot=4,
        old_ways=(1,), novel_ways=(1,), novel_shots=(2,), episode_pool=3,
        steps=2, validate_every=1, early_checkpoints=(1,), hidden=8,
        blend_grid=(0., .5), weight_grid=(.8, 1.), bias_grid=(0.,))


class CompetitiveTrainingTest(unittest.TestCase):
    def example(self):
        arrays, cfg, _ = fixture()
        o = settings()
        cfg, protocol, pool = prepare_context(arrays, cfg, o.seed, o.sequences, o.teacher_shot)
        records, trace = training.training_episodes(arrays, cfg, pool, o, identity_calibration())
        return arrays, cfg, o, protocol, pool, records, trace

    def test_fitting_excludes_reserved_queries_and_all_validation_classes(self):
        _, _, _, protocol, _, records, trace = self.example()
        fit = set(protocol['fitting_indices'])
        for ep in trace:
            self.assertTrue(set(ep['class_ids']) <= set(protocol['pseudo_base_class_ids']))
            for s, q in zip(ep['support_indices'], ep['query_indices']):
                self.assertFalse(set(s)&set(q))
                self.assertTrue((set(s)|set(q)) <= fit)
            self.assertNotIn('teacher_indices', ep)
        self.assertTrue(all(not value.requires_grad for row in records for value in row['fusion_probabilities'].values()))

    def test_auxiliary_preserves_episode_sampling_and_ordinary_queries(self):
        arrays, cfg, o, _, pool, original, trace = self.example()
        o.auxiliary, o.auxiliary_probability = 'safd', 1.
        perturbed, changed = training.training_episodes(arrays, cfg, pool, o, identity_calibration(), auxiliary_features=arrays['features'])
        for a, b, x, y in zip(trace, changed, original, perturbed):
            self.assertEqual({k:v for k,v in a.items() if k != 'auxiliary_edits'},
                             {k:v for k,v in b.items() if k != 'auxiliary_edits'})
            self.assertEqual(len(b['auxiliary_edits']), 1)
            self.assertTrue(torch.equal(x['query'], y['query']))

    def test_zero_and_optimized_parameters_have_live_gradients_and_frozen_inputs(self):
        _, cfg, o, _, _, records, _ = self.example()
        model = training.make_model(o, 'cpu')
        fixed = evaluation.identity_fixed(cfg)
        before = {k:v.clone() for k,v in model.state_dict().items()}
        history = training.optimize(model, records, cfg, identity_calibration(), fixed, o, 2)
        self.assertEqual(len(history), 2)
        self.assertTrue(any(not torch.equal(v, model.state_dict()[k]) for k,v in before.items()))
        self.assertTrue(all(p.grad is None for row in records for p in row['fusion_probabilities'].values()))
        self.assertTrue(all(np.isfinite(list(row.values())).all() for row in history))

    def test_fixed_selection_identity_and_both_group_guards(self):
        arrays, cfg, o, protocol, _, _, _ = self.example()
        records = evaluation.prepare_records(validation_records(arrays, cfg, protocol), cfg, identity_calibration())
        fixed, report = evaluation.select_fixed_fusion(records, cfg, identity_calibration(), 0., (.8, 1.), (0., .1))
        self.assertEqual(report['candidates'][0]['fixed'], evaluation.identity_fixed(cfg))
        self.assertTrue(evaluation.feasible(report['selected_validation'],
                        (report['reference'], report['calibrated_reference']), 0.))
        identity = evaluation.fusion_record_scores(records[0], cfg, identity_calibration(), evaluation.identity_fixed(cfg))
        self.assertIs(identity, records[0]['reference_votes'])
        diagnostics = evaluation.group_diagnostics(records, lambda row: evaluation.fusion_record_scores(row, cfg, identity_calibration(), fixed))
        self.assertEqual(diagnostics['counts']['queries'], sum(len(r['targets']) for r in records))

    def test_validation_teachers_cannot_affect_selection(self):
        arrays, cfg, o, protocol, _, _, _ = self.example()
        changed = {k:np.array(v, copy=True) for k,v in arrays.items()}
        for rows in protocol['audit_indices'].values():
            changed['features'][rows, 0] = 0.
            changed['features'][rows, 0, -1] = 1.
        metadata = {'complete_base_bank': True, 'has_text_features': True}
        with contextlib.redirect_stdout(io.StringIO()):
            _, a, _ = training.fit_competition(arrays, metadata, cfg, o)
            _, b, _ = training.fit_competition(changed, metadata, cfg, o)
        for key in ('calibration', 'fixed', 'selected_validation', 'selection_training_history'):
            self.assertEqual(a[key], b[key])

    def test_zero_steps_frozen_model_and_no_refit_oracle_selection(self):
        arrays, cfg, _ = fixture()
        o = settings(); o.steps = 0
        with contextlib.redirect_stdout(io.StringIO()), patch.object(training, 'training_episodes', wraps=training.training_episodes) as episodes:
            model, report, _ = training.fit_competition(arrays, {'complete_base_bank':True, 'has_text_features':True}, cfg, o)
        episodes.assert_not_called()
        self.assertEqual(report['selected_step'], 0)
        self.assertEqual(report['blend'], 0.)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        self.assertEqual(report['phase_order'][-3:], ['selection_ended', 'fresh_all_base_fixed_step_refit', 'freeze'])

    def test_anchor_replay_avoids_search_and_rejects_different_protocol(self):
        from engine.safd_metric_runtime import reference_settings, sample_identity_hash
        arrays, cfg, _ = fixture()
        o = settings(); o.steps = 0
        metadata = {'complete_base_bank':True, 'has_text_features':True, 'cache_sha256':'same-cache', 'clip_weight_sha256':'clip'}
        with contextlib.redirect_stdout(io.StringIO()):
            _, first, actual_cfg = training.fit_competition(arrays, metadata, cfg, o)
        anchor = {'cache_sha256':'same-cache', 'clip_weight_sha256':'clip', 'fitting_seed':o.seed,
            'reference_settings':reference_settings(actual_cfg), 'selection_protocol_sha256':first['selection_protocol']['sha256'],
            'base_sample_ids_sorted_sha256':sample_identity_hash(arrays['sample_id']),
            'calibration':first['calibration'], 'fixed':first['fixed'], 'model_sha256':'verified-by-CLI',
            'fixed_candidate_count':first['fixed_report']['candidate_count']}
        with patch.object(training, 'select_calibration', side_effect=AssertionError('Repeated search')), \
             patch.object(training, 'select_fixed_fusion', side_effect=AssertionError('Repeated search')), \
             contextlib.redirect_stdout(io.StringIO()):
            _, replay, _ = training.fit_competition(arrays, metadata, cfg, o, fixed_anchor=anchor)
        self.assertEqual(replay['fixed_reference'], first['fixed_reference'])
        self.assertEqual(replay['fixed_report']['candidate_count'], 0)
        with self.assertRaisesRegex(ValueError, 'anchor cache/seed/protocol'):
            training.fit_competition(arrays, metadata, cfg, o, fixed_anchor={**anchor, 'fitting_seed':2})

    def test_selected_duration_refits_fresh_on_all_base_after_selection(self):
        arrays, cfg, _ = fixture()
        o = settings()
        events = []
        original_episodes, original_optimize = training.training_episodes, training.optimize
        original_evaluate = training.evaluate_records
        def episodes(*args, **kwargs):
            events.append(('episodes', sorted(args[2])))
            return original_episodes(*args, **kwargs)
        def optimize(*args, **kwargs):
            events.append(('optimize', id(args[0]), args[6]))
            return original_optimize(*args, **kwargs)
        def evaluate(*args, **kwargs):
            events.append(('validation',))
            metrics = original_evaluate(*args, **kwargs)
            return {**metrics, 'harmonic_accuracy': metrics['harmonic_accuracy']+.01}
        with patch.object(training, 'training_episodes', side_effect=episodes), \
             patch.object(training, 'optimize', side_effect=optimize), \
             patch.object(training, 'evaluate_records', side_effect=evaluate), contextlib.redirect_stdout(io.StringIO()):
            _, report, _ = training.fit_competition(arrays, {'complete_base_bank':True, 'has_text_features':True}, cfg, o)
        self.assertGreater(report['selected_step'], 0)
        fits = [event for event in events if event[0] == 'optimize']
        self.assertEqual(len(fits), 2)
        self.assertNotEqual(fits[0][1], fits[1][1])
        self.assertEqual(fits[1][2], report['selected_step'])
        pools = [event[1] for event in events if event[0] == 'episodes']
        self.assertEqual(len(pools[0]), 3)
        self.assertEqual(pools[1], list(range(6)))
        self.assertEqual(len(report['final_training_history']), report['selected_step'])


if __name__ == '__main__':
    unittest.main()
