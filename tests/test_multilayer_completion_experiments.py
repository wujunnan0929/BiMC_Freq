"""Actual staged base-only engine on a complete, masked synthetic native cache."""

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import torch
from torch.nn import functional as F

from engine import multilayer_completion_experiments as engine
from engine.completion_calibration import completion_protocol
from tools import cache_multilayer_base_features as cache
from utils.attribute_descriptions import GROUPS, build_attribute_manifest
from test_attribute_descriptions import sentences
from test_guidance_integration import configuration
from test_residual_integration import synthetic_state


@contextlib.contextmanager
def artifact_directory():
    parent = Path(__file__).resolve().parents[1]/'.codex_tmp'/'multilayer_engine_tests'
    parent.mkdir(parents=True, exist_ok=True)
    directory = parent/uuid4().hex
    directory.mkdir()
    try:
        yield directory
    finally:
        if directory.resolve().parent != parent.resolve():
            raise RuntimeError('Unsafe synthetic experiment cleanup target')
        shutil.rmtree(directory)


def write_synthetic_cache(directory, samples_per_class=20):
    """A public test fixture for the engine and analysis-package integration."""
    directory = Path(directory)
    full = synthetic_state(num_classes=6, samples_per_class=samples_per_class)
    cfg = configuration(enabled=False, auto=False)
    cfg.defrost()
    cfg.TRAINER.BiMC.PROTOTYPE_COMPLETION.QUERY = 2
    cfg.TRAINER.BiMC.PROTOTYPE_COMPLETION.SHOT = 2
    cfg.TRAINER.BiMC.PROTOTYPE_COMPLETION.ENABLED = False
    cfg.freeze()
    arrays = {'final': full['images_features'].numpy().astype(np.float32),
              'labels': full['images_targets'].numpy().astype(np.int64),
              'sample_id': np.arange(len(full['images_targets']), dtype=np.int64)+1000,
              'text_features': full['text_features'].numpy().astype(np.float32),
              'description_features': full['description_features'].numpy().astype(np.float32),
              'description_targets': full['description_targets'].numpy().astype(np.int64),
              'description_proto': full['description_proto'].numpy().astype(np.float32),
              'structured_text': full['frequency_prompt_proto'].numpy().astype(np.float32)}
    generator = torch.Generator().manual_seed(602)
    for layer, dimension in ((3, 10), (6, 12), (9, 11)):
        mapping = torch.randn(8, dimension, generator=generator)
        values = full['images_features'] @ mapping + .04*torch.randn(len(arrays['labels']), dimension, generator=generator)
        arrays['layer_'+str(layer)] = F.normalize(values, dim=-1).numpy().astype(np.float32)
    names = ['base'+str(i) for i in range(6)]
    ordinary = {name: sentences(name) for name in names}
    ordinary['base3'] = ordinary['base3'][:6]
    supplied = {name: {group: sentences(name)[2*j:2*j+2] for j, group in enumerate(GROUPS)} for name in names}
    supplied['base0']['shape'] = []
    supplied['base1']['pattern'] = supplied['base1']['pattern'][:1]
    supplied['base2']['color'] = []
    manifest = build_attribute_manifest(ordinary, names, supplied)
    def encode(texts):
        output = []
        for text in texts:
            cls = int(re.search(r'base(\d+)', text).group(1))
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], 'little')
            rng = np.random.RandomState(seed)
            output.append(arrays['text_features'][cls]+.15*rng.standard_normal(8))
        return np.asarray(output, dtype=np.float32)
    arrays.update(cache.encode_attribute_text(manifest, encode, 8))
    metadata = {'format': cache.FORMAT, 'dataset': 'cub200', 'data_source': 'base_training_only',
                'source': 'native_vit_layers', 'complete_base_bank': True,
                'base_class_ids': list(range(6)), 'base_class_names': names,
                'selected_layers': [3, 6, 9], 'sample_count': len(arrays['labels']),
                'full_base_sample_count': len(arrays['labels']),
                'encoding_counts': {'original': len(arrays['labels']), 'auxiliary': 0,
                                    'benchmark_query': 0, 'selected_layer_forward_images': len(arrays['labels'])},
                'attribute_manifest': manifest, 'config_yaml': cfg.dump(),
                'benchmark_evaluation_performed': False, 'new_gradient_training_performed': False,
                'clip_weight_sha256': 'synthetic-no-model-download', 'seed': 1,
                'text_template': ['a photo of a {}.'], 'support_transform': 'synthetic-original-only'}
    support = {'sample_ids': arrays['sample_id'].tolist(), 'targets': arrays['labels'].tolist(),
               'data_source': 'base_training_only'}
    path = directory/'base.npz'
    cache.write_cache(path, arrays, metadata, support)
    loaded, certified = cache.load_cache(path)
    return path, loaded, certified


def run(cache_path, output, **extra):
    with contextlib.redirect_stdout(io.StringIO()):
        return engine.run_experiments(cache_path, output, fit_seeds=(2,), sequences=1,
                                      blend_grid=(0., .5, 1.), **extra)


class MultilayerCompletionExperimentsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_complete_study_keeps_missing_classes_and_zero_and_typed_repeat_are_exact(self):
        with artifact_directory() as directory:
            path, arrays, _ = write_synthetic_cache(directory)
            root = directory/'all'
            result = run(path, root)
            self.assertEqual(result['completed_runs'], 26)
            self.assertEqual(result['failed_runs'], 0)
            self.assertFalse(result['benchmark_evaluation_performed'])
            self.assertEqual({r['stage'] for r in result['runs']}, {'repeat', 'layers', 'attributes'})
            self.assertEqual(len({r['protocol_sha256'] for r in result['runs']}), 1)
            by_name = {(r['stage'], r['variant']): r for r in result['runs']}
            for row in result['runs']:
                folder = root/row['output_dir']
                protocol = json.loads((folder/'protocol.json').read_text())
                fit = set(protocol['fitting_indices'])
                reserved = {i for values in protocol['reserved_pseudo_base_query_indices'].values() for i in values}
                self.assertFalse(fit & reserved)
                self.assertEqual(set(arrays['labels'][list(fit)]), set(protocol['pseudo_base_class_ids']))
                self.assertEqual(len(protocol['pseudo_base_class_ids']), 3)
                self.assertEqual(len(protocol['pseudo_validation_class_ids']), 3)
                diagnostic = json.loads((folder/'diagnostics.json').read_text())
                teacher = diagnostic.get('teacher_diagnostic', {})
                if teacher.get('available'):
                    self.assertTrue(teacher['post_selection_only'])
                    self.assertFalse(teacher['selection_candidate'])
                    self.assertFalse(teacher['strict_performance_upper_bound'])
                    for detail in teacher['details']:
                        seq = protocol['sequences'][detail['sequence']]
                        position = seq['class_ids'].index(detail['class_id'])
                        used = seq['support_indices'][position]+seq['query_indices'][position]
                        self.assertFalse(set(detail['teacher_sample_ids']) & set(arrays['sample_id'][used].tolist()))
                        self.assertGreaterEqual(detail['teacher_count'], 2)
                stages = json.loads((folder/'stage_metrics.json').read_text())
                self.assertTrue(all(stage['exact_reference_scores'] for stage in stages['0.0']))
                self.assertTrue(all(stage['corrected'] == stage['damaged'] == 0 for stage in stages['0.0']))
                with np.load(folder/'predictions.npz', allow_pickle=False) as predictions:
                    np.testing.assert_array_equal(predictions['candidate_prediction'][0], predictions['reference_prediction'])
                    self.assertEqual(len(np.unique(predictions['target'])), 5)  # 3 old + two arriving classes.
            def predictions(name):
                row = by_name['repeat', name]
                with np.load(root/row['output_dir']/'predictions.npz', allow_pickle=False) as saved:
                    return saved['candidate_prediction'].copy()
            np.testing.assert_array_equal(predictions('typed_single'), predictions('typed_repeat'))
            attr = by_name['attributes', 'final_attribute_groups']
            diagnostics = json.loads((root/attr['output_dir']/'diagnostics.json').read_text())
            self.assertEqual(len(diagnostics['text_coverage_by_class']['shape']), 6)
            self.assertFalse(diagnostics['text_coverage_by_class']['shape'][0])
            self.assertFalse(diagnostics['text_coverage_by_class']['color'][2])
            self.assertTrue((root/attr['output_dir']/'group_diagnostics.json').is_file())
            from tools.package_multilayer_completion_results import package_results
            bundle = package_results(root, directory/'analysis.zip')
            self.assertTrue(bundle['complete'])
            self.assertEqual(bundle['audit']['errors'], [])
            self.assertEqual(bundle['audit']['completed_run_records'], 26)
            self.assertFalse(any(item['path'].endswith('/base.npz') for item in bundle['files']))

    def test_candidate_scorer_matches_shared_bimc_algebraic_implementation(self):
        from engine.prototype_evaluation import score_record
        from engine.safd_metric_training import _text_subset
        with artifact_directory() as directory:
            _, saved, metadata = write_synthetic_cache(directory)
            engine._runtime()
            arrays = {k: torch.as_tensor(v) for k,v in saved.items()}
            cfg = engine._configuration(metadata, 2, 1)
            protocol = completion_protocol(cfg, arrays['labels'], saved['sample_id'].tolist())
            for row in engine.build_records(arrays, cfg, protocol):
                dp, df, dt, text = _text_subset(arrays, row['class_ids'], 'cpu')
                shared = {'query': row['query'], 'reference_prototypes': row['reference_prototypes'],
                          'cov_inverse': row['inverse64'], 'text_features': text,
                          'description_proto': dp, 'description_features': df, 'description_targets': dt,
                          '_description_logits': row['description_logits'], 'num_old': row['num_old'],
                          'reference_votes': row['reference_votes']}
                moved = F.normalize(row['reference_prototypes']+.03*row['reference_prototypes'].roll(1, -1), dim=-1)
                actual = engine._scorer(row, cfg, moved)
                expected = score_record(shared, cfg, image_prototypes=moved)
                torch.testing.assert_close(actual, expected, atol=1e-7, rtol=2e-6)
                self.assertTrue(torch.equal(actual.argmax(1), expected.argmax(1)))

    def test_attribute_treatments_use_the_same_valid_pool_and_budget_without_filling_unknowns(self):
        with artifact_directory() as directory:
            _, saved, metadata = write_synthetic_cache(directory)
            engine._runtime()
            arrays = {k: torch.as_tensor(v) for k,v in saved.items()}
            cfg = engine._configuration(metadata, 2, 1)
            pool = engine._pool
            observed = {}
            snapshots = {k: arrays[k].clone() for k in ('attr_mask', 'ordinary_mask', 'attr_text')}
            for mode in ('ordinary', 'attribute_mean', 'attribute_groups', 'random_groups'):
                captured = []
                def recording(values, mask):
                    captured.append((values.clone(), mask.clone()))
                    return pool(values, mask)
                with patch.object(engine, '_pool', side_effect=recording):
                    texts, masks = engine.semantic_blocks(arrays, cfg, mode)
                self.assertEqual(len(captured), 1)
                values, mask = captured[0]
                values = values.reshape(6, -1, 8)
                mask = mask.reshape(6, -1)
                observed[mode] = [(values[i][mask[i]].tolist()) for i in range(6)]
                for name, valid in masks.items():
                    self.assertTrue(torch.all(texts[name][~valid] == 0))
            expected_counts = torch.minimum(arrays['attr_mask'].sum((1, 2)), arrays['ordinary_mask'].sum(1)).tolist()
            for mode, classes in observed.items():
                self.assertEqual([len(items) for items in classes], expected_counts, mode)
            for i in range(6):
                expected = sorted(observed['attribute_mean'][i])
                self.assertEqual(sorted(observed['attribute_groups'][i]), expected)
                self.assertEqual(sorted(observed['random_groups'][i]), expected)
            for key, value in snapshots.items():
                self.assertTrue(torch.equal(arrays[key], value), key)

    def test_only_fitting_rows_and_classes_enter_typed_statistics(self):
        with artifact_directory() as directory:
            _, saved, metadata = write_synthetic_cache(directory)
            engine._runtime()
            arrays = {k: torch.as_tensor(v) for k,v in saved.items()}
            cfg = engine._configuration(metadata, 2, 1)
            protocol = completion_protocol(cfg, arrays['labels'], saved['sample_id'].tolist())
            definition = next(r for r in engine.experiment_variants('attributes', 3)
                              if r['name'] == 'middle_attribute_groups')
            first = engine.fit_variant(definition, arrays, cfg, protocol, {})
            changed = {k: v.clone() for k,v in arrays.items()}
            excluded = list(set(range(len(saved['labels'])))-set(protocol['fitting_indices']))
            for key in ['final', 'layer_3', 'layer_6', 'layer_9']:
                changed[key][excluded] *= -2
            for cls in protocol['pseudo_validation_class_ids']:
                changed['attr_text'][cls] *= -3
                changed['text_features'][cls] *= -3
            second = engine.fit_variant(definition, changed, cfg, protocol, {})
            for key in ['basis', 'H', 'bias', 'mean', 'between', 'Q', 'W', 'fit_class_ids']:
                self.assertTrue(torch.equal(first['state'][key], second['state'][key]), key)
            self.assertEqual(first['state']['fit_sample_count'], len(protocol['fitting_indices']))

    def test_query_and_future_support_do_not_change_current_class_completion(self):
        with artifact_directory() as directory:
            _, saved, metadata = write_synthetic_cache(directory)
            engine._runtime()
            arrays = {k: torch.as_tensor(v) for k,v in saved.items()}
            cfg = engine._configuration(metadata, 2, 1)
            protocol = completion_protocol(cfg, arrays['labels'], saved['sample_id'].tolist())
            sequence = protocol['sequences'][0]
            definition = next(r for r in engine.experiment_variants('layers') if r['name'] == 'layer3')
            first = engine.fit_variant(definition, arrays, cfg, protocol, {})
            completed = engine.sequence_completions(first, arrays, sequence)
            changed = {k: v.clone() for k,v in arrays.items()}
            query_rows = [i for group in sequence['query_indices'] for i in group]
            future_rows = sequence['support_indices'][-1]
            for key in ['final', 'layer_3']:
                changed[key][query_rows] *= -1
                changed[key][future_rows] *= -1
            second = engine.fit_variant(definition, changed, cfg, protocol, {})
            other = engine.sequence_completions(second, changed, sequence)
            self.assertTrue(torch.equal(completed[0], other[0]))

    def test_layer_selection_is_required_then_saved_selection_resumes_into_attributes(self):
        with artifact_directory() as directory:
            path, _, _ = write_synthetic_cache(directory)
            with self.assertRaisesRegex(ValueError, 'layer selection'):
                run(path, directory/'missing', stages=('attributes',))
            root = directory/'staged'
            layers = run(path, root, stages=('layers',))
            self.assertEqual(layers['completed_runs'], 7)
            self.assertIn('2', layers['layer_selection'])
            attributes = run(path, root, stages=('attributes',))
            self.assertEqual(attributes['completed_runs'], 17)
            self.assertEqual(set(attributes['requested_stages']), {'layers', 'attributes'})
            selected = attributes['layer_selection']['2']['selected_layer']
            self.assertIn(selected, (3, 6, 9))

    def test_interrupted_study_reports_full_plan_and_resumes_only_incomplete_runs(self):
        from tools.package_multilayer_completion_results import package_results
        with artifact_directory() as directory:
            path, _, _ = write_synthetic_cache(directory)
            root = directory/'interrupted'
            original_fit = engine.fit_variant
            def interrupted(definition, *args, **kwargs):
                if definition['name'] == 'original32':
                    raise RuntimeError('injected interruption')
                return original_fit(definition, *args, **kwargs)
            with patch.object(engine, 'fit_variant', side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, 'injected interruption'):
                    run(path, root, stages=('repeat',))
            summary = json.loads((root/'summary.json').read_text())
            self.assertEqual((summary['planned_runs'], summary['completed_runs'],
                              summary['failed_runs'], summary['pending_runs']), (9, 1, 1, 7))
            self.assertFalse(summary['study_complete'])
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                package_results(root, directory/'refused.zip')
            partial = package_results(root, directory/'partial.zip', allow_incomplete=True)
            self.assertFalse(partial['complete'])
            completed_path = root/'seed_2/repeat/original16/predictions.npz'
            stamp = completed_path.stat().st_mtime_ns
            resumed = run(path, root, stages=('repeat',))
            self.assertTrue(resumed['study_complete'])
            self.assertEqual(resumed['completed_runs'], 9)
            self.assertEqual(completed_path.stat().st_mtime_ns, stamp)
            statuses = {r['variant']:r['status'] for r in resumed['runs']}
            self.assertEqual(statuses['original16'], 'skipped')
            self.assertEqual(statuses['original32'], 'completed')
            # A completed status cannot conceal a missing matrix artifact.
            (root/'seed_2/repeat/typed_repeat/diagnostics.npz').unlink()
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                package_results(root, directory/'missing_matrix.zip')
            repaired = run(path, root, stages=('repeat',))
            statuses = {r['variant']:r['status'] for r in repaired['runs']}
            self.assertEqual(statuses['typed_repeat'], 'completed')
            self.assertTrue(all(v == 'skipped' for k,v in statuses.items() if k != 'typed_repeat'))
            self.assertTrue(package_results(root, directory/'complete.zip')['complete'])

    def test_resume_reuses_matching_runs_and_rejects_changed_protocol(self):
        with artifact_directory() as directory:
            path, _, _ = write_synthetic_cache(directory)
            root = directory/'repeat'
            first = run(path, root, stages=('repeat',))
            self.assertEqual(first['completed_runs'], 9)
            prediction_path = root/'seed_2/repeat/typed_repeat/predictions.npz'
            stamp = prediction_path.stat().st_mtime_ns
            second = run(path, root, stages=('repeat',))
            self.assertTrue(all(r['status'] == 'skipped' for r in second['runs']))
            self.assertEqual(prediction_path.stat().st_mtime_ns, stamp)
            with self.assertRaisesRegex(ValueError, 'changed'):
                engine.run_experiments(path, root, fit_seeds=(2,), sequences=2,
                                       stages=('repeat',), blend_grid=(0., .5, 1.))
            with self.assertRaisesRegex(ValueError, 'not overwritten'):
                run(path, root, stages=('repeat',), resume=False)


if __name__ == '__main__':
    unittest.main()
