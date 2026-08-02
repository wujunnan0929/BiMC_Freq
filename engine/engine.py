import torch
import torch.nn as nn
import models.clip as clip
from datasets.data_manager import DatasetManager
from torch.nn import functional as F
from tqdm import tqdm
from utils.evaluator import AccuracyEvaluator
from models.bimc import BiMC
import numpy as np
import time
import os
from utils.frequency_analysis import (
    FourierBandStop,
    FrequencyContributionAccumulator,
    band_energy_fractions,
    equal_energy_band_edges,
    pearson_correlation,
    semantic_margins,
    spearman_correlation,
    summed_power_spectrum,
    summarize_frequency_records,
    write_frequency_records,
    write_predictivity_report,
)


class Runner:

    def __init__(self, cfg):
        self.cfg = cfg
        self.data_manager = DatasetManager(cfg,) 
        self.device = cfg.DEVICE.DEVICE_NAME

        self.model = BiMC(cfg, self.data_manager.template, self.device)

        # device
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
            self.is_distributed = True
        else:
            self.is_distributed = False
            

        self.acc_list = []
        self.task_acc_list = []
        self.evaluator = AccuracyEvaluator(self.data_manager.class_index_in_task)

        frequency_cfg = cfg.ANALYSIS.FREQUENCY
        self.frequency_analysis_enabled = frequency_cfg.ENABLED
        self.frequency_analysis_only = frequency_cfg.ANALYSIS_ONLY
        self.frequency_records = []
        self.frequency_text_features = None
        self.frequency_band_energy_fractions = None
        self.frequency_predictivity_reports = []
        if self.frequency_analysis_enabled:
            if frequency_cfg.SAMPLES_PER_CLASS <= 0:
                raise ValueError('ANALYSIS.FREQUENCY.SAMPLES_PER_CLASS must be positive')
            if frequency_cfg.BAND_MODE not in ('fixed', 'equal_energy'):
                raise ValueError(
                    "ANALYSIS.FREQUENCY.BAND_MODE must be 'fixed' or 'equal_energy'"
                )
            if frequency_cfg.ENERGY_SAMPLES_PER_CLASS <= 0:
                raise ValueError(
                    'ANALYSIS.FREQUENCY.ENERGY_SAMPLES_PER_CLASS must be positive'
                )
            predictivity_cfg = frequency_cfg.PREDICTIVITY
            if predictivity_cfg.EVAL_SCOPE not in ('final', 'all'):
                raise ValueError(
                    "ANALYSIS.FREQUENCY.PREDICTIVITY.EVAL_SCOPE must be 'final' or 'all'"
                )
            if predictivity_cfg.BAND not in frequency_cfg.BAND_NAMES:
                raise ValueError(
                    'ANALYSIS.FREQUENCY.PREDICTIVITY.BAND must name a configured band'
                )
            if predictivity_cfg.SOFT_GATE_TEMPERATURE <= 0:
                raise ValueError(
                    'ANALYSIS.FREQUENCY.PREDICTIVITY.SOFT_GATE_TEMPERATURE must be positive'
                )
            if predictivity_cfg.RELIABILITY_EPS <= 0:
                raise ValueError(
                    'ANALYSIS.FREQUENCY.PREDICTIVITY.RELIABILITY_EPS must be positive'
                )
            self.frequency_band_stop = FourierBandStop(
                frequency_cfg.BAND_NAMES,
                frequency_cfg.BAND_EDGES,
                fft_device=frequency_cfg.FFT_DEVICE,
            )
            self.frequency_output_dir = os.path.join(
                frequency_cfg.OUTPUT_DIR,
                '{}_seed{}_{}'.format(
                    cfg.DATASET.NAME.lower(), cfg.SEED, frequency_cfg.BAND_MODE
                ),
            )


    def merge_dicts(self, dict_list):
        result = {}

        keys_to_merge = [
            'description_proto', 
            'description_features', 
            'description_targets',
            'text_features', 
            'text_targets',
            'image_proto', 
            'images_features', 
            'images_targets'
        ]

        for key in keys_to_merge:
            result[key] = torch.cat([d[key] for d in dict_list], dim=0)


        weights = [len(d['class_index']) for d in dict_list]


        cov_keys = [
            'cov_image',
        ]
        cov_sums = {key: torch.zeros_like(dict_list[0][key]) for key in cov_keys}
        weight_sum = sum(weights)

        for i, d in enumerate(dict_list):
            for key in cov_keys:
                cov_sums[key] += d[key] * weights[i]

        for key in cov_keys:
            if weight_sum > 0: 
                result[key] = cov_sums[key] / weight_sum

        return result



    @torch.no_grad()
    def run(self):
        print(f'Start inferencing on all tasks: [0, {self.data_manager.num_tasks - 1}]')
        if self.frequency_analysis_enabled:
            self.prepare_frequency_bands()
        state_dict_list = []
        for i in range(self.data_manager.num_tasks):
            self.model.eval()

            current_class_name = np.array(self.data_manager.class_names)[self.data_manager.class_index_in_task[i]]

            if self.frequency_analysis_enabled:
                self.analyze_frequency_contribution(i)
                predictivity_cfg = self.cfg.ANALYSIS.FREQUENCY.PREDICTIVITY
                should_evaluate_predictivity = (
                    predictivity_cfg.ENABLED
                    and (
                        predictivity_cfg.EVAL_SCOPE == 'all'
                        or i == self.data_manager.num_tasks - 1
                    )
                )
                if should_evaluate_predictivity:
                    self.evaluate_frequency_predictivity(i)
                if self.frequency_analysis_only:
                    continue

            loader = self.data_manager.get_dataloader(i, source='train', mode='test', accumulate_past=False)            



            current_state_dict = self.model.build_task_statistics(current_class_name, loader,
                                                             class_index=self.data_manager.class_index_in_task[i], 
                                                             calibrate_novel_vision_proto=self.cfg.TRAINER.BiMC.VISION_CALIBRATION,)

            state_dict_list.append(current_state_dict)            
            merged_state_dict = self.merge_dicts(state_dict_list)

            start_time = time.time()
            acc = self.inference_task_covariance(i, merged_state_dict)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f'+++++++++++  task {i}, time: {elapsed_time} ++++++++++++++++')

            print(f'=> Task [{i}], Acc: {acc["mean_acc"]:.3f}')
            self.acc_list.append(round(acc["mean_acc"], 3))
            self.task_acc_list.append(acc['task_acc'])

        if self.frequency_analysis_enabled:
            self.write_combined_frequency_analysis()

        if not self.frequency_analysis_only:
            print(f'Final acc:{self.acc_list}')
            print('Task-wise acc:')
            for i, task_acc in enumerate(self.task_acc_list):
                print(f'task {i:2d}, acc:{task_acc}')
        else:
            print('Frequency analysis complete. Results: {}'.format(self.frequency_output_dir))


    @torch.no_grad()
    def prepare_frequency_bands(self):
        frequency_cfg = self.cfg.ANALYSIS.FREQUENCY
        if frequency_cfg.BAND_MODE == 'fixed':
            print(
                'Using fixed radial frequency edges: {}'.format(
                    list(self.frequency_band_stop.band_edges)
                )
            )
            return

        # Dataset construction samples classes with NumPy's global RNG. Restore
        # its state afterwards so the subsequent 5-shot analysis uses the same
        # samples it would have used without this calibration pass.
        numpy_rng_state = np.random.get_state()
        calibration_loader = self.data_manager.get_dataloader(
            0,
            source='train',
            mode='test',
            accumulate_past=False,
            shot_override=frequency_cfg.ENERGY_SAMPLES_PER_CLASS,
        )
        np.random.set_state(numpy_rng_state)

        print(
            'Estimating equal-energy frequency bands from {} base images per class ...'.format(
                frequency_cfg.ENERGY_SAMPLES_PER_CLASS
            )
        )
        power_spectrum = None
        for batch in tqdm(calibration_loader):
            batch_power = summed_power_spectrum(batch['image'])
            if power_spectrum is None:
                power_spectrum = batch_power
            else:
                power_spectrum += batch_power

        estimated_edges = equal_energy_band_edges(
            power_spectrum, len(frequency_cfg.BAND_NAMES)
        )
        self.frequency_band_stop.set_band_edges(estimated_edges)
        self.frequency_band_energy_fractions = band_energy_fractions(
            power_spectrum, estimated_edges
        )
        print('Estimated radial band edges: {}'.format(estimated_edges))
        print(
            'Measured base-set band energy fractions: {}'.format(
                self.frequency_band_energy_fractions
            )
        )


    @torch.no_grad()
    def analyze_frequency_contribution(self, task_id):
        frequency_cfg = self.cfg.ANALYSIS.FREQUENCY
        model = self.model.module if self.is_distributed else self.model

        if frequency_cfg.COMPETITOR_SCOPE == 'all':
            candidate_class_names = np.array(self.data_manager.class_names)
            competitor_scope = 'all dataset classes (offline diagnostic only)'
        elif frequency_cfg.COMPETITOR_SCOPE == 'accumulated':
            accumulated_class_ids = np.concatenate(
                self.data_manager.class_index_in_task[:task_id + 1]
            )
            candidate_class_names = np.array(self.data_manager.class_names)[accumulated_class_ids]
            competitor_scope = 'classes accumulated through task {}'.format(task_id)
        else:
            raise ValueError(
                "ANALYSIS.FREQUENCY.COMPETITOR_SCOPE must be 'all' or 'accumulated'"
            )
        if frequency_cfg.COMPETITOR_SCOPE == 'all' and self.frequency_text_features is not None:
            text_features = self.frequency_text_features
        else:
            text_features, _ = model.inference_text_feature(
                candidate_class_names, self.data_manager.template, cls_begin_index=0
            )
            if frequency_cfg.COMPETITOR_SCOPE == 'all':
                self.frequency_text_features = text_features

        loader = self.data_manager.get_dataloader(
            task_id,
            source='train',
            mode='test',
            accumulate_past=False,
            shot_override=frequency_cfg.SAMPLES_PER_CLASS,
        )
        accumulator = FrequencyContributionAccumulator(
            frequency_cfg.BAND_NAMES,
            weight_temperature=frequency_cfg.WEIGHT_TEMPERATURE,
        )

        print('Analyzing frequency contribution for task {} ...'.format(task_id))
        for batch in tqdm(loader):
            images, labels = self.parse_batch(batch)
            full_features = F.normalize(model.extract_img_feature(images), dim=-1)
            full_margins = semantic_margins(full_features, text_features, labels)

            contributions = {}
            counterfactual_batches = self.frequency_band_stop.remove_all(images)
            for band_name, counterfactual_images in zip(
                frequency_cfg.BAND_NAMES, counterfactual_batches
            ):
                counterfactual_features = F.normalize(
                    model.extract_img_feature(counterfactual_images), dim=-1
                )
                counterfactual_margins = semantic_margins(
                    counterfactual_features, text_features, labels
                )
                contributions[band_name] = full_margins - counterfactual_margins
            accumulator.update(labels, contributions)

        session_records = accumulator.records(
            self.data_manager.class_names, task_id=task_id
        )
        self.frequency_records.extend(session_records)
        session_summary = summarize_frequency_records(
            session_records, frequency_cfg.BAND_NAMES
        )
        metadata = self._frequency_metadata(competitor_scope=competitor_scope)
        json_path, csv_path = write_frequency_records(
            self.frequency_output_dir,
            'session_{:02d}_frequency_contribution'.format(task_id),
            session_records,
            session_summary,
            metadata,
        )
        print('Frequency summary for task {}: {}'.format(task_id, session_summary))
        print('Saved frequency records: {}, {}'.format(json_path, csv_path))


    def _frequency_metadata(self, competitor_scope):
        frequency_cfg = self.cfg.ANALYSIS.FREQUENCY
        return {
            'dataset': self.cfg.DATASET.NAME,
            'seed': int(self.cfg.SEED),
            'method': 'Fourier radial band-stop semantic-margin contribution',
            'band_names': list(frequency_cfg.BAND_NAMES),
            'band_mode': frequency_cfg.BAND_MODE,
            'configured_band_edges': [
                float(value) for value in frequency_cfg.BAND_EDGES
            ],
            'resolved_band_edges': list(self.frequency_band_stop.band_edges),
            'base_band_energy_fractions': self.frequency_band_energy_fractions,
            'energy_samples_per_base_class': int(
                frequency_cfg.ENERGY_SAMPLES_PER_CLASS
            ),
            'samples_per_class': int(frequency_cfg.SAMPLES_PER_CLASS),
            'weight_temperature': float(frequency_cfg.WEIGHT_TEMPERATURE),
            'competitor_scope': competitor_scope,
            'configured_competitor_scope': frequency_cfg.COMPETITOR_SCOPE,
            'configured_fft_device': frequency_cfg.FFT_DEVICE,
            'cuda_fft_fell_back_to_cpu': self.frequency_band_stop._cuda_fft_failed,
            'dc_component_preserved': True,
        }


    def write_combined_frequency_analysis(self):
        frequency_cfg = self.cfg.ANALYSIS.FREQUENCY
        summary = summarize_frequency_records(
            self.frequency_records, frequency_cfg.BAND_NAMES
        )
        summary['by_session_type'] = {
            session_type: summarize_frequency_records(
                [
                    record for record in self.frequency_records
                    if record['session_type'] == session_type
                ],
                frequency_cfg.BAND_NAMES,
            )
            for session_type in ('base', 'incremental')
        }
        metadata = self._frequency_metadata(
            competitor_scope=(
                'all dataset classes (offline diagnostic only)'
                if frequency_cfg.COMPETITOR_SCOPE == 'all'
                else 'session-specific accumulated classes'
            )
        )
        json_path, csv_path = write_frequency_records(
            self.frequency_output_dir,
            'all_classes_frequency_contribution',
            self.frequency_records,
            summary,
            metadata,
        )
        print('Combined frequency summary: {}'.format(summary))
        print('Saved combined frequency records: {}, {}'.format(json_path, csv_path))


    @torch.no_grad()
    def evaluate_frequency_predictivity(self, task_id):
        frequency_cfg = self.cfg.ANALYSIS.FREQUENCY
        predictivity_cfg = frequency_cfg.PREDICTIVITY
        model = self.model.module if self.is_distributed else self.model
        band_name = predictivity_cfg.BAND
        band_index = list(frequency_cfg.BAND_NAMES).index(band_name)

        accumulated_class_ids = np.concatenate(
            self.data_manager.class_index_in_task[:task_id + 1]
        )
        num_classes = len(accumulated_class_ids)
        class_names = np.array(self.data_manager.class_names)[accumulated_class_ids]
        text_features, _ = model.inference_text_feature(
            class_names, self.data_manager.template, cls_begin_index=0
        )

        support_records = {
            record['class_id']: record
            for record in self.frequency_records
            if record['class_id'] < num_classes
        }
        missing_classes = [
            class_id for class_id in range(num_classes)
            if class_id not in support_records
        ]
        if missing_classes:
            raise RuntimeError(
                'Missing support frequency records for classes {}'.format(missing_classes)
            )

        support_mean = torch.tensor([
            support_records[class_id]['bands'][band_name]['mean']
            for class_id in range(num_classes)
        ], dtype=torch.float32)
        support_std = torch.tensor([
            support_records[class_id]['bands'][band_name]['std']
            for class_id in range(num_classes)
        ], dtype=torch.float32)
        support_snr = support_mean / (
            support_std + predictivity_cfg.RELIABILITY_EPS
        )
        predicted_hard_gate = (support_mean >= 0).float()
        predicted_soft_gate = torch.sigmoid(
            support_mean / predictivity_cfg.SOFT_GATE_TEMPERATURE
        )

        test_loader = self.data_manager.get_dataloader(
            task_id, source='test', mode='test'
        )
        full_logits_list = []
        removed_logits_list = []
        labels_list = []
        test_contribution_list = []

        print(
            'Evaluating support-to-test frequency predictivity for task {} ...'.format(
                task_id
            )
        )
        for batch in tqdm(test_loader):
            images, labels = self.parse_batch(batch)
            full_features = F.normalize(model.extract_img_feature(images), dim=-1)
            removed_images = self.frequency_band_stop.remove(images, band_index)
            removed_features = F.normalize(
                model.extract_img_feature(removed_images), dim=-1
            )

            full_logits = (full_features.float() @ text_features.float().T)
            removed_logits = (removed_features.float() @ text_features.float().T)
            full_margins = semantic_margins(full_features, text_features, labels)
            removed_margins = semantic_margins(
                removed_features, text_features, labels
            )

            full_logits_list.append(full_logits.cpu())
            removed_logits_list.append(removed_logits.cpu())
            labels_list.append(labels.cpu())
            test_contribution_list.append((full_margins - removed_margins).cpu())

        full_logits = torch.cat(full_logits_list, dim=0)
        removed_logits = torch.cat(removed_logits_list, dim=0)
        labels = torch.cat(labels_list, dim=0).long()
        test_contributions = torch.cat(test_contribution_list, dim=0).float()
        full_predictions = full_logits.argmax(dim=1)
        removed_predictions = removed_logits.argmax(dim=1)

        class_records = []
        test_mean = torch.zeros(num_classes, dtype=torch.float32)
        full_class_accuracy = torch.zeros(num_classes, dtype=torch.float32)
        removed_class_accuracy = torch.zeros(num_classes, dtype=torch.float32)
        class_sample_counts = torch.zeros(num_classes, dtype=torch.long)
        for class_id in range(num_classes):
            class_mask = labels == class_id
            class_sample_counts[class_id] = int(class_mask.sum())
            test_mean[class_id] = test_contributions[class_mask].mean()
            full_class_accuracy[class_id] = (
                full_predictions[class_mask] == class_id
            ).float().mean()
            removed_class_accuracy[class_id] = (
                removed_predictions[class_mask] == class_id
            ).float().mean()

        accuracy_delta = full_class_accuracy - removed_class_accuracy
        oracle_margin_gate = (test_mean >= 0).float()
        oracle_accuracy_gate = (accuracy_delta >= 0).float()

        def gated_logits(gate):
            gate = gate.view(1, -1)
            return gate * full_logits + (1.0 - gate) * removed_logits

        def accuracy(logits):
            return float((logits.argmax(dim=1) == labels).float().mean())

        full_accuracy = accuracy(full_logits)
        removed_accuracy = accuracy(removed_logits)
        equal_mix_accuracy = accuracy(0.5 * (full_logits + removed_logits))
        predicted_hard_accuracy = accuracy(gated_logits(predicted_hard_gate))
        predicted_soft_accuracy = accuracy(gated_logits(predicted_soft_gate))
        oracle_margin_accuracy = accuracy(gated_logits(oracle_margin_gate))
        oracle_accuracy = accuracy(gated_logits(oracle_accuracy_gate))

        predicted_harmful = support_mean < 0

        def harmful_detection(actual_harmful):
            true_harmful = predicted_harmful & actual_harmful
            return {
                'predicted_harmful_classes': int(predicted_harmful.sum()),
                'actual_harmful_classes': int(actual_harmful.sum()),
                'true_harmful_classes': int(true_harmful.sum()),
                'precision': (
                    float(true_harmful.sum()) / float(predicted_harmful.sum())
                    if int(predicted_harmful.sum()) > 0 else None
                ),
                'recall': (
                    float(true_harmful.sum()) / float(actual_harmful.sum())
                    if int(actual_harmful.sum()) > 0 else None
                ),
                'sign_agreement': float(
                    ((support_mean >= 0) == (~actual_harmful)).float().mean()
                ),
            }

        for class_id in range(num_classes):
            class_records.append({
                'class_id': class_id,
                'class_name': str(self.data_manager.class_names[class_id]),
                'num_test_samples': int(class_sample_counts[class_id]),
                'support_contribution_mean': float(support_mean[class_id]),
                'support_contribution_std': float(support_std[class_id]),
                'support_reliability_snr': float(support_snr[class_id]),
                'test_contribution_mean': float(test_mean[class_id]),
                'full_accuracy': float(full_class_accuracy[class_id]),
                'high_removed_accuracy': float(removed_class_accuracy[class_id]),
                'accuracy_delta': float(accuracy_delta[class_id]),
                'predicted_hard_use_full': int(predicted_hard_gate[class_id]),
                'predicted_soft_use_full_weight': float(predicted_soft_gate[class_id]),
                'oracle_margin_use_full': int(oracle_margin_gate[class_id]),
                'oracle_accuracy_use_full': int(oracle_accuracy_gate[class_id]),
            })

        summary = {
            'task_id': int(task_id),
            'num_classes': num_classes,
            'num_test_samples': int(labels.numel()),
            'band': band_name,
            'accuracy': {
                'full': full_accuracy,
                'band_removed': removed_accuracy,
                'equal_mix': equal_mix_accuracy,
                'predicted_hard_gate': predicted_hard_accuracy,
                'predicted_soft_gate': predicted_soft_accuracy,
                'oracle_test_margin_gate': oracle_margin_accuracy,
                'oracle_test_accuracy_gate': oracle_accuracy,
            },
            'support_to_test_correlation': {
                'mean_vs_test_margin_pearson': pearson_correlation(
                    support_mean, test_mean
                ),
                'mean_vs_test_margin_spearman': spearman_correlation(
                    support_mean, test_mean
                ),
                'snr_vs_test_margin_pearson': pearson_correlation(
                    support_snr, test_mean
                ),
                'snr_vs_test_margin_spearman': spearman_correlation(
                    support_snr, test_mean
                ),
                'mean_vs_accuracy_delta_pearson': pearson_correlation(
                    support_mean, accuracy_delta
                ),
                'mean_vs_accuracy_delta_spearman': spearman_correlation(
                    support_mean, accuracy_delta
                ),
            },
            'harmful_band_detection': {
                'against_test_margin': harmful_detection(test_mean < 0),
                'against_test_accuracy': harmful_detection(accuracy_delta < 0),
            },
        }
        metadata = self._frequency_metadata(
            competitor_scope='classes accumulated through task {}'.format(task_id)
        )
        metadata['predictivity'] = {
            'strict_session_time': frequency_cfg.COMPETITOR_SCOPE == 'accumulated',
            'soft_gate_temperature': float(
                predictivity_cfg.SOFT_GATE_TEMPERATURE
            ),
            'reliability_eps': float(predictivity_cfg.RELIABILITY_EPS),
            'gate_interpretation': '1 uses full image; 0 uses band-removed image',
        }
        stem = 'session_{:02d}_frequency_predictivity'.format(task_id)
        json_path, csv_path = write_predictivity_report(
            self.frequency_output_dir,
            stem,
            class_records,
            summary,
            metadata,
        )
        self.frequency_predictivity_reports.append(summary)
        print('Frequency predictivity summary: {}'.format(summary))
        print('Saved frequency predictivity report: {}, {}'.format(json_path, csv_path))
    

    @torch.no_grad()
    def inference_task_covariance(self, task_id, state_dict):

        beta = self.cfg.DATASET.BETA

        image_proto = state_dict['image_proto']
        cov_image = state_dict['cov_image']
        text_features = state_dict['text_features']
        description_proto = state_dict['description_proto']
        description_features = state_dict['description_features']
        description_targets = state_dict['description_targets']

        num_base_class = len(self.data_manager.class_index_in_task[0])
        num_accumulated_class = max(self.data_manager.class_index_in_task[task_id]) + 1
        
        test_loader = self.data_manager.get_dataloader(task_id, source='test', mode='test')
        all_logits = []
        all_targets = []

        for i, batch in enumerate(tqdm(test_loader)):
            data, targets = self.parse_batch(batch)
            logits = self.model.forward_ours(data, num_accumulated_class, num_base_class,
                                                   image_proto, 
                                                   cov_image,
                                                   description_proto,
                                                   description_features, 
                                                   description_targets,
                                                   text_features,
                                                   beta=beta)

            all_logits.append(logits)
            all_targets.append(targets)

        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        eval_acc = self.evaluator.calc_accuracy(all_logits, all_targets, task_id) 
        print(f"Test acc mean: {eval_acc['mean_acc']}, task-wise acc: {eval_acc['task_acc']}")
        return eval_acc
    

    def parse_batch(self, batch):
        data = batch['image']
        targets = batch['label']
        data = data.to(self.device)
        targets = targets.to(self.device)
        return data, targets
