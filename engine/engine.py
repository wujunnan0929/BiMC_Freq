import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from datasets.data_manager import DatasetManager
from engine.residual_training import initialize_residual, make_reference_scorer
from models.bimc import BiMC
from utils.consensus_metrics import prediction_diagnostics
from utils.incremental_metrics import (
    compute_session_metrics, compute_forgetting, summarize_sessions,
)


def _cpu_state(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _cpu_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cpu_state(item) for item in value]
    return value


def _tensor_bytes(values):
    """Logical retained tensor bytes, excluding shared CLIP and temporary caches."""
    seen = set()

    def count(value):
        if torch.is_tensor(value):
            key = (str(value.device), value.data_ptr(), value.numel())
            if key in seen:
                return 0
            seen.add(key)
            return value.numel() * value.element_size()
        if isinstance(value, dict):
            return sum(count(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return sum(count(item) for item in value)
        return 0

    return count(values)


class Runner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_manager = DatasetManager(cfg)
        self.device = cfg.DEVICE.DEVICE_NAME
        # Custom cached-feature/session methods are not parallelized by DataParallel.
        self.model = BiMC(cfg, self.data_manager.template, self.device)
        self.acc_list = []
        self.task_acc_list = []
        self.sessions = []
        self.dictionary_report = None
        self.consensus_report = None
        self.output_dir = Path(cfg.OUTPUT_DIR) if cfg.OUTPUT_DIR else None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.data_manager.save_support_manifest(self.output_dir / 'support.json')
            (self.output_dir / 'config.yaml').write_text(cfg.dump(), encoding='utf-8')

    def merge_dicts(self, dict_list):
        # Individual support features must not become cross-session replay.
        keys = [
            'description_proto', 'description_features', 'description_targets',
            'text_features', 'text_targets', 'image_proto', 'raw_image_mean',
        ]
        if self.model.frequency_fusion_enabled:
            keys.extend([
                'frequency_image_proto', 'frequency_prompt_proto',
                'frequency_description_proto', 'frequency_semantic_proto',
                'frequency_calibrated_proto', 'frequency_uncertainty',
                'frequency_sample_counts', 'frequency_band_weights',
                'frequency_semantic_gates', 'frequency_alignment',
                'frequency_class_alpha', 'frequency_reliability',
                'raw_frequency_mean',
            ])
        if self.model.consensus_enabled:
            keys.extend(['raw_frequency_mean', 'frequency_consensus_text_proto'])
        result = {key: torch.cat([state[key] for state in dict_list]) for key in keys}
        weights = [len(state['class_index']) for state in dict_list]
        covariance = torch.zeros_like(dict_list[0]['cov_image'])
        for weight, state in zip(weights, dict_list):
            covariance += weight * state['cov_image']
        result['cov_image'] = covariance / sum(weights)
        result['class_index'] = [int(c) for state in dict_list for c in state['class_index']]
        return result

    def _fit_incremental(self, task_id, current, merged):
        settings = self.cfg.TRAINER.BiMC.RESIDUAL
        class_ids = torch.tensor(merged['class_index'], device=self.device)
        new_ids = torch.tensor(current['class_index'], device=self.device)
        if not torch.equal(class_ids, torch.arange(len(class_ids), device=self.device)):
            raise ValueError('The BiMC reference currently requires contiguous global class ids.')
        old_count = len(class_ids) - len(new_ids)
        scorer = make_reference_scorer(
            self.model, self.cfg, merged, len(self.data_manager.class_index_in_task[0])
        )
        features = current['images_features']
        support_scores = scorer(features, current.get('frequency_features'))
        anchors = merged['raw_image_mean'][:old_count]
        anchor_frequency = merged.get('raw_frequency_mean')
        if anchor_frequency is not None:
            anchor_frequency = anchor_frequency[:old_count]
        anchor_scores = scorer(anchors, anchor_frequency)
        return self.model.residual_head.fit_session(
            features, current['images_targets'], support_scores,
            class_ids, new_ids,
            anchor_features=anchors, anchor_reference_scores=anchor_scores,
            anchor_labels=class_ids[:old_count],
            steps=settings.TRAIN_STEPS, lr=settings.LR,
            optimizer=settings.OPTIMIZER,
            l2=settings.L2_WEIGHT, old_margin=settings.OLD_MARGIN,
            old_weight=settings.OLD_LOSS_WEIGHT, grad_clip=settings.GRAD_CLIP,
            temperature=settings.TEMPERATURE, seed=int(self.cfg.SEED) + task_id,
        )

    def _write_results(self, status):
        if self.output_dir is None:
            return
        payload = {
            'schema_version': 1, 'status': status,
            'dataset': self.cfg.DATASET.NAME, 'seed': int(self.cfg.SEED),
            'support_seed': int(self.data_manager.support_seed),
            'memory_protocol': 'class_visual_means_no_individual_feature_replay',
            'residual_input': 'original_clip_features',
            'dictionary': self.dictionary_report, 'sessions': self.sessions,
            'consensus': self.consensus_report,
            'image_encoding_counts': dict(self.model.image_encoding_counts),
            'summary': summarize_sessions(self.sessions) if self.sessions else None,
        }
        temporary = self.output_dir / 'metrics.json.tmp'
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
            encoding='utf-8',
        )
        temporary.replace(self.output_dir / 'metrics.json')

    def _save_checkpoint(self, task_id, states):
        if self.output_dir is None or not self.cfg.TRAINER.BiMC.RESIDUAL.SAVE_CHECKPOINT:
            return
        head, router = self.model.residual_head, self.model.frequency_router
        payload = {
            'schema_version': 1, 'session': task_id,
            'config_yaml': self.cfg.dump(),
            'clip_backbone': self.cfg.MODEL.BACKBONE.NAME,
            'class_groups': [group.tolist() for group in self.data_manager.class_index_in_task],
            'statistics': _cpu_state(states),
            'base_vision_prototype': _cpu_state(self.model.base_vision_prototype),
            'base_frequency_vision_prototype': _cpu_state(
                getattr(self.model, 'base_frequency_vision_prototype', None)
            ),
            'residual_shape': ({'feature_dim': head.dictionary.shape[0],
                                'rank': head.dictionary.shape[1],
                                'num_classes': head.codes.shape[0]} if head else None),
            'residual_state': _cpu_state(head.state_dict()) if head else None,
            'router_state': _cpu_state(router.state_dict()) if router else None,
            'consensus_state': _cpu_state(self.model.consensus_state),
            'consensus_calibration': self.consensus_report,
            'metrics': self.sessions,
        }
        # Boundary snapshot. Pretrained CLIP is identified, not duplicated.
        temporary = self.output_dir / 'checkpoint.pt.tmp'
        torch.save(payload, temporary)
        temporary.replace(self.output_dir / 'checkpoint.pt')

    def run(self):
        print(f'Start inferencing on all tasks: [0, {self.data_manager.num_tasks - 1}]')
        states = []
        if (self.cfg.TRAINER.BiMC.RESIDUAL.BASE_ONLY
                and not self.cfg.TRAINER.BiMC.RESIDUAL.ENABLED):
            raise ValueError('BASE_ONLY requires residual dictionary learning enabled.')
        self._write_results('running')
        for task_id in range(self.data_manager.num_tasks):
            self.model.eval()
            classes = self.data_manager.class_index_in_task[task_id]
            names = np.array(self.data_manager.class_names)[classes]
            loader = self.data_manager.get_dataloader(
                task_id, source='train', mode='test', accumulate_past=False
            )
            start = time.perf_counter()
            encoding_start = dict(self.model.image_encoding_counts)
            current = self.model.build_task_statistics(
                names, loader, class_index=classes,
                calibrate_novel_vision_proto=self.cfg.TRAINER.BiMC.VISION_CALIBRATION,
            )
            if self.model.frequency_router is not None:
                self.model.frequency_router.eval().requires_grad_(False)
            if task_id == 0 and self.model.consensus_enabled:
                from engine.consensus_calibration import initialize_consensus
                self.consensus_report = initialize_consensus(self.model, self.cfg, current)
                if self.output_dir is not None:
                    (self.output_dir / 'consensus_calibration.json').write_text(
                        json.dumps(self.consensus_report, indent=2, allow_nan=False),
                        encoding='utf-8',
                    )
                print('Frequency consensus calibration:', {
                    key: self.consensus_report[key] for key in (
                        'selected_lambda', 'scales', 'active_sources',
                        'reference_accuracy', 'top2_recall', 'candidate_class_counts',
                    )
                })
            states.append(current)
            merged = self.merge_dicts(states)
            fit_report = None
            if self.cfg.TRAINER.BiMC.RESIDUAL.ENABLED:
                if task_id == 0:
                    self.dictionary_report = initialize_residual(self.model, self.cfg, current)
                    print('Residual dictionary:', self.dictionary_report)
                else:
                    fit_report = self._fit_incremental(task_id, current, merged)
                    print('Residual fitting:', fit_report)
            if str(self.device).startswith('cuda'):
                torch.cuda.synchronize()
            fit_seconds = time.perf_counter() - start
            support_encodings = {
                key: self.model.image_encoding_counts[key] - encoding_start[key]
                for key in encoding_start
            }

            # Release sample-level image caches before proceeding to evaluation.
            for key in ('images_features', 'images_targets', 'frequency_features',
                        'frequency_description_candidates'):
                current.pop(key, None)
            if self.cfg.TRAINER.BiMC.RESIDUAL.BASE_ONLY:
                self._save_checkpoint(task_id, states)
                self._write_results('base_validation_completed')
                print('Base-only validation complete; no benchmark test images were evaluated.')
                return []
            start = time.perf_counter()
            metrics = self.inference_task_covariance(task_id, merged)
            if str(self.device).startswith('cuda'):
                torch.cuda.synchronize()
            metrics['eval_seconds'] = time.perf_counter() - start
            metrics['fit_seconds'] = fit_seconds
            metrics['support_image_encodings'] = support_encodings
            metrics['forgetting'] = compute_forgetting(metrics['task_acc'], self.sessions)
            metrics['fit'] = fit_report
            head_state = self.model.residual_head.state_dict() if self.model.residual_head else {}
            metrics['head_parameter_count'] = (
                self.model.residual_head.dictionary.numel()
                + self.model.residual_head.codes.numel()
                if self.model.residual_head else 0
            )
            router_state = (self.model.frequency_router.state_dict()
                            if self.model.frequency_router is not None else {})
            metrics['retained_tensor_bytes'] = _tensor_bytes([
                states, head_state, router_state, self.model.base_vision_prototype,
                getattr(self.model, 'base_frequency_vision_prototype', None),
                self.model.consensus_state,
            ])
            self.sessions.append(metrics)
            self.acc_list.append(round(metrics['accuracy'], 3))
            self.task_acc_list.append(metrics['task_acc'])
            print(f'=> Task [{task_id}], Acc: {metrics["accuracy"]:.3f}, '
                  f'reference: {metrics["reference_accuracy"]:.3f}')
            self._save_checkpoint(task_id, states)
            self._write_results('running')
        self._write_results('completed')
        print(f'Final acc:{self.acc_list}')
        print('Task-wise acc:')
        for task_id, accuracy in enumerate(self.task_acc_list):
            print(f'task {task_id:2d}, acc:{accuracy}')
        return self.sessions

    @torch.no_grad()
    def inference_task_covariance(self, task_id, state_dict):
        scorer = make_reference_scorer(
            self.model, self.cfg, state_dict,
            len(self.data_manager.class_index_in_task[0]),
        )
        loader = self.data_manager.get_dataloader(task_id, source='test', mode='test')
        scores, references, targets = [], [], []
        eligible_rows, encoded_rows, evidence_rows, sample_ids = [], [], [], []
        encoding_start = dict(self.model.image_encoding_counts)
        exact_reference = True
        for batch in tqdm(loader):
            images, labels = self.parse_batch(batch)
            features = self.model.extract_img_feature(images)
            frequency_features = (
                self.model.extract_frequency_img_feature(images)
                if self.model.frequency_fusion_enabled else None
            )
            reference = scorer(features, frequency_features)
            if self.model.consensus_enabled:
                output, detail = self.model.apply_frequency_consensus(
                    images, features, reference, state_dict['raw_frequency_mean'],
                    state_dict['frequency_consensus_text_proto'],
                )
                eligible_rows.append(detail['eligible'].cpu())
                encoded_rows.append(detail['encoded'].cpu())
                evidence_rows.append(detail['evidence'].cpu())
            else:
                output = self.model.apply_incremental_residual(features, reference)
            exact_reference = exact_reference and torch.equal(output, reference)
            identifiers = batch.get('global_index', batch.get('sample_id'))
            if identifiers is None:
                offset = sum(len(item) for item in targets)
                identifiers = torch.arange(offset, offset + len(labels))
            sample_ids.append(torch.as_tensor(identifiers).cpu())
            scores.append(output.cpu())
            references.append(reference.cpu())
            targets.append(labels.cpu())
        scores = torch.cat(scores).numpy()
        references = torch.cat(references).numpy()
        targets = torch.cat(targets).numpy()
        metrics = compute_session_metrics(
            scores, targets, task_id, self.data_manager.class_index_in_task
        )
        metrics['reference_accuracy'] = float(100 * np.mean(references.argmax(1) == targets))
        metrics['prediction_change_rate'] = float(100 * np.mean(scores.argmax(1) != references.argmax(1)))
        eligible = torch.cat(eligible_rows).numpy() if eligible_rows else None
        encoded = torch.cat(encoded_rows).numpy() if encoded_rows else None
        metrics['prediction_diagnostics'] = prediction_diagnostics(
            references, scores, targets, task_id, self.data_manager.class_index_in_task,
            eligible=eligible, encoded=encoded,
        )
        metrics['reference_scores_exact_equal'] = exact_reference
        metrics['query_original_encodings'] = (
            self.model.image_encoding_counts['original'] - encoding_start['original']
        )
        metrics['query_auxiliary_encodings'] = (
            self.model.image_encoding_counts['auxiliary'] - encoding_start['auxiliary']
        )
        if self.output_dir is not None and self.cfg.TRAINER.BiMC.CONSENSUS.SAVE_PREDICTIONS:
            first = references.argmax(1)
            remaining = references.copy()
            remaining[np.arange(len(first)), first] = -np.inf
            second = remaining.argmax(1)
            pair_scores = references[np.arange(len(first))[:, None],
                                     np.stack((first, second), axis=1)]
            payload = {
                'sample_id': torch.cat(sample_ids).numpy(), 'target': targets,
                'reference_prediction': first, 'second_candidate': second,
                'prediction': scores.argmax(1), 'reference_pair_scores': pair_scores,
            }
            if eligible is not None:
                payload.update(eligible=eligible, encoded=encoded,
                               evidence=torch.cat(evidence_rows).numpy())
            np.savez_compressed(self.output_dir / f'predictions_session_{task_id:02d}.npz',
                                **payload)
        return metrics

    def parse_batch(self, batch):
        return batch['image'].to(self.device), batch['label'].to(self.device)
