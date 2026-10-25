"""Safe export and frozen incremental use of a support-conditioned estimator."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from engine.safd_metric_runtime import reference_settings, validate_reference_metadata
from utils.frequency_descriptions import dataset_slug


FORMAT = 'support_prototype_estimator_v1'


def _calibration(value):
    expected = {'fused_temperature', 'cov_temperature', 'description_temperature', 'novel_bias'}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError('Estimator calibration must contain all three temperatures and novel_bias.')
    result = {}
    for key, supplied in value.items():
        if isinstance(supplied, bool) or not isinstance(supplied, (float, int)) or not np.isfinite(supplied):
            raise ValueError('Estimator calibration values must be finite real scalars.')
        if key.endswith('temperature') and supplied <= 0:
            raise ValueError('Estimator calibration temperatures must be positive.')
        result[key] = float(supplied)
    return result


def _blend(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Estimator blend must be finite and within [0,1].')
    return float(value)


def _validate_metadata(metadata):
    validate_reference_metadata(metadata)
    if metadata.get('data_source') != 'base_training_only':
        raise ValueError('Prototype estimator must certify base_training_only fitting.')
    if not metadata.get('base_sample_ids_sorted_sha256'):
        raise ValueError('Prototype estimator is missing the base-image identity hash.')
    if not isinstance(metadata.get('feature_dim'), int) or metadata['feature_dim'] < 1:
        raise ValueError('Prototype estimator feature_dim must be positive.')
    if not isinstance(metadata.get('model_config'), dict) or set(metadata['model_config']) != {'hidden', 'max_delta', 'use_competitors'}:
        raise ValueError('Prototype estimator model_config is incomplete.')
    _blend(metadata.get('blend'))
    _calibration(metadata.get('calibration'))


def export_estimator(path, model, metadata, training_report):
    """Write finite NPZ parameters and provenance; never overwrite a fitted artifact."""
    path = Path(path)
    if path.suffix != '.npz':
        raise ValueError('Estimator output must end in .npz.')
    paths = (path, path.with_suffix('.json'), path.with_suffix('.training.json'))
    if any(p.exists() for p in paths):
        raise ValueError('Refusing to overwrite an existing estimator; choose a fresh output.')
    metadata = dict(metadata)
    metadata['format'] = FORMAT
    _validate_metadata(metadata)
    if metadata['model_config'] != model.configuration():
        raise ValueError('Estimator model_config disagrees with the exported architecture.')
    count = sum(p.numel() for p in model.parameters())
    if metadata.get('trainable_parameters') != count:
        raise ValueError('Estimator parameter count disagrees with its exported architecture.')
    arrays = {'model__' + key: value.detach().cpu().numpy() for key, value in model.state_dict().items()}
    if not all(v.dtype.kind in 'fi' and np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('Estimator parameters must be finite real arrays.')
    # Validate all JSON data before creating any artifact.
    json.dumps(metadata, allow_nan=False)
    report_text = json.dumps(training_report, indent=2, ensure_ascii=False, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    metadata['model_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix('.training.json').write_text(report_text, encoding='utf-8')
    path.with_suffix('.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    return metadata


def load_estimator(path, cfg, clip_weight_sha256):
    """Load only numerical NPZ arrays, checking reference and model identity."""
    from models.support_estimator import SupportPrototypeEstimator
    path = Path(path)
    metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if metadata.get('format') != FORMAT:
        raise ValueError('Unsupported prototype estimator format.')
    _validate_metadata(metadata)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if metadata.get('model_sha256') != digest:
        raise ValueError('Prototype estimator checksum disagrees with its sidecar.')
    if dataset_slug(metadata.get('dataset', '')) != dataset_slug(cfg.DATASET.NAME):
        raise ValueError('Prototype estimator belongs to a different dataset.')
    if not clip_weight_sha256 or metadata.get('clip_weight_sha256') != clip_weight_sha256:
        raise ValueError('Prototype estimator CLIP weights differ from this run.')
    if metadata.get('base_class_ids') != list(range(cfg.DATASET.NUM_INIT_CLS)):
        raise ValueError('Prototype estimator does not match the original base-class bank.')
    if metadata.get('reference_settings') != reference_settings(cfg):
        raise ValueError('Prototype estimator BiMC reference settings differ from this run.')
    if cfg.TRAINER.BiMC.PREC != 'fp32':
        raise ValueError('Prototype estimator runtime requires fp32 CLIP encoding.')
    model = SupportPrototypeEstimator(**metadata['model_config'])
    expected = model.state_dict()
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != {'model__' + key for key in expected}:
            raise ValueError('Prototype estimator parameter names do not match the architecture.')
        weights = {}
        for key, template in expected.items():
            array = saved['model__' + key]
            if array.shape != tuple(template.shape) or array.dtype.kind not in 'fi' or not np.isfinite(array).all():
                raise ValueError('Prototype estimator parameter shape/value mismatch: ' + key)
            weights[key] = torch.as_tensor(array).to(template.dtype)
    model.load_state_dict(weights, strict=True)
    count = sum(p.numel() for p in model.parameters())
    if metadata.get('trainable_parameters') != count:
        raise ValueError('Prototype estimator parameter count disagrees with its architecture.')
    model.to(cfg.DEVICE.DEVICE_NAME).eval().requires_grad_(False)
    options = cfg.TRAINER.BiMC.PROTOTYPE_ESTIMATOR
    blend = metadata['blend'] if options.BLEND == -1 else options.BLEND
    return {'model': model, 'metadata': metadata, 'blend': _blend(blend),
            'calibration': _calibration(metadata['calibration']),
            'calibration_enabled': bool(options.CALIBRATION),
            'model_path': str(path.resolve()), 'model_sha256': digest,
            'base_bank_verified': False, 'reference_content_verified': False}


def _verified(state):
    if not state.get('base_bank_verified') or not state.get('reference_content_verified'):
        raise RuntimeError('Verify base training images and reference content before estimator use.')


@torch.no_grad()
def prepare_current(state, cfg, current, past_states):
    """Estimate a current novel class once; previous prototypes remain unchanged."""
    from models.support_estimator import correct_current
    _verified(state)
    reference = current['image_proto']
    if reference.shape[-1] != state['metadata']['feature_dim']:
        raise ValueError('Estimator feature dimension differs from current support features.')
    if int(current['class_index'][0]) == 0 or state['blend'] == 0:
        current['estimated_image_proto'] = reference
        return
    support = current['images_features']
    labels = current['images_targets']
    supports = [support[labels == int(c)] for c in current['class_index']]
    if any(not len(rows) for rows in supports):
        raise ValueError('Every current class needs its own support features.')
    b = cfg.TRAINER.BiMC
    coefficient = b.LAMBDA_T if b.TEXT_CALIBRATION else 0.
    semantic = F.normalize((1 - coefficient) * current['text_features']
                           + coefficient * current['description_proto'], dim=-1)
    past = torch.cat([s['estimated_image_proto'] for s in past_states]) if past_states else None
    current['estimated_image_proto'] = correct_current(
        state['model'], supports, reference, semantic,
        past_prototypes=past, blend=state['blend']).detach()


@torch.no_grad()
def apply_estimator(state, cfg, merged, features, reference_votes):
    """Score frozen estimated prototypes without query-time fitting or views."""
    from engine.prototype_evaluation import score_record, identity_calibration
    _verified(state)
    calibration = state['calibration'] if state['calibration_enabled'] else identity_calibration()
    estimated = merged['estimated_image_proto']
    if calibration == identity_calibration() and torch.equal(estimated, merged['image_proto']):
        return reference_votes
    # Cache only current-session class statistics, never query image features.
    key = (merged['cov_image'].data_ptr(), len(merged['image_proto']))
    if state.get('_session_key') != key:
        state['_session_key'] = key
        state['_session_statistics'] = {
            'num_old': cfg.DATASET.NUM_INIT_CLS,
            'reference_prototypes': merged['image_proto'],
            'covariance': merged['cov_image'],
            'cov_inverse': torch.pinverse(merged['cov_image'].float()),
            **{k: merged[k] for k in ('text_features', 'description_proto', 'description_features', 'description_targets')},
        }
    row = {**state['_session_statistics'], 'query': features, 'reference_votes': reference_votes}
    return score_record(row, cfg, image_prototypes=estimated, calibration=calibration)


def estimator_report(state):
    return {'model_path': state['model_path'], 'model_sha256': state['model_sha256'],
            'blend': state['blend'], 'calibration': state['calibration'],
            'calibration_enabled': state['calibration_enabled'],
            'base_bank_verified': state['base_bank_verified'],
            'reference_content_verified': state['reference_content_verified'],
            'model_config': state['metadata']['model_config'],
            'trainable_parameters_during_base_fitting': state['metadata']['trainable_parameters'],
            'incremental_trainable_parameters': 0,
            'selected_step': state['metadata'].get('selected_step'),
            'fitting_seed': state['metadata'].get('fitting_seed'),
            'selection_protocol_sha256': state['metadata'].get('selection_protocol_sha256'),
            'offline_fit_wall_seconds': state['metadata'].get('fit_wall_seconds'),
            'offline_cache_extraction_wall_seconds': state['metadata'].get('cache_extraction_wall_seconds'),
            'auxiliary_training': state['metadata'].get('auxiliary_training'),
            'training_protocol': state['metadata'].get('training_protocol'),
            'query_auxiliary_encodings': 0}


def checkpoint_state(state):
    if state is None:
        return None
    return {**{k: v for k, v in state.items() if k not in ('model', '_session_key', '_session_statistics')},
            'model_state': state['model'].state_dict()}
