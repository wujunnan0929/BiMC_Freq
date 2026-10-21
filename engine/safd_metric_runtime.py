"""Load a frozen base-fitted SAFD metric without pickle or test-time fitting."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from models.safd_metric import prepare_residual_classifier, apply_metric_residual
from utils.frequency_descriptions import dataset_slug


_INPUT_PROVENANCE_KEYS = ('dataset', 'train_size', 'train_targets_sha256',
                          'train_samples_sha256')


def validate_reference_metadata(metadata):
    """Reject incomplete reference provenance, including older metric artifacts."""
    if metadata.get('precision') != 'fp32':
        raise ValueError('Frozen metric precision must certify fp32 reference encoding.')
    for key in ('ordinary_description_sha256',):
        value = metadata.get(key)
        if (not isinstance(value, str) or len(value) != 64
                or any(c not in '0123456789abcdef' for c in value)):
            raise ValueError(f'Frozen metric is missing valid {key} reference provenance.')
    for key in ('text_template', 'base_class_names'):
        value = metadata.get(key)
        if not isinstance(value, list) or not value or any(not isinstance(v, str) or not v for v in value):
            raise ValueError(f'Frozen metric is missing valid {key} reference provenance.')
    if not isinstance(metadata.get('support_transform'), str) or not metadata['support_transform']:
        raise ValueError('Frozen metric is missing support_transform reference provenance.')
    provenance = metadata.get('input_provenance')
    if not isinstance(provenance, dict) or any(k not in provenance for k in _INPUT_PROVENANCE_KEYS):
        raise ValueError('Frozen metric is missing input_provenance reference provenance.')
    if not isinstance(provenance['dataset'], str) or not provenance['dataset']:
        raise ValueError('Frozen metric input_provenance needs a dataset.')
    if (isinstance(provenance['train_size'], bool)
            or not isinstance(provenance['train_size'], int) or provenance['train_size'] <= 0):
        raise ValueError('Frozen metric input_provenance needs a positive train_size.')
    for key in ('train_targets_sha256', 'train_samples_sha256'):
        value = provenance[key]
        if (not isinstance(value, str) or len(value) != 64
                or any(c not in '0123456789abcdef' for c in value)):
            raise ValueError(f'Frozen metric input_provenance needs valid {key}.')


def sample_identity_hash(sample_ids):
    values = [int(value) for value in sample_ids]
    if len(set(values)) != len(values):
        raise ValueError('Base sample identities must be unique.')
    return hashlib.sha256(json.dumps(sorted(values)).encode('utf-8')).hexdigest()


def reference_settings(cfg):
    b = cfg.TRAINER.BiMC
    return {'beta': float(cfg.DATASET.BETA),
            'ensemble_alpha': float(cfg.DATASET.ENSEMBLE_ALPHA),
            'vision_calibration': bool(b.VISION_CALIBRATION),
            'lambda_i': float(b.LAMBDA_I), 'tau': float(b.TAU),
            'text_calibration': bool(b.TEXT_CALIBRATION),
            'lambda_t': float(b.LAMBDA_T), 'using_ensemble': bool(b.USING_ENSEMBLE),
            'gamma_base': float(b.GAMMA_BASE), 'gamma_inc': float(b.GAMMA_INC)}


def load_frozen_metric(path, cfg, clip_weight_sha256, gain_override=-1.):
    """Validate provenance; the caller separately verifies the actual base bank."""
    path = Path(path)
    metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if metadata.get('data_source') != 'base_training_only':
        raise ValueError('Frozen metric must certify base_training_only fitting and selection.')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if metadata.get('model_sha256') != digest:
        raise ValueError('Frozen metric checksum disagrees with its JSON sidecar.')
    if dataset_slug(metadata.get('dataset', '')) != dataset_slug(cfg.DATASET.NAME):
        raise ValueError('Frozen metric belongs to another dataset.')
    if not clip_weight_sha256 or metadata.get('clip_weight_sha256') != clip_weight_sha256:
        raise ValueError('Frozen metric CLIP weights differ from this run.')
    if metadata.get('base_class_ids') != list(range(cfg.DATASET.NUM_INIT_CLS)):
        raise ValueError('Frozen metric does not match the original base-class bank.')
    if metadata.get('reference_settings') != reference_settings(cfg):
        raise ValueError('Frozen metric was selected with different BiMC reference settings.')
    if not metadata.get('base_sample_ids_sorted_sha256'):
        raise ValueError('Frozen metric is missing its base-image identity hash.')
    validate_reference_metadata(metadata)
    if cfg.TRAINER.BiMC.PREC != 'fp32':
        raise ValueError('Frozen metric runtime requires fp32 CLIP reference encoding.')
    with np.load(path, allow_pickle=False) as saved:
        matrix = np.asarray(saved['metric'])
    dimension = metadata.get('feature_dim')
    if (matrix.ndim != 2 or matrix.shape != (dimension, dimension)
            or matrix.dtype.kind not in 'fi' or not np.isfinite(matrix).all()):
        raise ValueError('Frozen metric must be a finite square matrix matching feature_dim.')
    tensor = torch.as_tensor(matrix, dtype=torch.float64)
    if not torch.allclose(tensor, tensor.T, rtol=1e-6, atol=1e-7):
        raise ValueError('Frozen metric must be symmetric.')
    tensor = (tensor + tensor.T) / 2
    if torch.linalg.eigvalsh(tensor).min() <= 0:
        raise ValueError('Frozen metric must be positive definite.')
    gain = metadata.get('gain') if gain_override == -1 else gain_override
    if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not np.isfinite(gain) or gain < 0:
        raise ValueError('Frozen metric gain must be finite and nonnegative.')
    return {'metric': tensor, 'gain': float(gain), 'metadata': metadata,
            'model_path': str(path.resolve()), 'model_sha256': digest,
            'base_bank_verified': False, 'reference_content_verified': False}


def verify_reference_content(state, cfg, manager):
    """Verify the reference that produced selection votes before any query read."""
    state['reference_content_verified'] = False
    metadata = state['metadata']
    validate_reference_metadata(metadata)
    if cfg.TRAINER.BiMC.PREC != 'fp32':
        raise ValueError('Frozen metric runtime requires fp32 CLIP reference encoding.')
    path = Path(cfg.DATASET.GPT_PATH)
    if not path.is_file():
        raise ValueError(f'Ordinary description file is unavailable: {path}')
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['ordinary_description_sha256']:
        raise ValueError('Frozen metric ordinary description content differs from selection.')
    if list(manager.template) != metadata['text_template']:
        raise ValueError('Frozen metric text template differs from selection.')
    if repr(manager.test_transform) != metadata['support_transform']:
        raise ValueError('Frozen metric support transform differs from selection.')
    current = manager._support_metadata()
    expected = metadata['input_provenance']
    for key in _INPUT_PROVENANCE_KEYS:
        actual = current.get(key)
        same = (isinstance(actual, str) and bool(actual)
                and dataset_slug(actual) == dataset_slug(expected[key])
                if key == 'dataset' else actual == expected[key])
        if not same:
            raise ValueError(f'Frozen metric input mapping differs from selection: {key}.')
    if list(manager.class_names[:cfg.DATASET.NUM_INIT_CLS]) != metadata['base_class_names']:
        raise ValueError('Frozen metric initial class names differ from selection.')
    state['reference_content_verified'] = True
    return True


def verify_base_bank(state, sample_ids, feature_dim):
    if feature_dim != state['metadata']['feature_dim']:
        raise ValueError('Frozen metric feature dimension differs from the base CLIP bank.')
    if sample_identity_hash(sample_ids) != state['metadata']['base_sample_ids_sorted_sha256']:
        raise ValueError('Frozen metric was fitted to a different set of base training images.')
    state['base_bank_verified'] = True


def prepare_session_metric(state, raw_image_means):
    if not state['base_bank_verified']:
        raise RuntimeError('Verify the original base training bank before using a frozen metric.')
    if not state.get('reference_content_verified', False):
        raise RuntimeError('Verify reference content before using a frozen metric.')
    return prepare_residual_classifier(raw_image_means, state['metric'])


def apply_session_metric(state, prepared, features, reference_votes):
    return apply_metric_residual(reference_votes, features, prepared, state['gain'])


def metric_report(state):
    return {'model_path': state['model_path'], 'model_sha256': state['model_sha256'],
            'gain': state['gain'], 'base_bank_verified': state['base_bank_verified'],
            'reference_content_verified': state['reference_content_verified'],
            'precision': state['metadata']['precision'],
            'training_protocol': state['metadata'].get('training_protocol'),
            'method': state['metadata'].get('method'),
            'source': state['metadata'].get('source'),
            'adaptation': 'base_only_fitted_frozen_during_incremental_sessions',
            'query_frequency_encodings': 0}
