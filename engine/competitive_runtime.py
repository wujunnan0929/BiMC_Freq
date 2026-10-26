"""Safe NPZ artifacts and frozen incremental competitive fusion."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from engine.prototype_evaluation import validate_calibration
from engine.prototype_estimator_runtime import _blend
from engine.safd_metric_runtime import reference_settings, validate_reference_metadata
from models.competitive_fusion import CompetitiveFusion, validate_fixed, support_quality, branch_probabilities, competitive_scores
from utils.frequency_descriptions import dataset_slug


FORMAT = 'competitive_fusion_v1'


def validate_metadata(d):
    validate_reference_metadata(d)
    if d.get('data_source') != 'base_training_only':
        raise ValueError('Fusion requires base_training_only provenance.')
    if not isinstance(d.get('feature_dim'), int) or d['feature_dim'] < 1 or not d.get('base_sample_ids_sorted_sha256'):
        raise ValueError('Missing fusion feature dimension/base-image identity.')
    keys = {'hidden', 'method', 'use_support_quality', 'group_limit', 'mix_limit'}
    if not isinstance(d.get('model_config'), dict) or set(d['model_config']) != keys:
        raise ValueError('Fusion model_config is incomplete.')
    _blend(d.get('blend')); validate_calibration(d.get('calibration'))
    if d.get('calibration') is None:
        raise ValueError('Fusion calibration is missing.')
    validate_fixed(d.get('fixed'))
    if not isinstance(d.get('selected_step'), int) or isinstance(d['selected_step'], bool) or d['selected_step'] < 0:
        raise ValueError('Fusion selected_step must be nonnegative.')
    if isinstance(d.get('fitting_seed'), bool) or not isinstance(d.get('fitting_seed'), int) or d['fitting_seed'] < 1 or not d.get('selection_protocol_sha256'):
        raise ValueError('Fusion fitting seed/protocol identity is missing.')


def export_fusion(path, model, metadata, report):
    path = Path(path)
    if path.suffix != '.npz' or any(p.exists() for p in (path, path.with_suffix('.json'), path.with_suffix('.training.json'))):
        raise ValueError('Choose a fresh .npz output; fusion artifacts are not overwritten.')
    d = dict(metadata, format=FORMAT)
    validate_metadata(d)
    if d['model_config'] != model.configuration() or d.get('trainable_parameters') != sum(p.numel() for p in model.parameters()):
        raise ValueError('Fusion architecture/parameter count mismatch.')
    arrays = {'model__'+key: value.detach().cpu().numpy() for key, value in model.state_dict().items()}
    if not all(value.dtype.kind in 'fi' and np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('Fusion parameters must be finite real arrays.')
    json.dumps(d, allow_nan=False)
    training_text = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    d['model_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix('.json').write_text(json.dumps(d, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    path.with_suffix('.training.json').write_text(training_text, encoding='utf-8')
    return d


def load_fusion(path, cfg, clip_weight_sha256):
    path = Path(path)
    d = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if d.get('format') != FORMAT:
        raise ValueError('Unsupported competitive fusion format.')
    validate_metadata(d)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if d.get('model_sha256') != digest:
        raise ValueError('Fusion checksum mismatch.')
    if dataset_slug(d.get('dataset', '')) != dataset_slug(cfg.DATASET.NAME):
        raise ValueError('Fusion dataset mismatch.')
    if not clip_weight_sha256 or d.get('clip_weight_sha256') != clip_weight_sha256:
        raise ValueError('Fusion CLIP weights mismatch.')
    if d.get('base_class_ids') != list(range(cfg.DATASET.NUM_INIT_CLS)):
        raise ValueError('Fusion base-class bank mismatch.')
    if d.get('reference_settings') != reference_settings(cfg) or cfg.TRAINER.BiMC.PREC != 'fp32':
        raise ValueError('Fusion reference settings/precision mismatch.')
    model = CompetitiveFusion(**d['model_config'])
    expected, weights = model.state_dict(), {}
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != {'model__'+key for key in expected}:
            raise ValueError('Fusion parameter names mismatch.')
        for key, template in expected.items():
            value = saved['model__'+key]
            if value.shape != tuple(template.shape) or value.dtype.kind not in 'fi' or not np.isfinite(value).all():
                raise ValueError('Fusion parameter shape/value mismatch.')
            weights[key] = torch.as_tensor(value).to(template.dtype)
    model.load_state_dict(weights, strict=True)
    if d.get('trainable_parameters') != sum(p.numel() for p in model.parameters()):
        raise ValueError('Fusion parameter count mismatch.')
    model.to(cfg.DEVICE.DEVICE_NAME).eval().requires_grad_(False)
    options = cfg.TRAINER.BiMC.COMPETITIVE_FUSION
    if options.MODE not in ('zero', 'calibrated', 'fixed', 'learned'):
        raise ValueError('Unknown fusion runtime MODE.')
    return {'model': model, 'metadata': d, 'model_path': str(path.resolve()), 'model_sha256': digest,
            'blend': _blend(d['blend'] if options.BLEND == -1 else options.BLEND), 'mode': options.MODE,
            'calibration': validate_calibration(d['calibration']), 'fixed': validate_fixed(d['fixed']),
            'base_bank_verified': False, 'reference_content_verified': False}


def verified(state):
    if not state.get('base_bank_verified') or not state.get('reference_content_verified'):
        raise RuntimeError('Verify original base images and reference content before fusion.')


@torch.no_grad()
def prepare_current(state, cfg, current):
    verified(state)
    labels = current['images_targets']
    supports = [current['images_features'][labels == int(c)] for c in current['class_index']]
    b = cfg.TRAINER.BiMC
    lam = b.LAMBDA_T if b.TEXT_CALIBRATION else 0.
    semantic = F.normalize((1-lam)*current['text_features']+lam*current['description_proto'], dim=-1)
    current['fusion_quality'] = support_quality(supports, current['image_proto'], semantic)


@torch.no_grad()
def apply_fusion(state, cfg, merged, features, reference_votes):
    from engine.prototype_evaluation import branch_logits, score_record
    from engine.competitive_evaluation import identity_fixed
    from models.competitive_fusion import fixed_scores
    verified(state)
    # A group gate has no incremental group at the base session.
    if state['mode'] == 'zero' or len(merged['image_proto']) == cfg.DATASET.NUM_INIT_CLS:
        return reference_votes
    key = (merged['cov_image'].data_ptr(), len(merged['image_proto']))
    if state.get('_session_key') != key:
        state['_session_key'] = key
        state['_session_statistics'] = {'num_old': cfg.DATASET.NUM_INIT_CLS,
            'reference_prototypes': merged['image_proto'], 'covariance': merged['cov_image'],
            'cov_inverse': torch.pinverse(merged['cov_image'].float()),
            **{k: merged[k] for k in ('text_features', 'description_proto', 'description_features', 'description_targets')}}
    row = {**state['_session_statistics'], 'query': features, 'reference_votes': reference_votes}
    calibration = state['calibration']
    if state['mode'] == 'calibrated':
        return score_record(row, cfg, calibration=calibration)
    p = branch_probabilities(branch_logits(row, cfg), calibration)
    anchor = (score_record(row, cfg, calibration=calibration) if state['fixed'] == identity_fixed(cfg) else
              fixed_scores(p, cfg.DATASET.NUM_INIT_CLS, state['fixed'], calibration))
    if state['mode'] == 'fixed':
        return anchor
    return competitive_scores(state['model'], p, cfg.DATASET.NUM_INIT_CLS, merged['fusion_quality'],
                              state['fixed'], calibration, state['blend'], anchor)


def fusion_report(state):
    return {**{key: state[key] for key in ('model_path', 'model_sha256', 'mode', 'blend', 'calibration', 'fixed',
             'base_bank_verified', 'reference_content_verified')},
        **{key: state['metadata'].get(key) for key in ('model_config', 'selected_step', 'fitting_seed',
              'selection_protocol_sha256', 'fit_wall_seconds', 'anchor_selection_wall_seconds', 'cache_extraction_wall_seconds', 'auxiliary')},
        'base_fit_parameter_budget': state['metadata']['trainable_parameters'],
        'incremental_trainable_parameters': 0, 'query_auxiliary_encodings': 0,
        'base_session_rule': 'unchanged_original_BiMC'}


def checkpoint_state(state):
    if state is None:
        return None
    return {**{k: v for k, v in state.items() if k not in ('model', '_session_key', '_session_statistics')},
            'model_state': state['model'].state_dict()}
