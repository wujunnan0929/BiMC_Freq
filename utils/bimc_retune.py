"""Manual BiMC configuration, with no dependency on torch or dataset loading."""

import hashlib
import math
from pathlib import Path

from yacs.config import CfgNode as CN


def merge_retune_config(cfg, filename):
    """Keep the dataset identity explicit; override only after loading defaults."""
    with open(filename, encoding='utf-8') as handle:
        profile = CN.load_cfg(handle)
    expected = profile.get('DATASET', {}).get('NAME')
    if expected is not None and expected != cfg.DATASET.NAME:
        raise ValueError(
            f'Retune profile is for {expected}, but --data_cfg selects '
            f'{cfg.DATASET.NAME}. Choose the matching configs/retune profile.'
        )
    cfg.merge_from_other_cfg(profile)


def _number(name, value, low, high=None, integer=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or (integer and not isinstance(value, int))
            or value < low or (high is not None and value > high)):
        domain = f'[{low}, {high}]' if high is not None else f'>= {low}'
        kind = 'integer' if integer else 'finite number'
        raise ValueError(f'{name} must be a {kind} in {domain}; got {value!r}.')


def prepare_retune_config(cfg):
    """Validate the final overrides and assign an auditable default output path.

    This is deliberately called only for --retune_cfg, so existing experiment
    configurations retain their current behavior. No parameter is auto-tuned.
    """
    b = cfg.TRAINER.BiMC
    for name, value in (
        ('DATASET.BETA', cfg.DATASET.BETA),
        ('DATASET.ENSEMBLE_ALPHA', cfg.DATASET.ENSEMBLE_ALPHA),
        ('TRAINER.BiMC.LAMBDA_I', b.LAMBDA_I),
        ('TRAINER.BiMC.LAMBDA_T', b.LAMBDA_T),
    ):
        _number(name, value, 0.0, 1.0)
    _number('TRAINER.BiMC.TAU', b.TAU, 1, integer=True)
    for name in ('GAMMA_BASE', 'GAMMA_INC'):
        _number('TRAINER.BiMC.' + name, b[name], 0.0)
    for name in ('VISION_CALIBRATION', 'TEXT_CALIBRATION', 'USING_ENSEMBLE'):
        if not isinstance(b[name], bool):
            raise ValueError('TRAINER.BiMC.' + name + ' must be True or False.')
    if b.PREC not in ('fp16', 'fp32', 'amp'):
        raise ValueError('TRAINER.BiMC.PREC must be fp16, fp32 or amp.')
    _number('SEED', cfg.SEED, 0, integer=True)
    _number('DATASET.SUPPORT_SEED', cfg.DATASET.SUPPORT_SEED, -1, integer=True)
    _number('DATASET.NUM_INC_SHOT', cfg.DATASET.NUM_INC_SHOT, 1, integer=True)

    # A retuned baseline must not silently include one of the experimental heads.
    optional = (
        'FREQUENCY', 'RESIDUAL', 'CONSENSUS', 'UNCERTAINTY',
        'DUAL_FREQUENCY', 'GUIDANCE', 'SAFD_METRIC',
        'PROTOTYPE_ESTIMATOR', 'PROTOTYPE_COMPLETION', 'COMPETITIVE_FUSION',
    )
    enabled = [name for name in optional if b[name].ENABLED]
    if b.FREQUENCY.ROUTER.ENABLED:
        enabled.append('FREQUENCY.ROUTER')
    for name in ('RESIDUAL', 'UNCERTAINTY', 'DUAL_FREQUENCY', 'GUIDANCE',
                 'PROTOTYPE_COMPLETION'):
        if b[name].BASE_ONLY:
            enabled.append(name + '.BASE_ONLY')
    if b.RESIDUAL.RESERVE_BASE_VALIDATION:
        enabled.append('RESIDUAL.RESERVE_BASE_VALIDATION')
    if enabled:
        raise ValueError('BiMC retuning requires optional branches disabled: '
                         + ', '.join(enabled))

    if not cfg.OUTPUT_DIR:
        # Include all resolved settings, not just the four most commonly tuned
        # scalars: precision, backbone, support seed and data paths matter too.
        digest = hashlib.sha256(cfg.dump().encode('utf-8')).hexdigest()[:12]
        variant = 'bimc_ensemble' if b.USING_ENSEMBLE else 'bimc'
        cfg.OUTPUT_DIR = str(
            Path('outputs') / 'retune' / cfg.DATASET.NAME.lower()
            / f'{variant}_shot{cfg.DATASET.NUM_INC_SHOT}_seed{cfg.SEED}_{digest}'
        )


def check_output_directory(directory):
    """Avoid silently replacing a previous manual calibration experiment."""
    path = Path(directory)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(
            f'Retune output already exists and is nonempty: {path}. '
            'Use --opts OUTPUT_DIR <new-directory> for another run.'
        )


def print_retune_settings(cfg):
    b = cfg.TRAINER.BiMC
    support = cfg.DATASET.SUPPORT_SEED
    print('BiMC manual retuning (fixed parameters; no automatic search)')
    print(f'  dataset={cfg.DATASET.NAME}, backbone={cfg.MODEL.BACKBONE.NAME}, '
          f'shots={cfg.DATASET.NUM_INC_SHOT}, seed={cfg.SEED}, '
          f'support_seed={support if support >= 0 else cfg.SEED}')
    print(f'  beta={cfg.DATASET.BETA}, lambda_t={b.LAMBDA_T}, '
          f'lambda_i={b.LAMBDA_I}, tau={b.TAU}')
    print(f'  text_calibration={b.TEXT_CALIBRATION}, '
          f'vision_calibration={b.VISION_CALIBRATION}, ensemble={b.USING_ENSEMBLE}')
    print(f'  ensemble_alpha={cfg.DATASET.ENSEMBLE_ALPHA}, '
          f'gamma_base={b.GAMMA_BASE}, gamma_inc={b.GAMMA_INC}')
    print(f'  output={cfg.OUTPUT_DIR}')
