"""Benchmark feature cache.

Layout for a cache stem ``S``:

``S.json``       metadata: dataset, class order, sessions, support draws, BiMC defaults
``S.train.npz``  every training image (test transform) + all text features
``S.test.npz``   every test image; only readable with a locked selection file

Visual arrays
-------------
``final``     [N, D] float32, unit-normalised original CLIP image embedding
``labels``    [N] int64 dataset class ids
``sample_id`` [N] int64 global dataset index (the DatasetManager index)
``blur``      [N, S, D] float16, unit embeddings of Gaussian-blurred views (sigma ladder)
``aug``       [N, A, D] float16, unit embeddings of non-frequency perturbations (control)
``dct``       [N, L, 3, W] float16, RMS DCT band energy of patch tokens (low/mid/high)
``dct_rand``  [N, L, 3, W] float16, same energies over random coefficient groups (control)

Text arrays (train file only)
-----------------------------
``text`` [C, D] prompt-template embeddings, ``desc`` [M, D] description embeddings,
``desc_targets`` [M] dataset class ids, ``desc_proto`` [C, D] normalised description means.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

FORMAT = 'fgdr_benchmark_v1'
VISUAL_KEYS = ('final', 'labels', 'sample_id')
OPTIONAL_VIEWS = ('blur', 'aug', 'dct', 'dct_rand')
TEXT_KEYS = ('text', 'desc', 'desc_targets', 'desc_proto')


def cache_paths(stem):
    stem = str(stem)
    for suffix in ('.json', '.train.npz', '.test.npz'):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
    return {'meta': Path(stem + '.json'), 'train': Path(stem + '.train.npz'),
            'test': Path(stem + '.test.npz')}


def file_sha256(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(chunk), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(payload):
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _check_arrays(arrays, split):
    for key in VISUAL_KEYS:
        if key not in arrays:
            raise ValueError(f'{split} cache is missing {key}')
    n = len(arrays['labels'])
    if arrays['final'].ndim != 2 or len(arrays['final']) != n or len(arrays['sample_id']) != n:
        raise ValueError(f'{split} cache rows disagree')
    if len(np.unique(arrays['sample_id'])) != n:
        raise ValueError(f'{split} cache has duplicate sample ids')
    for key in OPTIONAL_VIEWS:
        if key in arrays and len(arrays[key]) != n:
            raise ValueError(f'{split} view {key} rows disagree')
    if split == 'train':
        for key in TEXT_KEYS:
            if key not in arrays:
                raise ValueError(f'train cache is missing text array {key}')


def save_split(stem, split, arrays):
    if split not in ('train', 'test'):
        raise ValueError('split must be train or test')
    _check_arrays(arrays, split)
    path = cache_paths(stem)[split]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.npz')
    np.savez(tmp, **arrays)
    tmp.replace(path)
    return file_sha256(path)


def save_meta(stem, meta):
    meta = dict(meta)
    meta['format'] = FORMAT
    meta.pop('meta_sha256', None)
    meta['meta_sha256'] = canonical_sha(meta)
    path = cache_paths(stem)['meta']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    return path


def load_meta(stem):
    meta = json.loads(cache_paths(stem)['meta'].read_text(encoding='utf-8'))
    if meta.get('format') != FORMAT:
        raise ValueError('Not an FGDR benchmark cache: ' + str(stem))
    expected = meta.pop('meta_sha256', None)
    if expected != canonical_sha(meta):
        raise ValueError('Cache metadata checksum mismatch')
    meta['meta_sha256'] = expected
    return meta


def _load(path, expected_sha, verify, views=None):
    """Load a split; ``views`` restricts the optional view arrays (None = all)."""
    if verify and expected_sha and file_sha256(path) != expected_sha:
        raise ValueError('Checksum mismatch for ' + str(path))
    with np.load(path, allow_pickle=False) as saved:
        keys = [k for k in saved.files if views is None or k not in OPTIONAL_VIEWS or k in views]
        arrays = {key: saved[key] for key in keys}
    for key in ('final', 'blur', 'aug', 'text', 'desc', 'desc_proto'):
        if key in arrays:
            arrays[key] = arrays[key].astype(np.float32)
    for key in ('dct', 'dct_rand'):
        if key in arrays:
            arrays[key] = arrays[key].astype(np.float32)
    return arrays


def load_train(stem, verify=True, views=None):
    meta = load_meta(stem)
    arrays = _load(cache_paths(stem)['train'], meta.get('sha256', {}).get('train'), verify, views)
    _check_arrays(arrays, 'train')
    return arrays, meta


def load_lock(path):
    lock = json.loads(Path(path).read_text(encoding='utf-8'))
    expected = lock.pop('lock_sha256', None)
    if lock.get('status') != 'locked' or expected != canonical_sha(lock):
        raise ValueError('Selection file is not a valid locked selection: ' + str(path))
    lock['lock_sha256'] = expected
    return lock


def load_test(stem, lock_path, verify=True, views=None):
    """Test features are only released against a locked, checksummed selection."""
    lock = load_lock(lock_path)
    meta = load_meta(stem)
    if lock.get('cache_meta_sha256') != meta['meta_sha256']:
        raise ValueError('Selection was locked on a different cache')
    arrays = _load(cache_paths(stem)['test'], meta.get('sha256', {}).get('test'), verify, views)
    _check_arrays(arrays, 'test')
    return arrays, meta, lock
