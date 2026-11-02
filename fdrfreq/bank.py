"""Feature bank: a directory of .npy arrays plus meta.json.

Layout::

    <bank>/meta.json
    <bank>/arrays/<name>.npy

Array names (N_tr / N_te = all training / test images in the dataset's
original order, so row i is the manager's global sample index i):

    train_labels, test_labels                 int64 [N]
    train_feat_<view>, test_feat_<view>       float16/32 [N,D], unit norm
        views: orig, blur<s> (Gaussian sigma s px), flip, gray
    train_dct_L<l>_<kind>, test_dct_...       float16 [N,3,W] band RMS energies
        of ViT patch tokens after block l; kind = radial | random
    text_prompt                               [C,D] template-ensemble text
    desc_feat, desc_target, desc_proto        BiMC description bank
    kw_pair_class, kw_pair_word, kw_pair_tf   int64 [P] (class, word) pairs
    kw_score_final, kw_score_fdr              float16 [P,D] grounding scores
    support_seed<s>                           int64 [sum], train rows of the
    support_seed<s>_offsets                   engine's session supports

meta.json holds class names, split sizes, views, DCT layers, keyword
vocabulary, provenance and per-array shapes. Arrays are opened with
mmap_mode='r' so multi-GB banks are cheap to load.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

FORMAT = 'fdrfreq_bank_v1'


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def l2n(x, axis=-1, eps=1e-12):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), eps)


class Bank:
    """Read-only view of a feature bank (or an in-memory synthetic one)."""

    def __init__(self, arrays, meta):
        self._arrays = arrays
        self.meta = meta
        req = ('dataset', 'class_names', 'num_init', 'num_inc', 'base_shot', 'inc_shot')
        missing = [k for k in req if k not in meta]
        if missing:
            raise ValueError('Bank meta is missing: ' + ', '.join(missing))
        self.class_names = list(meta['class_names'])
        self.num_classes = len(self.class_names)
        self.num_init = int(meta['num_init'])
        self.num_inc = int(meta['num_inc'])
        self.base_shot = int(meta['base_shot'])
        self.inc_shot = int(meta['inc_shot'])
        for key in ('train_labels', 'test_labels', 'train_feat_orig', 'test_feat_orig',
                    'text_prompt', 'desc_feat', 'desc_target', 'desc_proto'):
            if key not in arrays:
                raise ValueError('Bank is missing array: ' + key)
        self.dim = int(arrays['train_feat_orig'].shape[1])
        if arrays['text_prompt'].shape != (self.num_classes, self.dim):
            raise ValueError('text_prompt must be [C,D].')
        self._cache = {}

    # ------------------------------------------------------------------ basic
    def has(self, name):
        return name in self._arrays

    def array(self, name):
        if name not in self._arrays:
            raise KeyError('Bank has no array ' + name)
        return self._arrays[name]

    def labels(self, split):
        return np.asarray(self.array(split + '_labels'), dtype=np.int64)

    def views(self, split='train'):
        prefix = split + '_feat_'
        return sorted(k[len(prefix):] for k in self._arrays if k.startswith(prefix))

    def feat(self, split, view='orig', rows=None):
        """Unit-norm float32 features; rows=None returns the full matrix (cached)."""
        name = f'{split}_feat_{view}'
        if rows is None:
            if name not in self._cache:
                self._cache[name] = l2n(self.array(name))
            return self._cache[name]
        if name in self._cache:
            return self._cache[name][rows]
        return l2n(np.asarray(self.array(name)[np.asarray(rows, dtype=np.int64)]))

    def dct(self, split, layer, kind='radial', rows=None):
        name = f'{split}_dct_L{int(layer)}_{kind}'
        value = self.array(name)
        out = value if rows is None else value[rows]
        return np.asarray(out, dtype=np.float32)

    def dct_layers(self):
        layers = set()
        for k in self._arrays:
            if k.startswith('train_dct_L'):
                layers.add(int(k.split('_')[2][1:]))
        return sorted(layers)

    # ------------------------------------------------------------------- text
    @property
    def text_prompt(self):
        return l2n(self.array('text_prompt'))

    @property
    def desc_proto(self):
        return l2n(self.array('desc_proto'))

    @property
    def desc_feat(self):
        return l2n(self.array('desc_feat'))

    @property
    def desc_target(self):
        return np.asarray(self.array('desc_target'), dtype=np.int64)

    # --------------------------------------------------------------- keywords
    def has_keywords(self):
        return all(self.has(k) for k in ('kw_pair_class', 'kw_pair_word', 'kw_pair_tf',
                                         'kw_score_final'))

    @property
    def kw_vocab(self):
        return list(self.meta.get('kw_vocab', []))

    # ---------------------------------------------------------------- support
    def stored_support(self, seed):
        key = f'support_seed{int(seed)}'
        if not self.has(key):
            return None
        flat = np.asarray(self.array(key), dtype=np.int64)
        offsets = np.asarray(self.array(key + '_offsets'), dtype=np.int64)
        return [flat[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]


def save_bank(path, arrays, meta, *, overwrite=False):
    path = Path(path)
    arr_dir = path / 'arrays'
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f'{path} exists; pass overwrite=True.')
    arr_dir.mkdir(parents=True, exist_ok=True)
    shapes, dtypes, hashes = {}, {}, {}
    for name, value in arrays.items():
        value = np.ascontiguousarray(value)
        file = arr_dir / f'{name}.npy'
        np.save(file, value, allow_pickle=False)
        shapes[name], dtypes[name] = list(value.shape), str(value.dtype)
        hashes[name] = _sha256(file)
    meta = dict(meta)
    meta.update(format=FORMAT, array_shapes=shapes, array_dtypes=dtypes, array_sha256=hashes)
    tmp = path / 'meta.json.tmp'
    tmp.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path / 'meta.json')
    return path


def finalize_bank(path, meta):
    """Write meta.json for arrays already written to <path>/arrays (e.g. memmaps)."""
    path = Path(path)
    shapes, dtypes, hashes = {}, {}, {}
    for file in sorted((path / 'arrays').glob('*.npy')):
        value = np.load(file, mmap_mode='r', allow_pickle=False)
        name = file.stem
        shapes[name], dtypes[name] = list(value.shape), str(value.dtype)
        hashes[name] = _sha256(file)
    meta = dict(meta)
    meta.update(format=FORMAT, array_shapes=shapes, array_dtypes=dtypes, array_sha256=hashes)
    (path / 'meta.json').write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding='utf-8')
    return path


def load_bank(path, *, verify=False, mmap=True):
    path = Path(path)
    meta = json.loads((path / 'meta.json').read_text(encoding='utf-8'))
    if meta.get('format') != FORMAT:
        raise ValueError(f'{path} is not a {FORMAT} bank.')
    arrays = {}
    for name, shape in meta['array_shapes'].items():
        file = path / 'arrays' / f'{name}.npy'
        if verify and _sha256(file) != meta['array_sha256'][name]:
            raise ValueError('Checksum mismatch: ' + name)
        value = np.load(file, mmap_mode='r' if mmap else None, allow_pickle=False)
        if list(value.shape) != shape:
            raise ValueError('Shape mismatch: ' + name)
        arrays[name] = value
    return Bank(arrays, meta)


def bank_fingerprint(bank):
    """Short identifier recorded in every result file."""
    meta = bank.meta
    payload = json.dumps({k: meta.get(k) for k in ('dataset', 'backbone', 'array_sha256')},
                         sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]
