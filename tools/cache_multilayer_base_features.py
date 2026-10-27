"""Cache native ViT CLS features and fixed attribute text from base training only.

One original-image forward supplies all selected layers. No image view,
benchmark query, patch-grid frequency transform, or language projection of
intermediate layers is used. --dry-run uses only the standard library.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.cache_safd_base_features import sha256, canonical_sha, output_paths, write_cache
from utils.attribute_descriptions import GROUPS, load_attribute_manifest

FORMAT = 'multilayer_base_features_v1'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=('cub200', 'cifar100', 'miniimagenet'), default='cub200')
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--layers', type=int, nargs='+', default=[3, 6, 9])
    p.add_argument('--attribute-descriptions', help='Optional fixed class -> shape/part/color/pattern string lists.')
    p.add_argument('--gpu', default='0')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--support-manifest')
    p.add_argument('--limit-per-class', type=int, help='Smoke subset; cannot be used for real development selection.')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    if not a.layers or len(set(a.layers)) != len(a.layers) or min(a.layers) < 1:
        p.error('--layers requires distinct positive one-based block indices.')
    if min(a.seed, a.batch_size) < 1 or a.workers < 0 or (a.limit_per_class is not None and a.limit_per_class < 1):
        p.error('seed/batch-size/limit must be positive; workers must be nonnegative.')
    if Path(a.output).suffix.lower() != '.npz':
        p.error('--output must end in .npz.')
    for value in (a.support_manifest, a.attribute_descriptions):
        if value and not Path(value).is_file():
            p.error('Input file does not exist: ' + value)
    return a


def build_cfg(args):
    from main import setup_cfg
    b = 'TRAINER.BiMC.'
    options = ['DATASET.ROOT', args.data_root, 'DEVICE.DEVICE_NAME', args.device,
               'DEVICE.GPU_ID', repr(args.gpu), 'SEED', str(args.seed),
               'DATASET.SUPPORT_SEED', str(args.seed), 'DATASET.SUPPORT_MANIFEST', args.support_manifest or '',
               'DATALOADER.NUM_WORKERS', str(args.workers), b+'PREC', 'fp32']
    for key in ('FREQUENCY.ENABLED', 'FREQUENCY.ROUTER.ENABLED', 'FREQUENCY.USE_EXPLICIT_DESCRIPTIONS',
                'GUIDANCE.ENABLED', 'CONSENSUS.ENABLED', 'UNCERTAINTY.ENABLED',
                'DUAL_FREQUENCY.ENABLED', 'RESIDUAL.ENABLED', 'SAFD_METRIC.ENABLED',
                'PROTOTYPE_ESTIMATOR.ENABLED', 'PROTOTYPE_COMPLETION.ENABLED', 'COMPETITIVE_FUSION.ENABLED'):
        options.extend((b+key, 'False'))
    return setup_cfg(str(ROOT/f'configs/datasets/{args.dataset}.yaml'),
                     str(ROOT/'configs/trainers/bimc_frequency_guidance.yaml'), options)


def collect_base_features(manager, model, layers=(3, 6, 9), *, batch_size=32,
                          workers=0, device='cpu', limit_per_class=None):
    """Cache task-zero training support only, with one encoder call per batch."""
    import numpy as np
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Subset
    dataset = manager.get_dataset(0, source='train', mode='test')
    classes = [int(c) for c in manager.class_index_in_task[0]]
    labels = np.asarray(dataset.labels)
    if set(labels.tolist()) != set(classes):
        raise ValueError('Base cache requires exactly the original base training classes.')
    rows = list(range(len(dataset)))
    if limit_per_class is not None:
        rows = [int(i) for c in classes for i in np.flatnonzero(labels == c)[:limit_per_class]]
    loader = DataLoader(Subset(dataset, rows), batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=str(device).startswith('cuda'))
    keys = ['final'] + ['layer_' + str(i) for i in layers]
    banks = {k: [] for k in keys}
    targets, sample_ids = [], []
    model.eval().requires_grad_(False)
    count = 0
    with torch.no_grad():
        for index, batch in enumerate(loader):
            images = batch['image'].to(device)
            values = model.clip_model.encode_image_layers(images, layers=tuple(layers))
            if set(values) != set(keys):
                raise ValueError('Selected-layer encoder returned the wrong feature keys.')
            for key in keys:
                value = values[key].float()
                if (value.ndim != 2 or len(value) != len(images) or not torch.isfinite(value).all()
                        or (value.norm(dim=-1) <= 1e-12).any()):
                    raise ValueError('Encoder returned invalid nonzero finite [N,D] features: ' + key)
                banks[key].append(F.normalize(value, dim=-1).cpu().numpy())
            identity = batch.get('global_index', batch.get('sample_id'))
            if identity is None:
                raise ValueError('Native layer caching requires stable global sample IDs.')
            targets.extend(int(i) for i in batch['label'])
            sample_ids.extend(int(i) for i in identity)
            model.image_encoding_counts['original'] += len(images)
            count += len(images)
            if index == 0 or (index+1) % 20 == 0:
                print(f'Multilayer base cache: {index+1}/{len(loader)} batches', flush=True)
    if not count or len(set(sample_ids)) != count:
        raise ValueError('Base cache is empty or contains duplicate global sample identities.')
    arrays = {k: np.concatenate(values).astype(np.float32) for k, values in banks.items()}
    arrays.update(labels=np.asarray(targets, dtype=np.int64), sample_id=np.asarray(sample_ids, dtype=np.int64))
    return arrays, {'original': count, 'auxiliary': 0, 'benchmark_query': 0,
                    'selected_layer_forward_images': count}, len(dataset)


def encode_attribute_text(manifest, encoder, dimension):
    """Encode valid strings only; unknown slots have zero vectors and false masks."""
    import numpy as np
    c, g, k = len(manifest['classes']), len(GROUPS), manifest['per_group']
    ordinary_count = manifest['ordinary_budget']
    arrays = {'attr_text': np.zeros((c, g, k, dimension), dtype=np.float32),
              'attr_mask': np.zeros((c, g, k), dtype=bool),
              'ordinary_text': np.zeros((c, ordinary_count, dimension), dtype=np.float32),
              'ordinary_mask': np.zeros((c, ordinary_count), dtype=bool)}
    texts, slots = [], []
    for index, record in enumerate(manifest['classes']):
        for group_index, group in enumerate(GROUPS):
            for j, item in enumerate(record['groups'][group]):
                texts.append(item['text'])
                slots.append(('attr_text', 'attr_mask', (index, group_index, j)))
        for j, item in enumerate(record['ordinary']):
            texts.append(item['text'])
            slots.append(('ordinary_text', 'ordinary_mask', (index, j)))
    if texts:
        values = np.asarray(encoder(texts), dtype=np.float32)
        if values.shape != (len(texts), dimension) or not np.isfinite(values).all():
            raise ValueError('Text encoder must return finite [valid_strings,D] features.')
        norm = np.linalg.norm(values, axis=-1, keepdims=True)
        if (norm <= 1e-12).any():
            raise ValueError('Text encoder returned a zero vector for valid content.')
        values = values / norm
        for value, (bank, mask, location) in zip(values, slots):
            arrays[bank][location], arrays[mask][location] = value, True
    return arrays


def collect_base_text(manager, model, cfg, attribute_path=None):
    import numpy as np
    import torch
    import models.clip.clip as clip
    from tools.cache_safd_base_features import collect_base_text as collect_ordinary
    ids = [int(c) for c in manager.class_index_in_task[0]]
    names = [str(manager.class_names[c]) for c in ids]
    arrays = collect_ordinary(manager, model, cfg)
    manifest = load_attribute_manifest(cfg.DATASET.GPT_PATH, names, attribute_path)
    def encoder(sentences):
        blocks = []
        with torch.no_grad():
            for start in range(0, len(sentences), 128):
                tokens = clip.tokenize(sentences[start:start+128]).to(model.device)
                blocks.append(model.clip_model.encode_text(tokens).detach().float().cpu().numpy())
        return np.concatenate(blocks)
    arrays.update(encode_attribute_text(manifest, encoder, arrays['text_features'].shape[-1]))
    with torch.no_grad():
        structured = model.inference_frequency_text_feature(names)
    arrays['structured_text'] = structured.detach().float().cpu().numpy().astype(np.float32)
    return arrays, manifest


def load_cache(path, *, require_complete=True):
    """Validate authenticated numerical cache and fixed text-mask provenance."""
    import numpy as np
    cache_path, meta_path, support_path = output_paths(path)
    metadata = json.loads(meta_path.read_text(encoding='utf-8'))
    expected = metadata.pop('metadata_payload_sha256', None)
    if (canonical_sha(metadata) != expected or metadata.get('format') != FORMAT
            or metadata.get('status') != 'completed' or metadata.get('data_source') != 'base_training_only'):
        raise ValueError('Invalid/incomplete multilayer cache metadata.')
    metadata['metadata_payload_sha256'] = expected
    if sha256(cache_path) != metadata.get('cache_sha256') or sha256(support_path) != metadata.get('base_support_sha256'):
        raise ValueError('Multilayer cache/support checksum mismatch.')
    if require_complete and metadata.get('complete_base_bank') is not True:
        raise ValueError('Development selection requires a complete base bank, not a smoke-test subset.')
    layers, base_ids = metadata.get('selected_layers'), metadata.get('base_class_ids')
    if (not isinstance(layers, list) or not layers or len(set(layers)) != len(layers)
            or any(isinstance(i, bool) or not isinstance(i, int) or i < 1 for i in layers)
            or not isinstance(base_ids, list) or base_ids != list(range(len(base_ids)))):
        raise ValueError('Cache requires distinct selected layers and contiguous original base IDs.')
    layer_keys = ['layer_' + str(i) for i in layers]
    names = ['final', *layer_keys, 'labels', 'sample_id', 'text_features', 'description_features',
             'description_targets', 'description_proto', 'structured_text',
             'attr_text', 'attr_mask', 'ordinary_text', 'ordinary_mask']
    with np.load(cache_path, allow_pickle=False) as saved:
        if set(saved.files) != set(names):
            raise ValueError('Multilayer cache has missing or unexpected arrays.')
        arrays = {k: saved[k] for k in names}
    n, c = len(arrays['labels']), len(metadata['base_class_ids'])
    if arrays['final'].ndim != 2:
        raise ValueError('Final visual features must have shape [N,D].')
    d = arrays['final'].shape[-1]
    if (not n or c < 1 or arrays['labels'].dtype != np.int64 or arrays['sample_id'].dtype != np.int64
            or arrays['labels'].shape != (n,) or arrays['sample_id'].shape != (n,)
            or len(np.unique(arrays['sample_id'])) != n
            or set(arrays['labels'].tolist()) != set(metadata['base_class_ids'])):
        raise ValueError('Invalid multilayer cache labels or sample identities.')
    def unit(key, shape=None):
        value = arrays[key]
        if (value.dtype != np.float32 or (shape is not None and value.shape != shape)
                or not np.isfinite(value).all() or not np.allclose(np.linalg.norm(value, axis=-1), 1., atol=1e-4)):
            raise ValueError('Expected finite float32 unit feature vectors: ' + key)
    for key in ['final', *layer_keys]:
        if arrays[key].ndim != 2 or arrays[key].shape[0] != n or arrays[key].shape[-1] < 1:
            raise ValueError('Native visual features must have shape [N,D]: ' + key)
        unit(key)
    unit('text_features', (c, d))
    unit('description_proto', (c, d))
    unit('structured_text', (c, 3, d))
    unit('description_features')
    desc, targets = arrays['description_features'], arrays['description_targets']
    if (desc.ndim != 2 or desc.shape[1] != d or targets.shape != (len(desc),)
            or targets.dtype != np.int64 or set(targets.tolist()) != set(metadata['base_class_ids'])):
        raise ValueError('Description bank does not cover the original base classes.')
    for key, mask_key, shape in [('attr_text', 'attr_mask', (c, 4, 2, d)),
                                  ('ordinary_text', 'ordinary_mask', (c, 8, d))]:
        value, mask = arrays[key], arrays[mask_key]
        if (value.shape != shape or value.dtype != np.float32 or mask.shape != shape[:-1]
                or mask.dtype != np.bool_ or not np.isfinite(value).all()
                or not np.allclose(np.linalg.norm(value[mask], axis=-1), 1., atol=1e-4)
                or np.any(value[~mask] != 0)):
            raise ValueError('Invalid attribute/ordinary text values or unknown masks: ' + key)
    manifest = metadata.get('attribute_manifest', {})
    if manifest.get('group_order') != list(GROUPS) or len(manifest.get('classes', [])) != c:
        raise ValueError('Cache is missing fixed attribute text provenance.')
    if metadata.get('base_class_names') is not None and [r.get('name') for r in manifest['classes']] != metadata['base_class_names']:
        raise ValueError('Fixed attribute text order differs from original base class names.')
    for index, record in enumerate(manifest['classes']):
        if [len(record['groups'][g]) for g in GROUPS] != arrays['attr_mask'][index].sum(-1).tolist():
            raise ValueError('Attribute provenance and encoded masks disagree.')
        if len(record['ordinary']) != int(arrays['ordinary_mask'][index].sum()):
            raise ValueError('Ordinary text provenance and encoded mask disagree.')
    support = json.loads(support_path.read_text(encoding='utf-8'))
    if support.get('sample_ids') != arrays['sample_id'].tolist() or support.get('targets') != arrays['labels'].tolist():
        raise ValueError('Base support sidecar identities disagree with cache rows.')
    if metadata.get('sample_count') != n or metadata.get('encoding_counts', {}).get('original') != n:
        raise ValueError('Multilayer cache sample/encoding counts disagree.')
    for key, value in arrays.items():
        if metadata.get('array_shapes', {}).get(key) != list(value.shape) or metadata.get('array_dtypes', {}).get(key) != str(value.dtype):
            raise ValueError('Cache arrays disagree with certified shapes/dtypes: ' + key)
    return arrays, metadata


def main(argv=None):
    a = parse_args(argv)
    if a.dry_run:
        print(json.dumps(vars(a), indent=2))
        return 0
    paths = output_paths(a.output)
    if not a.overwrite and any(p.exists() or p.with_suffix(p.suffix+'.tmp').exists() for p in paths):
        print('Refusing existing cache or sidecars; use a fresh --output.', file=sys.stderr)
        return 2
    import torch
    from utils.util import set_gpu, set_seed
    from datasets.data_manager import DatasetManager
    from models.bimc import BiMC
    set_gpu(a.gpu)
    set_seed(a.seed)
    start = time.perf_counter()
    cfg = build_cfg(a)
    manager = DatasetManager(cfg)
    model = BiMC(cfg, manager.template, a.device).eval().requires_grad_(False)
    arrays, counts, full_count = collect_base_features(manager, model, a.layers,
        batch_size=a.batch_size, workers=a.workers, device=a.device, limit_per_class=a.limit_per_class)
    text, attributes = collect_base_text(manager, model, cfg, a.attribute_descriptions)
    arrays.update(text)
    ids = [int(c) for c in manager.class_index_in_task[0]]
    provenance = manager._support_manifest()
    support = {k: provenance[k] for k in ('version', 'dataset', 'train_size', 'train_targets_sha256', 'train_samples_sha256', 'seed')}
    support.update(data_source='base_training_only', class_ids=ids,
                   sample_ids=arrays['sample_id'].tolist(), targets=arrays['labels'].tolist())
    source_files = ('tools/cache_multilayer_base_features.py', 'utils/attribute_descriptions.py',
                    'models/clip/model.py', 'models/bimc.py', 'datasets/data_manager.py', 'main.py',
                    f'configs/datasets/{a.dataset}.yaml', 'configs/trainers/bimc_frequency_guidance.yaml')
    metadata = {'schema_version': 1, 'format': FORMAT, 'dataset': a.dataset,
                'data_source': 'base_training_only', 'source': 'native_vit_layers', 'seed': a.seed,
                'base_class_ids': ids, 'base_class_names': [str(manager.class_names[c]) for c in ids],
                'selected_layers': list(a.layers), 'feature_dim': int(arrays['final'].shape[-1]),
                'feature_dimensions': {k: int(arrays[k].shape[-1]) for k in ['final', *['layer_'+str(i) for i in a.layers]]},
                'feature_semantics': 'Unit-normalized native CLS tokens; final uses the exact original final CLIP head.',
                'patch_features_cached': False, 'single_forward_per_image': True,
                'clip_weight_sha256': getattr(model.clip_model, 'clip_weight_sha256', None),
                'config_yaml': cfg.dump(), 'config_sha256': hashlib.sha256(cfg.dump().encode()).hexdigest(),
                'source_files_sha256': {name: sha256(ROOT/name) for name in source_files},
                'input_provenance': {k: provenance[k] for k in ('dataset', 'train_size', 'train_targets_sha256', 'train_samples_sha256')},
                'base_sample_ids_sorted_sha256': hashlib.sha256(json.dumps(sorted(int(i) for i in arrays['sample_id'])).encode()).hexdigest(),
                'ordinary_description_path': cfg.DATASET.GPT_PATH,
                'ordinary_description_sha256': sha256(cfg.DATASET.GPT_PATH),
                'attribute_description_path': str(Path(a.attribute_descriptions).resolve()) if a.attribute_descriptions else None,
                'attribute_description_sha256': sha256(a.attribute_descriptions) if a.attribute_descriptions else None,
                'attribute_manifest': attributes, 'text_template': list(manager.template),
                'support_transform': repr(manager.test_transform),
                'source_support_manifest': a.support_manifest,
                'source_support_manifest_sha256': sha256(a.support_manifest) if a.support_manifest else None,
                'encoding_counts': counts, 'sample_count': len(arrays['labels']),
                'full_base_sample_count': full_count, 'complete_base_bank': a.limit_per_class is None,
                'limit_per_class': a.limit_per_class,
                'purpose': 'smoke_test_subset' if a.limit_per_class else 'base_training_feature_cache',
                'has_text_features': True, 'benchmark_evaluation_performed': False,
                'new_gradient_training_performed': False, 'wall_seconds': time.perf_counter()-start}
    write_cache(a.output, arrays, metadata, support, overwrite=a.overwrite)
    load_cache(a.output, require_complete=a.limit_per_class is None)
    print('Multilayer base cache:', paths[0], 'rows=', metadata['sample_count'])
    print('Attribute coverage:', json.dumps(attributes['coverage'], ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
