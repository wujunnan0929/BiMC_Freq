"""Cache original and nested intervention CLIP features from base training only."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('cub200', 'cifar100', 'miniimagenet'), default='cub200')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True, help='New .npz file; metadata and base support sidecars use the same stem.')
    parser.add_argument('--source', choices=('safd', 'fft', 'augmentation', 'repeat'), default='safd')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--support-manifest', help='Optional existing paired support manifest; its path/seed must match the dataset.')
    parser.add_argument('--limit-per-class', type=int, help='Explicit smoke-test subset; never certified as a complete base cache.')
    parser.add_argument('--image-only', action='store_true', help='Omit ordinary text features; unsuitable for full-BiMC screening.')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Print the plan without loading data or CLIP.')
    args = parser.parse_args(argv)
    if min(args.seed, args.batch_size) < 1 or args.workers < 0 or (args.limit_per_class is not None and args.limit_per_class < 1):
        parser.error('seed/batch-size/limit-per-class must be positive and workers nonnegative.')
    if Path(args.output).suffix.lower() != '.npz':
        parser.error('--output must end in .npz')
    if args.support_manifest and not Path(args.support_manifest).is_file():
        parser.error('--support-manifest must be an existing file, not a new draw.')
    return args


def output_paths(output):
    output = Path(output).resolve()
    return output, output.with_suffix('.json'), output.with_suffix('.support.json')


def build_cfg(args):
    from main import setup_cfg
    b = 'TRAINER.BiMC.'
    options = ['DATASET.ROOT', args.data_root, 'DEVICE.DEVICE_NAME', args.device,
               'DEVICE.GPU_ID', repr(args.gpu), 'SEED', str(args.seed),
               'DATASET.SUPPORT_SEED', str(args.seed), 'DATALOADER.NUM_WORKERS', str(args.workers),
               'DATASET.SUPPORT_MANIFEST', args.support_manifest or '', b+'PREC', 'fp32']
    for key in ('FREQUENCY.ENABLED', 'FREQUENCY.ROUTER.ENABLED', 'FREQUENCY.USE_EXPLICIT_DESCRIPTIONS',
                'GUIDANCE.ENABLED', 'CONSENSUS.ENABLED', 'UNCERTAINTY.ENABLED',
                'DUAL_FREQUENCY.ENABLED', 'RESIDUAL.ENABLED', 'SAFD_METRIC.ENABLED'):
        options.extend((b + key, 'False'))
    return setup_cfg(str(ROOT / 'configs/datasets' / (args.dataset + '.yaml')),
                     str(ROOT / 'configs/trainers/bimc_frequency_guidance.yaml'), options)


def collect_base_features(manager, model, views, *, batch_size=32, workers=0,
                          device='cpu', limit_per_class=None):
    """Use precisely task zero's training support; no test loader is requested."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from models.safd_interventions import encode_intervention_features
    dataset = manager.get_dataset(0, source='train', mode='test')
    base_ids = [int(c) for c in manager.class_index_in_task[0]]
    labels = np.asarray(dataset.labels)
    if set(np.unique(labels).tolist()) != set(base_ids):
        raise ValueError('Base training dataset must contain exactly the original base classes.')
    selection = list(range(len(dataset)))
    if limit_per_class is not None:
        selection = [int(index) for cls in base_ids for index in np.flatnonzero(labels == cls)[:limit_per_class]]
    loader = DataLoader(Subset(dataset, selection), batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=str(device).startswith('cuda'))
    features, targets, sample_ids, distortions = [], [], [], []
    model.eval().requires_grad_(False)
    views = views.to(device).eval()
    original_count, auxiliary_count = 0, 0
    def auxiliary_encoder(images):
        model.image_encoding_counts['auxiliary'] += len(images)
        return model.clip_model.encode_image(images)
    with torch.no_grad():
        for index, batch in enumerate(loader):
            images = batch['image'].to(device)
            value, rmse = encode_intervention_features(images, views, model.extract_img_feature, auxiliary_encoder)
            features.append(value.cpu().numpy())
            distortions.append(rmse.cpu().numpy())
            targets.extend(int(v) for v in batch['label'])
            identifiers = batch.get('global_index', batch.get('sample_id'))
            if identifiers is None:
                raise ValueError('Cache requires stable dataset-global sample IDs.')
            sample_ids.extend(int(v) for v in identifiers)
            original_count += len(images)
            auxiliary_count += 0 if views.source == 'repeat' else 2 * len(images)
            if index == 0 or (index + 1) % 20 == 0:
                print(f'Base cache: {index+1}/{len(loader)} batches', flush=True)
    if not features:
        raise ValueError('Empty base training cache.')
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError('Duplicate global sample IDs in base cache.')
    arrays = {'features': np.concatenate(features).astype(np.float32),
              'labels': np.asarray(targets, dtype=np.int64),
              'sample_id': np.asarray(sample_ids, dtype=np.int64),
              'pixel_rmse': np.concatenate(distortions).astype(np.float32)}
    if set(arrays['labels'].tolist()) != set(base_ids):
        raise ValueError('Cache does not cover every original base class.')
    return arrays, {'original': original_count, 'auxiliary': auxiliary_count,
                    'benchmark_query': 0}, len(dataset)


def write_cache(output, arrays, metadata, base_support, overwrite=False):
    """Metadata is committed last, so an interrupted cache cannot certify completion."""
    import numpy as np
    paths = output_paths(output)
    if not overwrite and any(p.exists() or p.with_suffix(p.suffix + '.tmp').exists() for p in paths):
        raise FileExistsError('Output/sidecar already exists; choose a new output or explicitly use --overwrite.')
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    cache_path, meta_path, support_path = paths
    cache_temp = cache_path.with_suffix('.npz.tmp')
    support_temp = support_path.with_suffix('.json.tmp')
    meta_temp = meta_path.with_suffix('.json.tmp')
    # In overwrite mode invalidate the previous certificate before changing data.
    if overwrite and meta_path.exists():
        meta_path.unlink()
    with cache_temp.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    support_temp.write_text(json.dumps(base_support, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    metadata = {**metadata, 'status': 'completed', 'cache_sha256': sha256(cache_temp),
                'base_support_sha256': sha256(support_temp), 'array_shapes': {k: list(v.shape) for k, v in arrays.items()},
                'array_dtypes': {k: str(v.dtype) for k, v in arrays.items()}}
    metadata['metadata_payload_sha256'] = canonical_sha(metadata)
    meta_temp.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    cache_temp.replace(cache_path)
    support_temp.replace(support_path)
    meta_temp.replace(meta_path)
    return metadata


def collect_base_text(manager, model, cfg):
    """Encode original BiMC text banks for base class names only."""
    import numpy as np
    import torch
    ids = [int(c) for c in manager.class_index_in_task[0]]
    if ids != list(range(len(ids))):
        raise ValueError('Original BiMC text cache requires contiguous original base classes.')
    names = [str(manager.class_names[c]) for c in ids]
    with torch.no_grad():
        text, _ = model.inference_text_feature(names, manager.template, 0)
        descriptions, targets, proto, _ = model.inference_all_description_feature(names, cfg.DATASET.GPT_PATH, 0)
    return {'text_features': text.detach().float().cpu().numpy().astype(np.float32),
            'description_features': descriptions.detach().float().cpu().numpy().astype(np.float32),
            'description_targets': targets.detach().cpu().numpy().astype(np.int64),
            'description_proto': proto.detach().float().cpu().numpy().astype(np.float32)}


def load_cache(path):
    """Load an authenticated base cache without pickle or a model dependency."""
    import numpy as np
    cache_path, meta_path, support_path = output_paths(path)
    metadata = json.loads(meta_path.read_text(encoding='utf-8'))
    expected = metadata.pop('metadata_payload_sha256', None)
    if canonical_sha(metadata) != expected or metadata.get('status') != 'completed' or metadata.get('data_source') != 'base_training_only':
        raise ValueError('Invalid/incomplete base-cache metadata.')
    metadata['metadata_payload_sha256'] = expected
    if sha256(cache_path) != metadata['cache_sha256'] or sha256(support_path) != metadata['base_support_sha256']:
        raise ValueError('Cache/support file checksum mismatch.')
    names = ['features', 'labels', 'sample_id', 'pixel_rmse']
    if metadata.get('has_text_features'):
        names.extend(('text_features', 'description_features', 'description_targets', 'description_proto'))
    with np.load(cache_path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in names}
    n = len(arrays['labels'])
    features = arrays['features']
    if features.dtype != np.float32 or features.ndim != 3 or features.shape[:2] != (n, 3) or features.shape[2] < 1:
        raise ValueError('Expected float32 features [N,3,D].')
    if not n or arrays['labels'].shape != (n,) or arrays['sample_id'].shape != (n,) or arrays['pixel_rmse'].shape != (n, 2):
        raise ValueError('Cache arrays have incompatible shapes.')
    if arrays['labels'].dtype != np.int64 or arrays['sample_id'].dtype != np.int64 or len(np.unique(arrays['sample_id'])) != n:
        raise ValueError('Labels/sample IDs must be int64, with unique sample IDs.')
    if not all(np.isfinite(v).all() for v in arrays.values()) or (arrays['pixel_rmse'] < 0).any():
        raise ValueError('Invalid cache numerical values.')
    if not np.allclose(np.linalg.norm(features, axis=-1), 1, atol=1e-4):
        raise ValueError('Cached feature rows are not unit normalized.')
    if set(arrays['labels'].tolist()) != set(metadata['base_class_ids']):
        raise ValueError('Cache labels do not match original base class IDs.')
    if metadata.get('has_text_features'):
        c, d = len(metadata['base_class_ids']), features.shape[-1]
        if arrays['text_features'].shape != (c, d) or arrays['description_proto'].shape != (c, d):
            raise ValueError('Text prototypes must match base-class/feature dimensions.')
        desc, targets = arrays['description_features'], arrays['description_targets']
        if desc.ndim != 2 or desc.shape[1] != d or targets.shape != (len(desc),) or targets.dtype != np.int64:
            raise ValueError('Invalid description bank shapes or targets.')
        if set(targets.tolist()) != set(metadata['base_class_ids']):
            raise ValueError('Description bank must cover exactly original base classes.')
        for name in ('text_features', 'description_features', 'description_proto'):
            if arrays[name].dtype != np.float32 or not np.allclose(np.linalg.norm(arrays[name], axis=-1), 1, atol=1e-4):
                raise ValueError('Text features must be float32 unit vectors.')
    return arrays, metadata


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        print(json.dumps(vars(args), indent=2))
        return 0
    paths = output_paths(args.output)
    if not args.overwrite and any(p.exists() or p.with_suffix(p.suffix + '.tmp').exists() for p in paths):
        print('Refusing existing cache/sidecars; use a new --output.', file=sys.stderr)
        return 2
    import numpy as np
    import torch
    from utils.util import set_gpu, set_seed
    from datasets.data_manager import DatasetManager
    from models.bimc import BiMC
    from models.safd_interventions import NestedInterventionViews
    set_gpu(args.gpu)
    set_seed(args.seed)
    start = time.perf_counter()
    cfg = build_cfg(args)
    manager = DatasetManager(cfg)
    model = BiMC(cfg, manager.template, args.device).eval().requires_grad_(False)
    views = NestedInterventionViews(args.source, cfg.TRAINER.BiMC.FREQUENCY)
    arrays, counts, full_count = collect_base_features(manager, model, views,
        batch_size=args.batch_size, workers=args.workers, device=args.device,
        limit_per_class=args.limit_per_class)
    if not args.image_only:
        arrays.update(collect_base_text(manager, model, cfg))
    ids = [int(c) for c in manager.class_index_in_task[0]]
    manifest = manager._support_manifest()
    base_support = {key: manifest[key] for key in ('version', 'dataset', 'train_size', 'train_targets_sha256', 'train_samples_sha256', 'seed')}
    base_support.update(data_source='base_training_only', class_ids=ids,
                        sample_ids=arrays['sample_id'].tolist(), targets=arrays['labels'].tolist())
    files = ('tools/cache_safd_base_features.py', 'models/safd_interventions.py', 'models/safd.py',
             'models/frequency.py', 'models/bimc.py', 'datasets/data_manager.py', 'main.py',
             f'configs/datasets/{args.dataset}.yaml', 'configs/trainers/bimc_frequency_guidance.yaml')
    metadata = {'schema_version': 1, 'dataset': args.dataset, 'data_source': 'base_training_only',
                'seed': args.seed, 'base_class_ids': ids, 'base_class_names': [str(manager.class_names[c]) for c in ids],
                'class_names': [str(manager.class_names[c]) for c in ids], 'template': list(manager.template),
                'feature_dim': int(arrays['features'].shape[-1]),
                'base_sample_ids_sorted_sha256': hashlib.sha256(json.dumps(sorted(int(i) for i in arrays['sample_id'])).encode()).hexdigest(),
                'source': args.source, 'view_specification': views.specification(),
                'clip_weight_sha256': getattr(model.clip_model, 'clip_weight_sha256', None),
                'config_yaml': cfg.dump(), 'config_sha256': hashlib.sha256(cfg.dump().encode()).hexdigest(),
                'source_files_sha256': {name: sha256(ROOT/name) for name in files},
                'input_provenance': {key: manifest[key] for key in ('dataset', 'train_size', 'train_targets_sha256', 'train_samples_sha256')},
                'source_support_manifest': args.support_manifest,
                'source_support_manifest_sha256': sha256(args.support_manifest) if args.support_manifest else None,
                'has_text_features': not args.image_only,
                'ordinary_description_path': cfg.DATASET.GPT_PATH if not args.image_only else None,
                'ordinary_description_sha256': sha256(cfg.DATASET.GPT_PATH) if not args.image_only else None,
                'text_template': list(manager.template),
                'support_transform': repr(manager.test_transform), 'encoding_counts': counts,
                'sample_count': len(arrays['labels']), 'full_base_sample_count': full_count,
                'limit_per_class': args.limit_per_class, 'complete_base_bank': args.limit_per_class is None,
                'purpose': 'smoke_test_subset' if args.limit_per_class is not None else 'base_training_feature_cache',
                'class_counts': {str(c): int((arrays['labels']==c).sum()) for c in ids},
                'mean_pixel_rmse': arrays['pixel_rmse'].mean(0).tolist(),
                'wall_seconds': time.perf_counter()-start,
                'benchmark_evaluation_performed': False, 'new_gradient_training_performed': False}
    written = write_cache(args.output, arrays, metadata, base_support, args.overwrite)
    load_cache(args.output)
    print('Base cache:', paths[0], 'samples=', written['sample_count'], 'shape=', written['array_shapes']['features'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
