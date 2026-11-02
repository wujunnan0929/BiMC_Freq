"""Cache frozen-CLIP features of every train/test image for FGDR (run on the GPU machine).

Run from the repository root, e.g.::

  python tools/fgdr_cache_features.py --dataset cub200 --data-root ../datasets \
      --output outputs/fgdr/cache/cub200 --seeds 1 2 3 4 5 --shots 5 1

One original forward per image also yields patch-token DCT band energies (hooks on
the selected ViT blocks). Blur views and the non-frequency control views add one
forward each. Writes ``<output>.train.npz``, ``<output>.test.npz`` and
``<output>.json`` (see fgdr/cache.py). Support draws are read from
DatasetManager itself for every (shot, seed), so they match the engine exactly.
"""

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
AUG_NAMES = ('hflip', 'gray', 'contrast0.7')
DISALLOWED = ('FREQUENCY.ENABLED', 'FREQUENCY.ROUTER.ENABLED', 'GUIDANCE.ENABLED', 'CONSENSUS.ENABLED',
              'UNCERTAINTY.ENABLED', 'DUAL_FREQUENCY.ENABLED', 'RESIDUAL.ENABLED', 'SAFD_METRIC.ENABLED',
              'PROTOTYPE_ESTIMATOR.ENABLED', 'PROTOTYPE_COMPLETION.ENABLED', 'COMPETITIVE_FUSION.ENABLED')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', choices=('cub200', 'cifar100', 'miniimagenet'), required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True, help='Cache stem, e.g. outputs/fgdr/cache/cub200')
    p.add_argument('--trainer-config', default='configs/trainers/bimc.yaml')
    p.add_argument('--backbone', help='Override MODEL.BACKBONE.NAME, e.g. ViT-B/32 or ViT-L/14')
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    p.add_argument('--shots', type=int, nargs='+', default=[5, 1, 10])
    p.add_argument('--blur-sigmas', type=float, nargs='*', default=[1.0, 2.0, 4.0])
    p.add_argument('--no-aug', action='store_true', help='Skip the non-frequency control views')
    p.add_argument('--dct-layers', type=int, nargs='*', default=[8, 12],
                   help='One-based ViT blocks for patch-token DCT energies; empty list disables')
    p.add_argument('--no-dct-rand', action='store_true')
    p.add_argument('--splits', nargs='+', default=['train', 'test'], choices=('train', 'test'))
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', default='cuda')
    p.add_argument('--gpu', default='0')
    p.add_argument('--limit-per-class', type=int, help='Smoke test only')
    return p.parse_args(argv)


def build_cfg(a, seed, shot):
    from main import setup_cfg
    opts = ['DATASET.ROOT', a.data_root, 'DEVICE.DEVICE_NAME', a.device, 'DEVICE.GPU_ID', repr(a.gpu),
            'SEED', str(seed), 'DATASET.SUPPORT_SEED', str(seed), 'DATASET.SUPPORT_MANIFEST', '',
            'DATASET.NUM_INC_SHOT', str(shot), 'DATALOADER.NUM_WORKERS', str(a.workers),
            'TRAINER.BiMC.PREC', 'fp32', 'TRAINER.BiMC.USING_ENSEMBLE', 'True']
    if getattr(a, 'backbone', None):
        opts += ['MODEL.BACKBONE.NAME', a.backbone]
    cfg = setup_cfg(str(ROOT / f'configs/datasets/{a.dataset}.yaml'), str(ROOT / a.trainer_config), opts)
    for key in DISALLOWED:
        node = cfg.TRAINER.BiMC
        try:
            for part in key.split('.'):
                node = getattr(node, part)
        except AttributeError:
            continue
        if node is True:
            raise ValueError(f'TRAINER.BiMC.{key} must be disabled for a plain-BiMC feature cache')
    return cfg


# ----------------------------------------------------------------------------- image ops


def gaussian_blur(images, sigma):
    """Separable Gaussian blur with reflect padding.

    Blur is linear with unit-sum kernels and reflect padding, so blurring the
    CLIP-normalised tensor equals normalising the blurred pixels exactly.
    """
    import torch
    import torch.nn.functional as F
    radius = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-radius, radius + 1, device=images.device, dtype=images.dtype)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    c = images.shape[1]
    out = F.pad(images, (radius, radius, radius, radius), mode='reflect')
    out = F.conv2d(out, kernel.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(out, kernel.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def augment(images, name):
    import torch
    mean = torch.tensor(CLIP_MEAN, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    if name == 'hflip':
        return images.flip(-1)
    pixels = images * std + mean
    if name == 'gray':
        gray = 0.299 * pixels[:, 0:1] + 0.587 * pixels[:, 1:2] + 0.114 * pixels[:, 2:3]
        pixels = gray.repeat(1, 3, 1, 1)
    elif name == 'contrast0.7':
        centre = pixels.mean(dim=(1, 2, 3), keepdim=True)
        pixels = ((pixels - centre) * 0.7 + centre).clamp(0, 1)
    else:
        raise ValueError(name)
    return (pixels - mean) / std


def dct_matrix(n, device):
    import torch
    k = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    i = torch.arange(n, device=device, dtype=torch.float32)[None, :]
    m = torch.cos(math.pi * (2 * i + 1) * k / (2 * n))
    m[0] *= math.sqrt(1.0 / n)
    m[1:] *= math.sqrt(2.0 / n)
    return m


def band_masks(grid, device, random_seed=None):
    """Radial low/mid/high masks on the DCT grid (DC excluded), thresholds scaled from 14x14."""
    import torch
    u = torch.arange(grid, device=device, dtype=torch.float32)
    rho = torch.sqrt(u[:, None] ** 2 + u[None, :] ** 2)
    low_cut, mid_cut = 3.0 * grid / 14.0, 7.0 * grid / 14.0
    masks = torch.stack([(rho > 0) & (rho <= low_cut), (rho > low_cut) & (rho <= mid_cut), rho > mid_cut])
    if random_seed is None:
        return masks
    gen = torch.Generator().manual_seed(int(random_seed))
    flat = masks.reshape(3, -1).cpu()
    nonzero = torch.nonzero(flat.any(0)).flatten()
    perm = nonzero[torch.randperm(len(nonzero), generator=gen)]
    out = torch.zeros_like(flat)
    start = 0
    for b in range(3):
        size = int(flat[b].sum())
        out[b, perm[start:start + size]] = True
        start += size
    return out.reshape(3, grid, grid).to(device)


def band_energy(tokens, dct, masks):
    """tokens [B, G*G, W] -> RMS energy per band [B, 3, W]."""
    import torch
    b, n, w = tokens.shape
    g = int(round(math.sqrt(n)))
    x = tokens.float().transpose(1, 2).reshape(b, w, g, g)
    coeff = dct @ x @ dct.T
    power = coeff ** 2
    out = []
    for mask in masks:
        out.append((power * mask).sum(dim=(-1, -2)).div(mask.sum().clamp(min=1)).sqrt())
    return torch.stack(out, 1)


# ----------------------------------------------------------------------------- extraction


def encode_split(model, dataset, a, layers):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=False, num_workers=a.workers,
                        pin_memory=str(a.device).startswith('cuda'))
    visual = model.clip_model.visual
    captured, state = {}, {'on': False}
    hooks = []
    if layers:
        blocks = visual.transformer.resblocks
        for layer in layers:
            def hook(module, inputs, output, layer=layer):
                if state['on']:
                    captured[layer] = output.detach()
            hooks.append(blocks[layer - 1].register_forward_hook(hook))
    banks = {'final': [], 'blur': [], 'aug': [], 'dct': [], 'dct_rand': []}
    labels, ids = [], []
    dct = masks = rand_masks = None
    start = time.perf_counter()
    try:
        with torch.no_grad():
            for index, batch in enumerate(loader):
                images = batch['image'].to(a.device).float()
                state['on'] = bool(layers)
                final = model.clip_model.encode_image(images).float()
                state['on'] = False
                banks['final'].append(F.normalize(final, dim=-1).cpu().numpy().astype(np.float32))
                if layers:
                    energies, rand_energies = [], []
                    for layer in layers:
                        tokens = captured.pop(layer).permute(1, 0, 2)[:, 1:, :]
                        if dct is None:
                            grid = int(round(math.sqrt(tokens.shape[1])))
                            dct = dct_matrix(grid, tokens.device)
                            masks = band_masks(grid, tokens.device).float()
                            rand_masks = band_masks(grid, tokens.device, random_seed=20261102).float()
                        energies.append(band_energy(tokens, dct, masks))
                        if not a.no_dct_rand:
                            rand_energies.append(band_energy(tokens, dct, rand_masks))
                    banks['dct'].append(torch.stack(energies, 1).cpu().numpy().astype(np.float16))
                    if rand_energies:
                        banks['dct_rand'].append(torch.stack(rand_energies, 1).cpu().numpy().astype(np.float16))
                if a.blur_sigmas:
                    views = [F.normalize(model.clip_model.encode_image(gaussian_blur(images, s)).float(), dim=-1)
                             for s in a.blur_sigmas]
                    banks['blur'].append(torch.stack(views, 1).cpu().numpy().astype(np.float16))
                if not a.no_aug:
                    views = [F.normalize(model.clip_model.encode_image(augment(images, n)).float(), dim=-1)
                             for n in AUG_NAMES]
                    banks['aug'].append(torch.stack(views, 1).cpu().numpy().astype(np.float16))
                labels.extend(int(v) for v in batch['label'])
                ids.extend(int(v) for v in batch['global_index'])
                if index == 0 or (index + 1) % 20 == 0:
                    rate = (index + 1) / (time.perf_counter() - start)
                    print(f'  batch {index + 1}/{len(loader)} ({rate:.2f} batch/s)', flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    out = {k: np.concatenate(v) for k, v in banks.items() if v}
    out['labels'] = np.asarray(labels, dtype=np.int64)
    out['sample_id'] = np.asarray(ids, dtype=np.int64)
    return out


def limited_rows(targets, limit, keep=()):
    import numpy as np
    rows = set(int(i) for i in keep)
    for c in np.unique(targets):
        rows.update(int(i) for i in np.flatnonzero(targets == c)[:limit])
    return np.asarray(sorted(rows), dtype=np.int64)


def main(argv=None):
    a = parse_args(argv)
    import numpy as np
    import torch
    from datasets.data_manager import DatasetManager, TaskDataset
    from fgdr import cache as fc
    from fgdr.protocol import replica_support_indices, support_key
    from models.bimc import BiMC
    from utils.util import set_gpu, set_seed

    set_gpu(a.gpu)
    set_seed(a.seeds[0])
    t0 = time.perf_counter()
    cfg = build_cfg(a, a.seeds[0], a.shots[0])
    manager = DatasetManager(cfg)
    names = [str(n) for n in manager.class_names]
    sessions = [[int(c) for c in s] for s in manager.class_index_in_task]

    supports, replica_ok = {}, {}
    for shot in a.shots:
        for seed in a.seeds:
            m = DatasetManager(build_cfg(a, seed, shot))
            m._ensure_support_protocol()
            draw = [[int(i) for i in s] for s in m._session_support_indices]
            supports[support_key(shot, seed)] = draw
            replica = replica_support_indices(m.train_targets, m.class_index_in_task,
                                              m.num_base_shot, m.num_inc_shot, m.support_seed)
            replica_ok[support_key(shot, seed)] = all(list(map(int, r)) == d for r, d in zip(replica, draw))
    print('Support draws:', {k: v for k, v in replica_ok.items()})

    model = BiMC(cfg, manager.template, a.device).eval().requires_grad_(False)
    layers = list(a.dct_layers or [])
    meta = {'dataset': a.dataset, 'class_names': names, 'sessions': sessions, 'supports': supports,
            'support_replica_match': replica_ok, 'backbone': cfg.MODEL.BACKBONE.NAME,
            'bimc_defaults': {'beta': float(cfg.DATASET.BETA), 'ensemble_alpha': float(cfg.DATASET.ENSEMBLE_ALPHA),
                              'lambda_t': float(cfg.TRAINER.BiMC.LAMBDA_T), 'lambda_i': float(cfg.TRAINER.BiMC.LAMBDA_I),
                              'tau': float(cfg.TRAINER.BiMC.TAU), 'gamma_base': float(cfg.TRAINER.BiMC.GAMMA_BASE),
                              'gamma_inc': float(cfg.TRAINER.BiMC.GAMMA_INC), 'ensemble': True,
                              'vision_calibration': bool(cfg.TRAINER.BiMC.VISION_CALIBRATION),
                              'text_calibration': bool(cfg.TRAINER.BiMC.TEXT_CALIBRATION),
                              'cov_temperature': 512.0},
            'blur_sigmas': list(a.blur_sigmas), 'aug_names': [] if a.no_aug else list(AUG_NAMES),
            'dct_layers': layers, 'dct_bands': ['low', 'mid', 'high'], 'dct_random_seed': 20261102,
            'template': list(manager.template), 'gpt_path': cfg.DATASET.GPT_PATH,
            'gpt_sha256': fc.file_sha256(cfg.DATASET.GPT_PATH),
            'config_sha256': hashlib.sha256(cfg.dump().encode()).hexdigest(),
            'smoke_limit_per_class': a.limit_per_class, 'created': datetime.now(timezone.utc).isoformat(),
            'sha256': {}}

    with torch.no_grad():
        text, _ = model.inference_text_feature(names, manager.template, 0)
        desc, desc_targets, desc_proto, _ = model.inference_all_description_feature(names, cfg.DATASET.GPT_PATH, 0)
    text_arrays = {'text': torch.nn.functional.normalize(text.float(), dim=-1).cpu().numpy().astype(np.float32),
                   'desc': torch.nn.functional.normalize(desc.float(), dim=-1).cpu().numpy().astype(np.float32),
                   'desc_targets': desc_targets.cpu().numpy().astype(np.int64),
                   'desc_proto': torch.nn.functional.normalize(desc_proto.float(), dim=-1).cpu().numpy().astype(np.float32)}

    for split in a.splits:
        data = manager.train_data if split == 'train' else manager.test_data
        targets = np.asarray(manager.train_targets if split == 'train' else manager.test_targets)
        if a.limit_per_class:
            keep = sorted({i for draw in supports.values() for s in draw for i in s}) if split == 'train' else ()
            rows = limited_rows(targets, a.limit_per_class, keep)
        else:
            rows = np.arange(len(targets))
        images = [data[i] for i in rows] if isinstance(data, list) else data[rows]
        dataset = TaskDataset(images, targets[rows], manager.test_transform, sample_indices=rows)
        print(f'Encoding {split}: {len(rows)} images', flush=True)
        arrays = encode_split(model, dataset, a, layers)
        if split == 'train':
            arrays.update(text_arrays)
        meta['sha256'][split] = fc.save_split(a.output, split, arrays)
    meta['wall_seconds'] = time.perf_counter() - t0
    fc.save_meta(a.output, meta)
    print('Wrote cache', fc.cache_paths(a.output))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
