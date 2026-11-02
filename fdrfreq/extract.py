"""Build a feature bank with the project's frozen CLIP (GPU, torch). Run once per dataset.

    python -m fdrfreq.extract --dataset cub200 --data-root ../datasets \
        --out banks/cub200_vitb16 --sigmas 1 2 4 --aug flip gray \
        --dct-layers 6 9 12 --dct-splits train test --seeds 1 2 3 4 5 --keywords

What is encoded (all with the BiMC test transform, fp32, CLIP frozen):
  * every train and test image: final CLIP embedding of the original image,
    of Gaussian-blurred copies (sigma in pixels at 224 x 224) and of the
    augmentation controls (horizontal flip, grayscale);
  * optional ViT patch tokens after the chosen blocks, reduced on the fly to
    radial / random DCT band energies (no token is written to disk);
  * the BiMC text side for all classes (template ensemble, description bank);
  * optional keyword grounding scores for every (class, content word) pair,
    both in final CLIP coordinates (via utils.attribute_grounding) and with
    a re-implementation of FDR's KeywordGrounding (value-space coordinates);
  * the engine's support indices for each requested seed.

Blurring is applied to the normalised tensor; a normalised Gaussian kernel
with reflect padding commutes with the per-channel affine normalisation, so
this equals blurring the RGB image. ``--limit-per-class`` makes a smoke bank
(not valid for the paper: its row ids differ from the engine's).
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fdrfreq.bank import finalize_bank, _sha256  # noqa: E402
from fdrfreq.dctband import dct_matrix, radial_bands, random_bands, band_energies, RADIAL_EDGES  # noqa: E402
from fdrfreq.keywords import candidate_words, tokenize_words  # noqa: E402
from fdrfreq.protocol import task_classes, replica_support_indices  # noqa: E402

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
RANDOM_BAND_SEED = 20261102


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', choices=('cub200', 'cifar100', 'miniimagenet'), required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--backbone', default='ViT-B/16')
    p.add_argument('--sigmas', type=float, nargs='*', default=[1, 2, 4])
    p.add_argument('--aug', nargs='*', default=['flip', 'gray'], choices=('flip', 'gray'))
    p.add_argument('--dct-layers', type=int, nargs='*', default=[6, 9, 12])
    p.add_argument('--dct-splits', nargs='*', default=['train', 'test'], choices=('train', 'test'))
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    p.add_argument('--keywords', action='store_true')
    p.add_argument('--standard-manifest', help='Support manifest (DatasetManager format) for the standard '
                   'CEC/TOPIC session split; stored as seed 0.')
    p.add_argument('--standard-manifest-seed', type=int, default=1,
                   help='The seed recorded inside --standard-manifest.')
    p.add_argument('--max-words', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--gpu', default='0')
    p.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    p.add_argument('--limit-per-class', type=int)
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args(argv)


def build_cfg(a, seed, manifest=''):
    from main import setup_cfg
    b = 'TRAINER.BiMC.'
    options = ['DATASET.ROOT', a.data_root, 'DEVICE.DEVICE_NAME', a.device,
               'DEVICE.GPU_ID', repr(a.gpu), 'SEED', str(seed), 'DATASET.SUPPORT_SEED', str(seed),
               'DATASET.SUPPORT_MANIFEST', manifest or '', 'DATALOADER.NUM_WORKERS', str(a.workers),
               'MODEL.BACKBONE.NAME', a.backbone, b + 'PREC', 'fp32']
    for key in ('FREQUENCY.ENABLED', 'FREQUENCY.ROUTER.ENABLED', 'FREQUENCY.USE_EXPLICIT_DESCRIPTIONS',
                'GUIDANCE.ENABLED', 'CONSENSUS.ENABLED', 'UNCERTAINTY.ENABLED',
                'DUAL_FREQUENCY.ENABLED', 'RESIDUAL.ENABLED', 'SAFD_METRIC.ENABLED',
                'PROTOTYPE_ESTIMATOR.ENABLED', 'PROTOTYPE_COMPLETION.ENABLED', 'COMPETITIVE_FUSION.ENABLED'):
        options.extend((b + key, 'False'))
    return setup_cfg(str(ROOT / f'configs/datasets/{a.dataset}.yaml'),
                     str(ROOT / 'configs/trainers/bimc_frequency_guidance.yaml'), options)


# ------------------------------------------------------------------ vision
def encode_final_and_patches(visual, x, layers):
    """Final CLIP image embedding plus patch tokens after the given (1-based) blocks."""
    import torch
    x = visual.conv1(x)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                           dtype=x.dtype, device=x.device)
    x = torch.cat((cls, x), dim=1)
    x = visual.ln_pre(x + visual.positional_embedding.to(x.dtype))
    x = x.permute(1, 0, 2)
    patches = {}
    for i, block in enumerate(visual.transformer.resblocks, 1):
        x = block(x)
        if i in layers:
            patches[i] = x[1:].permute(1, 0, 2)  # [B, n*n, W]
    final = visual.ln_post(x.permute(1, 0, 2)[:, 0, :])
    if visual.proj is not None:
        final = final @ visual.proj
    return final, patches


def gaussian(x, sigma):
    from torchvision.transforms import functional as TF
    k = 2 * int(math.ceil(3 * sigma)) + 1
    return TF.gaussian_blur(x, [k, k], [float(sigma), float(sigma)])


def grayscale(x):
    import torch
    mean = torch.tensor(MEAN, device=x.device, dtype=x.dtype)[None, :, None, None]
    std = torch.tensor(STD, device=x.device, dtype=x.dtype)[None, :, None, None]
    rgb = x * std + mean
    g = (0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2])[:, None].expand(-1, 3, -1, -1)
    return (g - mean) / std


def torch_band_energies(tokens, D, bands):
    import torch
    B, L, W = tokens.shape
    n = D.shape[0]
    T = tokens.reshape(B, n, n, W).float()
    C = torch.einsum('uh,bhwc,vw->buvc', D, T, D)
    E = C ** 2
    return torch.stack([torch.sqrt((E * m[None, :, :, None]).sum(dim=(1, 2)) / m.sum()) for m in bands], dim=1)


def encode_split(model, dataset, out_dir, split, a, layers, device):
    import torch
    import torch.nn.functional as F
    from numpy.lib.format import open_memmap
    from torch.utils.data import DataLoader
    n = len(dataset)
    loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=False, num_workers=a.workers,
                        pin_memory=device.startswith('cuda'))
    visual = model.clip_model.visual
    dim = int(visual.output_dim)
    width = int(visual.conv1.out_channels)
    grid = int(visual.input_resolution // visual.conv1.kernel_size[0])
    views = ['orig'] + [f'blur{s:g}' for s in a.sigmas] + list(a.aug)
    feats = {v: open_memmap(out_dir / f'{split}_feat_{v}.npy', mode='w+', dtype=np.float16,
                            shape=(n, dim)) for v in views}
    dct_on = bool(split in a.dct_splits and layers)
    Dm = dct_matrix(grid)
    rb, xb = radial_bands(grid, RADIAL_EDGES), random_bands(grid, RANDOM_BAND_SEED, RADIAL_EDGES)
    Dt = torch.tensor(Dm, dtype=torch.float32, device=device)
    rbt = [torch.tensor(m, dtype=torch.float32, device=device) for m in rb]
    xbt = [torch.tensor(m, dtype=torch.float32, device=device) for m in xb]
    dct = {}
    if dct_on:
        for L in layers:
            for kind in ('radial', 'random'):
                dct[(L, kind)] = open_memmap(out_dir / f'{split}_dct_L{L}_{kind}.npy', mode='w+',
                                             dtype=np.float16, shape=(n, 3, width))
    labels = np.zeros(n, dtype=np.int64)
    seen_rows = np.zeros(n, dtype=bool)
    dtype = model.clip_model.dtype
    start = 0
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            x = batch['image'].to(device).type(dtype)
            rows = np.asarray(batch['idx'], dtype=np.int64)
            final, patches = encode_final_and_patches(visual, x, set(layers) if dct_on else set())
            if bi == 0:
                ref = model.clip_model.encode_image(x).float()
                if not torch.allclose(F.normalize(ref, dim=-1), F.normalize(final.float(), dim=-1), atol=1e-4):
                    raise RuntimeError('custom forward differs from encode_image')
            out = {'orig': final}
            for s in a.sigmas:
                out[f'blur{s:g}'] = model.clip_model.encode_image(gaussian(x, s))
            if 'flip' in a.aug:
                out['flip'] = model.clip_model.encode_image(torch.flip(x, dims=[3]))
            if 'gray' in a.aug:
                out['gray'] = model.clip_model.encode_image(grayscale(x))
            for v, val in out.items():
                feats[v][rows] = F.normalize(val.float(), dim=-1).cpu().numpy().astype(np.float16)
            for L in (layers if dct_on else []):
                tok = patches[L]
                er = torch_band_energies(tok, Dt, rbt)
                ex = torch_band_energies(tok, Dt, xbt)
                if bi == 0:
                    chk = band_energies(tok[:2].float().cpu().numpy(), Dm, rb)
                    if not np.allclose(chk, er[:2].cpu().numpy(), rtol=1e-3, atol=1e-4):
                        raise RuntimeError('torch DCT band energies disagree with numpy reference')
                dct[(L, 'radial')][rows] = er.cpu().numpy().astype(np.float16)
                dct[(L, 'random')][rows] = ex.cpu().numpy().astype(np.float16)
            labels[rows] = np.asarray(batch['label'], dtype=np.int64)
            seen_rows[rows] = True
            start += len(rows)
            if bi == 0 or (bi + 1) % 50 == 0:
                print(f'[{split}] {bi + 1}/{len(loader)} batches, {time.time() - t0:.0f}s', flush=True)
    if not seen_rows.all():
        raise RuntimeError(f'{split}: {(~seen_rows).sum()} rows were never written')
    for m in list(feats.values()) + list(dct.values()):
        m.flush()
    np.save(out_dir / f'{split}_labels.npy', labels)
    return {'views': views, 'dct': dct_on, 'width': width, 'grid': grid, 'seconds': time.time() - t0}


# -------------------------------------------------------------------- text
def keyword_grounding(model, names, gpt_path, max_words, device):
    """(class, word) scores in final CLIP coordinates and in FDR value coordinates."""
    import torch
    import models.clip.clip as clip
    from models.clip.simple_tokenizer import SimpleTokenizer
    from utils.attribute_grounding import text_token_contributions
    cm = model.clip_model
    tok = SimpleTokenizer()
    data = json.loads(Path(gpt_path).read_text(encoding='utf-8'))
    data = {k.lower().replace('_', ' '): v for k, v in data.items()}
    last = cm.transformer.resblocks[-1]
    q_w, k_w, v_w = last.attn.in_proj_weight.float().chunk(3, dim=0)
    width = q_w.shape[1]
    fdr_ok = width == cm.text_projection.shape[1]
    vocab, pairs = {}, []
    for c, name in enumerate(names):
        sentences = data[name.lower().replace('_', ' ')]
        cands = candidate_words(sentences, name, max_words=max_words)
        if not cands:
            continue
        tokens = clip.tokenize(sentences, truncate=True).to(device)
        with torch.no_grad():
            _, contrib = text_token_contributions(cm, tokens)       # [S,L,D] final coords
            x = cm.token_embedding(tokens).type(cm.dtype) + cm.positional_embedding.type(cm.dtype)
            x = x.permute(1, 0, 2)
            for blk in cm.transformer.resblocks[:-1]:
                x = blk(x)
            x = x.permute(1, 0, 2).float()                            # [S,L,W] (FDR: no ln_1)
        eos = tokens.argmax(dim=-1)
        word_ids = {w: [t for t in tok.encode(w)] for w, _ in cands}
        acc_final = {w: [] for w, _ in cands}
        acc_fdr = {w: [] for w, _ in cands}
        for s in range(len(sentences)):
            e = int(eos[s])
            ids = tokens[s, 1:e].tolist()
            B = contrib[s, 1:e].float()                                  # tokens x D
            Cf = torch.softmax(B, dim=0)
            if fdr_ok:
                feats = x[s, 1:e]
                qv = x[s, e] @ q_w.T
                kv = feats @ k_w.T
                vv = feats @ v_w.T
                att = torch.softmax((kv @ qv) / width ** 0.5, dim=0)
                Cd = torch.softmax(vv * att[:, None], dim=0)
            words_here = set(tokenize_words(sentences[s]))
            for w, _ in cands:
                if w not in words_here:
                    continue
                wid = word_ids[w]
                pos = [i for i in range(len(ids) - len(wid) + 1) if ids[i:i + len(wid)] == wid]
                if not pos:
                    continue
                span = list(range(pos[0], pos[0] + len(wid)))
                acc_final[w].append(Cf[span].sum(0))
                if fdr_ok:
                    acc_fdr[w].append(Cd[pos[0]])
        for w, tf in cands:
            if not acc_final[w]:
                continue
            wi = vocab.setdefault(w, len(vocab))
            sf = torch.stack(acc_final[w]).mean(0).cpu().numpy()
            sd = torch.stack(acc_fdr[w]).mean(0).cpu().numpy() if fdr_ok and acc_fdr[w] else np.zeros_like(sf)
            pairs.append((c, wi, tf, sf, sd))
    if not pairs:
        raise RuntimeError('no (class, word) pair could be grounded; check the description file')
    arrays = {'kw_pair_class': np.array([p[0] for p in pairs], np.int64),
              'kw_pair_word': np.array([p[1] for p in pairs], np.int64),
              'kw_pair_tf': np.array([p[2] for p in pairs], np.int64),
              'kw_score_final': np.stack([p[3] for p in pairs]).astype(np.float16),
              'kw_score_fdr': np.stack([p[4] for p in pairs]).astype(np.float16)}
    return arrays, sorted(vocab, key=vocab.get), fdr_ok


def main(argv=None):
    a = parse_args(argv)
    out = Path(a.out)
    arr_dir = out / 'arrays'
    if out.exists() and any(out.iterdir()) and not a.overwrite:
        print('Refusing to overwrite an existing bank; pass --overwrite.', file=sys.stderr)
        return 2
    arr_dir.mkdir(parents=True, exist_ok=True)
    import torch
    from utils.util import set_gpu, set_seed
    from datasets.data_manager import DatasetManager, TaskDataset
    from models.bimc import BiMC
    if a.device == 'cuda':
        set_gpu(a.gpu)
    set_seed(a.seeds[0])
    device = a.device
    cfg = build_cfg(a, a.seeds[0])
    manager = DatasetManager(cfg)
    model = BiMC(cfg, manager.template, device).eval().requires_grad_(False)
    names = [str(n) for n in manager.class_names]
    t_start = time.time()
    splits = {}
    for split, data, targets in (('train', manager.train_data, manager.train_targets),
                                 ('test', manager.test_data, manager.test_targets)):
        targets = np.asarray(targets)
        rows = np.arange(len(targets))
        if a.limit_per_class:
            rows = np.concatenate([np.flatnonzero(targets == c)[:a.limit_per_class]
                                   for c in range(len(names))])
        ds = TaskDataset(np.asarray(data, dtype=object)[rows] if not isinstance(data, np.ndarray)
                         else data[rows], targets[rows], manager.test_transform, None, names,
                         sample_indices=np.arange(len(rows)))
        splits[split] = encode_split(model, ds, arr_dir, split, a,
                                     sorted(set(a.dct_layers)), device)
    with torch.no_grad():
        text, _ = model.inference_text_feature(names, manager.template, 0)
        desc, desc_t, desc_p, _ = model.inference_all_description_feature(names, cfg.DATASET.GPT_PATH, 0)
    np.save(arr_dir / 'text_prompt.npy', text.float().cpu().numpy().astype(np.float32))
    np.save(arr_dir / 'desc_feat.npy', desc.float().cpu().numpy().astype(np.float32))
    np.save(arr_dir / 'desc_target.npy', desc_t.cpu().numpy().astype(np.int64))
    np.save(arr_dir / 'desc_proto.npy', desc_p.float().cpu().numpy().astype(np.float32))
    vocab, fdr_ok = [], None
    if a.keywords:
        kw_arrays, vocab, fdr_ok = keyword_grounding(model, names, cfg.DATASET.GPT_PATH, a.max_words, device)
        for k, v in kw_arrays.items():
            np.save(arr_dir / f'{k}.npy', v)
    train_labels = np.load(arr_dir / 'train_labels.npy')
    tasks = task_classes(len(names), cfg.DATASET.NUM_INIT_CLS, cfg.DATASET.NUM_INC_CLS)
    support_check = {}
    if not a.limit_per_class:
        if not np.array_equal(train_labels, np.asarray(manager.train_targets)):
            raise RuntimeError('train bank rows are not in the manager order')
        for seed in a.seeds:
            m = DatasetManager(build_cfg(a, seed))
            engine = [np.asarray(x, dtype=np.int64) for x in m._session_support_indices]
            replica = replica_support_indices(train_labels, tasks, cfg.DATASET.NUM_BASE_SHOT,
                                              cfg.DATASET.NUM_INC_SHOT, seed)
            support_check[seed] = all(np.array_equal(x, y) for x, y in zip(engine, replica))
            np.save(arr_dir / f'support_seed{seed}.npy', np.concatenate(engine))
            np.save(arr_dir / f'support_seed{seed}_offsets.npy',
                    np.cumsum([0] + [len(x) for x in engine]).astype(np.int64))
        if a.standard_manifest:
            if not Path(a.standard_manifest).is_file():
                raise FileNotFoundError(a.standard_manifest)
            m = DatasetManager(build_cfg(a, a.standard_manifest_seed, a.standard_manifest))
            std = [np.asarray(x, dtype=np.int64) for x in m._session_support_indices]
            np.save(arr_dir / 'support_seed0.npy', np.concatenate(std))
            np.save(arr_dir / 'support_seed0_offsets.npy',
                    np.cumsum([0] + [len(x) for x in std]).astype(np.int64))
            support_check['standard_manifest'] = str(a.standard_manifest)
    sources = ['fdrfreq/extract.py', 'fdrfreq/dctband.py', 'fdrfreq/keywords.py', 'models/bimc.py',
               'models/clip/model.py', 'datasets/data_manager.py', 'utils/attribute_grounding.py',
               f'configs/datasets/{a.dataset}.yaml']
    meta = {'dataset': a.dataset, 'backbone': a.backbone, 'class_names': names,
            'num_init': int(cfg.DATASET.NUM_INIT_CLS), 'num_inc': int(cfg.DATASET.NUM_INC_CLS),
            'base_shot': int(cfg.DATASET.NUM_BASE_SHOT), 'inc_shot': int(cfg.DATASET.NUM_INC_SHOT),
            'beta': float(cfg.DATASET.BETA), 'ensemble_alpha': float(cfg.DATASET.ENSEMBLE_ALPHA),
            'template': list(manager.template), 'gpt_path': cfg.DATASET.GPT_PATH,
            'gpt_sha256': _sha256(cfg.DATASET.GPT_PATH), 'sigmas': list(a.sigmas), 'aug': list(a.aug),
            'dct_layers': sorted(set(a.dct_layers)), 'dct_splits': list(a.dct_splits),
            'radial_edges': list(RADIAL_EDGES), 'random_band_seed': RANDOM_BAND_SEED,
            'kw_vocab': vocab, 'kw_fdr_coordinates_available': fdr_ok,
            'support_replica_matches_engine': {str(k): v for k, v in support_check.items()},
            'smoke_subset': a.limit_per_class, 'splits': splits,
            'clip_weight_sha256': getattr(model.clip_model, 'clip_weight_sha256', None),
            'source_sha256': {s: _sha256(ROOT / s) for s in sources if (ROOT / s).exists()},
            'extract_seconds': time.time() - t_start,
            'args': {k: (v if not isinstance(v, Path) else str(v)) for k, v in vars(a).items()}}
    finalize_bank(out, meta)
    print('bank written to', out, '| support replica == engine:', support_check)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
