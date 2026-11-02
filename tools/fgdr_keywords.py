"""Build the FDR-style keyword->channel library (frozen CLIP text encoder, GPU or CPU).

  python tools/fgdr_keywords.py --dataset cub200 --data-root ../datasets \
      --output outputs/fgdr/cache/cub200_keywords --m 8 --q 37

Keywords are tf-idf content words of the BiMC descriptions (fgdr/keywords.py).
Every (class, keyword) pair gets two top-q channel masks:
  final  last-layer attention path projected through out_proj, ln_final and
         text_projection into the joint embedding coordinates
         (utils/attribute_grounding.text_token_contributions)
  fdr    a faithful port of FDR's KeywordGrounding (value space of the last text
         block, no output projection), whose indices FDR applies directly to the
         joint embedding -- kept to test whether the coordinate mismatch matters.
The pipeline unions masks per keyword over the *current* base classes only.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', choices=('cub200', 'cifar100', 'miniimagenet'), required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--trainer-config', default='configs/trainers/bimc.yaml')
    p.add_argument('--m', type=int, default=8, help='keywords per class')
    p.add_argument('--q', type=int, default=37, help='channels per keyword mask (FDR script: 37)')
    p.add_argument('--min-classes', type=int, default=2)
    p.add_argument('--max-class-frac', type=float, default=0.5)
    p.add_argument('--device', default='cuda')
    p.add_argument('--gpu', default='0')
    return p.parse_args(argv)


def find_span(tokenizer, sentence, word):
    tokens = tokenizer.encode(sentence)
    target = tokenizer.encode(word)
    for i in range(len(tokens) - len(target) + 1):
        if tokens[i:i + len(target)] == target:
            return list(range(i, i + len(target)))
    return None


def fdr_contributions(clip_model, tokens):
    """Port of FDR models/Network.py KeywordGrounding + analyze_keyword_impact."""
    import torch
    x = clip_model.token_embedding(tokens).float() + clip_model.positional_embedding.float()
    x = x.permute(1, 0, 2).type(clip_model.dtype)
    for layer in clip_model.transformer.resblocks[:-1]:
        x = layer(x)
    x = x.float()
    eos = int(tokens[0].argmax())
    eos_feature, token_features = x[eos, 0], x[1:eos, 0]
    weight = clip_model.transformer.resblocks[-1].attn.in_proj_weight.float()
    qw, kw, vw = weight.chunk(3, dim=0)
    q, k, v = eos_feature @ qw.T, token_features @ kw.T, token_features @ vw.T
    attn = torch.softmax((q @ k.T) / (q.shape[-1] ** 0.5), dim=-1)
    return torch.softmax(v * attn[:, None], dim=0)          # [n_tokens, width]


def topq(vector, q):
    import torch
    mask = torch.zeros_like(vector, dtype=torch.bool)
    mask[torch.topk(vector.abs(), q).indices] = True
    return mask


def main(argv=None):
    a = parse_args(argv)
    import numpy as np
    import torch
    from datasets.data_manager import DatasetManager
    from fgdr.keywords import extract_keywords
    from models.bimc import BiMC
    from models.clip import clip
    from models.clip.simple_tokenizer import SimpleTokenizer
    from tools.fgdr_cache_features import build_cfg
    from utils.attribute_grounding import text_token_contributions
    from utils.util import set_gpu

    set_gpu(a.gpu)
    a.workers, a.trainer_config = 0, a.trainer_config
    cfg = build_cfg(a, 1, 5)
    manager = DatasetManager(cfg)
    names = [str(n) for n in manager.class_names]
    base_ids = [int(c) for c in manager.class_index_in_task[0]]
    raw = json.loads(Path(cfg.DATASET.GPT_PATH).read_text(encoding='utf-8'))
    lookup = {k.lower().replace('_', ' '): v for k, v in raw.items()}
    descriptions = [lookup[n.lower().replace('_', ' ')] for n in names]
    vocab, class_keywords, surface = extract_keywords(descriptions, names, base_ids, a.m,
                                                      a.min_classes, a.max_class_frac)
    model = BiMC(cfg, manager.template, a.device).eval().requires_grad_(False)
    clip_model = model.clip_model.eval()
    tokenizer = SimpleTokenizer()
    pair_class, pair_kw, final_masks, fdr_masks, failures = [], [], [], [], []
    with torch.no_grad():
        for cid in range(len(names)):
            for kw in class_keywords[cid]:
                sentence, word = surface[cid][kw]
                span = find_span(tokenizer, sentence, word)
                if span is None:
                    failures.append({'class': cid, 'keyword': vocab[kw], 'reason': 'bpe_span'})
                    continue
                tokens = clip.tokenize([sentence], truncate=True).to(a.device)
                eos = int(tokens[0].argmax())
                if span[-1] + 1 >= eos:
                    failures.append({'class': cid, 'keyword': vocab[kw], 'reason': 'truncated'})
                    continue
                _, projected = text_token_contributions(clip_model, tokens)
                final_vec = projected[0, [i + 1 for i in span]].sum(0)
                fdr_vec = fdr_contributions(clip_model, tokens)[span].sum(0)
                pair_class.append(cid)
                pair_kw.append(kw)
                final_masks.append(topq(final_vec, a.q).cpu().numpy())
                fdr_masks.append(topq(fdr_vec, a.q).cpu().numpy())
    stem = str(a.output)
    Path(stem).parent.mkdir(parents=True, exist_ok=True)
    np.savez(stem + '.npz', pair_class=np.asarray(pair_class, np.int64), pair_keyword=np.asarray(pair_kw, np.int64),
             mask_final=np.stack(final_masks), mask_fdr=np.stack(fdr_masks))
    meta = {'dataset': a.dataset, 'vocab': vocab, 'class_names': names,
            'class_keywords': {str(c): [k for k in v] for c, v in class_keywords.items()},
            'settings': {'m': a.m, 'q': a.q, 'min_classes': a.min_classes, 'max_class_frac': a.max_class_frac},
            'base_class_ids': base_ids, 'pairs': len(pair_class), 'failures': failures,
            'readable': {names[c]: [vocab[k] for k in class_keywords[c]] for c in range(len(names))}}
    Path(stem + '.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'{len(pair_class)} grounded pairs, {len(vocab)} vocabulary keywords, {len(failures)} failures')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
