"""Frozen-text heuristics mapping actual attribute keywords to CLIP channels.

This is FDR-inspired, not an exact reproduction. ``attention_value`` attributes
only the last text attention path: EOS attention times token values, through the
attention output projection, a fixed final-LayerNorm decomposition, and CLIP's
text projection. Residual and MLP paths are not assigned to attribute tokens.
``deletion`` instead measures a keyword deletion's final text-feature change.
Neither heuristic identifies an image region or establishes causal semantics.

Importing this module requires only the standard library. Numerical dependencies
are loaded when building or validating the numerical sidecar.
"""

import math
import re

from utils.attribute_descriptions import GROUPS, PATTERNS
from utils.frequency_descriptions import normalize_name


METHODS = ('attention_value', 'deletion')
ATTENTION_NOTE = ('Last text attention/value path projected through attention.out_proj, '
                  'fixed final LayerNorm centering/scaling/weight and text_projection; '
                  'residual/MLP paths are not attributed. A heuristic, not exact FDR reproduction.')


def keyword_spans(text, class_name, group, tokenizer, patterns=None):
    """Locate lexical content keywords, excluding every full class-name span.

    BPE positions include the leading SOS at position zero. An ambiguous prefix
    segmentation is rejected rather than silently grounding a different token.
    The selected keyword is never the full sentence or the species name.
    """
    patterns = PATTERNS if patterns is None else patterns
    if not isinstance(text, str) or not text.strip() or group not in patterns:
        raise ValueError('A nonempty attribute sentence and known group are required.')
    words = normalize_name(class_name).split()
    if not words:
        raise ValueError('Class name must be nonempty.')
    class_pattern = re.compile(r'\b' + r'[\s_-]+'.join(re.escape(w) for w in words) + r'\b', re.I)
    excluded = [(m.start(), m.end()) for m in class_pattern.finditer(text)]
    encoded = list(tokenizer.encode(text))
    accepted, rejected = [], []
    for match in patterns[group].finditer(text):
        start, end = match.span()
        if any(start < b and end > a for a, b in excluded):
            rejected.append({'keyword':match.group(), 'char_span':[start,end], 'reason':'class_name'})
            continue
        token_start = len(tokenizer.encode(text[:start]))
        token_end = len(tokenizer.encode(text[:end]))
        expected = list(tokenizer.encode(text[start:end]))
        if not expected or encoded[token_start:token_end] != expected:
            rejected.append({'keyword':match.group(), 'char_span':[start,end], 'reason':'ambiguous_bpe_alignment'})
            continue
        accepted.append({'keyword':match.group(), 'char_span':[start,end],
                         'token_span':[token_start+1,token_end+1], 'token_ids':expected})
    return accepted, rejected


def text_token_contributions(model, tokens):
    """Return unchanged encode_text output and projected last-attention paths.

    The output contribution tensor has shape [B,L,D] in the same output-channel
    coordinates as CLIP text/image embeddings. Its rows are *not* claimed to sum
    to the complete final text embedding, which also has unassigned paths.
    Temporary read-only hooks are always removed, even after an exception.
    """
    import torch
    from torch.nn import functional as F
    if model.training:
        raise ValueError('Text grounding requires an evaluation-mode frozen encoder.')
    if tokens.ndim != 2 or not tokens.shape[0] or tokens.shape[1] != model.context_length:
        raise ValueError('Token input must have shape [B,context_length].')
    block = model.transformer.resblocks[-1]
    attention = block.attn
    if (attention.in_proj_weight is None or attention.bias_k is not None
            or attention.bias_v is not None or attention.add_zero_attn):
        raise ValueError('Grounding supports the standard CLIP self-attention architecture only.')
    captured = {}
    hooks = [block.ln_1.register_forward_hook(lambda module, inputs, output: captured.update(normalized=output.detach())),
             block.register_forward_hook(lambda module, inputs, output: captured.update(final=output.detach()))]
    try:
        with torch.no_grad():
            encoded = model.encode_text(tokens)
    finally:
        for hook in hooks:
            hook.remove()
    x = captured['normalized']
    width, heads = x.shape[-1], attention.num_heads
    length, batch = x.shape[:2]
    if width % heads:
        raise ValueError('Attention width must divide into complete heads.')
    with torch.no_grad():
        q, k, v = F.linear(x, attention.in_proj_weight, attention.in_proj_bias).chunk(3, dim=-1)
        head_width = width // heads
        def split(value):
            return value.permute(1,0,2).reshape(batch,length,heads,head_width).permute(0,2,1,3)
        q, k, v = (split(value) for value in (q,k,v))
        eos = tokens.argmax(dim=-1)
        index = torch.arange(batch, device=tokens.device)
        query = q[index, :, eos, :]
        scores = (query.float()[:, :, None, :] * k.float()).sum(-1) / math.sqrt(head_width)
        valid = torch.arange(length, device=tokens.device)[None] <= eos[:,None]
        scores = scores.masked_fill(~valid[:,None,:], -torch.inf)
        # CLIP's mask is causal. Reject any different mask rather than silently
        # assigning contributions from a different attention operation.
        mask = block.attn_mask
        if mask is not None:
            expected = torch.full((length,length), -torch.inf, device=tokens.device).triu(1)
            if mask.shape != expected.shape or not torch.equal(mask.float(), expected):
                raise ValueError('Grounding requires the standard CLIP causal attention mask.')
        weights = scores.softmax(-1)
        weighted = (weights[...,None] * v.float()).permute(0,2,1,3).reshape(batch,length,width)
        projected = F.linear(weighted, attention.out_proj.weight.float(), None)
        # The final LayerNorm is decomposed at the *actual* final EOS input.
        # This places the selected attention path in the final output channels;
        # it does not assign the final MLP or residual paths to those tokens.
        final = captured['final'].permute(1,0,2)[index,eos].float()
        inv_std = (final.var(-1, unbiased=False, keepdim=True)+model.ln_final.eps).rsqrt()
        projected = (projected-projected.mean(-1,keepdim=True))*inv_std[:,None]
        if model.ln_final.elementwise_affine:
            projected = projected*model.ln_final.weight.float()
        projected = projected @ model.text_projection.float()
    return encoded.detach(), projected.detach()


def build_grounding(manifest, clip_model, tokenize, tokenizer, topq=50,
                    method='attention_value', batch_size=128, *, group_names=GROUPS,
                    patterns=None, max_sentences=2, include_text=False):
    """Return numerical channel scores/masks and a JSON-serializable audit.

    Every available group averages its valid actual sentence/keyword scores.
    Available groups select exactly min(topq,D) channels; unknown groups stay
    completely zero and are omitted by downstream measurements.
    """
    import numpy as np
    import torch
    from torch.nn import functional as F
    if isinstance(topq,bool) or not isinstance(topq,int) or topq < 1:
        raise ValueError('topq must be a positive integer.')
    if isinstance(batch_size,bool) or not isinstance(batch_size,int) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer.')
    if method not in METHODS:
        raise ValueError('Unknown grounding method.')
    if clip_model.training:
        raise ValueError('Grounding requires an evaluation-mode encoder.')
    group_names = tuple(group_names)
    if (not group_names or len(set(group_names)) != len(group_names)
            or any(not isinstance(g,str) or not g for g in group_names)
            or not isinstance(manifest,dict) or manifest.get('group_order') != list(group_names)):
        raise ValueError('Grounding requires a matching ordered attribute manifest.')
    if isinstance(max_sentences,bool) or not isinstance(max_sentences,int) or max_sentences < 1:
        raise ValueError('max_sentences must be positive.')
    classes = manifest.get('classes')
    if (not isinstance(classes,list) or not classes
            or any(not isinstance(c,dict) or not isinstance(c.get('name'),str) for c in classes)):
        raise ValueError('Invalid attribute class provenance.')
    names = [c['name'] for c in classes]
    if len(set(names)) != len(names):
        raise ValueError('Attribute class names must be unique.')
    dimension = int(clip_model.text_projection.shape[1])
    scores = np.zeros((len(classes),len(group_names),dimension),dtype=np.float32)
    text_sum = np.zeros_like(scores)
    available = np.zeros(scores.shape[:2],dtype=np.bool_)
    rows, records = [], []
    for c, record in enumerate(classes):
        groups = record.get('groups')
        if not isinstance(groups,dict) or set(groups) != set(group_names):
            raise ValueError('Class must contain all declared attribute group lists.')
        class_record = {'name':record['name'],'groups':{}}
        records.append(class_record)
        for g, group in enumerate(group_names):
            sentences = groups[group]
            if not isinstance(sentences,list) or len(sentences) > max_sentences:
                raise ValueError('Too many recorded sentences per attribute group.')
            audit = {'available':False,'reason':'unknown_attribute' if not sentences else 'no_groundable_keywords',
                     'sentences':[],'keyword_count':0}
            class_record['groups'][group] = audit
            for sentence in sentences:
                if not isinstance(sentence,dict) or not isinstance(sentence.get('text'),str):
                    raise ValueError('Attribute provenance sentence must contain text.')
                text = sentence['text']
                keywords, rejected = keyword_spans(text,record['name'],group,tokenizer,patterns)
                detail = {'text':text,'source_index':sentence.get('source_index'),
                          'keywords':keywords,'rejected_keywords':rejected,
                          'used':False,'reason':'pending' if keywords else 'no_content_keyword'}
                audit['sentences'].append(detail)
                if keywords:
                    rows.append({'class':c,'group':g,'text':text,'keywords':keywords,'audit':detail})
    device = next(clip_model.parameters()).device
    counts = np.zeros(available.shape,dtype=np.int64)
    with torch.no_grad():
        for start in range(0,len(rows),batch_size):
            batch = rows[start:start+batch_size]
            tokens = tokenize([row['text'] for row in batch], context_length=clip_model.context_length,
                              truncate=False).to(device)
            for row, token_row in zip(batch,tokens):
                original = tokenizer.encode(row['text'])
                eos = int(token_row.argmax())
                if eos != len(original)+1 or token_row[1:eos].tolist() != list(original):
                    raise ValueError('Tokenizer/BPE provenance disagrees with encoded CLIP tokens.')
            if method == 'attention_value':
                encoded, contribution = text_token_contributions(clip_model,tokens)
                eos = tokens.argmax(-1)
                valid = torch.arange(tokens.shape[1],device=device)[None] <= eos[:,None]
                contribution = contribution.masked_fill(~valid[...,None], -torch.inf)
                proportions = contribution.softmax(dim=1)
                vectors = []
                for index,row in enumerate(batch):
                    positions = sorted({p for keyword in row['keywords'] for p in range(*keyword['token_span'])})
                    vectors.append(proportions[index,positions].mean(0))
                vectors = torch.stack(vectors)
            else:
                encoded = clip_model.encode_text(tokens)
                original = F.normalize(encoded.float(),dim=-1)
                edited = []
                for row in batch:
                    text = row['text']
                    for a,b in sorted((k['char_span'] for k in row['keywords']),reverse=True):
                        text = text[:a]+' '+text[b:]
                    edited.append(text)
                    row['audit']['deleted_text'] = text
                altered = tokenize(edited,context_length=clip_model.context_length,truncate=False).to(device)
                vectors = (original-F.normalize(clip_model.encode_text(altered).float(),dim=-1)).abs()
            if not torch.isfinite(vectors).all():
                raise ValueError('Grounding produced non-finite channel scores.')
            vectors = vectors.cpu().numpy().astype(np.float32)
            embeddings = F.normalize(encoded.float(),dim=-1).cpu().numpy()
            for row,vector,embedding in zip(batch,vectors,embeddings):
                scale = float(vector.sum())
                if scale <= 1e-20:
                    row['audit']['reason'] = 'zero_grounding_response'
                    continue
                c,g = row['class'],row['group']
                scores[c,g] += vector/scale
                text_sum[c,g] += embedding
                counts[c,g] += 1
                row['audit'].update(used=True,reason=None)
    masks = np.zeros_like(scores,dtype=np.bool_)
    for c in range(len(classes)):
        for g,group in enumerate(group_names):
            audit = records[c]['groups'][group]
            if counts[c,g]:
                if include_text and np.linalg.norm(text_sum[c,g]) <= 1e-12:
                    scores[c,g] = 0
                    audit.update(reason='cancelled_text_mean')
                    for sentence in audit['sentences']:
                        if sentence['used']:
                            sentence.update(used=False,reason='cancelled_text_mean')
                    continue
                scores[c,g] /= counts[c,g]
                # Stable tie breaking: lower channel index first.
                indices = np.argsort(-scores[c,g],kind='stable')[:min(topq,dimension)]
                masks[c,g,indices] = True
                available[c,g] = True
                audit.update(available=True,reason=None,keyword_count=sum(len(s['keywords']) for s in audit['sentences'] if s['used']))
    audit = {'schema_version':1,'group_order':list(group_names),'method':method,
             'topq':topq,'effective_topq':min(topq,dimension),'classes':records,
             'class_count':len(classes),'feature_dimension':dimension,
             'group_available_counts':dict(zip(group_names,available.sum(0).tolist())),
             'no_image_encoding':True,'new_gradient_training_performed':False,
             'grounding_note':ATTENTION_NOTE if method == 'attention_value' else
                'Absolute final unit text-feature change after deleting actual content keywords; no gradient optimization.',
             'coverage_note':'All original valid attribute slots are considered; no ordinary-budget clipping or fabricated fillers.'}
    result = {'channel_score':scores,'channel_mask':masks,'group_available':available}
    if include_text:
        result['attribute_text'] = np.where(available[...,None],
            text_sum/np.maximum(np.linalg.norm(text_sum,axis=-1,keepdims=True),1e-12),0).astype(np.float32)
        audit['text_pooling'] = 'unit_mean_of_exactly_used_grounding_sentences'
    return result,audit
