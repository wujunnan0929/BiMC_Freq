"""Cache three frequency-semantic attributes for FDR-inspired residual transfer.

Reuses an authenticated base visual cache. Encodes only cleaned static text;
does not read images, call an LLM, train CLIP, or claim Fourier selectivity.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from tools.cache_attribute_grounding import FREQUENCY_FORMAT, load_grounding, write_grounding
from tools.cache_safd_base_features import sha256, output_paths
from utils.attribute_grounding import METHODS, build_grounding
from utils.frequency_attributes import BANDS, PATTERNS, build_frequency_manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',required=True)
    parser.add_argument('--descriptions',default='description/cub200_frequency_descriptions_deepseek.json')
    parser.add_argument('--output',required=True)
    parser.add_argument('--sentences-per-band',type=int,default=2)
    parser.add_argument('--method',choices=METHODS,default='attention_value')
    parser.add_argument('--topq',type=int,default=50)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--dry-run',action='store_true')
    args = parser.parse_args(argv)
    if min(args.sentences_per_band,args.topq,args.batch_size) < 1:
        parser.error('Sentence budget, topq and batch size must be positive.')
    if any(Path(p).suffix.lower()!='.npz' for p in (args.cache,args.output)):
        parser.error('--cache and --output must end in .npz.')
    if Path(args.cache).resolve() == Path(args.output).resolve():
        parser.error('Output must differ from the original visual cache.')
    return args


def build(payload, source, model, tokenize, tokenizer, *, sentences_per_band=2,
          method='attention_value', topq=50, batch_size=128):
    selection = build_frequency_manifest(payload,source['base_class_names'],sentences_per_band)
    arrays,audit = build_grounding(selection,model,tokenize,tokenizer,topq=topq,
        method=method,batch_size=batch_size,group_names=BANDS,patterns=PATTERNS,
        max_sentences=sentences_per_band,include_text=True)
    return arrays,selection,audit


def metadata_for(source, selection, audit, description_sha):
    return {'schema_version':1,'format':FREQUENCY_FORMAT,'data_source':'base_training_only',
        'source_cache_sha256':source['cache_sha256'],
        'source_metadata_sha256':source['metadata_payload_sha256'],
        'clip_weight_sha256':source['clip_weight_sha256'],
        'base_class_ids':source['base_class_ids'],'base_class_names':source['base_class_names'],
        'group_order':list(BANDS),'method':audit['method'],'topq':audit['topq'],
        'effective_topq':audit['effective_topq'],'grounding_manifest':audit,
        'frequency_manifest':selection,'description_sha256':description_sha,
        'no_image_encoding':True,'image_forward_count':0,'new_gradient_training_performed':False,
        'visual_frequency_grounding_performed':False,
        'frequency_semantics':'spatial_scale_text_hypothesis_not_measured_spectral_band'}


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        print(json.dumps({**vars(args),'group_order':list(BANDS),'image_forward_count':0,
            'frequency_semantics':'text_scale_hypothesis'},indent=2))
        return 0
    try:
        if any(p.exists() or p.with_suffix(p.suffix+'.tmp').exists() for p in output_paths(args.output)):
            raise FileExistsError('Frequency sidecar exists; use a fresh --output.')
        from tools.cache_multilayer_base_features import load_cache
        _,source = load_cache(args.cache,require_complete=True)
        description_path = Path(args.descriptions)
        payload = json.loads(description_path.read_text(encoding='utf-8'))
        # Fail malformed text before loading weights. No image-guided filtering.
        build_frequency_manifest(payload,source['base_class_names'],args.sentences_per_band)
        from yacs.config import CfgNode
        from models.bimc import load_clip_to_cpu
        from models.clip import clip
        from utils.util import set_gpu
        set_gpu(args.gpu)
        model = load_clip_to_cpu(CfgNode.load_cfg(source['config_yaml'])).float().to(args.device).eval().requires_grad_(False)
        if not source.get('clip_weight_sha256') or model.clip_weight_sha256 != source['clip_weight_sha256']:
            raise ValueError('Frozen CLIP weights differ from the visual cache.')
        started = time.perf_counter()
        arrays,selection,audit = build(payload,source,model,clip.tokenize,clip._tokenizer,
            sentences_per_band=args.sentences_per_band,method=args.method,
            topq=args.topq,batch_size=args.batch_size)
        metadata = metadata_for(source,selection,audit,sha256(description_path))
        sources = ('tools/cache_frequency_attribute_grounding.py','utils/frequency_attributes.py',
                   'utils/frequency_descriptions.py','utils/attribute_grounding.py',
                   'tools/cache_attribute_grounding.py','models/clip/model.py',
                   'models/clip/clip.py','models/clip/simple_tokenizer.py','models/bimc.py')
        metadata.update(description_path=str(description_path.resolve()),
            source_files_sha256={name:sha256(ROOT/name) for name in sources},
            wall_seconds=time.perf_counter()-started)
        write_grounding(args.output,arrays,metadata)
        load_grounding(args.output,source)
        print('Frequency attribute cache:',Path(args.output).resolve())
        print('Available groups:',audit['group_available_counts'],'image forwards: 0')
        print('These are semantic-scale channels; physical frequency selectivity is not established.')
        return 0
    except (OSError,ValueError,RuntimeError,KeyError,TypeError) as error:
        print('Frequency attribute grounding failed: '+str(error),file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
