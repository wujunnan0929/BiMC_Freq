"""Cache FDR-inspired attribute channel masks using frozen text encoding only.

The existing authenticated native-layer visual cache is reused. No dataset
manager, image loader, benchmark images, gradient optimization or LLM call is
used. --dry-run is standard-library-only and never loads the source cache.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from tools.cache_safd_base_features import canonical_sha, output_paths, sha256
from utils.attribute_descriptions import GROUPS
from utils.attribute_grounding import METHODS, build_grounding

FORMAT = 'attribute_grounding_v1'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',required=True,help='Existing complete multilayer base cache .npz.')
    parser.add_argument('--output',required=True,help='New grounding .npz plus .json/.support.json sidecars.')
    parser.add_argument('--method',choices=METHODS,default='attention_value')
    parser.add_argument('--topq',type=int,default=50)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--overwrite',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args = parser.parse_args(argv)
    if min(args.topq,args.batch_size) < 1:
        parser.error('--topq and --batch-size must be positive integers.')
    if any(Path(p).suffix.lower() != '.npz' for p in (args.cache,args.output)):
        parser.error('--cache and --output must end in .npz.')
    if Path(args.cache).resolve() == Path(args.output).resolve():
        parser.error('Grounding output must differ from the source visual cache.')
    return args


def _validate_arrays(arrays,metadata,source_metadata=None):
    import numpy as np
    if set(arrays) != {'channel_score','channel_mask','group_available'}:
        raise ValueError('Grounding has missing or unexpected arrays.')
    score, mask, available = (arrays[k] for k in ('channel_score','channel_mask','group_available'))
    if not all(isinstance(v,np.ndarray) for v in (score,mask,available)):
        raise ValueError('Grounding numerical values must be explicit NumPy arrays.')
    classes = metadata.get('base_class_ids')
    names = metadata.get('base_class_names')
    if (metadata.get('format') != FORMAT or metadata.get('data_source') != 'base_training_only'
            or metadata.get('group_order') != list(GROUPS)
            or not isinstance(classes,list) or classes != list(range(len(classes))) or not classes
            or not isinstance(names,list) or len(names) != len(classes)
            or any(not isinstance(n,str) or not n for n in names) or len(set(names)) != len(names)):
        raise ValueError('Invalid grounding class/group identities or format.')
    topq = metadata.get('topq')
    if (isinstance(topq,bool) or not isinstance(topq,int) or topq < 1
            or metadata.get('method') not in METHODS or metadata.get('no_image_encoding') is not True
            or metadata.get('new_gradient_training_performed') is not False):
        raise ValueError('Grounding must record positive topq and frozen text-only provenance.')
    if (score.dtype != np.float32 or score.ndim != 3 or score.shape[:2] != (len(classes),4)
            or score.shape[2] < 1 or mask.shape != score.shape or mask.dtype != np.bool_
            or available.shape != score.shape[:2] or available.dtype != np.bool_
            or not np.isfinite(score).all() or (score < 0).any()):
        raise ValueError('Expected finite nonnegative float32 [C,4,D] scores and boolean masks.')
    if (np.any(mask[~available]) or np.any(score[~available] != 0)
            or not np.allclose(score[available].sum(-1),1.,atol=1e-5)
            or not np.all(mask.sum(-1)[available] == min(topq,score.shape[-1]))):
        raise ValueError('Grounding unknowns, score normalization or mask topq count disagree.')
    for c,g in zip(*np.where(available)):
        expected = np.zeros(score.shape[-1],dtype=np.bool_)
        expected[np.argsort(-score[c,g],kind='stable')[:min(topq,score.shape[-1])]] = True
        if not np.array_equal(mask[c,g],expected):
            raise ValueError('Grounding mask disagrees with stable topq channel scores.')
    manifest = metadata.get('grounding_manifest',{})
    if not isinstance(manifest,dict) or not isinstance(manifest.get('classes'),list):
        raise ValueError('Grounding requires structured keyword provenance.')
    if (any(not isinstance(r,dict) for r in manifest['classes'])
            or [r.get('name') for r in manifest['classes']] != names
            or manifest.get('method') != metadata['method'] or manifest.get('topq') != topq
            or manifest.get('group_order') != list(GROUPS)
            or manifest.get('feature_dimension') != score.shape[-1]
            or manifest.get('effective_topq') != min(topq,score.shape[-1])
            or metadata.get('effective_topq',min(topq,score.shape[-1])) != min(topq,score.shape[-1])):
        raise ValueError('Grounding keyword provenance differs from its numerical metadata.')
    for c,record in enumerate(manifest['classes']):
        if set(record.get('groups',{})) != set(GROUPS):
            raise ValueError('Grounding keyword provenance lacks attribute groups.')
        for g,group in enumerate(GROUPS):
            item=record['groups'][group]
            if item.get('available') is not bool(available[c,g]):
                raise ValueError('Grounding group availability differs from recorded keyword provenance.')
            used=[s for s in item.get('sentences',[]) if s.get('used')]
            if available[c,g] and (not used or not any(s.get('keywords') for s in used)):
                raise ValueError('Available grounding group has no used content keyword.')
    for key in ('source_cache_sha256','source_metadata_sha256','clip_weight_sha256'):
        if not isinstance(metadata.get(key),str) or not metadata[key]:
            raise ValueError('Grounding is missing source identity: '+key)
    if source_metadata is not None:
        checks={'source_cache_sha256':'cache_sha256','source_metadata_sha256':'metadata_payload_sha256',
                'clip_weight_sha256':'clip_weight_sha256','base_class_ids':'base_class_ids',
                'base_class_names':'base_class_names'}
        if any(metadata.get(target) != source_metadata.get(source) for target,source in checks.items()):
            raise ValueError('Grounding source cache, class identities or CLIP weight SHA changed.')
        shapes=source_metadata.get('array_shapes',{})
        expected_dimension=source_metadata.get('feature_dim') or shapes.get('final',[None,None])[-1]
        if expected_dimension is not None and score.shape[-1] != expected_dimension:
            raise ValueError('Grounding dimension differs from the source CLIP feature dimension.')
        originals=source_metadata.get('attribute_manifest',{}).get('classes',[])
        if len(originals)!=len(classes):
            raise ValueError('Source cache lacks matching attribute provenance.')
        for c,original in enumerate(originals):
            for g,group in enumerate(GROUPS):
                recorded=[s['text'] for s in manifest['classes'][c]['groups'][group]['sentences']]
                expected=[s['text'] for s in original['groups'][group]]
                if recorded != expected:
                    raise ValueError('Grounding sentences differ from source cache attribute sentences.')


def write_grounding(path,arrays,metadata,*,overwrite=False):
    """Commit numerical and provenance sidecars, with the certificate last."""
    import numpy as np
    _validate_arrays(arrays,metadata)
    paths=output_paths(path)
    if not overwrite and any(p.exists() or p.with_suffix(p.suffix+'.tmp').exists() for p in paths):
        raise FileExistsError('Grounding output/sidecars already exist; choose a fresh output.')
    cache_path,meta_path,support_path=paths
    for p in paths:
        p.parent.mkdir(parents=True,exist_ok=True)
    temporary=[p.with_suffix(p.suffix+'.tmp') for p in paths]
    if overwrite and meta_path.exists():
        meta_path.unlink()
    with temporary[0].open('wb') as handle:
        np.savez_compressed(handle,**arrays)
    source_identity={k:metadata[k] for k in ('source_cache_sha256','source_metadata_sha256',
                                           'clip_weight_sha256','base_class_ids','base_class_names','group_order')}
    temporary[2].write_text(json.dumps(source_identity,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    result={**metadata,'status':'completed','cache_sha256':sha256(temporary[0]),
            'grounding_sha256':sha256(temporary[0]),'base_support_sha256':sha256(temporary[2]),
            'array_shapes':{k:list(v.shape) for k,v in arrays.items()},
            'array_dtypes':{k:str(v.dtype) for k,v in arrays.items()}}
    result.pop('metadata_payload_sha256',None)
    result['metadata_payload_sha256']=canonical_sha(result)
    temporary[1].write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary[0].replace(cache_path)
    temporary[2].replace(support_path)
    temporary[1].replace(meta_path)
    return result


def load_grounding(path,source_metadata=None):
    """Read certified channel masks, never pickle; optionally bind to a cache."""
    import numpy as np
    cache_path,meta_path,support_path=output_paths(path)
    metadata=json.loads(meta_path.read_text(encoding='utf-8'))
    expected=metadata.pop('metadata_payload_sha256',None)
    if canonical_sha(metadata)!=expected or metadata.get('status')!='completed':
        raise ValueError('Invalid or incomplete grounding metadata certificate.')
    metadata['metadata_payload_sha256']=expected
    if (sha256(cache_path)!=metadata.get('cache_sha256')
            or metadata.get('grounding_sha256')!=metadata.get('cache_sha256')
            or sha256(support_path)!=metadata.get('base_support_sha256')):
        raise ValueError('Grounding NPZ/provenance checksum mismatch.')
    with np.load(cache_path,allow_pickle=False) as data:
        arrays={key:data[key] for key in data.files}
    _validate_arrays(arrays,metadata,source_metadata)
    if any(metadata.get('array_shapes',{}).get(k)!=list(v.shape)
           or metadata.get('array_dtypes',{}).get(k)!=str(v.dtype) for k,v in arrays.items()):
        raise ValueError('Grounding certified shapes/dtypes disagree.')
    source_identity=json.loads(support_path.read_text(encoding='utf-8'))
    if any(source_identity.get(k)!=metadata.get(k) for k in ('source_cache_sha256','source_metadata_sha256',
            'clip_weight_sha256','base_class_ids','base_class_names','group_order')):
        raise ValueError('Grounding source identity sidecar disagrees.')
    return arrays,metadata


def main(argv=None):
    args=parse_args(argv)
    if args.dry_run:
        print(json.dumps({**vars(args),'no_image_encoding':True,'new_gradient_training_performed':False},indent=2))
        return 0
    try:
        paths=output_paths(args.output)
        if not args.overwrite and any(p.exists() or p.with_suffix(p.suffix+'.tmp').exists() for p in paths):
            raise FileExistsError('Grounding output/sidecars already exist; choose a new --output.')
        from tools.cache_multilayer_base_features import load_cache
        _,source=load_cache(args.cache,require_complete=True)
        from yacs.config import CfgNode
        from models.bimc import load_clip_to_cpu
        from models.clip import clip
        from utils.util import set_gpu
        set_gpu(args.gpu)
        cfg=CfgNode.load_cfg(source['config_yaml'])
        model=load_clip_to_cpu(cfg).float().to(args.device).eval().requires_grad_(False)
        if not source.get('clip_weight_sha256') or model.clip_weight_sha256!=source['clip_weight_sha256']:
            raise ValueError('Frozen CLIP weights do not match the original visual cache.')
        started=time.perf_counter()
        arrays,audit=build_grounding(source['attribute_manifest'],model,clip.tokenize,clip._tokenizer,
                                    topq=args.topq,method=args.method,batch_size=args.batch_size)
        sources=('utils/attribute_grounding.py','utils/attribute_descriptions.py',
                 'tools/cache_attribute_grounding.py','models/clip/model.py','models/clip/simple_tokenizer.py',
                 'models/clip/clip.py','models/bimc.py')
        metadata={'schema_version':1,'format':FORMAT,'data_source':'base_training_only',
                  'source_cache_sha256':source['cache_sha256'],
                  'source_metadata_sha256':source['metadata_payload_sha256'],
                  'clip_weight_sha256':source['clip_weight_sha256'],
                  'base_class_ids':source['base_class_ids'],'base_class_names':source['base_class_names'],
                  'group_order':list(GROUPS),'method':args.method,'topq':args.topq,
                  'effective_topq':min(args.topq,arrays['channel_score'].shape[-1]),
                  'no_image_encoding':True,'image_forward_count':0,'new_gradient_training_performed':False,
                  'grounding_manifest':audit,'source_files_sha256':{name:sha256(ROOT/name) for name in sources},
                  'wall_seconds':time.perf_counter()-started}
        write_grounding(args.output,arrays,metadata,overwrite=args.overwrite)
        _,saved=load_grounding(args.output,source_metadata=source)
        print('Grounding sidecar:',Path(args.output).resolve())
        print('Method:',saved['method'],'available:',audit['group_available_counts'],'image forwards: 0')
        return 0
    except (OSError,ValueError,RuntimeError,KeyError) as error:
        print('Attribute grounding failed: '+str(error),file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
