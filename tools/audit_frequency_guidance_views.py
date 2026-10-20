"""Audit SAFD/FFT responses on original base training images, without fitting."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=('cub200','cifar100','miniimagenet'),default='cub200')
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--backend',choices=('safd','fft'),default='safd')
    parser.add_argument('--samples',type=int,default=8)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--output')
    args=parser.parse_args(argv)
    if min(args.samples,args.batch_size,args.seed)<1:
        parser.error('samples, batch-size and seed must be positive.')
    from main import setup_cfg
    from utils.util import set_gpu,set_seed
    from datasets.data_manager import DatasetManager
    from models.bimc import BiMC
    import numpy as np
    import torch
    import torch.nn.functional as F
    set_gpu(args.gpu)
    set_seed(args.seed)
    cfg=setup_cfg(str(ROOT/f'configs/datasets/{args.dataset}.yaml'),
                  str(ROOT/'configs/trainers/bimc_frequency_guidance.yaml'),[
                      'DATASET.ROOT',args.data_root,'DEVICE.DEVICE_NAME',args.device,
                      'SEED',str(args.seed),'DATASET.SUPPORT_SEED',str(args.seed),
                      'TRAINER.BiMC.GUIDANCE.MODE','visual',
                      'TRAINER.BiMC.FREQUENCY.BACKEND',args.backend])
    manager=DatasetManager(cfg)
    dataset=manager.get_dataset(0,source='train',mode='test')
    classes=np.unique(dataset.labels)
    if args.samples>len(classes):
        parser.error('For this class-stratified audit, samples must not exceed the base class count.')
    chosen=classes[np.linspace(0,len(classes)-1,args.samples).round().astype(int)]
    rng=np.random.RandomState(args.seed)
    rows=[int(rng.choice(np.flatnonzero(np.asarray(dataset.labels)==c))) for c in chosen]
    model=BiMC(cfg,manager.template,args.device).eval()
    identifiers,responses,distortions=[],{'blur':[],'texture':[]},[]
    with torch.no_grad():
        for start in range(0,len(rows),args.batch_size):
            records=[dataset[i] for i in rows[start:start+args.batch_size]]
            images=torch.stack([r['image'] for r in records]).to(args.device)
            identifiers.extend(int(r['global_index']) for r in records)
            mean,std=model.frequency_decomposer.mean.to(images),model.frequency_decomposer.std.to(images)
            pixels=images*std+mean
            base=model.extract_frequency_img_feature(images)
            views=model.frequency_decomposer(images)
            distortions.extend(((views-images[:,None])*std[:,None]).square().mean((2,3,4)).sqrt().cpu().tolist())
            blur=F.avg_pool2d(F.pad(pixels,(3,3,3,3),mode='reflect'),7,stride=1)
            h,w=pixels.shape[-2:]
            checker=((torch.arange(h,device=args.device)[:,None]+torch.arange(w,device=args.device)[None,:])%2)*2-1
            texture=(pixels+.03*checker[None,None]).clamp(0,1)
            for name,perturbed in [('blur',blur),('texture',texture)]:
                encoded=model.extract_frequency_img_feature((perturbed-mean)/std)
                responses[name].extend((1-F.cosine_similarity(base,encoded,dim=-1)).cpu().tolist())
    payload={'purpose':'base_training_view_intervention_diagnostic_not_benchmark',
             'dataset':args.dataset,'backend':args.backend,'seed':args.seed,
             'class_ids':chosen.tolist(),'sample_ids':identifiers,
             'clip_weight_sha256':getattr(model.clip_model,'clip_weight_sha256',None),
             'bands':['low','middle','high'],
             'intervention_definition':{'blur':'RGB reflect-pad 7x7 average filter',
                                        'texture':'RGB +/-0.03 deterministic checkerboard, clipped to [0,1]'},
             'cosine_response':responses,
             'mean_cosine_response':{k:np.mean(v,axis=0).tolist() for k,v in responses.items()},
             'pixel_rmse_from_original':distortions,
             'mean_pixel_rmse_from_original':np.mean(distortions,axis=0).tolist(),
             'encoding_counts':model.image_encoding_counts,
             'config_yaml':cfg.dump(),
             'interpretation':'Diagnostic only; adaptive SAFD scales are not guaranteed physical Fourier bands.'}
    output=Path(args.output or ROOT/f'outputs/frequency_guidance_view_audit/{args.dataset}_{args.backend}.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
    print('View audit:',output)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
