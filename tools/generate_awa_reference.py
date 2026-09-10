#!/usr/bin/env python3
"""Generate independently executed official-body FP32-B AWA fixtures."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import torch
from awa_reference import AdaptiveWindowAttention, verify_sources


def sha(path):
    return hashlib.file_digest(path.open('rb'),'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    selection=parser.add_mutually_exclusive_group()
    selection.add_argument('--quick',action='store_true')
    selection.add_argument('--video-boundaries',action='store_true',
        help='Four new 128x80 short-video grid cases: latent T=3/5, 20 heads, 58 text tokens')
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit('Reference output directory must be empty')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    source=verify_sources()
    shapes=[((1,2,3),2,5),((2,3,4),2,5),((5,7,11),2,7),
            ((3,19,29),1,7),((9,5,7),2,1),((1,3,128),1,9),
            ((1,128,3),1,9),((1,2,3),20,58),((4,1,1),2,3),((33,2,3),1,7)]
    if args.quick: shapes=shapes[:2]
    if args.video_boundaries: shapes=[((3,5,8),20,58),((5,5,8),20,58)]
    cases=[]
    for index,(grid,heads,nt) in enumerate(shapes):
        for shifted in [False,True]:
            name=f't{grid[0]}-h{grid[1]}-w{grid[2]}-heads{heads}-text{nt}-'+('shifted' if shifted else 'regular')
            folder=args.output/name;folder.mkdir()
            g=torch.Generator().manual_seed(60117+index)
            norm=0.8+0.4*torch.rand(4,128,generator=g)
            video=torch.randn(*grid,3*heads*128,generator=g)
            text=torch.randn(nt,3*heads*128,generator=g)
            model=AdaptiveWindowAttention(heads,shifted).eval()
            model.set_norm_weights(norm)
            with torch.inference_mode(): reference_v,reference_t=model(video,text)
            def tensor(filename,t):
                path=folder/filename
                path.write_bytes(t.detach().contiguous().cpu().numpy().astype('<f4').tobytes())
                return {'path':filename,'sha256':sha(path),'shape':list(t.shape),'dtype':'f32le'}
            params=folder/'awa.ncnn.param'
            params.write_text('7767517\n3 4\nInput input0 0 1 in0\nInput input1 0 1 in1\n'
                             f'SeedVR2AWA awa 2 2 in0 in1 out0 out1 0={heads} 1={int(shifted)} 2=1.000000e-05 3=1\n')
            case={'schema_version':'awa-case-v1','case_id':name,'reference_profile':'FP32-B',
                  'precision':'fp32','grid':list(grid),'heads':heads,'text_length':nt,'shifted':shifted,
                  'seed':60117+index,'model_param':{'path':params.name,'sha256':sha(params)},
                  'model_bin':tensor('awa.ncnn.bin',norm),
                  'video_qkv':tensor('video-qkv.f32',video),'text_qkv':tensor('text-qkv.f32',text),
                  'reference_video':tensor('reference-video.f32',reference_v),
                  'reference_text':tensor('reference-text.f32',reference_t),
                  'export_status':'HANDWRITTEN_OPERATOR_GRAPH; PNNX_NOT_YET_APPLIED'}
            (folder/'case.json').write_text(json.dumps(case,indent=2)+'\n')
            cases.append({'path':f'{name}/case.json','sha256':sha(folder/'case.json')})
            print(name,flush=True)
    suite={'schema_version':'awa-reference-suite-v1','scope':'Synthetic projected QKV and RMS weights; official attention body FP32-B; no SeedVR2 checkpoint',
           'official_sources':source,'reference_adapter_sha256':sha(Path(__file__).with_name('awa_reference.py')),
           'generator_sha256':sha(Path(__file__)),'environment':{k:importlib.metadata.version(k) for k in ['torch','einops','rotary-embedding-torch','beartype']},
           'oracle':'official NaSwinAttention.forward with documented FP32-B adapters',
           'model_verified':False,'cases':cases}
    (args.output/'suite.json').write_text(json.dumps(suite,indent=2)+'\n')


if __name__=='__main__':main()
