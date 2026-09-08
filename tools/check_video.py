#!/usr/bin/env python3
"""Compare all 32 original DiT blocks and the temporal VAE for a native clip.

Uses unchanged pinned upstream bodies and shared raw noise. MP4 is a lossy
delivery format; numerical comparison is against retained pre-codec tensors.
"""
import argparse
import gc
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from export_vae_image import digest,tensor
from image_reference import make_dit,endpoint
from vae_reference import ROOT,make_reference,load_checkpoint,verify_sources
from pipeline_contract import reference_contract

ATOL,RTOL=1e-3,1e-3


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    run=json.loads((args.run/'run.json').read_text())
    if run['schema_version']!='seedvr2-video-run-v1' or run['status']!='SUCCEEDED' or digest(args.input)!=run['input']['sha256']:
        raise ValueError('Expected identified, successful native video run')
    for suffix in ('stdout','stderr'):
        log=args.run.parent/(args.run.name+'.'+suffix)
        if log.exists() and any(x in log.read_text().lower() for x in ('vuid-','validation error')):
            raise ValueError('Vulkan validation error in retained log')
    torch.set_num_threads(4);rows=[]
    def read(name):
        r=run['diagnostics'][name];path=args.run/r['path']
        if digest(path)!=r['sha256']: raise ValueError('Changed native tensor')
        return torch.from_numpy(np.fromfile(path,dtype='<f4').reshape(r['shape']).copy())
    def observe(name,expected):
        if name=='output-ada':
            tensor(args.output/(name+'.f32'),expected);return
        actual=read(name)
        if actual.shape!=expected.shape: raise ValueError(f'{name}: {actual.shape} != {expected.shape}')
        x,y=actual.double().numpy(),expected.detach().double().numpy();delta=np.abs(x-y)
        finite=bool(np.isfinite(x).all() and np.isfinite(y).all());violations=int(np.sum(delta>ATOL+RTOL*np.abs(y)))
        row=dict(stage=name,passed=finite and violations==0,finite=finite,violations=violations,
                 max_abs=float(delta.max()),rmse=float(np.sqrt(np.mean(delta**2))),reference=tensor(args.output/(name+'.f32'),expected))
        rows.append(row);(args.output/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(name,'PASS' if row['passed'] else 'FAIL',row['max_abs'],violations,flush=True)
    frames=run['clip']['decoded_frames'];padded=run['clip']['padded_frames'];lt=run['clip']['latent_frames']
    iw,ih=run['input']['width'],run['input']['height'];w,h=run['output']['width'],run['output']['height']
    raw=subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-threads','4','-i',str(args.input),
                        '-map','0:v:0','-frames:v',str(frames),'-pix_fmt','rgb24','-f','rawvideo','pipe:1'],capture_output=True,check=True,timeout=30).stdout
    pixels=np.frombuffer(raw,dtype=np.uint8).reshape(frames,ih,iw,3).copy()
    source=torch.from_numpy(pixels).permute(0,3,1,2).float()/255.
    scale=max(w,h)/max(iw,ih);rw,rh=round(iw*scale),round(ih*scale)
    resized=F.interpolate(source,size=(rh,rw),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
    oy,ox=round((rh-h)/2),round((rw-w)/2)
    prepared=(resized[:,:,oy:oy+h,ox:ox+w]-.5)/.5
    prepared=torch.cat((prepared,prepared[-1:].expand(padded-frames,-1,-1,-1)),0).permute(1,0,2,3).contiguous()
    observe('prepared',prepared)
    vae_state,lock=load_checkpoint(ROOT/'.cache/models/ema_vae.pth')
    encoder,memory=make_reference(vae_state,'encoder')
    with torch.inference_mode(),torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        posterior=encoder(prepared[None],memory_state=memory)[0]
    observe('posterior',posterior);del encoder;gc.collect()
    mean,logvar=posterior.chunk(2,0);conditioned=(mean+torch.exp(.5*logvar.clamp(-30,20))*read('posterior-noise'))*.9152
    observe('conditioned',conditioned);noise=read('noise')
    state_path=ROOT/'.cache/models/seedvr2_ema_3b.pth'
    if digest(state_path)!=next(x['lfs']['sha256'] for x in lock['files'] if x['rfilename']==state_path.name): raise ValueError('Changed DiT checkpoint')
    state=torch.load(state_path,weights_only=True,mmap=True,map_location='cpu')
    embedding=ROOT/'.cache/models/pos_emb.pt'
    if digest(embedding)!=next(x['lfs']['sha256'] for x in lock['files'] if x['rfilename']==embedding.name): raise ValueError('Changed text embedding')
    pos=torch.load(embedding,weights_only=True,map_location='cpu').float();model=make_dit(state,observe)
    lh,lw=h//8,w//8;cond=conditioned.permute(1,2,3,0)
    def predict(x,time):
        joined=torch.cat((x.permute(1,2,3,0),cond,torch.ones(lt,lh,lw,1)),-1)
        with torch.inference_mode(),torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            prediction=model(vid=joined.reshape(-1,33),txt=pos,vid_shape=torch.tensor([[lt,lh,lw]]),
                txt_shape=torch.tensor([[58]]),timestep=time,disable_cache=False).vid_sample
        velocity=prediction.reshape(lt,lh,lw,16).permute(3,0,1,2).contiguous();observe('velocity',velocity);return velocity
    with torch.inference_mode(): latent=endpoint(noise,predict)/.9152
    observe('latent',latent);del model,state;gc.collect()
    decoder,memory=make_reference(vae_state,'decoder')
    with torch.inference_mode(),torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        decoded=decoder(latent[None],memory_state=memory)[0]
    observe('decoded',decoded)
    identity=digest(args.run/run['output']['path'])==run['output']['sha256']
    report=dict(schema_version='seedvr2-video-parity-v1',reference_profile='FP32-B',passed=all(r['passed'] for r in rows) and identity,
                model_verified=False,model_certificate=None,scope='Whole temporal VAE and original 32-block 3D AWA trajectory; shared raw noise; pre-codec FP32; no long-video/cache or quality certification',
                tolerance=dict(atol=ATOL,rtol=RTOL,calibration='DIAGNOSTIC_NOT_MODEL_CERTIFICATION'),
                frames=frames,padded_frames=padded,latent_frames=lt,output_identity_valid=identity,native_run_sha256=digest(args.run/'run.json'),
                original_sources=verify_sources(),scripts={n:digest(ROOT/'tools'/n) for n in ('check_video.py','image_reference.py','vae_reference.py','dit_block_reference.py')},stages=rows)
    reference_contract(report,run,'video')
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('Complete temporal clip parity',report['passed'],flush=True)
    raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__':main()
