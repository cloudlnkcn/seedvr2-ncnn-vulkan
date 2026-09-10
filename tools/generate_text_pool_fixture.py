#!/usr/bin/env python3
"""Freeze the official repeated-text pooling order and FP32 mean, without weights."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from awa_reference import module, verify_sources
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4);source=verify_sources();na=module('text_pool_fixture','models/dit_v2/na.py')
cases=[]
for serial,lengths in enumerate([[17],[40]*3,[40]*5,[3,5,9,17],[16]*8,[48]*17,[16]*32]):
    nw=len(lengths);nt=58;hd=128;nv=sum(lengths)
    keys=na.repeat_concat(torch.arange(nv),torch.arange(nv,nv+nt),torch.tensor(lengths),torch.tensor([nt]),[nw])
    owners=na.repeat_concat(torch.full((nv,),-1),torch.arange(nt),torch.tensor(lengths),torch.tensor([nt]),[nw])
    which=torch.cat([torch.cat([torch.full((n,),-1),torch.full((nt,),i)]) for i,n in enumerate(lengths)])
    order=which[torch.argsort(keys)][nv:].reshape(nt,nw)
    i=torch.arange(nw*nt*hd,dtype=torch.int64)
    x=((i*1664525+1013904223)&0xffffffff).remainder(8192).float().reshape(nw,nt,hd)/512-8
    # Cancellation and tiny retained terms expose the significance of tie order.
    if nw>=3:
        x[0,:,::7]=8192;x[1,:,::7]=-8192;x[2,:,::7]=1/1024
    windows=[torch.cat([torch.zeros(n,1,hd),x[w].reshape(nt,1,hd)]) for w,n in enumerate(lengths)]
    _,unconcat=na.repeat_concat_idx(torch.tensor(lengths),torch.tensor([nt]),torch.tensor([nw]))
    _,y=unconcat(torch.cat(windows));y=y.reshape(nt,hd)
    names=[f'{serial}-input.f32',f'{serial}-reference.f32']
    for name,value in zip(names,[x,y]):value.numpy().astype('<f4').tofile(args.output/name)
    cases.append(dict(lengths=lengths,shape=list(x.shape),order=order.tolist(),input=names[0],reference=names[1]))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
report=dict(schema_version='seedvr2-text-pool-fixture-v1',torch_version=torch.__version__,oracle='Official na.repeat_concat_idx / unconcat_coalesce, FP32 CPU',generator_sha256=sha(Path(__file__)),source_sha256=sha(Path('tests/reference/seedvr/models/dit_v2/na.py')),cases=cases,files={p.name:sha(p) for p in args.output.iterdir()},model_verified=False)
(args.output/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
