#!/usr/bin/env python3
"""Retain the FP32-B RMSNorm expression on rows with disparate magnitudes."""
import argparse,hashlib,json
from pathlib import Path
import torch
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
i=torch.arange(37*2560,dtype=torch.int64).reshape(37,2560)
x=((i*1664525+1013904223)%65536).float()/32768-1
for row in range(37):
    x[row]*=[1.,.001,32.,1024.][row%4]
    x[row,(row*97)%2560]*=1024
with torch.inference_mode():y=x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+1e-5)
for name,t in [('input',x),('reference',y)]:t.numpy().astype('<f4').tofile(a.output/(name+'.f32'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
r=dict(schema_version='seedvr2-rms-norm-fixture-v1',torch_version=torch.__version__,shape=[37,2560],epsilon=1e-5,oracle='Original diffusers RMSNorm expression: x * rsqrt(mean(x.pow(2)) + eps), FP32 CPU, no affine',generator_sha256=sha(Path(__file__)),files={p.name:sha(p) for p in a.output.iterdir()},model_verified=False)
(a.output/'provenance.json').write_text(json.dumps(r,indent=2)+'\n')
