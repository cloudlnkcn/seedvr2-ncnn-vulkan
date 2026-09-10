#!/usr/bin/env python3
"""Freeze a small original PyTorch frame-wise GroupNorm arithmetic regression."""
import argparse, hashlib, json
from pathlib import Path
import torch
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
a=p.parse_args();a.output.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(2)
t,c,h,w=2,64,8,8
x=torch.empty(t,c,h,w)
for frame in range(t):
    for channel in range(c):
        mean=[0.,16.,1024.,4096.][(channel//2)%4]+frame*2
        x[frame,channel]=mean+((torch.arange(h*w).reshape(h,w)%2)*2-1).float()
gamma=1.+torch.arange(c).float()/128
beta=torch.arange(c).float()/1000
with torch.inference_mode(): y=torch.nn.functional.group_norm(x,32,gamma,beta,1e-6)
for name,tensor in [('input',x.transpose(0,1)),('reference',y.transpose(0,1)),('affine',torch.cat([gamma,beta]))]:
    tensor.contiguous().numpy().astype('<f4').tofile(a.output/(name+'.f32'))
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
report=dict(schema_version='seedvr2-frame-norm-reference-v1', torch_version=torch.__version__, shape=[c,t,h,w], groups=32, epsilon=1e-6,
    oracle='Original PyTorch CPU GroupNorm applied independently to each frame; variance one, distinct group and frame means',
    generator_sha256=sha(Path(__file__)), files={p.name:sha(p) for p in a.output.iterdir()}, model_verified=False)
(a.output/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
