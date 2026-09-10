#!/usr/bin/env python3
"""Isolate the unrotated Q/K channels of original FP32-B RMSNorm."""
import argparse,hashlib,json
from pathlib import Path
import torch
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
i=torch.arange(59*3*128,dtype=torch.int64).reshape(59,3,128)
x=((i*1664525+1013904223)%65536).float()/32768-1
for row in range(59):x[row,:,:126]*=[.001,.1,1.,32.][row%4]
x[:,:,126]=1.;x[:,:,127]=-.5
qk=x[:,:2];expected=qk*torch.rsqrt(qk.pow(2).mean(-1,keepdim=True)+1e-5)
for name,tensor in [('input',x),('reference',expected[:,:,126:])]:tensor.contiguous().numpy().astype('<f4').tofile(a.output/(name+'.f32'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
r=dict(schema_version='seedvr2-awa-rms-fixture-v1',torch_version=torch.__version__,shape=[59,3,128],text_length=58,epsilon=1e-5,oracle='Original FP32 RMSNorm on Q/K, unit affine, lanes 126 and 127 untouched by multimodal RoPE',generator_sha256=sha(Path(__file__)),files={p.name:sha(p) for p in a.output.iterdir()},model_verified=False)
(a.output/'provenance.json').write_text(json.dumps(r,indent=2)+'\n')
