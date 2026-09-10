#!/usr/bin/env python3
"""Freeze FP32-B attention with one active dot-product lane and close logits."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(4)
q = torch.zeros(3, 1, 4, 512)
k = torch.zeros_like(q)
v = torch.empty_like(q)
for frame in range(3):
    for token in range(4):
        q[frame, 0, token, 0] = 100 + frame * 7 + token * .03
        k[frame, 0, token, 0] = [100., 100.01, 99.99, 100.003][token]
        v[frame, 0, token] = (torch.arange(512).float() % 7 - 3) * (token - 1.5)
with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
    y = torch.nn.functional.scaled_dot_product_attention(q, k, v)
for name, value in [('q', q), ('k', k), ('v', v), ('reference', y)]:
    value[:, 0].transpose(1, 2).transpose(0, 1).reshape(512, 3, 2, 2).contiguous().numpy().astype('<f4').tofile(args.output / (name + '.f32'))

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

report = dict(schema_version='seedvr2-frame-sdpa-fixture-v1', torch_version=torch.__version__,
              shape=[512, 3, 2, 2], oracle='PyTorch FP32 CPU SDPBackend.MATH; independent attention per frame',
              generator_sha256=sha(Path(__file__)),
              files={p.name: sha(p) for p in args.output.iterdir()}, model_verified=False)
(args.output / 'provenance.json').write_text(json.dumps(report, indent=2) + '\n')
