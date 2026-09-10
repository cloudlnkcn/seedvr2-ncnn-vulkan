#!/usr/bin/env python3
"""Freeze independent FP32 CPU softmax rows, including ragged AWA lengths."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(4)
cases = []
for width in [7, 8, 9, 15, 16, 17, 58, 64, 160, 178, 378, 1024]:
    i = torch.arange(3 * 5 * width, dtype=torch.int64)
    x = (((i * 1664525 + 1013904223) & 0xffffffff) % 8192).float() / 256 - 16
    x = x.reshape(3, 5, width)
    x[1] += 256
    x[2] *= 4
    y = torch.softmax(x, dim=-1)
    names = [f'{width}-input.f32', f'{width}-reference.f32']
    for name, tensor in zip(names, [x, y]):
        tensor.numpy().astype('<f4').tofile(args.output / name)
    cases.append(dict(shape=list(x.shape), input=names[0], reference=names[1]))

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

report = dict(schema_version='seedvr2-softmax-fixture-v1', torch_version=torch.__version__,
              oracle='torch.softmax(x, dim=-1), FP32 CPU', cases=cases,
              generator_sha256=sha(Path(__file__)),
              files={p.name: sha(p) for p in args.output.iterdir()}, model_verified=False)
(args.output / 'provenance.json').write_text(json.dumps(report, indent=2) + '\n')
