#!/usr/bin/env python3
"""Freeze the FP32-B RGB causal convolution independently of model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(4)
if torch.__version__ != '2.9.0+cpu':
    raise ValueError('Use the locked FP32-B PyTorch environment')

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def save(name, value):
    path = args.output / name
    value.contiguous().numpy().astype('<f4').tofile(path)
    return dict(path=name, dtype='f32le', shape=list(value.shape), sha256=sha(path))

def values(count, seed, divisor):
    state = seed
    result = []
    for _ in range(count):
        state = (1664525 * state + 1013904223) & 0xffffffff
        result.append(((state >> 8) % 131071 - 65535) / divisor)
    return torch.tensor(result, dtype=torch.float32)

x = values(3 * 5 * 9 * 13, 1701, 32768).reshape(3, 5, 9, 13)
w = values(128 * 3 * 27, 91, 262144).reshape(128, 3, 3, 3, 3)
b = values(128, 29, 1048576)
input_item = save('input.f32', x)
weights_item = save('weights.f32', torch.cat((w.flatten(), b)))
cases = []
for stride in (1, 2):
    with torch.inference_mode():
        padded = F.pad(x[None], (0, 0, 0, 0, 2, 0), mode='replicate')
        y = F.conv3d(padded, w, b, stride=stride, padding=(0, 1, 1))[0]
    cases.append(dict(stride=stride, input=input_item, weights=weights_item,
                      reference=save(f'reference-{stride}.f32', y)))
report = dict(schema_version='seedvr2-rgb-conv-fixture-v1', torch_version=torch.__version__,
              oracle='FP32 CPU conv3d, RGB input, causal head replication, zero spatial padding',
              generator_sha256=sha(Path(__file__)), cases=cases, model_verified=False)
(args.output / 'provenance.json').write_text(json.dumps(report, indent=2) + '\n')
