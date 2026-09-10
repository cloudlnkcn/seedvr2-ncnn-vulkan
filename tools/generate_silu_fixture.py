#!/usr/bin/env python3
"""Generate an input-independent FP32 SiLU grid, including both saturation tails."""
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
# Binary fractions avoid an implementation-dependent linspace input oracle.
x = torch.arange(-20480, 20481, dtype=torch.float32) / 256
with torch.inference_mode():
    y = torch.nn.functional.silu(x)
for name, value in [('input', x), ('reference', y)]:
    value.numpy().astype('<f4').tofile(args.output / (name + '.f32'))

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

report = dict(schema_version='seedvr2-silu-fixture-v1', torch_version=torch.__version__,
              shape=[x.numel()], oracle='torch.nn.functional.silu, FP32 CPU',
              generator_sha256=sha(Path(__file__)),
              files={p.name: sha(p) for p in args.output.iterdir()}, model_verified=False)
(args.output / 'provenance.json').write_text(json.dumps(report, indent=2) + '\n')
