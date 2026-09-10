#!/usr/bin/env python3
"""Exercise Welford variance merging with independently generated varied groups."""
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
t, c, h, w = 3, 512, 10, 16
i = torch.arange(t * c * h * w, dtype=torch.int64).reshape(t, c, h, w)
x = ((i * 1664525 + 1013904223) % 65536).float() / 32768 - 1
# Distinct group means, scales and sparse outliers give nontrivial merge terms.
for frame in range(t):
    for group in range(32):
        part = x[frame, group * 16:(group + 1) * 16]
        part *= [0.125, 1., 16., 128.][group % 4]
        part[0, 0, (group * 3 + frame) % w] *= 256
        part += (group % 3 - 1) * 32 + frame * .125
gamma = 1 + torch.arange(c).float() / 64
beta = (torch.arange(c).float() % 11) / 32
with torch.inference_mode():
    y = torch.nn.functional.group_norm(x, 32, gamma, beta, 1e-6)
for name, value in [('input', x.transpose(0, 1)), ('reference', y.transpose(0, 1)),
                    ('affine', torch.cat([gamma, beta]))]:
    value.contiguous().numpy().astype('<f4').tofile(args.output / (name + '.f32'))

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

report = dict(schema_version='seedvr2-frame-norm-reference-v1', torch_version=torch.__version__,
              shape=[c, t, h, w], groups=32, epsilon=1e-6,
              oracle='Original PyTorch CPU GroupNorm; integer-generated varied groups with sparse outliers',
              generator_sha256=sha(Path(__file__)),
              files={p.name: sha(p) for p in args.output.iterdir()}, model_verified=False)
(args.output / 'provenance.json').write_text(json.dumps(report, indent=2) + '\n')
