#!/usr/bin/env python3
"""Create independent, small FP32 antialiased-resize reference tensors."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    rows = []
    for source, target in [(64, 128), (128, 64)]:
        y, x, c = np.indices((source, source, 3))
        pixels = ((x * 37 + y * 71 + c * 113 + (x * y) % 251) % 256).astype(np.uint8)
        name = f'{source}-to-{target}'
        pixels.tofile(args.output/(name+'.rgb8'))
        tensor = torch.from_numpy(pixels).permute(2, 0, 1).float()[None] / 255.
        result = F.interpolate(tensor, size=(target, target), mode='bicubic',
                               antialias=True, align_corners=False).clamp(0, 1)
        ((result[0] - .5) / .5).contiguous().numpy().astype('<f4').tofile(args.output/(name+'.f32'))
        rows.append(dict(name=name, input_shape=[source, source, 3], output_shape=[3, target, target]))
    report = dict(schema_version='seedvr2-preprocessing-reference-v1', torch_version=torch.__version__,
                  source='https://github.com/pytorch/pytorch/blob/v2.9.0/aten/src/ATen/native/cpu/UpSampleKernel.cpp',
                  oracle='PyTorch bicubic antialias FP32, align_corners=False; generated RGB texture',
                  model_verified=False, cases=rows,
                  generator_sha256=hashlib.file_digest(Path(__file__).open('rb'), 'sha256').hexdigest(),
                  files={p.name: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
                         for p in sorted(args.output.iterdir())})
    (args.output/'provenance.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
