#!/usr/bin/env python3
"""Export and independently compare the complete official single-frame VAE.

All 250 official tensors must load strictly into the official encoder/decoder.
FP32 diagnostic tolerances are fixed here before candidate runs; they are not
calibrated or sufficient for a SeedVR2 model certificate.
"""
import argparse
import gc
import hashlib
import json
import os
import subprocess
from pathlib import Path

import diffusers
import numpy as np
import torch

from vae_image_module import ImageVAE
from vae_reference import ROOT, evaluate, load_checkpoint, make_reference, verify_sources

ATOL, RTOL = 1e-4, 1e-3


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tensor(path, x):
    x.detach().cpu().contiguous().numpy().astype('<f4').tofile(path)
    return dict(path=path.name, sha256=digest(path), dtype='f32le', shape=list(x.shape))


def error(a, b):
    x, y = a.detach().double().numpy(), b.detach().double().numpy()
    delta = np.abs(x-y)
    return dict(passed=bool(np.isfinite(x).all() and np.isfinite(y).all() and
                            np.all(delta <= ATOL+RTOL*np.abs(y))),
                max_abs=float(delta.max()), rmse=float(np.sqrt(np.mean(delta**2))),
                violations=int(np.sum(delta > ATOL+RTOL*np.abs(y))))


def lower_shape_only_ops(path):
    """The one-head SDPA boundary uses native Vulkan Reshape for axis zero.

    Exact match only: add/remove a singleton head around SDPA. Both equivalent
    CPU and Vulkan paths are subsequently compared to the original 3D model.
    """
    lines = path.read_text().splitlines()
    counts = {'ExpandDims': 0, 'Squeeze': 0}
    for i, line in enumerate(lines[2:], 2):
        fields = line.split()
        if fields[0] not in counts:
            continue
        if len(fields) != 7 or fields[2:4] != ['1', '1'] or fields[6] != '-23303=1,0':
            raise ValueError('Unexpected VAE singleton-head conversion')
        kind = fields[0]
        counts[kind] += 1
        fields[0] = 'Reshape'
        fields[6:] = ['0=0', '1=0'] + (['2=1'] if kind == 'ExpandDims' else [])
        lines[i] = ' '.join(fields)
    if counts != {'ExpandDims': 3, 'Squeeze': 1}:
        raise ValueError(f'Unexpected VAE SDPA head layout: {counts}')
    original = path.with_suffix('.original.param')
    path.rename(original)
    path.write_text('\n'.join(lines)+'\n')
    return dict(original_sha256=digest(original), rewrites=counts,
                replacement='Reshape: insert/remove singleton head, axis zero')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'.cache/models/ema_vae.pth')
    parser.add_argument('--pnnx', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.pnnx = args.pnnx.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit('Output must be empty')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(60701)
    state, lock = load_checkpoint(args.checkpoint)
    if len(state) != 250:
        raise ValueError('Unexpected official VAE tensor count')
    report = dict(schema_version='vae-image-export-v1', scope='Complete single-frame VAE encoder distribution parameters and decoder; no posterior sampling, scaling, or DiT',
                  checkpoint=dict(repository=lock['repository'], revision=lock['revision'],
                                  filename='ema_vae.pth', sha256=digest(args.checkpoint), tensors=250),
                  official_sources=verify_sources(), reference_profile='FP32-B',
                  reference_adapters=['single rank', 'no slicing', 'PyTorch math SDPA',
                                      'inflation_mode none: checkpoint already contains 3D weights'],
                  specialization=['T=1 only', 'sum temporal convolution kernels',
                                  'keep upsample temporal phase zero', 'reorder PixelShuffle channels'],
                  environment=dict(torch=torch.__version__, diffusers=diffusers.__version__),
                  pnnx_sha256=digest(args.pnnx),
                  ncnn_commit=json.loads((ROOT/'engine-dependencies.lock.json').read_text())['ncnn']['commit'],
                  scripts={name: digest(ROOT/'tools'/name) for name in
                           ['vae_reference.py', 'vae_image_module.py', 'export_vae_image.py']},
                  tolerance=dict(atol=ATOL, rtol=RTOL, calibration='DIAGNOSTIC_NOT_MODEL_CERTIFICATION'),
                  model_verified=False, parts=[], cases=[])
    for part in ['encoder', 'decoder']:
        print(part, 'loading strict official model', flush=True)
        reference, memory_state = make_reference(state, part)
        candidate = ImageVAE(state, part).requires_grad_(False).eval()
        folder = args.output/part
        folder.mkdir()
        channels = 3 if part == 'encoder' else 16
        hw = (32, 48) if part == 'encoder' else (4, 6)
        sample = torch.randn(1, channels, *hw)
        with torch.inference_mode():
            traced = torch.jit.trace(candidate, sample, check_trace=True)
        traced.save(str(folder/'model.pt'))
        # Different token counts are essential: swapping H/W alone incorrectly
        # makes pnnx consider their product constant and specializes flatten.
        second = (48, 64) if part == 'encoder' else (6, 8)
        command = [str(args.pnnx), 'model.pt', f'inputshape=[1,{channels},{hw[0]},{hw[1]}]',
                   f'inputshape2=[1,{channels},{second[0]},{second[1]}]', 'fp16=0']
        print(part, 'pnnx export', flush=True)
        with (folder/'pnnx.log').open('w') as log:
            subprocess.run(command, cwd=folder, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=600, env={**os.environ, 'OMP_NUM_THREADS': '4'})
        param, binary = folder/'model.ncnn.param', folder/'model.ncnn.bin'
        lowering = lower_shape_only_ops(param)
        lines = param.read_text().splitlines()
        operators = sorted({line.split()[0] for line in lines[2:] if line.strip()})
        if any(op.startswith(('aten', 'prim', 'pnnx')) or '::' in op for op in operators):
            raise ValueError(f'Unlowered VAE operators: {operators}')
        report['parts'].append(dict(part=part, official_tensors=len(reference.state_dict()),
                                   operators=operators, shape_lowering=lowering,
                                   model_param_sha256=digest(param),
                                   model_bin_sha256=digest(binary)))
        for height, width in [(32, 48), (48, 32), (64, 64)]:
            case_id = f'vae-image-{part}-{height}x{width}'
            case = args.output/case_id
            case.mkdir()
            shape = (1, channels, height, width) if part == 'encoder' else (1, channels, height//8, width//8)
            x = torch.randn(shape)
            y = evaluate(reference, memory_state, x)
            with torch.inference_mode():
                z = traced(x)
            comparison = error(z, y)
            if not comparison['passed']:
                raise ValueError(f'{case_id} specialization mismatch: {comparison}')
            for original in [param, binary]:
                os.link(original, case/original.name)
            doc = dict(schema_version='ncnn-graph-case-v1', case_id=case_id,
                       component=f'vae-image-{part}', reference_profile='FP32-B', precision='fp32',
                       checkpoint_sha256=report['checkpoint']['sha256'],
                       model_param=dict(path=param.name, sha256=digest(param)),
                       model_bin=dict(path=binary.name, sha256=digest(binary)),
                       input=tensor(case/'input.f32', x[0]),
                       reference=tensor(case/'reference.f32', y[0]),
                       tolerance=report['tolerance'])
            path = case/'case.json'
            path.write_text(json.dumps(doc, indent=2)+'\n')
            report['cases'].append(dict(case_id=case_id, path=str(path.relative_to(args.output)),
                                        sha256=digest(path), specialization_vs_official=comparison))
            print(case_id, 'specialization PASS', comparison, flush=True)
        del candidate, traced, reference
        gc.collect()
    report['passed'] = True
    (args.output/'suite.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Complete VAE image export PASS; native execution still requires check_graph.py', flush=True)


if __name__ == '__main__':
    main()
