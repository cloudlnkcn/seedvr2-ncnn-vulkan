#!/usr/bin/env python3
"""Audit a complete native image run against the original FP32-B trajectory.

Diagnostic thresholds are declared here, before reference evaluation. They are
not the unfrozen release model policy. Raw shared noise isolates numerical
agreement from different random generators. The original input resize is tested
separately against PyTorch bicubic antialias, including crop and normalization.
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from PIL import Image
from torch.nn import functional as F

from export_vae_image import digest, tensor
from image_reference import make_dit, endpoint
from vae_reference import make_reference, evaluate, load_checkpoint, verify_sources

ATOL, RTOL = 1e-3, 1e-3
ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    run = json.loads((args.run/'run.json').read_text())
    if run['status'] != 'SUCCEEDED' or run['profile'] != 'seedvr2-3b-image-fp32-b-v1':
        raise ValueError('Expected a complete FP32-B image run')
    if digest(args.input) != run['input']['sha256']:
        raise ValueError('Input identity changed')
    for suffix in ['stdout', 'stderr']:
        log = args.run.parent/(args.run.name+'.'+suffix)
        if log.exists() and ('VUID-' in log.read_text() or 'Validation Error' in log.read_text()):
            raise ValueError('Vulkan validation failure in retained process output')
    torch.set_num_threads(4)
    rows = []

    def read(name):
        row = run['diagnostics'][name]
        path = args.run/row['path']
        if digest(path) != row['sha256']:
            raise ValueError('Native tensor identity changed')
        return torch.from_numpy(np.fromfile(path, dtype='<f4').reshape(row['shape']).copy())

    def observe(name, expected):
        if name == 'output-ada':
            tensor(args.output/(name+'.f32'), expected)
            return
        actual = read(name)
        if actual.shape != expected.shape:
            raise ValueError(f'{name} shape mismatch: {actual.shape} vs {expected.shape}')
        x, y = actual.double().numpy(), expected.detach().double().numpy()
        diff = np.abs(x-y)
        finite = bool(np.isfinite(x).all() and np.isfinite(y).all())
        violations = int(np.sum(diff > ATOL+RTOL*np.abs(y)))
        row = dict(stage=name, passed=finite and violations == 0, finite=finite,
                   max_abs=float(diff.max()), rmse=float(np.sqrt(np.mean(diff**2))),
                   violations=violations, reference=tensor(args.output/(name+'.f32'), expected))
        rows.append(row)
        (args.output/'progress.json').write_text(json.dumps(rows, indent=2)+'\n')
        print(name, 'PASS' if row['passed'] else 'FAIL', row['max_abs'], row['rmse'], violations, flush=True)

    source = torch.from_numpy(np.asarray(Image.open(args.input).convert('RGB')).copy()).permute(2, 0, 1).float()/255.
    long_side = max(run['output']['height'], run['output']['width'])
    scale = long_side/max(source.shape[-2:])
    rh, rw = (round(int(n)*scale) for n in source.shape[-2:])
    transformed = F.interpolate(source[None], size=(rh, rw), mode='bicubic', align_corners=False, antialias=True)[0].clamp(0, 1)
    height, width = run['output']['height'], run['output']['width']
    oy, ox = round((rh-height)/2), round((rw-width)/2)
    transformed = (transformed[:, oy:oy+height, ox:ox+width]-.5)/.5
    observe('prepared', transformed)
    vae_state, lock = load_checkpoint(ROOT/'.cache/models/ema_vae.pth')
    encoder, memory = make_reference(vae_state, 'encoder')
    posterior = evaluate(encoder, memory, transformed[None])[0]
    observe('posterior', posterior)
    del encoder
    gc.collect()
    distribution = DiagonalGaussianDistribution(posterior[None])
    # Explicit noise injection is equivalent to the original p.sample(), with
    # the exact same noise tensor as native instead of a different RNG stream.
    conditioned = (distribution.mean[0]+distribution.std[0]*read('posterior-noise'))*.9152
    observe('conditioned', conditioned)
    noise = read('noise')
    state_path = ROOT/'.cache/models/seedvr2_ema_3b.pth'
    expected_hash = next(x for x in lock['files'] if x['rfilename'] == 'seedvr2_ema_3b.pth')['lfs']['sha256']
    if digest(state_path) != expected_hash:
        raise ValueError('Official DiT checkpoint hash mismatch')
    state = torch.load(state_path, weights_only=True, mmap=True, map_location='cpu')
    pos = torch.load(ROOT/'.cache/models/pos_emb.pt', weights_only=True, map_location='cpu').float()
    model = make_dit(state, observe)
    lh, lw = height//8, width//8
    cond = conditioned.permute(1, 2, 0)[None]

    def predict(x, time):
        joined = torch.cat((x.permute(1, 2, 0)[None], cond, torch.ones(1, lh, lw, 1)), -1)
        with torch.inference_mode():
            prediction = model(vid=joined.reshape(-1, 33), txt=pos,
                vid_shape=torch.tensor([[1, lh, lw]]), txt_shape=torch.tensor([[58]]),
                timestep=time, disable_cache=False).vid_sample
        value = prediction.reshape(lh, lw, 16).permute(2, 0, 1).contiguous()
        observe('velocity', value)
        return value

    with torch.inference_mode():
        latent = endpoint(noise, predict)/.9152
    observe('latent', latent)
    del model, state
    gc.collect()
    decoder, memory = make_reference(vae_state, 'decoder')
    decoded = evaluate(decoder, memory, latent[None])[0]
    observe('decoded', decoded)
    pixels = ((decoded.clamp(-1, 1)*.5+.5)*255).round().byte().permute(1, 2, 0).numpy()
    Image.fromarray(pixels).save(args.output/'reference.png')
    actual_pixels = np.asarray(Image.open(args.run/'output.png')).astype(np.int16)
    pixel_delta = np.abs(actual_pixels-pixels.astype(np.int16))
    output_ok = digest(args.run/'output.png') == run['output']['sha256']
    report = dict(schema_version='seedvr2-image-parity-v1', reference_profile='FP32-B',
        passed=all(x['passed'] for x in rows) and bool(pixel_delta.max() <= 1) and output_ok,
        model_verified=False, model_certificate=None,
        scope='Complete single-image original NaDiT trajectory, all 32 blocks, output Ada cache, VAE posterior/scaling and original Euler endpoint; shared raw noise; no BF16/FlashAttention or video certification',
        tolerance=dict(atol=ATOL, rtol=RTOL, max_uint8_error=1, calibration='DIAGNOSTIC_NOT_MODEL_CERTIFICATION'),
        native_run_sha256=digest(args.run/'run.json'), original_sources=verify_sources(),
        scripts={n: digest(ROOT/'tools'/n) for n in ['check_image.py', 'image_reference.py', 'dit_block_reference.py', 'vae_reference.py', 'awa_reference.py']},
        stages=rows, output_identity_valid=output_ok,
        pixels=dict(max_abs=int(pixel_delta.max()), different=int(np.sum(pixel_delta != 0)), total=int(pixel_delta.size)))
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Complete image parity', report['passed'], report['pixels'], flush=True)
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
