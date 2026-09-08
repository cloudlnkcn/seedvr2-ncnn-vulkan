#!/usr/bin/env python3
"""Assemble a hashed, Python-free image package from verified source exports."""
import argparse
import json
import os
import subprocess
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from awa_reference import verify_sources
from dit_block_module import linear
from dit_block_reference import time_embedding
from export_vae_image import digest, tensor

ROOT = Path(__file__).resolve().parents[1]


class OutputProjection(nn.Module):
    def __init__(self, state, emb):
        super().__init__()
        self.norm = nn.Parameter(state['vid_out_norm.weight'], False)
        # Original NaDiT reuses emb_repeat_0_vid from block zero at output Ada.
        self.scale = nn.Parameter(emb[:, 1]+state['vid_out_ada.out_scale'], False)
        self.shift = nn.Parameter(emb[:, 0]+state['vid_out_ada.out_shift'], False)
        self.proj = linear(state, 'vid_out.proj')

    def forward(self, x):
        return self.proj(F.rms_norm(x, (2560,), self.norm, 1e-5)*self.scale+self.shift)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--blocks', type=Path, action='append', required=True)
    p.add_argument('--vae', type=Path, required=True)
    p.add_argument('--pnnx', type=Path, required=True)
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'manifest.json').exists():
        raise ValueError('A finalized package already exists')
    torch.set_num_threads(4)
    lock = json.loads((ROOT/'model-sources.lock.json').read_text())
    for name in ['seedvr2_ema_3b.pth', 'pos_emb.pt']:
        row = next(x for x in lock['files'] if x['rfilename'] == name)
        path = ROOT/'.cache/models'/name
        if path.stat().st_size != row['size'] or digest(path) != row['lfs']['sha256']:
            raise ValueError('Checkpoint identity mismatch')
    state = torch.load(ROOT/'.cache/models/seedvr2_ema_3b.pth', weights_only=True, mmap=True, map_location='cpu')
    pos = torch.load(ROOT/'.cache/models/pos_emb.pt', weights_only=True, map_location='cpu').float()
    with torch.inference_mode():
        emb = time_embedding(state)
        text = F.linear(pos, state['txt_in.weight'], state['txt_in.bias'])
    manifest = dict(schema_version='seedvr2-image-package-v1', model_id='seedvr2-3b',
        profile='seedvr2-3b-image-fp32-b-v1', precision='fp32', reference_profile='FP32-B',
        ncnn_commit=json.loads((ROOT/'engine-dependencies.lock.json').read_text())['ncnn']['commit'],
        model_sources=lock, official_sources=verify_sources(), model_verified=False,
        limits=dict(min_side=64, max_side=512, max_pixels=512*512, divisible_by=16),
        sampling=dict(steps=1, timestep=1000, cfg=1, latent_scale=0.9152, color_fix='none'),
        constants={}, graphs=[], exporter=dict(pnnx_sha256=digest(args.pnnx),
            script_sha256=digest(Path(__file__)), torch_version=torch.__version__))
    for name, value in [('text', text), ('time', emb)]:
        manifest['constants'][name] = tensor(out/(name+'.f32'), value)

    def descriptor(path):
        return dict(path=path.relative_to(out).as_posix(), sha256=digest(path), bytes=path.stat().st_size)

    def link_graph(source, name):
        folder = out/name
        folder.mkdir(exist_ok=True)
        entry = dict(id=name)
        for suffix, field in [('param', 'param'), ('bin', 'weights')]:
            a, b = source/('model.ncnn.'+suffix), folder/('model.ncnn.'+suffix)
            if b.exists():
                if digest(a) != digest(b):
                    raise ValueError('Existing package file differs')
            else:
                os.link(a, b)
            entry[field] = descriptor(b)
        manifest['graphs'].append(entry)

    vae_report = json.loads((args.vae/'suite.json').read_text())
    if vae_report['checkpoint']['sha256'] != next(x for x in lock['files'] if x['rfilename'] == 'ema_vae.pth')['lfs']['sha256']:
        raise ValueError('VAE source checkpoint mismatch')
    for part in ['encoder', 'decoder']:
        original = next(x for x in vae_report['parts'] if x['part'] == part)
        for suffix in ['param', 'bin']:
            if digest(args.vae/part/f'model.ncnn.{suffix}') != original[f'model_{suffix}_sha256']:
                raise ValueError('VAE export hash mismatch')
        link_graph(args.vae/part, part)
    available = {}
    for path in args.blocks:
        report = json.loads((path/'suite.json').read_text())
        if not report['passed'] or report['checkpoint']['sha256'] != next(x for x in lock['files'] if x['rfilename'] == 'seedvr2_ema_3b.pth')['lfs']['sha256']:
            raise ValueError('DiT export source mismatch')
        for block in report['blocks']:
            index = block['index']
            case = next(x for x in report['cases'] if x['path'].startswith(f'dit-block-{index}-'))
            cp = path/case['path']
            if digest(cp) != case['sha256']:
                raise ValueError('DiT case manifest changed')
            spec = json.loads(cp.read_text())
            folder = path/f'block-{index}'
            for suffix in ['param', 'bin']:
                if digest(folder/f'model.ncnn.{suffix}') != spec[f'model_{suffix}']['sha256']:
                    raise ValueError('DiT export hash mismatch')
            if index in available:
                raise ValueError('Duplicate block source')
            available[index] = folder
    if set(available) != set(range(32)):
        raise ValueError('The package requires all 32 blocks')
    for index in range(32):
        link_graph(available[index], f'block-{index:02d}')
        print('packaged block', index, flush=True)
    for name, model, channels in [('patch-in', linear(state, 'vid_in.proj'), 132),
                                  ('patch-out', OutputProjection(state, emb), 2560)]:
        folder = out/name
        folder.mkdir(exist_ok=True)
        with torch.inference_mode():
            traced = torch.jit.trace(model, torch.randn(6, channels))
        traced.save(str(folder/'model.pt'))
        with (folder/'pnnx.log').open('w') as log:
            subprocess.run([str(args.pnnx.resolve()), 'model.pt', f'inputshape=[6,{channels}]',
                            f'inputshape2=[24,{channels}]', 'fp16=0'], cwd=folder,
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
        param = folder/'model.ncnn.param'
        lines = param.read_text().splitlines()
        replacements = 0
        for i, line in enumerate(lines[2:], 2):
            row = line.split()
            if row[0] == 'MemoryData':
                if row[2:] != ['0', '1', row[4], '0=2560']:
                    raise ValueError('Unexpected projection constant')
                row[0] = 'SeedVR2Constant'
                lines[i] = ' '.join(row)
                replacements += 1
        if replacements != (2 if name == 'patch-out' else 0):
            raise ValueError('Projection constants changed')
        param.rename(folder/'model.original.ncnn.param')
        param.write_text('\n'.join(lines)+'\n')
        manifest['graphs'].append(dict(id=name, param=descriptor(param),
                                       weights=descriptor(folder/'model.ncnn.bin')))
    # The presence of this atomically written manifest marks a complete package.
    tmp = out/'manifest.json.tmp'
    tmp.write_text(json.dumps(manifest, indent=2)+'\n')
    tmp.replace(out/'manifest.json')
    print(json.dumps(dict(status='PACKAGED', manifest_sha256=digest(out/'manifest.json'),
                          graphs=len(manifest['graphs']), model_verified=False)), flush=True)


if __name__ == '__main__':
    main()
