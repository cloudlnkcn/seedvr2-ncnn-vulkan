#!/usr/bin/env python3
"""Export temporal VAE through pinned pnnx and compare to unchanged upstream."""
import argparse
import gc
import json
import os
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import torch

from export_vae_image import digest, error, tensor
from vae_reference import ROOT, load_checkpoint, make_reference, verify_sources
from vae_video_module import VideoVAE

MODULES = ['TemporalConv', 'FrameNorm', 'FrameSDPA', 'TemporalShuffle']


def lower(folder):
    param = folder/'model.ncnn.param'
    lines = param.read_text().splitlines()
    counts = {}
    # All weighted boundaries are opaque modules; pointwise graph nodes have
    # no weights. Re-serialize only validated pnnx attributes in layer order.
    with zipfile.ZipFile(folder/'model.pnnx.bin') as archive, (folder/'weights.tmp').open('wb') as stream:
        for i, line in enumerate(lines[2:], 2):
            row = line.split()
            kind = row[0].removeprefix('vae_video_module.')
            if kind not in MODULES:
                if row[0] not in ('Input', 'Swish', 'BinaryOp', 'Split'):
                    raise ValueError(f'Unlowered video VAE layer {row[0]}')
                continue
            ni, no = int(row[2]), int(row[3])
            if no != 1 or ni != (3 if kind == 'FrameSDPA' else 1):
                raise ValueError('Unexpected custom module arity')
            start = 4+ni+no
            def attr(name, dtype):
                return np.frombuffer(archive.read(row[1]+'.'+name), dtype=dtype)
            p = []
            if kind == 'TemporalConv':
                spec = attr('spec', '<i4').tolist()
                if len(spec) != 9 or spec[0] != 1:
                    raise ValueError('Unexpected temporal convolution spec')
                _, cin, cout, kt, ks, st, ss, pad, end = spec
                if not (1 <= cin <= 4096 and 1 <= cout <= 4096 and kt in (1,3) and ks in (1,3) and st in (1,2) and ss in (1,2) and pad in (0,1) and end in (0,1)):
                    raise ValueError('Invalid convolution bounds')
                w, b = attr('weight', '<f4'), attr('bias', '<f4')
                if len(w) != cout*cin*kt*ks*ks or len(b) != cout or not np.isfinite(w).all() or not np.isfinite(b).all():
                    raise ValueError('Invalid temporal convolution weights')
                # Cout,Cin,Kt,Kh,Kw is also Cout,(Cin*Kt),Kh,Kw without reorder.
                stream.write(w.tobytes()); stream.write(b.tobytes())
                p = [f'{j}={v}' for j,v in enumerate(spec[1:])]
            elif kind == 'FrameNorm':
                spec = attr('spec','<i4').tolist()
                if len(spec) != 3 or spec[0] != 1 or spec[2] != 32:
                    raise ValueError('Unexpected frame norm spec')
                w, b = attr('weight','<f4'), attr('bias','<f4')
                if len(w) != spec[1] or len(b) != spec[1] or not np.isfinite(w).all() or not np.isfinite(b).all():
                    raise ValueError('Invalid frame norm weights')
                stream.write(w.tobytes()); stream.write(b.tobytes())
                p = [f'0={spec[1]}', '1=32', '2=1.000000000e-06']
            elif kind == 'TemporalShuffle':
                spec = attr('spec','<i4').tolist()
                if len(spec) != 3 or spec[0] != 1 or spec[2] not in (1,2):
                    raise ValueError('Unexpected temporal shuffle spec')
                p = [f'0={spec[1]}', f'1={spec[2]}']
            row[0] = 'SeedVR2'+kind
            lines[i] = ' '.join(row[:start]+p)
            counts[kind] = counts.get(kind, 0)+1
    if set(counts) != set(MODULES) - ({'TemporalShuffle'} if folder.name == 'encoder' else set()):
        raise ValueError(f'Missing video modules {counts}')
    param.rename(folder/'model.original.ncnn.param')
    (folder/'model.ncnn.bin').rename(folder/'model.original.ncnn.bin')
    (folder/'weights.tmp').rename(folder/'model.ncnn.bin')
    param.write_text('\n'.join(lines)+'\n')
    return dict(modules=counts, binary_order='validated pnnx attributes in graph order',
                temporal_kernel='preserved; exact causal channel gather + native 2D convolution')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'.cache/models/ema_vae.pth')
    parser.add_argument('--pnnx', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Output must be empty')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(60705)
    state, lock = load_checkpoint(args.checkpoint)
    report = dict(schema_version='vae-video-export-v1', reference_profile='FP32-B',
                  scope='Whole-clip original causal VAE, B=1, memory disabled; no streaming cache',
                  checkpoint=dict(sha256=digest(args.checkpoint), tensors=len(state), revision=lock['revision']),
                  official_sources=verify_sources(), ncnn_commit=json.loads((ROOT/'converter-dependencies.lock.json').read_text())['ncnn']['commit'],
                  pnnx_sha256=digest(args.pnnx), model_verified=False, parts=[], cases=[],
                  tolerance=dict(atol=1e-4, rtol=1e-3, calibration='DIAGNOSTIC_NOT_MODEL_CERTIFICATION'))
    for part in ('encoder', 'decoder'):
        print(part, 'strict original reference and temporal candidate', flush=True)
        ref, memory_state = make_reference(state, part)
        model = VideoVAE(state, part).eval().requires_grad_(False)
        def script_boundaries(parent):
            for name, child in list(parent.named_children()):
                if type(child).__name__ in MODULES:
                    setattr(parent, name, torch.jit.script(child))
                else:
                    script_boundaries(child)
        script_boundaries(model)
        folder = args.output/part
        folder.mkdir()
        shape = (3, 5, 32, 48) if part == 'encoder' else (16, 2, 4, 6)
        second = (3, 9, 48, 32) if part == 'encoder' else (16, 3, 6, 4)
        with torch.inference_mode():
            traced = torch.jit.trace(model, torch.randn(shape), check_trace=False)
        traced.save(str(folder/'model.pt'))
        command = [str(args.pnnx.resolve()), 'model.pt', 'inputshape=['+','.join(map(str,shape))+']',
                   'inputshape2=['+','.join(map(str,second))+']',
                   'moduleop='+','.join('vae_video_module.'+m for m in MODULES), 'fp16=0']
        print(part, 'pnnx export', flush=True)
        with (folder/'pnnx.log').open('w') as log:
            subprocess.run(command, cwd=folder, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900,
                           env={**os.environ, 'OMP_NUM_THREADS':'4', 'MKL_NUM_THREADS':'4'})
        lowering = lower(folder)
        report['parts'].append(dict(part=part, lowering=lowering, official_tensors=len(ref.state_dict()),
                                   model_param_sha256=digest(folder/'model.ncnn.param'), model_bin_sha256=digest(folder/'model.ncnn.bin')))
        for frames, h, w in ((1,32,48),(5,32,48),(9,48,32),(17,32,32)):
            case_id = f'vae-video-{part}-t{frames}-{h}x{w}'
            case = args.output/case_id
            case.mkdir()
            x = torch.randn((3,frames,h,w) if part=='encoder' else (16,(frames-1)//4+1,h//8,w//8))
            with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                y = ref(x.unsqueeze(0), memory_state=memory_state)[0]
                z = traced(x)
            comparison = error(z,y)
            if not comparison['passed']:
                raise ValueError(f'{case_id} candidate differs from original: {comparison}')
            doc = dict(schema_version='ncnn-graph-case-v1', component='vae-video-'+part, case_id=case_id,
                       precision='fp32', reference_profile='FP32-B', checkpoint_sha256=report['checkpoint']['sha256'],
                       input=tensor(case/'input.f32',x), reference=tensor(case/'reference.f32',y), tolerance=report['tolerance'])
            for name in ('model.ncnn.param','model.ncnn.bin'):
                os.link(folder/name,case/name)
                doc['model_param' if name.endswith('param') else 'model_bin'] = dict(path=name,sha256=digest(folder/name))
            (case/'case.json').write_text(json.dumps(doc,indent=2)+'\n')
            report['cases'].append(dict(path=case_id+'/case.json',sha256=digest(case/'case.json'),scripted_vs_official=comparison))
            print(case_id, 'PASS', comparison, flush=True)
        del model,traced,ref
        gc.collect()
    report['passed']=True
    report['scripts']={name:digest(ROOT/'tools'/name) for name in ('vae_video_module.py','export_vae_video.py','vae_reference.py')}
    (args.output/'suite.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
