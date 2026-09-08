#!/usr/bin/env python3
"""Export complete checkpoint-backed DiT blocks with one preserved custom AWA."""
import argparse
import gc
import hashlib
import json
import os
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from dit_block_module import DiTBlock
from dit_block_reference import evaluate, make_block, time_embedding
from awa_reference import verify_sources
from export_vae_image import digest, error, tensor

ROOT = Path(__file__).resolve().parents[1]


def lower(folder, index):
    param = folder/'model.ncnn.param'
    original = folder/'model.original.ncnn.param'
    lines = (original if original.exists() else param).read_text().splitlines()
    matches = []
    for i, line in enumerate(lines[2:], 2):
        op = line.split()
        if op[0] == 'awa_export_module.AdaptiveWindowAttention':
            matches.append((i, op))
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one preserved AWA, found {len(matches)}')
    line, op = matches[0]
    if op[2:4] != ['2', '2'] or set(op[8:]) != {'-23310=1,1', '-23311=2,4,128', '-23312=1,21', '-23313=1,3'}:
        raise ValueError('Unexpected AWA module attributes')
    with zipfile.ZipFile(folder/'model.pnnx.bin') as archive:
        spec = np.frombuffer(archive.read(op[1]+'.spec'), dtype='<i4')
        epsilon = np.frombuffer(archive.read(op[1]+'.epsilon'), dtype='<f4')
        norms = np.frombuffer(archive.read(op[1]+'.norm_weights'), dtype='<f4')
        frequencies = np.frombuffer(archive.read(op[1]+'.rope_frequencies'), dtype='<f4')
    if spec.tolist() != [1, 20, index % 2] or epsilon.shape != (1,) or norms.shape != (512,) or not np.isfinite(norms).all():
        raise ValueError('AWA serialized metadata mismatch')
    expected = (1./(10000.**(torch.arange(0, 42, 2).float()/42.))).numpy()
    if not np.array_equal(frequencies, expected):
        raise ValueError('AWA frequency mismatch')
    op[0] = 'SeedVR2AWA'
    op[8:] = ['0=20', f'1={index % 2}', f'2={epsilon[0]:.9e}', '3=2']
    lines[line] = ' '.join(op)
    # AWA's opaque boundary carries THWD, with no batch axis. pnnx infers
    # a batch around the surrounding Linear layers; lower these four exact
    # reshape patterns to explicit ncnn THWD <-> token-row layouts.
    patterns = {
        ('0=2560', '1=-1', '12=233', '13=0'): ['0=2560', '1=-1'],
        ('6="7680,1h,1d"',): ['6="7680,1h,1d,1c"'],
        ('0=2560',): ['0=2560', '1=-1'],
        ('12=0', '13=233', '6="2560,1h,1d,1c"'): ['6="2560,1h,1d,1c"'],
    }
    counts = {pattern: 0 for pattern in patterns}
    constant_count = 0
    for i, text in enumerate(lines[2:], 2):
        row = text.split()
        if row[0] == 'MemoryData':
            if row[2:4] != ['0', '1'] or row[5:] != ['0=2560']:
                raise ValueError('Unexpected DiT modulation constant')
            row[0] = 'SeedVR2Constant'
            lines[i] = ' '.join(row)
            constant_count += 1
        if row[0] != 'Reshape':
            continue
        start = 4+int(row[2])+int(row[3])
        pattern = tuple(row[start:])
        # The six embedding Slice outputs also reshape to [2560]. Only the
        # first AWA output is a sequence and needs a two-dimensional view.
        if pattern == ('0=2560',) and row[4] != op[6]:
            continue
        if pattern in patterns:
            counts[pattern] += 1
            lines[i] = ' '.join(row[:start]+patterns[pattern])
    if any(count != 1 for count in counts.values()):
        raise ValueError(f'Unexpected pnnx block shape patterns: {counts}')
    if constant_count != (6 if index >= 10 else 12):
        raise ValueError(f'Unexpected DiT modulation constant count: {constant_count}')
    if not original.exists():
        param.rename(original)
    param.write_text('\n'.join(lines)+'\n')
    return dict(format=2, binary_rewritten=False, attributes='pnnx epsilon/norm_weights/rope_frequencies/spec',
                norm_weights_sha256=hashlib.sha256(norms.tobytes()).hexdigest(),
                constant_rewrites=constant_count,
                original_param_sha256=digest(original),
                shape_rewrites=[dict(before=list(p), after=patterns[p], count=counts[p]) for p in patterns])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'.cache/models/seedvr2_ema_3b.pth')
    parser.add_argument('--embeddings', type=Path, default=ROOT/'.cache/models/pos_emb.pt')
    parser.add_argument('--pnnx', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--blocks', default='0,1,10,11,31')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit('Output must be empty')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(60702)
    lock = json.loads((ROOT/'model-sources.lock.json').read_text())
    checkpoint_hash = digest(args.checkpoint)
    for name, path, actual in [('seedvr2_ema_3b.pth', args.checkpoint, checkpoint_hash),
                                ('pos_emb.pt', args.embeddings, digest(args.embeddings))]:
        entry = next(x for x in lock['files'] if x['rfilename'] == name)
        if actual != entry['lfs']['sha256'] or path.stat().st_size != entry['size']:
            raise ValueError('Official checkpoint hash mismatch')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    pos = torch.load(args.embeddings, map_location='cpu', weights_only=True).float()
    with torch.inference_mode():
        txt = F.linear(pos, state['txt_in.weight'], state['txt_in.bias'])
        emb = time_embedding(state)
    report = dict(schema_version='dit-block-export-v1',
        scope='Complete selected DiT blocks, actual official weights; independently sampled video inputs, official positive text embedding, timestep 1000. Not a 32-block denoising trajectory.',
        checkpoint=dict(repository=lock['repository'], revision=lock['revision'],
                        filename='seedvr2_ema_3b.pth', sha256=checkpoint_hash, tensors=len(state)),
        official_sources=verify_sources(), reference_profile='FP32-B',
        pnnx_sha256=digest(args.pnnx), torch_version=torch.__version__,
        ncnn_commit=json.loads((ROOT/'engine-dependencies.lock.json').read_text())['ncnn']['commit'],
        tolerance=dict(atol=1e-4, rtol=1e-3, calibration='DIAGNOSTIC_NOT_MODEL_CERTIFICATION'),
        scripts={name: digest(ROOT/'tools'/name) for name in
                 ['dit_block_reference.py', 'dit_block_module.py', 'awa_export_module.py', 'export_dit_block.py']},
        blocks=[], cases=[], model_verified=False)
    for index in [int(x) for x in args.blocks.split(',')]:
        if not 0 <= index < 32:
            raise ValueError('Invalid block index')
        print('block', index, 'strict official load', flush=True)
        reference, cache = make_block(state, index)
        candidate = DiTBlock(state, index).requires_grad_(False).eval()
        folder = args.output/f'block-{index}'
        folder.mkdir()
        sample = (torch.randn(1, 2, 3, 2560), txt, emb)
        with torch.inference_mode():
            traced = torch.jit.trace(candidate, sample, check_trace=False)
        traced.save(str(folder/'model.pt'))
        command = [str(args.pnnx.resolve()), 'model.pt',
            'inputshape=[1,2,3,2560],[58,2560],[2560,6]',
            'inputshape2=[2,3,4,2560],[59,2560],[2560,6]',
            'moduleop=awa_export_module.AdaptiveWindowAttention', 'fp16=0']
        print('block', index, 'pnnx export', flush=True)
        with (folder/'pnnx.log').open('w') as log:
            subprocess.run(command, cwd=folder, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=600, env={**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'})
        lowered = lower(folder, index)
        ops = sorted({line.split()[0] for line in (folder/'model.ncnn.param').read_text().splitlines()[2:]})
        if any('::' in op or op.startswith(('aten', 'prim', 'pnnx', 'awa_export')) for op in ops):
            raise ValueError(f'Unlowered DiT operator: {ops}')
        report['blocks'].append(dict(index=index, official_tensors=len(reference.state_dict()),
                                     operators=ops, awa_lowering=lowered))
        for grid in [(1, 2, 3), (2, 3, 4)]:
            case_id = f'dit-block-{index}-'+('x'.join(map(str, grid)))
            case = args.output/case_id
            case.mkdir()
            with torch.inference_mode():
                vid = F.linear(torch.randn(*grid, 132), state['vid_in.proj.weight'], state['vid_in.proj.bias'])
                expected = evaluate(reference, cache, vid, txt, emb)
                actual = traced(vid, txt, emb)
            comparison = [error(a, b) for a, b in zip(actual, expected)]
            if not all(c['passed'] for c in comparison):
                raise ValueError(f'{case_id} TorchScript mismatch: {comparison}')
            doc = dict(schema_version='dit-block-case-v1', case_id=case_id, component='dit-block',
                       block_index=index, checkpoint_sha256=checkpoint_hash, reference_profile='FP32-B', precision='fp32',
                       inputs=[tensor(case/f'input-{i}.f32', v) for i, v in enumerate((vid, txt, emb))],
                       reference_video=tensor(case/'reference-video.f32', expected[0]),
                       reference_text=tensor(case/'reference-text.f32', expected[1]))
            for name in ['model.ncnn.param', 'model.ncnn.bin']:
                os.link(folder/name, case/name)
                doc['model_param' if name.endswith('param') else 'model_bin'] = dict(path=name, sha256=digest(folder/name))
            (case/'case.json').write_text(json.dumps(doc, indent=2)+'\n')
            report['cases'].append(dict(path=case_id+'/case.json', sha256=digest(case/'case.json'),
                                       scripted_vs_official=comparison))
            print(case_id, 'scripted PASS', comparison, flush=True)
        del reference, candidate, traced
        gc.collect()
    report['passed'] = True
    (args.output/'suite.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Selected full-block export PASS; native CPU/Vulkan execution pending', flush=True)


if __name__ == '__main__':
    main()
