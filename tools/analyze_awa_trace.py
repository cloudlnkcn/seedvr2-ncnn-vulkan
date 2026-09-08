#!/usr/bin/env python3
"""Isolate real-block gather and SDPA rounding using retained native layer inputs.

This diagnostic uses teacher forcing, never replaces a reviewed full-run golden,
and does not certify the model. All comparisons are on the same QKV tensors.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from awa_reference import AdaptiveWindowAttention, verify_sources
from pipeline_contract import load_json, sha, tensor_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output already exists')
    source = load_json(args.trace/'report.json')
    case = load_json(args.case)
    if (source['schema_version'] != 'seedvr2-dit-layer-trace-v1'
            or source['status'] != 'EXECUTED' or source['case_sha256'] != sha(args.case)):
        raise ValueError('Trace/case identity mismatch')
    if (case['schema_version'] != 'dit-block-case-v1' or case['precision'] != 'fp32'
            or case['block_index'] != 16
            or [r['shape'] for r in case['inputs']] != [[5, 8, 8, 2560], [58, 2560], [2560, 6]]):
        raise ValueError('This probe expects real block 16 of the 17-frame 3B fixture')
    lock = load_json(Path(__file__).resolve().parents[1]/'model-sources.lock.json')
    expected = next(row['lfs']['sha256'] for row in lock['files']
                    if row['rfilename'] == 'seedvr2_ema_3b.pth')
    if sha(args.checkpoint) != expected:
        raise ValueError('Checkpoint identity mismatch')
    torch.set_num_threads(4)
    layers = {row['name']: row['outputs'] for row in source['layers']}
    awa_names = [row['name'] for row in source['layers'] if row['type'] == 'SeedVR2AWA']
    if len(layers) != len(source['layers']) or len(awa_names) != 1 or len(layers[awa_names[0]]) != 2:
        raise ValueError('Duplicate boundaries or missing AWA outputs')

    def tensor(name, index=0):
        row = layers[name][index]
        return torch.from_numpy(np.fromfile(tensor_path(args.trace, row), '<f4')
                                .reshape(row['shape']).copy())

    def difference(actual, reference):
        delta = (actual.double() - reference.double()).abs()
        return dict(max_abs=delta.max().item(), rmse=delta.square().mean().sqrt().item(),
                    finite=bool(torch.isfinite(actual).all() and torch.isfinite(reference).all()))

    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    index = case['block_index']
    prefix = f'blocks.{index}.attn.'
    weights = torch.stack([state[prefix+f'norm_{axis}.{"all" if index >= 10 else branch}.weight']
                           for branch, axis in [('vid', 'q'), ('vid', 'k'), ('txt', 'q'), ('txt', 'k')]])
    reference = AdaptiveWindowAttention(heads=20, shifted=bool(index % 2)).eval()
    reference.set_norm_weights(weights)
    captured = {}
    def capture(module, inputs, kwargs):
        captured.update({key: kwargs[key].detach().clone() for key in
                         ('q', 'k', 'v', 'cu_seqlens_q')})
    reference.reference.attn.register_forward_pre_hook(capture, with_kwargs=True)
    with torch.inference_mode():
        outputs = reference(tensor('linear_6').reshape(*case['inputs'][0]['shape'][:3], 7680),
                            tensor('linear_7'))
    report = dict(schema_version='seedvr2-awa-isolation-v1', diagnostic_only=True,
                  model_verified=False, checkpoint_sha256=expected,
                  native_trace_sha256=sha(args.trace/'report.json'),
                  case_sha256=sha(args.case), official_sources=verify_sources(),
                  script_sha256=sha(Path(__file__)), torch_version=torch.__version__, windows=[])
    boundaries = captured['cu_seqlens_q'].tolist()
    if len(boundaries) < 2 or boundaries[0] != 0 or any(b <= a for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError('Empty or invalid attention windows')
    expected = {f'awa.{kind}.{i}' for kind in ('qkv', 'sdpa') for i in range(len(boundaries)-1)}
    actual = {name for name in layers if name.startswith(('awa.qkv.', 'awa.sdpa.'))}
    if expected != actual:
        raise ValueError('Incomplete or unexpected attention windows')
    with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        for window, (lo, hi) in enumerate(zip(boundaries, boundaries[1:])):
            qkv = [tensor(f'awa.qkv.{window}', i) for i in range(3)]
            native = tensor(f'awa.sdpa.{window}')
            official_qkv = [captured[key][lo:hi].transpose(0, 1) for key in ('q', 'k', 'v')]
            official = F.scaled_dot_product_attention(*qkv)
            high_precision = F.scaled_dot_product_attention(*[x.double() for x in qkv])
            report['windows'].append(dict(
                index=window, gather_on_native_projection=[difference(a, b) for a, b in zip(qkv, official_qkv)],
                gather_by_axis={axis: [difference(a[..., begin:end], b[..., begin:end])
                                      for a, b in zip(qkv[:2], official_qkv[:2])]
                                for axis, begin, end in [('time', 0, 42), ('height', 42, 84),
                                                         ('width', 84, 126), ('unrotated', 126, 128)]},
                sdpa_on_native_qkv=difference(native, official),
                native_sdpa_to_fp64=difference(native, high_precision),
                official_sdpa_to_fp64=difference(official, high_precision)))
    report['awa_on_native_projection'] = [difference(tensor(awa_names[0], i).reshape(out.shape), out)
                                           for i, out in enumerate(outputs)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report['windows'], indent=2))


if __name__ == '__main__':
    main()
