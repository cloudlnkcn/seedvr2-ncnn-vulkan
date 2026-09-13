#!/usr/bin/env python3
"""Export a storage-only candidate from a retained trace; reuse immutable references.

Run with the export Python environment. This creates diagnostic cases, not a
reviewed application model package. Activation and arithmetic remain FP32.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Existing block export suite.json')
    parser.add_argument('--block', type=int, required=True)
    parser.add_argument('--pnnx', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weight-storage', choices=['fp32', 'fp16', 'fp16-ieee'], default='fp16')
    args = parser.parse_args()
    source = args.source.resolve()
    original = json.loads(source.read_text())
    if original['reference_profile'] != 'FP32-B' or not 0 <= args.block < 32:
        raise ValueError('Expected FP32-B reference and block 0..31')
    selected = []
    for item in original['cases']:
        path = source.parent / item['path']
        if sha(path) != item['sha256']:
            raise ValueError('Source case identity changed')
        case = json.loads(path.read_text())
        if case['block_index'] != args.block:
            continue
        if case['checkpoint_sha256'] != original['checkpoint']['sha256']:
            raise ValueError('Case checkpoint mismatch')
        for entry in case['inputs'] + [case['reference_video'], case['reference_text'], case['model_param'], case['model_bin']]:
            if sha(path.parent / entry['path']) != entry['sha256']:
                raise ValueError('Source tensor/model identity changed')
        selected.append((path, case))
    if not selected:
        raise ValueError('No matching reference cases')
    target = args.output.resolve()
    target.mkdir(parents=True, exist_ok=True)
    if any(target.iterdir()):
        raise ValueError('Output must be empty')
    graph = target / f'block-{args.block}'
    graph.mkdir()
    trace = source.parent / f'block-{args.block}' / 'model.pt'
    if args.weight_storage == 'fp16-ieee':
        from dit_weight_storage import rewrite
        import shutil
        old_graph = trace.parent
        shutil.copyfile(old_graph / 'model.ncnn.param', graph / 'model.ncnn.param')
        lowering = dict(storage_audit=rewrite(graph / 'model.ncnn.param',
                        old_graph / 'model.ncnn.bin', graph / 'model.ncnn.bin'),
                        converter_sha256=sha(Path(__file__).with_name('dit_weight_storage.py')))
        command = []
    else:
        os.link(trace, graph / 'model.pt')
        command = [str(args.pnnx.resolve()), 'model.pt',
                   'inputshape=[1,2,3,2560],[58,2560],[2560,6]',
                   'inputshape2=[2,3,4,2560],[59,2560],[2560,6]',
                   'moduleop=awa_export_module.AdaptiveWindowAttention',
                   f'fp16={int(args.weight_storage == "fp16")}']
        with (graph / 'pnnx.log').open('w') as log:
            subprocess.run(command, cwd=graph, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=600,
                           env={**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'})
        from export_dit_block import lower
        lowering = lower(graph, args.block)
    candidate = dict(schema_version='dit-storage-experiment-v1',
                     scope='One real checkpoint-backed DiT block; storage conversion only; not full-model validation.',
                     reference_profile=original['reference_profile'], checkpoint=original['checkpoint'],
                     tolerance=original['tolerance'], cases=[], model_verified=False,
                     source_suite_sha256=sha(source), trace_sha256=sha(trace),
                     pnnx_sha256=sha(args.pnnx), command=command, lowering=lowering,
                     precision=dict(weight_storage=args.weight_storage, activation='fp32', arithmetic='fp32',
                                    note='pnnx tagged weights only; opaque custom attributes may remain FP32.'))
    for path, case in selected:
        folder = target / case['case_id']
        folder.mkdir()
        for entry in case['inputs'] + [case['reference_video'], case['reference_text']]:
            os.link(path.parent / entry['path'], folder / entry['path'])
        case['storage_experiment'] = candidate['precision']
        for key in ['model_param', 'model_bin']:
            name = case[key]['path']
            os.link(graph / name, folder / name)
            case[key]['sha256'] = sha(folder / name)
        write(folder / 'case.json', case)
        candidate['cases'].append(dict(path=str((folder / 'case.json').relative_to(target)),
                                       sha256=sha(folder / 'case.json'), source_case_sha256=sha(path)))
    baseline_bin = selected[0][0].parent / json.loads(selected[0][0].read_text())['model_bin']['path']
    before, after = baseline_bin.stat().st_size, (graph / 'model.ncnn.bin').stat().st_size
    candidate['storage_bytes'] = dict(baseline=before, candidate=after, reduction_fraction=1-after/before)
    write(target / 'suite.json', candidate)
    print(json.dumps(candidate['storage_bytes']))


if __name__ == '__main__':
    main()
