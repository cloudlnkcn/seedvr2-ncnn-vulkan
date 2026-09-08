#!/usr/bin/env python3
"""Isolate accumulated DiT error using a reviewed trajectory and the native CLI.

official-start replaces only the first block's inputs; subsequent blocks consume
native outputs. teacher-forced replaces inputs at every block. Neither is an
end-to-end model acceptance run. The official reference is never rewritten.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from check_graph import compare
from pipeline_contract import bind_candidate, bind_reference, load_json, sha, tensor_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'sdk', 'reference', 'baseline', 'package', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--mode', choices=('official-start', 'native-start', 'teacher-forced'), required=True)
    parser.add_argument('--native-run', type=Path)
    parser.add_argument('--backend', choices=('cpu', 'vulkan'), default='vulkan')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--first', type=int, default=0)
    parser.add_argument('--last', type=int, default=31)
    args = parser.parse_args()
    if not 0 <= args.first <= args.last <= 31 or args.gpu < 0:
        parser.error('Choose ordered block indices 0..31 and a nonnegative GPU')
    if args.mode == 'native-start' and args.native_run is None:
        parser.error('native-start requires --native-run')
    ref, baseline, shapes, reference_sha, known = bind_reference(args.reference, args.baseline, 'video')
    if sha(args.package/'manifest.json') != known['model_manifest_sha256']:
        raise ValueError('Model package differs from reviewed trajectory')
    if shapes['patch-in'] != [320, 2560]:
        raise ValueError('This investigation expects the reviewed 17-frame 128x128 trajectory')
    official = {row['stage']: row['reference'] for row in ref['stages']}
    package = load_json(args.package/'manifest.json')
    graphs = {row['id']: row for row in package['graphs']}
    native = None
    if args.native_run is not None:
        native = load_json(args.native_run/'run.json')
        bind_candidate(native, baseline, 'video', shapes)
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output/'implementation'
    snapshot.mkdir()
    binary, sdk = snapshot/'seedvr2', snapshot/'libseedvr2.so.0'
    shutil.copy2(args.binary.resolve(), binary)
    shutil.copy2(args.sdk.resolve(), sdk)
    implementation = dict(executable_sha256=sha(binary), sdk_library_sha256=sha(sdk))
    env = dict(os.environ, LD_LIBRARY_PATH=str(snapshot.resolve()),
               VK_INSTANCE_LAYERS='VK_LAYER_KHRONOS_validation')
    report = dict(schema_version='seedvr2-dit-isolation-v1',
                  created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  mode=args.mode, backend=args.backend, first=args.first, last=args.last,
                  reference_report_sha256=reference_sha,
                  model_manifest_sha256=known['model_manifest_sha256'],
                  baseline_sha256=known['baseline_run_sha256'],
                  native_run_sha256=sha(args.native_run/'run.json') if native else None,
                  implementation=implementation, script_sha256=sha(Path(__file__)),
                  expected_blocks=list(range(args.first, args.last+1)), blocks=[],
                  completed=False, model_verified=False, tolerance=ref['tolerance'])
    previous = None
    for index in range(args.first, args.last+1):
        folder = args.output/f'block-{index:02d}'
        folder.mkdir()
        def copy_tensor(root, item, name, shape):
            source = tensor_path(root, item)
            shutil.copyfile(source, folder/name)
            return dict(path=name, dtype='f32le', shape=shape, sha256=sha(folder/name))
        names = ['patch-in', 'text-in'] if index == 0 else [f'block-{index-1:02d}-video', f'block-{index-1:02d}-text']
        if args.mode == 'teacher-forced' or (previous is None and args.mode == 'official-start'):
            sources = [(args.reference, official[name]) for name in names]
            input_kind = 'OFFICIAL'
        elif previous is None:
            sources = [(args.native_run, native['diagnostics'][name]) for name in names]
            input_kind = 'RETAINED_NATIVE'
        else:
            sources = [(previous[0], previous[1]['outputs'][branch]) for branch in ('video', 'text')]
            input_kind = 'NATIVE_PREVIOUS_BLOCK'
        inputs = [copy_tensor(root, item, f'input-{i}.f32', shape) for i, ((root, item), shape)
                  in enumerate(zip(sources, ([5, 8, 8, 2560], [58, 2560])))]
        inputs.append(copy_tensor(args.reference, official['time-in'], 'input-2.f32', [2560, 6]))
        case = dict(schema_version='dit-block-case-v1', case_id=folder.name,
                    component='dit-block', block_index=index, inputs=inputs,
                    precision='fp32', reference_profile='FP32-B',
                    provenance=dict(reference_report_sha256=reference_sha, input_kind=input_kind))
        for source_key, key, name in (('param', 'model_param', 'model.ncnn.param'),
                                     ('weights', 'model_bin', 'model.ncnn.bin')):
            item = graphs[folder.name][source_key]
            source = args.package/item['path']
            if source.stat().st_size != item['bytes'] or sha(source) != item['sha256']:
                raise ValueError('Graph identity differs from reviewed model')
            os.link(source, folder/name)
            case[key] = dict(path=name, sha256=item['sha256'])
        case_path = folder/'case.json'
        case_path.write_text(json.dumps(case, indent=2)+'\n')
        output = folder/'native'
        command = [str(binary.resolve()), 'engine', 'block', '--case', str(case_path.resolve()),
                   '--output', str(output.resolve()), '--backend', args.backend,
                   '--gpu', str(args.gpu), '--threads', '4']
        with (folder/'stdout.json').open('w') as stdout, (folder/'stderr.log').open('w') as stderr:
            completed = subprocess.run(command, env=env, stdout=stdout, stderr=stderr, timeout=180)
        run = load_json(folder/'stdout.json')
        if completed.returncode or run['status'] != 'EXECUTED' or run['implementation'] != implementation:
            raise ValueError('Native execution or frozen implementation identity differs')
        layers = run['layers']
        if (not layers or any(layer['backend'] != args.backend for layer in layers)
                or run['cpu_calls'] != (len(layers) if args.backend == 'cpu' else 0)
                or run['vulkan_calls'] != (len(layers) if args.backend == 'vulkan' else 0)
                or run['case_sha256'] != sha(case_path)):
            raise ValueError('Native dispatch/case contract differs')
        if any(token in (folder/'stderr.log').read_text().lower() for token in ('validation error', 'vuid-')):
            raise ValueError('Vulkan validation error')
        metrics = {}
        for branch in ('video', 'text'):
            target = dict(official[f'block-{index:02d}-{branch}'])
            target['shape'] = run['outputs'][branch]['shape']
            metrics[branch] = compare(target, run['outputs'][branch], args.reference, output, ref['tolerance'])
        row = dict(index=index, input_kind=input_kind, case_sha256=sha(case_path),
                   run_sha256=sha(folder/'stdout.json'), stderr_sha256=sha(folder/'stderr.log'), metrics=metrics)
        report['blocks'].append(row)
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(index, input_kind, {branch:{key:value[key] for key in ('max_abs','rmse','violations')} for branch,value in metrics.items()}, flush=True)
        previous = (output, run)
    if (implementation != dict(executable_sha256=sha(binary), sdk_library_sha256=sha(sdk))
            or [row['index'] for row in report['blocks']] != report['expected_blocks']):
        raise ValueError('Implementation/coverage changed during experiment')
    report['completed'] = True
    report['passed'] = all(m['passed'] for row in report['blocks'] for m in row['metrics'].values())
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
