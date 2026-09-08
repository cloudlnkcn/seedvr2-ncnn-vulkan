#!/usr/bin/env python3
"""Run isolated, explicit Vulkan precision probes and retain negative results.

This compares against the unchanged FP32 diagnostic budget. A low precision
miss is a measurement, not by itself an ncnn bug or a model certification.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from pipeline_contract import bind_reference, load_json, sha, tensor_path

MODES = ('fp32', 'fp16-storage', 'fp16-arithmetic', 'fp16-packed-arithmetic',
         'bf16-storage', 'bf16-packed')
CASES = ('erf', 'celu', 'reduction', 'reduction-local', 'sdpa', 'patch-in')


def prepare_patch_in(reference, baseline, package, output):
    ref, run, shapes, digest, known = bind_reference(reference, baseline, 'video')
    if sha(package/'manifest.json') != known['model_manifest_sha256']:
        raise ValueError('Package differs from reviewed official trajectory')
    if shapes['patch-in'] != [320, 2560] or shapes['conditioned'] != [16, 5, 16, 16]:
        raise ValueError('Probe expects the reviewed 17-frame, 128x128 fixture')
    stages = {row['stage']: row['reference'] for row in ref['stages']}
    conditioned_path = tensor_path(reference, stages['conditioned'])
    noise_path = tensor_path(baseline, run['diagnostics']['noise'])
    target_path = tensor_path(reference, stages['patch-in'])
    conditioned = np.fromfile(conditioned_path, dtype='<f4').reshape(16, 5, 16, 16)
    noise = np.fromfile(noise_path, dtype='<f4').reshape(conditioned.shape)
    pixels = np.concatenate((noise, conditioned, np.ones((1, 5, 16, 16), dtype='<f4')), axis=0)
    # Official [T, H, W, patch_h, patch_w, channels] patch order.
    patches = pixels.transpose(1, 2, 3, 0).reshape(5, 8, 2, 8, 2, 33).transpose(0, 1, 3, 2, 4, 5).reshape(320, 132)
    output.mkdir()
    patches.astype('<f4').tofile(output/'input.f32')
    shutil.copyfile(target_path, output/'reference.f32')
    def row(name, shape=None):
        p = output/name
        result = {'path': name, 'sha256': sha(p), 'bytes': p.stat().st_size}
        if shape is not None:
            result.update(dtype='f32le', shape=shape)
        return result
    graphs = load_json(package/'manifest.json')['graphs']
    selected = [g for g in graphs if g['id'] == 'patch-in']
    if len(selected) != 1:
        raise ValueError('Expected exactly one patch-in graph')
    for key, name in (('param', 'model.ncnn.param'), ('weights', 'model.ncnn.bin')):
        source = package/selected[0][key]['path']
        if sha(source) != selected[0][key]['sha256']:
            raise ValueError('Real component weights/graph changed')
        shutil.copyfile(source, output/name)
    doc = dict(schema_version='seedvr2-patch-in-precision-case-v1',
               input=row('input.f32', [320, 132]), reference=row('reference.f32', [320, 2560]),
               param=row('model.ncnn.param'), weights=row('model.ncnn.bin'),
               provenance=dict(reference_report_sha256=digest,
                               baseline_run_sha256=known['baseline_run_sha256'],
                               model_manifest_sha256=known['model_manifest_sha256'],
                               official_conditioned_sha256=sha(conditioned_path),
                               frozen_noise_sha256=sha(noise_path),
                               original_reference_sha256=sha(target_path),
                               generator_sha256=sha(Path(__file__))))
    path = output/'case.json'
    path.write_text(json.dumps(doc, indent=2)+'\n')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES[:-1]))
    parser.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--package', type=Path)
    args = parser.parse_args()
    if args.gpu < 0 or len(set(args.cases)) != len(args.cases) or len(set(args.modes)) != len(args.modes):
        parser.error('Use a nonnegative GPU index and distinct cases/modes')
    if 'patch-in' in args.cases and not all((args.reference, args.baseline, args.package)):
        parser.error('patch-in requires --reference, --baseline and --package')
    args.output.mkdir(parents=True, exist_ok=False)
    # Snapshot the executable so builds cannot change an active experiment.
    binary = args.output/'probe-binary'
    shutil.copy2(args.binary.resolve(), binary)
    binary = binary.resolve()
    binary_sha = sha(binary)
    graph_case = None
    if 'patch-in' in args.cases:
        graph_case = prepare_patch_in(args.reference, args.baseline, args.package,
                                     args.output/'patch-in-fixture').resolve()
    expected = [f'{case}/{mode}' for case in args.cases for mode in args.modes]
    report = dict(schema_version='ncnn-precision-matrix-v1',
                  created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  binary_sha256=binary_sha, runner_sha256=sha(Path(__file__)),
                  expected_cases=expected, gpu=args.gpu, cases=[], model_verified=False,
                  scope='Synthetic upstream kernels and optional same-input official patch-in; not full SeedVR2 low precision')
    root = Path(__file__).resolve().parents[1]
    report['sources'] = {name: sha(root/name) for name in (
        'tools/ncnn_precision_probe/main.cpp', 'tools/ncnn_precision_probe/CMakeLists.txt',
        'src/engine/ncnn/graph.hpp', 'tools/pipeline_contract.py',
        'tests/reference/reviewed-pipelines.json')}
    env = os.environ.copy()
    env['VK_INSTANCE_LAYERS'] = 'VK_LAYER_KHRONOS_validation'
    for case in args.cases:
        for mode in args.modes:
            stem = case+'-'+mode
            folder = args.output/stem
            command = [str(binary), case, mode, str(args.gpu), str(folder.resolve())]
            if case == 'patch-in':
                command.append(str(graph_case))
            stdout_path, stderr_path = args.output/(stem+'.json'), args.output/(stem+'.stderr')
            with stdout_path.open('w') as stdout, stderr_path.open('w') as stderr:
                try:
                    completed = subprocess.run(command, stdout=stdout, stderr=stderr, env=env, timeout=60)
                    code = completed.returncode
                except subprocess.TimeoutExpired:
                    code = -999
            try:
                result = load_json(stdout_path)
                if result.get('case') != case or result.get('precision') != mode:
                    raise ValueError('Case identity differs')
            except (ValueError, OSError) as exc:
                result = dict(status='ERROR', reason=f'Missing/malformed native report: {exc}')
            status = result['status']
            error_log = stderr_path.read_text(errors='replace')
            validation_errors = error_log.count('Validation Error') + error_log.count('VUID-')
            item = dict(id=case+'/'+mode, returncode=code, report=stdout_path.name,
                        report_sha256=sha(stdout_path), stderr=stderr_path.name,
                        stderr_sha256=sha(stderr_path), status=status,
                        validation_layer_requested=True, validation_error_mentions=validation_errors,
                        metrics=result.get('metrics'), reason=result.get('reason'))
            if status == 'EXECUTED' and code != (0 if result.get('metrics', {}).get('passed') else 1):
                item.update(status='ERROR', reason='Native status and return code disagree')
            if validation_errors:
                item.update(status='ERROR', reason='Vulkan validation error; retain logs')
            report['cases'].append(item)
            print(stem, item['status'], item.get('metrics'), flush=True)
    if [x['id'] for x in report['cases']] != expected or sha(binary) != binary_sha:
        raise ValueError('Experiment identity/coverage changed')
    report['executed'] = sum(x['status'] == 'EXECUTED' for x in report['cases'])
    report['skipped'] = sum(x['status'] == 'SKIPPED' for x in report['cases'])
    report['errors'] = sum(x['status'] == 'ERROR' for x in report['cases'])
    report['fp32_budget_passed'] = sum(x['status'] == 'EXECUTED' and x['metrics']['passed'] for x in report['cases'])
    report['passed'] = report['fp32_budget_passed'] == len(expected)
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
