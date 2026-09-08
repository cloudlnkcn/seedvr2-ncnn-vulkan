#!/usr/bin/env python3
"""Compare native submodel outputs to hashed, independently produced tensors."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def compare(ref, candidate, reference_root, output_root, tolerance):
    ref_file, out_file = reference_root/ref['path'], output_root/candidate['path']
    if sha(ref_file) != ref['sha256'] or sha(out_file) != candidate['sha256']:
        raise ValueError('Tensor hash mismatch')
    x = np.fromfile(ref_file, dtype='<f4').astype(np.float64)
    y = np.fromfile(out_file, dtype='<f4').astype(np.float64)
    if x.shape != y.shape or ref['shape'] != candidate['shape'] or x.size != np.prod(ref['shape']):
        raise ValueError('Tensor shape mismatch')
    delta = np.abs(x-y)
    violations = int(np.sum(delta > tolerance['atol']+tolerance['rtol']*np.abs(x)))
    finite = bool(np.isfinite(x).all() and np.isfinite(y).all())
    return dict(reference_sha256=ref['sha256'], candidate_sha256=candidate['sha256'],
                shape=ref['shape'], max_abs=float(delta.max()), rmse=float(np.sqrt(np.mean(delta**2))),
                violations=violations, finite=finite, passed=finite and violations == 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--backends', default='cpu,vulkan')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--validation-layer', action='store_true')
    parser.add_argument('--weight-io', choices=['buffered','mapped'], default='buffered')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit('Output must be empty')
    args.output.mkdir(parents=True, exist_ok=True)
    suite = json.loads(args.suite.read_text())
    rows = []
    for item in suite['cases']:
        case_path = args.suite.parent/item['path']
        if sha(case_path) != item['sha256']:
            raise ValueError('Case changed since reference generation')
        case = json.loads(case_path.read_text())
        dit = case['schema_version'] == 'dit-block-case-v1'
        for backend in args.backends.split(','):
            folder = args.output/(case['case_id']+'-'+backend)
            command = [str(args.binary.resolve()), 'engine', 'block' if dit else 'graph', '--case', str(case_path.resolve()),
                       '--output', str(folder.resolve()), '--backend', backend, '--gpu', str(args.gpu)]
            env = os.environ.copy()
            if args.weight_io!='buffered': command += ['--weight-io',args.weight_io]
            if args.validation_layer:
                env['VK_INSTANCE_LAYERS'] = 'VK_LAYER_KHRONOS_validation'
            result = subprocess.run(command, capture_output=True, text=True, timeout=180, env=env)
            (args.output/(folder.name+'.stderr.log')).write_text(result.stderr)
            (args.output/(folder.name+'.stdout.log')).write_text(result.stdout)
            row = dict(case_id=case['case_id'], backend=backend, exit_code=result.returncode, passed=False)
            try:
                run = json.loads(result.stdout)
            except json.JSONDecodeError:
                run = None
            if result.returncode == 0 and run is not None:
                comparisons = {name: compare(case['reference_'+name], run['outputs'][name],
                    case_path.parent, folder, suite['tolerance']) for name in ('video', 'text')} if dit else {
                    'output': compare(case['reference'], run['output'], case_path.parent, folder, suite['tolerance'])}
                layer_count = len(run['layers'])
                dispatch_ok = (layer_count > 0 and all(layer['backend'] == backend for layer in run['layers'])
                               and run['cpu_calls'] == (layer_count if backend == 'cpu' else 0)
                               and run['vulkan_calls'] == (layer_count if backend == 'vulkan' else 0))
                if dit:
                    dispatch_ok &= (run['awa_dispatch']['cpu_calls'] == int(backend == 'cpu')
                                    and run['awa_dispatch']['vulkan_calls'] == int(backend == 'vulkan'))
                diagnostics_ok = not any(token in result.stderr.lower() for token in
                                         ['validation error', 'vuid-', 'not match', 'failed'])
                diagnostics_ok &= not any(token in result.stdout.lower() for token in ['validation error', 'vuid-'])
                row.update(execution=run, comparison=comparisons if dit else comparisons['output'],
                           passed=all(c['passed'] for c in comparisons.values()) and dispatch_ok and diagnostics_ok)
            else:
                row['error'] = result.stdout[:2000] or result.stderr[:2000] or 'No JSON result'
            rows.append(row)
            print(case['case_id'], backend, 'PASS' if row['passed'] else 'FAIL',
                  row.get('comparison', row.get('error')), flush=True)
    report = dict(schema_version='ncnn-graph-validation-v1', scope=suite['scope'],
                  reference_profile=suite['reference_profile'], checkpoint=suite['checkpoint'], weight_io=args.weight_io,
                  suite_sha256=sha(args.suite), binary_sha256=sha(args.binary),
                  threshold=suite['tolerance'], validation_layer_requested=args.validation_layer,
                  cases=rows, passed=bool(rows) and all(row['passed'] for row in rows),
                  passed_count=sum(row['passed'] for row in rows), total=len(rows), model_verified=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
