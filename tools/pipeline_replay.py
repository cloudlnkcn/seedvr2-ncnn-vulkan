"""Shared replay for reviewed image and temporal-video references."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from check_graph import compare
from pipeline_contract import bind_candidate, bind_reference, load_json, sha, tensor_path


def replay(run_root, reference_root, baseline_root, kind):
    run_root, reference_root, baseline_root = map(Path, (run_root, reference_root, baseline_root))
    ref, baseline, expected, reference_digest, known = bind_reference(reference_root, baseline_root, kind)
    run = load_json(run_root/'run.json')
    bind_candidate(run, baseline, kind, expected)
    for name in ('posterior-noise', 'noise'):
        x, y = run['diagnostics'][name], baseline['diagnostics'][name]
        tensor_path(run_root, x)
        tensor_path(baseline_root, y)
        if x['sha256'] != y['sha256']:
            raise ValueError('Different raw noise')
    rows = []
    for stage in ref['stages']:
        name = stage['stage']
        tensor_path(reference_root, stage['reference'])
        tensor_path(run_root, run['diagnostics'][name])
        metrics = compare(stage['reference'], run['diagnostics'][name], reference_root, run_root, ref['tolerance'])
        tensor_path(reference_root, stage['reference'])
        tensor_path(run_root, run['diagnostics'][name])
        rows.append(dict(stage=name, **metrics))
    if sha(reference_root/'report.json') != reference_digest:
        raise ValueError('Reference changed during comparison')
    output = run['output']
    if output.get('path') != ('output.png' if kind == 'image' else 'output.mp4'):
        raise ValueError('Unexpected output artifact')
    if sha(run_root/output['path']) != output['sha256']:
        raise ValueError('Output identity differs')
    for suffix in ('stdout', 'stderr'):
        log = run_root.parent/(run_root.name+'.'+suffix)
        if log.exists() and any(x in log.read_text().lower() for x in ('vuid-', 'validation error')):
            raise ValueError('Retained Vulkan validation error')
    report = dict(schema_version=f'seedvr2-{kind}-replay-parity-v1', passed=all(x['passed'] for x in rows),
                  model_verified=False, backend=run['backend'], run_sha256=sha(run_root/'run.json'),
                  reference_report_sha256=reference_digest, reference_run_sha256=sha(baseline_root/'run.json'),
                  input_and_noise_identity_verified=True, tolerance=ref['tolerance'], stages=rows,
                  validation_contract='reviewed-reference-exact-73-v1',
                  scripts={name: sha(Path(__file__).parent/name) for name in
                           ('pipeline_contract.py', 'pipeline_replay.py', 'check_graph.py')})
    if kind == 'image':
        reference_png = reference_root/'reference.png'
        if sha(reference_png) != known['reference_png_sha256']:
            raise ValueError('Reference PNG identity differs')
        a = np.asarray(Image.open(run_root/'output.png').convert('RGB')).astype(np.int16)
        b = np.asarray(Image.open(reference_png).convert('RGB')).astype(np.int16)
        if a.shape != b.shape or a.shape != (run['output']['height'], run['output']['width'], 3):
            raise ValueError('Output PNG dimensions differ')
        delta = np.abs(a-b)
        report['pixel_max_abs'] = int(delta.max())
        report['pixel_mae'] = float(delta.mean())
        report['passed'] &= report['pixel_max_abs'] <= ref['tolerance']['max_uint8_error']
    return report


def main(kind):
    parser = argparse.ArgumentParser()
    for name in ('run', 'reference', 'reference-run', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output report already exists; preserve prior evidence')
    try:
        report = replay(args.run, args.reference, args.reference_run, kind)
        code = 0 if report['passed'] else 1
    except (ValueError, KeyError, TypeError, OSError) as error:
        report = dict(schema_version=f'seedvr2-{kind}-replay-parity-v1', passed=False,
                      model_verified=False, status='INVALID_EVIDENCE', error=str(error))
        code = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    rows = report.get('stages', [])
    print(report['passed'], sum(x['passed'] for x in rows), '/', len(rows), report.get('error', ''), flush=True)
    raise SystemExit(code)
