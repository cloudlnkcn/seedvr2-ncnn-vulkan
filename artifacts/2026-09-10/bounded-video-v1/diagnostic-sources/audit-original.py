#!/usr/bin/env python3
"""Recompute retained boundaries and source bindings without admitting references."""
import argparse
import json
from pathlib import Path

from check_graph import compare
from pipeline_contract import bind_candidate, load_json, reference_contract, sha, tensor_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execution', type=Path, required=True)
    p.add_argument('--fixtures', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve the previous audit; choose a new output path')
    plan = load_json(a.execution/'plan.json')
    fixture = load_json(a.fixtures/'manifest.json')
    if sha(a.fixtures/'manifest.json') != plan['fixture_manifest_sha256']:
        raise ValueError('Fixture declaration changed')
    frozen = a.execution/'implementation'
    for name, expected in plan['files'].items():
        if sha(frozen/name) != expected:
            raise ValueError('Frozen source or binary changed: '+name)
    official = load_json(frozen/'tests/reference/seedvr/sources.json')
    for entry in official['files']:
        if sha(frozen/'tests/reference/seedvr'/entry['path']) != entry['sha256']:
            raise ValueError('Pinned official body changed')
    result = []
    for case in fixture['cases']:
        name = case['case_id']
        native, reference = a.execution/name, a.execution/(name+'-reference')
        run, ref = load_json(native/'run.json'), load_json(reference/'report.json')
        expected = reference_contract(ref, run, 'video')
        bind_candidate(run, run, 'video', expected)
        if ref['native_run_sha256'] != sha(native/'run.json') or ref['original_sources'] != official:
            raise ValueError('Reference run or official-source binding differs')
        for script, digest in ref['scripts'].items():
            if digest != plan['files']['tools/'+script]:
                raise ValueError('Executed reference script differs')
        if run['model_manifest_sha256'] != plan['model_manifest_sha256'] or \
                run['input']['sha256'] != case['input']['sha256']:
            raise ValueError('Native model or input identity differs')
        for key in ('input', 'target'):
            if sha(a.fixtures/case[key]['path']) != case[key]['sha256']:
                raise ValueError('Fixture bytes differ')
        if (run['clip']['decoded_frames'], run['clip']['padded_frames'], run['clip']['latent_frames'],
                run['output']['width'], run['output']['height']) != (
                case['frames'], case['padded_frames'], case['latent_frames'], case['width'], case['height']):
            raise ValueError('Declared temporal/spatial geometry differs')
        if (run['implementation']['executable_sha256'] != plan['files']['install/bin/seedvr2'] or
                run['implementation']['sdk_library_sha256'] != plan['files']['install/lib64/libseedvr2.so.0.7.0']):
            raise ValueError('Executed native implementation differs')
        measurement = load_json(a.execution/(name+'.json'))
        if measurement['exit_code'] != 0 or measurement['timed_out']:
            raise ValueError('Native execution failed')
        for suffix in ('stdout', 'stderr'):
            text = (a.execution/(name+'.'+suffix)).read_text().lower()
            if 'vuid-' in text or 'validation error' in text:
                raise ValueError('Vulkan validation error')
        if run['backend'] != 'ncnn-vulkan' or len(run['stages']) != 36:
            raise ValueError('Incomplete native graph execution')
        if sha(native/'output.mp4') != run['output']['sha256']:
            raise ValueError('Native output identity differs')
        rows = []
        for old in ref['stages']:
            stage = old['stage']
            tensor_path(reference, old['reference'])
            tensor_path(native, run['diagnostics'][stage])
            new = compare(old['reference'], run['diagnostics'][stage], reference, native, ref['tolerance'])
            for key in ('max_abs', 'rmse', 'violations', 'finite', 'passed'):
                if new[key] != old[key]:
                    raise ValueError('Recomputed numerical result differs: '+name+'/'+stage+'/'+key)
            rows.append(dict(stage=stage, **new))
        if ref['passed'] != all(row['passed'] for row in rows):
            raise ValueError('Whole-report status differs')
        result.append(dict(case_id=name, source_and_tensor_audit_passed=True,
            native_run_sha256=sha(native/'run.json'), reference_report_sha256=sha(reference/'report.json'),
            numerical_passed=ref['passed'], passed_boundaries=sum(row['passed'] for row in rows),
            compared_elements=sum(__import__('math').prod(row['shape']) for row in rows), stages=rows))
    report = dict(schema_version='seedvr2-bounded-video-audit-v1', source_and_tensor_audit_passed=True,
        reference_registry_modified=False, model_verified=False, plan_sha256=sha(a.execution/'plan.json'),
        frozen_files_verified=len(plan['files']), official_files_verified=len(official['files']),
        scripts={n:sha(Path(__file__).parent/n) for n in
                 ('audit_bounded_video.py', 'pipeline_contract.py', 'check_graph.py')}, cases=result)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps([{k:r[k] for k in ('case_id', 'passed_boundaries', 'compared_elements')} for r in result], indent=2))


if __name__ == '__main__':
    main()
