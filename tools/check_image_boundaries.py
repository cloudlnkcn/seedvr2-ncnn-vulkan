#!/usr/bin/env python3
"""Reject corrupted packages and invalid image requests using the actual CLI.

Every fixture has its own small manifest. Large immutable graph files are hard
linked and never modified. These tests cannot certify a model.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.model/'manifest.json').read_text())
    cases = []

    def check(name, change=None, options=(), setup=None, raw=None):
        root = args.output/name
        model = root/'model'
        model.mkdir(parents=True)
        doc = copy.deepcopy(manifest)
        for graph in doc['graphs']:
            for field in ['param', 'weights']:
                path = Path(graph[field]['path'])
                (model/path).parent.mkdir(parents=True, exist_ok=True)
                os.link(args.model/path, model/path)
        for row in doc['constants'].values():
            os.link(args.model/row['path'], model/row['path'])
        if change:
            change(doc)
        (model/'manifest.json').write_text(raw if raw is not None else json.dumps(doc))
        if setup:
            setup(model, doc)
        call = subprocess.run([str(args.binary.resolve()), 'run', '--model', str(model),
            '--input', str(args.input), '--output', str(root/'result'), '--size', '128',
            '--backend', 'cpu', *options], capture_output=True, text=True, timeout=45)
        (root/'stdout').write_text(call.stdout)
        (root/'stderr').write_text(call.stderr)
        passed = call.returncode != 0 and not (root/'result/output.png').exists()
        cases.append(dict(case=name, passed=passed, exit_code=call.returncode))
        print(name, 'PASS' if passed else 'FAIL', flush=True)
        if not passed:
            raise AssertionError(name)

    check('unsupported-profile', lambda d: d.update(profile='unknown'))
    check('unsupported-ncnn-commit', lambda d: d.update(ncnn_commit='0'*40))
    check('unsupported-sampling', lambda d: d['sampling'].update(steps=2))
    check('missing-graph', lambda d: d['graphs'].pop())
    check('duplicate-graph', lambda d: d['graphs'][0].update(id=d['graphs'][1]['id']))
    check('incorrect-param-hash', lambda d: d['graphs'][0]['param'].update(sha256='0'*64))
    check('incorrect-weight-hash', lambda d: d['graphs'][0]['weights'].update(sha256='0'*64))
    check('escape-package', lambda d: d['graphs'][0]['param'].update(path='../model/manifest.json'))

    def symlink(model, doc):
        row = doc['graphs'][0]['param']
        (model/row['path']).unlink()
        (model/row['path']).symlink_to((args.model/row['path']).resolve())
    check('symlink-artifact', setup=symlink)
    check('duplicate-json-key', raw='{"profile":"one","profile":"two"}')
    check('invalid-output-multiple', options=['--size', '129'])
    check('oversized-output', options=['--size', '1024'])

    report = dict(schema_version='seedvr2-image-boundaries-v1', passed=all(r['passed'] for r in cases),
        model_verified=False, scope='Actual CLI rejection and absence of completed output; no model quality claim',
        binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(), cases=cases)
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
