#!/usr/bin/env python3
"""Adversarial tests of the evidence auditor. SYNTHETIC: not SeedVR2 model validation."""
import argparse
import copy
import hashlib
import json
import math
import struct
import subprocess
import tempfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('cli', type=Path)
parser.add_argument('--output', type=Path)
args = parser.parse_args()
cli = str(args.cli.resolve())
root = Path(__file__).resolve().parents[1]
base = json.loads((root / 'examples/model-evidence-empty/manifest.json').read_text())
policy = json.loads((root / 'policies/seedvr2-3b.v1.json').read_text())
results = []


def invoke(argv, code):
    p = subprocess.run([cli, *argv], capture_output=True, text=True, timeout=20)
    assert p.returncode == code, (argv, p.returncode, p.stdout, p.stderr)
    assert not any(marker in p.stderr for marker in ['AddressSanitizer', 'runtime error:']), p.stderr
    return json.loads(p.stdout)


def check(name, condition):
    assert condition, name
    results.append({'case': name, 'status': 'PASS'})


with tempfile.TemporaryDirectory(prefix='seedvr2-auditor-test-') as temporary:
    folder = Path(temporary) / 'evidence'; folder.mkdir()
    db = Path(temporary) / 'workspace.sqlite3'

    def audit(manifest, code=6, extra=None):
        (folder / 'manifest.json').write_text(json.dumps(manifest))
        return invoke(['--database', str(db), 'models', 'audit', '--bundle', str(folder), *(extra or [])], code)

    empty = audit(base)
    check('empty-evidence-blocks-certification', empty['status'] == 'BLOCKED' and not empty['model_verified'] and empty['certificate'] is None and len(empty['gates']) == 12)
    check('policy-hash-binds-exact-file', empty['policy_sha256'] == hashlib.sha256((root / 'policies/seedvr2-3b.v1.json').read_bytes()).hexdigest())
    check('missing-scope-is-explicit', len(empty['missing_scope']) == 11 and empty['scope_binding'] == 'DECLARED_UNVERIFIED')
    check('independent-official-reference-required', 'REFERENCE_NOT_REGISTERED' in empty['blockers'])

    def artifact(name, values):
        data = struct.pack('<' + 'f' * len(values), *values)
        (folder / (name + '.f32')).write_bytes(data)
        return {'id': name, 'role': 'tensor', 'path': name + '.f32', 'sha256': hashlib.sha256(data).hexdigest()}

    values = [0, 1, -2, 1e-7, 1000]
    good = copy.deepcopy(base)
    good['artifacts'] = [artifact('reference', values), artifact('candidate', values)]
    good['tensor_pairs'] = [{'id': 'synthetic-auditor-fixture', 'gate_id': 'M04', 'reference': 'reference', 'candidate': 'candidate', 'shape': [1,5], 'dtype': 'f32le'}]
    result = audit(good)
    check('raw-tensor-equality-measured', result['tensor_checks'][0]['max_abs'] == 0 and result['tensor_checks'][0]['diagnostic_status'] == 'PASS')
    check('tensor-pass-does-not-certify-model', result['status'] == 'BLOCKED' and not result['model_verified'] and result['certificate'] is None)
    bad = copy.deepcopy(good)
    bad_values = [0, 1.1, -2, 1e-7, 1000]
    bad['artifacts'][1] = artifact('candidate', bad_values)
    result = audit(bad)
    actual_delta = struct.unpack('<f', struct.pack('<f', 1.1))[0] - 1
    check('error-metrics-match-independent-calculation', math.isclose(result['tensor_checks'][0]['max_abs'], actual_delta, abs_tol=1e-12) and math.isclose(result['tensor_checks'][0]['rmse'], actual_delta / math.sqrt(5), abs_tol=1e-12))
    check('numeric-failure-propagates-to-gate', result['status'] == 'FAILED' and next(g for g in result['gates'] if g['id']=='M04')['status'] == 'FAILED')
    result = audit(good)
    check('tampered-file-hash-rejected', result['status'] == 'FAILED' and result['artifact_checks'][1]['status'] == 'HASH_MISMATCH' and result['tensor_checks'][0]['diagnostic_status'] == 'BLOCKED')
    for label, value in [('nan', float('nan')), ('infinity', float('inf'))]:
        nonfinite = copy.deepcopy(good)
        nonfinite['artifacts'][1] = artifact('candidate', [value, 1, -2, 1e-7, 1000])
        result = audit(nonfinite)
        check(label + '-cannot-pass', result['status'] == 'FAILED' and result['tensor_checks'][0]['nonfinite_pairs'] == 1 and result['tensor_checks'][0]['rmse'] is None)
    good['artifacts'][1] = artifact('candidate', values)
    oversized_shape = copy.deepcopy(good); oversized_shape['tensor_pairs'][0]['shape'] = [6]
    check('tensor-file-size-must-match-shape', audit(oversized_shape, 2)['error']['code'] == 'TENSOR_SIZE')
    for name, shape in [('zero', [0]), ('empty', []), ('float', [5.0]), ('boolean', [True]), ('overflow', [10**9,10**9])]:
        invalid = copy.deepcopy(good); invalid['tensor_pairs'][0]['shape'] = shape
        check(name + '-shape-rejected', audit(invalid, 2)['error']['code'] == 'TENSOR_SHAPE')
    invalid = copy.deepcopy(good); invalid['tensor_pairs'][0]['reference'] = 'unknown'
    check('unknown-artifact-id-rejected', audit(invalid, 2)['error']['code'] == 'ARTIFACT_REFERENCE')
    invalid = copy.deepcopy(good); invalid['artifacts'].append(invalid['artifacts'][0])
    check('duplicate-artifact-id-rejected', audit(invalid, 2)['error']['code'] == 'EVIDENCE_SCHEMA')
    for name, path in [('traversal','../outside'), ('absolute',str(Path(temporary)/'outside')), ('backslash','..\\outside')]:
        invalid = copy.deepcopy(good); invalid['artifacts'][0]['path'] = path
        check(name + '-path-rejected', audit(invalid, 2)['error']['code'] == 'ARTIFACT_PATH')
    (folder / 'link').symlink_to(folder / 'reference.f32')
    invalid = copy.deepcopy(good); invalid['artifacts'][0]['path'] = 'link'
    check('symlink-rejected', audit(invalid, 2)['error']['code'] == 'ARTIFACT_PATH')
    for field, value in [('model_verified',True), ('certificate',{'status':'PASS'}), ('calibration_status','FROZEN')]:
        invalid = copy.deepcopy(base); invalid[field] = value
        check('self-reported-' + field + '-rejected', audit(invalid, 2)['error']['code'] == 'EVIDENCE_SCHEMA')
    (folder/'manifest.json').write_text('{"schema_version":"1.0","schema_version":"1.0"}')
    check('duplicate-json-fields-rejected', invoke(['models','audit','--bundle',str(folder)], 2)['error']['code'] == 'DUPLICATE_KEY')
    all_roles = copy.deepcopy(base)
    for role in sorted({r for g in policy['gates'] for r in g['required_artifact_roles']}):
        all_roles['artifacts'].append({**good['artifacts'][0], 'id': role, 'role': role})
    result = audit(all_roles)
    check('complete-file-labels-cannot-forge-certification', result['status'] == 'BLOCKED' and all(g['status'] == 'EVIDENCE_PRESENT_UNREVIEWED' for g in result['gates']) and not result['model_verified'])
    result = audit(good, extra=['--save'])
    entries = invoke(['--database',str(db),'history','list','--kind','model-audit'], 0)
    check('audit-persists-across-process-restart', len(entries['items']) == 1 and entries['items'][0]['id'] == result['record_id'])
    saved = invoke(['--database',str(db),'history','get','--id',result['record_id']], 0)
    check('saved-audit-retains-negative-status-and-metrics', saved['status'] == 'BLOCKED' and saved['tensor_checks'][0]['max_abs'] == 0 and not saved['model_verified'])
    invalid = copy.deepcopy(base); invalid['certificate'] = True
    audit(invalid, 2, extra=['--save'])
    check('malformed-evidence-is-not-persisted', len(invoke(['--database',str(db),'history','list','--kind','model-audit'], 0)['items']) == 1)

report = {'scope':'Synthetic adversarial checks of the model auditor and SQLite persistence; NO SeedVR2 weights or model execution', 'status':'PASS', 'passed':len(results), 'cases':results}
if args.output: args.output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
