#!/usr/bin/env python3
"""Negative-case checks for native tensor/graph boundaries and persisted diagnostics."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import struct
import subprocess
import tempfile


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('binary', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    binary = str(args.binary.resolve())
    root = Path(__file__).resolve().parents[1]
    fixture = root/'tests/fixtures/awa/t2-h3-w4-heads2-text5-regular'
    rows = []

    def run(name, arguments, expected, predicate):
        result = subprocess.run([binary, *arguments], capture_output=True, text=True, timeout=30)
        data = json.loads(result.stdout)
        passed = result.returncode == expected and bool(predicate(data))
        rows.append(dict(case=name, passed=passed, exit_code=result.returncode))
        if not passed:
            raise AssertionError((name, result.returncode, data, result.stderr))
        return data

    with tempfile.TemporaryDirectory(prefix='seedvr2-engine-check-') as temporary:
        folder = Path(temporary)
        base = json.loads((fixture/'case.json').read_text())
        mutations = [
            ('float-grid', lambda d: d.update(grid=[2.0, 3, 4])),
            ('float-tensor-shape', lambda d: d['video_qkv'].update(shape=[2.0, 3, 4, 768])),
            ('wrong-dtype', lambda d: d['text_qkv'].update(dtype='f16le')),
            ('wrong-layout', lambda d: d['video_qkv'].update(shape=[3, 2, 4, 768])),
            ('nonboolean-shift', lambda d: d.update(shifted=1)),
            ('graph-shift-mismatch', lambda d: d.update(shifted=True)),
            ('wrong-precision', lambda d: d.update(precision='fp16')),
            ('tampered-hash', lambda d: d['video_qkv'].update(sha256='0'*64)),
            ('escaping-artifact', lambda d: d['video_qkv'].update(path='../outside.f32')),
            ('missing-artifact', lambda d: d['video_qkv'].update(path='missing.f32')),
            ('oversized-dimension', lambda d: d.update(grid=[257, 3, 4])),
            ('duplicate-key', None), ('nan-input', None), ('nan-weights', None),
            ('wrong-input-length', None), ('symlink-input', None), ('nonempty-output', None),
        ]
        for name, mutate in mutations:
            case = folder/name
            shutil.copytree(fixture, case)
            doc = copy.deepcopy(base)
            if mutate:
                mutate(doc)
            if name in ['nan-input', 'nan-weights', 'wrong-input-length']:
                key = 'model_bin' if name == 'nan-weights' else 'video_qkv'
                p = case/doc[key]['path']
                value = p.read_bytes()
                p.write_bytes(value+b'\0\0\0\0' if name == 'wrong-input-length'
                              else struct.pack('<f', float('nan'))+value[4:])
                doc[key]['sha256'] = sha(p)
            if name == 'symlink-input':
                p = case/doc['video_qkv']['path'];p.unlink();p.symlink_to(fixture/doc['video_qkv']['path'])
            content = json.dumps(doc)
            if name == 'duplicate-key':
                content = '{"heads":2,'+content[1:]
            (case/'case.json').write_text(content)
            out = folder/(name+'-output')
            if name == 'nonempty-output':
                out.mkdir();(out/'keep.txt').write_text('preserve')
            run(name, ['engine','awa','--case',str(case/'case.json'),'--output',str(out)], 2,
                lambda x: x['error']['code'] == 'AWA_EXECUTION_FAILED')
            if name == 'nonempty-output':
                assert (out/'keep.txt').read_text() == 'preserve'
        run('invalid-device-index', ['engine','self-test','--gpu','-2'], 2,
            lambda x: x['error']['code'] == 'CLI_USAGE')
        db = folder/'workspace.sqlite3'
        success = run('offline-cpu-self-test-save', ['--database',str(db),'engine','self-test','--save'], 0,
            lambda x: x['passed'] and x['status'] == 'PASS' and not x['model_verified'])
        record = run('saved-self-test-is-readable', ['--database',str(db),'history','get','--id',success['record_id']], 0,
            lambda x: x['cases'] == success['cases'] and x['passed'])
        assert 'record_id' not in record
        failed = run('failed-gpu-self-test-retained', ['--database',str(db),'engine','self-test',
                     '--backend','vulkan','--gpu','64','--save'], 7,
            lambda x: not x['passed'] and x['status'] == 'FAIL' and len(x['cases']) == 2)
        run('failure-record-readable', ['--database',str(db),'history','get','--id',failed['record_id']], 0,
            lambda x: x['status'] == 'FAIL' and not x['model_verified'])
        future = folder/'future.sqlite3'
        with sqlite3.connect(future) as connection:
            connection.execute('PRAGMA user_version=99')
        run('newer-workspace-refused', ['--database',str(future),'history','list'], 5,
            lambda x: 'newer schema' in x['error']['message'])
        with sqlite3.connect(future) as connection:
            assert connection.execute('PRAGMA user_version').fetchone()[0] == 99
    report = dict(scope='Native case rejection, real CPU diagnostic, retained GPU failure and workspace downgrade refusal',
                  cases=rows, passed=all(x['passed'] for x in rows), total=len(rows), binary_sha256=sha(args.binary))
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(f'{len(rows)}/{len(rows)} PASS')


if __name__ == '__main__':
    main()
