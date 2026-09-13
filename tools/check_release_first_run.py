#!/usr/bin/env python3
"""Real downloaded-model CPU execution plus network-isolated deterministic replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    work = a.output.resolve()
    work.mkdir(parents=True, exist_ok=False)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    image = work/'输入 照片.jpg'
    shutil.copyfile(ROOT/'tests/fixtures/natural/astronaut-degraded.jpg', image)
    common = [str(a.binary.resolve()), 'run', '--model', str(a.model.resolve()), '--input', str(image),
              '--size', '256', '--backend', 'cpu', '--threads', '2', '--seed', '666', '--diagnostic-tensors']
    report = dict(schema_version='seedvr2-clean-machine-functional-v1', passed=False,
                  official_numerical_reference_tested=False, backend='cpu',
                  binary_sha256=sha(a.binary), system=dict(os_release=Path('/etc/os-release').read_text(),
                  cpu=Path('/proc/cpuinfo').read_text().split('\n\n')[0]), executions=[])
    try:
        for name in ['first', 'offline']:
            command = common+['--output', str(work/name)]
            if name == 'offline':
                script = work/'offline.sh'
                script.write_text('set -eu\ntest ! -e '+shlex.quote(str(ROOT/'CMakeLists.txt'))+'\ntest "$(wc -l < /proc/net/route)" -eq 1\nreadlink /proc/self/ns/net > '+shlex.quote(str(work/'network.txt'))+'\nexec '+shlex.join(command)+'\n')
                command = ['bwrap', '--die-with-parent', '--unshare-net', '--ro-bind', '/', '/',
                           '--bind', str(work), str(work), '--tmpfs', str(ROOT), '--dev', '/dev', '--proc', '/proc',
                           '--setenv', 'PATH', '/usr/bin:/bin', '--unsetenv', 'LD_LIBRARY_PATH',
                           '--chdir', str(work), '/bin/sh', str(script)]
            start = time.monotonic()
            with (work/(name+'.stdout')).open('w') as out, (work/(name+'.stderr')).open('w') as err:
                result = subprocess.run(command, stdout=out, stderr=err, timeout=1200)
            report['executions'].append(dict(name=name, exit_code=result.returncode, seconds=time.monotonic()-start))
            if result.returncode:
                raise ValueError(name+' execution failed; raw logs retained')
        first, offline = [json.loads((work/name/'run.json').read_text()) for name in ['first', 'offline']]
        from pipeline_contract import expected_shapes, tensor_contract
        # Full trajectory and raw noise must be present; an empty output set cannot pass.
        shapes = expected_shapes(first, 'image')
        if expected_shapes(offline, 'image') != shapes:
            raise ValueError('Replay geometry changed')
        shapes.update(noise=shapes['conditioned'], **{'posterior-noise': shapes['conditioned']})
        expected = set(shapes)
        if set(first['diagnostics']) != expected or set(offline['diagnostics']) != expected:
            raise ValueError('Unexpected tensor boundary contract')
        same = first['output']['sha256'] == offline['output']['sha256']
        for key in expected:
            x, y = first['diagnostics'][key], offline['diagnostics'][key]
            tensor_contract(x, key, shapes[key])
            tensor_contract(y, key, shapes[key])
            from pipeline_contract import tensor_path
            tensor_path(work/'first', x)
            tensor_path(work/'offline', y)
            same &= x['sha256'] == y['sha256']
        for name, doc in [('first', first), ('offline', offline)]:
            if sha(work/name/'output.png') != doc['output']['sha256']:
                raise ValueError('Output file identity differs')
        child = (work/'network.txt').read_text().strip()
        report.update(passed=bool(same and child != os.readlink('/proc/self/ns/net')),
                      exact_replay=same, diagnostic_boundaries=len(expected),
                      isolated_network_namespace=child, output_sha256=first['output']['sha256'],
                      scope='Independent CI machine; same downloaded binary/model twice; source hidden and network isolated on replay')
    except Exception as e:
        report['error'] = str(e)
    a.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'system'}, indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
