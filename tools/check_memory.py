#!/usr/bin/env python3
"""Serial, real-component validation of Vulkan weight-placement policies.

Reuses saved official tensors and historical suite tolerances. A very large
reserve exercises auto->RAM without physically exhausting the GPU. This is
correctness validation, not a timing benchmark or full-model certification.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from check_graph import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--dit-suite', type=Path, required=True)
    parser.add_argument('--vae-suite', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new evidence directory; previous failures are retained')
    args.output.mkdir(parents=True)
    report = dict(schema_version='seedvr2-memory-components-v1', model_verified=False,
                  scope='Same-input real components; budgets are controlled, not physical exhaustion',
                  executable_sha256=sha(args.binary), passed=False, cases=[], modes=[])
    identities = {str(p.resolve()): sha(p) for p in (args.binary, args.dit_suite, args.vae_suite)}
    comparisons = {}
    try:
        status = subprocess.run([str(args.binary.resolve()), 'engine', 'status'],
                                capture_output=True, text=True, check=True, timeout=30)
        implementation = json.loads(status.stdout)['implementation']
        report['implementation'] = implementation
        for part, suite in [('dit', args.dit_suite), ('vae', args.vae_suite)]:
            for mode, flags in [('device', ['--weights', 'device']),
                                ('auto', ['--weights', 'auto']),
                                ('auto-host', ['--weights', 'auto', '--gpu-reserve-mib', '1048576'])]:
                for path, digest in identities.items():
                    if sha(Path(path)) != digest:
                        raise ValueError('Frozen binary or suite changed')
                stem = part+'-'+mode
                output = args.output/stem
                result_path = args.output/(stem+'.json')
                command = [sys.executable, str(Path(__file__).with_name('check_graph.py')),
                           '--binary', str(args.binary), '--suite', str(suite), '--output', str(output),
                           '--report', str(result_path), '--backends', 'vulkan', '--validation-layer', *flags]
                with (args.output/(stem+'.log')).open('w') as log:
                    completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
                row = dict(part=part, mode=mode, exit_code=completed.returncode, command=command,
                           suite_sha256=sha(suite), report=str(result_path.relative_to(args.output)))
                report['modes'].append(row)
                if completed.returncode:
                    raise ValueError('Component validation failed: '+stem)
                native = json.loads(result_path.read_text())
                expected = json.loads(suite.read_text())['cases']
                if not expected or len(native['cases']) != len(expected) or not native['passed']:
                    raise ValueError('Missing component comparisons: '+stem)
                for case in native['cases']:
                    run = case['execution']
                    memory = run['memory']
                    requested = memory['requested_memory']
                    if memory['policy'] != ('device' if mode == 'device' else 'auto'):
                        raise ValueError('Requested policy lost')
                    if mode == 'device' and requested != 'device':
                        raise ValueError('Device override lost')
                    if mode == 'auto-host' and (requested != 'host' or memory['reason'] != 'budget_pressure'):
                        raise ValueError('Controlled budget branch not exercised')
                    if (run['implementation'] != implementation or
                            run['implementation']['executable_sha256'] != report['executable_sha256']):
                        raise ValueError('Executable, loaded SDK or allocator identity changed')
                    outputs = run['outputs'] if part == 'dit' else {'output': run['output']}
                    hashes = {key: value['sha256'] for key, value in outputs.items()}
                    key = (part, case['case_id'])
                    if mode == 'device':
                        comparisons[key] = hashes
                    equal = comparisons[key] == hashes
                    report['cases'].append(dict(part=part, mode=mode, case_id=case['case_id'],
                        official_parity=case['passed'], bitwise_equal_to_device=equal,
                        memory=memory, hashes=hashes))
                    if not equal:
                        raise ValueError('Weight placement changed component output bytes')
                print(stem, len(native['cases']), 'PASS', flush=True)
        report['passed'] = bool(report['cases']) and all(
            row['official_parity'] and row['bitwise_equal_to_device'] for row in report['cases'])
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        report['error'] = str(error)
    finally:
        report['identities'] = identities
        (args.output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
