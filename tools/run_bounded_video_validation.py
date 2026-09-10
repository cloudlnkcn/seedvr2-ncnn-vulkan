#!/usr/bin/env python3
"""Run the three declared clips serially with frozen native/reference tooling.

Fresh reference reports require a separate source/tensor review before registry
admission. Numerical failure is retained and does not become model certification.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from pipeline_contract import load_json, sha

ROOT = Path(__file__).resolve().parents[1]


def snapshot_native(binary, install):
    """Freeze only this application's SDK, including every SONAME as real bytes."""
    prefix = binary.absolute().parent.parent
    libraries = [path for folder in ('lib', 'lib64')
                 for path in (prefix/folder).glob('libseedvr2.so*') if path.is_file()]
    if not libraries:
        raise ValueError('Installed SeedVR2 SDK missing from lib/lib64')
    native = install/'bin/seedvr2'
    native.parent.mkdir(parents=True)
    shutil.copy2(binary, native)
    for source in libraries:
        destination = install/source.relative_to(prefix)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination, follow_symlinks=True)
    return native


def run_child(command, timeout, **kwargs):
    """Also stop descendants when a reference or measurement times out."""
    proc = subprocess.Popen(command, start_new_session=True, **kwargs)
    try:
        proc.wait(timeout=timeout)
    except BaseException:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        raise
    return proc.returncode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('fixtures', 'binary', 'model', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--python', type=Path, default=ROOT/'.venv-export/bin/python')
    a = p.parse_args()
    a.output = a.output.absolute()
    a.fixtures, a.model = a.fixtures.resolve(), a.model.resolve()
    declaration = load_json(a.fixtures/'manifest.json')
    cases = declaration['cases']
    if [(r['case_id'], r['frames'], r['width'], r['height']) for r in cases] != [
            ('motion-9', 9, 128, 80), ('padding-8', 8, 128, 80), ('cut-17', 17, 128, 80)]:
        p.error('Expected the fixed three-case bounded-video fixture manifest')
    for row in cases:
        for key in ('input', 'target'):
            path = a.fixtures/row[key]['path']
            if not path.resolve().is_relative_to(a.fixtures) or sha(path) != row[key]['sha256']:
                p.error('Fixture path or identity differs')
    if a.output.exists():
        p.error('Use a new output directory; completed or interrupted runs are preserved')
    (ROOT/'.cache').mkdir(exist_ok=True)
    with (ROOT/'.cache/bounded-video-validation.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        a.output.mkdir(parents=True)
        frozen = a.output/'implementation'
        shutil.copytree(ROOT/'tools', frozen/'tools', ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(ROOT/'tests/reference', frozen/'tests/reference')
        for name in ('model-sources.lock.json', 'converter-dependencies.lock.json',
                     'dependencies.lock.json'):
            if (ROOT/name).is_file():
                shutil.copy2(ROOT/name, frozen/name)
        (frozen/'.cache').mkdir()
        (frozen/'.cache/models').symlink_to((ROOT/'.cache/models').resolve(), target_is_directory=True)
        native = snapshot_native(a.binary, frozen/'install')
        bindings = {str(f.relative_to(frozen)): sha(f) for f in sorted(frozen.rglob('*'))
                    if f.is_file() and not f.is_symlink() and '__pycache__' not in f.parts}
        plan = dict(schema_version='seedvr2-bounded-video-execution-v1',
            source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            fixture_manifest_sha256=sha(a.fixtures/'manifest.json'),
            model_manifest_sha256=sha(a.model/'manifest.json'),
            model_path=str(a.model), original_binary=str(a.binary.absolute()),
            reference_python=str(a.python.absolute()), files=bindings, cases=cases,
            protocol_sha256=sha(ROOT/'docs/BOUNDED-VIDEO-VALIDATION.md'),
            model_verified=False, reference_review='PENDING_INDEPENDENT_SOURCE_AND_TENSOR_REVIEW')
        (a.output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')

        def verify():
            for relative, expected in bindings.items():
                if sha(frozen/relative) != expected:
                    raise ValueError('Frozen implementation changed: '+relative)
            if sha(a.fixtures/'manifest.json') != plan['fixture_manifest_sha256'] or \
                    sha(a.model/'manifest.json') != plan['model_manifest_sha256']:
                raise ValueError('Frozen fixture/model manifest changed')

        rows = []
        env = os.environ.copy()
        env.update(VK_INSTANCE_LAYERS='VK_LAYER_KHRONOS_validation',
                   LD_LIBRARY_PATH=os.pathsep.join(str(frozen/'install'/name) for name in ('lib','lib64')),
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1')
        for case in cases:
            verify()
            name = case['case_id']
            run, reference = a.output/name, a.output/(name+'-reference')
            command = [str(native), 'run-video', '--model', str(a.model),
                       '--input', str(a.fixtures/case['input']['path']), '--output', str(run),
                       '--frames', str(case['frames']), '--size', '128', '--backend', 'vulkan',
                       '--gpu', '0', '--threads', '4', '--seed', '666',
                       '--diagnostic-tensors', '--weights', 'auto']
            measured = [str(a.python.absolute()), str(frozen/'tools/measure_native.py'),
                        '--report', str(a.output/(name+'.json')), '--timeout', '600',
                        '--sample-gpu', '--'] + command
            print(name, 'native start', flush=True)
            native_code = run_child(measured, env=env, timeout=650)
            if native_code:
                raise RuntimeError(name+': native execution failed; retained measurement/logs')
            check = [str(a.python.absolute()), str(frozen/'tools/check_video.py'),
                     '--run', str(run), '--input', str(a.fixtures/case['input']['path']),
                     '--output', str(reference)]
            print(name, 'official FP32-B start', flush=True)
            started = time.monotonic()
            with (a.output/(name+'-reference.stdout')).open('w') as out, \
                    (a.output/(name+'-reference.stderr')).open('w') as err:
                reference_code = run_child(['/usr/bin/time', '-v', '-o',
                                            str(a.output/(name+'-reference.resources.txt'))]+check,
                                           env=env, stdout=out, stderr=err, timeout=900)
            verify()
            if reference_code not in (0, 1) or not (reference/'report.json').is_file():
                raise RuntimeError(name+': official execution failed; see retained logs')
            report = load_json(reference/'report.json')
            rows.append(dict(case_id=name, native_exit_code=native_code,
                reference_exit_code=reference_code, reference_seconds=time.monotonic()-started,
                reference_report_sha256=sha(reference/'report.json'),
                native_run_sha256=sha(run/'run.json'),
                numerical_passed=report['passed'], passed_boundaries=sum(s['passed'] for s in report['stages']),
                total_boundaries=len(report['stages']), reference_review='PENDING', model_verified=False))
            (a.output/'progress.json').write_text(json.dumps(rows, indent=2)+'\n')
            print(name, rows[-1]['passed_boundaries'], '/', rows[-1]['total_boundaries'], flush=True)
        verify()
        (a.output/'execution.json').write_text(json.dumps(dict(cases=rows,
            execution_completed=True, reference_review='PENDING', model_verified=False), indent=2)+'\n')
        raise SystemExit(0 if all(r['numerical_passed'] for r in rows) else 1)


if __name__ == '__main__':
    main()
