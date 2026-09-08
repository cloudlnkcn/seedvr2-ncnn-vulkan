#!/usr/bin/env python3
"""Build pnnx from the independently locked ncnn converter commit and the private export environment."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from prepare_engine import locked_source


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--python', type=Path, default=root/'.venv-export/bin/python')
    args = parser.parse_args()
    if not 1 <= args.jobs <= 32:
        parser.error('--jobs must be 1..32')
    python = args.python.absolute()  # Do not resolve a venv symlink to the base interpreter.
    environment = json.loads(subprocess.check_output([str(python), '-c',
        'import json,sys,torch; from pathlib import Path; '
        'print(json.dumps(dict(python=sys.version_info[:3],torch=torch.__version__, '
        'torch_dir=str(Path(torch.__file__).parent))))'], text=True))
    if environment['python'][:2] != [3, 13] or environment['torch'] != '2.9.0+cpu':
        raise SystemExit('Use the pinned Python 3.13 / torch 2.9.0+cpu export environment')
    source, dep = locked_source(root, args.offline, 'converter-dependencies.lock.json')
    build = root/'.deps/build'/f"pnnx-{dep['commit'][:12]}-torch290"
    torch_dir = Path(environment['torch_dir'])
    commands = [
        ['cmake', '-S', str(source/'tools/pnnx'), '-B', str(build), '-G', 'Ninja',
         '-DCMAKE_BUILD_TYPE=Release', f'-DPython3_EXECUTABLE={python}',
         f'-DTorch_DIR={torch_dir}/share/cmake/Torch', '-DPNNX_TNN2PNNX=OFF'],
        ['cmake', '--build', str(build), '--target', 'pnnx', '--parallel', str(args.jobs)],
    ]
    log_path = root/'.cache/pnnx-build-private.log'
    print(f'Building official pnnx; log: {log_path}', flush=True)
    with log_path.open('w') as log:
        for command in commands:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    destination = root/'.deps/bin/pnnx'
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(build/'src/pnnx', destination)
    with destination.open('rb') as stream:
        sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
    (destination.parent/'pnnx-build.json').write_text(json.dumps(dict(
        ncnn_commit=dep['commit'], source_archive_sha256=dep['sha256'],
        pnnx_sha256=sha256, environment=environment,
        scope='Development export tool only; not a runtime dependency'), indent=2)+'\n')
    print(f'pnnx ready: {destination}', flush=True)


if __name__ == '__main__':
    main()
