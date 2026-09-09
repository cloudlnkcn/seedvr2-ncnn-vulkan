#!/usr/bin/env python3
"""Build and install the native tutorial from a clone; never download model weights."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def prerequisites(web):
    errors = []
    if sys.version_info < (3, 12):
        errors.append('Python 3.12 or newer is required')
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    commands = ['cmake', 'ninja', 'pkg-config', 'glslangValidator']
    if not compiler or not shutil.which(compiler[0]):
        errors.append('A C++20 compiler is required (set CXX to g++ or clang++)')
    if web:
        commands += ['node', 'npm']
    for command in commands:
        if not shutil.which(command):
            errors.append('Missing executable: ' + command)
    if shutil.which('cmake'):
        result = subprocess.run(['cmake', '--version'], capture_output=True, text=True)
        match = re.search(r'cmake version (\d+)\.(\d+)', result.stdout)
        if result.returncode or not match or tuple(map(int, match.groups())) < (3, 25):
            errors.append('CMake 3.25 or newer is required')
    if web and shutil.which('node'):
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        match = re.match(r'v(\d+)\.', result.stdout)
        if result.returncode or not match or int(match[1]) < 24:
            errors.append('Node.js 24 or newer is required for the Web build')
    if shutil.which('pkg-config'):
        packages = ['openssl', 'sqlite3', 'libjpeg', 'libavformat', 'libavcodec', 'libavutil', 'libswscale']
        if web:
            packages += ['zlib', 'uuid']
        for package in packages:
            if subprocess.run(['pkg-config', '--exists', package], capture_output=True).returncode:
                errors.append('Missing development package: ' + package)
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli-only', action='store_true', help='Build CLI/worker/SDK without Node.js or Web')
    parser.add_argument('--jobs', type=int, default=2, help='Build parallelism, 1..32 (default: 2)')
    parser.add_argument('--offline', action='store_true', help='Require cached dependency archives and npm packages')
    parser.add_argument('--prefix', type=Path, default=ROOT/'dist/tutorial', help='Install directory')
    parser.add_argument('--build-dir', type=Path, default=ROOT/'build/tutorial', help='CMake build directory')
    parser.add_argument('--check', action='store_true', help='Check host prerequisites without downloading or building')
    parser.add_argument('--plan', action='store_true', help='Print commands as JSON without executing them')
    args = parser.parse_args()
    if not 1 <= args.jobs <= 32:
        parser.error('--jobs must be 1..32')
    build, prefix = args.build_dir.resolve(), args.prefix.resolve()
    if build == ROOT or prefix == ROOT or build == prefix:
        parser.error('Build and install directories must be distinct and different from the source root')
    common = ['--jobs', str(args.jobs)] + (['--offline'] if args.offline else [])
    commands = [[sys.executable, str(ROOT/'tools/prepare_native.py'), *common,
                 *(['--cli-only'] if args.cli_only else [])],
                [sys.executable, str(ROOT/'tools/prepare_engine.py'), *common]]
    if not args.cli_only:
        commands += [['npm', 'ci', '--prefix', str(ROOT/'apps/studio'), '--ignore-scripts',
                      *(['--offline'] if args.offline else [])],
                     ['npm', 'run', 'build', '--prefix', str(ROOT/'apps/studio')]]
    commands += [
        ['cmake', '--fresh', '-S', str(ROOT), '-B', str(build), '-G', 'Ninja',
         '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_TESTING=ON',
         '-DSEEDVR2_BUILD_WEB=' + ('OFF' if args.cli_only else 'ON'),
         '-DCMAKE_INSTALL_PREFIX=' + str(prefix)],
        ['cmake', '--build', str(build), '--parallel', str(args.jobs)],
        ['ctest', '--test-dir', str(build), '--output-on-failure', '--no-tests=error'],
        ['cmake', '--install', str(build)],
        [str(prefix/'bin/seedvr2'), 'version'],
    ]
    if args.plan:
        print(json.dumps(dict(source=str(ROOT), build=str(build), install=str(prefix),
                              downloads_model_weights=False, commands=commands), indent=2))
        return 0
    errors = prerequisites(not args.cli_only)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        print('Install host dependencies using docs/TUTORIAL.md, then rerun this command.', file=sys.stderr)
        return 2
    if args.check:
        print('Host prerequisites found. This does not verify compilation or GPU execution.')
        return 0
    for index, command in enumerate(commands, 1):
        print(f'[{index}/{len(commands)}] {shlex.join(command)}', flush=True)
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode:
            print(f'Stopped at step {index}; earlier completed work is retained.', file=sys.stderr)
            return result.returncode if result.returncode > 0 else 1
    print(f'Installed native CLI/SDK: {prefix}')
    print('Model weights are separate. Start with engine self-test; see docs/TUTORIAL.md for export.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('Build interrupted; rerun to reuse downloaded archives and compiled objects.', file=sys.stderr)
        raise SystemExit(130)
