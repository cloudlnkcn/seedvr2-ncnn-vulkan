#!/usr/bin/env python3
"""Explicit, hash-checked dependency preparation. CMake itself never downloads."""
import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cli-only', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cache = root / '.cache'
    prefix = root / '.deps/install'
    sources = root / '.deps/src'
    deps = json.loads((root / 'native-dependencies.lock.json').read_text())['dependencies']
    selected = [d for d in deps if not args.cli_only or d['name'] == 'cli11']
    for dep in selected:
        archive = cache / 'archives' / f"{dep['name']}-{dep['commit']}.tar.gz"
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            if args.offline:
                raise SystemExit(f'Missing cached archive: {archive.name}')
            with urllib.request.urlopen(dep['url'], timeout=60) as response:
                data = response.read(20 * 1024 * 1024)
            if hashlib.sha256(data).hexdigest() != dep['sha256']:
                raise SystemExit(f"Archive hash mismatch: {dep['name']}")
            archive.write_bytes(data)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != dep['sha256']:
            raise SystemExit(f"Cached archive hash mismatch: {dep['name']}")
        target = sources / dep['name']
        if target.exists():
            # This script owns only ignored dependency trees, never project sources.
            shutil.rmtree(target)
        temp = sources / (dep['name'] + '-extract')
        temp.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as tar:
            tar.extractall(temp, filter='data')
        entries = list(temp.iterdir())
        if len(entries) != 1 or not entries[0].is_dir():
            raise SystemExit('Unexpected archive layout')
        entries[0].rename(target)
        temp.rmdir()
    if not args.cli_only:
        shutil.rmtree(sources / 'drogon/trantor')
        shutil.copytree(sources / 'trantor', sources / 'drogon/trantor')
    configs = {
        'cli11': ['-DCLI11_BUILD_TESTS=OFF', '-DCLI11_BUILD_EXAMPLES=OFF', '-DCLI11_BUILD_DOCS=OFF'],
        'jsoncpp': ['-DJSONCPP_WITH_TESTS=OFF', '-DJSONCPP_WITH_POST_BUILD_UNITTEST=OFF',
                    '-DBUILD_SHARED_LIBS=OFF', '-DBUILD_OBJECT_LIBS=OFF'],
        'drogon': ['-DBUILD_CTL=OFF', '-DBUILD_EXAMPLES=OFF', '-DBUILD_TESTING=OFF',
                   '-DBUILD_ORM=OFF', '-DBUILD_SHARED_LIBS=OFF', '-DBUILD_BROTLI=OFF',
                   '-DBUILD_YAML_CONFIG=OFF', '-DBUILD_DOC=OFF'],
    }
    for name in ['cli11'] if args.cli_only else ['cli11', 'jsoncpp', 'drogon']:
        build = root / '.deps/build' / name
        log_path = cache / f'{name}-build.log'
        commands = [
            ['cmake', '-S', str(sources / name), '-B', str(build), '-G', 'Ninja',
             '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CXX_STANDARD=20',
             '-DCMAKE_POSITION_INDEPENDENT_CODE=ON', '-DCMAKE_INSTALL_LIBDIR=lib',
             f'-DCMAKE_INSTALL_PREFIX={prefix}', f'-DCMAKE_PREFIX_PATH={prefix}', *configs[name]],
            ['cmake', '--build', str(build), '--parallel', str(args.jobs)],
            ['cmake', '--install', str(build)],
        ]
        print(f'Preparing {name}; log: {log_path}', flush=True)
        with log_path.open('w') as log:
            for command in commands:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    print(f'Native dependencies ready: {prefix}')


if __name__ == '__main__':
    main()
