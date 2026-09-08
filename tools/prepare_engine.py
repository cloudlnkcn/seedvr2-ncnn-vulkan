#!/usr/bin/env python3
"""Prepare the pinned, unmodified ncnn dependency in the private build tree."""
import argparse
import hashlib
import json
import subprocess
import tarfile
import urllib.request
from pathlib import Path


def locked_source(root, offline=False):
    dep = json.loads((root/'engine-dependencies.lock.json').read_text())['ncnn']
    archive = root/'.cache/archives'/f"ncnn-{dep['commit']}.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        if offline:
            raise SystemExit('The locked ncnn archive is not cached')
        with urllib.request.urlopen(dep['url'], timeout=60) as response:
            data = response.read(40*1024*1024)
        if hashlib.sha256(data).hexdigest() != dep['sha256']:
            raise SystemExit('ncnn archive hash mismatch')
        archive.write_bytes(data)
    with archive.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != dep['sha256']:
            raise SystemExit('Cached ncnn archive hash mismatch')
    source = root/'.deps/src'/f"ncnn-{dep['commit']}"
    if not source.exists():
        with tarfile.open(archive) as tar:
            tar.extractall(root/'.deps/src', filter='data')
    # A cached directory is not automatically trusted. Verify every source
    # file against the immutable archive before either ncnn or pnnx builds.
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            path = root/'.deps/src'/member.name
            if path.is_symlink() or not path.is_file():
                raise SystemExit(f'Locked source is missing or replaced: {member.name}')
            with path.open('rb') as actual, tar.extractfile(member) as expected:
                if hashlib.file_digest(actual, 'sha256').digest() != hashlib.file_digest(expected, 'sha256').digest():
                    raise SystemExit(f'Locked source was modified: {member.name}')
    return source, dep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 32:
        parser.error('--jobs must be 1..32')
    root = Path(__file__).resolve().parents[1]
    source, dep = locked_source(root, args.offline)
    prefix = root/'.deps/ncnn-install'
    build = root/'.deps/build'/f"ncnn-{dep['commit'][:12]}"
    options = {
        'CMAKE_BUILD_TYPE':'Release', 'CMAKE_INSTALL_PREFIX':str(prefix),
        'CMAKE_INSTALL_LIBDIR':'lib', 'CMAKE_POSITION_INDEPENDENT_CODE':'ON',
        'NCNN_VULKAN':'ON', 'NCNN_SYSTEM_GLSLANG':'ON',
        'NCNN_BUILD_TOOLS':'OFF', 'NCNN_BUILD_EXAMPLES':'OFF',
        'NCNN_BUILD_BENCHMARK':'OFF', 'NCNN_BUILD_TESTS':'OFF',
        'NCNN_AVX512':'OFF', 'NCNN_INT8':'OFF',
    }
    commands = [
        ['cmake','-S',str(source),'-B',str(build),'-G','Ninja',
         *[f'-D{k}={v}' for k,v in options.items()]],
        ['cmake','--build',str(build),'--parallel',str(args.jobs)],
        ['cmake','--install',str(build)],
    ]
    log_path = root/'.cache/ncnn-build.log'
    print(f'Building pinned ncnn; log: {log_path}', flush=True)
    with log_path.open('w') as log:
        for command in commands:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    (prefix/'seedvr2-ncnn-commit.txt').write_text(dep['commit']+'\n')
    print(f'ncnn ready: {prefix}', flush=True)


if __name__ == '__main__':
    main()
