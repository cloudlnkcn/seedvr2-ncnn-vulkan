#!/usr/bin/env python3
"""Package a tested Ubuntu 24.04 install with exact runtime inventory and source."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    return subprocess.check_output(args, text=True)


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--install', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if platform.freedesktop_os_release().get('VERSION_ID') != '24.04' or platform.freedesktop_os_release().get('ID') != 'ubuntu' or platform.machine() != 'x86_64':
        raise ValueError('This recipe supports Ubuntu 24.04 x86_64 only')
    prefix = a.install.resolve()
    for name in ['seedvr2', 'seedvr2-web', 'seedvr2-worker']:
        if not (prefix/'bin'/name).is_file():
            raise ValueError('Missing executable: '+name)
    dependencies, ldd = {}, {}
    for name in ['seedvr2', 'seedvr2-web', 'seedvr2-worker']:
        text = run('ldd', str(prefix/'bin'/name))
        ldd[name] = text
        if 'not found' in text:
            raise ValueError('Unresolved runtime dependency')
        for lib in re.findall(r'(?:=>\s+)?(/\S+)', text):
            path = Path(lib).resolve()
            if path.is_relative_to(prefix):
                continue
            found = subprocess.run(['dpkg-query', '-S', str(path)], capture_output=True, text=True)
            if found.returncode:
                found = subprocess.run(['dpkg-query', '-S', '*/'+path.name], capture_output=True, text=True)
            if found.returncode:
                raise ValueError('No Ubuntu package owns '+str(path))
            package = found.stdout.splitlines()[0].rsplit(': ', 1)[0]
            dependencies[package] = run('dpkg-query', '-W', '-f=${Version}', package)
    # Runtime dependencies are installed from Ubuntu, not silently copied from a development host.
    script = '#!/bin/sh\nset -eu\n. /etc/os-release\n[ "$ID" = ubuntu ] && [ "$VERSION_ID" = 24.04 ] || { echo "Ubuntu 24.04 required" >&2; exit 2; }\napt-get update\napt-get install -y '+shlex.join(sorted(dependencies))+'\n'
    (prefix/'install-runtime-ubuntu24.04.sh').write_text(script)
    os.chmod(prefix/'install-runtime-ubuntu24.04.sh', 0o755)
    commit = run('git', 'rev-parse', 'HEAD').strip()
    inventory = dict(schema_version='seedvr2-ubuntu-release-v1', source_commit=commit,
                     platform='Ubuntu 24.04 x86_64', build_package_versions=dependencies,
                     ldd=ldd, system_libraries_bundled=False, inference_requires_python=False,
                     offline_requires_runtime_dependencies_installed=True,
                     files={str(x.relative_to(prefix)): digest(x) for x in sorted(prefix.rglob('*')) if x.is_file()})
    (prefix/'release-manifest.json').write_text(json.dumps(inventory, indent=2)+'\n')
    shutil.copyfile('/usr/share/common-licenses/GPL-3', prefix/'COPYING-GPL-3')
    notes = '''# SeedVR2 native Ubuntu 24.04 x86_64\n\nCLI, local Web, worker and installable C++ SDK. This prerelease retains the existing FP32-B model execution path. Download the pinned official checkpoints and convert them locally with tools/convert_models.sh; see docs/TUTORIAL.md. Converted weights are not distributed in this release.\n\nExtract the program archive, run `sudo bash seedvr2/install-runtime-ubuntu24.04.sh` once on Ubuntu 24.04, then `seedvr2/bin/seedvr2 --help` or `seedvr2/bin/seedvr2-studio`. Runtime packages require internet during setup; inference is offline and does not require Python. This archive does not bundle system libraries or claim support for other distributions.\n\nThe installed manifest records the exact build commit, binary hashes and build-time Ubuntu package versions. The installer accepts current Ubuntu security updates; those are not bit-identical to the build environment. The clean-machine report records functional image execution and isolated offline replay; it does not certify AMD/Intel GPU execution, BF16, large resolutions, or restoration quality.\n\n## Source and licenses\n\nThese executable combinations use GPL-enabled FFmpeg/libx264 and are supplied under GPL version 3 with the complete project source and collected dependency source material alongside the binaries. Original project source files retain Apache-2.0; individual dependencies retain their notices. See COPYING-GPL-3, LICENSE, NOTICE and the bundled license directory. No system-library binaries are redistributed.\n\n`seedvr2-corresponding-source.tar.gz` contains the exact project checkout, prepared native dependency sources, frontend dependency sources and collected FFmpeg/x264/glslang/SPIRV-Tools source archives. Build with the checked-in release-linux workflow and tools/build_native.py on Ubuntu 24.04. Compiler/system package identities are retained in release-manifest.json.\n'''
    (prefix/'RELEASE.md').write_text(notes)
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/'RELEASE.md').write_text(notes)
    with tarfile.open(a.output/'seedvr2-ubuntu24.04-x86_64.tar.gz', 'w:gz') as archive:
        archive.add(prefix, arcname='seedvr2')
    with tarfile.open(a.output/'seedvr2-corresponding-source.tar.gz', 'w:gz') as archive:
        for name in run('git', 'ls-files', '-z').split('\0'):
            if name:
                archive.add(ROOT/name, arcname='seedvr2-source/'+name, recursive=False)
        for name in ['.deps/src', '.cache/archives', 'apps/studio/node_modules', 'dist/dependency-sources']:
            if (ROOT/name).exists():
                archive.add(ROOT/name, arcname='seedvr2-source/'+name)
    (a.output/'release-manifest.json').write_text(json.dumps(inventory, indent=2)+'\n')
    files = sorted(a.output.iterdir())
    if any(x.stat().st_size >= 2*1024**3 for x in files):
        raise ValueError('Asset exceeds GitHub size limit')
    (a.output/'SHA256SUMS').write_text(''.join(digest(x)+'  '+x.name+'\n' for x in files))


if __name__ == '__main__':
    main()
