#!/usr/bin/env python3
"""Inventory a trusted local install before binary publication; never bundle system libraries."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def installed_identity(original, installed, expected, libdir):
    result = dict(build_matches_frozen=original.is_file() and sha(original) == expected,
                  installed_sha256=sha(installed) if installed.is_file() else None,
                  matches_reproduced_install=False)
    if not result['build_matches_frozen'] or not installed.is_file():
        return result
    dynamic = subprocess.check_output(['readelf', '-d', str(original)], text=True)
    match = re.search(r'\((?:RUNPATH|RPATH)\).*?\[(.*?)\]', dynamic)
    if not match:
        return dict(result, matches_reproduced_install=sha(installed) == expected,
                    transformation='none')
    new = '$ORIGIN/../' + libdir + ':$ORIGIN'
    with tempfile.TemporaryDirectory(prefix='seedvr2-install-identity-') as directory:
        copy = Path(directory) / original.name
        shutil.copyfile(original, copy)
        script = Path(directory) / 'rpath.cmake'
        # CMake bracket arguments preserve dollar signs and do not evaluate source text.
        if any(']==]' in value for value in [str(copy), match[1], new]):
            raise ValueError('Unsupported CMake bracket delimiter in RPATH')
        script.write_text(f'file(RPATH_CHANGE FILE [==[{copy}]==] OLD_RPATH [==[{match[1]}]==] NEW_RPATH [==[{new}]==])\n')
        subprocess.run(['cmake', '-P', str(script)], capture_output=True, check=True, timeout=30)
        transformed = sha(copy)
    return dict(result, transformation='CMake RPATH_CHANGE to install-relative paths',
                install_rpath=new, reproduced_sha256=transformed,
                matches_reproduced_install=transformed == result['installed_sha256'])


def audit(prefix, build_dir):
    prefix = prefix.resolve()
    evidence = json.loads((ROOT / 'artifacts/2026-09-10/video-numerics-v2/summary.json').read_text())
    inventory, missing, libraries, errors = [], [], {}, []
    required = ['bin/seedvr2', 'bin/seedvr2-worker', 'include/seedvr2/pipeline.hpp',
                'share/seedvr2/LICENSE', 'share/seedvr2/NOTICE']
    for name in required:
        if not (prefix / name).is_file():
            missing.append(name)
    for path in sorted(prefix.rglob('*')):
        name = path.relative_to(prefix).as_posix()
        if name.startswith('models/'):
            continue
        if path.is_symlink():
            if not path.resolve().is_relative_to(prefix) or not path.exists():
                errors.append(f'Link leaves install or is broken: {name}')
            inventory.append(dict(path=name, symlink=str(path.readlink())))
        elif path.is_file():
            inventory.append(dict(path=name, bytes=path.stat().st_size, sha256=sha(path)))
    cli = prefix / 'bin/seedvr2'
    sdk_matches = list(prefix.glob('lib*/libseedvr2.so.0.7.0'))
    frozen = dict(cli=cli.is_file() and sha(cli) == evidence['cli_sha256'],
                  sdk=len(sdk_matches) == 1 and sha(sdk_matches[0]) == evidence['sdk_sha256'])
    derived = {}
    if len(sdk_matches) == 1:
        libdir = sdk_matches[0].parent.name
        derived = dict(cli=installed_identity(build_dir / 'seedvr2', cli, evidence['cli_sha256'], libdir),
                       sdk=installed_identity(build_dir / 'libseedvr2.so.0.7.0', sdk_matches[0],
                                              evidence['sdk_sha256'], libdir))
    identity_passed = bool(derived) and all(x['matches_reproduced_install'] for x in derived.values())
    for executable in ['seedvr2', 'seedvr2-worker', 'seedvr2-web']:
        path = prefix / 'bin' / executable
        if not path.is_file():
            continue
        process = subprocess.run(['ldd', str(path)], capture_output=True, text=True, timeout=30)
        if process.returncode:
            errors.append(f'ldd failed: {executable}')
        for line in process.stdout.splitlines():
            if 'not found' in line:
                errors.append(f'{executable}: {line.strip()}')
            match = re.match(r'\s*(\S+) => (/\S+) ', line)
            if match and not Path(match[2]).resolve().is_relative_to(prefix):
                libraries[match[1]] = match[2]
    # Inspect the exact installed CLI. These are trusted project binaries, not arbitrary downloads.
    version = subprocess.run([str(cli), 'version'], capture_output=True, text=True, timeout=30) if cli.is_file() else None
    if version is not None and version.returncode:
        errors.append('Installed CLI cannot run version')
    system = dict(platform=platform.system(), architecture=platform.machine(),
                  libc=list(platform.libc_ver()), distribution=platform.freedesktop_os_release().get('PRETTY_NAME'))
    blockers = ['No separately measured clean target-machine installation for this binary candidate',
                'System FFmpeg ABI/configuration and corresponding source/notice materials need a release recipe']
    if libraries:
        blockers.append(f'{len(libraries)} external shared-library names must be resolved on the target')
    return dict(schema_version='seedvr2-native-install-audit-v1',
                passed=not missing and not errors and identity_passed,
                raw_build_hash_equals_installed_hash=frozen, reproduced_install=derived, host=system,
                missing_files=missing, errors=errors, inventory=inventory,
                external_sonames=sorted(libraries), external_library_count=len(libraries),
                version_output=version.stdout.strip() if version and not version.returncode else None,
                portable_release_ready=False, publication_blockers=blockers,
                scope='This host inventory and frozen binary identity; no cross-distribution compatibility claim')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', required=True, type=Path)
    parser.add_argument('--build-dir', type=Path, default=ROOT / 'build/release',
                        help='Frozen build files used to reproduce CMake install RPATH changes')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--archive', type=Path, help='Optional local host-specific archive; no system libraries or models')
    args = parser.parse_args()
    report = audit(args.prefix, args.build_dir)
    if args.archive and not report['passed']:
        report['archive_refused'] = 'Install audit failed; no archive produced'
    elif args.archive:
        if args.archive.exists():
            raise SystemExit('Refusing to overwrite an existing archive')
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.archive, 'x:gz', dereference=False) as archive:
            for row in report['inventory']:
                archive.add(args.prefix / row['path'], arcname='seedvr2-host/' + row['path'], recursive=False)
        report['archive'] = dict(name=args.archive.name, bytes=args.archive.stat().st_size,
                                sha256=sha(args.archive), includes_models=False,
                                includes_system_libraries=False, published=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('inventory', 'version_output', 'external_sonames')},
                     ensure_ascii=False, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
