#!/usr/bin/env python3
"""Stage reviewed converted models, then install from a local bundle or HTTPS mirror.

The catalogue is local and reviewable. It contains the original manifests and a
deduplicated SHA-256 object inventory. Model maths and native trust policy do not
change. Publication is a separate operation; this tool never invents a mirror.
"""
import argparse
from contextlib import contextmanager, redirect_stdout
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
from urllib.parse import urlsplit

from prepare_models import download_file

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'seedvr2-model-distribution-v1'
SHA = re.compile(r'[0-9a-f]{64}')


def sha_file(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def no_symlinks(path):
    path = Path(os.path.abspath(path))
    for part in [path, *path.parents]:
        if part.is_symlink():
            raise ValueError(f'Refusing symbolic link: {part}')
    return path


def relative_name(value):
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or '..' in path.parts or '\\' in value
            or str(path) != value or value == '.'):
        raise ValueError(f'Unsafe package path: {value}')
    return value


def check_ref(row):
    if (not SHA.fullmatch(row.get('sha256', '')) or type(row.get('bytes')) is not int
            or row['bytes'] <= 0):
        raise ValueError('Invalid file hash or size')


def identity(manifest):
    payload = dict(manifest)
    payload.pop('exporter', None)
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def manifest_files(manifest, kind, reviewed):
    allowed = [x for x in reviewed['packages'] if x['kind'] == kind
               and x['profile'] == manifest.get('profile')]
    if len(allowed) != 1 or identity(manifest) != allowed[0]['payload_sha256']:
        raise ValueError(f'Unreviewed {kind} model payload; hashes alone cannot grant trust')
    if manifest.get('model_verified') is not False:
        raise ValueError('Distribution cannot promote a package to model certification')
    rows = []
    for graph in manifest['graphs']:
        rows += [dict(graph['param']), dict(graph['weights'])]
    for value in manifest['constants'].values():
        if value['dtype'] != 'f32le':
            raise ValueError('Unsupported constant storage')
        rows.append(dict(path=value['path'], sha256=value['sha256'], bytes=4 * math.prod(value['shape'])))
    names = set()
    for row in rows:
        name = relative_name(row['path'])
        check_ref(row)
        if name in names or name == 'manifest.json':
            raise ValueError(f'Duplicate package path: {name}')
        names.add(name)
    return sorted(rows, key=lambda x: x['path'])


def check_catalog(catalog, reviewed):
    if catalog.get('schema_version') != SCHEMA or not catalog.get('packages'):
        raise ValueError('Unknown or empty converted model catalogue')
    if catalog.get('model_verified') is not False:
        raise ValueError('A distribution catalogue cannot issue model certification')
    kinds, objects = set(), {}
    for package in catalog['packages']:
        kind = package['kind']
        if kind not in ('image', 'video') or kind in kinds:
            raise ValueError('Unknown or duplicate package kind')
        kinds.add(kind)
        expected = manifest_files(package['manifest'], kind, reviewed)
        if package['files'] != expected:
            raise ValueError('Catalogue file contract differs from the reviewed model manifest')
        check_ref(package['manifest_object'])
        if package['manifest_object']['bytes'] > 1024 * 1024:
            raise ValueError('Manifest object exceeds 1 MiB')
        for row in [package['manifest_object'], *expected]:
            sha = row['sha256']
            if sha in objects and objects[sha] != row['bytes']:
                raise ValueError('Inconsistent object size')
            objects[sha] = row['bytes']
    return objects


def verified(path, row):
    no_symlinks(path)
    if not path.is_file() or path.stat().st_size != row['bytes'] or sha_file(path) != row['sha256']:
        raise ValueError(f'File is missing or corrupt: {path}')


def stage(packages, output, reviewed, hardlink=False):
    output = no_symlinks(output)
    if output.exists():
        raise ValueError('Staging output already exists; use a new directory')
    catalog = dict(schema_version=SCHEMA, model_verified=False,
                   meaning='Reviewed export identity and byte integrity; numerical and quality evidence remain separate',
                   packages=[])
    origins = {}
    for kind, folder in packages.items():
        folder = no_symlinks(folder)
        path = folder / 'manifest.json'
        no_symlinks(path)
        manifest = json.loads(path.read_text())
        rows = manifest_files(manifest, kind, reviewed)
        reference = dict(sha256=sha_file(path), bytes=path.stat().st_size)
        catalog['packages'].append(dict(kind=kind, manifest=manifest, manifest_object=reference, files=rows))
        # Deduplicate by content, not by filenames or presumed shared DiT weights.
        for row in [dict(reference, path='manifest.json'), *rows]:
            origins.setdefault(row['sha256'], []).append((folder / row['path'], row))
    objects = check_catalog(catalog, reviewed)
    output.parent.mkdir(parents=True, exist_ok=True)
    needed = 0 if hardlink else sum(objects.values())
    if shutil.disk_usage(output.parent).free < needed + 64 * 1024 * 1024:
        raise ValueError(f'Insufficient space: need {needed} bytes plus staging reserve')
    temporary = Path(tempfile.mkdtemp(prefix=output.name + '.partial-', dir=output.parent))
    try:
        (temporary / 'objects').mkdir()
        checked_inodes = set()
        for index, (sha, rows) in enumerate(sorted(origins.items()), 1):
            for path, row in rows:
                no_symlinks(path)
                stat = path.stat()
                key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, sha)
                if key not in checked_inodes:
                    verified(path, row)
                    checked_inodes.add(key)
            path, row = rows[0]
            target = temporary / 'objects' / sha
            if hardlink:
                os.link(path, target)
            else:
                shutil.copyfile(path, target)
                verified(target, row)
            print(f'Object {index}/{len(origins)} checked and staged', file=sys.stderr, flush=True)
        (temporary / 'catalog.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + '\n')
        for source, name in [(ROOT / 'tests/reference/Apache-2.0.txt', 'LICENSE'),
                             (ROOT / 'model-sources.lock.json', 'model-sources.lock.json'),
                             (ROOT / 'policies/reviewed-packages.json', 'reviewed-packages.json')]:
            shutil.copyfile(source, temporary / name)
        (temporary / 'README.md').write_text(
            '# SeedVR2 3B converted ncnn models\n\n'
            'Derived from ByteDance-Seed/SeedVR2-3B at revision '
            '37255ff8cccfb01071b87f635a5948ca8d53117c (Apache-2.0). '
            'Converted by SeedVR2 ncnn Vulkan; this is a community export, not an official release.\n\n'
            'Image and temporal video FP32-B packages share content-addressed objects. '
            'Use tools/model_distribution.py install from the project repository; '
            'do not pass the objects directory directly to ncnn. '
            'The catalogue preserves the original manifests and exporter provenance. '
            'File authentication is separate from numerical parity and perceptual quality.\n\n'
            'Scope: https://github.com/mingshi2333/seedvr2-ncnn-vulkan/blob/main/docs/CURRENT-GAPS.md\n')
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return dict(passed=True, operation='stage', packages=list(packages), objects=len(objects),
                unique_bytes=sum(objects.values()),
                package_bytes=sum(sum(x['bytes'] for x in p['files']) + p['manifest_object']['bytes']
                                  for p in catalog['packages']),
                catalog_sha256=sha_file(output / 'catalog.json'), hardlinked=hardlink,
                published=False, model_verified=False)


@contextmanager
def install_lock(output):
    lock = no_symlinks(output.with_name(output.name + '.install.lock'))
    with lock.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another installer owns this destination') from None
        yield


def installed(folder, package):
    rows = [dict(package['manifest_object'], path='manifest.json'), *package['files']]
    expected = {row['path'] for row in rows}
    actual = set()
    for path in folder.rglob('*'):
        no_symlinks(path)
        if path.is_file():
            actual.add(path.relative_to(folder).as_posix())
    if actual != expected:
        raise ValueError('Installed package contains missing or unexpected files')
    for row in rows:
        verified(folder / row['path'], row)
    if json.loads((folder / 'manifest.json').read_text()) != package['manifest']:
        raise ValueError('Manifest bytes differ from the reviewed catalogue')


def install(catalog, kind, output, reviewed, source=None, base_url=None,
            offline=False, hardlink=False, plan=False):
    check_catalog(catalog, reviewed)
    matches = [x for x in catalog['packages'] if x['kind'] == kind]
    if len(matches) != 1:
        raise ValueError('Requested package is absent')
    package = matches[0]
    if source and base_url or hardlink and not source:
        raise ValueError('Choose one source; hardlinks require a local bundle')
    if base_url:
        parsed = urlsplit(base_url)
        if parsed.scheme != 'https' or not parsed.netloc or parsed.query or parsed.fragment or parsed.username:
            raise ValueError('Mirror must be an HTTPS base URL without credentials or query parameters')
    output = no_symlinks(output)
    rows = [dict(package['manifest_object'], path='manifest.json'), *package['files']]
    info = dict(operation='install', kind=kind, files=len(rows),
                bytes=sum(row['bytes'] for row in rows), model_verified=False,
                meaning='Reviewed export identity and integrity, not model certification')
    if plan:
        return dict(info, plan=True, writes=False, public_mirror_configured=bool(base_url))
    if not source and not base_url and not offline:
        raise ValueError('Specify --from-dir or --base-url; no public mirror is configured implicitly')
    output.parent.mkdir(parents=True, exist_ok=True)
    with install_lock(output):
        if output.exists():
            installed(output, package)
            return dict(info, passed=True, cached=True)
        temporary = no_symlinks(output.with_name(output.name + '.partial'))
        marker = temporary / '.seedvr2-install.json'
        token = dict(schema_version=SCHEMA, kind=kind,
                     manifest_sha256=package['manifest_object']['sha256'])
        if temporary.exists():
            no_symlinks(marker)
            if not marker.exists():
                # Recover the narrow crash window between final validation and rename.
                installed(temporary, package)
                temporary.rename(output)
                return dict(info, passed=True, cached=False, completed_partial=True)
            if not marker.is_file() or json.loads(marker.read_text()) != token:
                raise ValueError('Partial directory belongs to another package or operation')
        else:
            temporary.mkdir()
            marker.write_text(json.dumps(token) + '\n')
        if not hardlink:
            remaining = sum(row['bytes'] for row in rows if not (temporary / row['path']).is_file())
            if shutil.disk_usage(temporary).free < remaining + 64 * 1024 * 1024:
                raise ValueError(f'Insufficient space: {remaining} bytes plus reserve required')
        for index, row in enumerate(rows, 1):
            target = no_symlinks(temporary / row['path'])
            target.parent.mkdir(parents=True, exist_ok=True)
            if source:
                origin = no_symlinks(source / 'objects' / row['sha256'])
                verified(origin, row)
                if target.exists():
                    verified(target, row)
                elif hardlink:
                    os.link(origin, target)
                else:
                    part = no_symlinks(target.with_suffix(target.suffix + '.partial'))
                    shutil.copyfile(origin, part)
                    verified(part, row)
                    part.rename(target)
            else:
                url = (base_url or '').rstrip('/') + '/objects/' + row['sha256']
                with redirect_stdout(sys.stderr):
                    download_file(target, dict(size=row['bytes'], lfs=dict(sha256=row['sha256'])), url, offline)
            if index == 1 and json.loads(target.read_text()) != package['manifest']:
                raise ValueError('Downloaded manifest differs from reviewed catalogue')
            print(f'Installed {index}/{len(rows)}: {row["path"]}', file=sys.stderr, flush=True)
        marker.unlink()
        try:
            installed(temporary, package)
        except BaseException:
            marker.write_text(json.dumps(token) + '\n')
            raise
        temporary.rename(output)
    return dict(info, passed=True, cached=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    stage_parser = commands.add_parser('stage')
    stage_parser.add_argument('--image', type=Path)
    stage_parser.add_argument('--video', type=Path)
    stage_parser.add_argument('--output', type=Path, required=True)
    stage_parser.add_argument('--hardlink', action='store_true', help='Same filesystem only; keep source files unchanged')
    stage_parser.add_argument('--report', type=Path)
    setup = commands.add_parser('install')
    setup.add_argument('--catalog', type=Path, required=True)
    setup.add_argument('--kind', choices=['image', 'video'], required=True)
    setup.add_argument('--output', type=Path, required=True)
    source_group = setup.add_mutually_exclusive_group()
    source_group.add_argument('--from-dir', type=Path)
    source_group.add_argument('--base-url', help='Explicit immutable HTTPS mirror root')
    setup.add_argument('--offline', action='store_true')
    setup.add_argument('--hardlink', action='store_true')
    setup.add_argument('--plan', action='store_true')
    setup.add_argument('--report', type=Path)
    args = parser.parse_args()
    reviewed = json.loads((ROOT / 'policies/reviewed-packages.json').read_text())
    if args.command == 'stage':
        packages = {kind: getattr(args, kind) for kind in ('image', 'video') if getattr(args, kind)}
        if not packages:
            parser.error('At least one reviewed --image or --video package is required')
        result = stage(packages, args.output, reviewed, args.hardlink)
    else:
        result = install(json.loads(args.catalog.read_text()), args.kind, args.output, reviewed,
                         args.from_dir, args.base_url, args.offline, args.hardlink, args.plan)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
    except KeyboardInterrupt:
        print('Interrupted; installer partial files are retained for a verified resume.', file=sys.stderr)
        raise SystemExit(130)
