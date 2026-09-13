#!/usr/bin/env python3
"""Publish verified content-addressed model assets without replacing existing assets."""
import argparse
import json
from pathlib import Path
import subprocess
from model_distribution import ROOT, check_catalog, sha_file, verified


def gh(*args):
    return subprocess.check_output(['gh', *args], text=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--repo', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--target', required=True, help='Reviewed source commit')
    p.add_argument('--notes', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--publish', action='store_true')
    a = p.parse_args()
    policy = json.loads((ROOT/'policies/reviewed-packages.json').read_text())
    objects = check_catalog(json.loads((a.bundle/'catalog.json').read_text()), policy)
    files = []
    for digest, size in sorted(objects.items()):
        path = a.bundle/'objects'/digest
        verified(path, dict(bytes=size, sha256=digest))
        files.append(path)
    files += [a.bundle/name for name in ['catalog.json', 'LICENSE', 'README.md',
                                        'model-sources.lock.json', 'reviewed-packages.json']]
    inventory = {path.name: dict(bytes=path.stat().st_size, sha256=sha_file(path)) for path in files}
    # Read errors other than an absent tag must not create a duplicate release.
    releases = json.loads(gh('api', f'repos/{a.repo}/releases', '--paginate', '--slurp'))
    matches = [r for page in releases for r in page if r['tag_name'] == a.tag]
    if not matches:
        gh('release', 'create', a.tag, '--repo', a.repo, '--target', a.target, '--draft',
           '--title', 'SeedVR2 3B FP32-B converted model packages', '--notes-file', str(a.notes))
    releases = json.loads(gh('api', f'repos/{a.repo}/releases', '--paginate', '--slurp'))
    selected = [r for page in releases for r in page if r['tag_name'] == a.tag]
    if len(selected) != 1 or selected[0]['target_commitish'] != a.target:
        raise ValueError('Release source identity mismatch')
    endpoint = f'repos/{a.repo}/releases/{selected[0]["id"]}'
    release = json.loads(gh('api', endpoint))
    assets = {x['name']: x for x in release['assets']}
    if set(assets) - set(inventory):
        raise ValueError('Release has unexpected assets; refusing to modify it')
    for index, path in enumerate(files, 1):
        expected = inventory[path.name]
        if path.name not in assets:
            if not release['draft']:
                raise ValueError('Incomplete public release; refusing mutation')
            print(f'Upload {index}/{len(files)} {path.name} ({expected["bytes"]} bytes)', flush=True)
            subprocess.run(['gh', 'release', 'upload', a.tag, str(path), '--repo', a.repo], check=True)
        else:
            print(f'Resume {index}/{len(files)} {path.name}', flush=True)
    release = json.loads(gh('api', endpoint))
    if {x['name'] for x in release['assets']} != set(inventory):
        raise ValueError('Remote release inventory is incomplete')
    for asset in release['assets']:
        expected = inventory[asset['name']]
        if asset['size'] != expected['bytes'] or asset.get('digest') != 'sha256:'+expected['sha256']:
            raise ValueError('Remote asset digest or size mismatch: '+asset['name'])
    if a.publish and release['draft']:
        gh('release', 'edit', a.tag, '--repo', a.repo, '--draft=false', '--prerelease')
        release = json.loads(gh('api', endpoint))
    result = dict(schema_version='seedvr2-model-release-v1', passed=True,
                  published=not release['draft'], tag=a.tag, source_commit=a.target,
                  url=release['html_url'], object_bytes=sum(objects.values()),
                  verification='GitHub API server digests match locally verified bytes; client download separately tested',
                  assets=[dict(name=x['name'], bytes=x['size'], digest=x['digest'],
                               url=x['browser_download_url']) for x in release['assets']])
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
