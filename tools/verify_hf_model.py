#!/usr/bin/env python3
"""Maintainer check: download every immutable HF asset and verify local identities.

Requires huggingface_hub for authenticated private staging. Does not publish,
change the native model allowlist, or claim numerical certification.
"""
import argparse
import json
from pathlib import Path
import re
import shutil

from huggingface_hub import HfApi, hf_hub_download
from model_distribution import check_catalog, sha_file

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r'[0-9a-f]{40}', args.revision):
        parser.error('Use an immutable 40-character commit revision')
    if args.output.resolve() == args.source.resolve():
        parser.error('Readback must use a separate directory')
    catalog = json.loads((args.source/'catalog.json').read_text())
    check_catalog(catalog, json.loads((ROOT/'policies/reviewed-packages.json').read_text()))
    files = [p for p in args.source.rglob('*') if p.is_file()
             and '.cache' not in p.relative_to(args.source).parts]
    expected = {p.relative_to(args.source).as_posix():
                {'sha256': sha_file(p), 'bytes': p.stat().st_size} for p in files}
    info = HfApi().model_info(args.repo, revision=args.revision, files_metadata=True)
    actual = {p.rfilename: p for p in info.siblings if p.rfilename != '.gitattributes'}
    if set(actual) != set(expected):
        raise ValueError('Remote file inventory differs from the prepared model repository')
    args.output.mkdir(parents=True, exist_ok=True)
    needed = sum(row['bytes'] for name, row in expected.items() if not (args.output/name).exists())
    if shutil.disk_usage(args.output).free < needed + 1024**3:
        raise ValueError('Insufficient disk space for independent model readback plus reserve')
    results = []
    args.report.parent.mkdir(parents=True, exist_ok=True)
    for name, row in sorted(expected.items()):
        remote = actual[name]
        if remote.size != row['bytes']:
            raise ValueError('Remote size mismatch: '+name)
        if remote.lfs and remote.lfs.sha256 != row['sha256']:
            raise ValueError('Remote LFS hash mismatch: '+name)
        path = Path(hf_hub_download(args.repo, filename=name, revision=args.revision,
                                   local_dir=args.output))
        if path.stat().st_size != row['bytes'] or sha_file(path) != row['sha256']:
            raise ValueError('Downloaded bytes differ: '+name)
        results.append(dict(path=name, **row))
        args.report.write_text(json.dumps(dict(passed=False, status='IN_PROGRESS',
            repo=args.repo, revision=args.revision, files=results), indent=2)+'\n')
        print(f'{len(results)}/{len(expected)} downloaded and SHA-256 verified: {name}', flush=True)
    args.report.write_text(json.dumps(dict(passed=True, repo=args.repo,
        revision=args.revision, private_at_verification=info.private, files=results,
        total_bytes=sum(r['bytes'] for r in results), model_verified=False,
        scope='Independent remote byte readback; native execution and numerical parity are separate'), indent=2)+'\n')


if __name__ == '__main__':
    main()
