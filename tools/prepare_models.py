#!/usr/bin/env python3
"""Download immutable official checkpoints, verify LFS hashes, never execute them."""
import argparse
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path


def verified(path, row):
    if path.is_symlink():
        raise ValueError(f'Refusing a symbolic link: {path}')
    if not path.is_file() or path.stat().st_size != row['size']:
        return False
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() == row['lfs']['sha256']


def download_file(target, row, url, offline=False):
    if verified(target, row):
        print(target.name, 'hash verified (cached)', flush=True)
        return 'cached'
    if target.exists():
        raise ValueError(f'Refusing to replace a mismatched checkpoint: {target}')
    if offline:
        raise ValueError(f'Offline checkpoint is missing: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + '.partial')
    if partial.is_symlink():
        raise ValueError(f'Refusing a symbolic link: {partial}')
    offset = partial.stat().st_size if partial.exists() else 0
    # A previous process may have finished the bytes but stopped before rename.
    if offset == row['size']:
        if not verified(partial, row):
            raise ValueError(f'Complete partial file has the wrong SHA-256: {partial}')
        partial.rename(target)
        print(target.name, 'SHA256 verified (completed partial)', flush=True)
        return 'completed-partial'
    if offset > row['size']:
        raise ValueError(f'Partial file exceeds locked size: {partial}')
    headers = {'Range': f'bytes={offset}-'} if offset else {}
    request = urllib.request.Request(url, headers=headers)
    print(target.name, 'downloading, offset', offset, '/', row['size'], flush=True)
    with urllib.request.urlopen(request, timeout=60) as response:
        if offset:
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
            if (response.status != 206 or not match or int(match[1]) != offset
                    or int(match[2]) != row['size'] - 1 or int(match[3]) != row['size']):
                raise ValueError('Server did not honor the exact resume range; partial file is unchanged')
        elif response.status != 200:
            raise ValueError('Server did not return the complete checkpoint')
        last = time.monotonic()
        with partial.open('ab' if offset else 'wb') as out:
            while True:
                data = response.read(4 * 1024 * 1024)
                if not data:
                    break
                offset += len(data)
                if offset > row['size']:
                    raise ValueError('Checkpoint exceeds locked size')
                out.write(data)
                if time.monotonic() - last > 20:
                    print(target.name, offset, '/', row['size'], flush=True)
                    last = time.monotonic()
    if not verified(partial, row):
        raise ValueError(f'Checkpoint size or hash mismatch: {partial}; no final file was installed')
    partial.rename(target)
    print(target.name, 'SHA256 verified', row['lfs']['sha256'], flush=True)
    return 'downloaded'


def main():
    parser=argparse.ArgumentParser()
    root=Path(__file__).resolve().parents[1]
    parser.add_argument('--output',type=Path,default=root/'.cache/models')
    parser.add_argument('--only',choices=['vae','dit','embeddings','all'],default='all')
    parser.add_argument('--list',action='store_true',help='Show locked URLs and sizes without downloading')
    parser.add_argument('--offline',action='store_true',help='Verify existing final files; never access the network')
    args=parser.parse_args()
    lock=json.loads((root/'model-sources.lock.json').read_text())
    groups={'vae':['ema_vae.pth'],'dit':['seedvr2_ema_3b.pth'],'embeddings':['pos_emb.pt','neg_emb.pt']}
    rows=[]
    for row in lock['files']:
        name=row['rfilename']
        if args.only!='all' and name not in groups[args.only]:continue
        url=f"https://huggingface.co/{lock['repository']}/resolve/{lock['revision']}/{name}"
        rows.append(dict(filename=name,bytes=row['size'],sha256=row['lfs']['sha256'],url=url))
        if not args.list:
            download_file(args.output/name, row, url, args.offline)
    if args.list:
        print(json.dumps(dict(repository=lock['repository'],revision=lock['revision'],
                              output=str(args.output),total_bytes=sum(x['bytes'] for x in rows),
                              files=rows),indent=2))


if __name__=='__main__':
    try:
        main()
    except (ValueError, OSError) as error:
        raise SystemExit(str(error))
