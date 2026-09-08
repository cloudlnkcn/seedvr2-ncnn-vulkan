#!/usr/bin/env python3
"""Download immutable official checkpoints, verify LFS hashes, never execute them."""
import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('.cache/models'))
    parser.add_argument('--only',choices=['vae','dit','embeddings','all'],default='all')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    lock=json.loads((root/'model-sources.lock.json').read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    groups={'vae':['ema_vae.pth'],'dit':['seedvr2_ema_3b.pth'],'embeddings':['pos_emb.pt','neg_emb.pt']}
    for row in lock['files']:
        name=row['rfilename']
        if args.only!='all' and name not in groups[args.only]:continue
        target=args.output/name
        def verified(path):
            return path.exists() and path.stat().st_size==row['size'] and hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==row['lfs']['sha256']
        if verified(target):
            print(name,'hash verified (cached)',flush=True);continue
        if target.exists():raise SystemExit(f'Refusing to replace a mismatched checkpoint: {target}')
        partial=target.with_suffix(target.suffix+'.partial')
        offset=partial.stat().st_size if partial.exists() else 0
        url=f"https://huggingface.co/{lock['repository']}/resolve/{lock['revision']}/{name}"
        headers={'Range':f'bytes={offset}-'} if offset else {}
        request=urllib.request.Request(url,headers=headers)
        print(name,'downloading from',lock['revision'],'offset',offset,flush=True)
        with urllib.request.urlopen(request,timeout=60) as response:
            if offset and response.status!=206:raise SystemExit('Server did not honor resume range')
            last=time.monotonic()
            with partial.open('ab' if offset else 'wb') as out:
                while True:
                    data=response.read(4*1024*1024)
                    if not data:break
                    offset+=len(data)
                    if offset>row['size']:raise SystemExit('Checkpoint exceeds locked size')
                    out.write(data)
                    if time.monotonic()-last>20:
                        print(name,offset,'/',row['size'],flush=True);last=time.monotonic()
        if not verified(partial):raise SystemExit(f'Checkpoint size or hash mismatch: {name}')
        partial.rename(target)
        print(name,'SHA256 verified',row['lfs']['sha256'],flush=True)


if __name__=='__main__':main()
