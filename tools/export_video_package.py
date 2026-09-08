#!/usr/bin/env python3
"""Assemble a temporal package, retaining the same verified 32 DiT exports."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--image-package',type=Path,required=True)
    p.add_argument('--vae',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Output must be empty')
    manifest=json.loads((args.image_package/'manifest.json').read_text())
    if manifest['schema_version']!='seedvr2-image-package-v1' or len(manifest['graphs'])!=36:
        raise ValueError('Complete source package required')
    vae=json.loads((args.vae/'suite.json').read_text())
    lock=manifest['model_sources']
    if vae['schema_version']!='vae-video-export-v1' or vae['checkpoint']['sha256']!=next(x['lfs']['sha256'] for x in lock['files'] if x['rfilename']=='ema_vae.pth'):
        raise ValueError('Temporal VAE checkpoint identity mismatch')
    out=copy.deepcopy(manifest)
    out.update(schema_version='seedvr2-video-package-v1',profile='seedvr2-3b-video-fp32-b-v1',
               limits=dict(min_side=64,max_side=128,max_frames=17,divisible_by=16,streaming_cache=False),
               temporal=dict(vae='whole clip, causal, memory disabled',attention='3D adaptive windows',
                             frame_padding='repeat tail to 4n+1, trim output',audio=False),
               exporter=dict(script_sha256=sha(Path(__file__)),image_manifest_sha256=sha(args.image_package/'manifest.json'),
                             vae_suite_sha256=sha(args.vae/'suite.json'),pnnx_sha256=vae['pnnx_sha256']))
    args.output.mkdir(parents=True,exist_ok=True)
    for row in out['graphs']:
        temporal=row['id'] in ('encoder','decoder')
        part=next((x for x in vae['parts'] if x['part']==row['id']),None)
        for key,suffix in (('param','param'),('weights','bin')):
            source=args.vae/row['id']/f'model.ncnn.{suffix}' if temporal else args.image_package/row[key]['path']
            expected=part[f'model_{suffix}_sha256'] if temporal else row[key]['sha256']
            if sha(source)!=expected:
                raise ValueError('Changed source graph')
            target=args.output/row[key]['path'];target.parent.mkdir(parents=True,exist_ok=True)
            os.link(source,target)
            row[key].update(sha256=expected,bytes=target.stat().st_size)
    for row in out['constants'].values():
        source=args.image_package/row['path']
        if sha(source)!=row['sha256']:
            raise ValueError('Changed conditioning constant')
        os.link(source,args.output/row['path'])
    tmp=args.output/'manifest.tmp.json';tmp.write_text(json.dumps(out,indent=2)+'\n');tmp.rename(args.output/'manifest.json')
    print(json.dumps(dict(status='PACKAGED',sha256=sha(args.output/'manifest.json'),graphs=36,model_verified=False)))


if __name__=='__main__':
    main()
