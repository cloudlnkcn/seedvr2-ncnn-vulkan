#!/usr/bin/env python3
"""Download a pinned, ready-to-run ncnn package from Hugging Face. No conversion."""
import argparse
import json
from pathlib import Path
import re
import subprocess
from model_distribution import install, sha_file

ROOT=Path(__file__).resolve().parents[1]


def resolve_entry(precision):
    registry=json.loads((ROOT/'docs/distribution/huggingface-models.json').read_text())
    entry=registry['models'][precision]
    if not entry.get('download_verified') or not re.fullmatch('[0-9a-f]{40}',entry['revision']):
        raise ValueError('This model download has not been verified for publication')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',entry['repo_id']):
        raise ValueError('Invalid repository identity')
    catalog_path=ROOT/'docs/distribution'/entry['catalog']
    if catalog_path.resolve().parent != (ROOT/'docs/distribution').resolve() or sha_file(catalog_path)!=entry['catalog_sha256']:
        raise ValueError('Download catalogue identity differs')
    return entry,json.loads(catalog_path.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precision',choices=['fp32','dit-fp16'],default='dit-fp16')
    parser.add_argument('--kind',choices=['image','video'],default='image')
    parser.add_argument('--output',type=Path,help='Model directory; repeated calls verify and reuse it')
    parser.add_argument('--binary',type=Path,default=ROOT/'dist/tutorial/bin/seedvr2')
    parser.add_argument('--plan',action='store_true',help='Show download size without network access or writes')
    parser.add_argument('--offline',action='store_true',help='Verify/reuse downloaded files without network')
    parser.add_argument('--run',type=Path,help='After installation, restore this input with native CLI')
    parser.add_argument('--result',type=Path,help='New restoration output directory, required with --run')
    parser.add_argument('--backend',choices=['cpu','vulkan'],default='vulkan')
    parser.add_argument('--size',type=int,help='Output long side; defaults to 256 for image, 128 for video')
    parser.add_argument('--frames',type=int,default=17)
    args=parser.parse_args()
    if bool(args.run)!=bool(args.result):parser.error('--run and --result must be used together')
    if args.run and (not args.run.is_file() or args.result.exists()):parser.error('Input must exist and result directory must be new')
    size=args.size or (256 if args.kind=='image' else 128)
    if size%16 or not 64<=size<=(512 if args.kind=='image' else 128) or not 1<=args.frames<=17:
        parser.error('Size or frame count is outside native scope')
    entry,catalog=resolve_entry(args.precision)
    output=(args.output or ROOT/'models'/f'{args.precision}-{args.kind}').resolve()
    if not args.plan:
        if not args.binary.is_file():parser.error('Build the native application first, or supply --binary; --plan needs no binary')
        caps=json.loads(subprocess.check_output([str(args.binary.resolve()),'capabilities'],text=True))
        if args.precision=='dit-fp16' and 'dit-fp16' not in caps.get('model_storage_formats',[]):
            parser.error('This executable predates DiT FP16 model support; rebuild the current source')
    reviewed=json.loads((ROOT/'policies/reviewed-packages.json').read_text())
    result=install(catalog,args.kind,output,reviewed,
                   base_url=f'https://huggingface.co/{entry["repo_id"]}/resolve/{entry["revision"]}',
                   offline=args.offline,plan=args.plan)
    if not args.plan:
        subprocess.run([str(args.binary.resolve()),'models','verify','--kind',args.kind,'--model',str(output)],check=True)
    result.update(precision=args.precision,repo_id=entry['repo_id'],revision=entry['revision'],model_directory=str(output))
    print(json.dumps(result,indent=2))
    if args.run:
        command=[str(args.binary.resolve()),'run' if args.kind=='image' else 'run-video','--model',str(output),
                 '--input',str(args.run.resolve()),'--output',str(args.result.resolve()),'--backend',args.backend,'--size',str(size)]
        if args.kind=='video':command+=['--frames',str(args.frames)]
        if args.plan:print(json.dumps(dict(planned_native_command=command)))
        else:subprocess.run(command,check=True)

if __name__=='__main__':main()
