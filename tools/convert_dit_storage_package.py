#!/usr/bin/env python3
"""Convert reviewed FP32 packages to DiT FP16-storage candidates; no trust promotion."""
import argparse
import copy
import json
import os
from pathlib import Path
from dit_weight_storage import rewrite
from model_distribution import manifest_files, sha_file


def convert(source, output, reviewed, shared=None):
    manifest = json.loads((source/'manifest.json').read_text())
    kind = 'video' if manifest['schema_version'] == 'seedvr2-video-package-v1' else 'image'
    rows = manifest_files(manifest, kind, reviewed)
    for row in rows:
        file = source/row['path']
        if file.stat().st_size != row['bytes'] or sha_file(file) != row['sha256']:
            raise ValueError('Source package bytes changed')
    output.mkdir(parents=True, exist_ok=False)
    result = copy.deepcopy(manifest)
    result['profile'] = f'seedvr2-3b-{kind}-dit-fp16-storage-v1'
    result['storage_precision'] = dict(dit_linear_weights='fp16-ieee', other_weights='fp32',
                                       activation='fp32', arithmetic='fp32')
    result['exporter'] = dict(source_manifest_sha256=sha_file(source/'manifest.json'),
                              script_sha256=sha_file(Path(__file__)),
                              storage_converter_sha256=sha_file(Path(__file__).with_name('dit_weight_storage.py')))
    audit=[]
    for graph in result['graphs']:
        for key in ('param','weights'):
            row=graph[key]; destination=output/row['path']; destination.parent.mkdir(parents=True,exist_ok=True)
            if key == 'weights' and graph['id'].startswith('block-'):
                if shared:
                    other=json.loads((shared/'manifest.json').read_text())
                    prior=next(g for g in other['graphs'] if g['id']==graph['id'])
                    original_graph=next(g for g in manifest['graphs'] if g['id']==graph['id'])
                    if prior['param'] != original_graph['param']:
                        raise ValueError('Shared graph layout differs')
                    # Require the shared package to derive from the same reviewed source weights.
                    record=json.loads((shared/'storage-audit.json').read_text())
                    proof=next(g for g in record if g['id']==graph['id'])
                    if proof['source_sha256'] != row['sha256'] or sha_file(shared/row['path']) != prior['weights']['sha256']:
                        raise ValueError('Shared weights identity differs')
                    os.link(shared/row['path'],destination); audit.append(proof)
                else:
                    details=rewrite(source/graph['param']['path'],source/row['path'],destination)
                    audit.append(dict(id=graph['id'],source_sha256=row['sha256'],layers=details))
                row.update(bytes=destination.stat().st_size,sha256=sha_file(destination))
                print(kind,graph['id'],'converted',flush=True)
            else:
                os.link(source/row['path'],destination)
    for row in result['constants'].values():os.link(source/row['path'],output/row['path'])
    (output/'storage-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--shared-image',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    convert(args.source,args.output,json.loads((root/'policies/reviewed-packages.json').read_text()),args.shared_image)

if __name__=='__main__':main()
