#!/usr/bin/env python3
"""Extract actual same-input block cases from an already reviewed trajectory.

This reuses the official inputs and outputs, never native intermediate inputs.
It isolates local component error from accumulated end-to-end error. It does
not create a new end-to-end reference or approve one automatically.
"""
import argparse
import json
import os
from pathlib import Path
from pipeline_contract import bind_reference, load_json, sha, tensor_path, tolerance


def main():
    p = argparse.ArgumentParser()
    for name in ('reference', 'reference-run', 'package', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--blocks', type=int, nargs='+', default=[19, 31])
    args = p.parse_args()
    if not args.blocks or len(set(args.blocks)) != len(args.blocks) or any(x < 1 or x > 31 for x in args.blocks):
        p.error('Choose distinct blocks 1..31 with a retained preceding output')
    ref, run, shapes, digest, known = bind_reference(args.reference, args.reference_run, 'video')
    if sha(args.package/'manifest.json') != known['model_manifest_sha256']:
        raise ValueError('Package is not the reviewed trajectory package')
    package = load_json(args.package/'manifest.json')
    graphs = {x['id']: x for x in package['graphs']}
    stages = {x['stage']: x['reference'] for x in ref['stages']}
    grid = [run['clip']['latent_frames'], run['output']['height']//16, run['output']['width']//16, 2560]
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    for index in args.blocks:
        folder = args.output/f'official-trajectory-block-{index:02d}'
        folder.mkdir()
        def copy_tensor(stage, filename, shape):
            source = tensor_path(args.reference, stages[stage])
            (folder/filename).write_bytes(source.read_bytes())
            return dict(path=filename, dtype='f32le', shape=shape, sha256=sha(folder/filename))
        checkpoint = package['model_sources']
        checkpoint_sha = next(x['lfs']['sha256'] for x in checkpoint['files'] if x['rfilename']=='seedvr2_ema_3b.pth')
        case = dict(schema_version='dit-block-case-v1', case_id=folder.name, component='dit-block',
                    block_index=index, checkpoint_sha256=checkpoint_sha,
                    reference_profile='FP32-B', precision='fp32',
                    inputs=[copy_tensor(f'block-{index-1:02d}-video','input-0.f32',grid),
                            copy_tensor(f'block-{index-1:02d}-text','input-1.f32',[58,2560]),
                            copy_tensor('time-in','input-2.f32',[2560,6])],
                    reference_video=copy_tensor(f'block-{index:02d}-video','reference-video.f32',grid),
                    reference_text=copy_tensor(f'block-{index:02d}-text','reference-text.f32',[58,2560]),
                    provenance=dict(reference_report_sha256=digest, baseline_run_sha256=known['baseline_run_sha256'],
                                    model_manifest_sha256=known['model_manifest_sha256'], mode='official-inputs-official-outputs'))
        for key, filename, target in [('param','model.ncnn.param','model_param'),('weights','model.ncnn.bin','model_bin')]:
            artifact = graphs[f'block-{index:02d}'][key]
            source = args.package/artifact['path']
            if sha(source) != artifact['sha256']:
                raise ValueError('Graph differs from reviewed package')
            os.link(source, folder/filename)
            case[target] = dict(path=filename, sha256=artifact['sha256'])
        path = folder/'case.json'
        path.write_text(json.dumps(case, indent=2)+'\n')
        cases.append(dict(path=str(path.relative_to(args.output)), sha256=sha(path)))
    suite = dict(schema_version='same-input-dit-boundary-suite-v1',
                 scope='Blocks from a reviewed 17-frame official trajectory; local numerical diagnosis only',
                 reference_profile='FP32-B', checkpoint=checkpoint_sha, tolerance=tolerance('video'), cases=cases,
                 reference_report_sha256=digest, generator_sha256=sha(Path(__file__)))
    (args.output/'suite.json').write_text(json.dumps(suite, indent=2)+'\n')


if __name__ == '__main__':
    main()
