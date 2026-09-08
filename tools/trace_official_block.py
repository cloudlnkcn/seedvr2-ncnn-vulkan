#!/usr/bin/env python3
"""Retain official FP32-B block internals on fixed diagnostic inputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from dit_block_reference import evaluate, make_block
from awa_reference import verify_sources
from pipeline_contract import load_json, sha, tensor_path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    doc=load_json(args.case)
    sources=verify_sources()
    lock=load_json(Path(__file__).resolve().parents[1]/'model-sources.lock.json')
    expected=next(row['lfs']['sha256'] for row in lock['files'] if row['rfilename']=='seedvr2_ema_3b.pth')
    if sha(args.checkpoint)!=expected or doc['schema_version']!='dit-block-case-v1':
        raise ValueError('Checkpoint/case identity differs')
    torch.set_num_threads(4)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    block,cache=make_block(state,doc['block_index'])
    report=dict(schema_version='seedvr2-official-block-trace-v1',case_sha256=sha(args.case),
                checkpoint_sha256=expected,official_sources=sources,
                profile='FP32-B',torch_version=torch.__version__,model_verified=False,
                script_sha256=sha(Path(__file__)),layers=[])
    def record(name,tensors):
        if isinstance(tensors,torch.Tensor): tensors=(tensors,)
        row=dict(name=name,outputs=[])
        for tensor in tensors:
            if not isinstance(tensor,torch.Tensor) or not tensor.is_floating_point(): continue
            tensor=tensor.detach().float().contiguous()
            name=f'tensor-{len(report["layers"]):03d}-{len(row["outputs"])}.f32'
            path=args.output/name
            tensor.numpy().astype('<f4').tofile(path)
            row['outputs'].append(dict(path=name,dtype='f32le',shape=list(tensor.shape),sha256=sha(path)))
        report['layers'].append(row)
    for name in ('attn_norm','attn.proj_qkv','attn.norm_q','attn.norm_k','attn.rope',
                 'attn.attn','attn.proj_out','mlp_norm','mlp'):
        block.get_submodule(name).register_forward_hook(lambda module,inputs,output,name=name: record(name,output))
    block.ada.register_forward_hook(lambda module,inputs,kwargs,output:
        record('ada.'+kwargs['layer']+'.'+kwargs['mode'],output),with_kwargs=True)
    block.attn.attn.register_forward_pre_hook(lambda module,inputs,kwargs:
        record('sdpa.inputs',[kwargs[key] for key in ('q','k','v')]),with_kwargs=True)
    inputs=[torch.from_numpy(np.fromfile(tensor_path(args.case.parent,row),dtype='<f4').reshape(row['shape']).copy()) for row in doc['inputs']]
    output=evaluate(block,cache,*inputs)
    record('output',output)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Recorded',len(report['layers']),'official block boundaries',flush=True)


if __name__=='__main__':
    main()
