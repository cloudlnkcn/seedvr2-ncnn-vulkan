#!/usr/bin/env python3
"""Compare retained CPU encoder boundaries on identical inputs; no certification."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from pipeline_contract import load_json, sha, tensor_path
from vae_reference import ROOT, load_checkpoint, make_reference, verify_sources


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('trace','case','output'): p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise ValueError('Output already exists')
    doc=load_json(args.trace/'report.json');case=load_json(args.case)
    if doc['status']!='EXECUTED' or doc['case_sha256']!=sha(args.case): raise ValueError('Trace/case identity mismatch')
    if case['input']['shape']!=[3,17,128,128] or case['component']!='vae-video-encoder': raise ValueError('Unexpected fixture')
    rows={r['name']:r for r in doc['layers']}
    names={'conv_in','norm1','add_10','pnnx_unique_45','pnnx_unique_46'}
    if len(doc['layers'])!=5 or set(rows)!=names: raise ValueError('Incomplete trace')
    def read(root,r):
        return torch.from_numpy(np.fromfile(tensor_path(root,r),'<f4').reshape(r['shape']).copy())
    def compare(a,b):
        if a.shape!=b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all(): raise ValueError('Shape or finite-value failure')
        d=(a.double()-b.double()).abs()
        return dict(max_abs=d.max().item(),rmse=d.square().mean().sqrt().item(),elements=d.numel())
    torch.set_num_threads(4)
    state,lock=load_checkpoint(ROOT/'.cache/models/ema_vae.pth')
    encoder,memory=make_reference(state,'encoder')
    report=dict(schema_version='seedvr2-vae-isolation-v1',diagnostic_only=True,model_verified=False,
                case_sha256=sha(args.case),native_trace_sha256=sha(args.trace/'report.json'),
                script_sha256=sha(Path(__file__)),torch_version=torch.__version__,
                official_sources=verify_sources(),checkpoint_sha256=next(r['lfs']['sha256'] for r in lock['files'] if r['rfilename']=='ema_vae.pth'),boundaries={})
    with torch.inference_mode():
        source=read(args.case.parent,case['input'])
        first=read(args.trace,rows['conv_in'])
        reference=encoder.conv_in(source[None],memory_state=memory)[0]
        report['boundaries']['first_conv_same_prepared']=compare(first,reference)
        del source,reference
        for input_name,output_name,norm in [('conv_in','norm1',encoder.down_blocks[0].resnets[0].norm1),('add_10','pnnx_unique_45',encoder.conv_norm_out)]:
            x=first if input_name=='conv_in' else read(args.trace,rows[input_name])
            reference=norm(x.transpose(0,1)).transpose(0,1)
            actual=read(args.trace,rows[output_name])
            report['boundaries'][output_name+'_same_native_input']=compare(actual,reference)
            double=norm.to(dtype=torch.float64)(x.double().transpose(0,1)).transpose(0,1)
            report['boundaries'][output_name+'_native_vs_fp64']=compare(actual,double)
            report['boundaries'][output_name+'_official_vs_fp64']=compare(reference,double)
            del x,reference,actual,double
        report['boundaries']['posterior_full_encoder']=compare(read(args.trace,rows['pnnx_unique_46']),read(args.case.parent,case['reference']))
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report['boundaries'],indent=2))


if __name__=='__main__': main()
