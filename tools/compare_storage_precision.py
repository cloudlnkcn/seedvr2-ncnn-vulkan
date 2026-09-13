#!/usr/bin/env python3
"""Measure storage-induced drift against bound FP32 native and official trajectories.

Historical FP32 tolerances are descriptive only. Successful measurement requires
complete finite tensors, identical execution inputs/noise and package identity.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from pipeline_contract import bind_reference, bind_candidate, tensor_path, sha, load_json, tolerance, tensor_contract
from measure_image_quality import metrics
from measure_video_quality import pixels, temporal_errors


def compare(a,b,threshold):
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Tensor geometry or finite-value contract failed')
    x,y=a.astype(np.float64),b.astype(np.float64)
    delta=np.abs(x-y)
    return dict(max_abs=float(delta.max()),rmse=float(np.sqrt(np.mean(delta**2))),
                mean_abs=float(delta.mean()),
                fp32_threshold_violations=int(np.count_nonzero(delta>threshold['atol']+threshold['rtol']*np.abs(y))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('run','baseline','reference','model','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--fp32-run',type=Path,help='Retained repaired FP32 execution, bound to the same reference inputs')
    p.add_argument('--kind',choices=['image','video'],required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Output report must be new')
    ref,base,expected,ref_hash,known=bind_reference(a.reference,a.baseline,a.kind)
    fp32_root=a.fp32_run or a.baseline
    fp32=load_json(fp32_root/'run.json')
    bind_candidate(fp32,base,a.kind,expected)
    candidate=load_json(a.run/'run.json'); manifest=load_json(a.model/'manifest.json')
    profile=f'seedvr2-3b-{a.kind}-dit-fp16-storage-v1'
    if candidate['profile']!=profile or manifest['profile']!=profile or candidate['model_manifest_sha256']!=sha(a.model/'manifest.json'):
        raise ValueError('Candidate package binding differs')
    if candidate.get('storage_precision') != dict(dit_linear_weights='fp16-ieee',other_weights='fp32',activation='fp32',arithmetic='fp32'):
        raise ValueError('Candidate precision contract differs')
    # Only the explicitly intended model/profile changes are excluded from the
    # existing execution binding. Never rewrite archived candidate evidence.
    aligned=copy.deepcopy(candidate)
    aligned['profile']=base['profile'];aligned['model_manifest_sha256']=base['model_manifest_sha256']
    bind_candidate(aligned,base,a.kind,expected)
    if set(candidate['diagnostics'])!=set(base['diagnostics']) or set(candidate['diagnostics'])!=set(expected)|{'noise','posterior-noise','patches','patch-out'}:
        raise ValueError('Incomplete or extra diagnostic boundaries')
    for key in ('patches','patch-out'):
        row=candidate['diagnostics'][key]
        tensor_contract(row,key,base['diagnostics'][key]['shape'])
        values=np.fromfile(tensor_path(a.run,row),dtype='<f4')
        if values.size!=np.prod(row['shape']) or not np.isfinite(values).all():
            raise ValueError('Invalid auxiliary tensor')
    for key in ('noise','posterior-noise'):
        if sha(tensor_path(a.run,candidate['diagnostics'][key])) != sha(tensor_path(a.baseline,base['diagnostics'][key])):
            raise ValueError('Different noise input')
    official={row['stage']:row['reference'] for row in ref['stages']}
    rows=[]
    for name,shape in expected.items():
        def read(root,row):return np.fromfile(tensor_path(root,row),dtype='<f4').reshape(shape)
        x=read(a.run,candidate['diagnostics'][name]);y=read(fp32_root,fp32['diagnostics'][name]);z=read(a.reference,official[name])
        rows.append(dict(stage=name,shape=shape,vs_native_fp32=compare(x,y,tolerance(a.kind)),vs_official_fp32_b=compare(x,z,tolerance(a.kind))))
    shape=expected['decoded']
    x=np.fromfile(tensor_path(a.run,candidate['diagnostics']['decoded']),dtype='<f4').reshape(shape)
    y=np.fromfile(tensor_path(fp32_root,fp32['diagnostics']['decoded']),dtype='<f4').reshape(shape)
    if a.kind=='video':
        frames=candidate['output']['frames']; x,y=pixels(x,frames),pixels(y,frames)
    else:
        def image(v):return np.floor(((np.clip(v,-1,1)*np.float32(.5)+np.float32(.5))*np.float32(255)).astype(np.float64)+.5).clip(0,255).astype(np.uint8).transpose(1,2,0)[None]
        x,y=image(x),image(y)
    quality=dict(per_frame_vs_native_fp32=[metrics(u,v) for u,v in zip(x,y)],
                 scope='Decoded pre-encoding RGB preservation versus FP32; not task quality versus ground truth.')
    if a.kind=='video':quality['temporal_residual_vs_native_fp32']=temporal_errors(x,y)
    result=dict(schema_version='seedvr2-storage-drift-v1',kind=a.kind,measurement_complete=True,model_verified=False,
                historical_threshold_role='DESCRIPTIVE_NOT_LOW_PRECISION_ACCEPTANCE', boundaries=len(rows),stages=rows,
                reference_report_sha256=ref_hash,baseline_run_sha256=sha(a.baseline/'run.json'),fp32_run_sha256=sha(fp32_root/'run.json'),candidate_run_sha256=sha(a.run/'run.json'),
                model_manifest_sha256=sha(a.model/'manifest.json'),script_sha256=sha(Path(__file__)),quality=quality)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(measurement_complete=True,boundaries=len(rows),quality=quality)))

if __name__=='__main__':main()
