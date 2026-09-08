#!/usr/bin/env python3
"""Export dynamic TorchScript -> real pnnx -> checked ncnn custom-layer lowering."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile
import numpy as np
import torch
from awa_export_module import ExportModel


def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()


def lower(folder):
    raw=(folder/'awa.raw.ncnn.param').read_text().splitlines()
    if len(raw)!=5 or raw[0]!='7767517' or raw[1].split()!=['3','4']:
        raise ValueError('Unexpected pnnx graph structure')
    expected_inputs=[['Input','in0','0','1','in0'],['Input','in1','0','1','in1']]
    if [line.split() for line in raw[2:4]]!=expected_inputs:
        raise ValueError('Unexpected pnnx graph inputs')
    op=raw[4].split()
    if op[:8]!=['awa_export_module.AdaptiveWindowAttention','awa','2','2','in0','in1','out0','out1']:
        raise ValueError('Expected one two-output preserved AWA module')
    if set(op[8:])!={'-23310=1,1','-23311=2,4,128','-23312=1,21','-23313=1,3'}:
        raise ValueError('Unexpected serialized AWA attributes')
    with zipfile.ZipFile(folder/'awa.pnnx.bin') as archive:
        if sorted(archive.namelist())!=['awa.epsilon','awa.norm_weights','awa.rope_frequencies','awa.spec']:
            raise ValueError('Unexpected pnnx weight archive members')
        if [archive.getinfo(x).file_size for x in ['awa.epsilon','awa.norm_weights','awa.rope_frequencies','awa.spec']]!=[4,2048,84,12]:
            raise ValueError('Invalid pnnx tensor sizes')
        weights=archive.read('awa.norm_weights')
        spec=np.frombuffer(archive.read('awa.spec'),dtype='<i4')
        eps=float(np.frombuffer(archive.read('awa.epsilon'),dtype='<f4')[0])
        frequencies=np.frombuffer(archive.read('awa.rope_frequencies'),dtype='<f4')
    if spec[0]!=1 or not 1<=spec[1]<=20 or spec[2] not in [0,1] or not 0<eps<=0.01 or not np.isfinite(eps):
        raise ValueError('Unsupported AWA semantic parameters')
    if not np.isfinite(np.frombuffer(weights,dtype='<f4')).all():raise ValueError('Non-finite weights')
    expected=(1./(10000.**(torch.arange(0,42,2).float()/42.))).numpy()
    if not np.array_equal(frequencies,expected):raise ValueError('Unsupported RoPE frequency table')
    raw[4]=f'SeedVR2AWA awa 2 2 in0 in1 out0 out1 0={spec[1]} 1={spec[2]} 2={eps:.9e} 3=1'
    (folder/'awa.ncnn.param').write_text('\n'.join(raw)+'\n')
    (folder/'awa.ncnn.bin').write_bytes(weights)
    return {'heads':int(spec[1]),'shifted':bool(spec[2]),'epsilon':eps,'format_version':1}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--suite',type=Path,required=True)
    parser.add_argument('--pnnx',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):raise SystemExit('Export output directory must be empty')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    suite=json.loads(args.suite.read_text());new_cases=[];reports=[]
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    for item in suite['cases']:
        source=args.suite.parent/item['path']
        if sha(source)!=item['sha256']:raise ValueError('Reference case changed')
        case=json.loads(source.read_text());folder=args.output/case['case_id'];folder.mkdir()
        def load(key):
            row=case[key];p=source.parent/row['path']
            if sha(p)!=row['sha256']:raise ValueError('Reference artifact changed')
            return torch.from_numpy(np.fromfile(p,dtype='<f4').reshape(row['shape']).copy())
        module=ExportModel(case['heads'],case['shifted']).eval()
        with torch.no_grad():module.awa.norm_weights.copy_(load('model_bin'))
        scripted=torch.jit.script(module)
        scripted.save(str(folder/'awa.pt'))
        with torch.inference_mode(),torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            outputs=scripted(load('video_qkv'),load('text_qkv'))
        checks=[]
        for key,value in zip(['reference_video','reference_text'],outputs):
            expected=load(key);d=(value-expected).abs()
            torch.testing.assert_close(value,expected,atol=1e-5,rtol=1e-4)
            checks.append({'tensor':key,'max_abs':float(d.max()),'rmse':float(d.double().square().mean().sqrt())})
        width=3*case['heads']*128
        # Two different shapes force dynamic T/H/W and text length in pnnx metadata.
        command=[str(args.pnnx.resolve()),'awa.pt',f'inputshape=[2,3,4,{width}],[5,{width}]',
                 f'inputshape2=[3,5,7,{width}],[7,{width}]',
                 'moduleop=awa_export_module.AdaptiveWindowAttention','fp16=0',
                 'ncnnparam=awa.raw.ncnn.param','ncnnbin=awa.raw.ncnn.bin']
        result=subprocess.run(command,cwd=folder,text=True,capture_output=True,timeout=120,env=env)
        (folder/'pnnx.log').write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError('pnnx failed; see '+str(folder/'pnnx.log'))
        dynamic=(folder/'awa.pnnx.param').read_text()
        if f'(?,?,?,{width})f32' not in dynamic or f'(?,{width})f32' not in dynamic:
            raise ValueError('pnnx did not preserve dynamic input axes')
        params=lower(folder)
        if params['heads']!=case['heads'] or params['shifted']!=case['shifted']:raise ValueError('Lowered semantic parameters differ')
        if sha(folder/'awa.ncnn.bin')!=case['model_bin']['sha256']:raise ValueError('pnnx changed norm weights')
        for key in ['video_qkv','text_qkv','reference_video','reference_text']:
            shutil.copyfile(source.parent/case[key]['path'],folder/case[key]['path'])
        case['model_param']={'path':'awa.ncnn.param','sha256':sha(folder/'awa.ncnn.param')}
        case['model_bin']={'path':'awa.ncnn.bin','sha256':sha(folder/'awa.ncnn.bin'),'shape':[4,128],'dtype':'f32le'}
        case['export_status']='PNNX_MODULEOP_WITH_CHECKED_CUSTOM_LOWERING'
        (folder/'case.json').write_text(json.dumps(case,indent=2)+'\n')
        new_cases.append({'path':f"{case['case_id']}/case.json",'sha256':sha(folder/'case.json')})
        reports.append({'case_id':case['case_id'],'scripted_vs_official_B':checks,'dynamic_input_axes':True,
            'parameters':params,'files':[{ 'path':x,'sha256':sha(folder/x)} for x in ['awa.pt','awa.pnnx.param','awa.pnnx.bin','awa.raw.ncnn.param','awa.ncnn.param','awa.ncnn.bin']]})
        print(case['case_id'],'exported; scripted reference PASS',flush=True)
    suite['cases']=new_cases;suite['parent_suite_sha256']=sha(args.suite)
    (args.output/'suite.json').write_text(json.dumps(suite,indent=2)+'\n')
    lock=json.loads((Path(__file__).resolve().parents[1]/'engine-dependencies.lock.json').read_text())
    report={'schema_version':'awa-export-v1','pnnx_sha256':sha(args.pnnx),'ncnn_commit':lock['ncnn']['commit'],
        'torch':torch.__version__,'module_source_sha256':sha(Path(__file__).with_name('awa_export_module.py')),
        'lowerer_sha256':sha(Path(__file__)),'passed':True,'total':len(reports),'cases':reports,'model_verified':False}
    (args.output/'export-report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
