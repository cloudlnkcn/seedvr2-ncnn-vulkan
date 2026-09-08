#!/usr/bin/env python3
"""Check real ncnn outputs against independent raw FP32-B reference tensors."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np


def sha(path):return hashlib.file_digest(path.open('rb'),'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--suite',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--backends',default='cpu,vulkan')
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--validation-layer',action='store_true')
    args=parser.parse_args()
    backends=args.backends.split(',')
    if not backends or len(set(backends))!=len(backends) or any(b not in ('cpu','vulkan') for b in backends):
        parser.error('--backends must be a nonempty, unique selection of cpu,vulkan')
    if args.output.exists() and any(args.output.iterdir()):raise SystemExit('Output must be empty')
    args.output.mkdir(parents=True,exist_ok=True)
    suite=json.loads(args.suite.read_text())
    if not suite.get('cases'):raise SystemExit('Reference suite must contain cases')
    rows=[]
    for item in suite['cases']:
        case_path=args.suite.parent/item['path']
        if sha(case_path)!=item['sha256']:raise SystemExit('Reference case hash mismatch')
        case=json.loads(case_path.read_text())
        for backend in backends:
            folder=args.output/(case['case_id']+'-'+backend)
            command=[str(args.binary.resolve()),'engine','awa','--case',str(case_path.resolve()),
                     '--output',str(folder.resolve()),'--backend',backend,'--gpu',str(args.gpu)]
            env=os.environ.copy()
            if args.validation_layer:env['VK_INSTANCE_LAYERS']='VK_LAYER_KHRONOS_validation'
            result=subprocess.run(command,text=True,capture_output=True,timeout=120,env=env)
            (args.output/(case['case_id']+'-'+backend+'.stdout.log')).write_text(result.stdout)
            (args.output/(case['case_id']+'-'+backend+'.stderr.log')).write_text(result.stderr)
            row={'case_id':case['case_id'],'backend':backend,'exit_code':result.returncode,'comparisons':[],'passed':False}
            if result.returncode==0:
                try:
                    run=json.loads(result.stdout)
                except json.JSONDecodeError:
                    row['error']='Invalid runner JSON; see retained stdout/stderr logs'
                    rows.append(row)
                    print(case['case_id'],backend,'FAIL',flush=True)
                    continue
                row['execution']=run
                for name in ['video','text']:
                    ref=case['reference_'+name]
                    ref_path=case_path.parent/ref['path']
                    out=run['outputs'][name];out_path=folder/out['path']
                    if sha(ref_path)!=ref['sha256'] or sha(out_path)!=out['sha256']:raise SystemExit('Tensor hash mismatch')
                    a=np.fromfile(ref_path,dtype='<f4').astype(np.float64)
                    b=np.fromfile(out_path,dtype='<f4').astype(np.float64)
                    if a.shape!=b.shape or out['shape']!=ref['shape']:raise SystemExit('Tensor shape mismatch')
                    delta=np.abs(a-b)
                    finite=bool(np.isfinite(a).all() and np.isfinite(b).all())
                    violations=int(np.count_nonzero(delta>1e-5+1e-4*np.abs(a)))
                    row['comparisons'].append({'tensor':name,'reference_sha256':ref['sha256'],
                        'candidate_sha256':out['sha256'],'elements':int(a.size),
                        'max_abs':float(delta.max()),'rmse':float(np.sqrt(np.mean(delta*delta))),
                        'violations':violations,'finite':finite,'passed':finite and violations==0})
                backend_ok=(run['vulkan_calls']==1 and run['cpu_calls']==0) if backend=='vulkan' else (run['cpu_calls']==1 and run['vulkan_calls']==0)
                diagnostics_ok=not any(x in (result.stdout+'\n'+result.stderr).lower() for x in ['validation error','vuid-','not match','failed'])
                row['passed']=backend_ok and diagnostics_ok and all(x['passed'] for x in row['comparisons'])
            else:row['error']=result.stdout[-2000:]
            rows.append(row)
            print(case['case_id'],backend,'PASS' if row['passed'] else 'FAIL',flush=True)
    report={'schema_version':'awa-validation-v1','scope':suite['scope'],'oracle':suite['oracle'],
            'suite_sha256':sha(args.suite),'binary_sha256':sha(args.binary),
            'threshold':{'atol':1e-5,'rtol':1e-4,'status':'FP32 operator diagnostic threshold; not a calibrated whole-model gate'},
            'validation_layer_requested':args.validation_layer,'cases':rows,
            'passed':all(x['passed'] for x in rows),'passed_count':sum(x['passed'] for x in rows),
            'total':len(rows),'model_verified':False}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__':main()
