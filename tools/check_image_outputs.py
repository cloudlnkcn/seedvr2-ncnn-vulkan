#!/usr/bin/env python3
"""Compare additional actual native executions to a retained complete reference."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image

def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--reference-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    run=json.loads((a.run/'run.json').read_text());ref=json.loads((a.reference/'report.json').read_text());rows=[]
    baseline=json.loads((a.reference_run/'run.json').read_text())
    assert ref['passed'] and run['status']=='SUCCEEDED'
    assert digest(a.reference_run/'run.json')==ref['native_run_sha256']
    for field in ['profile','model_manifest_sha256','input','sampling','seed','noise_algorithm']:
        assert run[field]==baseline[field],f'Reference execution differs: {field}'
    for name in ['posterior-noise','noise']:
        x=run['diagnostics'][name];y=baseline['diagnostics'][name]
        assert x['shape']==y['shape'] and x['sha256']==y['sha256']
        assert digest(a.run/x['path'])==x['sha256'] and digest(a.reference_run/y['path'])==y['sha256']
    assert digest(a.run/'output.png')==run['output']['sha256']
    for suffix in ['stdout','stderr']:
        log=a.run.parent/(a.run.name+'.'+suffix)
        assert not log.exists() or not any(x in log.read_text() for x in ['VUID-','Validation Error'])
    tol=ref['tolerance']
    for stage in ref['stages']:
        name=stage['stage'];xrow=run['diagnostics'][name];yrow=stage['reference']
        xp=a.run/xrow['path'];yp=a.reference/yrow['path']
        assert digest(xp)==xrow['sha256'] and digest(yp)==yrow['sha256']
        x=np.fromfile(xp,dtype='<f4').astype(np.float64);y=np.fromfile(yp,dtype='<f4').astype(np.float64)
        assert x.shape==y.shape and xrow['shape']==yrow['shape']
        diff=np.abs(x-y);bad=int(np.sum(diff>tol['atol']+tol['rtol']*np.abs(y)))
        rows.append(dict(stage=name,passed=bool(bad==0 and np.isfinite(x).all() and np.isfinite(y).all()),max_abs=float(diff.max()),rmse=float(np.sqrt(np.mean(diff**2))),violations=bad))
    pixels=np.abs(np.asarray(Image.open(a.run/'output.png')).astype(np.int16)-np.asarray(Image.open(a.reference/'reference.png')).astype(np.int16))
    report=dict(schema_version='seedvr2-image-replay-parity-v1',passed=all(r['passed'] for r in rows) and bool(pixels.max()<=tol['max_uint8_error']),model_verified=False,backend=run['backend'],run_sha256=digest(a.run/'run.json'),reference_report_sha256=digest(a.reference/'report.json'),reference_run_sha256=digest(a.reference_run/'run.json'),input_and_noise_identity_verified=True,script_sha256=digest(Path(__file__)),tolerance=tol,pixel_max_abs=int(pixels.max()),stages=rows)
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(report['passed'],len(rows),report['pixel_max_abs']);raise SystemExit(0 if report['passed'] else 1)
if __name__=='__main__':main()
