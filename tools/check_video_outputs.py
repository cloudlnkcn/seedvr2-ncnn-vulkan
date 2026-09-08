#!/usr/bin/env python3
"""Compare a new native video run to identified retained official tensors."""
import argparse
import json
from pathlib import Path
import numpy as np
from check_graph import sha,compare


def main():
    p=argparse.ArgumentParser()
    for name in ('run','reference','reference-run','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run=json.loads((a.run/'run.json').read_text());ref=json.loads((a.reference/'report.json').read_text())
    baseline=json.loads((a.reference_run/'run.json').read_text())
    if run['status']!='SUCCEEDED' or ref['schema_version']!='seedvr2-video-parity-v1' or sha(a.reference_run/'run.json')!=ref['native_run_sha256']:
        raise ValueError('Invalid reference binding')
    for field in ('profile','model_manifest_sha256','input','sampling','seed','noise_algorithm','clip'):
        if run[field]!=baseline[field]:raise ValueError('Execution differs: '+field)
    for name in ('posterior-noise','noise'):
        x,y=run['diagnostics'][name],baseline['diagnostics'][name]
        if x['shape']!=y['shape'] or x['sha256']!=y['sha256'] or sha(a.run/x['path'])!=x['sha256'] or sha(a.reference_run/y['path'])!=y['sha256']:
            raise ValueError('Different raw noise')
    rows=[]
    for stage in ref['stages']:
        name=stage['stage'];r=compare(stage['reference'],run['diagnostics'][name],a.reference,a.run,ref['tolerance'])
        rows.append(dict(stage=name,**r))
    identity=sha(a.run/run['output']['path'])==run['output']['sha256']
    diagnostics=True
    for suffix in ('stdout','stderr'):
        log=a.run.parent/(a.run.name+'.'+suffix)
        if log.exists() and any(x in log.read_text().lower() for x in ('vuid-','validation error')):diagnostics=False
    report=dict(schema_version='seedvr2-video-replay-parity-v1',passed=all(x['passed'] for x in rows) and identity and diagnostics,
        model_verified=False,backend=run['backend'],run_sha256=sha(a.run/'run.json'),reference_report_sha256=sha(a.reference/'report.json'),
        reference_run_sha256=sha(a.reference_run/'run.json'),input_and_noise_identity_verified=True,script_sha256=sha(Path(__file__)),
        tolerance=ref['tolerance'],stages=rows)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(report['passed'],sum(x['passed'] for x in rows),'/',len(rows),[(x['stage'],x['violations']) for x in rows if not x['passed']],flush=True)
    raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__':main()
