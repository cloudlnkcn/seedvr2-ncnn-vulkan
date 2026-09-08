#!/usr/bin/env python3
"""Native first-use/preflight contract checks, with optional real package checks."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--model',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); binary=a.binary.resolve(); root=Path(__file__).resolve().parents[1]; rows=[]
    def run(name,args,code,predicate=lambda doc: True,json_output=True):
        start=time.monotonic(); r=subprocess.run([str(binary),*map(str,args)],capture_output=True,text=True,timeout=30)
        doc=json.loads(r.stdout) if json_output else r.stdout
        passed=r.returncode==code and predicate(doc)
        rows.append(dict(case=name,passed=bool(passed),exit_code=r.returncode,elapsed_seconds=time.monotonic()-start))
        if not passed: raise AssertionError((name,r.returncode,r.stdout,r.stderr))
        return doc
    with tempfile.TemporaryDirectory(prefix='seedvr2-路径 проверка ',dir=a.model.resolve().parent if a.model else None) as tmp:
        folder=Path(tmp); output=folder/'结果'; model=folder/'模型';model.mkdir()
        input_file=root/'tests/fixtures/natural/astronaut-degraded.jpg'
        base=['run','--model',model,'--input',input_file,'--output',output,'--backend','cpu','--check']
        error=lambda d: 'error' in d
        run('first-run-help',[],0,lambda d:'run-video' in d and 'models' in d,False)
        run('preflight-missing-model',base,2,lambda d:d['error']['field']=='model')
        run('preflight-invalid-size',[*base,'--size','129'],2,lambda d:d['error']['field']=='parameters')
        for flags in (['--weights','host'],['--weights','device'],['--gpu-reserve-mib','1']):
            run('cpu-memory-option-'+flags[-1],[*base,*flags],2,
                lambda d:d['error']['field']=='parameters' and 'require the Vulkan backend' in d['error']['message'])
        for name, flags in [('unknown-placement',['--weights','magic']),
                            ('negative-reserve',['--gpu-reserve-mib','-1']),
                            ('reserve-overflow',['--gpu-reserve-mib',str(2**64//(1024*1024))])]:
            run(name,[*base,*flags],2,lambda d:d['error']['code']=='CLI_USAGE')
        vulkan_base=list(base);vulkan_base[vulkan_base.index('--backend')+1]='vulkan'
        run('conflicting-host-reserve',[*vulkan_base,'--weights','host','--gpu-reserve-mib','1'],2,
            lambda d:d['error']['field']=='parameters' and 'requires automatic' in d['error']['message'])
        missing_input=list(base);missing_input[missing_input.index('--input')+1]=folder/'missing.png'
        run('preflight-missing-input',missing_input,2,lambda d:d['error']['field']=='input')
        output.mkdir();keep=output/'keep.txt';keep.write_text('keep')
        run('preflight-nonempty-output',base,2,lambda d:d['error']['field']=='output')
        assert keep.read_text()=='keep';keep.unlink();output.rmdir()
        run('verify-missing-model',['models','verify','--model',model],2,error)
        run('copy-preserves-existing',['models','copy','--model',model,'--output',model],2,error)
        assert model.is_dir()
        (model/'manifest.json').write_text('{"schema_version":"seedvr2-image-package-v1","profile":"seedvr2-3b-image-fp32-b-v1"}')
        run('self-described-model-not-authenticated',base,2,lambda d:'not reviewed' in d['error']['message'])
        if a.model:
            manifest=json.loads((a.model/'manifest.json').read_text())
            for row in manifest['graphs']:
                for key in ('param','weights'):
                    path=Path(row[key]['path']); (model/path).parent.mkdir(exist_ok=True,parents=True);os.link(a.model/path,model/path)
            for row in manifest['constants'].values():os.link(a.model/row['path'],model/row['path'])
            (model/'manifest.json').write_text(json.dumps(manifest))
            run('unicode-package-preflight',base,0,lambda d:d['model']['manifest_authenticated'] and not d['model']['weight_hashes_verified'])
            changed=copy.deepcopy(manifest);changed['exporter']['script_sha256']='1'*64
            (model/'manifest.json').write_text(json.dumps(changed))
            run('exporter-metadata-does-not-relabel-model-content',base,0)
            changed['model_verified']=True;(model/'manifest.json').write_text(json.dumps(changed))
            run('self-issued-model-certification-rejected',base,2,error)
            changed=copy.deepcopy(manifest);changed['graphs'][0]['weights']['sha256']='1'*64
            (model/'manifest.json').write_text(json.dumps(changed))
            run('self-consistent-new-weights-not-trusted',base,2,error)
            (model/'manifest.json').write_text(json.dumps(manifest))
            first=model/manifest['graphs'][0]['param']['path'];content=first.read_bytes();first.unlink();first.write_bytes(bytes([content[0]^1])+content[1:])
            run('actual-corrupted-graph-rejected',['models','verify','--model',model],2,lambda d:'hash mismatch' in d['error']['message'])
        assert not output.exists()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(schema_version='seedvr2-delivery-checks-v1',passed=all(r['passed'] for r in rows),
        cases=rows,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),model_verified=False),indent=2)+'\n')
    print(f'{len(rows)}/{len(rows)} native first-use checks passed')


if __name__=='__main__': main()
