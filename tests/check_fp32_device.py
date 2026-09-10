#!/usr/bin/env python3
"""Check the public preflight response to an observed arithmetic capability."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('binary',type=Path)
p.add_argument('input',type=Path)
a=p.parse_args()
binary=a.binary.resolve()
devices=subprocess.run([str(binary),'engine','devices'],capture_output=True,text=True,check=True)
report=json.loads(devices.stdout)
if not report['devices']:
    print('SKIP: no Vulkan device')
    raise SystemExit(77)
probe=report['devices'][0]['fp32_b_arithmetic']
assert probe['protocol']=='fp32-fma-residual-v1' and probe['checked']==8
assert len(probe['values'])==8
assert probe['failed']==sum(not row['matched'] for row in probe['values'])
assert probe['supported']==(probe['failed']==0)
with tempfile.TemporaryDirectory(prefix='seedvr2-fp32-device-') as folder:
    folder=Path(folder)
    run=subprocess.run([str(binary),'run','--check','--backend','vulkan','--gpu','0',
        '--model',str(folder/'missing-model'),'--input',str(a.input.resolve()),
        '--output',str(folder/'未创建 output'),'--size','128'],capture_output=True,text=True)
    assert run.returncode==2,(run.returncode,run.stdout,run.stderr)
    error=json.loads(run.stdout)['error']
    if probe['supported']:
        assert error['field']=='model',error
    else:
        assert error['field']=='device' and 'fused multiply-add residuals' in error['message'],error
    assert not (folder/'未创建 output').exists()
print(json.dumps(dict(passed=True,device=report['devices'][0]['name'],arithmetic=probe,
    preflight_error=error,scope='Capability and preflight contract; no model weights loaded'),indent=2))
