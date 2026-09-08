#!/usr/bin/env python3
"""Run an installed SDK in a new network namespace with the source tree hidden."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--install',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--with-model',action='store_true')
    a=p.parse_args();root=Path(__file__).resolve().parents[1];prefix=a.install.resolve();work=a.output.resolve()
    if not shutil.which('bwrap'):raise SystemExit('bwrap is required; offline isolation was not tested')
    work.mkdir(parents=True,exist_ok=False)
    for name in ['home','cache']: (work/name).mkdir()
    shutil.copytree(root/'examples/sdk',work/'consumer')
    shutil.copy2(root/'tests/fixtures/natural/astronaut-degraded.jpg',work/'输入 照片.jpg')
    mount='/opt/SeedVR2 空间'
    script='''#!/bin/sh
set -eu
test ! -e {source}
test "$(wc -l < /proc/net/route)" -eq 1
readlink /proc/self/ns/net > /tmp/work/network-namespace.txt
cat /proc/net/route > /tmp/work/network-routes.txt
{app}/bin/seedvr2 engine self-test --backend cpu > /tmp/work/cpu.json
cmake -S /tmp/work/consumer -B /tmp/work/build -G Ninja -DCMAKE_PREFIX_PATH={app} > /tmp/work/sdk-build.log 2>&1
cmake --build /tmp/work/build >> /tmp/work/sdk-build.log 2>&1
/tmp/work/build/seedvr2-sdk-example > /tmp/work/sdk-smoke.json
'''.format(source=shlex.quote(str(root/'src/engine/ncnn/image.cpp')),app=shlex.quote(mount))
    if a.with_model:
        script+='/tmp/work/build/seedvr2-sdk-example image '+shlex.quote(mount+'/models/image')+' '+shlex.quote('/tmp/work/输入 照片.jpg')+' '+shlex.quote('/tmp/work/结果 自然图像')+' vulkan 256 > /tmp/work/image.json\n'
    (work/'check.sh').write_text(script)
    command=['bwrap','--die-with-parent','--unshare-net','--ro-bind','/','/',
        '--tmpfs','/opt','--tmpfs','/tmp','--ro-bind',str(prefix),mount,'--bind',str(work),'/tmp/work','--dev-bind','/dev','/dev','--proc','/proc',
        '--tmpfs',str(root),'--setenv','HOME','/tmp/work/home','--setenv','XDG_CACHE_HOME','/tmp/work/cache',
        '--setenv','PATH','/usr/bin:/bin','--unsetenv','LD_LIBRARY_PATH','--setenv','VK_INSTANCE_LAYERS','VK_LAYER_KHRONOS_validation',
        '--chdir','/tmp/work','/bin/sh','/tmp/work/check.sh']
    started=time.monotonic()
    with (work/'stdout').open('w') as out,(work/'stderr').open('w') as err:
        result=subprocess.run(command,stdout=out,stderr=err,timeout=300)
    child=(work/'network-namespace.txt').read_text().strip() if (work/'network-namespace.txt').exists() else None
    parent=os.readlink('/proc/self/ns/net');errors=[line for line in (work/'stderr').read_text().splitlines() if any(x in line.lower() for x in ('vuid-','validation error'))]
    success=result.returncode==0 and child is not None and child!=parent and not errors
    report=dict(schema_version='seedvr2-offline-delivery-v1',passed=success,exit_code=result.returncode,
        source_tree_hidden=str(root),runtime_mount=mount,parent_network_namespace=parent,child_network_namespace=child,
        unicode_paths_requested=True,ld_library_path_unset_requested=True,network_isolation_established=child is not None and child!=parent,system_abi_scope='Same installed Linux system libraries, source/model originals inaccessible in child namespace',
        external_sdk_built_from_installed_package=True if result.returncode==0 else None,
        full_image_run_requested=a.with_model,full_image_succeeded=a.with_model and success,wall_seconds=time.monotonic()-started,validation_errors=errors,
        launcher_sha256=hashlib.sha256((work/'check.sh').read_bytes()).hexdigest())
    (work/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    raise SystemExit(0 if success else 1)


if __name__=='__main__':main()
