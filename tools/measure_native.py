#!/usr/bin/env python3
"""Measure one immutable native invocation; metrics are observations, not gates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--timeout',type=float,default=300)
    p.add_argument('--sample-gpu',action='store_true')
    p.add_argument('command',nargs=argparse.REMAINDER)
    args=p.parse_args();command=args.command
    if command and command[0]=='--':command=command[1:]
    if not command or args.report.exists():p.error('Command and a new report path are required')
    args.report.parent.mkdir(parents=True,exist_ok=True)
    samples=[];start=time.monotonic();timed_out=False
    def gpu():
        if not args.sample_gpu or not shutil.which('nvidia-smi'): return None
        try:
            raw=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],text=True,timeout=5)
            used,util,temp=map(int,raw.strip().split(','));return dict(whole_card_used_mib=used,utilization_percent=util,temperature_c=temp)
        except (subprocess.SubprocessError,ValueError):return None
    idle=gpu()
    with args.report.with_suffix('.stdout').open('w') as out,args.report.with_suffix('.stderr').open('w') as err:
        proc=subprocess.Popen(command,stdout=out,stderr=err)
        while proc.poll() is None:
            row=dict(elapsed_seconds=time.monotonic()-start)
            try:
                for line in Path(f'/proc/{proc.pid}/status').read_text().splitlines():
                    key,_,value=line.partition(':')
                    if key in ('VmRSS','VmHWM','RssAnon','RssFile','VmSwap'):row[key+'_bytes']=int(value.strip().split()[0])*1024
            except (OSError,ValueError):pass
            observed=gpu()
            if observed is not None:row.update(observed)
            samples.append(row)
            if time.monotonic()-start>args.timeout:
                proc.terminate();timed_out=True
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                break
            time.sleep(.5)
    report=dict(schema_version='seedvr2-native-measurement-v1',command=command,exit_code=proc.returncode,timed_out=timed_out,
        wall_seconds=time.monotonic()-start,idle_gpu=idle,sample_interval_seconds=.5,
        measurement_scope='Fresh child process; /proc RSS includes anonymous and mapped file pages. NVIDIA values cover the entire card including the desktop, not exclusive process VRAM. File cache outside RSS is not measured. No cache flushing.',
        peaks={key:max(row[key] for row in samples if key in row) for key in
            ('VmRSS_bytes','VmHWM_bytes','RssAnon_bytes','RssFile_bytes','VmSwap_bytes','whole_card_used_mib','temperature_c') if any(key in row for row in samples)},samples=samples)
    binary=Path(command[0]);report['executable_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
    text=args.report.with_suffix('.stdout').read_text()
    try:
        result=json.loads(text);report['native_status']=result.get('status');report['implementation']=result.get('implementation')
        report['native_total_ms']=result.get('total_ms');report['timing']=result.get('timing')
    except json.JSONDecodeError:report['native_json_parse_failed']=True
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('exit_code','wall_seconds','peaks','timed_out')}),flush=True)
    raise SystemExit(proc.returncode or int(timed_out))


if __name__=='__main__':main()
