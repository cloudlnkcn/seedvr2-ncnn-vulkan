#!/usr/bin/env python3
"""Real worker/HTTP lifecycle tests. Queue checks are distinct from model parity."""
import argparse,hashlib,json,os,queue,sqlite3,subprocess,tempfile,threading,time
from pathlib import Path
import urllib.request,urllib.error,urllib.parse

def main():
    p=argparse.ArgumentParser();p.add_argument('--server',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.server=a.server.resolve();a.model=a.model.resolve();a.output.mkdir(parents=True,exist_ok=False)
    db=a.output/'workspace.sqlite3';checks=[];process=None;run_index=0
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def check(name,ok):
        checks.append(dict(case=name,passed=bool(ok)));print(name,'PASS' if ok else 'FAIL',flush=True)
        if not ok:raise AssertionError(name)
    def start():
        nonlocal process,run_index
        run_index+=1
        log=(a.output/f'server-{run_index}.stderr').open('w')
        process=subprocess.Popen([str(a.server),'--port','0','--database',str(db),'--model',str(a.model)],stdout=subprocess.PIPE,stderr=log,text=True)
        line=queue.Queue();threading.Thread(target=lambda:line.put(process.stdout.readline()),daemon=True).start()
        first=line.get(timeout=15);log.close()
        return json.loads(first)['url'].rstrip('/')
    base=start()
    def call(path,body=None,headers=None):
        if isinstance(body,dict):body=json.dumps(body).encode()
        elif isinstance(body,str):body=body.encode()
        try:r=opener.open(urllib.request.Request(base+path,data=body,headers=headers or {}),timeout=15)
        except urllib.error.HTTPError as e:r=e
        with r:return r.status,r.read(),r.headers
    def data(path,body=None,headers=None):
        code,raw,_=call(path,body,headers)
        if code!=200:raise AssertionError((path,code,raw))
        return json.loads(raw)
    def auth():return {'Content-Type':'application/json','X-SeedVR2-Session':data('/api/v1/session')['session_token']}
    def wait_job(id,predicate,timeout=100):
        until=time.monotonic()+timeout
        while time.monotonic()<until:
            j=data('/api/v1/jobs/'+id)
            if predicate(j):return j
            time.sleep(.3)
        raise AssertionError(('job timeout',j))
    try:
        headers=auth()
        check('image-package-available',data('/api/v1/models/image')['installed'])
        raw=a.input.read_bytes()
        check('upload-requires-session',call('/api/v1/media',raw,{'X-File-Name':'image.png'})[0]==403)
        check('reject-invalid-image',call('/api/v1/media',b'invalid',{**headers,'X-File-Name':'broken.png'})[0]==422)
        media=data('/api/v1/media',raw,{**headers,'Content-Type':'application/octet-stream','X-File-Name':urllib.parse.quote('中文 测试.png')})
        check('actual-image-import',media['sha256']==hashlib.sha256(raw).hexdigest() and media['name']=='中文 测试.png')
        check('actual-image-serving',call('/api/v1/media/'+media['id'])[1]==raw)
        request=dict(schema_version='seedvr2-image-job-v1',media_id=media['id'],size=128,backend='vulkan',gpu=0,seed=666)
        for key,val in [('size',129),('size',True),('size',1024),('size',128.5),('seed',-1),('seed',4294967296),('backend','cuda'),('gpu',65),('media_id','a'*31)]:
            check(f'reject-{key}-{val}',call('/api/v1/jobs',{**request,key:val},headers)[0]==422)
        check('reject-unknown-field',call('/api/v1/jobs',{**request,'path':'/tmp/other'},headers)[0]==422)
        duplicate=json.dumps(request)[:-1]+',"size":128}'
        check('reject-duplicate-field',call('/api/v1/jobs',duplicate,headers)[0]==422)
        check('reject-nested-json',call('/api/v1/jobs',{**request,'seed':{'a':1}},headers)[0]==422)
        check('job-submit-requires-session',call('/api/v1/jobs',request,{'Content-Type':'application/json'})[0]==403)
        check('media-path-traversal',call('/api/v1/media/'+urllib.parse.quote('../workspace.sqlite3',safe=''))[0] in (404,422))
        first=data('/api/v1/jobs',request,headers)
        wait_job(first['id'],lambda j:j['status']=='RUNNING')
        queued=data('/api/v1/jobs',request,headers)
        check('second-job-waits',queued['status']=='QUEUED')
        extra=[data('/api/v1/jobs',request,headers) for _ in range(6)]
        check('queue-capacity-is-bounded',call('/api/v1/jobs',request,headers)[0]==422)
        for item in extra:data('/api/v1/jobs/'+item['id']+'/cancel',{},headers)
        cancelled=data('/api/v1/jobs/'+queued['id']+'/cancel',{},headers)
        check('cancel-queued',cancelled['status']=='CANCELLED')
        check('cancel-idempotent',data('/api/v1/jobs/'+queued['id']+'/cancel',{},headers)['sequence']==cancelled['sequence'])
        data('/api/v1/jobs/'+first['id']+'/cancel',{},headers)
        stopped=wait_job(first['id'],lambda j:j['status']=='CANCELLED',20)
        check('cancel-running-worker',stopped['result'] is None)
        check('no-cancelled-result',call('/api/v1/jobs/'+first['id']+'/files/output.png')[0]==404)
        bad=data('/api/v1/jobs',{**request,'gpu':64},headers)
        failed=wait_job(bad['id'],lambda j:j['status'] in ['FAILED','SUCCEEDED'],60)
        check('invalid-device-fails-without-fallback',failed['status']=='FAILED' and failed['error'] is not None and failed['result'] is None)
        check('failed-job-has-no-download',call('/api/v1/jobs/'+bad['id']+'/files/output.png')[0]==404)
        good=data('/api/v1/jobs',request,headers)
        complete=wait_job(good['id'],lambda j:j['status'] in ['SUCCEEDED','FAILED'],120)
        check('complete-real-worker-image',complete['status']=='SUCCEEDED')
        check('actual-36-graph-trajectory',len(complete['result']['stages'])==36 and all(x['cpu_layers']==0 for x in complete['result']['stages']))
        check('model-not-auto-certified',complete['result']['model_verified'] is False and data('/api/v1/models/status')['certificate'] is None)
        code,png,h=call('/api/v1/jobs/'+good['id']+'/files/output.png?download=1')
        check('download-actual-png',code==200 and png.startswith(b'\x89PNG') and hashlib.sha256(png).hexdigest()==complete['result']['output']['sha256'] and 'attachment' in h.get('Content-Disposition',''))
        check('result-path-allowlist',call('/api/v1/jobs/'+good['id']+'/files/worker.log')[0]==404)
        events=data('/api/v1/jobs/'+good['id']+'/events?after=0')
        seq=[e['sequence'] for e in events['items']]
        check('durable-ordered-events',seq==list(range(1,events['sequence']+1)) and events['items'][-1]['type']=='finished')
        check('event-cursor-resume',len(data('/api/v1/jobs/'+good['id']+'/events?after='+str(seq[-2]))['items'])==1)
        check('reject-event-cursor',call('/api/v1/jobs/'+good['id']+'/events?after=-1')[0]==422)
        crash=data('/api/v1/jobs',request,headers)
        wait_job(crash['id'],lambda j:j['status']=='RUNNING' and j['progress']['stage']=='validating')
        process.kill();process.wait(timeout=15);base=start();headers=auth()
        recovered=data('/api/v1/jobs/'+crash['id'])
        check('restart-marks-interrupted',recovered['status']=='INTERRUPTED' and recovered['result'] is None)
        check('completed-results-survive-restart',data('/api/v1/jobs/'+good['id'])['status']=='SUCCEEDED')
        check('media-survives-restart',call('/api/v1/media/'+media['id'])[1]==raw)
        with sqlite3.connect(db) as connection:
            check('sqlite-schema3-integrity',connection.execute('PRAGMA user_version').fetchone()[0]==3 and connection.execute('PRAGMA integrity_check').fetchone()[0]=='ok')
        (a.output/'actual-job.json').write_text(json.dumps(complete,indent=2)+'\n')
    finally:
        if process and process.poll() is None:process.terminate();process.wait(timeout=20)
        report=dict(passed=bool(checks) and all(x['passed'] for x in checks),checks=checks,scope='Real native worker and durable local HTTP jobs. Lifecycle tests are not numerical model evidence.')
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS',len(checks),flush=True)
if __name__=='__main__':main()
