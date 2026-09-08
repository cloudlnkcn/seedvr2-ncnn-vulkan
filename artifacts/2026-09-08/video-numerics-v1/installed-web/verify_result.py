# Recheck retained output only; never submits a new model job. Run from repository root.
import hashlib, json, subprocess, time, urllib.parse, urllib.request
from pathlib import Path
root=Path.cwd()
out=root/'artifacts/2026-09-08/video-numerics-v1/installed-web'
assert out.is_dir()
base='http://127.0.0.1:8877'
op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def raw(path,body=None,headers=None):
    if isinstance(body,dict): body=json.dumps(body).encode()
    with op.open(urllib.request.Request(base+path,data=body,headers=headers or {}),timeout=20) as r:
        return r.read()
def data(path,body=None,headers=None): return json.loads(raw(path,body,headers))
def save(name,value): (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p): return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
expected=json.loads((root/'.cache/video17-final-vulkan-v1/run.json').read_text())
before=[{'id':j} for j in json.loads((out/'previous-job-ids.json').read_text())]
job=json.loads((out/'job.json').read_text())
run=json.loads((out/'run.json').read_text())
movie=(out/'output.mp4').read_bytes()
events=json.loads((out/'events.json').read_text())
probe=json.loads((out/'ffprobe.json').read_text())
checks={
 'successful_native_job':run['status']=='SUCCEEDED',
 'all_36_graphs_vulkan':len(run['stages'])==36 and all(s['cpu_layers']==0 and s['backend']=='ncnn-vulkan' for s in run['stages']),
 'installed_sdk_identity':run['implementation']['sdk_library_sha256']==sha(root/'dist/seedvr2-0.6.0/lib64/libseedvr2.so')=='dcddfd914118ffcfabb6b6e37cbc7fd423320aedad59c9abe9a533d93ad11992',
 'native_worker_identity':run['implementation']['executable_sha256']==sha(root/'dist/seedvr2-0.6.0/bin/seedvr2-worker'),
 'runtime_pin':run['ncnn_commit']==expected['ncnn_commit'],
 'input_sha256':run['input']['sha256']==expected['input']['sha256'],
 'same_model_package':run['model_manifest_sha256']==expected['model_manifest_sha256'],
 'full_temporal_clip':run['clip']['decoded_frames']==17 and run['clip']['padded_frames']==17 and run['clip']['latent_frames']==5,
 'output_identity_matches_cli':hashlib.sha256(movie).hexdigest()==run['output']['sha256']==expected['output']['sha256'],
 'output_128x128_17_frames':run['output']['frames']==17 and run['output']['width']==128 and run['output']['height']==128,
 'decoded_17_h264_frames_no_audio':len(probe['streams'])==1 and int(probe['streams'][0]['nb_read_frames'])==17 and probe['streams'][0]['codec_name']=='h264',
 'ordered_original_timestamps':[int(f['pts']) for f in probe['frames']]==run['output']['timestamps_90khz'],
 'durable_events':events['items'][-1]['type']=='finished' and [e['sequence'] for e in events['items']]==list(range(1,events['sequence']+1)),
 'historical_jobs_retained':set(j['id'] for j in before)<=set(j['id'] for j in data('/api/v1/jobs')['items']),
 'no_model_certificate':run['model_verified'] is False and data('/api/v1/models/status')['certificate'] is None,
}
save('verification.json',dict(schema_version='seedvr2-installed-web-check-v1',scope='Installed native Web worker, actual retained 17-frame video; functional and CLI output identity check; numerical tensor gate is recorded separately',model_verified=False,job_id=job['id'],url=base+'/?v=numerics-20260908#/compare?job='+job['id'],passed=all(checks.values()),checks=checks,implementation=run['implementation'],previous_job_count=len(before),output_sha256=run['output']['sha256'],total_ms=run['total_ms']))
print(json.dumps(checks,indent=2),flush=True)
assert all(checks.values())
