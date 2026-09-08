#!/usr/bin/env python3
"""Exercise actual short-video workers, media seeking and durable recovery.

This is a functional integration test. Numerical model certification is separate.
"""
import argparse
import hashlib
import json
import queue
import sqlite3
import subprocess
import threading
import time
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    for name in ('server', 'model', 'video-model', 'input', 'image', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    db = (a.output / 'workspace.sqlite3').resolve()
    checks, process, index = [], None, 0
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def check(name, condition):
        checks.append(dict(case=name, passed=bool(condition)))
        print(name, 'PASS' if condition else 'FAIL', flush=True)
        if not condition:
            raise AssertionError(name)

    def start():
        nonlocal process, index
        index += 1
        with (a.output / f'server-{index}.stderr').open('w') as log:
            process = subprocess.Popen([str(a.server.resolve()), '--port', '0', '--database', str(db),
                '--model', str(a.model.resolve()), '--video-model', str(a.video_model.resolve())],
                stdout=subprocess.PIPE, stderr=log, text=True)
        first = queue.Queue()
        threading.Thread(target=lambda: first.put(process.stdout.readline()), daemon=True).start()
        return json.loads(first.get(timeout=15))['url'].rstrip('/')

    base = start()

    def call(path, body=None, headers=None):
        if isinstance(body, dict):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        try:
            response = opener.open(urllib.request.Request(base+path, data=body, headers=headers or {}), timeout=20)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, response.read(), response.headers

    def data(path, body=None, headers=None):
        code, raw, _ = call(path, body, headers)
        if code != 200:
            raise AssertionError((path, code, raw[:1000]))
        return json.loads(raw)

    def auth():
        return {'Content-Type': 'application/json', 'X-SeedVR2-Session': data('/api/v1/session')['session_token']}

    def wait_job(job, predicate, timeout=150):
        until = time.monotonic()+timeout
        while time.monotonic() < until:
            current = data('/api/v1/jobs/'+job['id'])
            if predicate(current):
                return current
            time.sleep(.25)
        raise AssertionError(('Job timeout', current))

    try:
        headers = auth()
        upload = {**headers, 'Content-Type': 'application/octet-stream', 'X-Media-Kind': 'video',
                  'X-File-Name': urllib.parse.quote('时序测试.mp4')}
        status = data('/api/v1/models/image')
        check('separate-video-package-ready', status['installed'] and status['video_model']['installed'] and status['video_model']['max_frames'] == 17)
        raw = a.input.read_bytes()
        check('video-upload-requires-session', call('/api/v1/media', raw, {'X-Media-Kind': 'video'})[0] == 403)
        check('malformed-video-rejected', call('/api/v1/media', b'not a movie', upload)[0] == 422)
        check('image-as-video-rejected', call('/api/v1/media', a.image.read_bytes(), upload)[0] == 422)
        for name, flags in [('hdr', ['-color_primaries', 'bt2020', '-color_trc', 'smpte2084', '-colorspace', 'bt2020nc']),
                            ('nonsquare', ['-vf', 'setsar=2/1'])]:
            fixture = a.output / (name + '.mp4')
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(a.input), '-frames:v', '1',
                '-c:v', 'libx264', *flags, str(fixture)], check=True, timeout=20)
            check(name+'-rejected', call('/api/v1/media', fixture.read_bytes(), upload)[0] == 422)
        media = data('/api/v1/media', raw, upload)
        check('actual-video-import', media['kind'] == 'video' and media['name'] == '时序测试.mp4' and media['sha256'] == digest(a.input))
        path = '/api/v1/media/'+media['id']
        code, body, h = call(path)
        check('original-video-serving', code == 200 and body == raw and h.get('Content-Type', '').startswith('video/mp4'))
        check('video-csp-and-seeking', "media-src 'self'" in h.get('Content-Security-Policy', '') and h.get('Accept-Ranges') == 'bytes')
        for name, value, expected in [('start', 'bytes=0-63', raw[:64]), ('suffix', 'bytes=-64', raw[-64:]),
                                      ('end', f'bytes={len(raw)-64}-', raw[-64:]),
                                      ('clamped', f'bytes={len(raw)-64}-{len(raw)+100}', raw[-64:])]:
            code, body, h = call(path, headers={'Range': value})
            check('range-'+name, code == 206 and body == expected and h.get('Content-Range', '').endswith('/'+str(len(raw))))
        for name, value in [('past-end', f'bytes={len(raw)}-'), ('multi', 'bytes=0-1,3-4'), ('negative', 'bytes=-0'),
                            ('reverse', 'bytes=3-2'), ('overflow', 'bytes=18446744073709551616-'), ('invalid', 'bytes=x-5')]:
            code, _, h = call(path, headers={'Range': value})
            check('range-reject-'+name, code == 416 and h.get('Content-Range') == 'bytes */'+str(len(raw)))
        image = data('/api/v1/media', a.image.read_bytes(), {**headers, 'X-File-Name': 'image.png'})
        request = dict(schema_version='seedvr2-video-job-v1', media_id=media['id'], size=64, backend='vulkan', gpu=0, seed=666, max_frames=6)
        for key, value in [('max_frames', 0), ('max_frames', 18), ('max_frames', True), ('max_frames', 6.0),
                           ('max_frames', 5.5), ('size', 144), ('size', 63), ('size', 65), ('backend', 'cuda'), ('media_id', image['id'])]:
            check('reject-'+key+'-'+str(value), call('/api/v1/jobs', {**request, key: value}, headers)[0] == 422)
        check('reject-unknown-option', call('/api/v1/jobs', {**request, 'audio': True}, headers)[0] == 422)
        check('reject-duplicate-option', call('/api/v1/jobs', json.dumps(request)[:-1]+',"max_frames":6}', headers)[0] == 422)
        check('reject-missing-frames', call('/api/v1/jobs', {k: v for k, v in request.items() if k != 'max_frames'}, headers)[0] == 422)
        check('video-cannot-use-image-pipeline', call('/api/v1/jobs', {**{k: v for k, v in request.items() if k != 'max_frames'}, 'schema_version': 'seedvr2-image-job-v1'}, headers)[0] == 422)
        cancelled = data('/api/v1/jobs', request, headers)
        wait_job(cancelled, lambda j: j['status'] == 'RUNNING')
        data('/api/v1/jobs/'+cancelled['id']+'/cancel', {}, headers)
        stopped = wait_job(cancelled, lambda j: j['status'] == 'CANCELLED', 25)
        check('cancel-real-video-worker', stopped['result'] is None)
        check('cancelled-video-not-downloadable', call('/api/v1/jobs/'+cancelled['id']+'/files/output.mp4')[0] == 404)
        good = data('/api/v1/jobs', request, headers)
        complete = wait_job(good, lambda j: j['status'] in ['SUCCEEDED', 'FAILED'])
        check('real-short-video-completes', complete['status'] == 'SUCCEEDED')
        run = complete['result']
        check('temporal-padding-and-crop', run['clip']['decoded_frames'] == 6 and run['clip']['padded_frames'] == 9 and run['clip']['latent_frames'] == 3 and run['output']['frames'] == 6 and run['clip']['truncated'])
        check('all-36-graphs-vulkan', len(run['stages']) == 36 and all(x['cpu_layers'] == 0 for x in run['stages']))
        check('no-model-certificate', run['model_verified'] is False and data('/api/v1/models/status')['certificate'] is None)
        result_path = '/api/v1/jobs/'+good['id']+'/files/output.mp4'
        code, movie, h = call(result_path+'?download=1')
        check('download-completed-mp4', code == 200 and hashlib.sha256(movie).hexdigest() == run['output']['sha256'] and 'attachment' in h.get('Content-Disposition', ''))
        output = a.output/'downloaded.mp4'
        output.write_bytes(movie)
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames', '-show_streams', '-show_frames', '-of', 'json', str(output)]))
        check('mp4-frame-count-and-no-audio', len(probe['streams']) == 1 and int(probe['streams'][0]['nb_read_frames']) == 6 and probe['streams'][0]['codec_name'] == 'h264')
        times = [int(f['pts']) for f in probe['frames']]
        check('decoded-mp4-timestamps', times == run['output']['timestamps_90khz'] and all(b > a for a, b in zip(times, times[1:])))
        code, piece, _ = call(result_path, headers={'Range': 'bytes=-128'})
        check('result-video-seek-range', code == 206 and piece == movie[-128:])
        check('comparison-video-available', call('/api/v1/jobs/'+good['id']+'/files/comparison-input.mp4')[0] == 200)
        events = data('/api/v1/jobs/'+good['id']+'/events?after=0')
        check('video-events-durable-ordered', [e['sequence'] for e in events['items']] == list(range(1, events['sequence']+1)) and events['items'][-1]['type'] == 'finished')
        crash = data('/api/v1/jobs', request, headers)
        wait_job(crash, lambda j: j['status'] == 'RUNNING' and j['progress']['stage'] == 'validating')
        process.kill(); process.wait(timeout=15)
        base = start()
        check('video-restart-interruption', data('/api/v1/jobs/'+crash['id'])['status'] == 'INTERRUPTED')
        check('video-history-survives-restart', data('/api/v1/jobs/'+good['id'])['status'] == 'SUCCEEDED' and call(result_path)[1] == movie)
        check('video-media-survives-restart', call(path)[1] == raw)
        with sqlite3.connect(db) as connection:
            check('schema3-remains-compatible', connection.execute('PRAGMA user_version').fetchone()[0] == 3 and connection.execute('PRAGMA integrity_check').fetchone()[0] == 'ok')
        (a.output/'actual-job.json').write_text(json.dumps(complete, indent=2)+'\n')
        (a.output/'mp4-probe.json').write_text(json.dumps(probe, indent=2)+'\n')
    finally:
        if process and process.poll() is None:
            process.terminate(); process.wait(timeout=20)
        report = dict(passed=bool(checks) and all(x['passed'] for x in checks), checks=checks,
            server_sha256=digest(a.server), worker_sha256=digest(a.server.parent/'seedvr2-worker'), script_sha256=digest(Path(__file__)),
            scope='Actual native video worker and HTTP lifecycle; not a numerical model certificate')
        (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('PASS', len(checks))


if __name__ == '__main__':
    main()
