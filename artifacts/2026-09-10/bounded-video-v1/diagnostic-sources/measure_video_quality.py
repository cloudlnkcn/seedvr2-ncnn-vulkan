#!/usr/bin/env python3
"""Describe a reviewed bounded clip's spatial and temporal errors, without a quality gate."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from measure_image_quality import metrics
from pipeline_contract import bind_candidate, bind_reference, load_json, sha, tensor_path


def pixels(values, frames):
    if values.ndim != 4 or values.shape[0] != 3 or not 1 <= frames <= values.shape[1]:
        raise ValueError('Expected C,T,H,W with enough real frames')
    if not np.isfinite(values).all():
        raise ValueError('Non-finite video tensor')
    # Match save_video's nonnegative std::lround and FP32 arithmetic.
    values = (np.clip(values.astype(np.float32), -1, 1)*np.float32(.5)+np.float32(.5))*np.float32(255)
    return np.floor(values.astype(np.float64)+.5).clip(0, 255).astype(np.uint8).transpose(1, 2, 3, 0)[:frames]


def temporal_errors(predicted, target, cut_before=None):
    if (predicted.shape != target.shape or predicted.ndim != 4 or predicted.shape[-1] != 3
            or any(size < 1 for size in predicted.shape)):
        raise ValueError('Expected equal F,H,W,RGB tensors')
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError('Non-finite quality input')
    if cut_before is not None and (type(cut_before) is not int or not 1 <= cut_before < len(target)):
        raise ValueError('Invalid artificial cut index')
    residual = predicted.astype(np.float64)-target.astype(np.float64)
    changes = np.mean(np.abs(np.diff(residual, axis=0)), axis=(1, 2, 3))
    rows = [dict(before_frame=i+1, artificial_cut=(i+1 == cut_before),
                 residual_change_mae=float(value)) for i, value in enumerate(changes)]
    continuous = [r['residual_change_mae'] for r in rows if not r['artificial_cut']]
    return dict(transitions=rows, continuous_transition_count=len(continuous),
                continuous_mean_mae=float(np.mean(continuous)) if continuous else None,
                cut_mae=float(changes[cut_before-1]) if cut_before is not None else None)


def media_contract(run_root, run):
    output = run['output']
    path = run_root/output['path']
    if output['path'] != 'output.mp4' or sha(path) != output['sha256']:
        raise ValueError('Output identity differs')
    data = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams',
        '-show_frames', '-show_entries',
        'stream=codec_type,width,height:frame=best_effort_timestamp_time',
        '-of', 'json', str(path)], text=True, timeout=30))
    if len(data['streams']) != 1 or data['streams'][0].get('codec_type') != 'video':
        raise ValueError('Expected one video stream and no audio')
    stream = data['streams'][0]
    times = [round(float(f['best_effort_timestamp_time'])*90000) for f in data['frames']]
    if (stream['width'], stream['height'], len(times)) != (
            output['width'], output['height'], output['frames']):
        raise ValueError('Encoded geometry or frame count differs')
    if times != output['timestamps_90khz'] or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Encoded timestamps differ')
    return dict(passed=True, frames=len(times), width=stream['width'], height=stream['height'],
                timestamps_90khz=times, audio=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('fixtures', 'run', 'reference', 'reference-run', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--case', required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Use a new output directory')
    fixture = load_json(a.fixtures/'manifest.json')
    rows = [r for r in fixture['cases'] if r['case_id'] == a.case]
    if len(rows) != 1:
        p.error('Expected one declared fixture case')
    case = rows[0]
    ref, baseline, expected, reference_digest, _ = bind_reference(a.reference, a.reference_run, 'video')
    run = load_json(a.run/'run.json')
    bind_candidate(run, baseline, 'video', expected)
    if run['input']['sha256'] != case['input']['sha256'] or (
            run['output']['frames'], run['output']['height'], run['output']['width']) != (
            case['frames'], case['height'], case['width']):
        raise ValueError('Run does not match the declared fixture')
    target_path = a.fixtures/case['target']['path']
    if sha(target_path) != case['target']['sha256']:
        raise ValueError('Target identity differs')
    target = np.fromfile(target_path, dtype=np.uint8).reshape(case['target']['shape'])

    def read(root, row):
        return np.fromfile(tensor_path(root, row), dtype='<f4').reshape(row['shape'])

    reference_decoded = next(r['reference'] for r in ref['stages'] if r['stage'] == 'decoded')
    images = dict(bicubic=pixels(read(a.run, run['diagnostics']['prepared']), case['frames']),
                  native=pixels(read(a.run, run['diagnostics']['decoded']), case['frames']),
                  official=pixels(read(a.reference, reference_decoded), case['frames']), target=target)
    if any(x.shape != target.shape for x in images.values()):
        raise ValueError('Image shape differs from the fixed target')
    per_frame = {name: [dict(frame=i, **metrics(frame, target[i])) for i, frame in enumerate(values)]
                 for name, values in images.items() if name != 'target'}
    aggregate = {name: dict(mean_frame_psnr_db=float(np.mean([r['psnr_db'] for r in values]))
                           if all(r['psnr_db'] is not None for r in values) else None,
                           mean_frame_ssim_rgb=float(np.mean([r['ssim_rgb'] for r in values])))
                 for name, values in per_frame.items()}
    delta = np.abs(images['native'].astype(np.int16)-images['official'].astype(np.int16))
    report = dict(schema_version='seedvr2-bounded-video-quality-v1', case_id=a.case,
        model_verified=False, quality_gate='NOT_DEFINED_DESCRIPTIVE_DEVELOPMENT_SAMPLE',
        scope=fixture['scope'], target_note=fixture['target_note'],
        protocol=dict(range=255, color='RGB equally averaged, no border crop',
            ssim='11x11 Gaussian sigma1.5, valid convolution, population covariance; K1=.01 K2=.03',
            temporal='MAE of consecutive changes in prediction-minus-target residuals; '
                     'NOT motion-compensated flicker, cut transition reported separately',
            pixels='Clamped/rounded pre-codec RGB8; padded output frames excluded'),
        fixture_manifest_sha256=sha(a.fixtures/'manifest.json'), native_run_sha256=sha(a.run/'run.json'),
        reference_report_sha256=reference_digest, media=media_contract(a.run, run),
        native_official=dict(max_abs=int(delta.max()), mae=float(delta.mean()),
            per_frame=[dict(frame=i, max_abs=int(d.max()), mae=float(d.mean())) for i, d in enumerate(delta)],
            temporal=temporal_errors(images['native'], images['official'], case['artificial_cut_before_frame'])),
        aggregate=aggregate, per_frame=per_frame,
        temporal={name: temporal_errors(values, target, case['artificial_cut_before_frame'])
                  for name, values in images.items() if name != 'target'},
        scripts={name:sha(Path(__file__).parent/name) for name in
                 ('measure_video_quality.py', 'measure_image_quality.py', 'pipeline_contract.py')})
    a.output.mkdir(parents=True)
    (a.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    indices = sorted(set([0, case['frames']//2, case['frames']-1] +
                        ([case['artificial_cut_before_frame']-1, case['artificial_cut_before_frame']]
                         if case['artificial_cut_before_frame'] is not None else [])))
    width, height = case['width'], case['height']
    panel = Image.new('RGB', (4*width, len(indices)*(height+24)), 'white')
    draw = ImageDraw.Draw(panel)
    for row, frame in enumerate(indices):
        for col, (name, values) in enumerate(images.items()):
            draw.text((col*width+3, row*(height+24)+4), f'{name} f{frame}', fill='black')
            panel.paste(Image.fromarray(values[frame]), (col*width, row*(height+24)+24))
    panel.save(a.output/'comparison.png')
    for name, values in images.items():
        Image.fromarray(values[0]).save(a.output/(name+'.png'))
        command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                   '-s', f'{width}x{height}', '-r', str(case['fps']), '-i', 'pipe:0', '-an',
                   '-c:v', 'libx264', '-crf', '18', '-threads', '1', '-pix_fmt', 'yuv420p',
                   '-movflags', '+faststart', str(a.output/(name+'.mp4'))]
        subprocess.run(command, input=values.tobytes(), capture_output=True, check=True, timeout=30)
    print(json.dumps(dict(aggregate=aggregate, native_official=report['native_official'],
                         media=report['media']), indent=2))


if __name__ == '__main__':
    main()
