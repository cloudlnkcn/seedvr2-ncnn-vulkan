#!/usr/bin/env python3
"""Render decoded FP16-storage/FP32/target comparisons from verified drift reports."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from pipeline_contract import load_json,sha,tensor_path
from measure_image_quality import metrics
from measure_video_quality import pixels,temporal_errors,media_contract


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','fp32-run','drift','fixtures','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--case',required=True)
    a=p.parse_args(); drift=load_json(a.drift);run=load_json(a.run/'run.json');base=load_json(a.fp32_run/'run.json')
    if not drift.get('measurement_complete') or drift['boundaries']!=73 or drift['candidate_run_sha256']!=sha(a.run/'run.json') or drift['fp32_run_sha256']!=sha(a.fp32_run/'run.json'):
        raise ValueError('Drift report binding differs')
    video=drift['kind']=='video';cut=None
    def read(root,row):return np.fromfile(tensor_path(root,row),dtype='<f4').reshape(row['shape'])
    if video:
        fixture=load_json(a.fixtures/'manifest.json');case=next(r for r in fixture['cases'] if r['case_id']==a.case)
        if run['input']['sha256']!=case['input']['sha256'] or sha(a.fixtures/case['target']['path'])!=case['target']['sha256']:
            raise ValueError('Fixture changed')
        target=np.fromfile(a.fixtures/case['target']['path'],dtype=np.uint8).reshape(case['target']['shape'])
        count=case['frames'];cut=case['artificial_cut_before_frame']
        def decode(v):return pixels(v,count)
        media=media_contract(a.run,run)
    else:
        fixture=load_json(a.fixtures/'provenance.json')
        for entry in fixture['files']:
            if entry['path'] in ('astronaut-target.png','astronaut-degraded.jpg') and sha(a.fixtures/entry['path'])!=entry['sha256']:
                raise ValueError('Image fixture identity differs')
        if run['input']['sha256']!=next(x['sha256'] for x in fixture['files'] if x['path']=='astronaut-degraded.jpg'):
            raise ValueError('Image input differs')
        target=np.array(Image.open(a.fixtures/'astronaut-target.png').convert('RGB'))[None];count=1;media=None
        def decode(v):return np.floor((np.clip(v,-1,1)*np.float32(.5)+np.float32(.5)).astype(np.float64)*255+.5).clip(0,255).astype(np.uint8).transpose(1,2,0)[None]
        # Compare encoded PNG directly for the image use case.
    if video:
        pictures={'FP32':decode(read(a.fp32_run,base['diagnostics']['decoded'])),'DiT FP16':decode(read(a.run,run['diagnostics']['decoded'])),'Target':target}
    else:
        pictures={'FP32':np.array(Image.open(a.fp32_run/'output.png').convert('RGB'))[None],'DiT FP16':np.array(Image.open(a.run/'output.png').convert('RGB'))[None],'Target':target}
    if any(v.shape!=target.shape for v in pictures.values()):raise ValueError('Output/target geometry differs')
    scores={k:[metrics(frame,target[i]) for i,frame in enumerate(v)] for k,v in pictures.items() if k!='Target'}
    summary={k:dict(mean_psnr_db=float(np.mean([x['psnr_db'] for x in rows])),mean_ssim=float(np.mean([x['ssim_rgb'] for x in rows]))) for k,rows in scores.items()}
    temporal={k:temporal_errors(v,target,cut) for k,v in pictures.items() if k!='Target'} if video else None
    report=dict(schema_version='storage-task-quality-v1',sample=a.case,scope='Development sample; target is fixed, not a representative benchmark.',target_quality=summary,per_frame=scores,temporal=temporal,media=media,drift_sha256=sha(a.drift),script_sha256=sha(Path(__file__)),fixture_sha256=sha(a.fixtures/('manifest.json' if video else 'provenance.json')),model_verified=False)
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    indices=sorted({0,count//2,count-1}|({cut-1,cut} if cut is not None else set()))
    height,width=target.shape[1:3];panel=Image.new('RGB',(width*3,len(indices)*(height+24)),'white');draw=ImageDraw.Draw(panel)
    for row,index in enumerate(indices):
        for col,(label,values) in enumerate(pictures.items()):
            draw.text((col*width+3,row*(height+24)+4),f'{label} frame {index}',fill='black')
            panel.paste(Image.fromarray(values[index]),(col*width,row*(height+24)+24))
    panel.save(a.output/'comparison.png');print(json.dumps(summary))

if __name__=='__main__':main()
