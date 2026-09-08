#!/usr/bin/env python3
"""One controlled natural-image example; not a representative quality benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def gaussian(x):
    axis = np.arange(-5, 6, dtype=np.float64)
    kernel = np.exp(-axis**2/(2*1.5**2)); kernel /= kernel.sum()
    for dim in (0, 1):
        x = np.einsum('...k,k->...', np.lib.stride_tricks.sliding_window_view(x, 11, axis=dim), kernel)
    return x


def metrics(x, y):
    x, y = x.astype(np.float64), y.astype(np.float64)
    mse = np.mean((x-y)**2)
    ux, uy = gaussian(x), gaussian(y)
    vx, vy, cov = gaussian(x*x)-ux*ux, gaussian(y*y)-uy*uy, gaussian(x*y)-ux*uy
    score = ((2*ux*uy+2.55**2)*(2*cov+7.65**2))/((ux*ux+uy*uy+2.55**2)*(vx+vy+7.65**2))
    return dict(psnr_db=float(10*np.log10(255**2/mse)) if mse else None, identical=bool(mse==0), ssim_rgb=float(score.mean()))


def main():
    p = argparse.ArgumentParser()
    for name in ('fixture','run','reference','output'): p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    files={'baseline':a.run/'comparison-input.png','native':a.run/'output.png','official_fp32_b':a.reference/'reference.png','target':a.fixture/'astronaut-target.png'}
    images={name:Image.open(path).convert('RGB') for name,path in files.items()}
    if len({im.size for im in images.values()})!=1: raise ValueError('Image sizes differ')
    target=np.asarray(images['target'])
    report=dict(schema_version='seedvr2-controlled-photo-quality-v1',model_verified=False,
        sample_count=1,scope='Single NASA/scikit-image public-domain image with documented synthetic JPEG degradation. Not representative, not a holdout or an acceptance threshold.',
        protocol={'range':255,'color':'RGB equally averaged, no Y conversion','border_crop':0,'ssim':'11x11 Gaussian sigma=1.5, valid convolution, population covariance, K1=.01 K2=.03','baseline':'Native aligned bicubic-antialias input; same geometry as result'},
        files={name:{'path':str(path),'sha256':digest(path)} for name,path in files.items()},
        metrics={name:metrics(np.asarray(im),target) for name,im in images.items() if name!='target'},
        fixture_provenance_sha256=digest(a.fixture/'provenance.json'),native_run_sha256=digest(a.run/'run.json'),script_sha256=digest(Path(__file__)))
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    w,h=images['target'].size; panel=Image.new('RGB',(4*w+50,h+66),'#f1f3f5');draw=ImageDraw.Draw(panel)
    try: font=ImageFont.truetype('/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf',16)
    except OSError: font=ImageFont.load_default()
    for index,(name,label) in enumerate([('baseline','Degraded input x2'),('native','Native ncnn Vulkan'),('official_fp32_b','Official FP32-B'),('target','Controlled target')]):
        x=10+index*(w+10);draw.text((x,10),label,fill='#17202a',font=font);panel.paste(images[name],(x,38))
    panel.save(a.output/'comparison.png'); print(json.dumps(report['metrics'],indent=2))


if __name__=='__main__':main()
