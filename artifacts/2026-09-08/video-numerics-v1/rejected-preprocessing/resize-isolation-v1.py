import numpy as np, torch
from torch.nn.functional import interpolate
from pathlib import Path
f=np.float32
def resize(x,n,axis,fused):
 x=np.moveaxis(x,axis,0);src=len(x);scale=f(src/n);supp=2*max(scale,1);inv=f(1/max(scale,1));rows=[]
 for i in range(n):
  center=f((i+.5)*scale);first=max(int(center-supp+.5),0);end=min(int(center+supp+.5),src)
  ws=[]
  for j in range(first,end):
   a=f(abs(f(f(j)-center+f(.5))*inv))
   w=f(f(f(f(1.5)*a-f(2.5))*a)*a+f(1)) if a<1 else f(f(f(f(f(-.5)*a+f(2.5))*a-f(4))*a)+f(2)) if a<2 else f(0)
   ws.append(w)
  s=f(0)
  for w in ws:s=f(s+w)
  ws=[f(w/s) for w in ws];acc=np.zeros_like(x[0])
  for j,w in enumerate(ws):
   v=x[first+j]
   acc=(v.astype('float64')*float(w)+acc.astype('float64')).astype('float32') if fused else (v*w+acc).astype('float32')
  rows.append(acc)
 return np.moveaxis(np.stack(rows),0,axis)
def metrics(a,b):
 d=np.abs(a.astype('float64')-b);return float(d.max()),int(np.count_nonzero(a!=b)),int((d>2e-7).sum())
for src,dst in [(64,128),(128,64)]:
 p=np.fromfile(f'tests/fixtures/preprocessing/{src}-to-{dst}.rgb8',dtype='uint8').reshape(src,src,3)
 t=torch.from_numpy(p).permute(2,0,1).float()[None]/255.
 npixels=p.transpose(2,0,1).astype('float32')/f(255)
 print('pixels',metrics(npixels,t[0].numpy()))
 h=interpolate(t,size=(src,dst),mode='bicubic',antialias=True,align_corners=False)[0].numpy()
 full=interpolate(t,size=(dst,dst),mode='bicubic',antialias=True,align_corners=False)[0].numpy()
 print(src,dst)
 for fh in [False,True]:
  nh=resize(npixels,dst,2,fh);print('h',fh,metrics(nh,h))
  for fv in [False,True]:
   nv=resize(nh,dst,1,fv);print('v',fv,metrics(nv,full),metrics((nv.clip(0,1)-f(.5))/f(.5),(full.clip(0,1)-f(.5))/f(.5)))
