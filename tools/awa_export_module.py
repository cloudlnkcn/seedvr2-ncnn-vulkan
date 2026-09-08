"""Scriptable dynamic AWA boundary, checked separately against the official body.

The pnnx moduleop pass preserves this module. Its serialized attributes are then
lowered to SeedVR2AWA; no traced window indices are copied into the ncnn graph.
"""
import math
from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F


class AdaptiveWindowAttention(nn.Module):
    def __init__(self,heads:int,shifted:bool,eps:float=1e-5):
        super().__init__()
        self.register_buffer('spec',torch.tensor([1,heads,int(shifted)],dtype=torch.int32))
        self.register_buffer('epsilon',torch.tensor([eps],dtype=torch.float32))
        self.norm_weights=nn.Parameter(torch.ones(4,128))
        self.register_buffer('rope_frequencies',1./(10000.**(torch.arange(0,42,2).float()/42.)))

    def rotate(self,x:torch.Tensor,position:torch.Tensor,frequencies:torch.Tensor)->torch.Tensor:
        # Interleaved adjacent pairs in three 42-dimensional axes, then 2 untouched channels.
        frequencies=(position.float().unsqueeze(-1)*frequencies).flatten(1)
        c=frequencies.cos().unsqueeze(1)
        s=frequencies.sin().unsqueeze(1)
        pairs=x[:,:,:126].reshape(x.shape[0],x.shape[1],63,2)
        a=pairs[:,:,:,0];b=pairs[:,:,:,1]
        rotated=torch.stack((a*c-b*s,a*s+b*c),dim=-1).flatten(2)
        return torch.cat((rotated,x[:,:,126:]),dim=-1)

    def forward(self,video:torch.Tensor,text:torch.Tensor)->Tuple[torch.Tensor,torch.Tensor]:
        spec=self.spec
        norm_weights=self.norm_weights
        frequencies=self.rope_frequencies
        eps=float(self.epsilon[0].item())
        heads=int(spec[1].item())
        shifted=bool(spec[2].item())
        assert int(spec[0].item())==1
        t,h,w,width=video.shape
        text_length=text.shape[0]
        hd=heads*128
        scale=math.sqrt(3600./float(h*w))
        wh=int(math.ceil(float(round(float(h)*scale))/3.))
        ww=int(math.ceil(float(round(float(w)*scale))/3.))
        wt=int(math.ceil(float(min(t,30))/4.))
        st=0.5 if shifted and wt<t else 0.
        sh=0.5 if shifted and wh<h else 0.
        sw=0.5 if shifted and ww<w else 0.
        nt=int(math.ceil((float(t)-st)/float(wt)))
        nh=int(math.ceil((float(h)-sh)/float(wh)))
        nw=int(math.ceil((float(w)-sw)/float(ww)))
        if shifted:
            nt=nt+1 if st>0 else 1
            nh=nh+1 if sh>0 else 1
            nw=nw+1 if sw>0 else 1
        vo=torch.zeros((t,h,w,hd),dtype=video.dtype,device=video.device)
        to=torch.zeros((text_length,hd),dtype=text.dtype,device=text.device)
        windows=0
        txt_pos=torch.arange(text_length,device=text.device).unsqueeze(1).repeat(1,3)
        for iw in range(nw):
            w0=max(int((float(iw)-sw)*float(ww)),0)
            w1=min(int((float(iw)-sw+1.)*float(ww)),w)
            if w1<=w0:continue
            for ih in range(nh):
                h0=max(int((float(ih)-sh)*float(wh)),0)
                h1=min(int((float(ih)-sh+1.)*float(wh)),h)
                if h1<=h0:continue
                for it in range(nt):
                    t0=max(int((float(it)-st)*float(wt)),0)
                    t1=min(int((float(it)-st+1.)*float(wt)),t)
                    if t1<=t0:continue
                    local=video[t0:t1,h0:h1,w0:w1].reshape(-1,3,heads,128)
                    nv=local.shape[0]
                    tx=text.reshape(-1,3,heads,128)
                    vq=local[:,0];vk=local[:,1];tq=tx[:,0];tk=tx[:,1]
                    vq=vq*torch.rsqrt(vq.square().mean(-1,keepdim=True)+eps)*norm_weights[0]
                    vk=vk*torch.rsqrt(vk.square().mean(-1,keepdim=True)+eps)*norm_weights[1]
                    tq=tq*torch.rsqrt(tq.square().mean(-1,keepdim=True)+eps)*norm_weights[2]
                    tk=tk*torch.rsqrt(tk.square().mean(-1,keepdim=True)+eps)*norm_weights[3]
                    ids=torch.arange(nv,device=video.device)
                    lp=torch.stack((ids//((h1-h0)*(w1-w0))+text_length,
                                    (ids//(w1-w0))%(h1-h0),ids%(w1-w0)),dim=1)
                    q=torch.cat((self.rotate(vq,lp,frequencies),self.rotate(tq,txt_pos,frequencies)),dim=0).transpose(0,1)
                    k=torch.cat((self.rotate(vk,lp,frequencies),self.rotate(tk,txt_pos,frequencies)),dim=0).transpose(0,1)
                    v=torch.cat((local[:,2],tx[:,2]),dim=0).transpose(0,1)
                    result=F.scaled_dot_product_attention(q,k,v).transpose(0,1).reshape(nv+text_length,hd)
                    vo[t0:t1,h0:h1,w0:w1]=result[:nv].reshape(t1-t0,h1-h0,w1-w0,hd)
                    to=to+result[nv:]
                    windows+=1
        return vo,to/float(windows)


class ExportModel(nn.Module):
    def __init__(self,heads:int,shifted:bool):
        super().__init__()
        self.awa=AdaptiveWindowAttention(heads,shifted)

    def forward(self,video:torch.Tensor,text:torch.Tensor)->Tuple[torch.Tensor,torch.Tensor]:
        video_out,text_out=self.awa(video,text)
        return video_out,text_out
