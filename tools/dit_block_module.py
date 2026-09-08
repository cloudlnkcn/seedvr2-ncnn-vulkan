"""Exportable full DiT block: native pnnx operations surrounding custom AWA."""
import torch
from torch import nn
from torch.nn import functional as F

from awa_export_module import AdaptiveWindowAttention


def linear(state, key):
    w = state[key+'.weight']
    bias = state.get(key+'.bias')
    layer = nn.Linear(w.shape[1], w.shape[0], bias=bias is not None, device='meta')
    layer.weight = nn.Parameter(w, False)
    if bias is not None:
        layer.bias = nn.Parameter(bias, False)
    return layer


class Branch(nn.Module):
    def __init__(self, state, prefix, branch):
        super().__init__()
        self.qkv = linear(state, prefix+f'attn.proj_qkv.{branch}')
        self.out = linear(state, prefix+f'attn.proj_out.{branch}')
        self.gate = linear(state, prefix+f'mlp.{branch}.proj_in_gate')
        self.value = linear(state, prefix+f'mlp.{branch}.proj_in')
        self.mlp_out = linear(state, prefix+f'mlp.{branch}.proj_out')
        for field in ['attn_shift', 'attn_scale', 'attn_gate', 'mlp_shift', 'mlp_scale', 'mlp_gate']:
            self.register_parameter(field, nn.Parameter(state[prefix+f'ada.{branch}.{field}'], False))

    def pre(self, x, e):
        y = F.rms_norm(x, (2560,), eps=1e-5)
        return self.qkv(y*(e[:, 1]+self.attn_scale)+(e[:, 0]+self.attn_shift))

    def post(self, x, a, e):
        y = x + self.out(a)*(e[:, 2]+self.attn_gate)
        z = F.rms_norm(y, (2560,), eps=1e-5)
        z = z*(e[:, 4]+self.mlp_scale)+(e[:, 3]+self.mlp_shift)
        z = self.mlp_out(F.silu(self.gate(z))*self.value(z))
        return y + z*(e[:, 5]+self.mlp_gate)


class DiTBlock(nn.Module):
    def __init__(self, state, index):
        super().__init__()
        prefix = f'blocks.{index}.'
        shared = index >= 10
        self.last = index == 31
        self.vid = Branch(state, prefix, 'all' if shared else 'vid')
        self.txt = self.vid if shared else Branch(state, prefix, 'txt')
        weights = torch.stack([state[prefix+f'attn.norm_{axis}.{"all" if shared else branch}.weight']
                               for branch, axis in [('vid', 'q'), ('vid', 'k'), ('txt', 'q'), ('txt', 'k')]])
        # Module boundary stays opaque to pnnx, while the surrounding block is
        # exported as standard ncnn layers with its actual checkpoint weights.
        awa = AdaptiveWindowAttention(20, bool(index % 2))
        awa.norm_weights = nn.Parameter(weights, False)
        self.awa = torch.jit.script(awa)

    def forward(self, vid, txt, emb):
        t, h, w = vid.shape[:3]
        v = vid.reshape(-1, 2560)
        vqkv = self.vid.pre(v, emb).reshape(t, h, w, 7680)
        tqkv = self.txt.qkv(F.rms_norm(txt, (2560,), eps=1e-5)) if self.last else self.txt.pre(txt, emb)
        a, b = self.awa(vqkv, tqkv)
        v = self.vid.post(v, a.reshape(-1, 2560), emb)
        # Official MMModule(vid_only=True) leaves text unchanged in the MLP;
        # the block's final residual addition therefore doubles that branch.
        txt = 2*(txt+self.txt.out(b)) if self.last else self.txt.post(txt, b, emb)
        return v.reshape(t, h, w, 2560), txt
