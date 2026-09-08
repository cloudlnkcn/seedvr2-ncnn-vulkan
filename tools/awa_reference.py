"""Run the pinned official NaSwinAttention body under an explicit FP32-B adapter.

Changes to official execution: single rank collectives; RMSNorm in FP32;
remove the three explicit BF16 casts; FlashAttention replaced by PyTorch math
SDPA; compute only the needed prefix of the identical language-RoPE table.
Projection layers are identity: the exported boundary starts at projected QKV
and ends before output projection. This is not the official BF16 reference A.
"""
import ast
import hashlib
import importlib.util
import json
import sys
from functools import lru_cache
from itertools import chain
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import torch
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _triple

SOURCE = Path(__file__).resolve().parents[1]/'tests/reference/seedvr'


def verify_sources():
    manifest = json.loads((SOURCE/'sources.json').read_text())
    for row in manifest['files']:
        if hashlib.file_digest((SOURCE/row['path']).open('rb'), 'sha256').hexdigest() != row['sha256']:
            raise RuntimeError(f"Official reference changed: {row['path']}")
    return manifest


def module(name, relative):
    spec = importlib.util.spec_from_file_location('_seedvr_reference_'+name, SOURCE/relative)
    result = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = result
    spec.loader.exec_module(result)
    return result


def classes(relative, namespace, remove_bf16=False):
    tree = ast.parse((SOURCE/relative).read_text())
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    if remove_bf16:
        class Float32Adapter(ast.NodeTransformer):
            count = 0
            def visit_Call(self, node):
                self.generic_visit(node)
                if isinstance(node.func, ast.Attribute) and node.func.attr == 'bfloat16' and not node.args and not node.keywords:
                    self.count += 1
                    return node.func.value
                return node
        adapter = Float32Adapter()
        tree = adapter.visit(tree)
        if adapter.count != 6:  # three in each of NaMMAttention and NaSwinAttention
            raise RuntimeError('Official BF16 call sites changed')
    exec(compile(ast.fix_missing_locations(tree), str(SOURCE/relative), 'exec'), namespace)


class FP32RMSNorm(nn.Module):
    def __init__(self, dim, eps, elementwise_affine):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight


class PairIdentity(nn.Module):
    def forward(self, vid, txt):
        return vid, txt


class MathAttention(nn.Module):
    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k, **kwargs):
        if not torch.equal(cu_seqlens_q, cu_seqlens_k):
            raise RuntimeError('Expected self attention')
        boundaries = cu_seqlens_q.tolist()
        out = []
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            for lo, hi in zip(boundaries, boundaries[1:]):
                out.append(F.scaled_dot_product_attention(
                    q[lo:hi].transpose(0, 1), k[lo:hi].transpose(0, 1),
                    v[lo:hi].transpose(0, 1), dropout_p=0., is_causal=False).transpose(0, 1))
        return torch.cat(out)


@lru_cache(maxsize=1)
def load_reference():
    verify_sources()
    cache = module('cache', 'common/cache.py')
    na = module('na', 'models/dit_v2/na.py')
    mm = module('mm', 'models/dit_v2/mm.py')
    win = module('window', 'models/dit_v2/window.py')
    ns = dict(torch=torch, nn=nn, F=F, lru_cache=lru_cache, Optional=Optional,
              Tuple=Tuple, Union=Union, rearrange=rearrange, chain=chain,
              RotaryEmbedding=RotaryEmbedding, apply_rotary_emb=apply_rotary_emb,
              Cache=cache.Cache)
    classes('models/dit_v2/rope.py', ns)
    def compact_freqs(self, vid_shape, txt_shape):
        dims = vid_shape.tolist()
        lengths = txt_shape[:, 0].tolist()
        if max(l+s[0] for l,s in zip(lengths,dims)) > 1024 or max(max(s[1:]) for s in dims) > 128:
            raise ValueError('Outside the official RoPE table')
        vf = self.get_axial_freqs(max(l+s[0] for l,s in zip(lengths,dims)),
                                 max(s[1] for s in dims), max(s[2] for s in dims))
        tf = self.get_axial_freqs(max(lengths))
        return (torch.cat([vf[l:l+t,:h,:w].reshape(-1,vf.size(-1)) for (t,h,w),l in zip(dims,lengths)]),
                torch.cat([tf[:l].repeat(1,3).reshape(-1,vf.size(-1)) for l in lengths]))
    ns['NaMMRotaryEmbedding3d'].get_freqs = compact_freqs
    ns.update(na=na, MMArg=mm.MMArg, MMModule=mm.MMModule,
              get_window_op=win.get_window_op, _triple=_triple,
              norm_layer_type=Callable, FlashAttentionVarlen=MathAttention,
              safe_pad_operation=F.pad,
              gather_seq_scatter_heads_qkv=lambda x, **kwargs: x,
              gather_heads_scatter_seq=lambda x, **kwargs: x)
    classes('models/dit_v2/nablocks/attention/mmattn.py', ns, remove_bf16=True)
    return ns['NaSwinAttention'], cache.Cache


class AdaptiveWindowAttention(nn.Module):
    """Export boundary. Metadata comes from 4-D video shape, never tensor values."""
    def __init__(self, heads=2, shifted=False, eps=1e-5):
        super().__init__()
        cls, _ = load_reference()
        self.heads, self.shifted, self.eps = heads, shifted, eps
        self.reference = cls(vid_dim=1, txt_dim=1, heads=heads, head_dim=128,
                             qk_bias=False, qk_norm=FP32RMSNorm, qk_norm_eps=eps,
                             rope_type='mmrope3d', rope_dim=128, shared_weights=False,
                             window=(4,3,3), window_method=('720pswin_by_size_bysize' if shifted else '720pwin_by_size_bysize'))
        self.reference.proj_qkv = PairIdentity()
        self.reference.proj_out = PairIdentity()

    def set_norm_weights(self, weights):
        with torch.no_grad():
            for parameter, value in zip(self.norm_parameters(), weights):
                parameter.copy_(value)

    def norm_parameters(self):
        r = self.reference
        return [r.norm_q.vid.weight, r.norm_k.vid.weight,
                r.norm_q.txt.weight, r.norm_k.txt.weight]

    def forward(self, video_qkv, text_qkv):
        _, cache = load_reference()
        t,h,w,_ = video_qkv.shape
        vid_shape = torch.tensor([[t,h,w]], dtype=torch.int64)
        txt_shape = torch.tensor([[text_qkv.shape[0]]], dtype=torch.int64)
        vid,txt = self.reference(video_qkv.reshape(-1,3*self.heads*128), text_qkv,
                                 vid_shape, txt_shape, cache())
        return vid.reshape(t,h,w,self.heads*128), txt
