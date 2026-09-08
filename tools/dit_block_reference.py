"""Run the complete pinned official transformer block under the FP32-B profile."""
from typing import Callable, List, Optional, Tuple

import torch
from diffusers.models.embeddings import get_timestep_embedding
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from awa_reference import classes, load_reference, module, verify_sources


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.register_parameter('weight', nn.Parameter(torch.ones(dim)) if elementwise_affine else None)

    def forward(self, x):
        y = x * torch.rsqrt(x.square().mean(-1, keepdim=True)+self.eps)
        return y if self.weight is None else y*self.weight


def make_block(state, index):
    verify_sources()
    attention, cache = load_reference()
    mm = module('mm_block', 'models/dit_v2/mm.py')
    ns = dict(torch=torch, nn=nn, F=F, rearrange=rearrange, Callable=Callable, List=List,
              Tuple=Tuple, Optional=Optional, Cache=cache, slice_inputs=lambda x, **kwargs: x,
              NaSwinAttention=attention, MMArg=mm.MMArg, MMModule=mm.MMModule,
              norm_layer_type=Callable)
    classes('models/dit_v2/modulation.py', ns)
    classes('models/dit_v2/mlp.py', ns)
    classes('models/dit_v2/nablocks/mmsr_block.py', ns)
    # RotaryEmbedding creates a small concrete buffer, so avoid a meta context
    # for the official attention. torch.empty initialization remains bounded to
    # this one block; checkpoint assignment releases initial parameters.
    block = ns['NaMMSRTransformerBlock'](vid_dim=2560, txt_dim=2560, emb_dim=15360,
        heads=20, head_dim=128, expand_ratio=4, norm=RMSNorm, norm_eps=1e-5,
        ada=ns['AdaSingle'], qk_bias=False, qk_norm=RMSNorm, mlp_type='swiglu',
        shared_weights=index >= 10, rope_type='mmrope3d', rope_dim=128,
        is_last_layer=index == 31, window=(4, 3, 3),
        window_method='720pswin_by_size_bysize' if index % 2 else '720pwin_by_size_bysize')
    prefix = f'blocks.{index}.'
    block.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)},
                          strict=True, assign=True)
    return block.requires_grad_(False).eval(), cache


def time_embedding(state, timestep=1000.):
    x = get_timestep_embedding(torch.tensor([timestep]), 256, flip_sin_to_cos=False, downscale_freq_shift=0)
    with torch.inference_mode():
        for name in ['proj_in', 'proj_hid', 'proj_out']:
            x = F.linear(x, state[f'emb_in.{name}.weight'], state[f'emb_in.{name}.bias'])
            if name != 'proj_out':
                x = F.silu(x)
    return x.reshape(2560, 6)


def evaluate(block, cache, vid, txt, emb):
    grid = vid.shape[:3]
    with torch.inference_mode():
        v, t, _, _ = block(vid.reshape(-1, 2560).clone(), txt.clone(),
                           torch.tensor([grid]), torch.tensor([[len(txt)]]), emb.reshape(1, 15360), cache())
    return v.reshape(*grid, 2560), t
