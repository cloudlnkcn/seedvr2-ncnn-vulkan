"""Complete FP32-B image reference with the pinned original NaDiT forward body.

Only device/distribution infrastructure and lazy checkpoint loading are adapters.
The common cache is retained across all 32 original blocks and the output Ada.
"""
from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
import gc
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple, Union

import torch
from diffusers.models.embeddings import get_timestep_embedding
from einops import rearrange
from torch import nn
from torch.nn.modules.utils import _triple

from awa_reference import classes, load_reference, module, verify_sources
from dit_block_reference import RMSNorm, make_block
from dit_block_module import linear


def make_dit(state, observe=lambda name, value: None):
    verify_sources()
    _, Cache = load_reference()
    na = module('image_na', 'models/dit_v2/na.py')
    ns = dict(globals(), Cache=Cache, na=na,
              slice_inputs=lambda x, **kwargs: x, gather_outputs=lambda x, **kwargs: x)
    classes('models/dit_v2/embedding.py', ns)
    classes('models/dit_v2/modulation.py', ns)
    classes('models/dit_v2/patch/patch_v1.py', ns)
    classes('models/dit_v2/nadit.py', ns)
    # Bypass only eager construction; all original forward methods are used.
    model = ns['NaDiT'].__new__(ns['NaDiT'])
    nn.Module.__init__(model)
    with torch.device('meta'):
        model.vid_in = ns['NaPatchIn'](33, (1, 2, 2), 2560)
        model.emb_in = ns['TimeEmbedding'](256, 2560, 15360)
        model.vid_out_norm = RMSNorm(2560, 1e-5, True)
        model.vid_out_ada = ns['AdaSingle'](2560, 15360, ['out'], ['in'])
        model.vid_out = ns['NaPatchOut'](16, (1, 2, 2), 2560)
    model.txt_in = linear(state, 'txt_in')
    for name in ['vid_in', 'emb_in', 'vid_out_norm', 'vid_out_ada', 'vid_out']:
        prefix = name+'.'
        getattr(model, name).load_state_dict(
            {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)},
            strict=True, assign=True)

    class LazyBlock(nn.Module):
        def __init__(self, index):
            super().__init__()
            self.index = index

        def forward(self, **kwargs):
            block, _ = make_block(state, self.index)
            result = block(**kwargs)
            observe(f'block-{self.index:02d}-video', result[0])
            observe(f'block-{self.index:02d}-text', result[1])
            del block
            gc.collect()
            return result

    model.blocks = nn.ModuleList(LazyBlock(i) for i in range(32))
    model.vid_in.register_forward_hook(lambda m, args, out: observe('patch-in', out[0]))
    model.txt_in.register_forward_hook(lambda m, args, out: observe('text-in', out))
    model.emb_in.register_forward_hook(lambda m, args, out: observe('time-in', out.reshape(2560, 6)))
    model.vid_out_ada.register_forward_hook(lambda m, args, out: observe('output-ada', out))
    return model.requires_grad_(False).eval()


def endpoint(noise, predict):
    types = module('image_diffusion_types', 'common/diffusion/types.py')
    ns = dict(globals(), PredictionType=types.PredictionType,
              SamplingDirection=types.SamplingDirection, SamplingTimesteps=object,
              expand_dims=lambda x, ndim: x.reshape(x.shape+(1,)*(ndim-x.ndim)))
    classes('common/diffusion/schedules/base.py', ns)
    classes('common/diffusion/schedules/lerp.py', ns)
    classes('common/diffusion/samplers/base.py', ns)
    classes('common/diffusion/samplers/euler.py', ns)
    sampler = ns['EulerSampler'].__new__(ns['EulerSampler'])
    sampler.schedule = ns['LinearInterpolationSchedule'](1000.)
    sampler.prediction_type = types.PredictionType.v_lerp
    sampler.timesteps = SimpleNamespace(timesteps=torch.tensor([1000.]),
                                       direction=types.SamplingDirection.backward)
    sampler.return_endpoint = True
    sampler.get_progress_bar = lambda: SimpleNamespace(update=lambda: None)
    return sampler.sample(noise, lambda args: predict(args.x_t, args.t))
