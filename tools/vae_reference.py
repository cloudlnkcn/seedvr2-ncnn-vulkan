"""Pinned official VAE, FP32, one rank, memory slicing disabled.

Loads the unchanged official class bodies. Only distributed, logging and
memory-limit infrastructure are replaced by explicit single-process adapters.
The encoder returns all 32 distribution parameters, before posterior sampling;
the decoder consumes unscaled 16-channel latents. No random sampler is hidden.
"""
import ast
import hashlib
import importlib.util
import json
import logging
import math
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Callable, List, Literal, Optional, Tuple, Union

import diffusers
import torch
import torch.distributed as dist
from diffusers.models.attention_processor import Attention, SpatialNorm
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.downsampling import Downsample2D
from diffusers.models.lora import LoRACompatibleConv
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.normalization import RMSNorm
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.unets.unet_2d_blocks import DownEncoderBlock2D, UpDecoderBlock2D
from diffusers.models.upsampling import Upsample2D
from diffusers.utils import is_torch_version
from diffusers.utils.accelerate_utils import apply_forward_hook
from einops import rearrange
from torch import Tensor, nn
from torch.nn import Conv3d, functional as F

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'tests/reference/seedvr'


def verify_sources():
    manifest = json.loads((SOURCE/'sources.json').read_text())
    for row in manifest['files']:
        with (SOURCE/row['path']).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != row['sha256']:
                raise ValueError(f"Official reference changed: {row['path']}")
    return manifest


def load_bodies(relative, namespace):
    path = SOURCE/relative
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    exec(compile(tree, str(path), 'exec'), namespace)


def load_classes():
    verify_sources()
    ns = dict(globals())
    # The actual primitive and block bodies remain unchanged, including the
    # first-frame rule, affine normalization, residuals and attention processor.
    spec = importlib.util.spec_from_file_location('_seedvr_vae_types', SOURCE/'models/video_vae_v3/modules/types.py')
    types = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = types
    spec.loader.exec_module(types)
    for key in ['MemoryState', 'CausalAutoencoderOutput', 'CausalDecoderOutput', 'CausalEncoderOutput',
                '_inflation_mode_t', '_memory_device_t', '_receptive_field_t', '_selective_checkpointing_t']:
        ns[key] = getattr(types, key)
    ns.update(get_logger=logging.getLogger, get_sequence_parallel_group=lambda: None,
              get_sequence_parallel_world_size=lambda: 1, get_sequence_parallel_rank=lambda: 0,
              get_norm_limit=lambda: float('inf'), set_norm_limit=lambda value: None,
              safe_pad_operation=F.pad, safe_interpolate_operation=F.interpolate,
              causal_conv_slice_inputs=lambda x, *args, **kwargs: x,
              causal_conv_gather_outputs=lambda x: x)
    load_bodies('models/video_vae_v3/modules/causal_inflation_lib.py', ns)
    load_bodies('models/video_vae_v3/modules/attn_video_vae.py', ns)
    return ns


def load_checkpoint(path):
    lock = json.loads((ROOT/'model-sources.lock.json').read_text())
    entry = next(r for r in lock['files'] if r['rfilename'] == 'ema_vae.pth')
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if path.stat().st_size != entry['size'] or actual != entry['lfs']['sha256']:
        raise ValueError('Official VAE checkpoint hash mismatch')
    return torch.load(path, weights_only=True, mmap=True, map_location='cpu'), lock


def make_reference(state, part):
    ns = load_classes()
    common = dict(block_out_channels=(128, 256, 512, 512), layers_per_block=2,
                  act_fn='silu', norm_num_groups=32, inflation_mode='none',
                  time_receptive_field='full')
    # No allocation of random checkpoint-sized tensors before assign=True.
    with torch.device('meta'):
        if part == 'encoder':
            module = ns['Encoder3D'](in_channels=3, out_channels=16, double_z=True,
                                    down_block_types=('DownEncoderBlock3D',)*4,
                                    temporal_down_num=2, **common)
        elif part == 'decoder':
            module = ns['Decoder3D'](in_channels=16, out_channels=3, temporal_up_num=2,
                                    up_block_types=('UpDecoderBlock3D',)*4, **common)
        else:
            raise ValueError('Expected encoder or decoder')
    prefix = part+'.'
    weights = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    module.load_state_dict(weights, strict=True, assign=True)
    module.requires_grad_(False).eval()
    return module, ns['MemoryState'].DISABLED


def evaluate(module, memory_state, x):
    with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        return module(x.unsqueeze(2), memory_state=memory_state).squeeze(2)
