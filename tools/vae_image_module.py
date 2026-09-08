"""Exact single-frame specialization of the pinned causal VAE.

Temporal kernel slices are summed because extend_head repeats the first frame.
For a twofold temporal upsample, official remove_head keeps phase zero. Reorder
the surviving (x,y,c) output channels to (c,x,y) for standard PixelShuffle.
This transform is invalid for T>1 and is never registered as a video VAE.
"""
import torch
from torch import nn
from torch.nn import functional as F


def conv(state, key, stride=1, padding=1):
    weight = state[key+'.weight']
    if weight.ndim != 5 or weight.shape[2] not in (1, 3):
        raise ValueError(f'Unsupported causal convolution: {key}')
    layer = nn.Conv2d(weight.shape[1], weight.shape[0], weight.shape[-2:], stride, padding)
    layer.weight = nn.Parameter(weight.sum(dim=2), requires_grad=False)
    layer.bias = nn.Parameter(state[key+'.bias'], requires_grad=False)
    return layer


def norm(state, key):
    layer = nn.GroupNorm(32, len(state[key+'.weight']), eps=1e-6)
    layer.weight = nn.Parameter(state[key+'.weight'], requires_grad=False)
    layer.bias = nn.Parameter(state[key+'.bias'], requires_grad=False)
    return layer


class Residual(nn.Module):
    def __init__(self, state, key):
        super().__init__()
        self.norm1, self.norm2 = norm(state, key+'.norm1'), norm(state, key+'.norm2')
        self.conv1, self.conv2 = conv(state, key+'.conv1'), conv(state, key+'.conv2')
        self.shortcut = (conv(state, key+'.conv_shortcut', padding=0)
                         if key+'.conv_shortcut.weight' in state else nn.Identity())

    def forward(self, x):
        return self.shortcut(x) + self.conv2(F.silu(self.norm2(self.conv1(F.silu(self.norm1(x))))))


class Attention2d(nn.Module):
    def __init__(self, state, key):
        super().__init__()
        self.norm = norm(state, key+'.group_norm')
        for name in ['to_q', 'to_k', 'to_v', 'to_out']:
            original = name if name != 'to_out' else 'to_out.0'
            w = state[key+'.'+original+'.weight']
            layer = nn.Conv2d(w.shape[1], w.shape[0], 1)
            layer.weight = nn.Parameter(w[:, :, None, None], requires_grad=False)
            layer.bias = nn.Parameter(state[key+'.'+original+'.bias'], requires_grad=False)
            setattr(self, name, layer)

    def forward(self, x):
        y = self.norm(x)
        # Official VAE mid attention: one head of width 512.
        q = self.to_q(y).flatten(2).transpose(1, 2).unsqueeze(1)
        k = self.to_k(y).flatten(2).transpose(1, 2).unsqueeze(1)
        v = self.to_v(y).flatten(2).transpose(1, 2).unsqueeze(1)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0., is_causal=False)
        y = y.squeeze(1).transpose(1, 2).reshape_as(x)
        return x + self.to_out(y)


class Up(nn.Module):
    def __init__(self, state, key, temporal):
        super().__init__()
        w, b = state[key+'.upscale_conv.weight'], state[key+'.upscale_conv.bias']
        c = w.shape[1]
        w = w.reshape(2, 2, temporal, c, c)[:, :, 0].permute(2, 0, 1, 3).reshape(4*c, c, 1, 1)
        b = b.reshape(2, 2, temporal, c)[:, :, 0].permute(2, 0, 1).reshape(4*c)
        self.upscale = nn.Conv2d(c, 4*c, 1)
        self.upscale.weight, self.upscale.bias = nn.Parameter(w, False), nn.Parameter(b, False)
        self.shuffle = nn.PixelShuffle(2)
        self.conv = conv(state, key+'.conv')

    def forward(self, x):
        return self.conv(self.shuffle(self.upscale(x)))


class Down(nn.Module):
    def __init__(self, state, key):
        super().__init__()
        self.conv = conv(state, key+'.conv', stride=2, padding=0)

    def forward(self, x):
        return self.conv(F.pad(x, (0, 1, 0, 1)))


class ImageVAE(nn.Module):
    def __init__(self, state, part):
        super().__init__()
        self.conv_in = conv(state, part+'.conv_in')
        layers = []
        def mid():
            key = part+'.mid_block'
            return [Residual(state, key+'.resnets.0'), Attention2d(state, key+'.attentions.0'),
                    Residual(state, key+'.resnets.1')]
        if part == 'decoder':
            layers += mid()
        for index in range(4):
            key = f'{part}.{"down" if part == "encoder" else "up"}_blocks.{index}'
            layers += [Residual(state, key+f'.resnets.{i}') for i in range(2 if part == 'encoder' else 3)]
            if index < 3:
                layers += ([Down(state, key+'.downsamplers.0')] if part == 'encoder' else
                           [Up(state, key+'.upsamplers.0', temporal=2 if index < 2 else 1)])
        if part == 'encoder':
            layers += mid()
        self.layers = nn.Sequential(*layers)
        self.norm = norm(state, part+'.conv_norm_out')
        self.conv_out = conv(state, part+'.conv_out')

    def forward(self, x):
        return self.conv_out(F.silu(self.norm(self.layers(self.conv_in(x)))))
