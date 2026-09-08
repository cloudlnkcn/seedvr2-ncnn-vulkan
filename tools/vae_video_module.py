"""Whole-clip, causal SeedVR2 VAE export boundaries (B=1, memory disabled).

Temporal weights are preserved. GroupNorm and mid-block attention act on each
frame; causal convolutions and pixel shuffle carry information through time.
The reference used to validate this graph is the unchanged upstream 3D VAE.
"""
import torch
from torch import nn
from torch.nn import functional as F


class TemporalConv(nn.Module):
    def __init__(self, weight, bias, stride=1, temporal_stride=1, down=False):
        super().__init__()
        if weight.ndim == 2:
            weight = weight[:, :, None, None, None]
        self.register_buffer('weight', weight.detach().float())
        self.register_buffer('bias', bias.detach().float())
        kt, kh, kw = weight.shape[2:]
        assert kt in (1, 3) and kh == kw and kh in (1, 3)
        pad = 0 if down or kh == 1 else 1
        self.register_buffer('spec', torch.tensor([
            1, weight.shape[1], weight.shape[0], kt, kh, temporal_stride,
            stride, pad, 1 if down else pad], dtype=torch.int32))
        self.kt, self.pad, self.end = kt, pad, 1 if down else pad
        self.stride, self.temporal_stride = stride, temporal_stride

    def forward(self, x):
        assert int(self.spec[0].item()) == 1
        kt, st, ss = int(self.spec[3].item()), int(self.spec[5].item()), int(self.spec[6].item())
        pad, end = int(self.spec[7].item()), int(self.spec[8].item())
        x = F.pad(x, (pad, end, pad, end))
        x = torch.cat((x[:, :1].expand(-1, kt-1, -1, -1), x), dim=1)
        return F.conv3d(x.unsqueeze(0), self.weight, self.bias,
                        stride=(st, ss, ss))[0]


class FrameNorm(nn.Module):
    def __init__(self, state, key):
        super().__init__()
        self.register_buffer('weight', state[key+'.weight'].float())
        self.register_buffer('bias', state[key+'.bias'].float())
        self.register_buffer('spec', torch.tensor([1, len(self.weight), 32], dtype=torch.int32))

    def forward(self, x):
        assert int(self.spec[0].item()) == 1 and int(self.spec[1].item()) == x.shape[0]
        return F.group_norm(x.transpose(0, 1), int(self.spec[2].item()), self.weight, self.bias, 1e-6).transpose(0, 1)


class FrameSDPA(nn.Module):
    def sequence(self, x):
        return x.permute(1, 2, 3, 0).flatten(1, 2).unsqueeze(1)

    def forward(self, q, k, v):
        # T is a batch of spatial attention problems, never the token axis.
        y = F.scaled_dot_product_attention(self.sequence(q), self.sequence(k), self.sequence(v))
        return y.squeeze(1).reshape(q.shape[1], q.shape[2], q.shape[3], q.shape[0]).permute(3, 0, 1, 2)


class TemporalShuffle(nn.Module):
    def __init__(self, channels, ratio):
        super().__init__()
        self.channels, self.ratio = channels, ratio
        self.register_buffer('spec', torch.tensor([1, channels, ratio], dtype=torch.int32))

    def forward(self, x):
        assert int(self.spec[0].item()) == 1
        channels, ratio = int(self.spec[1].item()), int(self.spec[2].item())
        y = x.reshape(2, 2, ratio, channels, x.shape[1], x.shape[2], x.shape[3])
        y = y.permute(3, 4, 2, 5, 0, 6, 1).reshape(
            channels, x.shape[1]*ratio, x.shape[2]*2, x.shape[3]*2)
        # Upstream remove_head keeps phase zero of the first frame.
        return torch.cat((y[:, :1], y[:, ratio:]), 1)


def conv(state, key, **kwargs):
    return TemporalConv(state[key+'.weight'], state[key+'.bias'], **kwargs)


class Residual(nn.Module):
    def __init__(self, state, key):
        super().__init__()
        self.norm1, self.norm2 = FrameNorm(state, key+'.norm1'), FrameNorm(state, key+'.norm2')
        self.conv1, self.conv2 = conv(state, key+'.conv1'), conv(state, key+'.conv2')
        self.shortcut = conv(state, key+'.conv_shortcut') if key+'.conv_shortcut.weight' in state else nn.Identity()

    def forward(self, x):
        return self.shortcut(x) + self.conv2(F.silu(self.norm2(self.conv1(F.silu(self.norm1(x))))))


class Attention(nn.Module):
    def __init__(self, state, key):
        super().__init__()
        self.norm = FrameNorm(state, key+'.group_norm')
        self.q, self.k, self.v = [conv(state, key+'.to_'+name) for name in ('q', 'k', 'v')]
        self.out = conv(state, key+'.to_out.0')
        self.attention = FrameSDPA()

    def forward(self, x):
        y = self.norm(x)
        return x + self.out(self.attention(self.q(y), self.k(y), self.v(y)))


class Up(nn.Module):
    def __init__(self, state, key, ratio):
        super().__init__()
        self.upscale = conv(state, key+'.upscale_conv')
        self.shuffle = TemporalShuffle(state[key+'.upscale_conv.weight'].shape[1], ratio)
        self.conv = conv(state, key+'.conv')

    def forward(self, x):
        return self.conv(self.shuffle(self.upscale(x)))


class VideoVAE(nn.Module):
    def __init__(self, state, part):
        super().__init__()
        self.conv_in = conv(state, part+'.conv_in')
        def mid():
            key = part+'.mid_block'
            return [Residual(state, key+'.resnets.0'), Attention(state, key+'.attentions.0'), Residual(state, key+'.resnets.1')]
        layers = mid() if part == 'decoder' else []
        for i in range(4):
            key = f'{part}.{"down" if part == "encoder" else "up"}_blocks.{i}'
            layers += [Residual(state, key+f'.resnets.{j}') for j in range(2 if part == 'encoder' else 3)]
            if i < 3:
                # Encoder time reduction is at down blocks 1 and 2; decoder
                # time expansion is at up blocks 0 and 1.
                layers += [conv(state, key+'.downsamplers.0.conv', stride=2, temporal_stride=2 if i >= 1 else 1, down=True)] if part == 'encoder' else [Up(state, key+'.upsamplers.0', 2 if i < 2 else 1)]
        if part == 'encoder':
            layers += mid()
        self.layers = nn.Sequential(*layers)
        self.norm = FrameNorm(state, part+'.conv_norm_out')
        self.conv_out = conv(state, part+'.conv_out')

    def forward(self, x):
        return self.conv_out(F.silu(self.norm(self.layers(self.conv_in(x)))))
