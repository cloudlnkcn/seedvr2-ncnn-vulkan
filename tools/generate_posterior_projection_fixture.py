#!/usr/bin/env python3
"""Independent oneDNN FP32 and FP64 probes for the VAE's blocked convolutions."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(4)

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def save(name, value):
    path = args.output / name
    value.contiguous().numpy().astype('<f4').tofile(path)
    return dict(path=name, dtype='f32le', shape=list(value.shape), sha256=sha(path))

def values(count, seed, divisor):
    state = seed
    result = []
    for _ in range(count):
        state = (1664525 * state + 1013904223) & 0xffffffff
        result.append(((state >> 8) % 131071 - 65535) / divisor)
    return torch.tensor(result, dtype=torch.float32)

x = values(512 * 3 * 3 * 4, 712, 32768).reshape(512, 3, 3, 4)
w = values(32 * 512 * 27, 216, 4194304).reshape(32, 512, 3, 3, 3)
b = values(32, 173, 1048576)
with torch.inference_mode():
    padded = F.pad(x[None].double(), (0, 0, 0, 0, 2, 0), mode='replicate')
    y = F.conv3d(padded, w.double(), b.double(), padding=(0, 1, 1))[0].float()
    # This compact probe is normally sent to Slow3d by torch's size heuristic.
    # Select the same oneDNN FP32 kernel family as the real video posterior.
    fp32 = F.conv3d(padded.float().to_mkldnn(), w, b, padding=(0, 1, 1)).to_dense()[0]
case = dict(stride=1, input=save('input.f32', x),
            weights=save('weights.f32', torch.cat((w.flatten(), b))),
            reference=save('reference.f32', y),
            onednn_fp32=save('onednn-fp32.f32', fp32))
cases = [case]
for index, (ci, co, kt, kernel, temporal_stride, stride) in enumerate([
        (16, 32, 3, 3, 1, 1), (128, 128, 3, 3, 1, 1),
        (256, 256, 3, 3, 2, 2), (128, 128, 1, 3, 1, 2),
        (128, 256, 1, 1, 1, 1), (256, 512, 1, 1, 1, 1),
        (512, 256, 1, 1, 1, 1), (256, 128, 1, 1, 1, 1)]):
    source = values(ci * 5 * 7 * 9, 900 + index, 32768).reshape(ci, 5, 7, 9)
    weight = values(co * ci * kt * kernel * kernel, 1100 + index, 4194304).reshape(co, ci, kt, kernel, kernel)
    bias = values(co, 1200 + index, 1048576)
    pad = 1 if stride == 1 and kernel == 3 else 0
    end = 1 if kernel == 3 else 0
    with torch.inference_mode():
        padded_source = F.pad(source[None], (0, 0, 0, 0, kt - 1, 0), mode='replicate')
        padded_source = F.pad(padded_source, (pad, end, pad, end))
        result = F.conv3d(padded_source.to_mkldnn(), weight, bias,
                          stride=(temporal_stride, stride, stride)).to_dense()[0]
        high_precision = F.conv3d(padded_source.double(), weight.double(), bias.double(),
                                  stride=(temporal_stride, stride, stride))[0].float()
    prefix = 'blocked-' + str(index)
    cases.append(dict(stride=stride, temporal_stride=temporal_stride, temporal_kernel=kt, kernel=kernel,
                      padding=pad, end_padding=end,
                      input=save(prefix + '-input.f32', source),
                      weights=save(prefix + '-weights.f32', torch.cat((weight.flatten(), bias))),
                      reference=save(prefix + '-fp64.f32', high_precision),
                      onednn_fp32=save(prefix + '-onednn.f32', result)))
report = dict(schema_version='seedvr2-posterior-projection-fixture-v2',
              torch_version=torch.__version__, generator_sha256=sha(Path(__file__)),
              oracle='Explicit oneDNN FP32 conv3d via torch mkldnn tensor; original FP64-rounded reference retained as an accuracy diagnostic',
              default_dense_backend=str(torch._C._select_conv_backend(padded.float(), w, b, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1)),
              oracle_backend='Mkldnn (explicit)',
              torch_build=torch.__config__.show(),
              tolerance='Exact oneDNN FP32 reference; FP64 differences reported separately; not a full-model gate',
              cases=cases, model_verified=False)
(args.output / 'provenance.json').write_text(json.dumps(report, indent=2) + '\n')
