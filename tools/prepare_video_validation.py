#!/usr/bin/env python3
"""Prepare three bounded development clips, without running or certifying a model."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from torch.nn import functional as F

SOURCE_COMMIT = '40e8f791aef186e9f051d60a2a1ee17319190127'
SOURCE_SHA256 = '5fde35f5a288ca86e216d2dc28188ab64b4560d3021f273faefdf0de80f38aa5'
SOURCE_URL = f'https://raw.githubusercontent.com/imageio/imageio-binaries/{SOURCE_COMMIT}/images/cockatoo.mp4'
CASES = [('motion-9', list(range(8, 17)), None),
         ('padding-8', list(range(8, 16)), None),
         ('cut-17', list(range(8, 16)) + list(range(80, 89)), 8)]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if sha(a.source) != SOURCE_SHA256:
        p.error('Expected the pinned ImageIO cockatoo.mp4; source identity differs')
    if a.output.exists():
        p.error('Use a new output directory; existing fixtures are never overwritten')
    a.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-threads', '1',
               '-i', str(a.source.resolve()), '-map', '0:v:0', '-an',
               '-vf', 'fps=8,scale=-2:80:flags=lanczos,crop=128:80,setsar=1',
               '-frames:v', '89', '-threads', '1', '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1']
    result = subprocess.run(command, capture_output=True, check=True, timeout=60)
    (a.output/'decode.stderr').write_bytes(result.stderr)
    if len(result.stdout) != 89 * 80 * 128 * 3:
        raise ValueError('The source did not produce the exact expected RGB frames')
    source = np.frombuffer(result.stdout, dtype=np.uint8).reshape(89, 80, 128, 3)
    rows = []
    for name, indices, cut in CASES:
        folder = a.output/name
        folder.mkdir()
        target = source[indices].copy()
        (folder/'target.rgb').write_bytes(target.tobytes())
        values = torch.from_numpy(target).permute(0, 3, 1, 2).float()/255
        low = F.interpolate(values, size=(40, 64), mode='bicubic',
                            align_corners=False, antialias=True).clamp(0, 1)
        low = (low*255).round().byte().permute(0, 2, 3, 1).contiguous().numpy()
        encode = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo',
                  '-pix_fmt', 'rgb24', '-s', '64x40', '-r', '8', '-i', 'pipe:0', '-an',
                  '-c:v', 'libx264', '-crf', '28', '-preset', 'medium', '-threads', '1',
                  '-pix_fmt', 'yuv420p', '-vf', 'setsar=1', '-movflags', '+faststart',
                  str((folder/'input.mp4').resolve())]
        result = subprocess.run(encode, input=low.tobytes(), capture_output=True,
                                check=True, timeout=30)
        (folder/'encode.stderr').write_bytes(result.stderr)
        frames = len(indices)
        padded = ((frames-1+3)//4)*4+1
        rows.append(dict(case_id=name, frames=frames, padded_frames=padded,
            latent_frames=(padded-1)//4+1, width=128, height=80, fps=8,
            source_frames_at_8fps=indices, artificial_cut_before_frame=cut,
            input=dict(path=f'{name}/input.mp4', sha256=sha(folder/'input.mp4')),
            target=dict(path=f'{name}/target.rgb', sha256=sha(folder/'target.rgb'),
                        shape=[frames, 80, 128, 3], dtype='rgb8'), encode_command=encode))
    report = dict(schema_version='seedvr2-bounded-video-fixtures-v1',
        scope='Three development cases from ONE source, with synthetic degradation. '
              'Not a representative quality benchmark or independent natural cuts.',
        source=dict(url=SOURCE_URL, commit=SOURCE_COMMIT, sha256=SOURCE_SHA256,
                    bytes=a.source.stat().st_size,
                    provenance='https://imageio.readthedocs.io/en/stable/user_guide/standardimages.html',
                    rights_note='ImageIO lists these standard images as public domain to its knowledge; '
                                'source attribution retained, not independent rights certification.'),
        target_note='Fixed RGB from an already compressed source; not camera-original ground truth.',
        decode_command=command,
        degradation='bicubic antialias 128x80 -> 64x40; clamp and nearest-even RGB8; H.264 CRF28 yuv420p',
        versions=dict(torch=torch.__version__, numpy=np.__version__,
                      ffmpeg=subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0]),
        generator_sha256=sha(Path(__file__)), cases=rows, model_verified=False)
    (a.output/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(manifest_sha256=sha(a.output/'manifest.json'),
                         cases=[r['case_id'] for r in rows]), indent=2))


if __name__ == '__main__':
    main()
