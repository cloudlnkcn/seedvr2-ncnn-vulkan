# Fixed bounded-video fixtures

Derived from ImageIO's [cockatoo.mp4](https://raw.githubusercontent.com/imageio/imageio-binaries/40e8f791aef186e9f051d60a2a1ee17319190127/images/cockatoo.mp4)
at commit `40e8f791aef186e9f051d60a2a1ee17319190127`.
Source SHA-256: `5fde35f5a288ca86e216d2dc28188ab64b4560d3021f273faefdf0de80f38aa5`.
[ImageIO's standard-media documentation](https://imageio.readthedocs.io/en/stable/user_guide/standardimages.html)
lists these media as public domain to the best of its knowledge. This attribution
records that statement; it is not an independent rights certification.

`manifest.json` is the exact pre-execution declaration. Historical absolute paths
inside recorded FFmpeg commands identify the original preparation; `input.path`
and `target.path` resolve relative to this directory for portable reuse.

Each case contains a small degraded H.264 `input.mp4` (64×40, 8 fps, no audio)
and fixed `target.rgb` (interleaved frame/height/width/RGB uint8, 128×80).
The target is derived from an already compressed video, not camera-original truth.
Frame selection, artificial cut location, degradation, tool versions and all file
hashes are declared in the manifest. Its SHA-256 is
`45ceb036bd5546304d44bdf3d50d0bb85d55bbd531fc248851b3c8d62f4b47e8`.

These three cases share one source and deliberately fixed synthetic degradation.
They are development fixtures, not a held-out quality benchmark. To regenerate
from the pinned source, use `tools/prepare_video_validation.py`; different FFmpeg
or encoder builds can change bytes. Use the committed payloads to reproduce the
same input identities. See [the results and commands](../../../docs/BOUNDED-VIDEO-RESULTS.md).
