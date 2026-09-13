---
license: apache-2.0
base_model: ByteDance-Seed/SeedVR2-3B
language:
- en
- zh
tags:
- ncnn
- vulkan
- seedvr2
- image-restoration
- video-restoration
- fp32
pipeline_tag: video-to-video
---
# SeedVR2 3B — ncnn CPU / Vulkan FP32-B packages

Community conversion of **ByteDance-Seed/SeedVR2-3B** for the native [SeedVR2 ncnn Vulkan application](https://github.com/mingshi2333/seedvr2-ncnn-vulkan). This is a format/graph conversion with explicit FP32-B execution adaptations, **not a newly trained model or a low-bit quantization**, and is not an official ByteDance or Tencent release.

## Quick start with the native downloader

A smaller [DiT FP16 weight-storage variant](https://huggingface.co/akashimio/SeedVR2-3B-ncnn-dit-fp16) is also public (image 10.40 GB / video 11.05 GB). Its activations and arithmetic remain FP32; [complete drift and quality measurements](https://github.com/mingshi2333/seedvr2-ncnn-vulkan/blob/main/docs/DIT-FP16-STORAGE.md) are retained separately. This repository keeps the original FP32-B weights for reference use.

After installing the [system build dependencies](https://github.com/mingshi2333/seedvr2-ncnn-vulkan/blob/main/docs/TUTORIAL.md), clone the current native project and run from its root:

```sh
python3 tools/build_native.py --cli-only --jobs 2 --prefix dist/tutorial
python3 tools/download_models.py --precision fp32 --kind image \
  --output dist/tutorial/models/image --run input.png --result results/image
```

Use an existing input and a new result directory. For video use `--kind video`, `--output dist/tutorial/models/video`, an MP4 input and a separate result directory. Omit `--run` / `--result` to install only; `--plan` shows sizes, `--offline` verifies/reuses installed models. Downloads need only Python standard library; native inference needs no Python. No PyTorch/pnnx conversion is required. Keep different storage variants in different directories. For Web, build without `--cli-only` and launch `dist/tutorial/bin/seedvr2-studio` after installation.

[First run and offline transfer](https://github.com/mingshi2333/seedvr2-ncnn-vulkan/blob/main/docs/FIRST-RUN.md) · [Technical Discussion](https://github.com/Tencent/ncnn/discussions/6991).

## Download and use

Build the native application from the linked GitHub project, then run this command from its root (Python 3.12+ standard library only; no PyTorch or pnnx needed):

```sh
python3 tools/model_distribution.py install --catalog docs/distribution/catalog.v1.json --kind image --base-url https://huggingface.co/akashimio/SeedVR2-3B-ncnn/resolve/9371e381e3d5581c05a0934a518b14d8aa15698b --output models/image
```

For video use `--kind video --output models/video`. The download is pinned to commit `9371e381e3d5581c05a0934a518b14d8aa15698b` and every file is SHA-256 checked. The installer supports interrupted-download recovery. Keep the full installed model directory for offline native inference.

Every uploaded asset was downloaded to a separate directory and checked against the prepared bytes. Both installed model packages passed native identity/integrity checks. One real 256px JPEG run from the downloaded image package passed the unchanged full 73-boundary FP32-B reference comparison. The retained video numerical experiments are separate; this upload does not rerun or expand their scope.

## Architecture and package layout

The native CLI, local Web worker and C++20 SDK share one inference implementation. Each package contains 36 ncnn graphs: VAE encoder, patch-in, 32 DiT blocks, patch-out and VAE decoder. Custom adaptive window attention and temporal VAE layers require this project's runtime; these files are not generic stock-ncnn networks or Transformers checkpoints.

This repository stores 53 SHA-256-addressed objects (21,441,543,211 bytes). The image and video packages share their DiT data. `catalog.json` preserves the original manifests and maps objects back to the expected paths. The project installer reconstructs a normal image or video package, checks every file and finalizes the directory only after validation. **Do not pass `objects/` directly to the native application.**

- `objects/<sha256>`: model parameters, weights, constants and original manifests.
- `catalog.json`: complete installation catalogue, including both package identities.
- `reviewed-packages.json`: the reviewed payload identities; hashes do not certify numerical or perceptual quality.
- `model-sources.lock.json`: original checkpoint revisions and hashes.
- `LICENSE`: Apache-2.0 license text.

## Reproducible sources

- Official checkpoints: `ByteDance-Seed/SeedVR2-3B`, revision `37255ff8cccfb01071b87f635a5948ca8d53117c`.
- Official source: `ByteDance-Seed/SeedVR`, commit `e4de8c24441a67e1b7df56abea10645059bb1185`.
- Runtime ncnn: `3b7bdba7fc8aea8fd46779533eee027df77c639d`, with the project's documented host-buffer overlay.
- pnnx converter source: `6a1bf000f363714839a36793addc8c879d3d899e`.
- Source-first application and installation tools: GitHub commit `64cd6e54729014eb12604f734c562de1a03c5d59`.

The catalogue retains each original exporter's metadata, including its pnnx hash and PyTorch version. Keep this identity with bug reports. Follow the source tutorial to reproduce or modify the conversion; newly produced payloads are not automatically added to the native allowlist.

## Validated scope and limitations

SeedVR2 **3B, FP32-B, one step, CFG=1**, on Linux x86_64 CPU and NVIDIA Vulkan. FP32-B is an explicit numerical reference profile; this upload does not claim official CUDA BF16/Apex/FlashAttention parity.

The retained application experiments cover six complete trajectories, each with **73/73 tensor boundaries** checked at the unchanged diagnostic `atol=rtol=0.001`: natural 9-frame motion, 8-frame tail padding, 17-frame cuts, synthetic 17-frame CPU/Vulkan runs and a natural JPEG image. See the [numerical repair record](https://github.com/mingshi2333/seedvr2-ncnn-vulkan/blob/64cd6e54729014eb12604f734c562de1a03c5d59/docs/NUMERICS-REPAIR.md).

- Image output: long edge up to 512; video: up to 17 frames, long edge up to 128.
- Video output is SDR MP4 without audio; no long-video streaming/cache claim.
- Native package size is approximately 20.44 GB for image or 21.10 GB for video. Working memory and runtime depend on shape/backend; these file sizes are not VRAM requirements.
- On the three retained low-resolution development videos, both native and official FP32-B outputs scored below bicubic on the fixed-target quality metrics. Numerical parity does not imply improved restoration quality. See [all images, videos and metrics](https://github.com/mingshi2333/seedvr2-ncnn-vulkan#实测结果与对照图).
- More devices, higher resolution, official BF16 reference and representative temporal quality validation remain separate work.

## License and attribution

Original SeedVR and SeedVR2 models and source are credited to ByteDance Seed and their authors. The official model card specifies Apache-2.0. This package changes the serialization/graph representation and uses the project's documented FP32 adaptations and custom operators; the original model provenance and license are retained. See the [official model card](https://huggingface.co/ByteDance-Seed/SeedVR2-3B) and [project licensing notes](https://github.com/mingshi2333/seedvr2-ncnn-vulkan/blob/main/docs/LICENSING.md). No application binaries or media libraries are included here.
