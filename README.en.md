# SeedVR2 ncnn Vulkan

[中文](README.md) · [Step-by-step tutorial (Chinese)](docs/TUTORIAL.md) · [Delivery evidence](artifacts/2026-09-10/tutorial-v1/README.md) · [Contributing](CONTRIBUTING.md) · [License](LICENSE)

An independent, executable case study in porting **SeedVR2 3B** from its official
PyTorch implementation to a native C++20 application using **ncnn CPU/Vulkan**.
It covers pnnx export, custom adaptive window attention, temporal VAE, numerical
comparison, weight placement, an installed C++ SDK, a CLI, and a local Web interface.
This is not an official ByteDance or Tencent release.

**Status: 0.7.0 native preview.** Real image and bounded whole-clip inference work.
Full model quality certification, long-video processing and portable binary releases
remain unfinished. This is an advanced porting tutorial, not a general Vulkan course.

## Start without downloading model weights

On Ubuntu 24.04, install:

```sh
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build python3 python3-venv \
  pkg-config ripgrep libssl-dev libjpeg-dev libsqlite3-dev libvulkan-dev \
  glslang-dev glslang-tools spirv-tools mesa-vulkan-drivers vulkan-validationlayers \
  libavformat-dev libavcodec-dev libavutil-dev libswscale-dev libomp-dev \
  uuid-dev zlib1g-dev

git clone https://github.com/mingshi2333/seedvr2-ncnn-vulkan.git
cd seedvr2-ncnn-vulkan
python3 tools/build_native.py --cli-only --check
python3 tools/build_native.py --cli-only --jobs 2
dist/tutorial/bin/seedvr2 engine self-test --backend cpu
dist/tutorial/bin/seedvr2 engine devices
dist/tutorial/bin/seedvr2 engine self-test --backend vulkan --gpu 0
```

The helper prepares hash-locked dependencies, builds, runs small native tests and
installs the CLI/worker/SDK to `dist/tutorial`. It stops at the first failure and does
not download checkpoints. `--plan` prints its commands; `--offline` requires cached
dependency archives. A missing Vulkan device is not a successful GPU test.

The default build also includes the local Web application and requires Node.js 24+
and npm; omit `--cli-only` when those prerequisites are installed. Runtime inference
does not require Python, Node.js, a cloud service or a CDN.

## Learning sequence

1. Build and compare the embedded regular/shifted AWA fixtures.
2. Generate independent official FP32-B references, export a preserved pnnx `moduleop`,
   lower only checked attributes, and compare ncnn CPU/Vulkan outputs.
3. Download the pinned official model and compare one real DiT block on identical inputs.
4. Assemble the image VAE, patch projections and all 32 DiT blocks into a 36-graph package.
5. Add the temporal VAE and whole-clip 3D attention, preserving causal and first-frame semantics.
6. Inspect numerical boundaries, SDK integration, installation, memory measurements and failures.

Commands, expected outcomes and source pointers are in [the tutorial](docs/TUTORIAL.md).
[Architecture](docs/ARCHITECTURE.md) maps the SDK, CLI, worker, model mathematics,
package validation and resource lifetime to their implementation files.

## Models and inference

```sh
python3 tools/prepare_models.py --list
python3 tools/prepare_models.py
```

This fetches four files (about 14.57 GB) from the pinned official
[ByteDance-Seed/SeedVR2-3B repository](https://huggingface.co/ByteDance-Seed/SeedVR2-3B/tree/37255ff8cccfb01071b87f635a5948ca8d53117c)
into `.cache/models`. Downloads support resume and SHA-256 verification. Cached valid
files are reused; `--offline` verifies final cached files without network access.
These are PyTorch checkpoints, **not ready-to-run ncnn packages**.

Export uses a separate Python 3.13 / PyTorch 2.9.0+cpu environment and a separately
pinned pnnx build. Follow the [export lessons](docs/TUTORIAL.md#2-看清自定义-awa-如何导出).
Image graphs total about 20.44 GB; video graphs total about 21.10 GB, with unchanged
DiT files reused through hard links during export. Checkpoints, intermediate files,
diagnostic tensors and independent installation copies require additional disk space.
There is no promise of a cross-device peak RAM or disk budget for conversion.

Once those lessons have produced the packages:

```sh
dist/tutorial/bin/seedvr2 run --model .cache/tutorial/image-package \
  --input tests/fixtures/natural/astronaut-degraded.jpg \
  --output .cache/tutorial/image-result --size 256 --backend vulkan --gpu 0

dist/tutorial/bin/seedvr2 run-video --model .cache/tutorial/video-package \
  --input docs/examples/video-demo-input.mp4 --output .cache/tutorial/video-result \
  --frames 6 --size 64 --backend vulkan --gpu 0
```

Output directories must be new or empty. Results include PNG/MP4 and `run.json`.
Add `--check` for preflight; actual execution verifies all model payload files.
Unknown package identities are rejected. Review export differences rather than
bypassing `policies/reviewed-packages.json`.

For the Web build, start `dist/tutorial/bin/seedvr2-web --model PATH --video-model PATH`
and open the printed loopback URL. The installed `seedvr2-studio` launcher expects
packages under `models/image` and `models/video`, or the corresponding environment
variables. It does not automatically download or export missing weights.

## Supported scope and evidence

| Area | Current scope |
| --- | --- |
| Images | PNG/JPEG, output long side 64–512, multiples of 16; RGB8 PNG |
| Video | First 1–17 frames jointly, output long side 64–128; 8-bit SDR, square pixels, no output audio |
| Runtime | Linux x86_64; CPU and RTX 4060 Laptop used for retained full-model tests |
| Memory | Per-graph automatic device/host weight placement; no activation offload or automatic OOM retry |
| Interface | Shared C++ SDK, standalone CLI, embedded React/TypeScript/Ant Design local Web |
| Not implemented | 7B, streaming cache, long-video chunking, HDR, large-resolution tiling, full model quality certification |

Retained 0.7.0 evidence includes 14 native tests, 40 AWA comparisons, 27 real-component
comparisons, two 73-boundary video comparisons, and a 73-boundary natural-image comparison.
These counts describe the [recorded candidate](artifacts/2026-09-08/memory-v1/README.md),
not every future checkout. Historical failures and invalidated SDK evidence remain visible.
The natural-image quality example has a retained negative result; numerical parity is
not perceptual-quality acceptance.

[GitHub Actions](https://github.com/mingshi2333/seedvr2-ncnn-vulkan/actions/workflows/native.yml)
builds/tests native Linux and software Vulkan without full model weights. Full 3B GPU
validation is a separate real-device protocol. Historical raw model/tensor paths in
reports belong to the developer's machine; the large files are not in Git.

## License

Original code and documentation: [Apache-2.0](LICENSE). Upstream code and fixtures
retain their original terms and attribution in [NOTICE](NOTICE).
The tested FFmpeg/libx264 configuration uses GPL components; distributing a combined
binary requires complying with the actual dependency licenses. See
[source and dependency licensing](docs/LICENSING.md).
This repository publishes source and small diagnostic fixtures, not model downloads or
a prebuilt portable application.
