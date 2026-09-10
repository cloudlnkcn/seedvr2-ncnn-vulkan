# SeedVR2 ncnn Vulkan

[中文](README.md) · [Measured results](#measured-results-and-visual-comparisons) · [Tutorial (Chinese)](docs/TUTORIAL.md) · [Architecture](docs/ARCHITECTURE.md) · [Contributing](CONTRIBUTING.md) · [License](LICENSE)

A native **C++20 / ncnn CPU/Vulkan** port of the official **SeedVR2 3B** image and short-video restoration model. The **standalone CLI, local Web application and installed C++ SDK** share one inference implementation. The tutorial covers pnnx export, custom adaptive window attention, temporal VAE and component-by-component validation.

**Status: 0.7.0 native preview; full model certification remains incomplete.** Image output long sides reach 512 pixels; video is limited to 17 frames and 128-pixel long sides, producing SDR MP4 without audio. Real inference works, with the numerical and quality limitations shown below. This is an advanced porting case study for readers with C++, PyTorch and basic Vulkan experience, not an official ByteDance or Tencent release.

The application includes model identity/integrity checks, preflight, progress/cancellation, offline model copying and per-graph device/host weight placement. React / TypeScript / Ant Design is embedded in the native Drogon host; inference needs no Python, Node.js or cloud service. See [first use and offline transfer](docs/FIRST-RUN.md) and [memory-policy measurements](docs/MEMORY-VALIDATION.md).

## Measured results and visual comparisons

Recorded **2026-09-10**, using the frozen **0.7.0 native CLI/SDK**, SeedVR2 **3B / single-step FP32-B**, **Linux x86_64 / RTX 4060 Laptop 8 GiB**, 32 GiB host RAM. Each clip starts as **64×40, 8 fps** and produces **128×80** output. The cases use one natural source with fixed synthetic degradation; the cut is an artificial splice. They are development examples, not a representative benchmark.

**All three clips complete inference. Two full numerical comparisons still fail, and fixed-target quality metrics are below bicubic in all three cases.** The examples below show those outcomes directly.

The four columns are **bicubic input baseline → native ncnn/Vulkan → official FP32-B → fixed target**. These are the original retained comparison images, rendered from pre-codec RGB8 with no extra enhancement. `f0` is the first frame. The target comes from an already compressed source, not camera-original ground truth.

### Natural motion · 9 frames

Frames 0, 4 and 8; no temporal padding.

![motion-9 — bicubic, native, official FP32-B and target at matching frames](artifacts/2026-09-10/bounded-video-v1/quality/motion-9/comparison.png)

[Input clip](tests/fixtures/video-bounded/motion-9/input.mp4) · [Native output MP4](artifacts/2026-09-10/bounded-video-v1/motion-9-output.mp4) · [Official preview MP4](artifacts/2026-09-10/bounded-video-v1/quality/motion-9/official.mp4) · [Per-frame data](artifacts/2026-09-10/bounded-video-v1/quality/motion-9/report.json)

### Tail padding · 8 frames → 9 → 8

Frames 0, 4 and 7. The last frame is repeated for model input, then the output is cropped back to 8 frames.

![padding-8 — bicubic, native, official FP32-B and target at matching frames](artifacts/2026-09-10/bounded-video-v1/quality/padding-8/comparison.png)

[Input clip](tests/fixtures/video-bounded/padding-8/input.mp4) · [Native output MP4](artifacts/2026-09-10/bounded-video-v1/padding-8-output.mp4) · [Official preview MP4](artifacts/2026-09-10/bounded-video-v1/quality/padding-8/official.mp4) · [Per-frame data](artifacts/2026-09-10/bounded-video-v1/quality/padding-8/report.json)

### Artificial cut · 17 frames

Frames 0, 7, 8 and 16. The splice is immediately before frame 8, so both sides of the cut remain visible.

![cut-17 — bicubic, native, official FP32-B and target at matching frames](artifacts/2026-09-10/bounded-video-v1/quality/cut-17/comparison.png)

[Input clip](tests/fixtures/video-bounded/cut-17/input.mp4) · [Native output MP4](artifacts/2026-09-10/bounded-video-v1/cut-17-output.mp4) · [Official preview MP4](artifacts/2026-09-10/bounded-video-v1/quality/cut-17/official.mp4) · [Per-frame data](artifacts/2026-09-10/bounded-video-v1/quality/cut-17/report.json)

### Numerical and output checks

| Case | Full tensor boundaries | RGB8 max difference¹ | Native graphs | Raw reference |
| --- | --- | --- | --- | --- |
| motion-9 | **63/73 · FAIL** | 1 | 36/36 Vulkan | [JSON](artifacts/2026-09-10/bounded-video-v1/motion-9-reference.json) |
| padding-8 | **71/73 · FAIL** | 1 | 36/36 Vulkan | [JSON](artifacts/2026-09-10/bounded-video-v1/padding-8-reference.json) |
| cut-17 | **73/73 · PASS for this case** | 1 | 36/36 Vulkan | [JSON](artifacts/2026-09-10/bounded-video-v1/cut-17-reference.json) |


¹ Native versus official, before video encoding, on a 0–255 channel scale. All three final `decoded` FP32 boundaries pass; this does not waive the earlier failures. Output frame count, dimensions, timestamps and no-audio checks pass. All graph layers execute on Vulkan with no CPU fallback; retained logs contain no Vulkan validation errors. The 73 boundaries use the unchanged `atol=rtol=0.001` diagnostic tolerance.

### Restoration quality

Each cell is **PSNR (dB) / RGB SSIM**, averaged over the real output frames, with no border crop. Higher means closer to this fixed target. Padded frames are excluded.

| Case | Bicubic baseline | Native ncnn/Vulkan | Official FP32-B |
| --- | --- | --- | --- |
| motion-9 | 26.93 / 0.8560 | 20.01 / 0.6187 | 20.01 / 0.6187 |
| padding-8 | 26.82 / 0.8539 | 19.70 / 0.5973 | 19.70 / 0.5973 |
| cut-17 | 26.80 / 0.8405 | 23.24 / 0.7686 | 23.24 / 0.7686 |


**Both native and official FP32-B outputs score below bicubic on these three examples.** The panels show changes in beak and feather detail relative to the target. Native and official values are calculated separately and look equal at this display precision. This is a retained negative result under the stated low-resolution setup; it is not an evaluation of the official default BF16/FlashAttention path. No post-hoc quality pass threshold is used.

### Runtime and memory

| Case | Native wall time (s) | Sampled process RSS (GiB) | Whole-card GPU use (MiB) |
| --- | --- | --- | --- |
| motion-9 | 28.09 | 1.184 | 3633 |
| padding-8 | 27.62 | 1.126 | 4202 |
| cut-17 | 31.31 | 0.893 | 5521 |


One sequential run per case, including package hashing, weight loading and computation; no speedup claim. Whole-card GPU memory includes desktop use and is not process-exclusive VRAM. RSS is not an allocation-class breakdown of weights, activations and workspace. Full sampling data is retained in [the summary](artifacts/2026-09-10/bounded-video-v1/summary.json).

### Operator checks and failure localization

| Check | Result | Scope |
| --- | --- | --- |
| AWA CPU / Vulkan | 8/8 | 4 pnnx exports; 16 output tensors; max abs 1.61e-6 |
| motion-9 · blocks 19–31 | 26/26 | Official inputs at every block |
| padding-8 · blocks 15–17 | 6/6 | Official inputs at every block |
| motion-9 · all 32 DiT blocks | 64/64 | Official initial tensors; subsequent inputs remain native |
| motion-9 · block 19 replay | **FAIL reproduced** | Both output tensors match the retained native failure byte for byte |


The AWA grids are `(3,5,8)` and `(5,5,8)`, each regular/shifted with 20 heads and 58 text tokens, using synthetic QKV. The DiT isolation checks use real checkpoint weights. These interventions support amplification of differences entering DiT, but do not isolate one upstream operator or fix the original 63/73 and 71/73 trajectories.

[Protocol and reproduction](docs/BOUNDED-VIDEO-RESULTS.md) · [Reports and failure logs](artifacts/2026-09-10/bounded-video-v1/README.md) · [Fixture provenance](tests/fixtures/video-bounded/SOURCE.md) · [Machine-readable results](artifacts/2026-09-10/bounded-video-v1/summary.json)

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
validation is a separate real-device protocol. The Ubuntu GCC CLI, Clang CLI and GCC
Web jobs passed at the [recorded CI commit](artifacts/2026-09-10/github-ci-v1/README.md);
raw reports and the artifact-log retention fix are retained in that archive.
Historical raw model/tensor paths in
reports belong to the developer's machine; the large files are not in Git.

## License

Original code and documentation: [Apache-2.0](LICENSE). Upstream code and fixtures
retain their original terms and attribution in [NOTICE](NOTICE).
The tested FFmpeg/libx264 configuration uses GPL components; distributing a combined
binary requires complying with the actual dependency licenses. See
[source and dependency licensing](docs/LICENSING.md).
This repository publishes source and small diagnostic fixtures, not model downloads or
a prebuilt portable application.
