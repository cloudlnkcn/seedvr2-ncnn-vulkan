---
license: apache-2.0
library_name: ncnn
pipeline_tag: image-to-image
tags:
- seedvr2
- ncnn
- vulkan
- video-to-video
- fp16
base_model: ByteDance-Seed/SeedVR2-3B
---
# SeedVR2 3B ncnn — DiT FP16 weight storage

Ready-to-install image and temporal-video model packages for [SeedVR2 ncnn Vulkan](https://github.com/mingshi2333/seedvr2-ncnn-vulkan), a native C++20 application with a CLI, local Web interface and installable C++ SDK. These are converted `.param` / `.bin` models with constants and manifests. Downloading this package does not require running PyTorch or pnnx conversion.

## What is stored and what is computed

All 32 DiT blocks store linear matrix weights as IEEE FP16. ncnn expands them to FP32 when loading. **Activations and arithmetic remain FP32.** VAE, patch projection, biases, modulation constants and custom attention attributes remain unchanged. This is storage compression, not INT8, BF16 arithmetic or full FP16 computation; it does not imply half the VRAM or a speedup. Memory placement accounts for expanded weights before choosing GPU/RAM placement.

The application executes 36 component graphs: VAE encoder → patch-in → 32 DiT blocks → patch-out → VAE decoder. Its custom adaptive window attention preserves window indexing, Q/K normalization, multimodal 3D RoPE and text aggregation. Image and video packages share DiT weights but have different VAE graphs and reviewed profiles. See [architecture](https://github.com/mingshi2333/seedvr2-ncnn-vulkan/blob/main/docs/ARCHITECTURE.md).

| Package | Installed bytes | Approximate size |
| --- | ---: | ---: |
| Image | 10,395,017,619 | 10.40 GB |
| Short video | 11,052,868,862 | 11.05 GB |

The repository uses content-addressed objects: 53 unique objects total **11,397,496,168 bytes** (11.40 GB). The installer reconstructs ordinary model directories. Separate image/video installs use separate local copies; remote deduplication does not imply shared local disk usage. The [FP32-B packages](https://huggingface.co/akashimio/SeedVR2-3B-ncnn) remain available for reference use.

## Download, verify and run

Build the current native application first. From a clean source checkout, install the [documented system development dependencies](https://github.com/mingshi2333/seedvr2-ncnn-vulkan/blob/main/docs/TUTORIAL.md#1-从干净克隆开始), then run:

```sh
git clone https://github.com/mingshi2333/seedvr2-ncnn-vulkan.git
cd seedvr2-ncnn-vulkan
python3 tools/build_native.py --cli-only --jobs 2 --prefix dist/tutorial
python3 tools/download_models.py --precision dit-fp16 --kind image \
  --output dist/tutorial/models/image --run input.png --result results/image
```

Use an existing input path and a new result directory. Omit `--run` and `--result` to install only. `--plan` shows exact bytes without downloading. The download tool uses Python 3.12+ standard library, requires no Hugging Face account, resumes interrupted downloads, checks SHA-256 and performs native model validation. Original checkpoints, PyTorch and pnnx are not needed for this route. The native application is currently built from source; this model repository is not a portable binary distribution.

```sh
# Video: default output long side 128, up to 17 frames
python3 tools/download_models.py --precision dit-fp16 --kind video \
  --output dist/tutorial/models/video --run input.mp4 --result results/video --frames 17
# Offline reuse after installation
python3 tools/download_models.py --precision dit-fp16 --kind image \
  --output dist/tutorial/models/image --offline
# Native inference itself has no Python dependency
dist/tutorial/bin/seedvr2 run --model dist/tutorial/models/image \
  --input input.png --output results/another-image --size 256 --backend vulkan
```

For the local Web interface, build without `--cli-only`, install the models at the paths above, then run `dist/tutorial/bin/seedvr2-studio` and open the printed local URL. Rebuild older executables: the new model profiles must be recognized by the native package loader.

Downloads are pinned to model revision `5c17b05641fbc84f16752b2f3c59aa417cf2475b`; subsequent model-card edits do not change those bytes. The application’s reviewed catalogue is `docs/distribution/catalog.dit-fp16.v1.json`, SHA-256 `ecac033376e8b4a4ae61593fdbc990c53290b543048211962998083b6b23d7f1`. If downloading manually, the files under `objects/` require assembly by the installer; downloading only a `.bin` does not produce a complete package.

## Measured numerical differences

Linux x86_64, RTX 4060 Laptop 8 GiB, approximately 32 GiB host RAM. Fixed 3B, one step, CFG=1, seed 666, identical inputs and raw noise. Five complete executions checked all 73 model boundaries plus auxiliary input/output tensors; all measured tensors were finite. Historical FP32 thresholds remain descriptive and are not low-precision acceptance gates.

| Case | Backend | Decoded RGB fidelity PSNR vs native FP32 | Minimum SSIM |
| --- | --- | ---: | ---: |
| 256×256 natural image | Vulkan | 40.03 dB | 0.99158 |
| 128×80 motion, 9 frames | Vulkan | 61.96–64.32 dB | 0.99986 |
| 128×80 padded tail, 8 frames | Vulkan | 62.50–64.25 dB | 0.99987 |
| 128×80 artificial cut, 17 frames | Vulkan | 66.48–67.86 dB | 0.99993 |
| 256×256 natural image | CPU | 40.03 dB | 0.99158 |

These are fidelity comparisons against FP32, not restoration quality against ground truth. Fixed-target image PSNR was 20.0184 dB for FP32 and 20.0093 dB for DiT FP16 storage. Corresponding video means were 20.0078/20.0074, 19.6978/19.6987 and 23.2410/23.2414 dB. Three clips derive from one source; the cut is artificial. This is a development sample, not a representative quality benchmark. Frame residual changes are recorded separately from cuts and do not prove an absence of flicker.

[Full measurements, images, per-boundary errors, resource observations and limitations](https://github.com/mingshi2333/seedvr2-ncnn-vulkan/blob/main/docs/DIT-FP16-STORAGE.md). Existing single-step quality limitations remain documented; compression does not establish improved restoration quality.

## Delivery validation and limits

All uploaded assets were downloaded into an independent directory at the fixed revision and checked byte-for-byte by SHA-256. Both assembled model packages passed native identity/integrity verification. The installed SDK also completed a real Vulkan image restoration in a separate network namespace, with the source tree hidden and Unicode paths. All 77 diagnostic tensor hashes matched the frozen development FP16 run. A real local Web worker also restored an image from the downloaded package; all 39 lifecycle checks (including cancellation, result download and restart recovery) passed. Public access probes are recorded separately from the complete authenticated readback performed while staging.

The application accepts images up to long side 512 and short videos up to 17 frames / long side 128. This release’s full low-precision measurements are the sizes listed above. Video output is SDR MP4 without audio. It does not add long-video streaming, unbounded resolution, temporal cache, INT8/BF16 compute, or AMD/Intel hardware validation. A compatible native runtime and system ABI are required; isolated offline execution on the development Linux host is not a fresh-machine or cross-distribution portability test.

## Provenance and license

Official model: [ByteDance-Seed/SeedVR2-3B](https://huggingface.co/ByteDance-Seed/SeedVR2-3B), pinned revision `37255ff8cccfb01071b87f635a5948ca8d53117c`. Official code and converter/runtime identities are preserved in manifests and source lockfiles. See the included `LICENSE` and `NOTICE` and the native project’s [third-party notices](https://github.com/mingshi2333/seedvr2-ncnn-vulkan/blob/main/NOTICE). This community port is not an official ByteDance or Tencent release.

Feedback and technical discussion: [Tencent/ncnn Discussion #6991](https://github.com/Tencent/ncnn/discussions/6991).
