# SeedVR2 → ncnn / Vulkan: complete image pipeline and local Studio

Build: 0.4.0-image-preview
Pinned official ncnn commit: 6a1bf000f363714839a36793addc8c879d3d899e
Complete single-image restoration: implemented (preview, output long side up to 512). Video restoration: not implemented. Model certification: not issued.

## Available implementation

- Local React / TypeScript / Ant Design workspace served by Drogon; independent C++ CLI.
- Complete native image pipeline: VAE encoder, posterior sampling/scaling, patch projections, all 32 DiT blocks, Euler endpoint and VAE decoder. Weights are loaded one block at a time.
- Isolated worker, SQLite job/event journal, bounded queue, cancellation, restart recovery, image import, aligned result comparison and PNG export.
- ncnn linked: true. AWA CPU: true; AWA Vulkan: true.
- Embedded offline AWA self-test: true; real local device detection and saved numerical reports.
- Single-frame VAE diagnostics: true; complete individual DiT block diagnostics: true.
- pnnx submodel export tools: true. Reference profile: FP32-B, adapted from pinned official code with FP32 math attention; not the official BF16/FlashAttention execution profile.
- Exported graphs and input tensors are hashed; submodel execution explicitly dispatches each layer to the requested backend.
- Model evidence auditing recomputes file SHA-256 and raw tensor differences. Diagnostic PASS never issues a model certificate.

## Selected image task

Job: d3dec86e500d2010e302a6322c605abb
Status: SUCCEEDED
Input: image-demo-input-128.png (128 x 87)
Backend: vulkan; seed: 666
Output: 512 x 336; total wall time: 63.4 seconds, including model hashing/loading and transfers.
Output SHA-256: a9660bfcf638264117a3c8502772786fdfc699940d8b455da46819f2cff1e6bb
Model package SHA-256: 64c7c5a48bf6fffc193435535c39c6f559be71059e8510778c7eb3068b851298
Recorded ncnn commit: 6a1bf000f363714839a36793addc8c879d3d899e
Recorded worker build: 0.4.0
Worker executable SHA-256: 97143a175bfbdb23b3f1b9b26c4114aaa5ec07a9736ae0d22f4c3d8fff48f1cb

This is a recorded local run, not a speed benchmark or a model certificate. It uses one frame, FP32, one Euler endpoint, CFG 1 and fixed positive conditioning. The application uses a versioned native noise generator; same-seed identity with PyTorch is not claimed.

## Model acceptance

Policy: seedvr2-3b-validation-v1-draft
Policy SHA-256: 9ecd258361bd50fc9455985e0c8e47044fd17e6a4fd061c236cd7bb0c267c8c9
Calibration: NOT_FROZEN

- M01 权重与模型身份: MISSING_EVIDENCE
- M02 导出与算子覆盖: MISSING_EVIDENCE
- M03 独立参考与随机输入: MISSING_EVIDENCE
- M04 AWA 全计算: MISSING_EVIDENCE
- M05 VAE 编码与解码: MISSING_EVIDENCE
- M06 DiT 32 层中间结果: MISSING_EVIDENCE
- M07 整网端到端一致性: MISSING_EVIDENCE
- M08 视频时间与分块一致性: MISSING_EVIDENCE
- M09 真实画质与保留集: MISSING_EVIDENCE
- M10 Vulkan 与精度配置: MISSING_EVIDENCE
- M11 资源与稳定性: MISSING_EVIDENCE
- M12 可复现模型验收档案: MISSING_EVIDENCE

## Remaining work

- Temporal VAE, causal cache, video chunk boundaries and long-video consistency.
- FP16/BF16 accuracy calibration, representative quality checks, measured memory budgets and performance profiling.
- Larger image sizes, additional image metadata handling and portable worker adapters.
- Complete evidence bundle, frozen acceptance thresholds and platform qualification.

This local draft distinguishes compiled capabilities from the selected actual test record. Complete-image diagnostics are documented in docs/image-validation.md. Historical submodel reports are in docs/awa-export-validation.json, docs/vae-image-validation.json and docs/dit-block-validation.json; review their exact scope before attaching them. Only an explicitly selected image run contributes timing. No memory benchmark or image-quality acceptance is claimed. Downloading this draft does not publish it.
