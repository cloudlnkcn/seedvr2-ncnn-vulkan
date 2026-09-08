# SeedVR2 → ncnn / Vulkan: image and short-video preview with local Studio

Build: 0.5.0-video-preview
Pinned official ncnn commit: 6a1bf000f363714839a36793addc8c879d3d899e
Complete single-image restoration: implemented (preview, output long side up to 512). Short-video restoration: implemented for the first 1..17 frames, output long side up to 128, no audio or streaming cache. Model certification: not issued.

## Available implementation

- Local React / TypeScript / Ant Design workspace served by Drogon; independent C++ CLI.
- Complete native image pipeline: VAE encoder, posterior sampling/scaling, patch projections, all 32 DiT blocks, Euler endpoint and VAE decoder. Weights are loaded one block at a time.
- Isolated worker, SQLite job/event journal, bounded queue, cancellation and restart recovery.
- Image/video import, aligned image comparison, synchronized video players with frame stepping, PNG and MP4 export.
- Temporal VAE with preserved 3D kernels, framewise normalization/attention, temporal pixel shuffle, and native Vulkan dispatch.
- ncnn linked: true. AWA CPU: true; AWA Vulkan: true.
- Embedded offline AWA self-test: true; real local device detection and saved numerical reports.
- Single-frame VAE diagnostics: true; complete individual DiT block diagnostics: true.
- pnnx submodel export tools: true. Reference profile: FP32-B, adapted from pinned official code with FP32 math attention; not the official BF16/FlashAttention execution profile.
- Exported graphs and input tensors are hashed; submodel execution explicitly dispatches each layer to the requested backend.
- Model evidence auditing recomputes file SHA-256 and raw tensor differences. Diagnostic PASS never issues a model certificate.

## Selected media task

Job: bc4fe196bf90bda25872d1cf23f93309
Status: SUCCEEDED
Input: video-demo-17.mp4 (64 x 64)
Backend: vulkan; seed: 666
Output: 128 x 128; total wall time: 49.1 seconds, including model hashing/loading and transfers.
Output SHA-256: 3200b20ddb231ecb49142d36b26d491b8006ab7607638f913aa3d4274263da9e
Model package SHA-256: fe38835c367a2a389c689d89ce85c952b0e347d1edfd879b008d796a65374028
Recorded ncnn commit: 6a1bf000f363714839a36793addc8c879d3d899e
Recorded worker build: 0.5.0
Worker executable SHA-256: aab2315e53b19ad59e737c8d71ebdb73f0ba5fe6ffee5ba40f82a8a7e6439ce2

This is a recorded local run, not a speed benchmark or a model certificate. It uses 17 video frames, a whole-clip causal VAE and 3D adaptive windows, FP32, one Euler endpoint, CFG 1 and fixed positive conditioning. The application uses a versioned native noise generator; same-seed identity with PyTorch is not claimed.

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

- One 6-frame 64-pixel clip passed 73/73 FP32-B checkpoints. The 17-frame 128-pixel clip failed strict intermediate checks (Vulkan 60/73, CPU 70/73); final decoded tensors passed in both. Inspect docs/video-validation.md for the exact builds and retained failures. These synthetic clips are not a quality benchmark.
- Streaming causal cache, video chunk boundaries, audio retention and long-video consistency.
- FP16/BF16 accuracy calibration, representative quality checks, measured memory budgets and performance profiling.
- Larger image sizes, additional image metadata handling and portable worker adapters.
- Complete evidence bundle, frozen acceptance thresholds and platform qualification.

This local draft distinguishes compiled capabilities from the selected actual test record. Complete-image diagnostics are documented in docs/image-validation.md. Historical submodel reports are in docs/awa-export-validation.json, docs/vae-image-validation.json and docs/dit-block-validation.json; review their exact scope before attaching them. Only an explicitly selected media run contributes timing. No memory benchmark or image-quality acceptance is claimed. Downloading this draft does not publish it.
