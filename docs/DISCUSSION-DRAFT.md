# SeedVR2 3B image and short-video restoration with ncnn/Vulkan

Local engineering preview, 2026-09-08. This draft has not been published.

The project runs the official SeedVR2 3B weights through a native C++ pipeline: temporal VAE, patch projections, all 32 DiT blocks with adaptive window attention, the original single-step Euler endpoint, and VAE decoding. The local Web application, standalone CLI and installed C++ SDK share the same implementation. Python and pnnx are used for export and independent reference generation, not application inference.

The runtime is pinned to unmodified upstream ncnn `3b7bdba7fc8aea8fd46779533eee027df77c639d`. Existing exports retain their independently pinned pnnx source `6a1bf000f363714839a36793addc8c879d3d899e`. Official model code and checkpoint revisions are recorded in the repository lock files. Upgrading the runtime does not relabel old export evidence.

Adaptive windows are exported as a custom layer. Native CPU/Vulkan implementations preserve window gather/scatter, Q/K normalization, multimodal 3D RoPE, joint attention and equal averaging of the text result across windows. Standard attention work uses upstream ncnn SDPA. Four temporal VAE operators preserve the causal convolution, normalization and layout behavior required for joint clip processing.

Current application limits are image long sides of 64–512 pixels and the first 1–17 video frames at long sides of 64–128 pixels. Video output is an SDR MP4 without audio. Clips are processed jointly; this preview does not implement long-video streaming, temporal cache reuse, tiled high-resolution inference or the 7B model.

The latest natural JPEG example passes all 73 retained FP32-B tensor boundaries against the independent official reference with shared raw noise. Maximum output difference is one 8-bit value. That test also ran through an externally built C++ SDK consumer with networking isolated, the source tree hidden and Unicode installation/input/output paths. A JPEG decoding mismatch was found and corrected before this result; the earlier failure remains in the evidence directory.

Numerical limitations remain. The retained 6-frame 64-pixel clip passes 73/73 checks; a 17-frame 128-pixel Vulkan trajectory passes 60/73 and therefore fails the fixed diagnostic protocol. The retained CPU trajectory passes 70/73. Supplying identical official inputs to blocks 19 and 31 gives passing CPU/Vulkan component results, but this does not close the accumulated-error investigation. The reference profile is explicit CPU FP32-B, not the official BF16/Apex/FlashAttention execution path.

Correctness against the reference is also separate from restoration quality. On the single controlled natural-photo example, native and official outputs match closely, but RGB PSNR/SSIM against the target are 20.01 dB/0.6772, below the bicubic input baseline of 23.79 dB/0.7568. This is one synthetic degradation example, not a representative quality benchmark or model certification.

Weights are loaded one graph at a time. Four paired 17-frame runs on an RTX 4060 Laptop produced byte-identical saved tensors and MP4 output with buffered and mapped readers. Read-only mapping reduced sampled anonymous RSS, but did not show a speed benefit in these observations, so buffered loading remains the default. The measurements include package hashing and graph loading; desktop GPU use, temperature and file-cache state were not controlled. Allocator-specific activation and workspace peaks have not been measured.

The delivery includes model identity/integrity checks, preflight validation, progress and cancellation, an offline model-copy command, an installed CMake SDK target and Linux native CI configuration. Local checks include 6/6 CTest tests, 39/39 real image-job lifecycle checks and 49/49 real video-job checks. The remote CI matrix has not run; full-model real-device evidence is recorded separately from CI smoke tests.

The next concrete work is investigating the remaining 17-frame discrepancies, expanding natural-video quality and temporal-consistency coverage, and qualifying additional devices and distribution environments. Complete numerical reports, source identities, measured resource data and known limitations accompany the code. No full-model certificate or general portable-binary claim is made.
