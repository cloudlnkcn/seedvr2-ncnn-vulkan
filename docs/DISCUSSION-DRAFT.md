# SeedVR2 3B 图片与短视频修复：原生 C++ / ncnn / Vulkan 移植

> 已于 2026-09-10 发布到 Tencent/ncnn 的 Show and tell：[Discussion #6991](https://github.com/Tencent/ncnn/discussions/6991)。下文为已发布正文，保留原文件路径以兼容已有链接；数值修复及远端 CI 在提交 `7e56479` 核验。[发布记录](DISCUSSION-PUBLICATION.json)。

我做了一个 [SeedVR2 ncnn Vulkan](https://github.com/mingshi2333/seedvr2-ncnn-vulkan) 项目，把官方 SeedVR2 3B 接到原生 C++20 应用中。输入图片或短片，输出 PNG / MP4 和 JSON 报告；提供独立 CLI、本地 Web 和可安装的 C++ SDK。模型准备完成后可以离线使用，推理无需 Python。

项目是一个完整移植案例：从官方权重、pnnx 导出和自定义算子，到原生流水线、相同输入验证和应用交付。当前验证配置为 **3B、FP32-B、单步、CFG=1**。

## 实现结构

CLI 和 Web worker 调用同一个 SDK。模型拆为 **VAE encoder → patch-in → 32 个 DiT block → patch-out → VAE decoder**，共 36 个图；C++ 负责噪声、条件、Euler 和媒体处理，权重按图装载和释放。

AWA 在 pnnx 导出时保留为自定义模块，原生实现根据实际时空网格生成裁剪窗口，再执行 Q/K RMSNorm、3D RoPE、视频/文本联合注意力和窗口文本平均。Vulkan 复用锁定 ncnn 的 SDPA QK/PV 着色器，并补充 FP32 softmax 和 gather/scatter。时序 VAE 保留因果卷积、首帧规则与尾帧补齐，视频按整个短片联合处理。

运行库固定为 ncnn `3b7bdba7fc8aea8fd46779533eee027df77c639d`，pnnx 导出单独固定为 `6a1bf000f363714839a36793addc8c879d3d899e`。来源、分配器适配和实际加载 SDK 的哈希都写入报告。[架构与源码导航](https://github.com/mingshi2333/seedvr2-ncnn-vulkan/blob/main/docs/ARCHITECTURE.md)。

## 对照结果

Linux x86_64，RTX 4060 Laptop 8 GiB，约 32 GiB RAM。对照使用锁定官方数学实现的 CPU FP32-B 参考，共享**原始后验噪声和扩散噪声**，逐元素检查全部 73 个张量边界。

| 测试输入 | 后端 / 输出 | 完整数值结果 |
| --- | --- | --- |
| 自然运动 9 帧 | Vulkan，128×80 | 原 63/73 → **73/73** |
| 尾帧补齐 8 帧 | Vulkan，128×80 | 原 71/73 → **73/73** |
| 人工镜头切换 17 帧 | Vulkan，128×80 | **73/73** |
| 合成运动 17 帧 | CPU / Vulkan，128×128 | 两条轨迹均 **73/73** |
| 自然 JPEG | Vulkan，256×256 | **73/73**，RGB8 最大差 1 |

本轮没有修改官方参考、权重、原始噪声或 `atol=rtol=0.001` 的历史全链门槛。误差定位用相同输入的真实组件回放，修复了 VAE 分块卷积和 shortcut bias 顺序、归一化、注意力中的 FP32 舍入，以及 DiT 长点积的误差补偿。还抓到并修复了 CPU 留出视频的 72/73 回归。失败候选和诊断记录都保留在[数值修复报告](https://github.com/mingshi2333/seedvr2-ncnn-vulkan/blob/main/docs/NUMERICS-REPAIR.md)。

下面四列依次为 **bicubic → 原生 ncnn/Vulkan → 官方 FP32-B → 固定目标**；来自实际输出，没有额外增强。

![9 帧自然运动的真实对照](https://raw.githubusercontent.com/mingshi2333/seedvr2-ncnn-vulkan/main/artifacts/2026-09-10/video-numerics-v2/quality/motion-9/comparison.png)

完整张量对照通过之后，画质仍需单独看。三段低分辨率开发视频的原生 PSNR 为 **20.01 / 19.70 / 23.24 dB**，bicubic 为 **26.93 / 26.82 / 26.80 dB**；SSIM 也较低，官方 FP32-B 在这些样例上有相同表现。这些素材来自同一来源和固定人工退化，不是代表性画质基准。[全部图、视频与逐帧数据](https://github.com/mingshi2333/seedvr2-ncnn-vulkan#实测结果与对照图)。

## 使用与当前边界

[教程](https://github.com/mingshi2333/seedvr2-ncnn-vulkan/blob/main/docs/TUTORIAL.md)从干净克隆和不需要大权重的算子测试开始，再进入官方权重下载、pnnx 转换和真实模型运行。下载脚本支持固定 revision、续传和 SHA-256 校验；下载的是官方权重，还需要转换。仓库暂未提供预编译 Release 或可直接下载的完整 ncnn 模型包。

应用已包含参数预检、Unicode 路径、进度与取消、模型身份/完整性校验、离线模型复制和安装 SDK。本机原生测试 **36/36**。Mesa 的小型测试为 **10 项通过、12 项能力跳过**：本机 llvmpipe 不保留这些补偿算子要求的 FMA 残差，程序会在模型加载前明确拒绝。能力探测和数值验收分开；大模型实机证据不由 CI 小型测试替代。

提交 `7e56479` 的 Ubuntu GCC CLI、Clang CLI、GCC Web 远端 CI 全部通过：每项原生套件 **24 项通过、12 项能力跳过**，Web 另有 **54/54** 接口检查。[下载归档的原始报告](https://github.com/mingshi2333/seedvr2-ncnn-vulkan/blob/main/artifacts/2026-09-10/github-ci-numerics-v2/README.md)保留依赖日志、安装验证和跳过原因；远端 CI 没有运行完整 3B 权重。

当前图片输出长边 ≤512；视频 ≤17 帧、长边 ≤128，输出无音轨 SDR MP4。官方 CUDA BF16/Apex/FlashAttention 默认路径、更高分辨率、长片、音轨、更多设备和代表性时序画质仍是后续工作。[具体缺口与验收范围](https://github.com/mingshi2333/seedvr2-ncnn-vulkan/blob/main/docs/CURRENT-GAPS.md)。

希望交流两个实现问题：如何在不同 Vulkan 驱动上可靠地保留补偿算法需要的 FMA 行为；以及在保留同输入数值证据的前提下，怎样减少这类逐图大模型执行的权重装载和图间传输成本。欢迎带设备、驱动、参数与 `run.json` 的复现报告。
