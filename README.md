# SeedVR2 ncnn Vulkan

[English](README.en.md) · [架构设计](#架构设计) · [实测对照](#实测结果与对照图) · [从零开始的教程](docs/TUTORIAL.md) · [源码导航](docs/ARCHITECTURE.md) · [ncnn Discussion](https://github.com/Tencent/ncnn/discussions/6991) · [贡献方法](CONTRIBUTING.md) · [许可证](LICENSE)

将官方 **SeedVR2 3B** 移植为使用 **ncnn CPU/Vulkan** 的原生 C++20 图片与短视频修复应用。提供 **独立 CLI、本地 Web 和可安装的 C++ SDK**，共用同一推理实现；教程覆盖 pnnx 导出、自定义 adaptive window attention、时序 VAE 和逐组件验证。

**已验证模型与配置：SeedVR2 3B，FP32-B、单步 CFG=1。** 本轮修复后，三段原始短片、留出的 17 帧合成视频 CPU/Vulkan 和 256×256 自然图片，**六条轨迹均通过官方参考的 73/73 项张量边界对照**。原来的自然运动 63/73、尾帧补齐 71/73 数值失败已关闭；权重、官方参考、原始噪声和误差门槛保持不变。[修复过程](docs/NUMERICS-REPAIR.md) · [完整记录](artifacts/2026-09-10/video-numerics-v2/summary.json)。

当前支持图片输出长边最高 512，视频最多 17 帧、长边最高 128，输出为无音轨 SDR MP4。项目面向已有 C++、PyTorch 和基本 Vulkan 经验的读者，不属于 ByteDance 或 Tencent 官方发行。

支持模型身份/完整性校验、参数预检、进度与取消、离线模型复制、逐图 GPU/RAM 权重选择。React / TypeScript / Ant Design 界面内嵌于 Drogon C++ 服务；运行时无需 Python、Node 或云服务。首次使用见[安装与离线搬移](docs/FIRST-RUN.md)，内存策略及代价见[内存验证](docs/MEMORY-VALIDATION.md)。

项目改进按[来源 → 核对 → 知识页 → 检索与检查 → 验证回写](docs/wiki/README.md)推进；[同类设计对照](docs/wiki/synthesis/design-comparison.md)固定到具体源码版本。已增加[转换模型分发工具](docs/distribution/README.md)：图片与视频包按内容去重约 21.44 GB，可从离线目录或显式镜像安装；公共模型镜像和通用预编译程序仍待发布。

## 架构设计

架构围绕三个实际需求展开：**同一模型能从界面、命令行和外部程序调用；完整权重包能在有限显存设备上分阶段执行；每个结果都能追溯到输入、模型与运行实现。** 因此应用采用共享原生核心、本地任务服务和离线转换工具三部分。下面描述的是当前代码已经实现的结构。

### 应用分层：三个入口，一套原生计算

```mermaid
flowchart TB
    Web["本地 Web · React / TypeScript"] --> Service["Drogon HTTP + 有界任务队列"]
    Service <--> Store[("SQLite · 任务 / 事件 / 文件引用")]
    Service --> Worker["独立 worker 进程"]
    CLI["独立 CLI · CLI11"] --> SDK["公共 C++ SDK · seedvr2::restore"]
    App["外部 C++ 应用"] --> SDK
    Worker --> SDK
    SDK --> Pipeline["参数预检 / 包校验 / 图片或短片编排"]
    Package[("模型包 · manifest / 图 / 常量")] --> Pipeline
    Pipeline --> Executor["逐图执行器 · 权重预算 / 装载 / 执行 / 释放"]
    Executor --> Runtime["ncnn CPU / Vulkan · 标准层 + 自定义 AWA / 时序 VAE"]
```

Web 提交任务后，C++ 服务把请求和事件写入工作区，再启动 worker 调用 SDK。浏览器通过任务接口读取进度和结果；刷新页面不会重新启动推理。CLI 和外部 C++ 应用直接调用同一个 SDK，输出媒体与 `run.json`，可以独立于 Web 使用。

| 层次 | 负责什么 | 为什么这样划分 / 源码入口 |
| --- | --- | --- |
| 使用入口 | 文件、参数、进度和结果展示 | 界面不实现模型数学；[CLI](apps/cli/main.cpp)、[Web](apps/studio/src) 和 [SDK 示例](examples/sdk) 复用同一计算入口 |
| 本地任务服务 | 队列、worker 监督、取消、持久事件和异常中断记录 | 独立进程让服务能处理推理失败；最多 8 个未完成任务、同时执行 1 个，约束资源竞争。见 [service.cpp](src/jobs/service.cpp)、[process.cpp](src/jobs/process.cpp) |
| 公共 SDK | `RestoreRequest`、`preflight`、`restore`、模型验证/复制、回调与错误 | 调用方使用标准 C++ 类型，不依赖 Web 或 ncnn 内部类型；CLI 测试和外部集成落到同一实现。见 [pipeline.hpp](include/seedvr2/pipeline.hpp)、[适配层](src/engine/ncnn/pipeline.cpp) |
| 模型流水线 | VAE、后验采样、条件拼接、32 个 DiT 块、Euler 和输出编排 | 图片与视频保留各自布局和时间语义，共用图执行器。见 [image.cpp](src/engine/ncnn/image.cpp)、[video.cpp](src/engine/ncnn/video.cpp) |
| 模型包 | 来源、采样配置、图与常量的身份、大小和哈希 | 导出产物独立于应用安装；离线复制前后均校验。见 [package.cpp](src/engine/ncnn/package.cpp)、[审阅身份表](policies/reviewed-packages.json) |
| ncnn 执行层 | 图装载、后端选择、逐层调度、资源释放和自定义算子 | 标准算子复用 ncnn，SeedVR2 特有语义由自定义层承担。见 [inference.hpp](src/engine/ncnn/inference.hpp)、[graph.hpp](src/engine/ncnn/graph.hpp)、[awa.cpp](src/engine/ncnn/awa.cpp)、[video_layers.cpp](src/engine/ncnn/video_layers.cpp) |

这也决定了测试入口：不需要界面时用 CLI；验证嵌入能力时用独立 SDK 使用程序；测试排队、取消和恢复时才经过 Web 服务。详细职责与修改位置见[架构与源码导航](docs/ARCHITECTURE.md)。

### 模型流水线：为什么拆成 36 个图

当前 3B 包由 **1 个 VAE encoder + 1 个 patch-in + 32 个 DiT block + 1 个 patch-out + 1 个 VAE decoder = 36 个 ncnn 图**组成。按组件拆图可以逐块复用官方输入做对照，也让权重随当前阶段装载、释放。采样、布局和媒体处理由 C++ 流水线编排。

```mermaid
flowchart TB
    Input["图片或短片 · RGB 输入"] --> Prepare["缩放 / 裁剪 / 归一化；视频补齐到 4n+1 帧"]
    Prepare --> Encoder["VAE encoder · 图 1"]
    Encoder --> Condition["后验采样与缩放 · 16 通道条件潜变量"]
    PosteriorNoise["后验噪声"] --> Condition
    Condition --> Patches["扩散噪声 + 条件 + 掩码 · 33 通道；patch 1×2×2"]
    DiffusionNoise["扩散噪声"] --> Patches
    Patches --> PatchIn["patch-in · 图 2 · 投影到 2560 维"]
    PatchIn --> DiT["32 个 DiT block · 图 3–34；每块更新视频和文本"]
    Constants["固定正向文本 + 时间条件"] --> DiT
    DiT --> PatchOut["patch-out · 图 35 · 还原速度预测"]
    PatchOut --> Euler["单步 Euler endpoint · 还原 16 通道潜变量"]
    DiffusionNoise --> Euler
    Euler --> Decoder["VAE decoder · 图 36"]
    Decoder --> Output["裁回真实帧数 / 编码 · PNG 或无音轨 MP4 + run.json"]
```

关键张量关系如下。`H/W` 为处理后的像素尺寸，`Tₚ` 为补齐后的帧数，`L=(Tₚ−1)/4+1`，`N=L×(H/16)×(W/16)`；图片取 `Tₚ=L=1`。

| 边界 | 当前 3B 的逻辑形状 / 含义 |
| --- | --- |
| VAE 输入与后验 | 输入 `3×Tₚ×H×W`；后验 `32×L×(H/8)×(W/8)`，分为 16 通道均值和 16 通道 log-variance |
| DiT 条件输入 | 16 通道扩散噪声 + 16 通道采样后验 + 1 通道掩码；`1×2×2` patch 展开为 `N×132` |
| DiT 主干 | 视频 `N×2560`，块内保留时间/空间网格；文本 `58×2560`；20 个注意力头，每头 128 维 |
| 输出 | patch-out 为 `N×64`，还原为 16 通道速度预测；Euler 后经 VAE 解码回像素 |

**视频在一整段潜变量上执行时序 VAE 和 3D attention。** 例如 8 帧先重复尾帧补到 9 帧，VAE 形成 3 个潜变量时间位置，解码后裁回 8 帧。当前保留时序卷积及首帧规则，跨片段 VAE cache 关闭；输入片段、补齐和输出裁剪都记入报告。固定正向文本与时间条件随模型包提供，当前接口不接受自由文本提示。

### AWA：导出保留语义，Vulkan 执行窗口计算

AWA（adaptive window attention）的窗口大小和边界随时间、空间网格变化；普通 trace 在一个尺寸下展开窗口循环，不能代表其他尺寸。因此导出分成两个步骤：先让 pnnx 通过 **`moduleop` 保留 AWA 模块边界**，再由转换脚本检查属性、输入输出、归一化权重与 RoPE 频率，把节点映射为 **`SeedVR2AWA`**。窗口索引由运行时根据实际形状生成。

完整路径是 **官方实现与权重 → PyTorch 导出表达 → TorchScript / pnnx → 受检查的自定义层映射 → `.param/.bin` 模型包 → 原生 CPU/Vulkan**。Python、PyTorch 和 pnnx 用于准备及验证，生成的模型包交给 C++ 应用执行。相关入口为 [awa_export_module.py](tools/awa_export_module.py)、[export_awa.py](tools/export_awa.py) 和[真实 DiT 块导出](tools/export_dit_block.py)。

每个 AWA 层执行以下语义：

1. 根据实际网格生成 regular/shifted 的裁剪窗口；shifted 使用半窗口偏移并裁剪边界，不做循环回卷。
2. 在每个窗口收集视频 Q/K/V，并重复加入完整文本 Q/K/V；执行 Q/K 归一化和多模态 3D RoPE。
3. 计算窗口内的视频/文本联合注意力；Vulkan 复用锁定 ncnn 的 SDPA QK/PV 着色器，配合保留 FP32 舍入的 softmax。视频结果写回对应位置，文本结果在所有窗口间等权平均。

Vulkan 实现使用 [gather](src/engine/ncnn/shaders/awa_gather.comp)、[scatter](src/engine/ncnn/shaders/awa_scatter.comp)、[text mean](src/engine/ncnn/shaders/awa_text_mean.comp) 着色器配合 [FP32 SDPA 适配](src/engine/ncnn/softmax.cpp)；CPU 实现用于同输入对照。请求 Vulkan 时检查每层能力和 AWA 实际调用计数，不自动切换到 CPU。媒体编解码、布局、噪声和 Euler 仍在主机执行，图间边界张量会下载并在下一图上传。

### 内存与模型包：控制生命周期，保留选择依据

一次图调用的顺序是 **创建并检查图结构 → 查询预算 → 选择放置并载入权重 → 执行并取回输出 → 销毁 Net 与分配器 → 进入下一图**。这样无需让整个约 20 GB 的 FP32 图文件包同时驻留；代价是每次运行的文件读取、逐图装载和主机/GPU 传输。相邻图之间保留必要的输出，本次运行共享着色器 pipeline cache。

`auto / device / host` 是逐图权重策略。自动模式读取设备预算，按图文件大小估计权重准备空间并留出余量，再请求设备内存或主机可见内存；`host` 模式仍执行 Vulkan。策略与查询分别位于 [memory_policy.cpp](src/runtime/memory_policy.cpp) 和 [memory.hpp](src/engine/ncnn/memory.hpp)。请求的放置位置、决策理由和装载/释放前后的预算均写入报告；实际驻留由驱动决定。该估计未覆盖全部激活、工作区和分配器开销，不能保证任意设备都不会内存不足。

默认按缓冲方式读权重，可选只读 `mmap`；映射保持到对应 Net 销毁之后。模型不使用自回归 KV cache，当前也没有跨任务常驻的整包权重缓存。预检先核对参数、输入、路径、输出空间、设备和包身份，运行前再完整核验权重。离线搬移校验复制两端，并在成功后发布新目录。包身份来自项目审阅记录；数值与画质有各自的验证报告。

### 验证设计：让下面的数据可追溯

官方参考与候选导出分开维护。现用 **FP32-B** 参考在固定官方数学实现上使用 CPU PyTorch 适配；它与官方默认 BF16/Apex/FlashAttention 路径有明确区分。相同输入对照同时绑定原始后验噪声和扩散噪声，不能仅凭相同 seed 假定跨框架随机序列一致。

全链合同固定检查 **73 个张量边界：32 个 DiT 块各 2 个输出，加 9 个输入、VAE、条件和末端边界**。参考来源、形状、数据类型、完整性及输入身份先通过检查，才计算误差；缺失或重复边界直接失败。运行报告另记录模型 manifest、转换器与运行库提交、可执行文件及实际加载 SDK 的哈希。见 [pipeline_contract.py](tools/pipeline_contract.py)、[pipeline_replay.py](tools/pipeline_replay.py) 与 [provenance.hpp](src/engine/ncnn/provenance.hpp)。

据此分别报告构建与接口、小型算子、真实组件、完整执行、张量误差、任务画质和性能。同输入组件测试帮助定位错误，全链测试检查误差传播，固定目标与 bicubic 对照衡量本例修复效果；后两者回答不同问题。下面的图和表保留各自的通过项、失败项和测量条件。

## 实测结果与对照图

记录日期 **2026-09-10**。使用冻结的 **0.7.0 数值修复版 CLI/SDK**、SeedVR2 **3B / 单步 FP32-B**，设备为 **Linux x86_64 / RTX 4060 Laptop 8 GiB**，主机内存 32 GiB。输入为 **64×40、8 fps**，输出均为 **128×80**。三个案例来自同一自然素材，退化方式固定，镜头切换为人工拼接；这是开发样例，尚不构成代表性视频质量基准。

**三段原始短片均完整执行并通过 73/73 数值对照。** 固定目标画质指标仍低于 bicubic，原生和官方 FP32-B 都有这一结果；下面同时保留修复后的输出和画质数据。

每张图从左到右依次为 **bicubic 输入基线 → 原生 ncnn/Vulkan → 官方 FP32-B → 固定目标**。直接引用已归档的编码前 RGB8 对照图，没有再次增强；`f0` 表示第 0 帧。固定目标来自已压缩的源视频，不是相机原始真值。

### 自然运动 · 9 帧

展示第 0、4、8 帧，无需时序补齐。

![motion-9：相同帧的 bicubic、原生、官方 FP32-B 与目标对照](artifacts/2026-09-10/video-numerics-v2/quality/motion-9/comparison.png)

[输入短片](tests/fixtures/video-bounded/motion-9/input.mp4) · [原生实际输出 MP4](artifacts/2026-09-10/video-numerics-v2/motion-9-output.mp4) · [官方预览 MP4](artifacts/2026-09-10/video-numerics-v2/quality/motion-9/official.mp4) · [逐帧数据](artifacts/2026-09-10/video-numerics-v2/quality/motion-9/report.json)

### 尾帧补齐 · 8 帧 → 9 帧 → 8 帧

展示第 0、4、7 帧。模型输入重复尾帧补至 9 帧，实际输出再裁回 8 帧。

![padding-8：相同帧的 bicubic、原生、官方 FP32-B 与目标对照](artifacts/2026-09-10/video-numerics-v2/quality/padding-8/comparison.png)

[输入短片](tests/fixtures/video-bounded/padding-8/input.mp4) · [原生实际输出 MP4](artifacts/2026-09-10/video-numerics-v2/padding-8-output.mp4) · [官方预览 MP4](artifacts/2026-09-10/video-numerics-v2/quality/padding-8/official.mp4) · [逐帧数据](artifacts/2026-09-10/video-numerics-v2/quality/padding-8/report.json)

### 人工镜头切换 · 17 帧

展示第 0、7、8、16 帧。切换发生在第 8 帧之前，图中保留切换前后两帧。

![cut-17：相同帧的 bicubic、原生、官方 FP32-B 与目标对照](artifacts/2026-09-10/video-numerics-v2/quality/cut-17/comparison.png)

[输入短片](tests/fixtures/video-bounded/cut-17/input.mp4) · [原生实际输出 MP4](artifacts/2026-09-10/video-numerics-v2/cut-17-output.mp4) · [官方预览 MP4](artifacts/2026-09-10/video-numerics-v2/quality/cut-17/official.mp4) · [逐帧数据](artifacts/2026-09-10/video-numerics-v2/quality/cut-17/report.json)

### 数值与输出检查

| 案例 | 原数值结果 → 修复后 | RGB8 最大差¹ | 原生图执行 | 修复报告 |
| --- | --- | --- | --- | --- |
| motion-9 | 63/73 → **73/73** | 1 | 36/36 Vulkan | [JSON](artifacts/2026-09-10/video-numerics-v2/motion-9.replay.json) |
| padding-8 | 71/73 → **73/73** | 1 | 36/36 Vulkan | [JSON](artifacts/2026-09-10/video-numerics-v2/padding-8.replay.json) |
| cut-17 | 73/73 → **73/73** | 1 | 36/36 Vulkan | [JSON](artifacts/2026-09-10/video-numerics-v2/cut-17.replay.json) |

¹ 原生与官方在编码前的 0–255 通道值差异。输出帧数、尺寸、时间戳和无音轨检查均通过；图层由 Vulkan 执行，无 CPU 回退，保留日志无 Vulkan 校验错误。全部 73 个边界沿用 `abs(error) <= 0.001 + 0.001 * abs(reference)` 的诊断门槛，包括中间张量。

留出的 **17 帧、128×128 合成视频**在 CPU/Vulkan 上均通过 73/73；**256×256 自然 JPEG**也通过 73/73、RGB8 最大差 1。[六条轨迹及实现身份](artifacts/2026-09-10/video-numerics-v2/summary.json)。

### 修复质量数据

每格为 **PSNR（dB）/ RGB SSIM**，对实际输出帧取均值，不裁边；数值越高表示越接近本例固定目标，补齐帧不参与统计。

| 案例 | bicubic 基线 | 原生 ncnn/Vulkan | 官方 FP32-B |
| --- | --- | --- | --- |
| motion-9 | 26.93 / 0.8560 | 20.01 / 0.6187 | 20.01 / 0.6187 |
| padding-8 | 26.82 / 0.8539 | 19.70 / 0.5973 | 19.70 / 0.5973 |
| cut-17 | 26.80 / 0.8405 | 23.24 / 0.7686 | 23.24 / 0.7686 |


**这三个样例中，原生与官方 FP32-B 的指标都低于 bicubic。** 对照图可见喙部与羽毛细节偏离目标。原生和官方数值分别计算，在表格显示精度下接近；这是当前低分辨率设置中的负结果，不代表已评估官方默认 BF16/FlashAttention 路径。没有根据这些结果事后设置画质通过门槛。

### 耗时与内存

| 案例 | 原生总耗时（秒） | 进程 RSS 采样峰值（GiB） | 整卡显存采样峰值（MiB） |
| --- | --- | --- | --- |
| motion-9 | 28.11 | 0.918 | 2884 |
| padding-8 | 32.20 | 0.888 | 2871 |
| cut-17 | 32.25 | 0.927 | 3357 |


每例只做一次串行实测，总耗时包含模型包哈希、权重载入、计算和诊断张量保存，不据此宣称正常运行速度或加速。整卡显存包含桌面等其他进程，不能视为模型独占显存；RSS 也不是权重、激活和工作区的独立分项峰值。完整采样见[原始汇总](artifacts/2026-09-10/video-numerics-v2/summary.json)。

### 数值修复与算子验证

误差来自 FP32 运算顺序在完整模型中的累积与放大。修复保留实际模型结构，分别处理预处理插值、VAE 分块卷积与 shortcut bias、RMSNorm/FrameNorm、Q/K 预缩放、RoPE、窗口文本平均、SiLU/softmax 和 DiT 长点积补偿。

| 检查 | 本轮结果 / 范围 |
| --- | --- |
| CPU/Vulkan DiT RMSNorm | 各 94,720 个独立夹具数值逐字节一致；CPU 的 31 项超限已修复 |
| VAE 投影 / shortcut | 431,328 个 oneDNN FP32 夹具输出逐字节一致，device/host 两种权重放置 |
| 真实 encoder 定位 | 同输入后验投影 15,360 个输出逐字节一致；完整 encoder 保留误差传播记录 |
| 原生构建、算子和接口 | 本机 **36/36**；外部安装 SDK 与 CLI 加载同一库 |
| 软件 Vulkan CI | **10 项通过，12 项能力跳过**；llvmpipe 不满足 FMA 残差要求，完整 FP32-B 运行在预检时拒绝 |

`engine devices` 现在包含独立的 FP32 算术探测；设备探测通过不等于模型认证。CI 的跳过项不计作数值通过，初次 Mesa 失败日志继续保留。CPU 与 NVIDIA 的完整模型记录分别归档。

[修复原理与复现](docs/NUMERICS-REPAIR.md) · [本轮报告、失败候选与测量](artifacts/2026-09-10/video-numerics-v2/README.md) · [原始失败记录](artifacts/2026-09-10/bounded-video-v1/README.md) · [素材来源](tests/fixtures/video-bounded/SOURCE.md) · [当前缺口](docs/CURRENT-GAPS.md)

## 第一次克隆：先做不需要大模型的实验

先按 [教程第 1 课](docs/TUTORIAL.md#1-从干净克隆开始) 安装 Linux 开发依赖。随后：

```sh
git clone https://github.com/mingshi2333/seedvr2-ncnn-vulkan.git
cd seedvr2-ncnn-vulkan
python3 tools/build_native.py --cli-only --check
python3 tools/build_native.py --cli-only --jobs 2
dist/tutorial/bin/seedvr2 engine self-test --backend cpu
dist/tutorial/bin/seedvr2 engine self-test --backend vulkan --gpu 0
```

构建脚本串起锁定依赖准备、编译、小型测试和安装，不下载模型权重。新克隆没有 `dist/` 或转换模型；下面的已有安装说明不能替代构建和导出。无 Vulkan 设备时先完成 CPU 实验，并保留设备缺失原因。

继续按 [完整教程](docs/TUTORIAL.md) 做 AWA 导出、单个真实 DiT 块、完整图片和时序视频。官方权重下载已经自动化：

```sh
python3 tools/prepare_models.py --list
python3 tools/prepare_models.py
```

固定官方 revision、约 14.57 GB，支持续传和 SHA-256 校验；`--offline` 仅验证缓存。下载后还需 pnnx 转换，详见教程。源仓库包含小型测试夹具和历史报告，未上传官方大权重、转换包或便携二进制。原生 CI 的实际结果见 [GitHub Actions](https://github.com/mingshi2333/seedvr2-ncnn-vulkan/actions/workflows/native.yml)，大模型实机记录单独保留。

数值修复版提交 `7e56479` 的 Ubuntu GCC CLI、Clang CLI、GCC Web 三项远端 CI 全部通过；每项原生测试为 **24 项通过、12 项能力跳过**，Web 另有 **54/54** 接口检查。对应提交、原始日志和跳过原因见[本轮远端验证归档](artifacts/2026-09-10/github-ci-numerics-v2/README.md)；完整 3B 的 CPU/NVIDIA 实测单独记录，不由远端软件 Vulkan 测试替代。

## 本地 Web

已有本地构建和完整模型包时，从项目目录启动：

```sh
dist/tutorial/bin/seedvr2-web --model .cache/tutorial/image-package \
  --video-model .cache/tutorial/video-package
```

打开打印的地址，默认 `http://127.0.0.1:8877/`。导入 PNG/JPEG 或 MP4/MOV/WebM/MKV，选择输出长边、设备和片段帧数，开始处理。刷新或关闭浏览器后，服务进程会继续处理。任务页保留历史；服务退出会中断未完成任务，重开设置会创建一次新运行。

完成教程中的导出后，用 `--model` 和 `--video-model` 指定 `.cache/tutorial/image-package` 与 `.cache/tutorial/video-package`。模型文件独立保存；启动脚本默认查找安装目录的 `models/image` 和 `models/video`，缺失时不会自动下载或转换。Git 克隆与 `cmake --install` 不附带大模型。已提供的下载、导出、测试脚本及其边界见 [首次使用的自动化说明](docs/FIRST-RUN.md#已转换模型与自动化范围)。

- 每张输入最多 32 MiB、32 M 像素；输出长边 64–512、16 的倍数，裁剪后短边至少 64。Web 提供 128/256/384/512 四档。
- 视频最多 256 MiB、8 位 SDR、方形像素且方向已转正；短片长边 64–128、最多 17 帧，输出无音轨 MP4。当前不支持 HDR、长视频分块或流式缓存，详见[短视频范围](docs/video-runtime.md)。
- 当前预处理保持比例后居中裁至 16 的倍数。PNG 按 RGB8 输出，暂不保留透明度、ICC、EXIF 方向及其他元数据。
- 队列上限为 8 个未完成任务，每次执行 1 个；失败与取消不会暴露半成品下载。
- 端口冲突时使用 `--port 0`。`--database PATH` 可选择工作区；启动脚本使用 `SEEDVR2_MODEL_DIR` 与 `SEEDVR2_VIDEO_MODEL_DIR` 覆盖默认的安装模型目录。
- 图片包图文件约 20.44 GB，视频包约 21.10 GB，准备工具以硬链接复用未变的 DiT 权重。文件完整性会在每次处理前核验，总耗时包含哈希、权重装载和计算。

无需权重的 AWA CPU/Vulkan 自测位于“测试记录”。“模型验收”保留 12 项严格要求及缺项；一次处理或自测成功不会签发模型证书。“报告与分享”可附加实际处理记录，下载 Discussion 草稿及原始报告。

## 独立 CLI

```sh
dist/tutorial/bin/seedvr2 run-video --model .cache/tutorial/video-package \
  --input /path/to/input.mp4 --output /path/to/new-video-result \
  --frames 17 --size 128 --backend vulkan --gpu 0 --seed 666
```

视频完整时序链路、包准备和复现命令见 [video-runtime.md](docs/video-runtime.md)。

```sh
dist/tutorial/bin/seedvr2 run --model .cache/tutorial/image-package \
  --input /path/to/input.png --output /path/to/new-result \
  --size 512 --backend vulkan --gpu 0 --seed 666

# 无需启动 Web，CPU 使用同一条完整处理链
dist/tutorial/bin/seedvr2 run --model .cache/tutorial/image-package \
  --input /path/to/input.jpg --output /path/to/another-result \
  --size 256 --backend cpu --threads 4
```

图片输出目录须不存在或为空，包含 `output.png`、与结果对齐的 `comparison-input.png` 和 `run.json`。标准输出为结果 JSON，标准错误输出进度与诊断。`--diagnostic-tensors` 额外保存中间张量供开发数值验证。CLI 运行结果保存在指定目录，不会自动加入 Web 队列。

```sh
dist/tutorial/bin/seedvr2 engine status
dist/tutorial/bin/seedvr2 engine devices
dist/tutorial/bin/seedvr2 engine self-test --backend cpu --save
dist/tutorial/bin/seedvr2 engine self-test --backend vulkan --gpu 0 --save
dist/tutorial/bin/seedvr2 plan --request examples/plan-720p.json --save
dist/tutorial/bin/seedvr2 models status
dist/tutorial/bin/seedvr2 models audit --bundle examples/model-evidence-empty
dist/tutorial/bin/seedvr2 history list --kind operator-test
```

空证据包审计退出 **6** 是预期结果，表示没有模型证书。`models status` 查询退出 0 只表示查询成功。规划记录为 `PLANNED`，与真实图像任务分开保存。CLI 的全局 `--database PATH` 置于子命令前，可与 Web 共用工作区。

## 构建

开发依赖：C++20、CMake 3.25+、Ninja、Python 3.12+、Node 24+；OpenSSL、SQLite、libjpeg、zlib、uuid、Vulkan headers/loader、glslang 与 SPIRV-Tools；pkg-config 与 FFmpeg 开发库 libavformat/libavcodec/libavutil/libswscale，视频输出需要 libx264 编码器。运行时需要对应共享库 ABI，具体版本/许可在 `third_party/media`。导出还需要项目私有 PyTorch 环境，见[完整模型包准备](docs/image-runtime.md)。

```sh
python3 tools/prepare_native.py
python3 tools/prepare_engine.py --jobs 4
npm ci --prefix apps/studio --ignore-scripts
npm run build --prefix apps/studio
cmake --preset release
cmake --build --preset release --parallel 4
ctest --preset release
cmake --install build/release --prefix dist/tutorial
```

依赖准备显式联网，版本和归档 SHA-256 已锁定；缓存齐全后使用 `--offline`。CMake 不下载依赖。运行库锁定为 2026-09-08 查询时的官方 HEAD `3b7bdba7fc8aea8fd46779533eee027df77c639d`，转换器独立固定 `6a1bf000f363714839a36793addc8c879d3d899e`，没有使用相邻本地工作树。0.7.0 应用编译了经过哈希校验的 `host-buffer-v1` 分配器修正副本，原始 ncnn 源目录与归档不变；运行报告明确记录该修正和实际加载 SDK 的身份。旧导出及实验仍绑定原版本，不重新标注为新运行库；构建不会自动追随远端 HEAD。

只构建 CLI 和 worker：

```sh
python3 tools/prepare_native.py --cli-only
python3 tools/prepare_engine.py --jobs 4
cmake --preset cli
cmake --build --preset cli --parallel 4
```

该构建不寻找 Drogon、不构建 React，不要求 Node 或启动浏览器。

## 历史验证与技术文档

较早的 SDK、JPEG 修正、离线验证、组件复核、ABBA 内存/时间测量与 CI 记录见 [DELIVERY-RESULTS.md](docs/DELIVERY-RESULTS.md)。以下旧图像/视频报告保留各自执行版本，不能代替当前构建的实测。公共 SDK 源码导航见 [ARCHITECTURE.md](docs/ARCHITECTURE.md)。

已完成两种尺寸的官方 FP32-B 全链路对比：每条轨迹 73 个中间结果逐元素通过，8 位输出最大相差 1。CPU 与最终优化后的 Vulkan 构建也对保留参考进行了复核。**这些是有明确范围的开发数值证据，不是图像质量验收或完整模型认证。** 视频早期的时序 VAE 16/16 个 CPU/Vulkan 子模型检查、6 帧 64×64 完整轨迹 73/73 通过记录继续保留。17 帧 128×128 的历史结果曾为 Vulkan 60/73、CPU 70/73，严格判失败；后续数值修复后同一保留轨迹在 CPU/Vulkan 均达到 73/73，见 [视频数值修复](docs/VIDEO-NUMERICS.md)。历史失败未删除，当前 0.7.0 两种内存策略的复核见 [内存验证](docs/MEMORY-VALIDATION.md)。官方 BF16/FlashAttention、长视频和正式验收阈值尚未完成。

- [短视频运行时](docs/video-runtime.md) / [包含负结果的视频验证](docs/video-validation.md)
- [最新 ncnn 的 FP16/BF16 实测与真实组件误差](docs/PRECISION-VALIDATION.md)
- [完整单图链路、模型包与复现命令](docs/image-runtime.md)
- [当前数值验证](docs/NUMERICS-REPAIR.md) / [剩余范围](docs/CURRENT-GAPS.md) / [机器可读状态](docs/status.json)
- [应用框架和模块职责](docs/full-stack-framework.md) / [任务架构决定](docs/adr/0007-image-runtime-and-jobs.md)
- [严格模型验收](docs/model-validation.md) / [模型内部结构](docs/architecture.md)
- [本地 Web API](docs/local-web.md) / [OpenAPI](schemas/local-api.v1.openapi.json)
- [原生算子与导出设计](docs/native-backend.md) / [来源与许可](docs/provenance.md)

`video-*` 记录本轮视频扩展，`image-*` 保留 0.4.0 完整单图证据；`native-*`、`framework-*`、`foundation-*`、`web-*` 保留各自历史版本，不能用旧报告证明新构建。程序不会自动公开发布 Discussion。

## 开源与复用

原创源码和文档使用 [Apache-2.0](LICENSE)，原始第三方代码保留各自许可证和 [NOTICE](NOTICE)。完整 GPL-enabled FFmpeg/libx264 二进制的分发另有依赖许可要求；本仓库发布源码和小型证据，不宣称已完成便携二进制发行资格。详见 [许可范围](docs/LICENSING.md)。

欢迎复用组件拆分、独立参考、同输入误差定位和资源测量方法。新模型应重新核实维度、算子、设备与阈值；不要直接套用本项目的 3B 参数或把自测 PASS 当作完整模型认证。
