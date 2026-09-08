# 0.6 原生后端与应用验证

2026-09-07，项目版本 0.3.0-ncnn-dev。真实 ncnn/pnnx、自定义 AWA、单帧 VAE、完整独立 DiT 块及 Web/CLI 自测已运行。**本报告没有完整 SeedVR2 图像/视频恢复结果，没有模型证书。**

## 模型计算证据

| 套件 | 实际范围 | CPU | Vulkan | 参考与结论 |
| --- | --- | --- | --- | --- |
| AWA | 20 组形状，普通/移位、20 heads、极端宽高比、T=33、文本归并 | 20/20 | 20/20 | 合成 projected QKV/RMS；固定官方 attention 方法体的 FP32-B 参考 |
| 单帧 VAE | 官方权重；编码/解码 × 32×48、48×32、64×64 | 6/6 | 6/6 | 原始官方 3D 类在 T=1 的输出；独立 2D 候选经真实 pnnx 导出 |
| 完整 DiT 块 | 官方权重块 0、1、10、11、31，各两种 token grid | 10/10 | 10/10 | 独立官方完整块参考，包含 ada、投影、AWA、SwiGLU、residual 及视频/文本输出 |

共 **72/72 次后端执行通过**。计数包含 CPU 与 Vulkan，并非 72 个独立场景。最新结果：[AWA](awa-export-validation.json)、[VAE](vae-image-validation.json)、[DiT](dit-block-validation.json)；AWA 原始 pnnx 导出身份另见 [awa-pnnx-export.json](awa-pnnx-export.json)。所有逐元素比较均无非有限值、无超差元素；Vulkan 使用真实 RTX 4060，不依赖请求参数推断实际后端。

AWA 容差为 `1e-5 + 1e-4*abs(reference)`；VAE/DiT 为 `1e-4 + 1e-3*abs(reference)`，均在 float64 中比较原始 FP32 文件。以下最大误差只是所测输出的诊断摘要，不能用作整网精度或图像质量指标。

| 套件 | CPU 最大绝对误差 | Vulkan 最大绝对误差 |
| --- | --- | --- |
| AWA | 1.3113021850585938e-6 | 2.3543834686279297e-6 |
| 单帧 VAE | 4.0531158447265625e-5 | 3.719329833984375e-5 |
| DiT 块 | 1.52587890625e-4 | 1.678466796875e-4 |

采用逐元素绝对与相对混合容差；不能把最后一行单独与 atol 比较。容差是开发诊断值，没有经过模型验收校准。参考是显式 FP32-B 适配，尚未运行官方 BF16/FlashAttention A 路径。

VAE T=1 不覆盖时间缓存，独立块也不覆盖 32 层误差累积。AWA 的 20 组数据不使用官方 checkpoint；真实权重证据来自 VAE 和完整 DiT 块。它们各自证明表中的范围。

## 工程与使用路径

| 检查 | 结果 | 实际范围 |
| --- | --- | --- |
| GCC C++20 构建、Studio 类型检查/生产构建 | PASS | 原生 engine、CLI、嵌入 Web 资源；Debug 主程序、Release ncnn |
| CTest | 4/4 | 核心契约、规划 CLI、worker 能力边界、真实 CPU AWA 内置自测 |
| CLI 参数与协议 | 10/10 | 路径、几何、JSON、未实现整网 run 的明确拒绝 |
| 模型证据审计工具 | 31/31 | 合成文件/张量负向测试、实际误差、拒绝伪造认证；不是模型证据 |
| 原生输入与失败边界 | 23/23 | 错误 shape/dtype/hash、NaN、路径越界、链接、输出保护、失败保存和数据库未来版本拒绝 |
| HTTP 与 Web/CLI 互读 | 59/59 | 真实 CPU/Vulkan 自测、v1→v2 数据保留、设备、会话/Origin/Host/JSON、计划/审计回归 |
| 安装后 HTTP/CLI | 59/59 | 从项目外 `/tmp` 启动安装后的程序，再做实际自测和跨入口验证 |
| 安装内容与独立运行 | 10/10 | 三个二进制身份、11 个许可/来源文件、锁/接口文件、空 cwd 与无开发 PATH 的 CPU 自测 |
| Chromium 实际操作 | 11/11 | 显卡选择、两种自测、刷新重开、JSON/Discussion 实际下载、模型边界、桌面及手机布局 |

机器可读记录：[CTest](native-ctest.xml)、[CLI](native-cli-validation.json)、[审计工具](native-audit-validation.json)、[原生边界](native-engine-boundary-validation.json)、[HTTP](native-web-validation.json)、[安装](native-install-validation.json)、[安装 HTTP](native-install-http-validation.json)、[浏览器](native-browser-validation.json)。这些检查有重叠，不应相加成“独立模型测试总数”。

浏览器实际使用 1440×1000 与 390×844 viewport。测试页与分享页 document width 均为 390；测试表格在 316 像素容器内滚动 660 像素内容。实际下载的 Vulkan 报告与 SQLite 保存 JSON 一致；草稿包含所选记录 ID、真实 backend、ncnn commit、PASS 和未完成整网的说明。浏览器错误和警告均为 0；刷新后捕获的请求均为本机资源。没有执行公开发布。

当前能力快照：[native-build-capabilities.json](native-build-capabilities.json)。完整模型状态：[native-model-status.json](native-model-status.json)，始终 `model_verified=false`、`certificate=null`；策略 `NOT_FROZEN`，签发关闭。

## 实际身份与主机

- ncnn/pnnx 官方源码：`6a1bf000f363714839a36793addc8c879d3d899e`；远端 HEAD 查询于 2026-09-06 20:38 UTC，此后按提交锁定，不自动追随更新。
- 官方源码归档 SHA256：`3f5fb14e3393859dd1adffabb79b1f0530272d5fe1eb8b470adc29682f939f6d`，14,683,974 字节。源码未改动，也未使用相邻 ncnn 工作树。
- 私有 pnnx SHA256：`732a25bf3c8e17b1d071b3eed8c6131a14da5450564bbe1a157d6dc11c2a10ba`；Python 3.13.12 / PyTorch 2.9.0+cpu。最新三类导出均绑定它。
- CLI SHA256：`a266bc06569e58e83c1634606f9d05f43bd2bf124956ee823fcfc4a2356c34e1`；三个子模型结果绑定此二进制，安装后的 CLI 哈希相同。
- Web SHA256：`d7318d65f534c67b5ba69b85fa23540c308aa1f0f1413ab9f892ffd694cf6b8a`，浏览器实际使用此版本。
- 本机：Fedora Linux x86_64，kernel 7.1.13-200.fc44，GCC 16.2.1；RTX 4060 Laptop GPU，驱动 595.91.07，设备报告显存 8188 MiB。
- ncnn Vulkan 开启；FP32、关闭 BF16/FP16、关闭测试图中的自动 CPU 回退。系统 glslang / SPIRV-Tools 仍是构建/运行依赖；没有全封闭工具链保证。

精确源码、报告、shader、参考清单与二进制身份见 [native-manifest.json](native-manifest.json)。哈希用于识别本轮文件，不宣称跨机器按位可复现，也不代替模型验证。

## 发现并修复的问题

1. pnnx 把 DiT 的 token 轴推断为 batch，并固化 AWA 周围若干 reshape。导出工具保留原 param，对四种精确结构进行 lowering，检查匹配数量与真实输入/输出形状。
2. DiT 的调制参数按每个 hidden 维交错保存，不能按六个大段切开；官方 learned scale 也不能再加 1。完整块参考暴露并校正了这两处语义。
3. 单帧 VAE 的时间上采样相位与空间 PixelShuffle 排布不同；候选保留原始 3D 参考，显式选取第一帧相位并重排通道。
4. ncnn MemoryData 的 Vulkan clone 使用了缺少 TRANSFER_SRC 标志的 buffer。虽然数值接近，validation layer 仍报 `VUID-vkCmdCopyBuffer-srcBuffer-00118`。应用内加入 `SeedVR2Constant`，为小常量分配有复制权限的 buffer；没有修改上游 ncnn。
5. Vulkan validation 信息可能写到 stdout。验证工具保存 stdout/stderr，并把非 JSON 或 VUID 作为失败；不能只看退出码或 stderr。最终 72 次执行无相应诊断。
6. case 文件声明的 head/shift/shape 必须与实际图一致；输入与权重中的非有限数值、重复 JSON 字段和路径越界均明确拒绝。数据库 v2 在事务内迁移保留旧记录，并拒绝把未来版本降级。

早期失败的导出、GPU validation 和构建记录仍保留在 `.cache`；最新通过结果没有覆盖旧原始张量目录。当前没有新的 ASan/UBSan 全后端报告，不能沿用 framework-* 的旧 sanitizer 结果证明新 engine。

## 尚未通过的模型与发行项

全部 32 层串联、官方输入/输出投影、posterior 与噪声、sampler/scaling、整张图恢复、视频 VAE 缓存与分块、FP16/BF16、画质与保留集、峰值显存/性能、worker 监督/取消、媒体 IO、实际输出对比、Windows 与便携包仍未完成。RTX 4060 上的小形状检查不能证明 8GB/720p 能运行。

下一阶段按 [native-backend.md](native-backend.md) 的验收顺序推进，先交付完整单张图的逐层证据。完整模型验证与发行仍需独立验收。
