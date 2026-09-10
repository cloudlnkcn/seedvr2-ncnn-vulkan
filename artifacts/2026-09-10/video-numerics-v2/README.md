# SeedVR2 3B FP32-B 数值修复证据

2026-09-10，Linux x86_64 / RTX 4060 Laptop 8 GiB。最终冻结候选 **`release-fp32-v2`** 的六条完整轨迹全部通过原有 **73/73** 张量边界。原始 `motion-9` 的 63/73、`padding-8` 的 71/73 已关闭；官方参考、模型权重、输入、原始噪声和 `atol=rtol=0.001` 均保持不变。

| 入口 | 内容 |
| --- | --- |
| [summary.json](summary.json) | 六条完整轨迹的结果、参考与实现身份、一次实测的耗时与资源采样 |
| [index.json](index.json) | 轨迹、诊断、安装和本地大文件目录的导航 |
| [manifest.json](manifest.json) | 本目录文件的 SHA-256；不包含自身 |
| [implementation-plan.json](implementation-plan.json) / [implementation.diff](implementation.diff) | 冻结时的源码基点、差异、工具与新增源文件哈希，以及实际 CLI/SDK 身份 |
| [candidate-history.json](candidate-history.json) / [history](history) | 保留的中间候选结果，包含数值失败和执行错误；这些顺序实验不构成独立消融 |
| [diagnostics](diagnostics) | 相同输入的真实 VAE/DiT 定位、CPU RMSNorm 修复前后、FMA 设备探测 |
| [quality](quality) | 从最终输出生成的真实对照图、预览视频、逐帧画质和时间一致性描述数据 |
| [native-ci](native-ci/scope.json) | 本机 36/36 原生测试、Mesa 10 项通过 / 12 项能力跳过、安装 SDK 与接口检查 |
| [native-ci-initial](native-ci-initial/mesa-regressions.log) | Mesa 初次 11 项严格数值失败的原始日志 |
| [installed](installed/verification.json) | 安装后的 CLI 完整运行 9 帧视频，73/73 原始参考通过，并与冻结候选的全部边界逐字节一致 |
| [修复说明](../../../docs/NUMERICS-REPAIR.md) / [当前缺口](../../../docs/CURRENT-GAPS.md) | 根因、适用范围、复现命令和后续交付边界 |

## 最终完整轨迹

| 案例 | 后端 / 输出 | 结果 | 原始报告 |
| --- | --- | --- | --- |
| motion-9 | Vulkan / 128×80 | 73/73 | [回放](motion-9.replay.json) / [执行](motion-9-run.json) |
| padding-8 | Vulkan / 128×80 | 73/73 | [回放](padding-8.replay.json) / [执行](padding-8-run.json) |
| cut-17 | Vulkan / 128×80 | 73/73 | [回放](cut-17.replay.json) / [执行](cut-17-run.json) |
| synthetic-17-cpu | CPU / 128×128 | 73/73 | [回放](synthetic-17-cpu.replay.json) / [执行](synthetic-17-cpu-run.json) |
| synthetic-17-vulkan | Vulkan / 128×128 | 73/73 | [回放](synthetic-17-vulkan.replay.json) / [执行](synthetic-17-vulkan-run.json) |
| natural-256-vulkan | Vulkan / 256×256 | 73/73 | [回放](natural-256-vulkan.replay.json) / [执行](natural-256-vulkan-run.json) |

每个完整候选使用同一冻结实现，不能从不同候选拼接通过项。安装会改写动态库查找路径，因此安装后二进制哈希另记；独立 SDK 使用程序与安装 CLI 已确认加载同一库，安装视频也逐张量确认了与冻结候选的一致性。

六条回放的 73 项合同由既有比较器校验，报告包含来源、形状、有限性、逐元素超限数、最大误差和 RMSE。原始失败记录保留在 [bounded-video-v1](../bounded-video-v1/README.md)。本目录的 `model_verified: false` 表示没有签发通用模型证书，不否定上述固定配置与输入已经通过的数值对照。

## 结果边界

三段自然素材来自同一源片，固定人工退化，输出仅 128×80。原生和官方 FP32-B 的 PSNR/SSIM 都低于 bicubic，图和数据原样保留。该修复关闭数值失败，不证明代表性视频画质、更长片段、官方默认 BF16 路径或其他设备已通过。

本机 llvmpipe 不保留补偿算法要求的 FMA 残差；实际应用在大权重加载前拒绝该路径，CI 把相关测试记为能力跳过。跳过不计为数值通过，首次失败日志仍可查看。小型 CI 和本目录的完整大模型实机运行分别报告。

性能是开启诊断张量保存、一次串行执行的观测，包括模型校验、文件读取、计算和诊断输出；不是受控速度基准。整卡显存包含桌面与其他进程；RSS 不代表独立测量的权重、激活和工作区。CPU 回归存在交换区使用，详见其资源报告。

Git 不包含大权重、全部中间张量或冻结二进制。对应本地路径在索引中；干净克隆需要按[教程](../../../docs/TUTORIAL.md)准备模型与官方参考，才能重新计算完整 73 项误差。夹具、来源记录、最终实现和比较工具则包含在源码仓库内。文件哈希清单用于完整性检查，不是外部签名或模型认证。
