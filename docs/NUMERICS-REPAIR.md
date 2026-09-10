# 原始视频数值失败的修复

2026-09-10。SeedVR2 **3B / FP32-B / 单步 / CFG=1**。原始 `motion-9` 的 **63/73**、`padding-8` 的 **71/73** 已修复。最终数值候选 `release-fp32-v2` 对三段原始视频、留出的合成视频 CPU/Vulkan 和自然图片均通过原来的 **73/73** 边界检查。[机器可读汇总](../artifacts/2026-09-10/video-numerics-v2/summary.json)。

这次修复保留官方参考、真实权重、模型 manifest、输入、原始后验/扩散噪声和 `abs(error) <= 0.001 + 0.001 * abs(reference)`。没有删除中间边界，也没有用最后一张图的相似程度豁免失败。当前结论对应这些固定输入和设备；修复质量另外评估。

## 固定范围和证据

| 对象 | 身份 / 范围 |
| --- | --- |
| 官方 SeedVR 代码 | `e4de8c24441a67e1b7df56abea10645059bb1185` |
| 官方权重 revision | `37255ff8cccfb01071b87f635a5948ca8d53117c` |
| 既有 pnnx 导出 | `6a1bf000f363714839a36793addc8c879d3d899e` |
| ncnn 运行库 | `3b7bdba7fc8aea8fd46779533eee027df77c639d`，原有 `host-buffer-v1` 分配器适配另记哈希 |
| 官方数值参考 | 锁定 PyTorch 2.9.0 CPU FP32-B；oneDNN 3.7.1、MKL 2024.2、AVX512；不是官方默认 BF16/Apex/FlashAttention 路径 |
| 实机 | Linux x86_64，RTX 4060 Laptop 8 GiB，约 32 GiB 主机 RAM |
| 真实模型路径 | 原来的 36 个图、32 个 DiT 块；Vulkan 图层没有切回 CPU |

每个完整结果包含 CLI、**实际加载的 SDK**、模型 manifest、参考报告、输入和张量身份。[候选冻结清单](../artifacts/2026-09-10/video-numerics-v2/implementation-plan.json)绑定实现文件；[历史候选表](../artifacts/2026-09-10/video-numerics-v2/candidate-history.json)保留失败和无收益方案。中间候选完整源文件和大张量留在本地忽略目录，公开目录提供报告与身份；不声称这些历史候选的全部二进制都已发布。

## 如何找到误差

先重放原来的失败，再把真实组件放到相同官方输入上。motion 的 DiT 从官方 patch/text/time 起点执行时可以通过全部 64 个输出，但消费旧原生 VAE 起点时失败。因此继续在 VAE 内保留输入、归一化、下采样、shortcut 和后验边界，检查误差如何进入 DiT。局部通过只是定位证据，最终仍要求整个流水线通过。

修复涉及几个不同来源，不能归因于单一算子：

| 位置 | 发现与实现 | 独立证据 |
| --- | --- | --- |
| 图像预处理 | 对齐 FP32 插值系数和乘加的舍入边界 | 8 个形状，289,536 个数值对照 |
| VAE 空间/时序卷积 | 官方真实后验使用 oneDNN 的 16 通道分块累加；原先整体点积的顺序不同。按实际核的循环、分块和 bias 顺序实现 Vulkan 投影 | 同一真实输入的 15,360 个后验输出逐字节一致；小型夹具覆盖 3D、空间下采样和四种 channel shortcut |
| VAE channel shortcut | channel-changing 的 oneDNN 1×1 shortcut 在 FMA 累加开始前加入 bias；注意力中的 Linear 保留末尾加 bias | 两个真实 encoder shortcut 的 512 个抽样输出，正确顺序逐字节一致 |
| 归一化 | RMSNorm、Q/K RMSNorm、FrameNorm 的平方物化、级联求和、Welford 合并及 sqrt/reciprocal 舍入需要分别保留 | CPU/Vulkan DiT RMSNorm 各 94,720 个夹具数值逐字节一致；FrameNorm 覆盖小型和实际规模归约 |
| 注意力 | 对齐 Q/K 的预缩放、冻结 FP32 RoPE 常量、窗口文本排序/平均和稳定 softmax | 12 种 softmax 宽度、7 种窗口数量，另保留真实 AWA 对照 |
| DiT 长点积 | FP32 Dot2 同时补偿乘积舍入与相消；只作用于 DiT 块的 Vulkan InnerProduct | 独立解析点积夹具与真实 block 对照；两个累加器均为 FP32，不依赖设备 FP64 |
| SiLU | 使用有来源记录的 SLEEF FP32 exp 与除法修正，保留尾部行为 | 1D、2D、4D 共 163,934 个夹具数值 |

相关实现集中在 [ncnn 执行目录](../src/engine/ncnn)、[着色器目录](../src/engine/ncnn/shaders)、[夹具生成工具](../tools)和 [CTest 注册](../CMakeLists.txt)。oneDNN/SLEEF 来源及许可证在 [third_party/engine](../third_party/engine)。这些计算顺序针对锁定参考环境推导，不是 PyTorch 对所有 CPU、编译器和版本承诺的逐位规范。

一个无收益方案也保留了：把后验卷积换成更接近 FP64 的补偿点积，虽然减小了相对 FP64 的误差，却仍不能通过原来的完整 FP32-B 回放。最终采用有实际 oneDNN 源码依据的 FP32 顺序，不把“更接近无限精度”和“对齐冻结实现”混为一个目标。

## 完整回归

| 输入 | 后端 / 输出 | 原记录或本轮中间失败 | 修复结果 |
| --- | --- | --- | --- |
| 自然运动 9 帧 | Vulkan，128×80 | 63/73 | **73/73** |
| 尾帧补齐 8 帧 | Vulkan，128×80 | 71/73 | **73/73** |
| 人工切换 17 帧 | Vulkan，128×80 | 73/73 | **73/73** |
| 合成运动 17 帧 | Vulkan，128×128 | 留出回归 | **73/73** |
| 合成运动 17 帧 | CPU，128×128 | 本轮发现 72/73 | **73/73** |
| 自然 JPEG | Vulkan，256×256 | 留出回归 | **73/73**，RGB8 最大差 1 |

CPU 留出回归曾在 `block-30-video` 的 `[150,1326]` 超限：绝对差 `0.00103759765625`，允许值 `0.0010006787776947021`。CPU RMSNorm 的独立夹具同时有 31 项超限；修正求和顺序后夹具全部逐字节一致，完整视频恢复 73/73。这项负结果及修复前后日志也在[诊断目录](../artifacts/2026-09-10/video-numerics-v2/diagnostics)。

修复 CPU RMSNorm 前后的三段原始 Vulkan 视频，全部 73 个输出张量及 MP4 均逐字节一致。这验证了 CPU 改动没有改变已通过的 GPU 结果；所有对照由摘要脚本再次检查身份，而非拼接不同候选的通过项。

安装后的 CLI 另做了一次原始 9 帧视频回放：完整 73/73 通过，并与冻结候选的 73 个输出张量逐字节一致；使用含中文和空格的输出目录，未设置额外动态库搜索路径。外部 SDK 使用程序也确认载入同一个安装库。[安装回归](../artifacts/2026-09-10/video-numerics-v2/installed/verification.json) / [SDK 身份](../artifacts/2026-09-10/video-numerics-v2/native-ci/sdk-identity.json)。

## 设备能力与 CI

本机 NVIDIA 的原生测试通过。软件 Vulkan llvmpipe 26.1.8 的初次严格数值运行有 **11 个失败**，保留在 [原始 Mesa 日志](../artifacts/2026-09-10/video-numerics-v2/native-ci-initial/mesa-regressions.log)。检查驱动源码并用运行时输入重现后，确认其把 FMA 拆成乘法和加法，丢掉补偿算法需要的残差。

核心 Vulkan 的 GLSL `Fma` 不能直接视为 IEEE 单次舍入保证；本项目的 `NoContraction` 处理也不构成跨驱动保证。参见 [Vulkan 浮点精度规则](https://github.khronos.org/Vulkan-Site/spec/latest/appendices/spirvenv.html)、[Mesa 26.1.8 llvmpipe 编译选项](https://gitlab.freedesktop.org/mesa/mesa/-/blob/mesa-26.1.8/src/gallium/drivers/llvmpipe/lp_screen.c)。

现在 [独立算术探测](../src/engine/ncnn/fp32_device.hpp)使用 8 组解析可知的输入检测该条件。NVIDIA 为 8/8，本机 llvmpipe 为 0/8。**公共 preflight 和真实组件入口会在大模型加载前拒绝不满足要求的设备**；`engine devices` 报告检测值，用户可以显式选 CPU。这个条件不按品牌硬编码，也不代表探测通过后所有模型或精度都已认证。

CI 保留普通算子执行，并把依赖该算术条件的严格 FP32 测试记为能力跳过；另有公共接口测试检查“不支持时确实拒绝运行”。跳过不计作数值通过。最终数量、名称和原因见 [CI 汇总](../artifacts/2026-09-10/video-numerics-v2/native-ci/scope.json)，原始错误不删除。当前未实现软件 FMA，也没有修改系统 Mesa 驱动。

## 复现与升级

已有经过审阅的模型包可直接复用；本次没有修改转换产物。重新构建运行程序即可：

```sh
python3 tools/build_native.py --cli-only --jobs 2
dist/tutorial/bin/seedvr2 engine devices
```

不下载大模型也能运行新增的夹具测试：

```sh
ctest --test-dir build/tutorial --output-on-failure
```

大模型回放使用同一冻结参考目录及其原始 `run.json`。例如 motion：

```sh
dist/tutorial/bin/seedvr2 run-video \
  --model .cache/tutorial/video-package \
  --input tests/fixtures/video-bounded/motion-9/input.mp4 \
  --output outputs/motion-repaired --frames 9 --size 128 \
  --backend vulkan --gpu 0 --threads 4 --seed 666 --diagnostic-tensors
python3 tools/check_video_outputs.py \
  --run outputs/motion-repaired \
  --reference .cache/bounded-video-v1/full/motion-9-reference \
  --reference-run .cache/bounded-video-v1/full/motion-9 \
  --output outputs/motion-repaired-parity.json
```

新克隆不包含这些大参考张量；先按[教程](TUTORIAL.md)和[原始三段视频协议](BOUNDED-VIDEO-RESULTS.md)准备官方参考。比较器要求已审阅的身份及完整 73 项合同；不应为了让新环境通过而修改身份表或门槛。生成夹具也需要其 provenance 中对应的参考环境。

修复后的图、视频、逐帧画质和测量在 [README](../README.md#实测结果与对照图)。三个低分辨率开发样例的原生与官方画质仍低于 bicubic；本轮关闭的是原始数值失败，没有据此宣称画质、长视频或跨平台交付完成。剩余范围见[当前缺口](CURRENT-GAPS.md)。
