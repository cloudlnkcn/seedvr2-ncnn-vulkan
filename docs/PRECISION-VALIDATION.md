# 最新 ncnn 的 FP16/BF16 实测

2026-09-08。使用官方当前 HEAD `3b7bdba7fc8aea8fd46779533eee027df77c639d`，与旧运行库 `6a1bf000f363714839a36793addc8c879d3d899e` 对照。两个库都来自项目保存的官方归档，没有修改上游源码。**新版本的 BF16 reduction 修复已被真实 GPU 实验复现；完整 SeedVR2 的 FP16/BF16 支持仍未交付。** 应用、CLI、Web 和 SDK 继续使用 FP32。

[汇总和文件身份](../artifacts/2026-09-08/precision-v1/summary.json) · [逐文件 SHA-256](../artifacts/2026-09-08/precision-v1/SHA256SUMS.json)

## 实验范围与身份

- Fedora 44、GCC 16.2.1、NVIDIA 595.91.07、RTX 4060 Laptop 8 GB，subgroup 32；测试前显存占用约 1,772 MiB、GPU 利用率 0%。桌面仍使用 GPU。
- 独立 C++ 测试程序链接指定的 ncnn 静态库，记录库 SHA、提交、测试程序 SHA、实际输入输出 SHA。进程启动前复制程序快照，测试逐个执行。
- 显式调用 Vulkan 层；没有 Extractor 的 CPU 回退。请求与最终精度选项必须一致，上传张量的实际存储位宽必须符合请求，输出形状固定。设备不支持时记录 SKIPPED，不降级为 FP32 后报告通过。
- 请求 Vulkan validation，成功执行的 54 项没有捕获校验错误。此处 54 项是新库 36 项、旧库 18 项，不包含下述两次无 subgroup 的失败。
- 上游 `tests/testutil.cpp` 的通用 GPU 测试主动把 `use_fp16_arithmetic` 设为 false，并跳过原本请求该选项的用例。本工具显式覆盖该模式；通用测试通过不能代替它。
- 小算子包括 Erf、CELU、Reduction、SDPA。SDPA 是合成输入，使用 20 heads、head width 128、65 tokens；不称为官方 AWA 组件对照。
- 真实组件是 17 帧、128×128 官方轨迹中的 **patch-in Gemm**，输入 `[320,132]`，输出 `[320,2560]`。从已审阅的官方 conditioned 张量和固定的原始噪声重建 patch 顺序，使用原模型包的真实权重，对照已保存的官方 patch-in 输出。
- 官方整个参考报告 SHA 为 `9bad6fae546a23752f479b11087fdf2c95752605282b85a9b9fdc1ace9503d58`，先经现有完整 73 项合同和审阅清单校验，再抽取该组件。没有重新生成或重新批准官方参考。

所有模式使用现有 FP32 开发诊断预算 `abs(error) <= 0.001 + 0.001 * abs(reference)`，用于直接观察代价。**它没有被校准为 FP16/BF16 的正式验收标准。超差不单独证明 ncnn 有缺陷，通过也不签发模型证书。** 没有修改旧阈值或删除旧视频失败。

## BF16 修复的直接证据

构造 65,536 个可被 BF16 精确表示的输入：一半为 `+0.5`、一半为 `-0.5`，其中前 128 个正数各增加 `1/256`。真实和为 `0.5`。GPU 部分和先变大再相消，用于检查共享累加器是否丢失余量。

| 精度模式 | 旧 `6a1bf00` 输出 | 新 `3b7bdba` 输出 | 参考值 |
| --- | ---: | ---: | ---: |
| FP32 | 0.5 | 0.5 | 0.5 |
| FP16 存储、FP32 运算 | 0.5 | 0.5 | 0.5 |
| FP16 存储与运算 | 0 | 0 | 0.5 |
| FP16 packed 与运算 | 0 | 0 | 0.5 |
| BF16 storage、FP32 运算 | **0** | **0.5** | 0.5 |
| BF16 packed、FP32 运算 | **0** | **0.5** | 0.5 |

输入和参考文件在所有模式、新旧版本间完全一致。普通求和用例中，BF16 packed 的最大绝对误差也从 `0.90625` 降为 `0.09375`。这些结果与 [上游 df3e53b / PR #6964](https://github.com/Tencent/ncnn/commit/df3e53b1827e887a6ba4ca7965aa87c1998c01ff) 将共享部分和从 `lfp` 改为 `afp` 的实现吻合。该修复让 BF16 的 FP32 部分和保留在 FP32 中；它没有把 FP16 arithmetic 全部改为 FP32 累加。

## 真实 patch-in 组件

同一官方输入、真实权重和官方 FP32-B 输出；共 819,200 个输出元素。

| 模式 | 最大绝对误差 | RMSE | 超出既有 FP32 预算的元素 |
| --- | ---: | ---: | ---: |
| FP32 | 5.96046e-7 | 2.50017e-8 | **0** |
| FP16 存储、FP32 运算 | 0.00206232 | 9.59697e-5 | **3** |
| FP16 存储与运算 | 0.0102215 | 0.000371306 | **6,775** |
| FP16 packed 与运算 | 0.0102215 | 0.000371306 | **6,775** |
| BF16 storage、FP32 运算 | 0.0215936 | 0.000768492 | **46,177** |
| BF16 packed、FP32 运算 | 0.0215936 | 0.000768521 | **46,178** |

各模式对应的新旧 ncnn 输出逐字节一致。因此，最新 BF16 reduction 修复有效，但没有改变这个真实投影组件的低精度误差。FP16 存储、FP32 运算是目前较有依据的后续候选；仍需对完整组件边界和最终时序质量独立验证。

最新库的 36 个成功执行用例中，21 个符合既有 FP32 预算，15 个超差，0 个能力跳过。Erf 和 CELU 在 FP16 arithmetic 下实际编译并执行成功；Erf 仍有该预算下的数值超差。不能把着色器编译修复等同于全模型数值通过。

## 保留的失败与限制

1. **无 subgroup 的独立 Reduction**：显式设置 `use_subgroup_ops=false` 时，新旧版本都出现 `subgroup op: requires SPIR-V 1.3` 编译错误，随后测试子进程以 `-11` 退出。两份完整 stderr 已保留；不是能力跳过，也不是新库回归。该额外用例不在上面的 36/18 项成功执行计数中。当前应用默认开启 subgroup，本次没有修改上游实现。
2. **首轮测试工具无效**：初稿在下载时被 ncnn 重新打包，且求和系数写成整数参数字面值 `2=1`。修正为明确的 CPU pack1 下载及浮点参数 `2=1.0` 后重新测试。首轮在 `harness-invalid-initial/` 保留并标记 `valid_for_numerical_conclusions=false`，其数值不用于结论。
3. **完整低精度模型未运行**：AWA 和时序 VAE 的精度约束尚未适配；本次只有一个真实投影组件，没有宣称整段 FP16/BF16 视频通过。最新运行库已有的 17 帧 FP32 完整执行仍是 60/73 FAIL。
4. **性能未认证**：原生报告保存一次装载、上传、计算、下载的合计耗时，包含冷启动成本；没有测稳态吞吐、独立显存峰值或端到端质量，不能据此宣称加速或显存收益。
5. **设备范围有限**：本次新实验使用 NVIDIA 真机；没有模拟 shaderInt16 缺失设备，也没有复测 AMD/Intel 或软件 Vulkan。BF16 packed 的跨设备编译修复不能据此宣布全平台通过。

## 复现与独立复算

测试工具不进入已安装 SDK，也不修改应用的精度开关。准备过锁定的 ncnn 后：

```sh
cmake -S tools/ncnn_precision_probe -B .cache/precision-probe-build-new \
  -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build .cache/precision-probe-build-new --parallel 2

python3 tools/check_precision.py \
  --binary .cache/precision-probe-build-new/ncnn-precision-probe \
  --output .cache/my-precision-check --gpu 0 \
  --cases erf celu reduction reduction-cancellation sdpa patch-in \
  --reference .cache/video-reference-v2 --baseline .cache/video-run-v2 \
  --package .cache/video-package-fp32

python3 tools/replay_precision.py .cache/my-precision-check \
  --output .cache/my-precision-check/replay.json
```

去掉 `patch-in` 和最后三项来源参数即可执行无需大模型权重的合成用例。程序因超出诊断预算返回 1 是保留的负结果；检查各行 status，不把它与编译/执行失败混为一类。旧库比较可通过单独构建目录和 `SEEDVR2_NCNN_PREFIX` 指向已准备的旧库，不能覆盖正在运行的程序。

三批有效结果已分别用 NumPy 从保存的原始输出独立复算：30/30、6/6、18/18 的统计与原生报告一致。`statistics_verified=true` 表示报告统计可信，不表示全部数值符合预算。原始 tensor、真实组件的小模型副本和可执行文件在汇总中的 `.cache` 路径；版本控制中保存 JSON、stderr、源码快照与哈希。增加相消用例前后的测试程序具有各自的 SHA，没有把早期实验重新标注为后来二进制。

源码入口：[原生 probe](../tools/ncnn_precision_probe/main.cpp)、[顺序执行与来源绑定](../tools/check_precision.py)、[独立复算](../tools/replay_precision.py)。
