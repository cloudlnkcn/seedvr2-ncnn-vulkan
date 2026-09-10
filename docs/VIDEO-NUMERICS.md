# 17 帧视频数值修复记录

**后续修复，2026-09-10：** 128×80 自然运动/补帧/人工切换案例的原始结果为
63/73、71/73、73/73，前两例的失败现已关闭。最终同一份修复实现的六条完整轨迹
均通过原来的 73/73 门槛，详见[当前修复记录](NUMERICS-REPAIR.md)；
[原始失败](BOUNDED-VIDEO-RESULTS.md)继续保留。本文以下内容仍记录 2026-09-08 的合成短片修复及其原始身份。

2026-09-08，0.6.0 预览的后续修复。**保留的 17 帧、128×128 合成短片，在最终同一份 SDK 上 CPU 和 Vulkan 均通过 73/73 项 FP32-B 对照。** 原始 Vulkan 60/73、CPU 70/73 失败继续保留；没有更换官方 golden、噪声、模型包、输入或误差门槛。

证据目录为 [video-numerics-v1](../artifacts/2026-09-08/video-numerics-v1/)，机器可读入口为 [summary.json](../artifacts/2026-09-08/video-numerics-v1/summary.json)。它是开发数值回归，**不签发模型证书**，不覆盖自然视频质量、官方 BF16/FlashAttention 路径、长片或其他设备。

## 固定身份与范围

| 项目 | 身份或范围 |
| --- | --- |
| 官方模型体 | SeedVR `e4de8c24441a67e1b7df56abea10645059bb1185`，原文与哈希未改 |
| 模型 revision | SeedVR2-3B `37255ff8cccfb01071b87f635a5948ca8d53117c` |
| ncnn runtime | 官方 `3b7bdba7fc8aea8fd46779533eee027df77c639d`，没有修改上游源码 |
| 转换器 | 独立锁定 `6a1bf000f363714839a36793addc8c879d3d899e`，未重新导出权重 |
| 视频模型包 SHA | `fe38835c367a2a389c689d89ce85c952b0e347d1edfd879b008d796a65374028` |
| 官方视频参考报告 SHA | `9bad6fae546a23752f479b11087fdf2c95752605282b85a9b9fdc1ace9503d58` |
| 构建目录 SDK SHA | `f683258f30962fb51ff7c958e814a72ebe8ce948545e6eb555ae80c8cf3b0196` |
| 构建目录 CLI SHA | `412de27447c8b59d0e25aaf5a045facae60e2e1f7b6f4cdb480eef26384ddd09` |
| 安装目录 SDK SHA | `dcddfd914118ffcfabb6b6e37cbc7fd423320aedad59c9abe9a533d93ad11992` |
| 安装目录 CLI SHA | `d8797b4ebbcf1abb0ce69037cd1f31c05212b6d4481d94676a42c1d64a50f9f4` |
| 输入 | 17 帧 64×64 testsrc2，8 fps；SHA `71b061fb9277cfeccad53c9d739003929ac7004a24c5925707c144d0ffeabc67` |
| 推理 | 输出 128×128，时序 VAE latent T=5，32 个真实 3B DiT 块，FP32，seed 666，4 CPU 线程 |
| 机器 | Fedora 44 x86_64，RTX 4060 Laptop 8 GB，NVIDIA 595.91.07；桌面仍在使用 GPU |

协议保持 `abs(error) <= 0.001 + 0.001 * abs(reference)`。73 项是固定形状的 9 个阶段和 32×2 个 DiT 输出；验证器检查全部边界、文件哈希、原始噪声、官方参考登记、真实后端与输出身份。任何一个中间阶段失败都判整链失败。此门槛不是官方标准，也没有据本次候选重新校准。

CMake 安装时改写 ELF `.dynstr` 中的 RUNPATH，因此安装文件与构建文件的整体哈希不同。[安装身份记录](../artifacts/2026-09-08/video-numerics-v1/install-provenance.json) 保存两组身份，并核对该字符串区之外的全部字节一致。完整数值对照使用冻结的构建文件；安装后另外通过外部 SDK 和真实 Web worker 验证。

## 三项修复及证据

### 1. Vulkan RoPE 相位表

从真实 block 16 的同一份投影 QKV 出发，未旋转的两个通道只相差约 `9.54e-7`，旋转后时间轴通道最大差达到 `7.74e-5`。独立解析测试固定 Q/K 为 1，使用首个频率 1 与双精度三角函数为 oracle；3792 个检查中，旧 Vulkan 实现有 2738 项超出局部 `2e-6` 相位预算，最大差 `4.62e-5`。CPU 原实现通过。

修复在装载 AWA 时按原 FP32 角度计算 sin/cos 表，Vulkan gather 读取常量。官方语言 RoPE 的位置上界是 1024；每个常驻块新增 `1024×21×2×4 = 172032` 字节的表，主机与设备各一份。张量归一化、旋转、注意力和散射仍在 Vulkan 上执行，不回退 CPU 张量计算。模型包格式保持原样。

修复后同一解析测试 3792/3792 通过，最大差 `2.04e-7`；真实 block 16 的同输入 Q/K 最大差降至 `2.86e-6`。仅修此处的完整 Vulkan 轨迹为 **66/73，仍失败**，原始结果保留。21 个序列化频率与本机计算频率逐字节一致，因此没有把频率生成误称为根因。

### 2. Q/K 在点积前缩放

[PyTorch 2.9 的数学 SDPA 实现](https://github.com/pytorch/pytorch/blob/v2.9.0/aten/src/ATen/native/transformers/attention.cpp) 在矩阵乘法前分别缩放 Q 和 K。原生路径此前在点积后缩放，数学等价但浮点舍入与溢出行为不同。

新增独立稳定性回归：放大归一化权重，使未缩放点积溢出，而缩放后的 logits 仍有限；V 全为 1，所以正确输出可独立确定为 1。旧 Vulkan 的 40448 个输出均非有限，新实现 CPU/Vulkan 均通过。旧日志中的 `max_abs=0` 不代表精确：NaN 未计入最大值，但有限性检查已判失败。最终测试明确分别报告非有限元素数与有限值最大差。

使用 ncnn 原生 BinaryOp 分别乘 Q/K，再令 SDPA 的 scale 为 1。每个窗口多两个 Vulkan dispatch，无额外 Q/K 张量分配，不修改 ncnn。与相位表一起，完整 Vulkan 轨迹达到 **73/73**。block 31 video 的最大差由 `0.00673032` 降至 `0.000282288`，RMSE 由约 `1.10e-4` 降至 `9.10e-6`，超限元素由 520 降至 0。

### 3. CPU 时序编码器的卷积策略

两项 AWA 修复后 CPU 整链仍为 **70/73**。以官方初始张量进入完整 32 块 DiT 时，64 个块输出全部通过；同输入 block 16 的 CPU AWA 误差也没有显示 Vulkan 的相位问题。定位转向编码器。

同一份官方 prepared 输入下，ncnn CPU 编码器采用 SGEMM 卷积时，posterior 最大差 `6.38962e-5`、RMSE `8.76825e-6`；直接卷积时降至 `1.81198e-5`、`2.85841e-6`。诊断计算含指定边界落盘，观察耗时分别为 12.77 秒和 20.75 秒。归一化同输入探针则只有小量浮点差异，没有修改其数学实现。

先将时序 encoder/decoder 都切为直接卷积，CPU 整链通过 73/73，但总耗时 130.04 秒。测量显示 decoder 计算约 55.84 秒，旧 SGEMM 路径约 36.28 秒。最终仅在 **CPU 时序 encoder** 禁用 SGEMM，decoder 保持原有策略；独立图诊断与共享 SDK 流水线使用相同策略，阶段报告记录该选择。最终 CPU 仍为 **73/73**，单次观察总耗时 112.47 秒。Vulkan 和单图 VAE 不受此 CPU 策略影响。

| 最终 17 帧轨迹 | posterior 最大差 | block 31 video 最大差 | block 31 RMSE | decoded 最大差 | 超限阶段 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vulkan | 3.24249e-5 | 2.82288e-4 | 9.10462e-6 | 5.42998e-5 | 0/73 |
| CPU | 1.62125e-5 | 1.31559e-3 | 2.19921e-5 | 5.28097e-5 | 0/73 |

CPU 某些最大绝对差大于 0.001 仍可能通过，因为原协议包含按参考值计算的相对误差项；没有删去这部分定义或隐藏超限元素。

## 保留的无收益候选与失败

- **插值 FMA 候选未合入**：17 帧 prepared 与官方逐字节一致，但 CPU 整链仍为 70/73；独立缩小样例仍有 46 个元素超出新设局部 `2e-7` 探针预算。扩大范围后的结果不支持普遍逐字节匹配。原预处理保持不变，候选源码、输入、参考和日志单独归档。
- **只修 RoPE 未解决整链**：保留 66/73 完整失败及官方起始张量隔离结果。局部误差降低不保证所有后段误差单调下降。
- **测试程序曾发生 validation 装载崩溃**：原因是静态 SPIRV-Tools 符号被动态 validation layer 绑定，引入与现有 SDK 相同的 Linux `--exclude-libs,ALL` 后解决。该失败不计作模型数值差异。
- **直接卷积覆盖 decoder 的较慢候选未保留为默认**：保留通过结果和成本，最终限定到已测得误差传播来源的 encoder。
- 原始 60/73、70/73、自然照片质量负结果、低精度未通过项继续保留在之前的交付和精度证据中。

## 最终回归与安装验证

以下均来自最终实现，未重新生成官方参考。

| 验证层级 | 结果 | 范围 |
| --- | --- | --- |
| 完整视频 CPU / Vulkan | 各 73/73 | 同一段合成短片，17 帧，128×128 |
| 完整视频 Vulkan 回归 | 73/73 | 前 6 帧，64×64，补齐 9 帧后裁回 6 帧 |
| 自然 JPEG Vulkan 回归 | 73/73；最大像素差 1 | 128→256，单张 NASA 照片；数值对照不判定修复画质 |
| AWA 原始模型体对照 | 40/40 | 20 个算子样例 × CPU/Vulkan |
| 真实时序 VAE CPU | 8/8 | 新编码器卷积策略与原解码器策略 |
| 原生 CTest | 10/10 | 包括新增 CPU/Vulkan 相位与有限值回归 |
| Mesa Vulkan 显式回归 | 2/2 | 软件设备、validation layer，无设备跳过 |
| 参考协议拒绝与完整性 | 11/11 | 缺项、重复、篡改身份及容差等必须拒绝 |
| CLI / 引擎边界 / 首次使用 | 10/10、23/23、8/8 | 安装后原生程序 |
| 安装后 SDK / Web | 通过；Web 15/15 检查 | 外部 SDK 构建及能力调用、实际 17 帧 worker 任务 |

最终 Vulkan 完整运行的全部保存张量，与两项 AWA 修复首次通过时逐字节一致。新收紧的 AWA 隔离报告也已在原有 block 16 数据上执行：三个窗口均有完整记录，AWA video/text 的同投影最大差分别为 `6.68e-6` 和 `2.86e-6`。

安装后的 [Web 任务证据](../artifacts/2026-09-08/video-numerics-v1/installed-web/verification.json) 对应任务 `eca7a5e64fd3b2319cea4bda24198407`。它实际运行全部 36 个 Vulkan 子图，输出 17 帧、128×128、无音轨 H.264；下载 MP4 的 SHA 与最终 CLI 完全相同。原有六条任务保留，中文媒体名、进度记录、时间戳、下载与当前 SDK/worker 身份均已核对。浏览器已打开该任务，显示同步播放、逐帧时间线和运行报告入口；本轮没有重跑此前已通过的播放交互与完整任务生命周期套件。

首次安装后检查脚本误写共享库文件名，发生在视频成功生成之后；修正为实际 `libseedvr2.so` 后，仅重新核对保存的结果，没有重复大模型运行。该检查器错误与初始日志也保留。原生 CI 在本机执行通过，远端 CI 尚未触发；全模型实机记录与小型 CI 分开存放。

## 复核和日常使用

CLI、本地 Web worker 与外部 SDK 共用更新后的原生库。安装和使用入口不变，见 [首次使用](FIRST-RUN.md)。UI 中的状态说明与 Discussion 草稿已更新；日常运行报告仍保持 `model_verified=false`，处理完成不会自动生成开发对照报告或模型证书。

```sh
cmake --build build/release --parallel 4
ctest --test-dir build/release --output-on-failure
cmake --install build/release --prefix dist/seedvr2-0.6.0

dist/seedvr2-0.6.0/bin/seedvr2 run-video \
  --model .cache/video-package-fp32 --input .cache/video-demo-17.mp4 \
  --output .cache/my-video-check --frames 17 --size 128 \
  --backend vulkan --gpu 0 --threads 4 --seed 666 --diagnostic-tensors
python3 tools/check_video_outputs.py --run .cache/my-video-check \
  --reference .cache/video-reference-v2 --reference-run .cache/video-run-v2 \
  --output .cache/my-video-parity.json
```

这些参考目录是本机已经保存并在登记表中审阅的官方轨迹；首次克隆仓库不会自带大模型或完整原始张量，不能用一个新候选输出替代它们。`--backend cpu` 使用相同输入和历史协议。

开发诊断入口按实际问题拆分：

- [trace_dit.py](../tools/trace_dit.py)：官方起始、原生起始或逐块相同输入隔离，严格保留块序列与边界覆盖。
- [trace_dit_layers.cpp](../tools/trace_dit_layers.cpp) / `seedvr2-dit-trace`：复用实际图调度器及 AWA，读取保留的真实块输入，保存内部边界。仅显式构建，不进入默认大模型 CI。
- [trace_official_block.py](../tools/trace_official_block.py)、[analyze_awa_trace.py](../tools/analyze_awa_trace.py)：原始 FP32-B 模型体、同 QKV 对照和 FP64 注意力 oracle；当前 AWA 定位工具限定在真实 3B block 16。
- [trace_vae_encoder.cpp](../tools/trace_vae_encoder.cpp) / `seedvr2-vae-trace`、[analyze_vae_trace.py](../tools/analyze_vae_trace.py)：同输入 CPU 编码器卷积策略及归一化诊断。
- [awa_rope_tests.cpp](../tests/awa_rope_tests.cpp)：无大权重的解析相位与有限值回归。CI 另在 Mesa Vulkan 上明确执行这两项，软件设备结果不替代完整模型实机验证。

Vulkan 最终单次总耗时 35.29 秒，其中图计算 8.38 秒、包校验 13.25 秒、图载入 12.24 秒。前后运行页缓存、桌面负载和温度未受控，**不据此声称整体加速**。CPU 策略选择有明确的精度收益和计算成本；权重仍逐图加载，没有新增全模型常驻或时序/KV 缓存，也没有把进程 RSS 当作独立激活/工作区峰值。

后续验收重点是自然运动、镜头切换、重复纹理和时序稳定性；另需官方默认 BF16 路径、AMD/Intel 真机及跨平台交付资格。当前视频上限仍为 17 帧/128 像素、8 位 SDR、无音轨，图片上限仍为 512 像素。本次没有扩张支持范围或公开发布。
