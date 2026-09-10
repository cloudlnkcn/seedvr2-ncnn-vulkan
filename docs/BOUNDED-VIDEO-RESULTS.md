# 有界短视频实测与定位 · 2026-09-10

> 后续修复：原始 motion-9 与 padding-8 的数值失败已在不变参考和门槛下关闭，见 [2026-09-10 修复记录](NUMERICS-REPAIR.md)。本文保留当时的实验结论和失败轨迹。

**本轮执行完成，完整数值验收仍未全部通过。** 新增的自然运动、补帧和人工切换三个案例分别为
**63/73、71/73、73/73**；原生与官方编码前 RGB8 每通道最大差均为 1。
三个案例中，当前原生和官方 FP32-B 输出相对固定目标的 PSNR/SSIM 均低于 bicubic 基线。
这些结论同时保留，不能合并成“视频已全面修好”或模型认证通过。

本轮遵循执行前写定的[范围与停止条件](BOUNDED-VIDEO-VALIDATION.md)。
原始报告、日志、诊断失败和可查看结果在[证据目录](../artifacts/2026-09-10/bounded-video-v1/README.md)，
机器可读入口为 [summary.json](../artifacts/2026-09-10/bounded-video-v1/summary.json)。
所有模型任务串行执行；没有重复实现未变的历史大模型实验。

## 固定对象与输入

| 项目 | 本轮实际范围 |
| --- | --- |
| 模型/采样 | SeedVR2 3B、单步、FP32-B；CFG 1、seed 666、color fix none |
| 设备/后端 | Linux x86_64，RTX 4060 Laptop 8 GiB，ncnn Vulkan；主机 32 GiB RAM |
| 原生候选 | 保留的 0.7.0 安装版 CLI/SDK；本轮未修改或重建模型计算 |
| 官方模型体 | `e4de8c24441a67e1b7df56abea10645059bb1185`，32 个原始文件逐一核验 |
| ncnn / pnnx | 运行库 `3b7bdba7fc8aea8fd46779533eee027df77c639d`；转换器 `6a1bf000f363714839a36793addc8c879d3d899e`；既有 `host-buffer-v1` 修正身份保留 |
| Python 参考 | Python 3.13.12 / PyTorch 2.9.0+cpu；共享原始噪声，官方模型体独立执行 |
| 空间/时间 | 输出均为 128×80；8 fps；三个固定片段，最长 17 帧 |
| 数值门槛 | 完整轨迹继续使用 `atol=rtol=0.001`；AWA 使用 `atol=1e-5, rtol=1e-4` |

模型清单 SHA 为 `fe38835c367a2a389c689d89ce85c952b0e347d1edfd879b008d796a65374028`。
CLI SHA 为 `2c2d08ac27d1a35ce2f2a8a92d32c307f7774ab00398999b6173d672893112b6`；
SDK SHA 为 `ff86dcd0b05ea66f5ab384701701aced273582e0ee95fc0cfa12c9c4608f30f2`。
脚本、参考源和安装文件共 108 项在执行前冻结，结束后重算摘要。
当前源码与历史 0.7.0 原生模型源码一致；安装文档的变化不构成模型重新验证。

素材来自固定提交的 ImageIO `cockatoo.mp4`，来源说明和精确输入已放入
[video-bounded 夹具](../tests/fixtures/video-bounded/SOURCE.md)，约 1.06 MB。
先得到 128×80 固定 RGB 目标，再下采样到 64×40、经 H.264 CRF 28 压缩。
目标来自已经压缩的视频，不能称为相机原始真值；三个案例共享一个来源，且切换为人工拼接。
帧索引、切换位置、工具版本及每份输入/目标的摘要均在执行前固定。

## 执行、数值与媒体分别判断

| 案例 | 时序处理 | 完整边界 | 编码前 RGB8 最大差 | 原生总耗时 | 采样 RSS 峰值 |
| --- | --- | ---: | ---: | ---: | ---: |
| motion-9 | 9 帧，无需补齐 | **63/73，未通过** | 1 | 28.09 s | 1.184 GiB |
| padding-8 | 8 帧补至 9，输出裁回 8 | **71/73，未通过** | 1 | 27.62 s | 1.126 GiB |
| cut-17 | 17 帧，第 8 帧前人工切换 | **73/73，通过此案例** | 1 | 31.31 s | 0.893 GiB |

三个原生任务均正常退出，完成 36 个 Vulkan 图，无图层 CPU 回退，保留日志无 Vulkan 校验错误。
MP4 解码后的帧数、尺寸、单调时间戳与报告一致，输出无音轨。三个 `decoded` 浮点边界均通过。
官方检查器在前两例完成参考生成后因数值失败退出 1，第三例退出 0；批量执行器收齐三个案例后退出 1。

完整边界被再次独立计算，逐项核对了形状、有限性、摘要、误差、超限元素数和通过状态；
每侧共检查 54,208,000 个标量。新参考在原始来源、原始张量、输出抽帧和报告绑定审阅后登记。
登记允许后续严格重放，不抹去候选的数值失败，也不签发模型证书。

`motion-9` 的最早超限是 block 19 video；共 9 个 video 边界和 block 31 text 超限。
`padding-8` 仅 block 15/17 text 超限，各 1 个元素。逐边界值见原始报告，未调宽门槛。

资源数据是一次连续实测，包含模型哈希、载入和计算。整卡显存采样峰值为 3633/4202/5521 MiB，
包含桌面等其他进程；不能当作模型独占显存。原生进程采样交换量为 0。
官方 CPU 检查器分别用时 33.96/34.50/40.54 s，最大 RSS 由 GNU time 单独记录。
页缓存、温度和桌面负载未受控，不据此比较加速比；RSS 也未分解为权重、激活与工作区。

## 同输入定位与实际修复

| 隔离实验 | 改动的输入 | 结果 |
| --- | --- | --- |
| motion-9，block 19 重放 | 使用原生保存的 block 18 输出 | 失败稳定复现，两个输出与原失败轨迹逐字节一致 |
| motion-9，block 19–31 | 每块都使用对应官方上一块输出 | 26/26 输出通过 |
| motion-9，全部 32 块 | 仅起点替换为官方 patch/text/time，随后消费原生上一块输出 | 64/64 输出通过 |
| padding-8，block 15–17 | 每块都使用对应官方上一块输出 | 6/6 输出通过 |

`motion-9` 原始输入进入 DiT 前已有微小差异：prepared 最大差约 `5.96e-7`，
posterior 约 `3.48e-5`，patch-in 约 `4.19e-6`；后续 block 19 video 最大差约 `0.006155`。
相同官方输入下，该块 video 最大差约 `4.01e-5`，无超限元素。

这些干预支持**进入 DiT 的差异经轨迹传播后影响后段数值门槛**。
它们没有把来源进一步唯一归因到预处理、VAE、采样或投影中的某一个算子，
也没有修复两条原始完整轨迹。停止在这个已复现、可继续隔离的边界，避免无依据试改模型数学。

本轮修复了复现时实际遇到的三个工具问题：

- `trace_dit.py` 从已审阅轨迹推导 `(3,5,8)` 等视频网格，去掉只认 17 帧/128×128 的限制；仍服从既有短片范围。
- 从 Linux ELF 读取实际 SDK SONAME，修复硬编码 `libseedvr2.so.0` 无法载入 0.7 SDK 的问题；先检查进程退出，再解析 JSON，并核验 CLI/SDK 身份和实际后端。
- 画质工具拒绝空视频、非有限值、形状不匹配和非法切换位置；补帧不进入输出画质统计。空输入的失败测试与修复后结果均保留。

这些修改作用于开发诊断和质量统计，原生应用数学、权重、模型包与默认参数未变。

交付检查另将批量工具的安装快照限定为 SeedVR2 SDK，兼容 `lib`/`lib64` 并冻结符号链接指向的实际字节，
避免复制安装前缀中的其他库或依赖后来改变的链接。两个回归检查和实际安装版 CLI/SDK 的无权重载入检查通过。
完整实验执行时的旧脚本保留在归档中；没有因快照收紧而重复模型推理。

## 算子新增覆盖

新增 AWA 网格 `(3,5,8)` 和 `(5,5,8)`，各测 regular/shifted，20 heads、58 个文本位置。
四例均经过 pnnx 自定义边界导出；CPU/Vulkan 共 **8/8**，16 组输出比较，最大绝对差
`1.6093254e-6`。QKV 与归一化权重是确定性合成输入；真实权重组件证据来自上述 DiT 同输入隔离。
局部通过与完整轨迹未通过并存。

## 内容质量与限制

| 案例 | bicubic PSNR / SSIM | 原生 PSNR / SSIM | 官方 FP32-B PSNR / SSIM |
| --- | --- | --- | --- |
| motion-9 | 26.93 dB / 0.8560 | 20.01 dB / 0.6187 | 20.01 dB / 0.6187 |
| padding-8 | 26.82 dB / 0.8539 | 19.70 dB / 0.5973 | 19.70 dB / 0.5973 |
| cut-17 | 26.80 dB / 0.8405 | 23.24 dB / 0.7686 | 23.24 dB / 0.7686 |

均为逐帧 RGB 指标的均值，无边缘裁剪；完整设置与逐帧数据保留在 quality 报告。
抽帧可见部分喙部和羽毛形状偏离目标；原生和官方视觉上非常接近。
当前极低分辨率、固定退化和 FP32-B 设置下，不能承诺模型修复优于简单插值。
这不是对官方默认 BF16 配置或更广泛素材的总体质量结论。

时间统计使用 `MAE(Δ(输出 − 目标))`，人工切换项单列，其余项求均值。
三例原生值约为 18.08/18.38/11.09，bicubic 为 9.38/9.41/8.40。
它衡量同一目标下误差如何随帧变化，**不是光流补偿的闪烁指标**；本轮也没有做视频盲评。
逐帧对照图和四路预览随报告保存，原生实际 MP4 与重新编码的诊断预览分别标注。

## 如何重用

不下载权重的检查使用已准备的导出环境（见[教程第 2 课](TUTORIAL.md#2-看清自定义-awa-如何导出)）：

```sh
.venv-export/bin/python -m unittest discover -s tests -p test_video_quality.py -v
.venv-export/bin/python -m unittest discover -s tests -p test_pipeline_contract.py -v

.venv-export/bin/python tools/generate_awa_reference.py \
  --video-boundaries --output .cache/bounded-check/awa-reference
.venv-export/bin/python tools/export_awa.py \
  --suite .cache/bounded-check/awa-reference/suite.json \
  --pnnx .deps/bin/pnnx --output .cache/bounded-check/awa-export
.venv-export/bin/python tools/check_awa.py \
  --binary dist/seedvr2-0.7.0/bin/seedvr2 \
  --suite .cache/bounded-check/awa-export/suite.json \
  --output .cache/bounded-check/awa-native --report .cache/bounded-check/awa.json \
  --validation-layer
```

已有完整视频包时，普通 CLI 可以直接试固定输入；此命令只生成，不代表数值验收：

```sh
dist/seedvr2-0.7.0/bin/seedvr2 run-video \
  --model dist/seedvr2-0.7.0/models/video \
  --input tests/fixtures/video-bounded/motion-9/input.mp4 \
  --output .cache/bounded-preview --frames 9 --size 128 --backend vulkan --gpu 0 --seed 666
```

完整验证还需要固定官方 checkpoints（`tools/prepare_models.py`）和导出环境；
安装/转换方法见[教程](TUTORIAL.md)。批量工具按三个固定案例串行运行，冻结实现，保留超时与失败：

```sh
.venv-export/bin/python tools/run_bounded_video_validation.py \
  --fixtures tests/fixtures/video-bounded \
  --binary dist/seedvr2-0.7.0/bin/seedvr2 \
  --model dist/seedvr2-0.7.0/models/video --output .cache/bounded-check/full

.venv-export/bin/python tools/audit_bounded_video.py \
  --execution .cache/bounded-check/full --fixtures tests/fixtures/video-bounded \
  --output .cache/bounded-check/audit.json
```

选择新的输出目录。批量工具的数值失败退出码不能忽略；审计通过仅说明来源、完整边界和计算可复算。
新生成参考保持 `PENDING`，需要按现有审阅流程核对来源、输入、输出及误差后才能加入
`tests/reference/reviewed-pipelines.json`，禁止仅凭候选 PASS 自动登记。
本次三个参考已审阅登记；在本机保留的大张量仍存在时，可直接重算质量而不重跑模型：

```sh
.venv-export/bin/python tools/measure_video_quality.py \
  --fixtures tests/fixtures/video-bounded --case motion-9 \
  --run .cache/bounded-video-v1/full/motion-9 \
  --reference .cache/bounded-video-v1/full/motion-9-reference \
  --reference-run .cache/bounded-video-v1/full/motion-9 \
  --output .cache/bounded-quality-recheck
```

干净克隆含输入、目标、报告和预览；不含大权重、CLI/SDK 二进制和完整中间张量。
重新构建和生成的候选有自己的身份，不能继承此次通过结论。

## 检查与后续边界

新增 6 项质量/固定夹具测试和现有 11 项完整轨迹证据契约测试在本机通过。
另有 2 项 SDK 快照契约检查通过，覆盖 `lib`/`lib64`、缺失 SDK、无关文件及链接变更。
质量测试已接入 Linux CI，依赖 NumPy/Pillow，不下载模型；**本提交的远端 CI 尚未执行**。
原生 C++ 未改动，因此没有为此次 Python/文档更新重复历史构建及完整 CPU 轨迹。

下一项有直接证据支持的工作是：在保留 `motion-9` 输入与官方轨迹上，进一步拆分预处理、VAE 与投影的差异传播，
一次只验证一个原因；修复后只重跑受影响的完整案例。本轮保留这项未解决的数值问题。
其他素材的质量、官方默认 BF16、AMD/Intel、Windows/macOS、长视频、音轨、HDR 与高分辨率仍在本轮范围外。
