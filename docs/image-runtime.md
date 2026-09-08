# 完整单图运行时

0.4.0-image-preview / 0.7-durable-image-jobs，2026-09-07。本版已执行完整单图 FP32 链路；模型认证保持关闭。实测见 [image-validation.md](image-validation.md)。

## 固定执行范围

| 项目 | 当前实现 |
| --- | --- |
| 模型包 | `seedvr2-3b-image-fp32-b-v1`，36 张图：encoder、decoder、patch-in/out、block-00..31 |
| 权重 | 官方 3B DiT、VAE、正向 embedding；固定来源 revision 与 SHA-256 |
| Conditioning | 官方正向文本投影 58×2560、timestep 1000 的六路调制 2560×6，导出时固化 |
| 采样 | 单步、CFG 1、VAE stochastic posterior、latent scale 0.9152、Euler endpoint；color_fix=none |
| 输出 | 长边 64–512、16 的倍数，裁剪后短边 ≥64；RGB8 PNG |
| 后端 | ncnn CPU 或明确逐层 Vulkan；不做神经网络层的 CPU 回退 |
| 主机处理 | 图像解码、bicubic antialias、布局、随机噪声、posterior 与 Euler 算术、PNG 编码 |
| 未实现 | 视频、时间缓存、低精度、高分辨率/tile、ICC/EXIF/alpha、完整模型证书 |

RNG 为 `splitmix64-box-muller-v1`；不承诺与 PyTorch 相同 seed 得到相同随机序列。独立参考验证注入保留的实际 posterior-noise 与 DiT noise，比较运算本身。复现还需要相同二进制、输入、模型、参数及设备环境。

## 执行与内存边界

`include/seedvr2/image.hpp` 只暴露标准 C++ 类型、进度观察与取消回调。`image.cpp` 实现完整管线；`graph.hpp` 是共享 ncnn 图执行适配器；`image_io.cpp` 负责受限 PNG/JPEG IO 和预处理。输入统一到 RGB，按输出长边调整大小并居中裁至 16 的倍数。透明度与图片元数据暂不保留。

模型文件哈希通过后，按 encoder → patch-in → 32 blocks → patch-out → decoder 依次装载和释放权重。图内部在最后一次消费后释放激活引用，Vulkan 每 8 层提交并检查取消。一个任务共用 ncnn PipelineCache，减少重复编译相同块的 shader。此策略已降低本机运行耗时，但没有完整峰值显存验收，也不代表生产性能已优化。

每块接收上一块实际的视频与文本输出。timestep 六路调制按原始交错布局排布。原始 NaDiT 的输出 AdaSingle 使用 block-0 写入的 `emb_repeat_0_vid` 缓存；候选的输出 scale/shift 因此取该缓存对应的第一组。参考保留原始 forward 与缓存行为，不能单独对 output Ada 调用后忽略缓存。

图层来自固定允许集合，并检查输入/输出个数、输出形状、每块唯一 AWA、20 heads 和移位奇偶性。36 张图、常量和相对路径工件均验证 SHA-256。重复/未知图、目录逃逸、符号链接、哈希不匹配和错误采样 profile 会失败。哈希只证明文件与清单一致；来源身份由导出证据追溯，不提供签名信任或模型证书。

`run.json` 记录二进制哈希、模型清单哈希、输入/输出哈希、ncnn 提交、设备、实际后端、每块计算/装载耗时、RNG、采样参数和模型未认证状态。PNG 与报告先写临时文件再重命名；Web 仅在 worker 成功退出且输出哈希复核通过后提供下载。原子重命名不承诺断电后的文件持久性。

## 准备完整模型包

普通用户使用已准备的包时不需要 Python。以下为开发者从官方权重重新导出的流程；输出目录必须新建或为空，避免覆盖历史证据。模型图文件约 20.44 GB，导出中间文件和官方 checkpoint 还会占用额外空间。

```sh
uv venv --python 3.13 .venv-export
uv pip install --python .venv-export/bin/python torch==2.9.0+cpu --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-export/bin/python -r tools/export-requirements.lock.txt
python3 tools/prepare_engine.py --jobs 4
python3 tools/prepare_pnnx.py --jobs 4
python3 tools/prepare_models.py

.venv-export/bin/python tools/export_vae_image.py \
  --pnnx .deps/bin/pnnx --output .cache/image-exports/vae
.venv-export/bin/python tools/export_dit_block.py \
  --pnnx .deps/bin/pnnx --output .cache/image-exports/dit \
  --blocks 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
.venv-export/bin/python tools/export_image_package.py \
  --pnnx .deps/bin/pnnx --vae .cache/image-exports/vae \
  --blocks .cache/image-exports/dit --output .cache/image-package-fp32
```

最后一步验证各子图导出身份、导出 patch-in/out、写入实际 conditioning 并原子提交包清单。大权重使用硬链接，包和导出目录须在同一文件系统。包由清单引用的文件组成，运行时不使用 `.pt` 或 Python 文件。复制到其他文件系统可按清单复制所引用文件并保留相对路径。

本机实际包复用了前一轮 VAE 和五块导出，其余 27 块另行导出：`.cache/vae-image-export-private`、`.cache/dit-block-export-private`、`.cache/dit-block-export-remaining`；`export_image_package.py` 可以多次提供 `--blocks DIR`，但索引必须刚好覆盖 0–31，不能重复。

模型路径优先顺序：显式 `--model`、环境变量 `SEEDVR2_MODEL_DIR`、用户数据目录 `models/seedvr2-3b-image-fp32-b-v1`、可执行文件旁的 `models/`、安装 `share/seedvr2/models/`、开发项目 `.cache/image-package-fp32`。模型文件不随开发安装复制。Linux 默认数据目录为 `$XDG_DATA_HOME/seedvr2`，或 `~/.local/share/seedvr2`。

## 独立数值复核

```sh
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
  build/dev/seedvr2 run --model .cache/image-package-fp32 \
  --input /path/to/input.png --output .cache/image-run-new \
  --size 256 --backend vulkan --gpu 0 --diagnostic-tensors \
  > .cache/image-run-new.stdout 2> .cache/image-run-new.stderr

.venv-export/bin/python tools/check_image.py \
  --run .cache/image-run-new --input /path/to/input.png \
  --output .cache/image-reference-new
```

参考使用固定官方 NaDiT、全部原始 block、VAE 原始 3D 类的 T=1 路径、官方 LinearInterpolationSchedule 和 Euler endpoint。显式 FP32-B 适配替换 RMSNorm/attention 的外部加速实现；官方 BF16/FlashAttention A 路径未验收。

开发诊断阈值在评估前定义：`abs(x-y) <= 1e-3 + 1e-3*abs(y)`，最终 RGB8 差值 ≤1。这不是正式模型 policy 的冻结阈值。`check_image_outputs.py --run NEW --reference REF --reference-run ORIGINAL --output REPORT` 可将相同输入、模型、实际噪声的额外 CPU/Vulkan 执行与保留参考比较，并重新核对张量/运行身份。

## 任务、取消与恢复

独立 worker 的 `seedvr2-worker-v1` NDJSON 包含 progress/result/error。监督进程使用无 shell 的固定参数启动 worker、读取有界消息，在 SQLite 中事务保存状态与事件。队列最多 8 个未完成任务、1 个执行；取消先发 SIGTERM，超过 10 秒再 SIGKILL，始终回收子进程。Linux 父进程死亡信号防止 Web 崩溃后留下继续计算的孤儿 worker。

进度有 38 个计数步，包含校验、VAE、采样、投影、32 blocks 和解码。它表示完成的阶段数，不是剩余时间预测。事件具有递增 sequence，可按 cursor 续读；目前使用轮询，不声称 SSE 已实现。重启后未完成任务标记 `INTERRUPTED`，可在保留输入和参数的基础上重试。CLI 直接调用同一引擎，当前 Ctrl+C 为进程中断，不承诺与 Web 相同的持久取消记录。
