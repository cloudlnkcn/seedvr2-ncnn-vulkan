# 0.5.0 短视频运行时

此版本在同一个本地 Web、独立 CLI 和 worker 中加入整段短片推理。使用官方 3B 权重、时序 VAE、全部 32 个 DiT block 和 3D adaptive window attention；Vulkan 路径不会把网络层静默改到 CPU。视频作为联合时空张量进入网络。

当前为 **实验性预览**：开头 1–17 帧，输出长边 64–128、16 的倍数，短边至少 64。本机保留的 17 帧、128 像素合成轨迹已在 CPU/Vulkan 各通过 73/73 项 FP32-B 开发对照，见 [修复记录](VIDEO-NUMERICS.md)；[历史失败](video-validation.md) 继续保留。无长视频分块、流式缓存、音轨保留或正式模型证书。

## 使用

```sh
build/dev/seedvr2-web --port 8877 \
  --model .cache/image-package-fp32 \
  --video-model .cache/video-package-fp32
```

从“新建处理”导入 MP4/MOV/M4V、WebM 或 MKV。界面显示短片范围，选择开头 5/9/17 帧、尺寸、CPU/Vulkan 和 seed 后提交。视频与图片共用持久队列、取消和重开功能。完成后可同步播放、暂停到同一帧、逐帧前进后退、拖动时间线、下载 MP4 和运行记录。

CLI 不需要 Web 或浏览器，也能指定任意 1–17 帧数量：

```sh
build/dev/seedvr2 run-video --model .cache/video-package-fp32 \
  --input /path/to/input.mp4 --output /path/to/new-video-result \
  --frames 17 --size 128 --backend vulkan --gpu 0 --seed 666

build/dev/seedvr2 run-video --model .cache/video-package-fp32 \
  --input /path/to/input.webm --output /path/to/new-cpu-result \
  --frames 6 --size 64 --backend cpu --threads 4 --seed 666
```

输出目录须不存在或为空。包含 `output.mp4`、对齐的 `comparison-input.mp4`、两侧第一帧 PNG 封面，以及 `run.json`。`--diagnostic-tensors` 额外保留未经过视频编码的 FP32 中间结果。MP4 为有损交付格式，数值验收比较编码前张量。

`--video-model` / `SEEDVR2_VIDEO_MODEL_DIR` 与图片模型选项独立。未指定时依次使用用户数据目录、安装目录、开发缓存中的 `video-package-fp32`。包存在只表示可尝试执行；每次任务先核对实际图文件和权重身份，不等于模型已认证。

## 媒体范围

- 输入最多 256 MiB，单边 16–4096、总计最多 4096×2160 像素，1–240 fps，8 位 SDR、方形像素、像素方向已经转正。拒绝非零旋转声明、HDR/PQ/HLG、BT.2020、超过 8 位和不支持的色彩矩阵。
- 只允许本地文件和 MP4/MOV、Matroska/WebM 解复用器；不接受流媒体 URL 或播放列表。解码最多读取目标帧数加一帧，后一帧用于判断是否截短。
- 按声明的 YCbCr 矩阵/量程转 RGB8；未声明矩阵时采用 FFmpeg 的 BT.601 默认。没有完整色彩管理、HDR 色调映射或 ICC 工作流。
- 每帧按同一尺寸做 bicubic antialias、居中裁至 16 的倍数，归一化到 [-1,1]。空间处理规则与单图一致。
- 原始可用时间戳平移至首帧为零，使用 90 kHz 时间基并写入报告；缺失时间戳按帧率生成，非递增时间戳拒绝。未保留完整容器元数据。
- 输出 H.264 / yuv420p / CRF 18 / BT.709，faststart MP4，无音轨。输出尺寸与时间长度应以实际报告为准。
- 17 帧在 24 fps 下约 0.71 秒，在 30 fps 下约 0.57 秒。这是短片调试范围，尚不适合完整影片或 720p/1080p 修复。

## 时序计算与自定义导出

输入帧数 N 向上补到 `4n+1`，补帧重复最后一帧。VAE 编码为 `ceil(N/4)` 个 latent 帧，解码得到 `4*(latent_frames-1)+1`，最终裁回实际读入的 N 帧。例如 **6 → 补至 9 → 3 latent 帧 → 解码 9 → 输出 6**。

`tools/vae_video_module.py` 是独立的 pnnx 可导出表达；完整保留官方 3D 卷积权重、首帧因果填充、逐帧 GroupNorm/空间 attention 和时间 pixel shuffle。encoder 的时间降采样在 down block 1、2；decoder 的前两个 up block 恢复时间。`MemoryState.DISABLED` 表示整段输入，不实现缓存式连续推理。

`export_vae_video.py` 将 TemporalConv、FrameNorm、FrameSDPA、TemporalShuffle 保存为有明确边界的 scripted modules。`spec` 常量必须参与脚本图，避免 pnnx 前的 tracing 删除算子属性。lowering 验证属性、形状与权重后，生成四个原生自定义层。

| 原生层 | CPU/Vulkan 实现 |
| --- | --- |
| SeedVR2TemporalConv | 因果时间 gather，将 Cin×Kt 展开为 2D 卷积输入通道；帧间加入隔离空间，调用固定官方 ncnn Convolution 后 scatter。保留完整 3D 权重，不把时间核压成单帧核。 |
| SeedVR2FrameNorm | 每帧每组独立归一化；Vulkan 归约 shader，不把 T 合并进 GroupNorm 的空间统计。 |
| SeedVR2FrameSDPA | 每帧单独 gather 成空间 token，调用 ncnn SDPA 后还原；这是 VAE 的空间 attention。 |
| SeedVR2TemporalShuffle | 严格按官方 dy/dx/time/channel 相位排列，并处理首帧特殊相位裁剪。 |

DiT 继续复用图片包中的 32 个动态图，输入包含实际 latent T。自定义 AWA 根据 T/H/W 生成普通/移位的 3D 窗口，保留 joint video/text attention、RoPE、逆散射和文本聚合。17 帧的 latent T=5 实际触发时间窗口，不是 17 次独立图片推理。

`inference.hpp` 共享包身份核验、图注册、显式后端调度、权重生命周期和噪声算法；`image.cpp` / `video.cpp` 分别组织对应媒体的完整轨迹。VAE 每 4 层提交 GPU 指令以释放临时缓冲，DiT 每 8 层；通常只驻留当前图的权重。当前卷积 gather 单张量上限 1 GiB，普通视频张量上限 512 MiB。这是拒绝异常分配的界限，**不是测量得到的峰值内存承诺**。

## 准备视频包

前提是已按 [单图运行时](image-runtime.md) 准备官方权重、私有导出环境、固定 pnnx 和完整图片包。开发导出需要 Python/PyTorch，运行时不需要。

```sh
.venv-export/bin/python tools/export_vae_video.py \
  --pnnx .deps/bin/pnnx --output .cache/vae-video-export-new

.venv-export/bin/python tools/check_graph.py --binary build/dev/seedvr2 \
  --suite .cache/vae-video-export-new/suite.json \
  --output .cache/vae-video-native-new --report .cache/vae-video-native-new.json \
  --validation-layer

.venv-export/bin/python tools/export_video_package.py \
  --image-package .cache/image-package-fp32 \
  --vae .cache/vae-video-export-new --output .cache/video-package-new
```

每个输出目录应为空或不存在。实际已准备包为 `.cache/video-package-fp32`，manifest SHA-256 `fe38835c367a2a389c689d89ce85c952b0e347d1edfd879b008d796a65374028`，图文件合计约 21.10 GB。未改动的 DiT 文件以硬链接复用图片包，避免重复占用磁盘；两份 manifest 与时序 VAE 身份独立。结构契约在 [video-package schema](../schemas/video-package.v1.schema.json)，结构校验不能替代实际文件哈希和图执行。

## 运行库与发行

构建新增 pkg-config 和 FFmpeg 开发库 `libavformat`、`libavcodec`、`libavutil`、`libswscale`。程序直接调用这些共享库；H.264 输出需要库中启用 `libx264`。不通过 shell 调用 FFmpeg，不依赖 Python/Node 服务，也不下载外部网页资源。

本机验证使用 FFmpeg 8.0.1。共享库 ABI、发行包许可标识、配置和许可证原文记录在 `third_party/media`，随开发安装附带。当前系统 FFmpeg 为启用 GPL/libx264 的构建，安装目录未打包系统库。完整应用许可、跨平台编解码器和便携发行资格尚未完成，不能把本机安装说成即用的跨平台二进制发行。
