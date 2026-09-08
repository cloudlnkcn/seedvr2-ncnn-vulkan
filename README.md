# SeedVR2 ncnn Vulkan

**0.5.0 video preview：官方 3B 权重已能在 ncnn CPU / Vulkan 上完成真实图片和整段短片处理。** 本地 Web 提供图片/视频导入、持久队列、进度、取消/重试、图片滑动对比、视频同步播放/逐帧比较、PNG/MP4 与运行记录下载。CLI 可独立处理两种媒体。图片长边最高 512；短片为开头 1–17 帧、长边最高 128。**视频数值验收仍有失败，功能为实验性预览，完整模型认证尚未通过。**

React / TypeScript / Ant Design 界面内嵌于 Drogon C++ 服务。共享 C++20 推理核心，由独立 worker 执行，SQLite 保存任务与有序事件。运行时无需 Node、Python、云服务或 CDN。本机验证平台为 Linux x86_64 / RTX 4060 Laptop GPU；尚未形成跨平台便携发行版。

## 现在使用

已有本地构建和完整模型包时，从项目目录启动：

```sh
build/dev/seedvr2-web --model .cache/image-package-fp32 \
  --video-model .cache/video-package-fp32
```

打开打印的地址，默认 `http://127.0.0.1:8877/`。导入 PNG/JPEG 或 MP4/MOV/WebM/MKV，选择输出长边、设备和片段帧数，开始处理。刷新或关闭浏览器后，服务进程会继续处理。任务页保留历史；服务退出会中断未完成任务，重开设置会创建一次新运行。

- 每张输入最多 32 MiB、32 M 像素；输出长边 64–512、16 的倍数，裁剪后短边至少 64。Web 提供 128/256/384/512 四档。
- 视频最多 256 MiB、8 位 SDR、方形像素且方向已转正；短片长边 64–128、最多 17 帧，输出无音轨 MP4。当前不支持 HDR、长视频分块或流式缓存，详见[短视频范围](docs/video-runtime.md)。
- 当前预处理保持比例后居中裁至 16 的倍数。PNG 按 RGB8 输出，暂不保留透明度、ICC、EXIF 方向及其他元数据。
- 队列上限为 8 个未完成任务，每次执行 1 个；失败与取消不会暴露半成品下载。
- 端口冲突时使用 `--port 0`。`--database PATH` 可选择工作区，`SEEDVR2_MODEL_DIR` 可指定模型包。未指定时查找用户数据目录、安装目录和开发目录的模型包。
- 图片包图文件约 20.44 GB，视频包约 21.10 GB，准备工具以硬链接复用未变的 DiT 权重。文件完整性会在每次处理前核验，总耗时包含哈希、权重装载和计算。

无需权重的 AWA CPU/Vulkan 自测位于“测试记录”。“模型验收”保留 12 项严格要求及缺项；一次处理或自测成功不会签发模型证书。“报告与分享”可附加实际处理记录，下载 Discussion 草稿及原始报告。

## 独立 CLI

```sh
build/dev/seedvr2 run-video --model .cache/video-package-fp32 \
  --input /path/to/input.mp4 --output /path/to/new-video-result \
  --frames 17 --size 128 --backend vulkan --gpu 0 --seed 666
```

视频完整时序链路、包准备和复现命令见 [video-runtime.md](docs/video-runtime.md)。

```sh
build/dev/seedvr2 run --model .cache/image-package-fp32 \
  --input /path/to/input.png --output /path/to/new-result \
  --size 512 --backend vulkan --gpu 0 --seed 666

# 无需启动 Web，CPU 使用同一条完整处理链
build/dev/seedvr2 run --model .cache/image-package-fp32 \
  --input /path/to/input.jpg --output /path/to/another-result \
  --size 256 --backend cpu --threads 4
```

图片输出目录须不存在或为空，包含 `output.png`、与结果对齐的 `comparison-input.png` 和 `run.json`。标准输出为结果 JSON，标准错误输出进度与诊断。`--diagnostic-tensors` 额外保存中间张量供开发数值验证。CLI 运行结果保存在指定目录，不会自动加入 Web 队列。

```sh
build/dev/seedvr2 engine status
build/dev/seedvr2 engine devices
build/dev/seedvr2 engine self-test --backend cpu --save
build/dev/seedvr2 engine self-test --backend vulkan --gpu 0 --save
build/dev/seedvr2 plan --request examples/plan-720p.json --save
build/dev/seedvr2 models status
build/dev/seedvr2 models audit --bundle examples/model-evidence-empty
build/dev/seedvr2 history list --kind operator-test
```

空证据包审计退出 **6** 是预期结果，表示没有模型证书。`models status` 查询退出 0 只表示查询成功。规划记录为 `PLANNED`，与真实图像任务分开保存。CLI 的全局 `--database PATH` 置于子命令前，可与 Web 共用工作区。

## 构建

开发依赖：C++20、CMake 3.25+、Ninja、Python 3.12+、Node 24+；OpenSSL、SQLite、zlib、uuid、Vulkan headers/loader、glslang 与 SPIRV-Tools；新增 pkg-config 与 FFmpeg 开发库 libavformat/libavcodec/libavutil/libswscale，视频输出需要 libx264 编码器。运行时需要对应共享库 ABI，具体版本/许可在 `third_party/media`。导出还需要项目私有 PyTorch 环境，见[完整模型包准备](docs/image-runtime.md)。

```sh
python3 tools/prepare_native.py
python3 tools/prepare_engine.py --jobs 4
npm ci --prefix apps/studio --ignore-scripts
npm run build --prefix apps/studio
cmake --preset dev
cmake --build --preset dev --parallel 4
ctest --preset dev
cmake --install build/dev --prefix dist/native-dev
```

依赖准备显式联网，版本和归档 SHA-256 已锁定；缓存齐全后使用 `--offline`。CMake 不下载依赖。ncnn / pnnx 来自独立获取的官方提交 `6a1bf000f363714839a36793addc8c879d3d899e`，没有使用相邻本地工作树。这是已验证的固定版本，构建不会自动追随远端 HEAD。

只构建 CLI 和 worker：

```sh
python3 tools/prepare_native.py --cli-only
python3 tools/prepare_engine.py --jobs 4
cmake --preset cli
cmake --build --preset cli --parallel 4
```

该构建不寻找 Drogon、不构建 React，不要求 Node 或启动浏览器。

## 模型实测与架构

完整处理包含 VAE 编码、随机后验采样与缩放、patch 投影、32 层 DiT、自定义 AWA、单步 Euler endpoint 和 VAE 解码。图层按指定后端执行；请求 Vulkan 时不自动回退 CPU。预处理、张量布局、噪声和 Euler 标量运算在主机侧完成，并在报告中注明。

已完成两种尺寸的官方 FP32-B 全链路对比：每条轨迹 73 个中间结果逐元素通过，8 位输出最大相差 1。CPU 与最终优化后的 Vulkan 构建也对保留参考进行了复核。**这些是有明确范围的开发数值证据，不是图像质量验收或完整模型认证。** 视频新增时序 VAE：16/16 个 CPU/Vulkan 子模型检查通过；6 帧 64×64 完整轨迹 73/73 通过；17 帧 128×128 完整轨迹 Vulkan 60/73、CPU 70/73，严格判失败，最终解码张量通过不覆盖这些中间层失败。官方 BF16/FlashAttention、长视频和正式验收阈值尚未完成。

- [短视频运行时](docs/video-runtime.md) / [包含负结果的视频验证](docs/video-validation.md)
- [完整单图链路、模型包与复现命令](docs/image-runtime.md)
- [当前验证报告与限制](docs/image-validation.md) / [机器可读状态](docs/status.json)
- [应用框架和模块职责](docs/full-stack-framework.md) / [任务架构决定](docs/adr/0007-image-runtime-and-jobs.md)
- [严格模型验收](docs/model-validation.md) / [模型内部结构](docs/architecture.md)
- [本地 Web API](docs/local-web.md) / [OpenAPI](schemas/local-api.v1.openapi.json)
- [原生算子与导出设计](docs/native-backend.md) / [来源与许可](docs/provenance.md)

`video-*` 记录本轮视频扩展，`image-*` 保留 0.4.0 完整单图证据；`native-*`、`framework-*`、`foundation-*`、`web-*` 保留各自历史版本，不能用旧报告证明新构建。程序不会自动公开发布 Discussion。
