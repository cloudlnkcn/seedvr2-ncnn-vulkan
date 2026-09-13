# 首次使用与离线搬移

0.7.0 原生预览，Linux x86_64。图片和短片通过同一套 C++ SDK 执行，CLI 不依赖 Web。已通过固定 SeedVR2 3B / FP32-B 的六条图片与短片数值轨迹，每条 73/73 个张量边界；适用设备、输入与画质结果见 [数值修复](NUMERICS-REPAIR.md)，资源边界见 [内存及算子验证](MEMORY-VALIDATION.md)。

**首次从 GitHub 克隆的读者请从 [教程第 1 课](TUTORIAL.md#1-从干净克隆开始) 开始。** 下文以教程构建默认输出 `dist/tutorial` 为安装目录。程序和权重需要分别准备，不随 Git 克隆自动提供。

2026-09-10 的后续数值修复冻结在维护者本机 `dist/numerics-v2` 安装，修复版 CPU/Vulkan 共享同一套 SDK。新读者按当前源码构建；旧 `dist/seedvr2-0.7.0` 目录是早期安装，不能仅凭目录中的版本号判断是否已包含修复。早期 17 帧数值修复和直接卷积的成本保留在 [历史记录](VIDEO-NUMERICS.md)，后续修复以 [NUMERICS-REPAIR](NUMERICS-REPAIR.md) 为准。

## 安装目录与启动

完成构建及模型准备后，`dist/tutorial/` 包含 CLI、本地 Web、worker、共享 SDK、头文件、CMake 配置，`models/image` 和 `models/video` 则保存另行安装的图片和视频包。进入该安装目录后运行：

```sh
./bin/seedvr2-studio
```

打开终端打印的本地地址，默认 http://127.0.0.1:8877/ 。已有服务占用端口时运行 `./bin/seedvr2-studio --port 0`，使用系统分配的端口。界面支持导入、队列、进度、取消、重新运行、对比和下载；每次执行一个任务，未完成队列最多八个。

启动脚本按自身所在的安装目录定位 `models/image` 和 `models/video`，所以也可以从任意工作目录使用脚本的完整路径启动。`SEEDVR2_MODEL_DIR` 与 `SEEDVR2_VIDEO_MODEL_DIR` 可指定其他已校验包。`--database "/路径/工作区.sqlite3"` 选择任务工作区。模型包和工作区分开保存，搬移模型不会自动搬移历史任务。

模型是独立文件，不内嵌在脚本或二进制里。启动脚本不会下载或转换模型，也不会自动打开浏览器：图片包缺失时会给出错误；视频包存在时才同时传给 Web。终端中按 Ctrl+C 可停止服务；只关闭浏览器不会停止服务。

## 已转换模型与自动化范围

当前已经审阅的完整 ncnn 包如下。自行转换可按教程执行，收到转换包后可使用[分发安装工具](distribution/README.md)直接部署，无需再次下载官方 checkpoint：

| 安装目录中的位置 | 模型包 | 大小 |
| --- | --- | --- |
| `models/image/` | `seedvr2-3b-image-fp32-b-v1`，36 张图 | 约 20.44 GB |
| `models/video/` | `seedvr2-3b-video-fp32-b-v1`，36 张图 | 约 21.10 GB |

每个包包含 `.param` 图结构、`.bin` 权重、常量和 `manifest.json`。这些大文件保存在本机忽略目录中，不随 Git 克隆或 `cmake --install` 自动获取；目前没有默认公开托管地址。2026-09-13 已增加 [转换包分发与安装工具](distribution/README.md)，可从离线分发目录或显式 HTTPS 镜像组装、断点续装和校验已转换模型。图片与视频共同的分发内容按哈希去重后约 21.44 GB；默认安装产生独立副本，空间另计。

源码目录已有分阶段自动化工具：

| 工具 | 已实现的工作 |
| --- | --- |
| `tools/seedvr2-studio`（安装后为 `bin/seedvr2-studio`） | 定位安装模型，启动本地 Web 与任务服务 |
| `tools/build_native.py` | 检查开发依赖，串起原生依赖准备、构建、小型测试和安装；不下载模型 |
| `tools/prepare_native.py`、`tools/prepare_engine.py`、`tools/prepare_pnnx.py` | 按各自锁文件准备原生依赖、ncnn 和转换器 |
| `tools/prepare_models.py` | 下载固定 revision 的官方 checkpoint，支持断点续传，校验文件大小与 SHA-256；已校验缓存会复用 |
| `tools/model_distribution.py` | 组装去重转换包，从离线目录或显式 HTTPS 镜像安装；保留原始模型身份，未配置公共默认镜像 |
| `tools/audit_native_install.py` | 核对冻结程序经 CMake RPATH 安装变换后的身份，盘点系统依赖，准备本机程序归档 |
| `tools/export_vae_image.py`、`tools/export_dit_block.py`、`tools/export_image_package.py` | 按顺序导出图片组件并组装 ncnn 图片包 |
| `tools/export_vae_video.py`、`tools/export_video_package.py` | 导出时序 VAE，并复用图片 DiT 组装视频包 |
| `tools/native_ci.sh` | 对已有构建和安装执行小型测试、参数边界、Mesa Vulkan 与外部 SDK 检查；完整 3B 实机验证另行执行 |
| `seedvr2 models verify` / `models copy` | 校验模型身份与文件完整性，或复制并复核离线模型包 |

从源码准备应用可运行 `python3 tools/build_native.py --jobs 2`，或执行下方分解命令。模型按教程中的图片/视频导出顺序准备；尚未提供一条命令完成空机器系统依赖安装、权重下载、转换和模型部署的总脚本。`prepare_models.py --list` 可先看下载清单，`--offline` 校验缓存；它下载官方权重，不会直接产出可运行的 ncnn 包。

## CLI：先检查，再执行

以下命令在安装目录执行。路径可以含中文和空格，使用引号包围即可。输出目录必须不存在或为空，已有结果不会被覆盖。

```sh
./bin/seedvr2 engine devices
./bin/seedvr2 engine self-test --backend cpu
./bin/seedvr2 engine self-test --backend vulkan --gpu 0

./bin/seedvr2 run --model models/image --input "/路径/照片.jpg" \
  --output "/路径/新结果" --size 256 --backend vulkan --gpu 0 --check

./bin/seedvr2 run --model models/image --input "/路径/照片.jpg" \
  --output "/路径/新结果" --size 256 --backend vulkan --gpu 0 --seed 666

./bin/seedvr2 run-video --model models/video --input "/路径/短片.mp4" \
  --output "/路径/视频结果" --size 128 --frames 17 --backend vulkan --gpu 0
```

`--check` 检查参数、输入解码、输出目录与空间、设备、包清单身份和工件路径/尺寸；**不会读取二十多 GB 权重完成哈希校验**。`READY_FOR_WEIGHT_VERIFICATION` 表示可以进入权重校验阶段。实际运行先验证全部文件的 SHA-256，再执行计算。也可单独运行：

```sh
./bin/seedvr2 models verify --kind image --model models/image
./bin/seedvr2 models verify --kind video --model models/video
```

`AUTHENTICATED_AND_INTEGRITY_CHECKED` 表示包的有效载荷属于项目审阅过的导出身份，且当前文件完整；它不是官方数字签名，也不证明画质或全部数值验收通过。`model_verified` 仍为 `false`。

标准输出是 JSON，标准错误显示阶段进度。Ctrl+C 请求取消；Web 的取消按钮停止独立 worker。成功图片输出为 `output.png`、对齐输入 `comparison-input.png` 和 `run.json`；视频对应 `output.mp4`、对齐输入视频及报告。`--diagnostic-tensors` 保存完整中间结果供独立对照，会增加磁盘使用。用 `--backend cpu --threads 4` 可选择 CPU；选择 Vulkan 后不会因设备不可用而静默改成 CPU。

## 输入、设备与资源范围

图片：PNG、RGB/YCbCr/灰度 JPEG，最大 32 MiB、32 M 像素；输出长边 64–512，16 的倍数，居中裁剪后短边至少 64。CMYK JPEG 需先转成 RGB。透明度、ICC、EXIF 方向不保留。JPEG 使用 libjpeg 的 ISLOW 和 fancy upsampling，与保留的 Pillow 解码样例逐像素一致。

视频：输入最大 256 MiB，8 位 SDR、方形像素、方向已转正；处理开头 1–17 帧，输出长边 64–128，无音轨 MP4。使用整段时序 VAE 和三维注意力；不具备长视频缓存、分块或音频回填。更大尺寸的规划记录不代表运行引擎已经支持。

实机范围是本机 RTX 4060 Laptop 8 GB 与 CPU。软件 Vulkan 只执行小型 CI 测试。预检不能精确预测整个任务的显存峰值，设备分配失败会保留错误；8 GB 不是其他设备的统一保证。当前 FP32 图片包约 20.44 GB，视频包约 21.10 GB，两份独立离线副本合计约 41.54 GB；工作区和诊断张量另计。

Vulkan 默认 `--weights auto`，按当前图的权重估算与设备预算选择 GPU 或主机内存；日常使用无需手动设置。它不保证整个任务不会内存不足，也不包含激活卸载或分配失败后的重试，详见 [0.7.0 测量与边界](MEMORY-VALIDATION.md)。

默认 `--weight-io buffered`。可选 `mapped` 是 Linux 实验项：本轮减少了匿名内存，但未显示速度收益，详见 [测量记录](DELIVERY-RESULTS.md)。运行期间请保持模型文件不变。

## 从源码准备

需要 C++20、CMake 3.25+、Ninja、Python 3.12+、pkg-config、OpenSSL、SQLite、JPEG、FFmpeg、Vulkan/glslang/SPIRV-Tools 开发包。构建 Web 还需 Node 24、zlib 和 uuid；运行时无需 Python/Node。

```sh
python3 tools/prepare_native.py --jobs 4
python3 tools/prepare_engine.py --jobs 4
npm ci --prefix apps/studio --ignore-scripts
npm run build --prefix apps/studio
cmake --preset release
cmake --build --preset release --parallel 4
ctest --preset release
cmake --install build/release --prefix dist/tutorial
```

依赖脚本按锁文件显式下载；已有完整归档缓存时加 `--offline`。CMake 自身不下载依赖。只构建 CLI/SDK 时，准备依赖加 `--cli-only`，CMake 设置 `-DSEEDVR2_BUILD_WEB=OFF`。转换模型另需专用 PyTorch/pnnx 环境，完整命令见 [图片导出](image-runtime.md) 和 [视频导出](video-runtime.md)。转换器与运行库分别锁定，重新构建运行库不要求无理由重导出已验证模型。

## 搬移安装和模型

运行原生模型复制功能会校验源目录，复制到新的临时目录，再校验副本，最后将其改名为目标目录；失败或取消会清理本次临时副本，已有目标不会覆盖。

```sh
./bin/seedvr2 models copy --kind image --model "/原目录/图片包" --output models/image
./bin/seedvr2 models copy --kind video --model "/原目录/视频包" --output models/video
```

完整复制安装目录中的 `bin/`、`lib/` 或 `lib64/`、`include/`、`share/`、`models/`，保留共享库符号链接。二进制通过相对路径寻找 SDK。复制完成后在目标机先执行 `engine self-test` 和 `models verify`，再运行小尺寸输入。

本轮在新网络命名空间中，把安装目录映射到含中文的全新路径，隐藏整个源码目录、原模型、开发环境，清空 `LD_LIBRARY_PATH`，成功构建外部 SDK 使用程序并完成真实图像推理。**系统共享库仍来自同一台 Fedora 主机**；该结果不证明二进制可以直接跨 Linux 发行版、Windows 或 macOS 运行。目标机需提供匹配的 C++、FFmpeg、OpenSSL、SQLite、JPEG、Vulkan 等 ABI 和 GPU 驱动。

## C++ SDK

```cmake
find_package(seedvr2 0.7 CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE seedvr2::sdk)
```

配置外部工程时设置 `-DCMAKE_PREFIX_PATH=/安装目录`。公共入口是 `seedvr2/pipeline.hpp`，以 `RestoreRequest` 调用 `preflight()` 和 `restore()`，通过回调接收进度和请求取消。可编译示例在 [examples/sdk](../examples/sdk)。调用方无需安装 ncnn、CLI11 或 JSON 头文件。当前 SDK 为 0.7 预览 ABI，旧版调用方需重新构建；复用外部示例构建目录时使用 `cmake --fresh` 清除旧安装的路径缓存。同一进程的并发多任务推理尚未资格验证，当前应用逐任务执行。
