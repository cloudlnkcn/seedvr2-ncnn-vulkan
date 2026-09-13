# 转换模型分发与安装

## 下载已转换模型（可选）

推荐先看 [首次使用的直接下载入口](../FIRST-RUN.md#已转换模型与自动化范围)。支持公开的 [DiT FP16 存储包](https://huggingface.co/akashimio/SeedVR2-3B-ncnn-dit-fp16) 和 [FP32-B](https://huggingface.co/akashimio/SeedVR2-3B-ncnn)。在仓库根目录运行：

```sh
python3 tools/download_models.py --precision dit-fp16 --kind image \
  --output dist/tutorial/models/image
```

先构建当前原生程序；`--plan` 不需要程序或网络，`--offline` 验证本机已安装包，`--run input.png --result results/image` 可在安装后直接执行。视频改用 `--kind video --output dist/tutorial/models/video`。DiT FP16 是权重存储格式，激活与运算仍为 FP32，安装大小约 10.40 / 11.05 GB。固定版本见 [下载登记](huggingface-models.json)。下面保留底层 FP32 安装命令和源码转换/离线组装方法。


已验证的图片和时序视频 FP32-B 模型托管于 [Hugging Face：akashimio/SeedVR2-3B-ncnn](https://huggingface.co/akashimio/SeedVR2-3B-ncnn)。构建原生程序后，可直接安装模型，省去本机 PyTorch / pnnx 转换；源码转换入口继续保留。

```sh
python3 tools/model_distribution.py install --catalog docs/distribution/catalog.v1.json --kind image --base-url https://huggingface.co/akashimio/SeedVR2-3B-ncnn/resolve/9371e381e3d5581c05a0934a518b14d8aa15698b --output models/image
```

视频改用 `--kind video --output models/video`。下载固定到不可变提交，安装器支持续传和逐文件 SHA-256 校验。图片 / 视频安装约需 20.44 / 21.10 GB。模型下载工具只需 Python 标准库，原生推理不需要 Python。发布验证包括完整远端字节回读、两个模型包的原生校验及一张真实图片的 73/73 官方 FP32-B 对照；不扩大已有视频、精度或画质结论。

本目录的 [catalog.v1.json](catalog.v1.json) 记录项目已审阅的 SeedVR2 3B 图片和视频 FP32-B 包，保留原始 manifest 和文件哈希。它随源码提供；权重本体不进入 Git。

**已提供上面的 Hugging Face 转换模型下载；原生程序仍按源码构建。** 获取官方 checkpoint 仍使用 [教程](../TUTORIAL.md) 和 `tools/prepare_models.py`。这里的工具用于安装已经转换好的包，省去接收者重新转换的步骤。

## 接收离线分发目录

需要项目源码中的 Python 工具和一个包含 `catalog.json`、`objects/`、`LICENSE` 的完整分发目录。普通安装会复制文件，图像与视频分别需要约 20.44 / 21.10 GB；分发目录自身另占约 21.44 GB。

```sh
python3 tools/model_distribution.py install \
  --catalog docs/distribution/catalog.v1.json \
  --from-dir "/路径/转换模型分发目录" \
  --kind image --output dist/tutorial/models/image --offline

python3 tools/model_distribution.py install \
  --catalog docs/distribution/catalog.v1.json \
  --from-dir "/路径/转换模型分发目录" \
  --kind video --output dist/tutorial/models/video --offline

dist/tutorial/bin/seedvr2 models verify --kind image --model dist/tutorial/models/image
dist/tutorial/bin/seedvr2 models verify --kind video --model dist/tutorial/models/video
```

安装前可加 `--plan` 查看文件数和逻辑大小，不写入输出、不联网。安装写入相邻 `.partial` 目录，只有完整清单及 SHA-256 都通过后才改名。中断时保留已校验文件，使用相同命令继续；损坏文件不会被当作缓存命中，也不会覆盖已经存在但不匹配的最终目录。一个目标目录只允许一个安装进程写入。

同文件系统可显式加 `--hardlink` 复用分发目录中的文件，避免再次占用权重空间。此时各目录引用相同文件内容，必须保持源包、分发目录和安装文件不被原地修改；需要独立可写副本时使用默认复制。哈希仍会完整检查。

## 维护者准备分发目录

```sh
python3 tools/model_distribution.py stage \
  --image .cache/image-package-fp32 --video .cache/video-package-fp32 \
  --output dist/model-distribution-v1 --hardlink
```

输出按 SHA-256 存储对象，图片与视频相同的 DiT 权重和图参数只保留一份。各包的路径在安装时恢复；不能把 `objects/` 直接传给 ncnn。组装器先核对 `policies/reviewed-packages.json` 的完整 payload 身份，再检查每个文件。转换参数或模型数学改变后，需要新的审阅和新的清单，不能只生成一套新哈希绕过身份检查。

发布时上传整个分发目录，保留原模型许可与来源。确认远端所有对象后记录固定 revision，再公布安装命令。工具支持显式的 `--base-url` HTTPS 根地址和断点续传，替代 `--from-dir`；地址应指向固定 revision，目录下必须包含 `objects/<sha256>`。公共镜像采用上面的固定 Hugging Face revision；不要用可变 main 替代锁定提交。

## 程序分发的边界

```sh
python3 tools/audit_native_install.py --prefix dist/numerics-v2 \
  --output artifacts/local/native-install.json \
  --archive dist/seedvr2-fedora-host.tar.gz
```

该命令面向可信的本机安装：比对冻结数值实验的 CLI/SDK 哈希、记录安装清单与动态库名称，可生成不含模型和系统库的本机归档。它不解决系统 ABI 或许可证材料的交付，因此报告分别给出 `passed` 与 `portable_release_ready`。当前后者为 `false`；发布跨发行版程序需要独立构建配方、实际 FFmpeg 配置及相应材料、干净目标机与 GPU 验证，见 [许可证说明](../LICENSING.md)。

本轮真实安装和数值回放入口见 [交付记录](../wiki/synthesis/delivery-2026-09-13.md)。模型包完整性、数值参考通过和画质结果保持独立。
