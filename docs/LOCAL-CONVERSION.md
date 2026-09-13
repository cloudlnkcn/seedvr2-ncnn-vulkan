# 从官方权重到本地应用

## 下载已转换模型（可选）

推荐先看 [首次使用的直接下载入口](FIRST-RUN.md#已转换模型与自动化范围)。支持公开的 [DiT FP16 存储包](https://huggingface.co/akashimio/SeedVR2-3B-ncnn-dit-fp16) 和 [FP32-B](https://huggingface.co/akashimio/SeedVR2-3B-ncnn)。在仓库根目录运行：

```sh
python3 tools/download_models.py --precision dit-fp16 --kind image \
  --output dist/tutorial/models/image
```

先构建当前原生程序；`--plan` 不需要程序或网络，`--offline` 验证本机已安装包，`--run input.png --result results/image` 可在安装后直接执行。视频改用 `--kind video --output dist/tutorial/models/video`。DiT FP16 是权重存储格式，激活与运算仍为 FP32，安装大小约 10.40 / 11.05 GB。固定版本见 [下载登记](distribution/huggingface-models.json)。下面保留底层 FP32 安装命令和源码转换/离线组装方法。


已验证的图片和时序视频 FP32-B 模型托管于 [Hugging Face：akashimio/SeedVR2-3B-ncnn](https://huggingface.co/akashimio/SeedVR2-3B-ncnn)。构建原生程序后，可直接安装模型，省去本机 PyTorch / pnnx 转换；源码转换入口继续保留。

```sh
python3 tools/model_distribution.py install --catalog docs/distribution/catalog.v1.json --kind image --base-url https://huggingface.co/akashimio/SeedVR2-3B-ncnn/resolve/9371e381e3d5581c05a0934a518b14d8aa15698b --output models/image
```

视频改用 `--kind video --output models/video`。下载固定到不可变提交，安装器支持续传和逐文件 SHA-256 校验。图片 / 视频安装约需 20.44 / 21.10 GB。模型下载工具只需 Python 标准库，原生推理不需要 Python。发布验证包括完整远端字节回读、两个模型包的原生校验及一张真实图片的 73/73 官方 FP32-B 对照；不扩大已有视频、精度或画质结论。

项目以源码、转换工具和验证记录交付。使用者在本机构建程序，从固定官方来源下载 checkpoint，再转换为 ncnn 模型包。预编译程序和转换模型的 Release 不作为使用或 CI 的前提。

## 首次使用

先按[教程第 1 课](TUTORIAL.md#1-从干净克隆开始)安装 Linux 构建依赖。CLI 与 Web 共用模型和 SDK；默认构建包含 Web，需要 Node.js 24 及 npm。只需要命令行时加 `--cli-only`。

```sh
python3 tools/build_native.py --check
python3 tools/build_native.py --jobs 2 --prefix dist/tutorial
```

另外安装 `uv`，转换脚本会用它准备 Python 3.13、PyTorch 2.9.0+cpu 和锁定导出依赖。先做不下载、不写文件的预检：

```sh
bash tools/convert_models.sh --check image models/image dist/tutorial/bin/seedvr2
bash tools/convert_models.sh image models/image dist/tutorial/bin/seedvr2
```

视频包使用相同入口，选择新目录：

```sh
bash tools/convert_models.sh --check video models/video dist/tutorial/bin/seedvr2
bash tools/convert_models.sh video models/video dist/tutorial/bin/seedvr2
```

官方 `.pth` 不能直接由 ncnn 加载。转换依次执行官方文件 SHA-256 校验、锁定 pnnx 构建、32 个 DiT block 与 VAE 导出、36 图模型包组装和原生身份/完整性校验。私有 Python 环境显式传给 pnnx 构建器，不依赖机器上已有的 `.venv-export`。

生成模型后，按[图片运行示例](TUTORIAL.md#4-接通图片的-36-张图)或[视频教程](TUTORIAL.md)运行 CLI。Web 可以直接指定两个模型：

```sh
dist/tutorial/bin/seedvr2-web --model models/image --video-model models/video
```

只准备了图片模型时省略 `--video-model`。浏览器地址以启动输出为准。

## 失败、资源与复用

- 相对路径统一从仓库根目录解析；带空格或 Unicode 的路径使用引号。目标模型目录必须不存在，脚本不会覆盖已有模型。
- 日志保存在目标目录同级的 `conversion.XXXXXXXX/conversion.log`；失败时打印阶段名称和退出码，并保留导出中间文件。
- 官方权重约 14.57 GB，下载支持续传并复用已校验文件。FP32 转换模型、中间结果、Python 和编译环境还需要额外磁盘空间；不能把下载大小当作总空间要求。新的完整流程资源峰值仍待测量。
- 下载支持续传，转换阶段暂不自动续跑。失败后依据日志处理问题；逐组件复用请按教程操作。视频入口目前仍重新导出共享 DiT；已有图片包的用户可以按教程复用其 32 个 DiT 图，避免重复导出。
- 最终校验失败时不将模型标记为可用，不自动将新哈希加入允许列表。工具链差异产生未知有效载荷时，应保存报告并审阅原因。
- 转换完成后保留原生安装目录及完整模型目录，即可在兼容环境离线推理。Python、PyTorch、pnnx 只用于转换和参考验证；应用运行不需要它们。迁移后仍应执行模型校验和设备自测。

## 验证边界

常规 [Native Linux delivery](../.github/workflows/native.yml) 检查构建、算子、CLI/Web/SDK 与安装，不下载完整权重，也不依赖 Release。

[手动完整模型工作流](../.github/workflows/model-validation.yml) 在 Ubuntu 24.04 从源码构建、下载官方权重、本机转换图片包，随后执行真实 CPU 图片修复与隔离网络的离线回放。它只保存 Actions 验证报告，不创建或发布 Release。耗时、内存、磁盘不足和转换失败均属于需要记录的结果；该新工作流尚未完整通过。

新脚本的帮助、参数与依赖预检不等于全模型验收。离线自身回放相同也不等于官方数值通过；完整数值、设备资格和修复质量仍分别依据对应实验报告。
