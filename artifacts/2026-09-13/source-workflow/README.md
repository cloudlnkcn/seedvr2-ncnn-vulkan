# Sources → delivery → evidence，2026-09-13

基础源码为 `65aa871ccfeaddb1b0b151428770feb8aed00730`。本轮修改知识/分发/安装工具和说明，模型运行代码、权重与既有参考未变。操作入口见 [Wiki](../../../docs/wiki/README.md) 与 [分发说明](../../../docs/distribution/README.md)。

| 层次 | 实测结果 | 原始记录 |
| --- | --- | --- |
| 小型合同测试 | 33/33，0 失败、0 跳过 | [结构化汇总](contracts.json)、[日志](contracts.log) |
| 来源与链接 | 11 组来源、7 个页面；离线 lint 通过 | [lint](knowledge-lint.json)、[固定远程原文回读](ingest-aichi.json) |
| 源码交付 | 文档路径、官方/第三方文件哈希及受跟踪文件清单检查通过 | [检查](source-package.json) |
| 模型组装 | 2 包、53 个去重对象、21,441,543,211 字节 | [组装](model-stage.json)、[目录](../../../docs/distribution/catalog.v1.json) |
| 安装工具 | 图片/视频各恢复 75 个文件 | [图片](image-install.json)、[视频](video-install.json) |
| 原生包身份 | 两包均为 AUTHENTICATED_AND_INTEGRITY_CHECKED | [图片](image-native-verify.json)、[视频](video-native-verify.json) |
| 程序身份 | 冻结构建经 CMake RPATH_CHANGE 后逐字节等于安装文件；本机归档 53,808,289 字节 | [文件清单与依赖](native-install.json) |
| 实际运行 | 新程序路径及新模型路径的 256×256 图片执行成功 | [运行](image-run.json) |
| 完整张量对照 | 73/73，atol=rtol=0.001，官方参考像素最大误差 1 | [回放](image-replay.json) |
| 被拒绝的证据 | 首次误选历史 baseline，参考绑定失败，0 张量获准比较 | [原始失败](image-replay.invalid-reference.json) |

## 复现命令与边界

小型测试不需要大模型：

```sh
python3 -m unittest discover -s tests -p test_model_download.py -v
python3 -m unittest discover -s tests -p test_model_distribution.py -v
python3 -m unittest discover -s tests -p test_knowledge.py -v
python3 -m unittest discover -s tests -p test_native_install_audit.py -v
python3 tools/knowledge.py lint
python3 tools/check_source_package.py
```

本机模型组装/安装使用硬链接，保持所有源文件不变：

```sh
python3 tools/model_distribution.py stage \
  --image .cache/image-package-fp32 --video .cache/video-package-fp32 \
  --output dist/model-distribution-v1 --hardlink
python3 tools/model_distribution.py install \
  --catalog docs/distribution/catalog.v1.json --from-dir dist/model-distribution-v1 \
  --kind image --output 'dist/交付验证 模型/image' --hardlink --offline
python3 tools/model_distribution.py install \
  --catalog docs/distribution/catalog.v1.json --from-dir dist/model-distribution-v1 \
  --kind video --output 'dist/交付验证 模型/video' --hardlink --offline
```

组装输出目录必须新建，安装命令允许校验命中已有正确目录；独立复制模式由小型合同测试覆盖。分发内容与普通安装的逻辑大小不同，硬链接也不代表独立副本。

程序从本机归档解开到 `dist/交付验证 程序/seedvr2-host`，使用同一台 Fedora 44 的系统库。下面命令记录了实际推理和纠正后的参考目录；原始大模型与大张量缓存不随 Git 发布，已有目录需使用新输出名：

```sh
'dist/交付验证 程序/seedvr2-host/bin/seedvr2' run \
  --model 'dist/交付验证 模型/image' \
  --input tests/fixtures/natural/astronaut-degraded.jpg \
  --output .cache/distribution-replay-2026-09-13 \
  --size 256 --backend vulkan --gpu 0 --threads 4 --seed 666 --diagnostic-tensors
python3 tools/check_image_outputs.py \
  --run .cache/distribution-replay-2026-09-13 \
  --reference .cache/delivery-natural-reference \
  --reference-run '.cache/delivery-offline-v2/结果 自然图像' \
  --output artifacts/local/image-replay.json
```

输出 PNG SHA-256 为 `2669e07499ce9db0c44cfa861bde9e32bfa993a33df1b66091b52cdece200b2f`，与此前冻结结果一致。报告中的约 30.94 秒是这一次带诊断执行的运行时间，没有做重复统计，不作为性能优化收益。

`native-install.json` 的 `passed` 表示本机安装检查成功；`portable_release_ready=false` 明确保留跨发行版、系统媒体库及相应发布材料的缺项。没有公开模型镜像或程序 Release。完整六条旧轨迹、视频画质负结果以及设备能力范围继续引用 [2026-09-10 记录](../../2026-09-10/video-numerics-v2/README.md)。
