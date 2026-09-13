# DiT FP16 存储：Hugging Face 交付验证

公开模型库：[akashimio/SeedVR2-3B-ncnn-dit-fp16](https://huggingface.co/akashimio/SeedVR2-3B-ncnn-dit-fp16)。固定模型 revision `5c17b05641fbc84f16752b2f3c59aa417cf2475b`，模型卡更新不改变下载内容。操作见 [首次使用](../../../../docs/FIRST-RUN.md)。

| 证据 | 结果与范围 |
| --- | --- |
| [readback.json](readback.json) | 58 个准备资产完整下载到独立目录并通过 SHA-256；包括 53 个内容对象。回读时模型库为私有 |
| [image-install.json](image-install.json)、[video-install.json](video-install.json) | 用回读文件组装图片/视频包，原生身份与字节校验分别见对应 `*-native-verify.json` |
| [offline.json](offline.json) | 新网络命名空间、隐藏整个源码目录、Unicode 安装和输入输出路径、安装 SDK 外部使用程序完成真实 256×256 Vulkan 图片 |
| [offline-replay.json](offline-replay.json)、[offline-drift.json](offline-drift.json) | 77/77 张量哈希与冻结 FP16 执行一致；73 个模型边界继续对原生 FP32 / 官方 FP32-B 报告误差 |
| [web-integration.json](web-integration.json) | 下载模型的真实 Web/worker 图片执行，39 项导入、排队、取消、结果下载、重启恢复等检查通过；完整任务见 [web-actual-job.json](web-actual-job.json) |
| [publication.json](publication.json)、[public-download.json](public-download.json) | 已公开；匿名读取每个资产前 64 KiB，小文件读取完整。完整远端字节检查与匿名访问探测分别记录 |
| [download-entry.json](download-entry.json) | 统一入口计划、匿名补齐一个缺失图文件、原生校验、64px 图片运行及断网复用通过。其余文件复用已完整回读内容，此项不冒充第二次全量下载 |
| [network-retry.json](network-retry.json)、[range-fetch.json](range-fetch.json) | 记录 TLS 超时后的传输切换及分段补取；随后仍按完整 SHA-256 验证 |

该实验使用同一台 Linux 主机的系统库，不证明干净新机器或跨发行版二进制可移植。模型包校验不是模型画质认证，FP16 存储也不等于 FP16 激活或运算。五条完整轨迹、逐帧质量和资源数据见 [精度报告](../../../../docs/DIT-FP16-STORAGE.md)。
