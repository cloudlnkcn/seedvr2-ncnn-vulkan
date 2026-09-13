# 同类移植的来源入口

核对日期：2026-09-13。Tencent/ncnn 的 Discussion 是社区作者的项目分享；出现在官方仓库中不等于 ncnn 官方对整个应用或模型签发认证。以下是选定来源的阅读结果，没有在本机复现其他项目的大模型性能。

| 项目与入口 | 本轮固定版本 | 核对内容 |
| --- | --- | --- |
| [AiChiTuDouPian / SeedVR2-2b-ncnn](https://github.com/Tencent/ncnn/discussions/6988) | `e8f9294066dc9e2d903f9f3f645767dce56944e6` | README、DiT 图编排和 AWA。名称含 2b，说明中的模型是 3B；视频入口为逐帧独立处理；公开转换包与 BF16 路径 |
| [Chisato623 / seedvr2_ncnn](https://github.com/Tencent/ncnn/discussions/6968) | `16b2535198855fe79bac63f94919dbb382f00279` | VAE query 分块、时间状态、合并 DiT 图、媒体处理及共享投影代码 |
| [everythingfornothing / ernie-image-ncnn-vulkan](https://github.com/Tencent/ncnn/discussions/6998) | `77b4cbbd90bb49723efda95b9009c0bab4f0f6f5` | CUDA BF16、PyTorch FP32 与原生执行分层对照的说明及固定参考范围 |
| [futz12 / ernie-image-ncnn-vulkan](https://github.com/Tencent/ncnn/discussions/6996) | `8dcd6e4411137d8abe92c9d78581c4c96d5182c6` | 原生 CLI、BF16、可选提示增强和模型准备流程 |
| [nihui / zimage-ncnn-vulkan](https://github.com/nihui/zimage-ncnn-vulkan) | `c1938eef9cd7e03cb216da4f4c89a0c041543e7a` | CLI、低显存实现、跨平台 release workflow；作为维护者发布方式的例子 |

版本化原文 URL、SHA-256 和阅读范围见 [来源登记](../sources.json)。README 的能力声明、源码中实际存在的路径、作者报告的实测、本项目独立复现分别标注；未检查的路径保持未知。下载链接存在也不表示我们已经下载和运行过模型。

关联：[本项目](seedvr2-native.md)、[设计对照](../synthesis/design-comparison.md)、[数值与质量](../concepts/precision-and-evidence.md)、[视频与内存](../concepts/video-and-memory.md)。
