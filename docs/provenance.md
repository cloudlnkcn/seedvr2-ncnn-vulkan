# 来源与派生范围

当前版本 0.6.0-native-preview，2026-09-08。当前实现与独立版本证据见 [DELIVERY-RESULTS.md](DELIVERY-RESULTS.md)，此前图像/视频证据保留各自身份。native-*、foundation-*、web-*、framework-* 是以前版本的历史记录；其“未链接 ncnn”等结论不描述当前版本。

## ncnn 与 pnnx

本轮从 [Tencent/ncnn 官方远端](https://github.com/Tencent/ncnn/commit/3b7bdba7fc8aea8fd46779533eee027df77c639d) 查询 HEAD，运行库锁定 `3b7bdba7fc8aea8fd46779533eee027df77c639d`，身份在 `engine-dependencies.lock.json`。转换器独立固定 `6a1bf000f363714839a36793addc8c879d3d899e`，身份在 `converter-dependencies.lock.json`。两者均为未修改的官方源码，准备脚本复用缓存前逐文件与已验证归档比对，没有链接相邻的 ncnn 工作树。旧模型包的转换器字段仍描述当时导出，兼容新运行库不等于重新导出。

本机 ncnn 构建启用 Vulkan、系统 glslang，关闭 INT8 和 AVX512。pnnx 由项目私有 Python 3.13 / PyTorch 2.9.0+cpu 开发环境构建，身份记录在 `.deps/bin/pnnx-build.json`。Python 和 pnnx 不进入应用运行时。

系统 glslang / SPIRV-Tools、Vulkan loader/driver 仍是平台依赖；本轮没有锁定整个系统工具链。`third_party/engine` 保存 ncnn、glslang、SPIRV-Tools 的许可证与来源哈希。其 license manifest 不宣称锁定系统库二进制版本。当前安装目标包含这些声明。

## SeedVR 官方代码与权重

参考来源是 [ByteDance-Seed/SeedVR](https://github.com/ByteDance-Seed/SeedVR/tree/e4de8c24441a67e1b7df56abea10645059bb1185)，提交 `e4de8c24441a67e1b7df56abea10645059bb1185`。使用的 attention、RoPE、窗口、DiT block、modulation、embedding、MLP、VAE 类和配置原文保存在 `tests/reference/seedvr`；每个文件的 SHA256 在 `sources.json`，许可证保持原文。旧窗口基线另外保存在 `tests/reference/official_window.py`，其 SHA256 为 `61b3e0235e25d3e901271186bbaa43a90dcc1552f003daa3d2844b5126ce7e84`。

官方权重来自 [ByteDance-Seed/SeedVR2-3B 的固定 revision](https://huggingface.co/ByteDance-Seed/SeedVR2-3B/tree/37255ff8cccfb01071b87f635a5948ca8d53117c)。`model-sources.lock.json` 固定四个文件：3B DiT、VAE、正向与负向文本 embedding；实际下载后逐个核对官方 LFS SHA256 和精确字节数。原文件位于忽略目录 `.cache/models`，不进入 Git。0.6 本机安装的 `models/image`、`models/video` 包含经过校验的独立导出副本，未公开上传。

`tools/awa_reference.py`、`vae_reference.py`、`dit_block_reference.py`、`image_reference.py` 执行上述原始类的方法体，对分布式/缓存外壳、Apex RMSNorm、FlashAttention 及 BF16 转换做显式 FP32-B 适配。原文不修改，适配逻辑可独立审查。AWA 使用合成 projected QKV/RMS；VAE 与 DiT 块使用真实官方 checkpoint。候选导出模块独立实现后与参考比较，不能把候选自己的输出作为 golden。

`src/planning/windows.cpp` 是窗口算法的 C++ 移植；`src/engine/ncnn/awa.cpp` 及 shaders 实现 Q/K RMSNorm、局部 RoPE、窗口 joint attention、视频逆散射和文本等权平均。`tools/vae_image_module.py` 是官方视频 VAE 的 T=1 特化；`dit_block_module.py` 是完整块的 pnnx 可导出表达。实际 lowering、派生范围和已发现的上游 Vulkan buffer 问题见 native-backend.md。本轮没有复制其他个人移植项目的 C++ 或 shader。

数值结论覆盖注明范围的 FP32-B 子模型、完整单图和短片 32 层轨迹；短片结果为 MIXED，17 帧轨迹仍有中间层严格失败。官方 BF16 / FlashAttention 路径 A、长视频恢复和模型阈值校准尚未完成；权重来源与哈希验证也不构成模型证书。

## 应用框架

`dependencies.lock.json` 固定 [nlohmann/json 3.12.0](https://github.com/nlohmann/json/releases/tag/v3.12.0) 单头文件与 MIT 许可证。CMake 校验头文件 SHA256；JSON 类型不进入公共 C++ 接口。旧 cpp-httplib 0.54.1 与 HTML 原型保留在历史目录，不参与当前构建。

`native-dependencies.lock.json` 固定 Drogon 1.9.13、匹配的 trantor 子模块、JsonCpp 1.9.8、CLI11 2.7.2 的精确提交、归档地址和 SHA256。准备脚本显式联网，CMake 本身不下载。OpenSSL / SQLite / zlib / uuid 属于系统依赖；当前程序未被认定为跨 Linux 发行版便携包。

前端依赖锁在 `apps/studio/package-lock.json`；生产依赖许可证汇总在 `third_party/studio/NOTICES.txt`，补充原文及哈希在 `license-sources.json`。生产资源嵌入 C++ Web 主程序，运行时不从 CDN 加载。模型证据审计器、工作区、自测与报告逻辑属于本项目实现；框架合成测试与模型数值证据分开保存。

## 发行状态

本轮已形成可安装的 SDK/CLI/Web、独立模型副本和同系统 ABI 的离线验证，没有发布公开权重包、跨系统便携发行版或外部 Discussion。第三方许可证随安装附带；公开发行仍需项目自身许可决定、完整运行库与模型分发资格、目标平台实测。本说明不代替这些尚未完成的事项。

## 0.6.0 JPEG 与自然样例

当前 JPEG 解码链接系统 libjpeg-turbo 3.1.3，以 ISLOW 和 fancy upsampling 对齐独立 Pillow 像素参考；PNG 继续使用固定运行库附带的 stb。`third_party/media/libjpeg/` 保存本机许可原文及哈希。自然照片原件来自 scikit-image 官方固定标签的数据集，为 NASA 公有领域图像；输入、受控降质与派生夹具身份在 `tests/fixtures/natural/provenance.json`，不以该单例证明普遍画质合格。

## 0.4.0 图像 IO 与上游检查

PNG/JPEG IO 使用固定 ncnn 提交附带的 stb_image.h / stb_image_write.h，MIT 许可原文在 `third_party/engine/stb-MIT.txt`，随安装附带。完整单图参考新增原文文件及 SHA-256，包括 NaDiT、patch、sampler、schedule 和预处理代码，清单共 32 个文件。

2026-09-07（本地日期）再次读取官方 HEAD 时，上游已前进到 `88a29d8ef42be81b0d67f5fc53ba7695d380cc64`（父提交即本版固定版本；VS2026 CI 和 x86 相关改动）。本版继续使用本轮已完成数值验证的 `6a1bf000f363714839a36793addc8c879d3d899e`；不宣称仍是当前 HEAD。上游提交：[88a29d8](https://github.com/Tencent/ncnn/commit/88a29d8ef42be81b0d67f5fc53ba7695d380cc64)。

## 0.5.0 时序 VAE 与视频 IO

`tools/vae_video_module.py` 保留官方 VAE 的时间卷积、首帧语义、逐帧 norm/attention 和时间相位。`video_layers.cpp` 与 video shaders 为本项目实现，使用固定且未修改的官方 ncnn 2D Convolution/SDPA。完整轨迹独立参考见 `tools/check_video.py`，实际导出和负结果见 [视频验证](video-validation.md)。

媒体 IO 直接链接本机 FFmpeg 8.0.1 的 libavformat/libavcodec/libswscale/libavutil；libx264 提供 MP4 H.264 编码。`third_party/media/sources.json` 记录实际 Fedora 包、GPL 配置、版本和原文许可哈希，安装附带该目录。没有打包系统共享库，也未完成完整应用许可与便携分发验收。视频测试输入由本机 FFmpeg testsrc2 生成，原始命令和哈希另存，不使用它声称自然视频画质通过。
