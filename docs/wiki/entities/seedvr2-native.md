# 本项目：SeedVR2 3B 原生应用

本页是代码与证据的导航，具体结构见 [架构与源码导航](../../ARCHITECTURE.md)。官方模型、原始源码、转换器和运行库分别锁定，文件哈希列在 [来源登记](../sources.json) 的 `official-model`、`official-code`、`runtime-converter` 和 `seedvr2-native`。

React / TypeScript / Ant Design 页面通过 Drogon 服务管理 SQLite 任务和事件；独立 worker 调用 C++ SDK。CLI 与安装后的 SDK 也调用同一套原生执行代码。公共 SDK 使用标准 C++ 类型，目前接收文件路径。Python 留在导出与参考工具中。

模型包有 36 张图：VAE encoder、patch-in、32 个 DiT block、patch-out、VAE decoder。运行时逐图创建 ncnn Net，按图装载和释放权重；当前每图边界有主机与设备传输。AWA 保留自适应窗口、clipped shifted window、Q/K 归一化与多模态 RoPE。没有自回归生成式 KV cache。

当前通过记录限定为 3B / FP32-B / 单步 CFG=1。[数值修复归档](../../../artifacts/2026-09-10/video-numerics-v2/summary.json) 中六条固定轨迹各通过 73/73 张量边界；它们覆盖指定图片/短片与 CPU、RTX 4060 Laptop 的组合，不能外推成所有输入与设备的证明。任务画质保留在同一归档和 [修复说明](../../NUMERICS-REPAIR.md) 中，三个自然短片的固定目标指标仍低于 bicubic。

视频使用整段时序 VAE 与三维注意力，当前长边最多 128、最多 17 帧，图片长边最多 512。输入输出细节见 [首次使用](../../FIRST-RUN.md)。对运行范围的修改需要新组件和流水线证据。

关联：[同类移植](peer-ports.md)、[数值与质量](../concepts/precision-and-evidence.md)、[视频与内存](../concepts/video-and-memory.md)、[本轮交付记录](../synthesis/delivery-2026-09-13.md)。
