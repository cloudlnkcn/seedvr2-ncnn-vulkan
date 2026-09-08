# 模型执行边界

0.3.0 已有实际 ncnn CPU/Vulkan 诊断执行；完整恢复管线仍待实现。详见 [原生后端与证据](../../docs/native-backend.md)。

- `ncnn/awa.cpp` 与 `shaders/`：自定义 AWA 全计算，FP32、真实 ncnn SDPA，GPU gather/scatter/text mean。
- `ncnn/constant.cpp`：带正确 Vulkan transfer usage 的调制常量，保持 pnnx 二进制排列。
- `ncnn/runtime.cpp`：哈希用例、VAE/DiT 块逐层指定后端、GPU 设备、内置 AWA 自测；无整网修复入口。
- 公开接口只使用标准 C++ 类型，HTTP/React/SQLite 不进入 engine。
- Python 仅用于开发侧参考与 pnnx 导出，不是部署运行依赖。
- 后续 `pipeline/` 应接入 32 层串联、噪声/sampler、VAE latent 语义、预后处理。视频因果缓存必须单独验证。

worker 目前仍只有能力查询。Web 自测直接调用有界 application 用例，不伪装成已完成的独立 worker 或持久任务队列。
