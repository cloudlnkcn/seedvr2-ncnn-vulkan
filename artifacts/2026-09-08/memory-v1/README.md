# 权重内存验证证据索引

当前实现说明与实测分析见 [MEMORY-VALIDATION.md](../../../docs/MEMORY-VALIDATION.md)。此目录保留成功、失败和被作废的测试，不能用扫描到的 PASS 字符串代替完整验收。

当前数值候选以 `source-bindings-v2.json` 中的生产源码和安装二进制为准：CLI SHA `2c2d08ac27d1a35ce2f2a8a92d32c307f7774ab00398999b6173d672893112b6`，SDK SHA `ff86dcd0b05ea66f5ab384701701aced273582e0ee95fc0cfa12c9c4608f30f2`。随后修改的是安装验证脚本、SDK 使用示例版本要求和说明；未修改候选图计算或内存策略。

| 文件 | 含义 |
| --- | --- |
| `native-ci-v2/` | 当前安装验证，包括 14 项 CTest、11 项参考合同、5 项 Mesa Vulkan、参数边界及外部 SDK 实际库身份检查 |
| `gemm-final.json` | NVIDIA 实际四次 GEMM、受控预算、精确数学期望和权重内存类型 |
| `awa.json` | 20 个官方输入 × CPU/Vulkan，40 次小算子对照 |
| `components/summary.json` | 27 次真实 DiT/VAE 对照、逐次策略和输出一致性；同目录六份详细报告 |
| `video-*-parity.json` | 两种策略的完整 73 边界官方参考重放 |
| `video-equivalence.json` | 77 份诊断张量及 MP4 在两种策略之间逐字节一致，且核对实际库及日志 |
| `video-*-measurement.*` | 逐次测量、stdout、stderr；显存为含桌面的整卡口径 |
| `image-sdk-fixed-*`、`image-sdk-validation.json` | 正确新 SDK 的完整自然图验证和日志身份 |
| `first-use-real-package.json` | 20 项真实模型包/首次使用/内存参数边界检查 |
| `web.json` | 58 项本地 HTTP/界面资源/真实 AWA 合同 |
| `live-web-*.json` | 当前 0.7.0 本地安装、历史保留、真实 worker 视频任务和最终库/输出身份 |
| `installed-*-package.json` | 独立 Btrfs 写时复制模型副本的完整文件校验；不是模型质量认证 |

## 保留的失败和无效范围

- `gemm-upstream-validation.*`：原版固定 ncnn 的主机内存导入声明、复制源用途两个 VUID。数值通过不能覆盖这些错误。`gemm-fixed.*` 为初次修正后的结果，当前小测试见 `gemm-final.json`。
- `ctest.log`：新 GEMM 测试最初把存储填充当作逻辑元素数，13/14；数值期望未修改。
- `mesa.log`：新测试最初未使用 Net 按设备修正后的选项，发生绑定数量不匹配和崩溃；修正后 5/5，未跳过。
- `sdk-cache-failure.json`、`native-ci/sdk-*`、`image-sdk-auto-*`：首次外部示例错误加载旧 `.so.0`，数值结果仅属于旧 SDK，**不能计入 0.7.0 验收**。后续 `native-ci-v2`、`image-sdk-fixed-*` 才是当前结果。
- `installed-video-wrong-kind.*`：调用包验证时未指定 `--kind video`，图片验证合同正确拒绝视频包；显式指定类型后的结果为 `installed-video-package.json`。
- `source-bindings.json`、`implementation.patch`：第一轮开发快照，补丁不包含当时未跟踪文件；不是最终源码交付包。最终完整源码由本仓库提交和 `source-bindings-v2.json` 的逐文件身份确定。

原始权重、官方原始张量及较大的候选诊断张量保存在忽略目录 `.cache`；其路径和哈希在相应报告中。复制到本目录的 JSON、PNG/MP4 用于复核和展示，不将候选输出重新登记为官方参考。
