# FP32-B 视频数值修复证据

2026-09-08，SeedVR2 3B。最终同一份构建在保留的 17 帧、128×128 合成短片上，CPU 和 Vulkan 各通过 73/73 项原有开发协议。模型、输入、官方参考、随机噪声和容差均保持不变。完整模型认证仍未通过。

- [summary.json](summary.json)：结果、修复成本、未合入候选、已知限制和测试范围。
- [index.json](index.json)：轨迹和诊断报告入口，以及本机原始数据目录。
- [manifest.json](manifest.json)：本目录文件 SHA-256；不包含自身。
- [修复说明](../../../docs/VIDEO-NUMERICS.md)：相同输入定位过程、来源锁定、复测方法及源码导航。
- [安装后的实际网页任务](installed-web/verification.json)：36 个模型子图全 Vulkan，下载结果与冻结 CLI 一致，旧任务保留。
- [native-ci](native-ci/)：本机原生小型 CI、Mesa Vulkan、安装后 SDK；没有把大模型实机实验算进 CI。

`runs/` 同时保留 66/73、70/73 的失败与最终通过结果；`implementations/` 保存各次源码快照和二进制哈希。`rejected-preprocessing/` 是未合入的 FMA 插值实验，默认构建不使用它。`installed-web/initial-checker.py` 与错误日志保留首次检查器文件名错误；`verify_result.py` 仅核对已保存的成功结果，不会提交新任务。

大模型、完整张量和冻结二进制仍位于索引记录的 `.cache/` 路径，不随 Git 提交；首次克隆后不能只靠这些摘要重新计算 73 项误差。`preview/` 提供该合成短片和一张自然照片的实际输出，不能用作代表性画质证明。计时包含校验与文件读取，页缓存和桌面负载未受控，不据此声称整体加速。
