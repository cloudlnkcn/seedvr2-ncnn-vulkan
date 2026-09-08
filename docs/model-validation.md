# SeedVR2 严格模型验收

版本 1.0-draft，2026-09-06。用户要求认证模型本身，而不仅是工程项目。此处定义项目的模型验收结论及证据条件。**当前模型状态为 NOT_RUN / 未认证，阈值 NOT_FROZEN，程序没有签发模型证书的代码路径。**

2026-09-07 / 0.4.0 实现检查点：官方权重已校验，完整单图链路已贯通全部 32 个 DiT block。两种尺寸与原始官方类的 FP32-B 参考逐元素对比，每条轨迹 73 项均通过，CPU 与最终 Vulkan 构建也复核了保留参考；详见 [单图验证](image-validation.md)。整网开发诊断容差为 `1e-3 + 1e-3*abs(reference)`，8 位输出最大误差为 1。前一轮算子/子模型证据保留在 [原生后端](native-backend.md)。这里的 NOT_RUN 指正式验收包与策略门槛，没有否定这些已经执行的开发对比；它们尚未满足下述 12 项要求，也没有改变策略冻结或证书签发状态。

## 1. 三种结论必须分别保存

| 结论 | 所回答的问题 | 不能替代 |
| --- | --- | --- |
| 工程可靠性 | 能否构建、协议是否一致、记录是否持久、程序是否越界 | AWA / VAE / 整网是否数值正确 |
| 模型实现一致性 | 在固定输入与配置下，是否忠实实现官方模型语义 | 修复画面是否优于输入或其他模型 |
| 修复质量与适用性 | 在指定数据集、设备、资源下是否达到目标 | 不同模型、驱动、精度、分块模式的全面保证 |

认证范围为模型权重 + 导出产物 + runtime/ncnn/shader + 设备/驱动 + 精度 + shape profile + chunk policy + 数据集/噪声身份。修改其中任一项就必须重新判断哪些证据失效。不能给一个模型名字永久绿色勾。

## 2. 两条独立参考

**A：冻结官方执行路径。** 固定 SeedVR commit `e4de8c24441a67e1b7df56abea10645059bb1185`、3B 权重清单、配置、BF16、Apex/FlashAttention 和实际运行环境。记录全部包装修改。官方代码：[SeedVR](https://github.com/ByteDance-Seed/SeedVR/tree/e4de8c24441a67e1b7df56abea10645059bb1185)。

**B：显式 FP32 数学参考。** 同一份权重、窗口和时间语义，拆开 QKV/norm/RoPE/attention/text mean，便于定位算子误差。改变 Apex、噪声生成或 VAE posterior 包装的运行不得标为未经修改的 A。

A 用于复现发布路径，B 用于定位数学/实现错误；A/B 之间的差异也要记录。不能用候选 C++ 自己生成“参考答案”。同 seed 不够：保存 posterior noise、DiT noise 的实际 tensor、dtype、shape、hash；参考与候选消费同一随机输入。

## 3. 验收矩阵

机器可读策略在 [policies/seedvr2-3b.v1.json](../policies/seedvr2-3b.v1.json)。12 项均为必需；当前文件清单只支持收集及诊断，不能自动满足下表的完整语义。

| ID | 模型要求 | 必需证据与失败例 |
| --- | --- | --- |
| M01 | 权重身份 | 所有分片、VAE、文本条件、配置的文件 hash、冻结来源、模型许可；缺一个分片即不完整 |
| M02 | 导出与覆盖 | pnnx/ncnn/脚本提交、trace 条件、输入 profile、子图 I/O、算子清单、fallback；仅 pnnx 命令成功不算通过 |
| M03 | 参考与随机输入 | A/B 生成 manifest、环境、脚本 hash、真实输入和噪声 tensor；自称 official 的 JSON 不足以确认来源 |
| M04 | AWA 全数学 | QKV、head norm、局部 RoPE 的 126 rotated + 2 unchanged、SDPA、video scatter、每窗等权 text mean；shifted 是截断/裁边而非循环移位 |
| M05 | VAE | encoder/decoder、因果 Conv3D、每层时间 cache、norm 与空间 attention；首帧、4n+1、补帧、裁边均要对照 |
| M06 | DiT 层级 | 32 个 block，前 10 个双分支，逐层 video/text 更新；不能只比较最后一张图 |
| M07 | 整网输出 | 单步 CFG=1 基线；preprocess、encoder latent、条件、各 block、采样、decoder float、color fix 前后、最终输出 |
| M08 | 视频连续性 | 静止/运动/切镜、开头/结尾、非整除形状、分块接缝；完整 clip 对照，不靠单帧 PSNR |
| M09 | 修复质量 | 独立可分发样例、hash/来源、保留集；文字、人像、纹理、压缩及失真；负面和盲审记录保留 |
| M10 | 设备与精度 | CPU/Vulkan 同输入，真实逐算子 backend，device UUID/PCI身份、driver、precision、shader hash；开 Vulkan 选项不等于全网 GPU |
| M11 | 性能稳定性 | 冷启动与至少 3 次暖跑的原始行、峰值定义、取消/OOM/长视频；8GB/720p 是待测目标 |
| M12 | 策略冻结及复核 | 阈值校准来源、冻结策略 hash、case 覆盖、独立复核者与内容 hash；阈值不可在看完候选结果后放宽 |

首轮覆盖的几何至少包括图片、17 帧短 clip、正常/shifted、非整除宽高、小于一个窗口、时间窗口上限两侧、ties-to-even、前后 block text 长度/状态。全量 case 清单必须由套件定义并先冻结，当前 16 个官方窗口几何参考只覆盖元数据，不自动满足 M04。

## 4. 数值准则与阈值冻结

- 整数索引、窗口边界、permutation/scatter、shape 和 frame count 必须精确一致。
- FP32 单算子诊断采用 `abs(candidate-reference) <= atol + rtol * abs(reference)`，当前 `atol=1e-5`、`rtol=1e-4`。这是定位工具的诊断值。
- 当前 C++ 审计器实际计算逐元素 violations、max absolute error、worst index、MAE、RMSE、NRMSE、NaN/Inf 数量。NRMSE 为 `sqrt(sum(error²)/sum(reference²))`；参考全零而误差非零时为 null，不伪造有限值。出现非有限值时不显示仅对部分有限数据计算的汇总指标。
- BF16/FP16 的 block / 整网阈值尚未确定。先用 A 的重复运行和 A/B/精度对照校准，再冻结；不把 FP32 单算子阈值直接推广到全模型。
- 除全局误差，还要检查边界最大误差、分位数、误差热图、误差随 block/帧数增长的曲线。这些完整指标尚未实装。
- PSNR/SSIM 只有在比较对象和色彩空间定义清楚时才有意义；实现 parity 的高 PSNR 不等于修复质量高。
- 视频时间残差可比较 `(candidate[t]-candidate[t-1]) - (reference[t]-reference[t-1])`，并单列 clip 接缝/运动区域。该指标需与逐帧误差和视觉审查配合，不能单独证明无闪烁。
- 精确 query/FFN 分块与近似空间/时间分块用不同 profile。近似模式的结果不能继承精确模式证书。

## 5. 本轮已实现的审计器

入口：`seedvr2 models audit --bundle DIRECTORY [--save]`。目录包含 [manifest schema](../schemas/model-evidence-bundle.v1.schema.json) 和工件。空证据示例：[manifest.json](../examples/model-evidence-empty/manifest.json)。

```json
{
  "id": "block00.awa.video",
  "gate_id": "M04",
  "reference": "ref-block00-video",
  "candidate": "ncnn-block00-video",
  "shape": [1, 18000, 2560],
  "dtype": "f32le"
}
```

这里是 tensor pair 示例，引用 ID 必须存在于 `artifacts`。每个 artifact 有 `id/role/path/sha256`。tensor 为无头 little-endian float32 连续文件，文件长度必须等于 shape 乘积 × 4。FP16/BF16 转成 f32 dump 的过程也需要在未来参考 runner 中记录，不改动原数值语义。

当前执行：

1. 读取最多 1 MiB 清单，拒绝重复键、未知字段、错误 schema、未知模型/证据类型、重复工件与 pair ID。
2. 工件路径必须留在 bundle 内；拒绝绝对路径、`..`、反斜线及符号链接。不从清单执行脚本或下载文件。
3. 以 SHA256 流式读取每个实际文件。缺失项显示 MISSING，内容改动显示 HASH_MISMATCH；hash matching 仅说明这次读取与清单一致。
4. 读取有效的 tensor pair，核对 shape、dtype、精确字节数，实际计算误差；比较后再次复核 tensor hash 检测运行期间的改动。当前最多 512 个工件及 512 个 pair，单 pair 最多 10 亿元素；仅 CLI 可发起大文件审计。
5. 按 12 项策略列缺失证据类型。即使所有文件标签和 hash 都齐，仍显示 EVIDENCE_PRESENT_UNREVIEWED。`scope_binding=DECLARED_UNVERIFIED` 不把用户写的设备/来源身份当已证实。
6. 生成 BLOCKED / FAILED 记录，固定 `model_verified=false`、`certificate=null`；可保存 SQLite 并由 Web 查看。错误清单不持久化成成功审计。

这一阶段**没有**可信参考注册、语义套件覆盖判定、低精度整网阈值、质量/时间指标计算器、校准工具、复核签名或模型证书签发。工具的合成负向测试只证明拒绝逻辑，不能成为 SeedVR2 的 M01–M12 证据。

## 6. 完整证据包与未来签发设计

完整产物树：`identity/`（权重/导出/构建）→ `reference-a/` 与 `reference-b/` → `runs/<scope>/`（输入、噪声、tensors、实际后端 trace）→ `metrics/` → `review/` → `report.json + report.html`。巨型 tensor 在文件工件存储，不进 JSON/SQLite。最终保存内容寻址目录，明确不可变边界、数量/大小上限和失败恢复。

可信参考 registry 要同时核对受控生成器及其运行产物；“hash 与自己写的 manifest 一致”不能证明模型来源。每个数值结果绑定 reference hash、candidate hash、policy hash、case id、scope；独立复核还要确认使用的确实是该模型与 backend。

签发条件设计为所有必需 gate PASS、没有跳过/缺项/非有限值、策略已经冻结、保留集未参与调参、scope 完整并可证实。证书写入精确 scope、证据根 hash、生成器、校准策略、复核记录和撤销/失效条件。公开 Discussion 从证书及原始记录生成 claim-to-evidence 表，而不是从日志措辞推导“已支持”。

在上述路径实现并验收之前，代码保持无法签发模型通过结论。
