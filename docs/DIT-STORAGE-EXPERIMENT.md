# DiT 权重存储精度实验：2026-09-13

> 历史组件实验。后续已按独立低精度协议完成图片与视频执行，见 [DiT FP16 存储支持](DIT-FP16-STORAGE.md)。下文的 0/4 仅指原 FP32 保真门槛，不能解释为量化不可用。

本轮已实现独立候选转换和真实组件验证。**FP16 文件缩小约 50%，但两种候选均未通过现有 FP32-B 组件门槛，因此默认模型仍保持 FP32。** 这不是 INT8 实现，也没有交付 BF16/FP16 激活计算或完整低精度模型包。

## 实现边界

[实验入口](../tools/experiment_dit_storage.py)消费已保留的 pnnx trace、官方权重来源及带哈希的参考张量，输出独立候选目录。原始输入、参考、参数和权重哈希在转换前校验；不重新生成参考，不修改审阅模型包。

提供三个候选模式：`fp32`、锁定 pnnx 的 `fp16`、本项目 `fp16-ieee`。后者由[严格转换器](../tools/dit_weight_storage.py)按已知 DiT 参数布局逐层读取，只转换 `InnerProduct` 的带类型标记权重，保留 bias、modulation 常量和 AWA 属性原始字节。使用 IEEE FP16 舍入并保留 subnormal 小数值；未知层、量化输入、截断、尾随字节及非有限/溢出权重拒绝输出，临时文件验证成功后才发布。

运行时仍使用原来的 ncnn 模型加载器，将半精度存储解码成 FP32。CPU/Vulkan 激活和算术仍为 FP32；Vulkan 自定义线性层仍使用现有 FP32 实现。**磁盘节约不能推出运行时权重或显存节约。** 自定义线性层目前拒绝 INT8 scale 参数，INT8 需要另行实现并验证。

## 固定输入与结果

SeedVR2 3B 的第 16 块，真实 checkpoint，视频激活形状 `1×2×3×2560` 和 `2×3×4×2560`，官方正向文本嵌入，独立保留的 FP32-B 参考。这里的视频激活是采样张量，不能等同于完整视频轨迹。每种模式运行两种形状 × CPU/Vulkan，共四例，每例比较 video/text 两个输出，并检查后端分派。Vulkan 请求 Khronos validation layer。

运行库为锁定的 ncnn `3b7bdba7fc8aea8fd46779533eee027df77c639d` 加已记录项目补丁；使用冻结的 `dist/numerics-v2/bin/seedvr2`。转换器 ncnn/pnnx 源码固定为 `6a1bf000f363714839a36793addc8c879d3d899e`。二进制、套件及脚本 SHA 见[实验摘要](../artifacts/2026-09-13/precision-v1/summary.json)。设备为 Linux x86_64 / RTX 4060 Laptop 8 GiB。

| 候选 | 单块 bin 字节 | 固定组件门槛通过 | 两种形状中 Vulkan video 最大绝对误差 |
| --- | ---: | ---: | ---: |
| 原 FP32 | 634,472,588 | 4/4 | 0.00003815 |
| pnnx `fp16=1` | 317,278,348 | 0/4 | 0.02330017 |
| IEEE FP16 存储 | 317,278,348 | 0/4 | 0.02125168 |

门槛保持 `atol=1e-4, rtol=1e-3`，逐元素判定。所有候选输出有限且可以执行；零项通过指固定数值协议失败，并不表示应用不能产生图像。该实验没有测量最终图片/视频质量，不能用这些组件误差直接判定画质好坏。

当前锁定 pnnx 的 `tools/pnnx/src/utils.cpp::float32_to_float16` 会把不能表示为 normal FP16 的小数值清零。IEEE 候选修正这部分存储损失后，仍有大量正常权重舍入，因此未关闭误差。逐矩阵 changed、最大误差、小数值计数和舍入为零计数保留在摘要中。不能将全部差异归因于小数值清零。

完整结果：[FP32 控制](../artifacts/2026-09-13/precision-v1/fp32-control-block16.json)、[pnnx FP16](../artifacts/2026-09-13/precision-v1/fp16-storage-block16.json)、[IEEE FP16](../artifacts/2026-09-13/precision-v1/ieee-storage-block16.json)。七项转换器测试通过，涵盖小数值、bias 保真、截断、未知算子、量化输入、非有限值和禁止覆盖；最终转换器重写得到与实测候选相同的 SHA。

## 时间观察的限制

在较大形状的 Vulkan 单次运行中，FP32 / pnnx FP16 / IEEE FP16 的含传输计算时间分别为 71.39 / 82.82 / 85.97 ms，总时间分别为 1595.12 / 1358.77 / 1499.50 ms。总时间包含装载等开销，不能用两者之差命名为纯装载时间。未做重复分布、冷缓存控制及 GPU 峰值采集，而且当时有 HF 后台网络上传；这些只是运行记录，不能声称获得推理加速。

## 复现

在已有官方转换缓存的仓库中执行（Python 使用现有导出环境）：

```sh
.venv-export/bin/python tools/experiment_dit_storage.py \
  --source .cache/dit-block-export-remaining/suite.json \
  --block 16 --pnnx .deps/bin/pnnx \
  --weight-storage fp16-ieee --output .cache/my-fp16-component

.venv-export/bin/python tools/check_graph.py \
  --binary dist/numerics-v2/bin/seedvr2 \
  --suite .cache/my-fp16-component/suite.json \
  --output .cache/my-fp16-component/runs \
  --report .cache/my-fp16-component/result.json --validation-layer

.venv-export/bin/python -m unittest discover -s tests -p test_dit_weight_storage.py -v
```

当前候选检查预期以非零退出码报告数值失败。全新机器先按[本机转换说明](LOCAL-CONVERSION.md)准备官方来源和导出环境；本实验依赖开发期保留的 block trace/suite，并非一键安装包。候选目录必须为空。

## 后续决策

保留 FP32 默认和既有 73 边界基线。继续低精度需要独立标明参考路径、舍入和质量预算，再做真实轨迹的敏感层诊断；不能放宽旧门槛后重新标注历史结果。BF16 官方参考、低精度激活、选择性 INT8、完整低精度模型验证与可分发包均未由本轮完成。

知识 lint 当前仍报告前一轮未提交 VAE query-chunk 改动导致 `pipeline.hpp`、`inference.hpp` 来源哈希变化；没有用更新哈希掩盖尚未完成的该轮审阅。HF 上传的 FP32 包及其冻结发布材料未改动。
