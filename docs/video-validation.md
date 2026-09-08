# 0.5.0 视频预览验证记录

2026-09-07，Linux x86_64 / RTX 4060 Laptop GPU。**已完成真实整段短片推理和 MP4 输出；数值验证为 MIXED，模型认证仍未通过。** 本页保留通过和失败的测试，各 JSON 绑定自己的代码/二进制或运行记录身份。当前范围与使用方法见 [video-runtime.md](video-runtime.md)。

## 参考与输入

官方 SeedVR 源码固定 `e4de8c24441a67e1b7df56abea10645059bb1185`，原文及哈希清单为 `tests/reference/seedvr/sources.json`。VAE 与完整 NaDiT 使用独立参考包装执行原始方法体，显式切换为 FP32-B / PyTorch math attention / 整段 VAE memory disabled。官方 BF16/FlashAttention A 路径未运行。

VAE 模块测试使用确定性合成张量，覆盖 encoder/decoder、1/5/9/17 帧、方形及两个方向的长方形。完整视频输入为 FFmpeg `testsrc2` 生成的运动测试图案，64×64、8 fps、17 帧。输入 SHA-256 为 `71b061fb9277cfeccad53c9d739003929ac7004a24c5925707c144d0ffeabc67`，生成和工具身份见 [样例来源](examples/video-demo-input-provenance.json)。它只能证明功能及所测张量的一致性，**不能证明自然视频画质提升、去闪烁能力或普遍鲁棒性**。

完整参考共享候选保留的原始 posterior/DiT noise。它检查实际字节和身份，不假设 ncnn 与 PyTorch 的同 seed 产生同一随机序列。比较发生在有损 MP4 编码之前。

## 数值结果

VAE 单模块阈值为 `1e-4 + 1e-3*abs(reference)`；整网沿用图像阶段诊断阈值 `1e-3 + 1e-3*abs(reference)`，没有为通过视频测试而放宽。73 个检查点是同一轨迹的中间张量，不是 73 个独立视频。

| 检查 | 范围 | 结果 | 证据 |
| --- | --- | --- | --- |
| 候选 PyTorch 时序 VAE / 官方方法体 | encoder+decoder，1/5/9/17 帧 | 8/8 PASS | [导出参考 suite](video-vae-export-suite.json) |
| 最终原生时序 VAE / 官方参考 | 上述 8 组 × CPU/Vulkan，启用 Vulkan validation layer | 16/16 PASS | [原生 VAE](video-vae-native-validation.json) |
| 完整短片 / Vulkan | 6 个输入帧，64×64，补到 9 帧、latent T=3，再裁回 6 帧 | 73/73 PASS | [原始参考](video-reference-six.json) / [最终构建复核](video-parity-final-six.json) |
| 完整短片 / Vulkan | 17 帧，128×128，latent T=5 | **60/73 PASS，整体 FAIL** | [原始参考](video-reference-seventeen.json) / [最终构建复核](video-parity-final-seventeen.json) |
| 完整短片 / CPU | 同一 17 帧、128×128、同一噪声与参考 | **70/73 PASS，整体 FAIL** | [CPU 复核](video-parity-cpu-seventeen.json) |
| 图片回归 / 最终 Vulkan CLI | 原有 128×80 单图及保留的独立参考 | 73/73 PASS，RGB8 最大差 1 | [图片复核](video-image-parity-final.json) |

17 帧 Vulkan 的失败在 block 19、22–31 的 video 输出，以及 velocity、latent。CPU 的失败为 block 29–31 video 输出；CPU 的 velocity/latent 通过。两者最终 decoder 张量均通过，但这不能覆盖中间层失败。Vulkan 原始最终 decoded 最大绝对误差约 `2.6923e-4`。

CPU 对照说明问题不只存在于 Vulkan 路径；现有证据尚未把原因严格定位到某个算子或单纯舍入累积。需要用同一官方输入逐层隔离后段 DiT，再扩展形状/内容，才能关闭此缺项。**不将“小误差”“画面看似正常”或最终 decoder 通过作为整网验收通过的替代。**

完整 run 报告有 36 图逐图计数，Vulkan 路径网络层 CPU 调用为零。媒体编解码、resize、布局变换、噪声和 Euler 标量计算是显式主机操作；“无网络层回退”不等于全部程序工作都在 GPU。

## 保留的开发失败

1. 首次导出时 tracing 删除没有被引用的 spec 属性，pnnx lowering 不能验证边界。改为保留有实际引用的 scripted module 后重新导出，不保留为有效模型包。
2. 早期 encoder 将时间降采样放错 block，T=1 无法发现，T=5 的官方对照明确失败。按官方 block 1、2 修正后，重新得到全部 8 个参考通过。
3. 第一次 17×128 完整 GPU 运行在 decoder `pnnx_unique_51` 被 512 MiB gather 守卫拒绝，没有生成成功 run。审查实际展开尺寸后仅把 gather 守卫增加到 1 GiB，重新完整运行成功；没有禁用尺寸检查。失败 stderr 和后续 run 均保留在本地证据清单。
4. 17 帧整网严格阈值失败仍然存在，见上表。失败报告的 `passed` 保持 false，不能当成证书或绿色发行门禁。

## 应用验证与复现

- [真实视频任务](video-jobs-validation.json)：49 项实际 HTTP/worker 检查，包括媒体与任务类型、非法帧数/尺寸/重复字段、HDR/非方形像素拒绝、取消、完成 6 帧任务、MP4 数量与时间戳、下载哈希、Range 206/416、顺序事件和重启恢复。
- [原有图片任务回归](video-image-jobs-validation.json)：39 项，原有任务协议与完整图片处理保留。
- [HTTP/CLI/审计边界](video-web-validation.json) 和 CTest：[开发构建](video-ctest.xml)、[独立 CLI 构建](video-cli-ctest.xml)。工程测试和模型数值结论分开。
- 浏览器实际上传、执行、播放、暂停同帧、时间线、逐帧、MP4 下载、重开、窄屏与 Discussion 草稿验证见 [浏览器记录](video-browser-validation.json)。

```sh
python3 tools/check_video_jobs.py --server build/dev/seedvr2-web \
  --model .cache/image-package-fp32 --video-model .cache/video-package-fp32 \
  --input .cache/video-demo-17.mp4 --image docs/examples/image-demo-input.png \
  --output .cache/video-jobs-new

.venv-export/bin/python tools/check_video.py --run /path/to/diagnostic-run \
  --input /path/to/exact-input.mp4 --output /path/to/new-reference

.venv-export/bin/python tools/check_video_outputs.py --run /path/to/new-run \
  --reference /path/to/retained-reference --reference-run /path/to/original-run \
  --output /path/to/replay.json
```

参考脚本在失败时仍写入报告并以非零退出；重放检查原始 run/输入/噪声/包身份及逐张量哈希。不能通过改写报告、改阈值或复用不同输入的参考来获得通过。

## 尚缺的模型与发行工作

后段 DiT 差异定位、自然视频和运动/遮挡/镜头切换覆盖、冻结阈值与质量保留集、官方 A 路径、低精度、流式 causal cache、分块边界、长视频连续性、音轨、较高分辨率、峰值 RAM/VRAM 和统计性能测试。完整模型仍为 `model_verified=false`，无证书；尚未发布外部 Discussion、模型包或便携发行版。

`image-*` 保留 0.4.0 原始证据，`video-*` 记录本次扩展；每份报告只证明其绑定的构建和输入。当前机器状态见 [status.json](status.json)，当前来源和工件身份见 [video-manifest.json](video-manifest.json)。
