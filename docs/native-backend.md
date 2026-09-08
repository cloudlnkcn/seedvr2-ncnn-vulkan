# 原生后端与当前模型证据

此文保留 0.3.0 的子图设计与历史实测范围。0.5.0 新增时序 VAE 和整段短片，见 [视频运行时](video-runtime.md) 与 [包含失败的视频验证](video-validation.md)。0.4.0 已完成全 32 层的单图链路、worker 和 Web 任务，当前状态与复现见 [image-runtime.md](image-runtime.md)，最新证据见 [image-validation.md](image-validation.md)。模型仍没有证书。

## 已验证的范围

| 范围 | 实际输入与参考 | CPU / Vulkan 检查 | 不包含 |
| --- | --- | --- | --- |
| 自定义 AWA | 20 组 projected QKV、RMS 权重及官方 FP32-B 参考；普通/移位窗口、长宽极端比、T=33、20 heads/58 text tokens | 40/40，逐元素比较；实际 AWA / SDPA 调用计数 | 全 DiT、BF16/FlashAttention 官方路径 A |
| 单帧 VAE | 官方 ema_vae.pth；编码器、解码器，各三种尺寸；与原始 3D 官方类独立比较 | 12/12，逐层指定后端 | T>1、因果缓存、时序分块、完整恢复 |
| 完整 DiT 块 | 官方 3B 权重的块 0、1、10、11、31；真实正向文本 embedding、timestep 1000、两种视频 token grid | 20/20，视频与文本两路逐元素比较、逐层后端记录 | 全部 32 层串联、真实去噪轨迹、其他 timestep/提示词 |
| 随程序自测 | 2 组内置小型 AWA 合成用例 | Web 与 CLI 真正调用 CPU / Vulkan；PASS、FAIL 均可保存 | 官方 checkpoint 和模型验收 |

机器可读结果：[AWA](awa-export-validation.json)、[VAE](vae-image-validation.json)、[DiT](dit-block-validation.json)。三份结果绑定同一个实际 CLI 二进制哈希。Vulkan 检查在 RTX 4060 Laptop GPU 上启用 Khronos validation layer；逐层执行记录不能由“请求了 Vulkan”字段代替。

容差：AWA 使用 `abs(candidate-reference) <= 1e-5 + 1e-4*abs(reference)`；VAE/DiT 子模型诊断使用 `1e-4 + 1e-3*abs(reference)`。两者均在 float64 中比较原始 float32 文件。这些是开发诊断容差，**不是已校准的模型验收阈值**，也不适用于 FP16/BF16。

## 来源和参考独立性

- ncnn：从官方远端 HEAD 查询得到 `6a1bf000f363714839a36793addc8c879d3d899e`，按不可变归档和 SHA-256 锁定。`engine-dependencies.lock.json` 保留查询时间；不表示未来运行时自动追随 HEAD。
- ncnn 与 pnnx 均从这个提交构建，源码未打补丁；不链接旁边的 ncnn 工作树。准备工具在复用缓存前逐文件比对官方归档。
- SeedVR：官方提交 `e4de8c24441a67e1b7df56abea10645059bb1185`；所用源文件保持原文，清单在 `tests/reference/seedvr/sources.json`。
- 权重：官方 `ByteDance-Seed/SeedVR2-3B`，revision `37255ff8cccfb01071b87f635a5948ca8d53117c`；DiT、VAE、正负 embeddings 已按官方 LFS SHA-256 与字节数核对。锁在 `model-sources.lock.json`。
- Python 参考在开发环境执行官方类的方法体；仅把分布式/缓存外壳、Apex RMSNorm、FlashAttention 和 BF16 强制转换适配为明确的 FP32-B 路径。候选 VAE 和 DiT 导出模块独立编写，再与官方参考比较。所有适配均在 `tools/*_reference.py` 中可审查。
- 官方 A 路径（BF16 / FlashAttention）尚未运行，不能把本轮 FP32-B 报告称作官方生产路径一致性。

## AWA 的接口与实现

视频 QKV 为 `[T,H,W,3*heads*128]`，文本 QKV 为 `[text_length,3*heads*128]`；输出分别为 `[T,H,W,heads*128]` 和 `[text_length,heads*128]`。ncnn 映射为 w=特征、h=W、d=H、c=T；不使用 ncnn batch 轴。

CPU 与 Vulkan 都完成：窗口划分、Q/K RMSNorm、局部 3D RoPE、视频和文本联合 SDPA、视频逆散射、文本按窗口等权平均。文本不能按窗口面积加权；移位窗口采用边界裁剪，不做循环平移。RoPE 使用 3×42 维，保留最后两维。

GPU 路径由本项目 gather / scatter / text-mean shader 与 ncnn SDPA 组成。窗口、heads、文本长度和保守 512 MiB scratch 上界在分配前检查。当前是 FP32 正确性基线：尚未实现融合 attention、在线 softmax、FP16 存储或高分辨率显存策略。

`SeedVR2AWA` param：0=heads、1=shifted、2=epsilon、3=weight_format。format 1 读取 512 个 RMS 权重；format 2 依次读取 pnnx 原始 `epsilon/norm_weights/rope_frequencies/spec`，校验后使用。完整块保留整个 pnnx 权重二进制，不在大文件中搜索或替换字节。

## VAE 单帧 specialization

T=1 时，因果 padding 令同一帧重复进入时间卷积，3D 卷积可沿时间核求和，得到精确的 2D 候选。验证仍执行官方原始 3D 类，不用候选作为参考。时间上采样只保留官方第一帧的 z=0 相位，重排 `(x,y,z,c)` 权重后使用 PixelShuffle。

编码器输出 32 通道 posterior 参数；解码器输入未施加额外 scaling 的 16 通道 latent。此接口没有做后验采样、去噪、色彩修复。图中 attention、GroupNorm、残差均保留，不能简化成纯卷积 VAE。

pnnx 对 ExpandDims/Squeeze 的输出做四个严格的 shape-only lowering；转换时记录原始 param 哈希及匹配数量。导出使用不同像素数的两个输入形状，避免 trace 把 token 数固化。

## DiT 块的关键语义

- 块 0–9 视频/文本分支分开；10–31 共享权重。0、1、10、11 验证上述两类与普通/移位窗口的组合。
- hidden=2560、20 heads、head_dim=128；SwiGLU hidden=6912。
- timestep 调制按每维 `[attn_shift, attn_scale, attn_gate, mlp_shift, mlp_scale, mlp_gate]` 交错排布，不能切成六个连续大段。
- learned scale 已包含基线，不能再加一个无依据的 `+1`。
- 最后块的 `vid_only` 行为也在独立官方类中验证，包括其文本分支的最后一次 residual 行为。
- pnnx 对 AWA 边界周围四个 reshape 推断出的 batch 轴需要改为明确 token 布局；严格检查结构后才 lowering。
- 本轮发现 pinned ncnn 的 MemoryData Vulkan clone 从缺少 TRANSFER_SRC usage 的 weight buffer 复制，触发 `VUID-vkCmdCopyBuffer-srcBuffer-00118`。本项目 `SeedVR2Constant` 为调制常量使用有复制权限的小型 Vulkan buffer，保持复制语义；没有修改 ncnn。失败日志仍在本地缓存。

`engine graph` / `engine block` 在诊断执行中逐层调用指定的 CPU 或 Vulkan overload，缺少 Vulkan 层直接拒绝；无 Extractor 自动 CPU 回退。层间按 ncnn packing 契约转换。0.4.0 的共享执行器会在最后一次消费后释放 blob，并对图像任务分段提交 Vulkan 指令；峰值显存尚未完成正式测量。

## 复现导出与检查

普通用户运行内置自测不需要以下 Python 环境，也不需要下载权重。

```sh
# 开发专用：独立 Python 3.13、CPU PyTorch。固定依赖在 lock 文件。
uv venv --python 3.13 .venv-export
uv pip install --python .venv-export/bin/python torch==2.9.0+cpu --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-export/bin/python -r tools/export-requirements.lock.txt
python3 tools/prepare_engine.py --jobs 4
python3 tools/prepare_pnnx.py --jobs 4
python3 tools/prepare_models.py

# AWA 参考和真实 pnnx 导出
.venv-export/bin/python tools/generate_awa_reference.py --output .cache/awa-reference
.venv-export/bin/python tools/export_awa.py --suite .cache/awa-reference/suite.json --pnnx .deps/bin/pnnx --output .cache/awa-export

# 实际权重的子模型导出
.venv-export/bin/python tools/export_vae_image.py --pnnx .deps/bin/pnnx --output .cache/vae-image-export
.venv-export/bin/python tools/export_dit_block.py --pnnx .deps/bin/pnnx --output .cache/dit-block-export

# 每次指定空的输出目录。其余两类同理；AWA 使用 check_awa.py。
.venv-export/bin/python tools/check_graph.py --binary build/dev/seedvr2 \
  --suite .cache/dit-block-export/suite.json --output .cache/dit-block-check \
  --report .cache/dit-block-check.json --gpu 0 --validation-layer
```

已有缓存时，准备依赖使用 `--offline`；模型准备工具会重新校验已有文件。各导出/验证目录必须为空，不覆盖旧结果。导出工具、参考源、checkpoint 与 pnnx 的身份写入 suite 或 export-report；AWA 的导出报告另存为 [awa-pnnx-export.json](awa-pnnx-export.json)。当前实际私有环境复跑的目录是 `.cache/{awa,vae-image,dit-block}-export-private`，结果张量在 `.cache/native-awa-validation-retained`、`.cache/native-vae-validation`、`.cache/native-dit-validation`；三类均保留 stdout/stderr 以检查 Vulkan validation 输出。缓存不作为待发布源码。

## 0.3.0 时记录的后续实现链（历史）

| 顺序 | 需要实现 | 完成时必须拿出的证据 |
| --- | --- | --- |
| 1 | 全部 32 层导出、输入/输出投影与 timestep / conditioning，支持逐块装载 | 同一真实输入的每一层中间输出及最终 DiT 输出；不能只测独立随机块 |
| 2 | 固定噪声、官方单步 schedule、VAE posterior / scaling、图像预后处理 | 一张输入图的完整 CPU 路径，再验证 Vulkan；保留原始 latent 与像素误差 |
| 3 | 视频 VAE 因果 3D 卷积、时间相位、缓存和分块 | T=1/T>1、chunk 边界、首尾帧与不分块官方参考对比 |
| 4 | 有界 GPU 权重装载、FP16/BF16、融合 AWA 与调度 | 实际峰值显存、墙钟时间、无 CPU 回退证据、冻结后的精度/质量策略 |
| 5 | worker 监督、任务事件、取消/恢复、媒体 IO、实际结果对比 | 异常终止、重启、输出原子提交及 Web/CLI 相同结果 |
| 6 | 完整模型证据包与发行档案 | 12 项门槛、保留集、来源复核、跨平台实测、许可/打包；才考虑签发和公开 Discussion |

自测、审计器单元测试、官方权重哈希一致和子模型诊断，各自证明自己的范围。模型证书保持 `null`；策略仍 `NOT_FROZEN`，签发关闭。
