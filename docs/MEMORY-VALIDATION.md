# 0.7.0 权重内存策略与验证

2026-09-08。本轮实现每个图装载前的 GPU/RAM 权重选择，沿用官方 SeedVR2 3B 的单步 FP32 数学、模型包和已保存参考。先验证小算子，再验证真实组件，最后验证完整流水线。结果、失败与执行身份在 [memory-v1](../artifacts/2026-09-08/memory-v1/)。完整模型认证仍为 false。

## 实现和使用

CLI、Web worker、外部 C++ SDK 共用 `RestoreRequest` 和同一原生图执行层。默认 `auto` 在每个图装载前重新读取实际设备本地堆的 `VK_EXT_memory_budget`，使用 `budget - usage`，并明确记录查询不可用。该扩展提供驱动估计，并不保证后续分配一定成功，见 [Khronos 定义](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceMemoryBudgetPropertiesEXT.html)。

```sh
# 默认自动选择；不需要在 Web 中额外配置
dist/seedvr2-0.7.0/bin/seedvr2 run-video --model .cache/video-package-fp32 \
  --input .cache/video-demo-17.mp4 --output output/memory-auto \
  --backend vulkan --gpu 0 --size 128 --frames 17 --weights auto

# 显式使用 RAM 权重；计算仍在 Vulkan 上执行
dist/seedvr2-0.7.0/bin/seedvr2 run-video --model .cache/video-package-fp32 \
  --input .cache/video-demo-17.mp4 --output output/memory-host \
  --backend vulkan --gpu 0 --size 128 --frames 17 --weights host
```

`run`、`run-video`、`engine graph`、`engine block` 都接受 `--weights auto|device|host`。自动模式还接受 `--gpu-reserve-mib N`。默认 0 表示保留所查询堆预算的四分之一；非零值是用户要求的余量。这个比例是当前启发式设置，没有测量校准成工作区上界。权重准备估计为已校验 FP32 图文件大小的两倍，允许准备副本，但不计入激活、工作区或分配器额外开销。查询缺失或返回零预算时请求 RAM；显式 device/host 则遵从请求。CPU 后端拒绝 GPU 专用设置，非自动模式拒绝同时指定余量。

SDK 的等价设置为 `request.memory.weights = seedvr2::WeightPlacement::automatic`，余量以字节写入 `request.memory.gpu_reserve_bytes`。新字段改变了预览 SDK 的结构布局，版本升级为 0.7.0，SONAME 为 `libseedvr2.so.0.7`；旧 SDK 使用程序需要重新编译，不会将旧 `.so.0` 原位替换为不兼容的新库。

报告中每个 stage 保存选择理由、准备估计、保留余量、装载前/后的堆预算与使用估计，完整执行还保存释放后的观测。汇总是放置**请求次数**，不是分配器实际驻留量。图执行完成、输出下载后才销毁 Net 和权重映射；随后 GPU 激活池仍可能保留内存，因此释放后堆使用量不要求为零，也不能把两次观测相减当作权重大小。

## 模块边界与 ERNIE 复用

| 层 | 源码 | 职责 |
| --- | --- | --- |
| 公共设置 | `include/seedvr2/memory.hpp` | 标准 C++ 类型，无 ncnn 或 JSON |
| 纯策略 | `src/runtime/memory_policy.hpp/.cpp` | 预算、保留余量、溢出保护、查询缺失和选择理由 |
| ncnn 适配 | `src/engine/ncnn/memory.hpp` | 查询真实堆、设置权重分配选项、组织报告 |
| 生命周期 | `src/engine/ncnn/inference.hpp` | 同步完成、下载、销毁、释放后观测 |
| 算子验证 | `tests/memory_policy_tests.cpp`、`tests/memory_vulkan_tests.cpp` | 独立预算边界和真实 GEMM GPU→RAM→GPU 切换 |
| 真实组件 | `tools/check_memory.py` | 保留的官方输入、既有误差门槛、三种策略、输出字节一致性 |

核对了 ERNIE 的 `codex/surpass-reference` 工作树 `00b6e642636cf95af1fe65d396e4d91eca301206`：已有 `WeightPlacement`、`WeightSession` 和 `host_memory`，包括逐块预算重读、可选 RAM 权重缓存和 RAM 余量限制。它有自己的尺寸条件、查询缺失回退、缓存验收与默认余量。本轮未修改 ERNIE。

两项目可复用纯预算判断、可注入预算读取器、真实小矩阵验证方法与 GPU 完成后释放的原则。当前 SeedVR2 只有一次去噪，同一个 DiT 块通常只经过一次，没有为它增加跨步权重缓存；ERNIE 多步执行的缓存收益需使用它自己的完整输入测量。SeedVR2 的时序 VAE cache、ERNIE 的提示增强 KV cache、模型形状约束均不能直接共用。这里没有创建新的通用框架包。

## 算子验证发现的 ncnn 问题

在固定官方源码 `3b7bdba7fc8aea8fd46779533eee027df77c639d` 上，首次小 GEMM 虽然数值正确，RAM 权重路径仍出现两个 Vulkan 错误：

1. `VUID-vkBindBufferMemory-memory-02985`：导入主机指针的内存所绑定缓冲区缺少对应外部内存类型声明。要求见 [vkBindBufferMemory](https://docs.vulkan.org/refpages/latest/refpages/source/vkBindBufferMemory.html)。
2. `VUID-vkCmdCopyBuffer-srcBuffer-00118`：权重布局转换从权重缓冲区复制，但该缓冲区缺少 `TRANSFER_SRC` 用途。

原始 stdout/stderr 保存在 `gemm-upstream-validation.*`。校验层会向 stdout 输出；只检查 stderr 或退出码会漏掉该失败。CTest 和原生 CI 现在同时把这些诊断作为失败。

`cmake/SeedVR2NcnnAllocator.cmake` 校验原始 `allocator.cpp` SHA 后，在构建目录生成带修正的独立编译副本，并替代静态库中的同一编译单元。官方源码目录、下载归档和已安装 ncnn 静态库没有修改；**应用实际运行时包含 `host-buffer-v1` 修正**，不能将其描述成完全原版 ncnn 二进制。运行报告同时记录原始/编译副本 SHA 和实际加载 SDK SHA。换用新的上游 allocator 时哈希门槛会要求重新审阅。修复范围限于所测试的权重缓冲区路径，没有宣称全部平台或所有 ncnn 分配方式已验证。

另保留新测试自身的两个失败：第一次使用包含填充的 `Mat::total()` 判断逻辑元素数；Mesa 首次执行没有使用 Net 按设备调整后的选项，导致着色器绑定数量不匹配。分别修正为逻辑形状和实际 `net.opt`，没有放宽数学期望或跳过 Mesa。

## 验证协议

`source-bindings-v2.json` 固定生产源码、输入、可执行文件和 SDK；组件工具还核对每次子进程的完整实现身份。同一输出目录不允许覆盖既有实验。CUDA/PyTorch 官方参考没有重新生成，既有参考身份和数值阈值保持原样。

```sh
python3 tools/check_memory.py --binary dist/seedvr2-0.7.0/bin/seedvr2 \
  --dit-suite .cache/dit-trace-cases-v1/suite.json \
  --vae-suite .cache/vae-video-export-v3/suite.json --output .cache/memory-new-check
```

三种模式为显式 device、默认 auto、auto 加 1 TiB 保留余量。最后一个是无需物理耗尽显存的受控分支测试，**不是日常推荐设置，也不是 OOM 压力测试**。每次使用 Vulkan validation、相同真实权重和官方输入；既要符合原有组件误差门槛，也要与 device 输出逐字节一致。真实块为官方轨迹 block 16；时序 VAE 包含 T=1/5/9/17 的 8 个编码器/解码器用例。

完整短片另验证 17 帧、128×128、seed 666、相同原始噪声的全部 73 个边界。采样 RSS、匿名页、交换区及整卡显存；激活、工作区、文件缓存的独立峰值仍未知。两种策略各一次的桌面实测只描述这两个运行，不是统计性能基准。

## 本轮结果

同一安装候选 CLI SHA 为 `2c2d08ac27d1a35ce2f2a8a92d32c307f7774ab00398999b6173d672893112b6`，共享 SDK SHA 为 `ff86dcd0b05ea66f5ab384701701aced273582e0ee95fc0cfa12c9c4608f30f2`。外部 SDK 示例有自己的可执行文件 SHA，但载入同一 SDK。RTX 4060 Laptop 8 GiB；桌面同时使用 GPU。下列数值门槛沿用原协议，未因新增内存路径放宽。

| 类别 | 结果 | 证据与口径 |
| --- | --- | --- |
| Release 原生构建、安装与外部 SDK 链接 | PASS | `install-v2.log`、`native-ci-v2/sdk-build.log`；新的 0.7 SONAME |
| CTest | **14/14 PASS** | `native-ci-v2/ctest.log`，包含 20 个独立预算/参数断言和真实 GEMM |
| NVIDIA 小 GEMM | **4 次精确通过** | `gemm-final.json`；实际权重内存类型 device-local 为 true/false/true/false，跨 Net 保留的 GPU 激活继续正确计算 |
| Mesa Vulkan | **5/5 PASS** | `native-ci-v2/mesa-regressions.log`；要求 Vulkan 校验，未跳过 GEMM |
| AWA 官方 FP32-B 小算子 | **40/40 PASS** | 20 个输入 × CPU/Vulkan；`awa.json`，最大绝对误差 `1.3709068298339844e-6`；门槛 `1e-5 + 1e-4 × abs(reference)` |
| 真实 DiT block 16 | **3/3 PASS** | `components/dit-*.json`；最大绝对误差 `0.000244140625`，三种模式输出逐字节一致；门槛 `0.001 + 0.001 × abs(reference)` |
| 真实时序 VAE | **24/24 PASS** | 8 用例 × 3 模式；最大绝对误差 `5.0067901611328125e-5`，输出逐字节一致；门槛 `0.0001 + 0.001 × abs(reference)` |
| 17 帧自动选择 | **完整执行、73/73 PASS** | `video-auto-parity.json`，本机 36 个图均请求 device |
| 17 帧受控低预算 | **完整执行、73/73 PASS** | `video-auto-host-parity.json`，36 个图均请求 host，理由均为预算不足 |
| 视频跨策略一致性 | **77/77 诊断张量及 MP4 逐字节一致** | `video-equivalence.json`；MP4 同时等于先前修复后的基线，SHA `43a15cd3eea6da3db932ef3bc20c0206bda4def0d484cc978806d8b302189550` |
| 外部 SDK 自然图片、中文输出路径 | **完整执行、73/73 PASS，像素最大差 1** | `image-sdk-fixed-parity.json`，128→256，原始 JPEG、固定噪声、独立官方参考 |
| 参数、首次使用、真实包边界 | **20/20 PASS** | `first-use-real-package.json`，含 7 个新增内存参数拒绝条件、Unicode 和文件篡改 |
| 参考合同负向测试 | **11/11 PASS** | `native-ci-v2/evidence-contract.log`，包含逐一移除 73 边界的检查 |
| 既有 CLI / 引擎边界 | **10/10、23/23 PASS** | `native-ci-v2/cli.json`、`engine-boundaries.json` |
| 本地 Web 合同 | **58/58 PASS** | `web.json`，包含 CPU/Vulkan AWA；完整模型质量仍分别报告 |
| 当前 Web → worker → SDK | **真实 17 帧视频 PASS** | `live-web-result.json`；36 图使用 auto，实际新 SDK，MP4 与已验证 CLI 逐字节一致；保留原有 7 条任务历史 |
| 远端 CI / 其他真机 | NOT_RUN | 工作流已包含新增小测试；本轮未推送或声称远端通过 |

三种策略、真实组件和完整执行均保留 stdout/stderr，并检查校验错误。外部 SDK 的首次测试因 CMake 缓存选中了旧库而作废，记录为 `sdk-cache-failure.json`；`image-sdk-auto-*` 仅保留旧库实测，当前验收使用修正后的 `image-sdk-fixed-*`。CI 现使用 `cmake --fresh`、0.7 版本要求，并在大模型测试前检查实际加载库的 SHA。原始失败不覆盖；`ctest.log`、`mesa.log`、`gemm-upstream-validation.*` 是修正前的负结果。

| 17 帧同输入运行 | 墙钟 | 采样总 RSS 峰值 | 采样匿名 RSS 峰值 | 整卡显存峰值，含桌面 |
| --- | ---: | ---: | ---: | ---: |
| 默认 auto（本次全部选 device） | 41.36 s | 1021.47 MiB | 778.77 MiB | 6846 MiB |
| 受控 auto→host（本次全部选 RAM） | 88.98 s | 1579.86 MiB | 1336.74 MiB | 6249 MiB |

这两个运行的整卡峰值相差 597 MiB，RAM 模式耗时为自动模式的约 2.15 倍。两者采样进程交换区均为零；这不表示整个操作系统没有使用交换区。主机内存增加、访问成本上升，不能把 RAM 路径当作默认加速方法。驱动堆预算、进程 RSS 和整卡显存是不同观测，均未冒充独立权重/激活/工作区峰值。没有清空页缓存、独占 GPU 或重复统计测量。

| 新增选项 | 正确性证据 | 收益与代价 | 默认状态 |
| --- | --- | --- | --- |
| 逐图自动权重选择 | 独立预算边界、真实组件和完整图像/视频 | 按当前堆压力选择；启发式估计不保证峰值适配 | **开启** |
| RAM 权重 | 精确小 GEMM、真实 VAE/DiT、17 帧整链 | 本次显存峰值较低，耗时和主机内存增加 | 自动决策或显式 host |
| 保留余量覆盖 | 溢出、冲突及受控低预算整链 | 方便限制显存准备空间和复现分支；大余量可能显著变慢 | 0，使用四分之一堆预算 |
| 原有 mmap 权重读取 | 保留先前同输入证据 | 本轮未改变读取器实现；独立于 GPU/RAM 放置 | 保持 buffered，mmap 显式选用 |

## 当前边界

本版没有激活卸载、异步预取、OOM 自动恢复、动态画质降级、跨步权重缓存或总 RAM 硬上限。RAM 权重可能增加主机内存和 PCIe 访问成本，实际驻留也取决于设备内存类型。AMD/Intel 真机、Windows/macOS 完整模型和官方 BF16 路径仍未测；Mesa 小测试不代替那些设备验收。单例自然图画质负结果和模型未认证状态继续保留。

本地已安装 `dist/seedvr2-0.7.0`，图片与视频模型使用独立 Btrfs 写时复制副本并重新完整核验。入口为 `bin/seedvr2-studio`；SDK 示例、本文和本轮证据一同安装。SDK 缓存问题修正后，只重新执行受影响的 SDK 图片验证；已通过且计算实现未变的组件、视频没有再次执行。之后的 CMake 变更只补充证据安装目录，生产二进制与冻结候选仍逐字节一致。
