# 框架基础验证

日期：2026-09-06。版本：0.1.0-framework。范围：实际 C++ 主机规划、状态、协议与命令行，框架结构图的交互设计。

## 已验证

| 检查 | 实际结果 |
| --- | --- |
| GCC 16.2.1 / CMake 4.3.0 / Ninja，Linux x86_64 Debug | 编译成功，CTest 2/2 通过 |
| Clang 22.1.8 / ASan + UBSan | 编译成功，CTest 2/2 通过，本轮用例未报告内存或未定义行为错误 |
| 核心测试内容 | 7 组：几何、资源/整数边界、官方窗口参考、状态图、旧事件、严格 JSON、输出真实性 |
| 官方窗口对照 | 8 个形状 × 普通/shifted 共 16 组，有序边界、packed offset 与每个 token 单次覆盖通过 |
| 取整边界 | 代理轴 12.5 取到偶数 12，与 Python round 相同 |
| 状态规则 | 11 种状态的允许/拒绝关系、两阶段取消、终态不回退、run 身份与连续状态版本 |
| CLI 集成 | 10 个案例全部通过，见 cli-validation.json |
| 独立目录安装 | 在临时安装目录运行同一示例成功，随包示例、schema 和两份第三方许可证齐全 |
| 当前二进制依赖 | 实际检查仅链接 C/C++ 系统运行库，没有 Qt/ncnn/Vulkan 依赖 |
| 可视化 | 桌面结构图、模块切换、任务规划逐步查看已检查 |
| 390×844 窄屏 | 4 个页面主体均无横向溢出，临时 viewport 已恢复，控制台无 error 记录 |
| 文档与依赖 | 24 条本地链接、HTML ID 与脚本语法、固定 JSON 库 SHA256 均通过 |

核心原始输出见 [core-test-output.txt](core-test-output.txt)。其中 610255 次断言主要来自逐 token 覆盖检查，不表示 610255 个独立测试场景。

命令行覆盖中文/空格路径、缺失文件、未知参数、负数与极大维度、重复 JSON 键、过大文件、旧 JobSpec 拒绝、shifted 元数据和未实现引擎拒绝。正常示例结果见 [example-plan-result.json](example-plan-result.json)，10 个案例的状态与退出码见 [cli-validation.json](cli-validation.json)。

配置重生成时发现依赖哈希检查使用了工作目录相对路径，已改为源目录绝对路径并通过重新构建。C++20 路径构造替换了已弃用接口，随后构建没有报告该警告。

## 当前边界

未运行真实权重、attention 数值、RoPE、VAE、CPU/Vulkan 模型推理、GPU 设备检测、显存/吞吐测试、Qt 原生 UI、媒体处理、持久队列恢复或 Windows 程序。状态 reducer 通过不等于持久队列已实现。窗口划分通过不等于 AWA 已完成。

本轮 `plan` 退出码 0 表示声明几何与窗口计算成功。输出始终携带 `runnable=false`、`execution_status=NOT_RUN`，实际总峰值显存保持 null。当前主机规划上限不是模型支持矩阵。

框架结构图是静态本地 HTML，显示的 720p 数字来自本轮 CLI 输出。它不在浏览器运行 C++，不提交推理任务。

独立目录安装只在本机完成，不等于干净系统或跨发行版便携性验收。最终源文件、参考夹具和本机二进制身份记录于 [foundation-manifest.json](foundation-manifest.json)。

## 后续本地 Web 接入

本页为 0.3-framework 的基础验收快照，Qt 选型已由 0.4-local-web 替代。Web 增量验收见 [web-validation.md](web-validation.md)。foundation-manifest.json 保留旧快照；当前版本身份见 web-manifest.json。
