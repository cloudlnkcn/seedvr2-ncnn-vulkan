# 0.5 完整框架验证记录

日期：2026-09-06。项目版本 0.2.0-framework。**以下是框架与审计工具验证，不是 SeedVR2 模型验证。当前可执行文件未链接 ncnn，未运行模型或 Vulkan。**

## 结果

| 检查 | 结果 | 范围 |
| --- | --- | --- |
| C++ GCC Debug 构建 | PASS | 本机 Linux x86_64，C++20；Drogon/CLI11/SQLite/验收模块 |
| React 类型检查与 Vite 生产构建 | PASS | Node 24.18.0、TypeScript 7.0.2；6 个 lazy routes；静态资源内嵌 |
| CTest | 3/3 PASS | 原核心契约、CLI 几何、worker 能力边界；不含模型算子 |
| 独立 CLI 契约 | 10/10 PASS | UTF-8 路径、几何、窗口、参数/JSON 负向、未实现推理拒绝 |
| 审计器负向/数值工具检查 | 31/31 PASS | 合成 tensor、独立 Python 误差计算、篡改、NaN/Inf、错误 shape/path、伪造 PASS 字段、持久记录 |
| Drogon HTTP 与跨入口检查 | 40/40 PASS | Web/CLI 完整规划一致、模型状态一致、Host/Origin/session/body、SQLite 保存与互读、并发 |
| Clang ASan/UBSan | CTest 3/3、审计 31/31、HTTP 40/40 PASS | 项目目标有 instrumentation；本机系统库及准备的第三方库并非全部经过 sanitizer 重编译 |
| 安装目录检查 | HTTP 40/40 PASS | `/tmp/seedvr2-framework-install-uyk9zcvm`，同一 Linux 主机，7 个许可/声明文件 |
| 浏览器实际操作 | PASS | 规划 1280×720、保存、列表、重开、输入改变清空旧结果、12 项缺证据、CLI 空证据审计查看 |
| 响应式与视觉检查 | PASS | 1440 桌面审阅；390 窄屏 6 个工作区及审计详情均无 document 横向溢出；浏览器 error 日志为空 |

C++ 单元测试继承固定官方窗口几何参考；无 QKV、RoPE、SDPA、text mean 数值执行。审计器使用合成数据专门检查“不能伪造模型认证”，未把这些数据注册成真实 SeedVR2 参考。

## 可重跑证据

- [CLI 检查](framework-cli-validation.json)
- [审计器检查](framework-audit-validation.json) / [sanitizer 审计检查](framework-sanitize-audit-validation.json)
- [HTTP / 跨入口检查](framework-web-validation.json) / [sanitizer HTTP 检查](framework-sanitize-web-validation.json)
- [CTest XML](framework-ctest.xml) / [sanitizer CTest XML](framework-sanitize-ctest.xml)
- [安装验证](framework-install-validation.json)
- [浏览器实际检查记录](framework-browser-validation.json)
- [真实空证据审计输出](framework-empty-model-audit.json)：BLOCKED，model_verified=false，certificate=null；演示缺项查看流程
- [本轮源文件与构建产物清单](framework-manifest.json)

运行方式见 [README](../README.md)。独立 CLI 动态依赖检查包含系统 crypto/SQLite/C++ 等库，不需要 Drogon daemon、Node、浏览器或 ncnn。

## 本轮发现并修复

- SQLite 参数绑定 helper 名称与 `std::bind` 的重载查找冲突，导致记录 ID 没有按意图绑定。改为明确的 `bind_text`，通过 CLI/HTTP 互读与重开检查。
- Drogon 在 HTTP parser 层拒绝过大 body 时可返回空 body 的 413。客户端改为按 HTTP 状态报告，不尝试强制把空 body 当 JSON。
- Drogon CMake 包对未设置的 CXX_STANDARD 保存/恢复为空；项目显式固定 C++20。
- 生产 JS chunk 超过部分编译器保证的字符串字面量长度。嵌入器改为二进制字节数组，保持资源内容并消除该警告。

## 尚未验证

SeedVR2 权重来源与全量 hash、官方参考执行、pnnx 导出、AWA 全数学、VAE、32 层 DiT、CPU/Vulkan parity、整网画质、时间一致性、性能/峰值、8GB/720p、Windows 和便携发行包均未验证。worker 只提供能力查询；没有真实任务控制、SSE、取消/恢复。浏览器文件选择和下载落盘收据未自动验证。

旧 `foundation-*` 和 `web-*` 报告继续保留作历史，不能用来替代本轮验证或模型验收。
