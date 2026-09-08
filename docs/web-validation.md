# 本地 Web 增量验证

2026-09-06，Linux x86_64。架构版本 0.4-local-web，程序 0.1.0-framework + local-web。验证范围为本地 HTTP / 浏览器规划入口；没有模型推理或 GPU 性能结论。

| 项目 | 结果 | 证据 |
| --- | --- | --- |
| GCC 16.2.1 开发构建 | PASS | CLI 与 seedvr2-web 均构建完成 |
| 基础 CTest | 2/2 PASS | 原几何/窗口/状态/JSON 合同与 CLI 示例 |
| CLI 集成 | 10/10 PASS | [cli-validation.json](cli-validation.json) |
| 实际 HTTP 边界与 CLI 一致性 | 26/26 PASS | [web-validation.json](web-validation.json) |
| Clang 22.1.8 ASan + UBSan | 2/2 CTest、26/26 HTTP PASS | [web-sanitize-validation.json](web-sanitize-validation.json)，所测路由未发现 sanitizer 报告 |
| 安装目录运行 | 26/26 PASS | [web-install-validation.json](web-install-validation.json)，同一 Linux 主机；内嵌资源，不依赖源目录 |
| 浏览器真实交互 | PASS，限已列案例 | [web-browser-validation.json](web-browser-validation.json) |
| 390×844 窄屏 | 六工作区无页面横向溢出 | 浏览器 DOM 的 scrollWidth 等于 clientWidth；导航在容器内横向滚动 |
| 用户运行时依赖 | 当前 host 仅链接 C/C++ 系统库 | ldd 实查；无需 Python、Node.js、Qt、CDN |
| 安装许可文件 | 三项存在 | SeedVR Apache 2.0、nlohmann JSON MIT、cpp-httplib MIT |

HTTP 检查包括：同一请求的完整 Web/CLI 响应一致、非对齐图片尺寸、真实 capability、session、来源与 Host 检查、错误类型/重复键/旧 schema、1 MiB 上限、并发隔离、未知文件和任务路径。正常外部链接可打开主页，跨站读取 session/API 被拒绝。

浏览器检查了连接、视频 1280×720/17 帧/27 与 48 窗、图片单帧、改参数清除旧计划、实际能力查询、六工作区切换与窄屏。观察时浏览器 error 日志为空。系统文件选择器与下载文件落盘未自动验收；它们不能算作完整媒体导入/结果工件验收。

复查命令：

```sh
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
python3 tools/check_web.py build/dev/seedvr2-web build/dev/seedvr2 --output docs/web-validation.json
cmake --preset sanitize
cmake --build --preset sanitize
ctest --preset sanitize
python3 tools/check_web.py build/sanitize/seedvr2-web build/sanitize/seedvr2 --output docs/web-sanitize-validation.json
```

尚未验收：React/TypeScript 完整组件工程、真实媒体上传/解码、模型、AWA 数学/VAE/Vulkan、显卡探测、长连接事件、持久队列、worker、报告回导、启动器、Windows 和其它浏览器。规划通过不代表模型可运行，安装测试也不代表完成跨机器便携包。

`foundation-manifest.json` 保留 Web 接入前的历史快照；当前来源、依赖和二进制身份见 [web-manifest.json](web-manifest.json)。静态完整流程原型与新的 C++ Web 入口各自标明演示/实现边界。
