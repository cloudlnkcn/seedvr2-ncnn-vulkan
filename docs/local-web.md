# 本地 Web 服务（0.8-media-jobs）

Drogon C++ 服务提供内嵌 React 应用和同源 API。运行：

```sh
seedvr2-web --model /path/to/image-package --video-model /path/to/video-package --port 8877
```

仅监听 `127.0.0.1`，`--port 0` 选择空闲端口并打印 URL。前台服务进程继续处理任务，浏览器关闭不会停止推理。运行时无需 Node、Python 或 CDN；独立 CLI 不依赖 Web。未显式指定模型时使用[模型发现规则](image-runtime.md)。

## 图片与视频任务 API

| 方法 | 路径 | 行为 |
| --- | --- | --- |
| GET | `/api/v1/models/image` | 包清单是否存在、worker 是否可用、范围/尺寸/队列限制；不声称已完成哈希检查或认证 |
| POST | `/api/v1/media` | PNG/JPEG 或视频原始字节；视频指定 `X-Media-Kind: video`，`X-File-Name` 为百分号编码文件名；返回 ID、大小、尺寸和哈希 |
| GET | `/api/v1/media/{id}` | 仅按已导入资源 ID 读取输入 |
| POST | `/api/v1/jobs` | 校验 ImageJob/VideoJob 请求并核对实际媒体类型，入队并返回持久任务 |
| GET | `/api/v1/jobs` | 最近 100 条任务摘要，列表省略各层明细 |
| GET | `/api/v1/jobs/{id}` | 完整任务、状态、进度、结果或错误 |
| POST | `/api/v1/jobs/{id}/cancel` | `{}`，取消排队或执行；重复取消幂等 |
| GET | `/api/v1/jobs/{id}/events?after=N` | sequence 大于 N 的已提交事件；单页最多 256 项 |
| GET | `/api/v1/jobs/{id}/files/{name}` | 仅成功任务的 `output.png`、`comparison-input.png`、`run.json`，视频另有 `output.mp4`、`comparison-input.mp4`；`?download=1` 触发下载 |

ImageJob 请求只接受六个字段：

```json
{"schema_version":"seedvr2-image-job-v1","media_id":"0123456789abcdef0123456789abcdef","size":256,"backend":"vulkan","gpu":0,"seed":666}
```

拒绝重复/未知字段、嵌套值、错误数值类型、非法资源 ID、尺寸或设备范围。size 必须为 64–512 间的 16 倍数，seed 为 uint32；gpu=-1 表示自动选择，其他允许值 0–64 仍须由 worker 核对实际设备。请求通过不表示该 GPU 一定存在。详细契约见 [OpenAPI](../schemas/local-api.v1.openapi.json)。

VideoJob 使用 `schema_version="seedvr2-video-job-v1"`，额外且必需的第七个字段为 `max_frames`（整数 1–17）；size 收紧为 64–128、16 的倍数。视频和图片的媒体 ID 不可混用。视频实际时间填充/裁切、无音轨和 8 位 SDR 范围见 [video-runtime.md](video-runtime.md)。

输入媒体与完成结果支持单个 HTTP `Range`：`bytes=start-end`、`bytes=start-`、`bytes=-suffix` 返回 206 和 Content-Range；多段、非法及不可满足区间返回 416。完整/分段读取都设置 Accept-Ranges: bytes，支持浏览器拖动定位。

状态：QUEUED → RUNNING → SUCCEEDED/FAILED，取消经过 CANCELLING → CANCELLED，重启恢复为 INTERRUPTED。8 个未完成任务的上限、1 个执行进程；只在完整输出且成功退出后提供下载。进度轮询和持久事件游标已实现，SSE 未实现。

## 共享诊断与验收 API

| 方法 | 路径 | 行为 |
| --- | --- | --- |
| GET | `/api/v1/session` | 本次服务会话 token，不写 URL/日志 |
| GET | `/api/v1/capabilities` | 实际编译能力；image/short-video inference=true、model_certification=false |
| GET | `/api/v1/engine/devices` | 真实 Vulkan 设备索引，可能含软件实现 |
| POST | `/api/v1/engine/self-test` | backend/gpu，执行两组内置 AWA 并保存 PASS/FAIL |
| POST | `/api/v1/plan` | 与 CLI 相同的纯几何规划 |
| POST | `/api/v1/plans` | 保存 PLANNED 记录，不执行模型 |
| GET | `/api/v1/records?kind=…` | 最近 100 条 plan/model-audit/operator-test 摘要 |
| GET | `/api/v1/records/{id}` | 已保存规划、审计、自测完整 JSON |
| GET | `/api/v1/models/status` | 正式模型策略及缺项，不根据最近一次成功运行签发证书 |
| GET | `/api/v1/models/policy` | 12 项验收策略原文 |

外部模型验收包使用独立 CLI 审计并保存；Web 可读取相同工作区中的报告。草稿选择实际任务/自测记录后生成本地 Markdown，下载不会发布。

## HTTP 与文件边界

Host 必须等于实际 socket 的 `127.0.0.1:port`。Origin 必须同源，Sec-Fetch-Site 拒绝跨站 API/资源访问；允许外站导航打开首页。写操作需要 `X-SeedVR2-Session`，控制请求要求 `application/json`；媒体导入允许原始字节，无 CORS。

控制 body 最大 1 MiB，图片 body 最大 32 MiB，视频最大 256 MiB；图像解码前检查尺寸（单边 16–16384，总计最多 32 M 像素）。64 个 HTTP 连接、1 个 I/O loop、4 个应用线程、最多 16 个在途/排队用例，满时 429。模型运行在独立 worker；小型内置自测仍在服务进程的应用线程。网页不暴露任意路径或可执行命令，静态资源逐项内嵌，模型文件没有下载接口。

JSON/业务错误返回结构化 Error。Drogon HTTP parser 拒绝超大 body 时可能返回空 body 的 413；客户端按状态处理。CSP 允许 Ant Design 必需的 inline style，script-src 和 media-src 仅 self；响应禁止缓存和嵌入 frame。工作区私有目录保存输入和结果，不承诺抵御拥有同一系统账户权限的并发文件修改。

开发模式 `npm run dev --prefix apps/studio` 使用 loopback Vite 和同源代理。生产部署不需要 Vite。无自动浏览器启动器、托盘或跨平台发行安装器；Linux 进程管理已测试，其他平台尚待适配。
