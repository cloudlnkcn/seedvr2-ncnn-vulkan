# 数值修复版的远端原生 CI

完整通过的源提交为 `7e56479c0c1069ee6b6c3ff88f8d57af12955418`。三个 Ubuntu 24.04 原生任务全部成功。[GitHub 原始执行](https://github.com/mingshi2333/seedvr2-ncnn-vulkan/actions/runs/34483304341) / [机器可读结果](summary.json)。

本目录保留实际下载的报告、依赖编译日志、完整 workflow 输出和 GitHub 元数据。全模型实机证据在 [video-numerics-v2](../video-numerics-v2/README.md)，远端 CI 没有下载或运行完整 3B 权重。

| 构建 | 原生测试通过 / 总项数 | 原生能力跳过 | Mesa 通过 / 总项数 | Mesa 能力跳过 | Web 接口 | 任务耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| gcc-cli | 24/36 | 12 | 10/22 | 12 | 未构建 | 488 s |
| clang-cli | 24/36 | 12 | 10/22 | 12 | 未构建 | 339 s |
| gcc-web | 24/36 | 12 | 10/22 | 12 | 54/54 | 634 s |

Mesa 条目是原生测试集合中的 Vulkan 子集，再显式指定软件驱动复测，不能相加为独立测试总数。跳过项的名称和算术前置条件在每个任务的 `scope.json`、`mesa-arithmetic.json` 与 JUnit 文件中；跳过不计为数值通过。公共预检会实际拒绝不满足 FMA 残差要求的设备。

三个任务还检查安装后的 SDK/CLI 库身份、CLI 与 engine 参数边界、首次使用、官方参考合同、模型下载和视频指标工具。GCC Web 另外执行 54 项接口检查。任务时间包括依赖下载和编译，不代表模型推理速度。

数值实现由 `dbe8bbb` 引入，后续 `7e56479` 只补齐构建工具的 `spirv-val` 预检。[源码对象核对](delivery-source-identity.json)确认模型实现、CMake、夹具与回放工具没有改变。原 `dbe8bbb` 的远端运行在被新提交取代后取消，元数据保留在 [superseded-run.json](superseded-run.json)，没有将它标记为通过。

[manifest.json](manifest.json)记录本目录文件的 SHA-256，不包含自身。`.cache` 目录仅包含 workflow 明确上传的依赖构建日志。追加本目录的文档归档提交使用 `[skip ci]`，避免为未改动的数值实现重复构建和测试。
