# Ubuntu 24.04 发布与独立机器验收

本流程正在首次执行。只有成功的 release workflow 和实测报告才能将候选包标记为已交付。

- 模型：`models-fp32b-v1`，原始审阅过的 FP32-B 字节；53 个 SHA-256 对象，逐对象校验 GitHub 服务端摘要。客户端下载回读另测。
- 程序：Ubuntu 24.04 x86_64，CLI/Web/worker/SDK；系统运行库通过随包脚本安装，不打包 Fedora 开发机的库，不宣称通用 Linux。
- 构建：[release-linux.yml](../../.github/workflows/release-linux.yml) 在新 GitHub runner 上准备锁定依赖、构建及测试；保留源码、前端依赖、原生依赖和媒体源码归档。
- 新机器：另一个 runner 从 Release 下载程序与转换模型，执行真实 CPU 图片修复；第二次隐藏仓库源码并隔离网络，比较完整张量集合与 PNG 字节。
- 这项新机器测试验证首次运行和离线复用。官方 FP32-B 数值对照、GPU 设备资格、修复质量分别记录；不会由两次自身回放相同推出官方数值通过。

模型安装示例（Release 完整发布后可用）：

```sh
python3 tools/model_distribution.py install \
  --catalog docs/distribution/catalog.v1.json --kind image \
  --base-url https://github.com/mingshi2333/seedvr2-ncnn-vulkan/releases/download/models-fp32b-v1 \
  --layout flat --output models/image
```

发布工具不会覆盖已有同名资产；已发布且不完整的 Release 会拒绝修改。候选程序只有在独立机器检查成功后才公开，失败保留在 workflow artifacts。已转换模型与程序分别发布，以免把某个程序的失败误报为权重变更。
