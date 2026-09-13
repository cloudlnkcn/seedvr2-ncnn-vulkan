# 视频与内存中的不同问题

来源：`seedvr2-native`、`chisato-port`、`aichi-port`；路径和哈希见 [来源登记](../sources.json)。

| 名称 | 保存/分割的对象 | 本项目当前状态 | 同类代码可借鉴处 |
| --- | --- | --- | --- |
| 逐帧图片处理 | 各帧互不传递时序特征 | 现有短片走联合时序计算 | `aichi-port` 的帧目录模式便于批量图片处理，但不等同于联合视频恢复 |
| VAE query 分块 | 将注意力的 Q 分批，仍使用完整 K/V | 尚未实现该优化 | `chisato-port` 的 `VAEChunkedMHA` 默认 128 个 query，一次 attention 中复用投影后的 K/V |
| VAE 时间状态 | 因果卷积跨分段需要的历史帧/特征 | 整段计算，跨片段 cache 关闭 | `chisato-port` 有 INITIALIZING / ACTIVE 状态及结束后的 reset |
| 权重放置 | 当前 ncnn 图的权重驻留位置 | auto / device / host；逐图释放 | 根据预算选择，不能代替激活预算 |
| 图内连续执行 | 相邻算子/层的设备张量 | 36 图边界存在传输 | 同类合并多层 DiT 可以减少同步，但收益仍需测量 |
| 文件页缓存 | 操作系统保留的模型文件数据 | 影响热启动与读取时间 | 冷暖状态必须写入性能记录 |

VAE query 分块中的 K/V 是同一次非自回归 attention 的投影复用，与语言模型每生成一个 token 累积的 KV cache 不同。时间状态也只保证 VAE 相关边界；长视频还要定义 DiT 上下文、跨段重叠、镜头切换、首尾帧和拼接质量，不能只加一个 cache 开关就宣布长视频完成。

本项目 [内存验证](../../MEMORY-VALIDATION.md) 的权重估算不包含激活和工作区，`--weights auto` 不保证不会 OOM，也没有分配失败后的自动重试。这些边界应持续出现在用户预检和性能报告中。

关联：[本项目](../entities/seedvr2-native.md)、[同类移植](../entities/peer-ports.md)、[设计对照](../synthesis/design-comparison.md)、[改进顺序](../synthesis/delivery-plan.md)。
