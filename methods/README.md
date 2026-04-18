# methods/

三条跨中心泛化研究方法线的分类入口。

| 方向 | 目录 | 状态 |
|---|---|---|
| AugMix + ECG 数据增强 | [`augmix/`](augmix/) | ✅ 已实现（5 个算子 + meta + JSD + Dataset，39 测试通过）|
| ECGTwin 条件生成目标域样本 | [`ecgtwin_gen/`](ecgtwin_gen/) | 🚧 待填充（center token 已实现） |
| AdvDiff 对抗生成 | [`advdiff/`](advdiff/) | ✅ 已实现（代码在 `adversarial/`） |

每个子目录有自己的 README 说明起点、参考实现、建议设计。

## 为什么不把代码直接放 `methods/` 下？

- **AdvDiff 已在 `adversarial/`**：历史命名且 import 链路稳定，不重命名。`methods/advdiff/README.md` 做跳转文档。
- **AugMix / ECGTwin 尚未实现**：先留 stub，等实际写代码时再决定目录粒度。
