# 干净工作树全链路复现日志

日期：2026-08-06
工作树：`/home/linbinhao/ECG_manual_refactor_clean`
起始提交：`fce01866153deb760f38bd1aebf729a7f4ff958f`

## 目标

从当前手工重构白名单出发，重新验证以下链路：

```text
PTB-XL source checkpoint
-> 单目标中心固定 K500
-> 两链 AugMix + SimCLR
-> clean + rotating depth2/depth3 监督适配
-> exact-label attack-then-contract VAE-LHAT BCE
-> 排除 K500 身份后的 PN2021 Clean / PN2021-C
-> drop_all_zero macro AUROC / AUPRC
```

成功标准：

1. 配置闭包、数据身份、模型权重和随机种子可追溯。
2. EfficientNet1DV2 与 ECGFounder 均完成四中心训练和评估。
3. 复现结果与 `configs/active_evidence_registry.yaml` 中锁定的开发指标
   对比，并解释数值差异。
4. 每个受管运行留下 run card、配置快照、日志、checkpoint、评估结果和
   文件索引。

## 当前进度

| 阶段 | 状态 | 证据 |
|---|---|---|
| Git/边界预检 | 通过 | 工作树起始时干净，分支与远端对齐 |
| 基础数据与算子契约 | 通过 | `73 passed, 10 skipped` |
| EfficientNet/Ningbo 主线 dry-run | 通过 | 解析 14 个配置闭包文件；未加载数据、模型或 GPU |
| 全矩阵受管配置 | 通过 | 两骨干四中心共 8 个训练入口和 8 个评估入口均完成 dry-run |
| EfficientNet 四中心训练/评估 | 通过 | 四个 run card 均为 `complete`；正式 `drop_all_zero` 结果见下文 |
| ECGFounder 四中心训练/评估 | 通过 | 四个 run card 均为 `complete`；正式 `drop_all_zero` 结果见下文 |
| 运行产物闭包 | 通过 | 8 个训练和 8 个评估均具备 run card、manifest、summary、file index 及 checkpoint/结果 |

## 问题记录

### R001：迁移后缺少本机 ECGTwin 模型句柄

- 现象：`configs/train/vae.yaml` 指向
  `model/ECGTwin/checkpoints/vae_model.pth`，但新工作树的
  `model/ECGTwin` 不存在。
- 根因：Git worktree 不携带本机 `LOCAL-ONLY` 外部模型软链接。
- 影响：真正进入 VAE-LHAT 资源加载时会报 checkpoint 不存在。
- 处理：按 keep manifest 的 `LOCAL-ONLY` 规则重建
  `model/ECGTwin -> /home/linbinhao/ECG_adv_data/models/ECGTwin`。
- 验证：真实 VAE checkpoint 存在，SHA256 为
  `c2b4ef8060e412bd503f46b66dd44fc460e67a89de0ae1809dcc771358b3fa93`。
- 状态：已修复；该本机链接不得纳入 Git。

### R002：active index 引用了未迁移的测试

- 现象：`configs/active_scripts.yaml` 的 CPU 验证列表包含
  `test_manual_run_experiment.py`、`test_online_trainer.py`、
  `test_method_graph.py`、`test_vae_lhat_augmix.py` 和
  `test_pn2021_evaluation.py`，但干净工作树中不存在这些文件。
- 根因：清理测试树后没有同步更新 active index，或迁移时遗漏了所声明的
  主线测试。
- 影响：active index 当前不能作为“主线 CPU 已验证”的完整证据。
- 处理：把 active index 的 CPU 验证列表收敛为 A9 实际保留的五个基础契约
  测试，并显式把主线 GPU 复现状态保持为 `pending`，不再声称已运行不存在的
  测试。
- 状态：索引漂移已修复；在线 trainer 已由本次两骨干四中心 GPU 复现验证。

### R003：两骨干四中心主线 YAML 矩阵不完整

- 现象：受管主线只显式提供
  `manual_refactor_pn2021_effnet_augmix_simclr_lhat_ningbo.yaml`。
- 根因：迁移时只保留了一个 prospective replication 示例。
- 影响：无法在不使用动态 CLI 覆盖或临时脚本的前提下，通过唯一 launcher
  完成两骨干四中心正式复现。
- 处理：只复用现有 `train_pn2021` 与 `evaluate_pn2021` 入口，补齐 8 个训练
  YAML 和绑定对应 checkpoint 的 8 个评估 YAML；没有新增 Python runner。
- 状态：配置矩阵已补齐并同步 keep manifest/active index；16 个入口均完成
  dry-run，其中 8 个训练和 8 个评估随后全部实际执行成功。

### R004：历史 `manual_refactor` 运行根是指向 `/data` 的软链接

- 现象：把 `--run-dir` 放在
  `/home/linbinhao/ECG_adv_data/runs/manual_refactor/...` 时，解析后的真实目录
  位于 `/data/linbinhao/...`。
- 影响：新 AGENTS 约束要求本次写入保持在 `/home/linbinhao`，除非用户另行
  明确允许写 `/data`。
- 处理：本次新运行使用不经过该软链接的
  `/home/linbinhao/ECG_adv_data/runs/manual_refactor_clean_repro_20260806_r2/`。
- 状态：已规避。

### R005：Stage-2 clean logit-anchor 在 backward 前恢复 BatchNorm

- 现象：EfficientNet/Ningbo 真实探针完成 Stage-1 1024 步并保存
  `stage1.pt`，进入 Stage-2 第一批后，backward 报 640 维 CUDA 张量发生
  inplace version mismatch。
- 根因：clean logit-anchor 的第二次 student forward 被包在局部
  BatchNorm snapshot/restore context 中；该 context 在构建 loss 后、执行
  backward 前恢复了 running buffer。BatchNorm backward 仍引用这些 buffer，
  提前 `copy_` 使 version counter 从 0 变为 1。
- 影响：主线无法进入 Stage-2，因此旧的 dry-run/CPU 基础契约不足以证明在线
  trainer 可执行。
- 处理：把 clean logit-anchor 的 BatchNorm/RNG preservation context 挂到
  当前 exposure 的外层 `ExitStack`，使恢复发生在 backward 完成后。
- 失败证据：
  `/home/linbinhao/ECG_adv_data/runs/manual_refactor_clean_repro_20260806/effnet/ningbo/`。
- 验证：修复后的
  `/home/linbinhao/ECG_adv_data/runs/manual_refactor_clean_repro_20260806_r2/effnet/ningbo/`
  完成 Stage-1、23 轮 Stage-2、92 次 optimizer update，并成功产出
  `last.pt` 和正式评估结果；其余三个 EfficientNet 中心也完成相同链路。
- 状态：已修复并由四中心完整运行验证。

### R006：新工作树结果与历史开发锁不是逐位相同

- 现象：EfficientNet 四中心等权均值为
  `0.8736698738 / 0.6464508591 / 0.8593198569 / 0.6199489471`
  （Clean AUROC/AUPRC、PN2021-C AUROC/AUPRC）；相对历史开发锁分别为
  `+0.000521 / +0.262251 / +0.045502 / +0.061188 pp`。ECGFounder 对应
  均值为 `0.9143198878 / 0.7150944895 / 0.8916545579 / 0.6755787637`，
  相对历史锁为 `-0.010724 / -0.017164 / -0.080193 / -0.193773 pp`。
- 影响：EfficientNet 四项均达到历史锁定值；ECGFounder 最大绝对差距小于
  `0.2 pp`。两者都不能声称 checkpoint 或评估 JSON 的字节级重放。
- 当前证据：source checkpoint、随机种子配置、数据映射、K500 排除协议和
  训练 recipe 均已锁定；本次训练 checkpoint SHA 与历史候选 SHA 不同。
- 核对结果：当前 evaluator 明确记录 sklearn `1.8.0`、raw logits、
  `average_precision_score`；GPU 确定性算法在锁定 recipe 中为 `false`。历史
  候选 checkpoint 本体已不在干净运行面，只有 SHA 和汇总指标，因此无法
  对 state dict 做逐张量比较。
- 状态：指标级复现通过；字节级复现未证明，保留为明确限制。

### R007：ECGFounder 的 source 身份字段与 EfficientNet 名称不同

- 现象：初看 `train_result.model.checkpoint_identity` 时，ECGFounder 值为
  `null`，容易误判为 PTB-XL source checkpoint 未加载。
- 根因：模型契约按加载层级分别记录身份：EfficientNet 使用
  `checkpoint_identity`，ECGFounder 的完整任务 checkpoint 使用
  `task_checkpoint_identity`。
- 验证：四个 ECGFounder 运行的 `task_checkpoint_identity` 均为严格加载，
  `missing_keys=[]`、`unexpected_keys=[]`、509 个 state keys，SHA256 为
  `c23ec4361b097b74a4efed427488c4dd6a3753adfc75f7d1fc281e4df0989948`。
- 状态：不是漏加载；审计时必须按模型契约读取正确字段。

## 正式复现结果

以下均为 K500 identity 排除、v7 Super5 映射、`drop_all_zero`、
四中心等权均值；顺序为 Clean AUROC/AUPRC、PN2021-C depth2+3
AUROC/AUPRC。

| 骨干 | 当前干净工作树 | 历史开发锁 | 当前减历史（pp） |
|---|---|---|---|
| EfficientNet1DV2 | `0.873670 / 0.646451 / 0.859320 / 0.619949` | `0.873665 / 0.643828 / 0.858865 / 0.619337` | `+0.0005 / +0.2623 / +0.0455 / +0.0612` |
| ECGFounder | `0.914320 / 0.715094 / 0.891655 / 0.675579` | `0.914427 / 0.715266 / 0.892456 / 0.677516` | `-0.0107 / -0.0172 / -0.0802 / -0.1938` |

结论：

- EfficientNet 四项均达到或超过历史开发锁。
- ECGFounder 四项与历史锁的最大绝对偏差为 `0.1938 pp`；属于指标级近似
  复现，但不能写成逐位相等。
- 两个骨干的 VAE-LHAT raw-search sample-anyflip ASR 分别为
  `0.6094`、`0.6030`；回缩后的训练 view ASR 均为 `0`，无效解码率均为
  `0`。这与 attack-then-contract 契约一致，不能把回缩后 ASR=0 误写成攻击
  没有运行。
- 本次仍是锁定 recipe 的单 seed 开发复现，不构成三次独立重复或论文终局
  声明。

运行根：

```text
/home/linbinhao/ECG_adv_data/runs/manual_refactor_clean_repro_20260806_r2
```

## 已核对的外部权重

| 权重 | SHA256 | 状态 |
|---|---|---|
| EfficientNet PTB-XL source `best.pt` | `fff9bd293677be54a606e28775c66d410d55b0ad7440a4e7fbde55357ec13e68` | 与 source registry 一致 |
| ECGFounder PTB-XL source `best.pt` | `c23ec4361b097b74a4efed427488c4dd6a3753adfc75f7d1fc281e4df0989948` | 与 source registry 一致 |
| 官方 ECGFounder backbone | `ee199f3781f4ae1f732973267f003da0a759ea12bddb0dd28a77faa60aca7997` | 与 source registry 一致 |
| ECGTwin VAE | `c2b4ef8060e412bd503f46b66dd44fc460e67a89de0ae1809dcc771358b3fa93` | 文件存在且可读 |

## 后续追加规则

每个新问题继续使用 `RNNN` 编号，并记录：

```text
现象 -> 根因 -> 影响 -> 处理 -> 验证 -> 状态
```

不得把尚未运行的配置、历史报告或旧工作树结果写成当前 clean worktree
已经复现成功。
