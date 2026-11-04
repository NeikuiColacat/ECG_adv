# Traditional SimCLR：两阶段方法与开发交接

本分支为 `direction/traditional-simclr`。主方法锁定为 **PTB-XL source → 目标中心 K500 → 两链 AugMix + SimCLR → 监督适配 + contracted VAE-LHAT**，双骨干为 EfficientNet1DV2 与 ECGFounder。新增实验应从此方法建立独立对照，保留主配方及旧结果。

交接状态日期：2026-11-04。先读 [AGENTS.md](AGENTS.md)；本 README 是本分支的入口，[方向目录](docs/directions.md)用于查找同仓库的其他研究方向。

## 1. 锁定的方法

| 项目 | 配置 / 身份 |
| --- | --- |
| 主配方 | [augmix_simclr_lhat.yaml](configs/train/methods/augmix_simclr_lhat.yaml)，recipe ID `augmix_simclr_lhat` |
| 主配方 spec SHA256 | `6245e89f572b49bd24f5d688bda7d3f8b16fa4972051353afa311d007cd9f675` |
| 训练参数 | [PN2021.yaml](configs/train/PN2021.yaml) |
| AugMix / VAE / LHAT | [augmix.yaml](configs/train/augmix.yaml)、[vae.yaml](configs/train/vae.yaml)、[lhat.yaml](configs/train/lhat.yaml) |
| 新实验的评估 | [PN2021_online_gpu.yaml](configs/eval/PN2021_online_gpu.yaml)，`online_torch_gpu_v1` |
| 主线与证据索引 | [active_scripts.yaml](configs/active_scripts.yaml)、[active_evidence_registry.yaml](configs/active_evidence_registry.yaml) |

两个 stage 在同一次训练任务中顺序执行：

1. **Stage 1，表征适配**：clean 与一个两链 AugMix strong view 做 NT-Xent，温度 0.5；冻结分类头，训练骨干与投影头；保留冻结 source logits 的 soft-target BCE anchor，权重 5。没有 VAE、source replay 或 VICReg。
2. **Stage 2，监督适配**：解冻分类头，重建优化器；关闭 SimCLR 和 source anchor。clean/corruption BCE 权重 0.5/0.5，轮换两个 depth-2 和两个 depth-3 腐蚀视图，并启用 contracted VAE-LHAT。只取最后 checkpoint，无验证集选模。

| 骨干 | Stage 1 steps / lr | Stage 2 epochs / optimizer steps | Batch / Stage 2 lr |
| --- | --- | --- | --- |
| EfficientNet1DV2 | 1024 / 1e-3 | 25 / 100 | 128 / 5e-5 |
| ECGFounder | 256 / 1e-3 | 30 / 240 | 64 / 3e-5 |

Stage 2 使用 AdamW、weight decay 1e-4、gradient clip 1、bf16；两者 cosine scheduler horizon 都为 30 epochs。EfficientNet 的 25 epochs 不能改写成 scheduler horizon 25。

LHAT 固定：exact-label、非自身、nearest M=20；hull lambda=1，标准化 latent L2 epsilon=12，一步 Adam attack，lr=0.25。之后按 `preflip_maxloss_grid_v1` 在 t=[0.25,0.5,0.75,1] 中选择不新增 clean-correct 翻转的最高 BCE，margin retention=0.5，使用线性端点残差校正。base/auxiliary 梯度直接相加，无 PCGrad。

`augmix_simclr_matched_no_vae` 是保留相同 Stage 1 的两阶段消融。单阶段 `augmix_simclr_joint_no_vae` 只作为已完成的辅助实验保留，未晋升主线；[其协议和负结果](docs/simclr_joint_protocol.md)单独记录。JSD 属于另一方向。

## 2. 环境、数据与权重

### 获取代码与 Python 环境

```bash
git clone --single-branch --branch direction/traditional-simclr \
  git@github.com:NeikuiColacat/ECG_adv.git
cd ECG_adv
git branch --show-current

ECG_PY=/home/linbinhao/miniforge3/envs/ECGTwin/bin/python
"$ECG_PY" -c 'import sys, torch; print(sys.version); print(torch.__version__, torch.version.cuda)'
```

本机验证环境为 Python 3.11.5、PyTorch 2.1.1+cu118 / CUDA build 11.8、NumPy 1.26.4、scikit-learn 1.8.0。始终显式使用项目 Python；无需修改系统 CUDA 或驱动。

新机器的环境定义见 [environments/README.md](environments/README.md)：先创建 [Conda 环境](environments/ecgtwin-miniforge.yml)，安装 [pip lock](environments/ecgtwin-pip-lock.txt)，再校验并安装 [manifest](environments/pytorch-cu118-wheel-manifest.sha256) 中三个 CUDA 11.8 wheels（`--no-deps`）。这些 wheel 是外置依赖，需要自行准备；Git clone 不包含环境、数据或模型权重。环境迁移后执行 `"$ECG_PY" -m pip check` 和下文检查。

本机 `pip check` 目前报告额外安装的 `imagecorruptions` 缺少其声明的 `opencv-python`；该包未列入上述 lock，也未被本分支传统训练/评估路径导入。本次未修改共享环境，详见交接验证记录。

### 本服务器的外置资源

| 资源 | 路径 / 配置 |
| --- | --- |
| PTB-XL raw100 cache | `/data/linbinhao/ECG_manual_refactor_cache/ptbxl_100hz_qc_v3_20260716` |
| PN2021 raw100 cache | `/data/linbinhao/ECG_manual_refactor_cache/pn2021_100hz_linear_qc_v3_20260716` |
| 固定 K500 split root | `/home/linbinhao/ECG_adv_data/splits/manual_refactor_super5_v2_k500_tune_seed20260501` |
| PN2021-C 元数据目录 | `/data/linbinhao/ECG_manual_refactor_cache/pn2021c_paper_anchored_s5_depth23_5center_dualrate_v1` |
| 两个 source checkpoints | [ptbxl_source_v1.yaml](configs/baselines/ptbxl_source_v1.yaml) 的 `selected_checkpoint` / SHA256 |
| VAE checkpoint | `model/ECGTwin/checkpoints/vae_model.pth`，SHA256 `c2b4ef8060e412bd503f46b66dd44fc460e67a89de0ae1809dcc771358b3fa93` |
| 新 run 输出位置 | `/home/linbinhao/ECG_adv_data/runs/`，由实验 YAML 指定 |

数据根目录由 [splits.yaml](configs/data/splits.yaml) 声明，加载参数及账本由 [data_load.yaml](configs/data/data_load.yaml) 声明。PN2021-C 目录仅保留身份、标签、20 种组合及算子参数等元数据；已退役的腐蚀波形数组不需要恢复。

新 clone 在确认路径存在且有读取授权后，建立本地 VAE 权重绑定（只在尚未绑定时执行）：

```bash
mkdir -p model
ln -s /home/linbinhao/ECG_adv_data/models/ECGTwin model/ECGTwin
sha256sum model/ECGTwin/checkpoints/vae_model.pth
```

`model/ECGTwin` 是忽略的本地链接，不能提交。模型架构实现均在本仓库 `models/`，不跨工作树导入外部代码。其他账号/机器需准备自己可读的数据、权重和新输出目录，并在自己的版本化配置中明确迁移路径、重建对应账本；不要用随意更换哈希的方式绕过身份检查。

## 3. 从 dry-run 到训练和评估

交接提供两组明确配对的 YAML，均使用完整两阶段主配方和固定 K500：

| 骨干 | 训练 | 在线 GPU 评估 |
| --- | --- | --- |
| EfficientNet1DV2 | [efficientnet_ningbo_train.yaml](configs/experiments/simclr_handoff/efficientnet_ningbo_train.yaml) | [efficientnet_ningbo_eval.yaml](configs/experiments/simclr_handoff/efficientnet_ningbo_eval.yaml) |
| ECGFounder | [ecgfounder_ningbo_train.yaml](configs/experiments/simclr_handoff/ecgfounder_ningbo_train.yaml) | [ecgfounder_ningbo_eval.yaml](configs/experiments/simclr_handoff/ecgfounder_ningbo_eval.yaml) |

从仓库根目录执行。第一步只解析配置闭包，不读取数据/权重、不使用 GPU、不写 run：

```bash
ECG_PY=/home/linbinhao/miniforge3/envs/ECGTwin/bin/python
CUDA_VISIBLE_DEVICES='' "$ECG_PY" boot_scripts/run_experiment.py \
  --config configs/experiments/simclr_handoff/efficientnet_ningbo_train.yaml --dry-run
CUDA_VISIBLE_DEVICES='' "$ECG_PY" boot_scripts/run_experiment.py \
  --config configs/experiments/simclr_handoff/efficientnet_ningbo_eval.yaml --dry-run
```

默认输出为 `runs/traditional_simclr_handoff_v1/<model>/ningbo` 与 `ningbo_eval`。执行前确认目录未被使用。复跑时复制这组 YAML，给训练与评估分别设置新 `output.run_dir`，同时把评估的 `--train-result` 改为新训练目录下 `training/train_result.json`，再 dry-run。仅修改训练的 CLI `--run-dir` 不会自动重写评估所引用的训练路径。

检查 `nvidia-smi`，选确实空闲且获授权的卡；下例假设物理 GPU 0 空闲。先完成训练，再评估：

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$ECG_PY" boot_scripts/run_experiment.py \
  --config configs/experiments/simclr_handoff/efficientnet_ningbo_train.yaml

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$ECG_PY" boot_scripts/run_experiment.py \
  --config configs/experiments/simclr_handoff/efficientnet_ningbo_eval.yaml
```

ECGFounder 换成表中相应 YAML。宁波仅是接入示例；其他中心应复制成各自明确的训练/评估配置，同时修改两侧 `--center`、输出位置和训练结果引用。完整网格为双骨干 × 四中心，每个模型只在其目标中心 K500 上适配。

唯一运行入口为 `boot_scripts/run_experiment.py`。不要直接调用底层脚本绕过 launcher，也不要覆盖已有目录。训练产物包括 `training/train_result.json`、`training/training_history.json`、`training/checkpoints/last.pt` 和资源身份；评估产物为 `evaluation/evaluation_result.json`、`evaluation/evaluation_progress.json`。run 根目录的 manifest、配置快照和 file index 用于追溯。

## 4. 数据、在线腐蚀与指标口径

- Super5 顺序：`CD,HYP,MI,NORM,STTC`；mapping `v7_super5_sjr_rgq_review_20260528` / `555ec85d5b51`。
- 四个逻辑中心：`ningbo`、`chapman_shaoxing`、`cpsc_2018`（包含 Extra）、`georgia`。
- 主线每中心固定 500 个非全零标签样本用于适配；评估排除这些 ID。目标中心其他记录只作开发评估，不参与适配或选 checkpoint。
- 传统模型输入 raw mV `(B,1000,12)`、100 Hz、PTB-XL 导联序；先腐蚀再做逐样本 global z-score。Founder 在设备上插值至 5000 点再标准化。
- 训练腐蚀与新评估使用已有 Torch 算子：raw100 → GPU500 → GPU100，后续模型域适配、标准化、推理也在 GPU。clean cache 仍以只读 mmap 加载，不生成或保存腐蚀波形 cache。
- 评估固定 severity 5、depth-2/depth-3 各 10 种组合。随机流以有序记录 ID、组合和 cache version 为键，固定 32 条微批，不包含骨干、方法或训练 seed；128/512 的模型 batch 得到相同腐蚀视图。
- `online_torch_gpu_v1` 与历史 NumPy cache 的随机实现不同。配对比较必须使用同一 backend、同一 cohort；旧缓存结果不能直接作为在线结果的因果对照。

主指标为 **macro AUROC 和 macro AP**，使用 `drop_all_zero`。结果字段中的 `macro_auprc` 实际为 `sklearn.average_precision_score`，即 Average Precision，未使用梯形 PR 积分。AP 和 AUROC 数值不必处于同一量级。原始字段为 0–1；表中乘 100，0.01 的绝对差记为 **1 个百分点**。

先在每中心对 20 个腐蚀视图等权平均，再对四中心等权平均；多训练种子再对每个 seed 的均值求 mean / sample SD。`all_zero_kept` 仅为附加审计视图。

比较时注明参照：source 原模型、clean-only 微调、直接腐蚀监督、两阶段 no-VAE 是不同基线。评估 VAE 的增益应使用相同 Stage 1、source、K500、seed、预算与 backend 的 `augmix_simclr_matched_no_vae`；完整流程相对 clean-only 的差值不能全部归因于 VAE。LHAT 保留 raw ASR、contracted ASR、接受率、BCE gain、选中 t、invalid decode rate，不单看收缩后的 ASR。

## 5. 已有结果与当前待办

经典完整版已有 **3 个独立训练种子 × 2 骨干 × 4 中心**，24 份训练和 24 份评估子任务完成并核验。以下为历史 NumPy-cache backend、固定 K500、drop_all_zero 的百分制 mean ± sample SD：

| 骨干 | Clean AUROC | Clean AP | PN2021-C AUROC | PN2021-C AP |
| --- | --- | --- | --- | --- |
| EfficientNet1DV2 | 87.25 ± 0.06 | 64.70 ± 0.11 | 85.98 ± 0.07 | 62.34 ± 0.06 |
| ECGFounder | 91.58 ± 0.10 | 71.89 ± 0.09 | 89.15 ± 0.04 | 67.81 ± 0.23 |

报告：`/home/linbinhao/ECG_adv_data/runs/jsd_simclr_width_repeat_20260912_v1/dual_jsd_workflow/training/width_comparison.json` 的 `seed_summary[model]["simclr"]`，SHA256 `58b75a4010b5bda792a8aa7dd66b11d0d8ea7aaf2e90cb7594d5e215131fd1bc`。子任务证据已验证；历史父 workflow 的收尾状态未修复，不能据此声称整个父 run 已完整归档。

另有已完成的 [单阶段/no-VAE 对照](docs/simclr_joint_protocol.md)，96 个子任务。该 joint 配方的 PN2021-C AP 比配对两阶段 no-VAE 低约 7.67 / 7.09 个百分点，故主方法保持两阶段。此结果只适用于该配方与预算。

[K500 重抽样实验](docs/k500_sampling_protocol.md)测试的是抽样敏感性：三个新 K500、固定训练 seed、共同评估集排除三次 K500 并集；它与训练 seed 重复不同。目前原队列完成 2/48 训练，0/48 评估，首次评估因把元数据总数固定为 10 而退出。新增 `RETIREMENT_20261017.md` 造成误拒绝。

本次修复逐一验证十个**指定**元数据文件的大小和全内容哈希，明确记录未验证的 sidecar，继续拒绝缺失、篡改及符号链接；旧的完整缓存门禁保持严格。原训练结果、失败日志及 source snapshots 保留。修复验证与完整队列恢复的状态见 [交接验证记录](docs/simclr_handoff_validation.md)。

**完整抽样队列尚未恢复。** 原工程准入绑定旧源码哈希，新代码不能直接复用该准入或覆盖失败目录。后续需独立的新恢复配置、确认旧训练 artifact 的复用边界、重新做工程准入，再按空闲 GPU 运行。不得手改旧 source hash/admission 或将剩余任务写成已完成。

现有配方及中心有历史开发调参；三次重复也不能自动升级为论文最终独立测试。论文最终结论仍需冻结方案后的独立测试证据。

## 6. 验证与开发边界

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$ECG_PY" -m pytest -q
git diff --check
```

完整 CPU suite 包含其他方向的遗留测试；本机已知基线失败及本次验证结果见 [验证记录](docs/simclr_handoff_validation.md)。不得为使测试通过而放宽数据身份或路径安全检查。

| 修改内容 | 优先位置 |
| --- | --- |
| 两阶段训练 / 梯度、BN、RNG | `core/online_trainer.py`、`util/tests/test_online_trainer.py` |
| 配方身份 | `core/methods/registry.py`、`configs/train/methods/`、`test_method_graph.py` |
| AugMix / GPU 腐蚀 | `core/augmix.py`、`core/corruption.py`、`util/augmentations/` |
| VAE / LHAT | `models/vae.py`、`core/lhat.py`、`core/latent_pool.py` |
| 数据、账本、K500 | `data_preprocess/`、`util/tests/test_data_runtime.py` |
| 指标、结果身份 | `util/evaluation/pn2021.py`、`util/pn2021_artifact_contract.py` |
| 有限实验调度 | `util/pulse_training_queue.py`、`util/simclr_joint_queue.py` |
| 新配置 / 证据登记 | `configs/experiments/`、`configs/active_scripts.yaml`、`configs/active_evidence_registry.yaml` |

开始修改前核对 branch、dirty state、活跃任务及其 source guard；每位同学使用独立 clone/worktree，运行中的源码不改。新消融建新 selector/profile，保留主配方 SHA 和已冻结参数。提交前按 [artifact-git-guard](.codex/skills/artifact-git-guard/SKILL.md)逐项检查，只 stage 任务相关源码、配置和文档；不提交数据、权重、日志、环境、token、本地模型链接。文件保留边界由 [keep manifest](docs/refactor_cleanup/manual_refactor_keep_manifest.md)管理。
