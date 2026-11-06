# Traditional JSD

本分支 `direction/traditional-jsd` 只保留传统 ECG 分类器的方法、数据处理和运行入口。
主方法为既有 R18 单阶段配方：JSD 1.5，LHAT 监督质量 0.20 替换 clean loss。
两个骨干为 EfficientNet1DV2 和 ECGFounder；这些是开发配方，不能直接称为论文最终测试。
先读 [AGENTS.md](AGENTS.md)，方法与指标见 [METHOD](docs/METHOD.md)，历史证据与恢复见 [EVIDENCE](docs/EVIDENCE.md)。

## 环境与外置资源

本机显式使用项目 Python，无需改变系统环境、CUDA 或驱动：

```bash
ECG_PY=/home/linbinhao/miniforge3/envs/ECGTwin/bin/python
"$ECG_PY" -c 'import sys, torch; print(sys.version); print(torch.__version__, torch.version.cuda)'
```

新机器先用 [Conda 定义](environments/ecgtwin-miniforge.yml) 建用户级环境，再安装
[pip lock](environments/ecgtwin-pip-lock.txt)。
在本地 wheel 目录按 [SHA256 清单](environments/pytorch-cu118-wheel-manifest.sha256)
校验三个 CUDA 11.8 wheels，然后用新环境的 Python 执行 `pip install --no-deps <wheel路径>`。
最后运行 `"$ECG_PY" -m pip check`、下文 CPU 检查和所需 GPU smoke test。
环境目录、wheels、数据及权重都不随 Git clone 分发；迁移环境后以实际检查结果为准。

| 资源 | 配置或本机位置 |
| --- | --- |
| raw100 cache、split roots | [splits.yaml](configs/data/splits.yaml)、[data_load.yaml](configs/data/data_load.yaml) |
| 数据内容账本 | `configs/data/data_content_ledger_v1.jsonl`，由 data_load 配置绑定 |
| source checkpoints / SHA256 | [ptbxl_source_v1.yaml](configs/baselines/ptbxl_source_v1.yaml) |
| VAE checkpoint | `model/ECGTwin/checkpoints/vae_model.pth` |
| 新运行输出 | 实验 YAML 的 `output.run_dir`，置于 `/home/linbinhao/ECG_adv_data/runs/` |

数据路径需可读；新机器应在自己的配置中迁移路径并重建账本，不能绕过身份检查。
VAE 权重是外置资源；确认本账号具有读取权限且目标尚不存在时可建立本地绑定：

```bash
mkdir -p model
ln -s /home/linbinhao/ECG_adv_data/models/ECGTwin model/ECGTwin
sha256sum model/ECGTwin/checkpoints/vae_model.pth
```

VAE SHA256 为 `c2b4ef8060e412bd503f46b66dd44fc460e67a89de0ae1809dcc771358b3fa93`。
`model/ECGTwin` 本地链接不能提交；架构代码全部由仓库 `models/` 提供。

## 一个训练与评估

十个示例入口均通过 `boot_scripts/run_experiment.py` 启动，目标适配示例采用 Ningbo：

| 骨干 | K500 训练 / 评估 | PTB-XL source 重训 / source 迁移评估 | 四中心结果汇总 |
| --- | --- | --- | --- |
| EfficientNet | `efficientnet_train.yaml` / `efficientnet_eval.yaml` | `efficientnet_source_train.yaml` / `efficientnet_source_eval.yaml` | `efficientnet_matrix.yaml` |
| ECGFounder | `ecgfounder_train.yaml` / `ecgfounder_eval.yaml` | `ecgfounder_source_train.yaml` / `ecgfounder_source_eval.yaml` | `ecgfounder_matrix.yaml` |

以上文件都位于 `configs/experiments/`。从仓库根目录先 dry-run：

```bash
CUDA_VISIBLE_DEVICES='' "$ECG_PY" boot_scripts/run_experiment.py \
  --config configs/experiments/efficientnet_train.yaml --dry-run
CUDA_VISIBLE_DEVICES='' "$ECG_PY" boot_scripts/run_experiment.py \
  --config configs/experiments/efficientnet_eval.yaml --dry-run
```

Dry-run 只解析配置闭包，不读取数据、权重或 GPU，也不写运行目录。
检查 `nvidia-smi`、负载、内存和磁盘后显式选择空闲卡；下例假设物理 GPU 0 可用。
训练完成后才运行其配对评估：

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$ECG_PY" boot_scripts/run_experiment.py --config configs/experiments/efficientnet_train.yaml
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$ECG_PY" boot_scripts/run_experiment.py --config configs/experiments/efficientnet_eval.yaml
```

ECGFounder 换成对应入口。source_train 在新目录重训 PTB-XL，不会自动晋升为锁定 checkpoint；
source_eval 根据 source registry 评估已注册模型在 PN2021 的直接迁移表现，不会读取新重训输出。
矩阵入口在四中心训练/评估全部完成后汇总结果，不启动新训练或推理。
其 [PN2021_matrix.yaml](configs/eval/PN2021_matrix.yaml) 锁定主配方、四中心与身份；
复制到新运行时同步修改各 --result 路径，改变配方或重复编号时更新对应 cohort 锁。

## 配置修改指南

主配方选择器为 [a1_rot4_two_chain_balanced_jsd1p5_vae_lhat_replace0p2.yaml](configs/train/methods/a1_rot4_two_chain_balanced_jsd1p5_vae_lhat_replace0p2.yaml)，
训练 profile 为 [PN2021_a1_joint_long_budget_r13.yaml](configs/train/PN2021_a1_joint_long_budget_r13.yaml)；
新评估固定使用 [PN2021_online_gpu.yaml](configs/eval/PN2021_online_gpu.yaml)。
保留的基线为 `a0_clean_v1`（clean-only）和 `a1_corrupt_ft_rot4_v1`（直接腐蚀监督）。
R18 的 0.20 是监督质量替换，不是额外增加 20% loss；不要改成两阶段 JSD-12。

新中心或消融复制训练/评估 YAML 成一对，修改两侧 `--model`、`--center`、
`--method-config` 和各自的 `output.run_dir`，并让评估 `--train-result`
指向新训练目录的 `training/train_result.json`。更换骨干时还要选择其已注册 source checkpoint。
不要只改训练输出而忘记评估引用，也不要覆盖已有目录。
独立训练重复应复制主训练 profile，设置 `random_seed.replicate_id`，
让训练入口的 `entrypoint.config` 引用新 profile；评估通过新 `--train-result` 继承其身份。
训练和评估都用相同的 `--method-config`，评估的 `entrypoint.config` 仍为在线评估配置。
学习率、epoch、batch 和重复编号由训练 profile 拥有。方法 ID 与损失契约由
`core/methods/registry.py` 拥有；新增算法或损失权重需同步修改契约与测试，
不能只重命名 selector。保留主配方，用独立配置记录变体。

数据预处理入口与账本工具在 `data_preprocess/`；数据 YAML 在 `configs/data/`。
仅保存 canonical raw100 clean cache；腐蚀在线执行，不生成磁盘腐蚀波形 cache。
评估排除目标 K500，主指标为 `drop_all_zero` macro AUROC / sklearn AP，
先等权平均腐蚀视图，再等权平均中心；详细口径见 METHOD。

## 开发与验证

`core/` 拥有训练、AugMix、LHAT，`models/` 拥有两个骨干与 VAE，
`util/evaluation/` 拥有评估，`boot_scripts/` 只提供托管入口。
新实验使用明确 YAML，不增加历史队列或一次性调度器；旧实验配置可从 Git 恢复。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$ECG_PY" -m pytest -q
git diff --check
```

CPU 检查不能证明真实模型/GPU 等价。修改数值路径需对应的受限验证，
文档修改只需范围、链接与 diff 检查。
提交前使用 [artifact-git-guard](.codex/skills/artifact-git-guard/SKILL.md)，只选择本任务文件。
