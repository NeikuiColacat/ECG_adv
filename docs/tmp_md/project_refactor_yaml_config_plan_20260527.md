# ECG_adv_Gen YAML 配置管理与项目重构计划

日期：2026-05-27

关联阅读：

- `docs/tmp_html/project_refactor_plan_20260527.html`
- `docs/tmp_md/vae_online_at_closing_summary_20260525.md`
- `AGENTS.md`
- `.codex/skills/ecg-vae-online-at/SKILL.md`

## 1. 核心结论

建议引入 YAML 配置管理，而且应作为当前项目重构的第一批落地事项之一。

原因不是“YAML 更好看”，而是当前实验已经进入论文主线阶段，最容易出错的地方不是模型 forward，而是：

- PN2021 Super5 mapping 版本是否一致；
- K=500 target-center ref ids 是否被正确 exclude；
- direct fine-tune 和 VAE online AT 是否使用同一批 K500；
- final selector 是否只使用 K500 internal validation 和 source floor；
- all-zero kept 和 drop-all-zero 指标是否被混算；
- 旧机器 `/root/autodl-tmp` 路径是否被误用；
- 长 bash 命令中某个关键参数是否漏传。

YAML 的目标是把“实验定义”从超长 bash 命令中拿出来，让每个论文结果都能被一个 tracked config、一个 local config、一个 resolved run config 和一个 manifest 追溯。

## 2. 设计原则

### 2.1 Bash 只负责启动，不负责定义实验

推荐启动方式：

```bash
micromamba run -n ECGTwin python scripts/run_experiment.py \
  --config configs/experiments/effnet_vae_lhat_k500_v6.yaml \
  --local-config configs/local/linbinhao_server.yaml \
  --run-id effnet_vae_lhat_v6_k500_seed20260531_20260527 \
  --dry-run
```

正式运行时先运行 `nvidia-smi`，确认目标 GPU 空闲，再显式设置
`CUDA_VISIBLE_DEVICES=...` 并使用 `--execute`，不要只删除 `--dry-run`。

Bash 中只保留：

- 使用哪张 GPU；
- 使用哪个实验 YAML；
- 使用哪个本地路径 YAML；
- 本次 `run_id`；
- 是否 dry-run / resume / force。

不要把 20-50 个训练参数长期写在 bash 里。

### 2.2 Tracked YAML 只保存论文实验定义

应该进 Git 的 YAML：

- backbone；
- mapping version/hash expectation；
- centers；
- K-shot policy；
- preprocessing；
- model training hyperparameters；
- VAE latent-hull / AugMix / PGD 参数；
- selection policy；
- evaluation views；
- random seeds；
- artifact requirements。

不应该进 Git 的 YAML：

- 本机绝对数据路径；
- GPU id；
- 私人临时目录；
- 当前机器 Python 路径；
- 外部模型本地 symlink 目标；
- 任何私钥、token 或认证信息。

### 2.3 本地 YAML 只保存机器环境

本地 YAML 放在 `configs/local/`，并加入 `.gitignore`。

它只回答“这台机器上数据和模型在哪里”，不回答“论文实验怎么跑”。

### 2.4 每次运行必须冻结 resolved config

YAML 可以分层、继承、被 CLI override，但最终 run 目录里必须保存展开后的完整配置：

```text
run_config.resolved.yaml
run_config.resolved.json
run_manifest.json
command.sh
```

以后论文审稿、复现、debug 都看 resolved config，不再猜当时 bash 传了什么。

## 3. 推荐目录结构

```text
configs/
  README.md
  schemas/
    experiment_config.schema.json
    local_config.schema.json
  defaults/
    common_v6_super5.yaml
    effnet1dv2_defaults.yaml
    ecgfounder_defaults.yaml
    vae_lhat_defaults.yaml
  experiments/
    effnet_direct_k500_v6.yaml
    effnet_vae_lhat_k500_v6.yaml
    effnet_vae_lhat_augmix_k500_v6.yaml
    ecgfounder_direct_k500_v6.yaml
    ecgfounder_inithead_fullft_k500_v6.yaml
    ecgfounder_vae_lhat_k500_v6.yaml
    pn2021_eval_v6_refexcluded.yaml
  local/
    .gitignore
    linbinhao_server.example.yaml
```

`configs/local/.gitignore` 内容建议：

```gitignore
*
!.gitignore
!*.example.yaml
```

这样本机真实路径不会被提交，但可以保留 example。

## 4. 配置分层规则

配置合并优先级从低到高：

1. code defaults：只放极少量安全默认值；
2. `configs/defaults/*.yaml`：模型/方法默认参数；
3. `configs/experiments/*.yaml`：论文实验定义；
4. `configs/local/*.yaml`：本机路径和资源；
5. CLI override：只允许少量字段，如 `run_id`、`resume`、`dry_run`。

路径解析优先级：

1. CLI 显式参数；
2. local YAML；
3. 环境变量：`ECG_ADV_GEN_PROJECT_ROOT`、`ECG_ADV_GEN_DATA_ROOT`、`ECG_ADV_GEN_MODEL_ROOT`、`ECG_ADV_GEN_PYTHON`；
4. 当前迁移 host fallback；
5. 报错。

写入路径必须经过 `Path.resolve()` 后检查，默认必须在 `/home/linbinhao` 下。不能默认写到 `/root`。

## 5. 实验 YAML 字段草案

### 5.1 顶层字段

```yaml
experiment:
  name: effnet_vae_lhat_k500_v6
  version: 1
  description: EfficientNet1DV2 direct-K500 plus ECGTwin VAE latent-hull online AT.
  owner: linbinhao
  status: active-main

paper_protocol:
  mapping_version: v6_super5_clinician_review_20260524
  mapping_hash: 3adc673a60ad
  class_order: [CD, HYP, MI, NORM, STTC]
  centers: [ningbo, chapman_shaoxing, cpsc_2018, georgia]
  kshot:
    mode: fixed_k
    k: 500
    seed: 20260531
    subset_seed: 20260531
    exclude_refs_from_eval: true
  selection:
    policy: k500_internal_val_plus_source_floor
    forbid_heldout_target_labels: true
    forbid_full_target_distribution_tuning: true
```

### 5.2 数据与预处理

```yaml
data:
  source_dataset: ptbxl
  target_dataset: pn2021
  pn2021_centers: [ningbo, chapman_shaoxing, cpsc_2018, georgia]
  exclude_centers: [ptb-xl, ptbxl]
  cache:
    pn2021_cache_version: v6_super5_clinician_review_20260524
    prefer_mmap: true

preprocess:
  classifier_fs: 100
  classifier_len: 1000
  mode: minimal_resample
  norm_mode: per_sample_global
  lead_order: ptbxl
  ecgtwin_decode:
    input_len: 1024
    output_len: 1000
    reorder_indices: [0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11]
```

### 5.3 模型与训练

```yaml
model:
  backbone: efficientnet1dv2
  num_classes: 5
  init_checkpoint: ${paths.data_root}/triple_labels/super5_minresample_full10_perglobal_20260503/best_model.pt

training:
  epochs: 20
  batch_size: 128
  num_workers: 4
  amp: true
  amp_dtype: bf16
  optimizer:
    name: adamw
    lr: 0.0001
    weight_decay: 0.0001
  checkpoint:
    save_best_clean: true
    save_best_robust_val: true
    save_last: true
```

### 5.4 VAE latent-hull online AT

```yaml
adaptation:
  method: vae_latent_hull_online_at
  vae:
    name: ecgtwin_vae
    latent_shape: [4, 128]
    latent_scale: 0.18215
    standardize_latents: true
  anchors:
    source: target_center_k500
    include_clean_anchor_in_batch: true
    partner_policy: local_same_label_or_compatible
    forbid_norm_abnormal_mix_main: true
    rare_class_anchor_quota: true
  hull:
    include_anchor: true
    coefficient_mode: anchor_dominant
    hull_lambda: 0.05
  attack:
    name: latent_pgd
    steps: 5
    epsilon: 2.0
    step_size: 0.5
    random_start: true
    target_asr_range: [0.30, 0.70]
  loss:
    clean_weight: 1.0
    adv_weight: 0.3
    adv_weight_warmup_epochs: 10
    label_mode: latent_mixed_teacher
  quality:
    hard_reject_invalid_decode: true
    use_victim_score_gate: false
```

### 5.5 评估与日志

```yaml
evaluation:
  views:
    - pn2021_all_zero_kept_refexcluded
    - pn2021_drop_all_zero_refexcluded
    - ptbxl_fold10_source_floor
  metrics:
    - macro_auroc
    - macro_auprc
    - per_center
    - per_class
  min_pos: 10

logging:
  required_epoch_metrics:
    - clean_bce
    - adv_bce
    - loss_gain
    - atk_init
    - atk_anchor
    - asr_anyflip
    - invalid_decode_rate
    - source_floor_auroc
    - source_floor_auprc
  launch_artifacts:
    - run_config.resolved.yaml
    - run_config.resolved.json
    - run_manifest.json
    - command.sh
    - k500_ref_ids.json
    - selection.json
  child_artifacts:
    - training_log.json
    - train_result.json
    - eval_result.json
  postprocess_artifacts:
    - metrics_long.csv
```

## 6. 本地 YAML 字段草案

`configs/local/linbinhao_server.example.yaml`：

```yaml
host:
  name: linbinhao_shared_server
  shared_server: true
  home_root: /home/linbinhao

paths:
  project_root: /home/linbinhao/ECG_adv_Gen
  data_root: /home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp
  model_root: /home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/models
  output_root: /home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs
  cache_root: /home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/cache
  tmp_root: /home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/tmp

python:
  executable: /home/linbinhao/micromamba/envs/ECGTwin/bin/python
  micromamba_env: ECGTwin

resources:
  default_num_workers: 4
  prefer_single_gpu: true
  require_nvidia_smi_check: true
  default_bind_host: 127.0.0.1

safety:
  forbid_sudo: true
  forbid_system_env_changes: true
  write_boundary: /home/linbinhao
  forbid_commit_large_artifacts: true
```

真实 `linbinhao_server.yaml` 可以基于 example 复制，但不要提交。

## 7. 运行目录产物规范

每次 run 的输出目录建议：

```text
${output_root}/<date>/<run_id>/
  run_config.resolved.yaml
  run_config.resolved.json
  run_manifest.json
  command.sh
  stdout.log
  stderr.log
  training_log.json
  train_result.json
  eval_result_v6_super5_refexcluded.json
  metrics_long.csv
  selection.json
  k500_ref_ids.json
  checkpoints/
    best_clean.pt
    best_robust_val.pt
    last.pt
```

`run_manifest.json` 至少记录：

- git commit；
- dirty files summary；
- script entry；
- argv；
- resolved config hash；
- mapping version/hash；
- K500 record ids file；
- input checkpoint path；
- output artifacts；
- start/end time；
- status；
- whether held-out target labels were used for selection。

当前实现中，整体 manifest 有创建/更新时间；每条实际执行的 legacy child
command 会写入顶层 `command_runs`，每条 managed reporting command 会写入顶层
`postprocess_runs`。记录字段包含 command index、name、returncode、status、
combined log path、split stdout/stderr log paths、UTC start/end timestamp；
失败命令也会先落盘记录再返回错误。run 目录同时会维护 aggregate
`stdout.log` / `stderr.log`，便于快速查看所有 child/postprocess 输出。正式
execute 还会写入 `execution_started_at_utc`、最终 `finished_at_utc` 和
`duration_seconds`，覆盖有 postprocess 与无 postprocess 两种路径。

## 8. CLI 与实现建议

新增一个统一入口：

```text
scripts/run_experiment.py
```

第一阶段不需要重写所有训练逻辑。它可以先做 wrapper：

1. 读取 YAML；
2. 合并 local YAML；
3. 做 schema / safety / path dry-run；
4. 生成旧脚本需要的命令；
5. 写 `run_config.resolved.*` 和 `command.sh`；
6. 调用现有脚本。

推荐 CLI：

```bash
micromamba run -n ECGTwin python scripts/run_experiment.py \
  --config configs/experiments/effnet_vae_lhat_k500_v6.yaml \
  --local-config configs/local/linbinhao_server.yaml \
  --run-id effnet_vae_lhat_v6_k500_seed20260531 \
  --dry-run
```

允许的 CLI override：

- `--run-id`
- `--output-dir`
- `--resume`
- `--dry-run`
- `--force`
- `--set key=value`，仅限白名单字段；当前实现只允许
  `resources.default_num_workers`、`training.*` 中的安全训练超参，以及
  `adaptation.hull/loss/latent_augmix` 中的数值方法超参。禁止通过
  `--set` 修改 mapping、K500、中心列表、路径、checkpoint、环境变量或
  raw runner argv。所有 accepted override 都会写入 resolved config 和
  `run_manifest.json`。

不建议允许任意 override，因为这会重新制造“长 bash 难复现”的问题。

## 9. 最小落地路线

### Phase 0：文档与 schema 冻结

产物：

- `configs/README.md`
- `configs/schemas/experiment_config.schema.json`
- `configs/schemas/local_config.schema.json`
- `configs/local/.gitignore`
- `configs/local/linbinhao_server.example.yaml`

验收：

- 不改训练逻辑；
- 不启动 GPU；
- 文档明确 tracked YAML 与 local YAML 边界。

### Phase 1：path resolver 与 config loader

产物：

- `util/project_paths.py` 或未来 `ecg_adv_gen/paths.py`
- `util/config_loader.py` 或未来 `ecg_adv_gen/config.py`
- CPU-only 单测

验收：

- 无 `/root/autodl-tmp` 的环境下，能通过 local YAML 正确解析路径；
- 写入路径默认必须在 `/home/linbinhao` 下；
- `--dry-run` 不创建大文件、不使用 GPU。

### Phase 2：先包三条主线

优先 YAML：

```text
configs/experiments/effnet_direct_k500_v6.yaml
configs/experiments/effnet_vae_lhat_k500_v6.yaml
configs/experiments/ecgfounder_direct_k500_v6.yaml
configs/experiments/ecgfounder_inithead_fullft_k500_v6.yaml
configs/experiments/ecgfounder_vae_lhat_k500_v6.yaml
```

验收：

- 能生成与旧 bash 等价的命令；
- 每个 run 都写 resolved config 和 manifest；
- K500 ref-meta paths、mapping hash、selection policy、input checkpoint/head、
  frozen `k500_ref_ids.json`、child output root 和 expected eval artifacts
  被强制写入 manifest。

当前已落地状态：

- `effnet_direct_k500_v6`、`effnet_vae_lhat_k500_v6`、
  `ecgfounder_direct_k500_v6`、`ecgfounder_inithead_fullft_k500_v6`、
  `ecgfounder_vae_lhat_k500_v6` 和 `pn2021_eval_v6_refexcluded`
  都能通过 YAML 生成 legacy 命令；
- VAE-LHAT wrapper 已显式使用 direct-K500 `--init_ckpt` 和 compatsoft
  direct-init 参数，不再隐式回退 source-only checkpoint；
- `run_manifest.json` 已加入 `manifest_schema_version=2` 与
  `artifact_trace`，可追溯 K500 ref-meta、anchors、init checkpoints/heads、
  selection policy、child output roots 和 expected eval JSON；
- write-plan/execute 会把 K500 ref-meta 展开为统一的 `k500_ref_ids.json`，
  记录每中心 ref ids 与 source ref-meta hash；
- write-plan/execute 会生成 `selection.json`，明确记录 selection policy 只
  允许 K500 internal validation 与 PTB-XL source floor，不允许 held-out
  target labels 或 full target distribution tuning；
- VAE-LHAT 主线 config 已声明 `logging.required_epoch_metrics`，launcher
  artifact verification 会检查 `training_log.json` 至少包含攻击强度/成功率、
  latent AugMix 或 ECGFounder attack-vs-anchor、target validation 与 source
  floor 相关字段，避免“跑完但缺少关键诊断日志”；
- launcher execute 现在同时写 per-command combined log、per-command
  stdout/stderr split logs，以及 run-level aggregate `stdout.log` /
  `stderr.log`；manifest 的 `command_runs` / `postprocess_runs` 会记录这些
  路径，失败命令也保留 split log 入口；
- runner/postprocess protocol audit 现在会拒绝不包含 resolved
  `runtime.run_id` 的 managed 输出路径，覆盖 `--out_root`、`--out_dir`、
  `--output_path`、`--output-dir`、`--output_dir`，降低重复实验覆盖旧结果的风险；
- experiment config schema 现在强制要求 `logging` 生命周期字段，并要求
  YAML-managed `postprocess.commands` 的 expected artifacts 至少包含
  `role` 与 `path`，避免 postprocess 产物只存在于文档而没有进入 manifest
  校验；
- `configs/local/.gitignore` 的边界已被测试锁定为只允许 `.gitignore` 和
  `*.example.yaml` 入库，真实 local YAML、私有路径和凭据不会被误提交；
- generated legacy commands 会经过协议审计，拦截缺失关键参数、K500/seed
  不一致、seed42 回退、缺 direct-init checkpoint、YAML 覆盖
  `CUDA_VISIBLE_DEVICES` 等错误；
- `scripts/run_experiment.py` 已支持白名单 `--set key=value`，在 interpolation
  前应用，允许小规模 smoke/资源超参调整，同时拒绝 paper protocol、路径、
  center/K-shot 和 runner argv 漂移；
- 新增 `configs/experiments/pn2021_eval_v6_refexcluded.yaml`，作为
  `eval_crosscenter.py` 的 managed ref-excluded 评估入口；它按四个 target
  center 展开 direct-K500 model_dir，并在每条 eval command 中传入四个
  K500 ref-meta，强制输出 all-zero-kept 和 drop-all-zero PN2021 v6 视图；
- 新增 `configs/experiments/ecgfounder_direct_k500_v6.yaml`，作为
  `run_ecgfounder_kshot_head_ft_20260517.py` 的 managed ECGFounder
  frozen-feature direct K500 head fine-tune baseline；它追踪 v6 linear-probe
  head/features、四个 K500 ref-meta 和每中心 `best_head.pt`/`eval_result.json`；
- 新增 `configs/experiments/ecgfounder_vae_lhat_k500_v6.yaml`，作为
  `run_ecgfounder_vae_only_lhat_head_ft_20260523.py` 的 managed VAE-LHAT
  入口；它按四个 target center 展开，追踪 ECGFounder checkpoint、
  linear-probe feature cache、K500 direct base head、K500 latent anchors，
  并强制 `selection_source=target_real_val`，GPU id 仍由外层
  `CUDA_VISIBLE_DEVICES` 控制；
- `--execute` 在 legacy child scripts 启动前会验证 required inputs，在
  child scripts 返回 0 后还会验证 launch/child artifacts 和 eval JSON
  mapping metadata，验证失败不会标记 succeeded；
- `scripts/run_experiment.py` 已支持 YAML-managed `postprocess.commands`：
  child artifacts 先通过校验，再执行受 allowlist 限制的 reporting/export
  postprocess，最后把 postprocess artifacts 一并纳入 manifest 校验；
- 6 个 active paper/evaluation config 都已声明 managed reporting
  postprocess：5 个 config 导出 `metrics_long.csv`、all-zero-kept table 和
  drop-all-zero table；`ecgfounder_direct_k500_v6` 由于 legacy direct-head
  结果不含 drop-all-zero view，目前只导出 `metrics_long.csv` 与
  all-zero-kept table；
- 新增 `scripts/audit_managed_configs.py` 和 `ecg_adv_gen.config.audit`，
  可对 `configs/active_scripts.yaml` 中所有 `active_wrapped` config 做
  CPU-only preflight：解析 YAML、展开 legacy command、运行 protocol audit、
  构建 manifest trace，并可选验证 required input artifacts；
- CPU-only refactor 测试当前为 `229 passed`，其中
  `util/tests/test_adaptation_lhat.py` 覆盖 pure Latent-Hull helper、
  `StratifiedPoolWalker`、`SameLabelLatentIndex` 和 array-level latent
  AugMix core，`util/tests/test_config_loader.py` 覆盖白名单 `--set`
	  override、PN2021 ref-excluded eval wrapper、ECGFounder VAE-LHAT wrapper
	  及其协议漂移拒绝逻辑，并新增 selection policy gate，要求
	  `allowed_data` 只能是 K500 train split、K500 internal validation 和 PTB-XL
  source floor；`util/tests/test_run_naming.py` 覆盖 ECGFounder full-FT、
  EfficientNet direct-K500 和 EfficientNet VAE-LHAT/AugMix 的
  legacy-compatible child run leaf；
	  source floor，且 runner argv 不能退回 `pn2021_heldout` selection；
	  现在还覆盖 ECGFounder full-FT manifest child run-dir 与 shared
	  `run_naming` helper 的一致性，避免 launcher expected artifact path
	  和 legacy runner 实际输出路径漂移；同时新增 subprocess import
	  回归，确认 `ecg_adv_gen.config.loader` 不会加载 `torch`；
  现在还覆盖 `required_epoch_metrics`，会拒绝缺少 VAE attack success / ASR
  等关键 epoch 指标的 `training_log.json`；
  同时新增 run-id-scoped output gate，防止 managed child/postprocess 输出
  路径绕过 `runtime.run_id`，并覆盖 schema 对 `logging` 生命周期字段、
  postprocess expected artifact shape 的强约束；
  `util/tests/test_active_script_index.py` 现在还会
  扫描 tracked YAML 源文件，禁止把本机绝对路径、`CUDA_VISIBLE_DEVICES`、
  Python executable 或明显密钥字段写入 defaults/experiments/index，并锁定
  `configs/local/.gitignore` 只能允许 `.gitignore` 与 `*.example.yaml`；
  `util/tests/test_real_anchors.py` 覆盖 real-anchor latent base 查找、
  PN2021 cache record-id/center 对齐、legacy min-one K-shot anchor selection
  和 explicit selected-id pool loading；
  `util/tests/test_evaluation_selection.py` 现在同时覆盖 ECGFounder full-FT
  目标中心 K-shot train/val split 的 random、stratified、zero-val 和错误输入
  分支，以及 source-only、target-val 和 source+target-val blend selection
  score helper；
  `util/tests/test_pn2021_index_kshot.py` 现在还覆盖 legacy K-shot
  ref-meta JSON 中 `record_ids`、`ref_record_ids`、`selected_ref_record_ids`、
  top-level row list 和 `items` 的 selected id 解析，以及 ECGFounder
  direct-head exact-K/source-K ref-meta fallback 和 min-one primary-class
  K-shot selection；
  `util/tests/test_pn2021_metric_views.py` 现在还覆盖 ECGFounder legacy
  target-ref-excluded PN2021 view loop 与 package helper 的等价行为，以及
  `eval_crosscenter.py` PN2021 per-center row 形状，包括 `n_scanned`、
  cache metadata、drop-all-zero 字段和零值 `n_include_kept`；
  `util/tests/test_evaluation_metrics.py` 覆盖共享 ECG macro
  AUROC/AUPRC helper 的 sklearn 数值一致性、`min_pos`/all-positive class
  gate、unknown-label masking、EfficientNet legacy wrapper schema、
  dict wrapper schema 和输入 shape 校验；
  `util/tests/test_models_ecgfounder.py` 现在还覆盖 ECGFounder full-FT
  `model.dense` 从 Super5 linear head checkpoint 初始化的成功路径、缺字段、
  缺 dense、dense 类型错误和 shape mismatch，以及 ECGFounder 100Hz ->
  500Hz torch input conversion / per-sample global z-score helper、full signal
  model prediction 和 indexed split evaluation helper；
  `util/tests/test_anchor_sampling.py` 现在还覆盖 ECGFounder full-FT
  anchor class-weight 解析、带 repeat cap 的 weighted class quota，以及
  weighted index sampling fallback、signal-model anchor hard-BCE/uncertainty
  difficulty score 和 positive-boundary classwise score；
  `util/tests/test_training_losses.py` 覆盖 shared masked BCE / `pos_weight`
  helper、masked per-sample BCE、stream-weighted masked BCE、ECGFounder
  VAE-LHAT attack-success diagnostics、full-FT VAE adversarial batch/epoch
  diagnostics、pairwise ranking loss 与 legacy 兼容出口；
  `util/tests/test_run_naming.py` 覆盖 ECGFounder full-FT legacy run leaf、
  method tag、selection tag、explicit `run_name` override 和字符串布尔值兼容；
  `util/tests/test_training_splits.py` 覆盖 direct EfficientNet 与
  ECGFounder K-shot runner 共用的 deterministic random train/val split
  helper，并锁定 zero-val identity/permuted 两种 legacy policy；
  `util/tests/test_signal_streams.py` 覆盖 ECGFounder full-FT signal-level
  cached/memory dataset wrapper、stream id、teacher logits shape、weighted
  source/target/adversarial stream loader 和 no-active-stream 错误分支；
  `util/tests/test_signal_cache.py` 覆盖 full-FT signal cache 的 metadata
  schema、label shape 校验、cache shape 计算和 `.npy`/`.npz` roundtrip；
  `util/tests/test_training_torch_utils.py` 覆盖包内 module freeze/unfreeze
  helper 对所有参数 `requires_grad` 的切换。

### Phase 3：统一 metrics exporter

产物：

- `metrics_long.csv`
- `artifact_manifest.json`
- paper table exporter

验收：

- 从现有 JSON/CSV 回填，不需要重跑实验；
- exporter 拒绝混用 v3/v5/v6；
- all-zero kept 和 drop-all-zero 是两个明确 view；
- `metrics_long.csv` 同时保留 raw `view` 和 `canonical_view`，使
  ECGFounder `target_*_refexcluded` 单中心结果能与 PN2021 ref-excluded
  结果进入同一 paper table 口径；
- paper table exporter 按 canonical view 过滤和计算 delta，同时输出
  `raw_views` 供审计。

当前已落地状态：

- exporter 已写入 `canonical_view`；
- paper table exporter 已按 canonical view 汇总，并保留 `raw_views`；
- exporter 已支持显式 method-level `--run-id`，并可用
  `--filter-to-target-center` 从每个 target-center artifact 只保留该目标中心
  的指标，避免把 cross-center eval JSON 中的非目标中心结果错误并入方法均值；
- 新增 `scripts/merge_metrics_long.py`，可把多个方法级 `metrics_long.csv`
  合并为一个 paper-table 输入，同时拒绝混用 mapping 和重复 metric key；
- 已将 2026-05-25 v6 结果回填到：
  `/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/metrics_export/refactor_v6_20260528/`；
- 已生成三个表格视图：
  `paper_table_all_zero_kept_no_delta/`、`paper_table_effnet_all_zero_kept_delta/`
  和 `paper_table_drop_all_zero/`；
- `configs/experiments/pn2021_eval_v6_refexcluded.yaml` 已声明 managed
  postprocess：自动从四个 ref-excluded eval JSON 导出 `metrics_long.csv`，
  再导出 all-zero-kept 与 drop-all-zero 两个 paper table；
- 其余 5 个 active mainline config 也已声明 managed postprocess，未来
  训练/eval execute 完成后可自动生成方法级 `metrics_long.csv` 和对应
  paper table；
- postprocess write-plan smoke 已通过：
  `/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/dry_runs/refactor_postprocess_plan_smoke_20260528/`；
- postprocess integrated execute 已通过：
  `refactor_execute_postprocess_pn2021_eval_20260528`。同一个 launcher run
  完成四个 ref-excluded PN2021 eval、`metrics_long.csv` 导出、all-zero-kept
  paper table 和 drop-all-zero paper table，并由
  `run_manifest.json` 校验 `n_checked=16`、6 个 artifacts 标记
  `postprocess=true` 且存在、`missing=0`、`mapping_errors=0`；
- 该 integrated execute 的自动导出表格 center mean：
  all-zero-kept AUROC/AUPRC `0.854321 / 0.487896`，
  drop-all-zero AUROC/AUPRC `0.881637 / 0.634631`；
- 最新完整 CPU-only refactor 回归为 `229 passed`；同时已通过
  `compileall`、生产包禁止 `scripts.*` 反向导入检查、`git diff --check`
  和 6 个 active config 的 YAML launcher `--write-plan`；
- 最新 all-mainline postprocess 审计输出：
  `/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/config_audit/refactor_all_mainline_postprocess_verify_20260528/`，
  `managed_experiment_count=6`、`passed_count=6`、`failed_count=0`；
- ECGFounder direct training smoke 修复后再次运行 active-config 审计，输出：
  `/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/config_audit/refactor_post_training_smoke_verify_20260528/`，
  仍为 `managed_experiment_count=6`、`passed_count=6`、`failed_count=0`；
- EfficientNet direct smoke 和文档固化后再次运行 active-config 审计，输出：
  `/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/config_audit/refactor_after_effnet_smoke_docs_20260528/`，
  仍为 `managed_experiment_count=6`、`passed_count=6`、`failed_count=0`；
- VAE-LHAT smoke 后再次运行 active-config 审计，输出：
  `/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/config_audit/refactor_after_effnet_vae_lhat_smoke_20260528/`，
  仍为 `managed_experiment_count=6`、`passed_count=6`、`failed_count=0`；
- 新增 `configs/experiments/effnet_direct_k500_v6_smoke.yaml`，作为
  EfficientNet1DV2 direct-K500 的真实训练 smoke 配置。该配置保留 v6
  mapping、K500 seed20260531、四中心 ref-exclusion 和 managed postprocess，
  但限制为 1 epoch、`num_workers=0`、嵌套 PN2021 eval
  `--eval_pn2021_limit 64` / `--eval_min_pos 2`，用于验证真实 GPU launcher
  训练链路而非论文结果。
- EfficientNet direct smoke 已在 GPU 0 通过：
  `refactor_smoke_effnet_direct_limit64_ep1_20260528`，
  `status=succeeded`、`input_verification.passed=true`、
  `artifact_verification.passed=true`、`n_checked=32`、`missing=0`、
  `mapping_errors=0`、6 个 postprocess artifacts 通过校验；
- 新增 `configs/experiments/effnet_vae_lhat_k500_v6_smoke.yaml`，作为
  EfficientNet1DV2 VAE-LHAT 主线 wrapper 的真实训练 smoke 配置。该配置保留
  v6 mapping、K500 seed20260531、四中心 ref-exclusion、K500 internal
  validation 和 managed postprocess，但限制为 1 epoch、`num_workers=0`、
  `k_anchor=32`、`hull_M=4`、`hull_steps=1`、`pgd_batch=8`、latent AugMix
  `width=2/depth=1/severity=1`、嵌套 PN2021 eval
  `--eval_pn2021_limit 64` / `--eval_min_pos 2`。
- VAE-LHAT smoke 首次运行暴露了一个有效配置约束：latent AugMix
  `width=1` 非法，因为第 0 条分支保留给 VAE adversarial sample；smoke
  配置已改为最小合法 `width=2`。
- VAE-LHAT smoke 已在 GPU 0 通过：
  `refactor_smoke_effnet_vae_lhat_width2_limit64_ep1_20260528`，
  `status=succeeded`、`input_verification.passed=true`、
  `artifact_verification.passed=true`、`n_checked=32`、`missing=0`、
  `mapping_errors=0`、6 个 postprocess artifacts 通过校验。该 run 真实覆盖
  ECGTwin VAE decode、latent-hull PGD、latent AugMix、target K500 internal
  validation、最终 ref-excluded PN2021 eval 和 managed metrics/table export；
- 最新 all-mainline dry-run/write-plan 输出：
  `/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/dry_runs/refactor_all_postprocess_plan_*_20260528/`，
  6 个 manifest 均为 `status=dry_run`，均包含 non-empty
  `postprocess_commands`，且 `command.sh` 均有 `# Managed postprocess commands`
  段落。

### Phase 4：逐步抽库包

在 wrapper 稳定后，再把复用逻辑迁出 `scripts/`：

```text
ecg_adv_gen.labels
ecg_adv_gen.ecg
ecg_adv_gen.data
ecg_adv_gen.models
ecg_adv_gen.training
ecg_adv_gen.adaptation
ecg_adv_gen.reporting
```

验收：

- 生产库不再 import `scripts.*`；
- 旧脚本只做兼容 wrapper；
- CPU/import-only smoke test 通过。

当前已落地状态：

- 已新增 `ecg_adv_gen.labels.super5`，作为 config/reporting 层使用的
  Super5 metadata facade，并新增 canonical `label_mapping.pn2021_super5`
  artifact payload helper；
- `ecg_adv_gen.config.loader` 和 `ecg_adv_gen.reporting.metrics_export` 已改为
  使用该 facade，不再直接 import `scripts.triple_labels.label_schemes`；
- 已新增 `ecg_adv_gen.data.contracts`，集中记录当前 PTB-XL -> PN2021 主线的
  target/eval centers、leak-excluded centers、100Hz x 1000 classifier input、
  PTB-XL lead order、ECGTwin 1024 -> 1000 decode/reorder 协议；
- `ecg_adv_gen.config.loader` 已对 YAML 的 data/preprocess 字段执行 contract
  校验，防止中心列表、采样率、长度和导联顺序无意漂移；
- 已新增 `ecg_adv_gen.data.manifest` 和 `scripts/export_data_manifest.py`，
  可生成轻量 `data_manifest.json`，记录 PTB-XL/PN2021 roots、cache dirs、
  PN2021 center dirs 和 shared-server write boundary 状态，不加载波形、不递归扫描大数据；
  `--include-counts` 可额外只统计每个 PN2021 center 目录及其下一层分组目录的
  `*.hea` 数量，作为中心样本规模快照；
- 已新增 `ecg_adv_gen.data.pn2021_index` 和 `ecg_adv_gen.data.kshot`，
  先抽出 PN2021 header/SNOMED 解析、record id basename、leak-center guard、
  ref/include meta loader、selected ref-id JSON variant parser、K-shot artifact
  schema check、primary-label proportional K-shot selection 等纯元数据逻辑，
  不加载波形、不写 cache；
- `scripts/run_experiment.py --write-plan` 现在会随 resolved config 一起写出
  `data_manifest.json`，作为每次 managed run 的 launch artifact；
- 已新增 `ecg_adv_gen.preprocessing.signals`，提供纯 NumPy 的 waveform axis
  inference、ECGTwin -> PTB-XL lead reorder、线性重采样、以及 ECGTwin decoded
  到 classifier channel-last `(1000, 12)` 的转换 helper；
- 已新增 `ecg_adv_gen.evaluation.views`，集中管理 PN2021 all-zero-kept、
  drop-all-zero、ECGFounder `target_*_refexcluded` raw view 到 canonical view
  的映射，以及 macro metric aliases；
- 已新增 `ecg_adv_gen.evaluation.pn2021_metric_views`，先抽出
  labels/scores/centers/record_ids 已存在之后的 PN2021 ref-excluded、
  all-zero-kept、drop-all-zero 纯 NumPy 指标视图组装逻辑，不运行 inference、
  不读取波形、不构建 cache；
- 已新增 `ecg_adv_gen.adaptation.lhat`，先抽出 VAE-LHAT online AT 中
  K500 internal validation mask、source/class weight parser、anchor quota、
  K-shot inverse-frequency class weight、anchor-preserving soft label、
  class-trust、global z-score、Dirichlet branch cap、no-revisit stratified
  pool walker、same-label latent candidate index、以及 array-level latent
  AugMix mixer、anchor sampling mode 解析和 adversarial stream linear warmup
  等无 GPU/无模型状态纯逻辑；
  `synth_online_at_super5.py` 保留训练 loop/PGD/ECGTwin decode/DataLoader，
  只改为导入这些 helper，并保留旧模块命名空间 re-export 兼容已有 dated scripts；
- 已新增 `ecg_adv_gen.models.ecgfounder`，先抽出 ECGFounder linear-probe
  head/features、direct K-shot head run dir、VAE-LHAT run dir 和 K500 base-head
  lookup 等无 GPU/无模型 import 的路径契约；`config.loader` 的 ECGFounder
  manifest trace 已改用这些 helper，减少路径模式散落；
- 已新增 `ecg_adv_gen.models.ecgfounder_heads`，抽出 ECGFounder VAE-LHAT 的
  `ResidualAdapterHead`、`FeatureAdapterHead`、linear head unwrap/clone 逻辑；
  zero-init adapter 在初始时严格等价于 base head，`clone_linear_head` 会冻结
  direct/base head 副本用于固定 anchor scoring；现在也抽出 ECGFounder
  full-FT 从 frozen-feature Super5 linear head checkpoint 初始化 `model.dense`
  的 state-dict 形状校验和复制逻辑；
- 已新增 `ecg_adv_gen.models.ecgfounder_inference`，抽出 cached-feature
  ECGFounder head batch prediction、clipped sigmoid、PTB-XL fold feature-head
  eval、PN2021 feature-head eval、full signal-model prediction / indexed split
  eval 和 generic feature-head metric callback；
  包内不直接导入 legacy metrics，而是由 runner 显式传入 metric/view 函数；
- 已新增 `ecg_adv_gen.models.ecgfounder_torch`，抽出 ECGFounder full-FT
  的 100Hz channel-time ECG -> 500Hz ECGFounder input conversion 和
  per-sample global z-score；该 torch helper 不挂到
  `ecg_adv_gen.models.__init__`，避免 config/audit 轻量路径导入时加载 torch；
- 已新增 `ecg_adv_gen.adaptation.anchor_sampling`，抽出 ECGFounder VAE-LHAT
  hard/uncertain anchor difficulty weights、归一化、seeded hard-anchor sampling
  和 ESS/weight 统计；该 torch helper 不挂到 `ecg_adv_gen.adaptation.__init__`
  上，避免普通 YAML/config audit 导入时加载 torch；现在也抽出 ECGFounder
  full-FT 的 anchor class-weight 解析、带 repeat cap 的 weighted class quota
  和 weighted index sampling；
- 已新增 `ecg_adv_gen.training.stream_sampling`，抽出 ECGFounder VAE-LHAT
  source/target-real/adversarial 三流 feature DataLoader 组装、multi-label
  row sample weight、class-weight multiplier 和 stream id 追踪逻辑；
- 已新增 `ecg_adv_gen.training.signal_streams`，抽出 ECGFounder full-FT
  signal-level cached/memory dataset wrapper、source/target/adversarial stream
  id、teacher logits 和 weighted DataLoader 组装；full-FT runner 只保留
  source fold/limit 选择和薄兼容 wrapper；
- 已新增 `ecg_adv_gen.adaptation.latent_hull_torch`，抽出 ECGFounder
  VAE-LHAT/full-FT 共享的 `initial_hull_latent` deterministic start 诊断逻辑，
  支持 `optimized`、`one_hot`、`uniform` 和 stochastic mode fallback；
- 已新增 `ecg_adv_gen.data.real_anchors`，抽出 ECGFounder VAE-LHAT/full-FT
  共用的 real-anchor latent base 查找、PN2021 cache record-id/center 对齐、
  legacy min-one primary-class K-shot anchor selection，以及 explicit
  selected-id anchor pool loading；VAE-LHAT runner 保留兼容 wrapper，full-FT
  不再为了找 anchor 反向 import VAE-LHAT 脚本；
- 已新增 `ecg_adv_gen.evaluation.target_splits`，抽出 ECGFounder full-FT
  的目标中心 K-shot train/val split，保留 legacy `split_seed+1701` random
  split 和 rarity/coverage-biased stratified split，用于 K500 internal
  validation；
- `ecg_adv_gen.evaluation.selection` 现在还抽出 ECGFounder full-FT
  `source_auprc` / `target_val_auprc` /
  `source_plus_target_val_auprc` 三种 selection score 计算，runner 不再内联
  该选择逻辑；
- 已新增 `ecg_adv_gen.training.losses`，先抽出 masked BCE-with-logits、
  per-class clipped `pos_weight`、普通 multi-label per-sample BCE、
  masked per-sample BCE、stream-weighted masked BCE、ECGFounder VAE-LHAT
  attack-success diagnostics / merge 逻辑和 selected-class pairwise ranking
  loss；`train_ptbxl.py` 继续作为 legacy 兼容出口，
  ECGFounder linear-probe、K-shot head、full fine-tune、VAE-LHAT，以及
  EfficientNet direct-K500 / legacy VAE-LHAT 编排脚本已改为直接从包内 helper
  导入；
- 已新增 `ecg_adv_gen.training.torch_utils`，抽出 full-FT 中 freeze/unfreeze
  model parameters 的 `set_module_requires_grad` helper；full-FT runner 不再
  保留本地 `set_requires_grad`；
- 已新增 `ecg_adv_gen.training.splits`，抽出 direct EfficientNet K500 和
  ECGFounder K-shot head fine-tune 共用的 deterministic random train/val split
  helper；两个 legacy runner 只保留薄兼容 wrapper，并分别保留
  zero-val identity 与 zero-val permuted 的历史行为；
- 已完成一次真实单 GPU cached-feature smoke：
  `ecgfounder_direct_k500_v6`，`CUDA_VISIBLE_DEVICES=0`，
  `training.epochs=2`、`batch_size=256`、`eval_batch_size=8192`；
  manifest 位于
  `/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/launcher_smoke/refactor_smoke_ecgfounder_direct_ep2_20260528/run_manifest.json`，
  结果为 `status=succeeded`、`artifact_verification.passed=true`、
  `missing=0`、`mapping_errors=0`、18 个 artifact 已检查；
- 第一次 ep1 smoke 暴露出 ECGFounder direct K500 legacy 结果缺少
  `label_mapping.pn2021_super5`，launcher 正确拒绝；随后已用 package
  mapping payload helper 修复并由 ep2 smoke 验证通过；
- 完整标签转换逻辑暂不移动，仍由 `scripts/triple_labels/label_schemes.py`
  负责，避免在 data/preprocessing 尚未抽出前破坏历史实验入口；
- 新增 CPU-only 测试校验 labels facade 与 legacy label scheme 同步，并校验
  data/preprocess contract、tracked YAML、training loss helper 与 legacy
  兼容出口同步。

## 10. 必须防止的错误

1. 不要把本机真实 `configs/local/*.yaml` 提交到 Git。
2. 不要把 GPU id 写死进 tracked experiment YAML。
3. 不要让 tracked YAML 依赖 `/home/linbinhao` 或 `/root` 绝对路径。
4. 不要让 final selector 使用 held-out target labels。
5. 不要把 v3/v5/v6 mapping 的结果放在同一个表里平均。
6. 不要只保存原始 YAML，不保存 resolved config。
7. 不要把 YAML 做成几千行巨型配置；方法默认值应拆到 `configs/defaults/`。
8. 不要为了重构立即移动大脚本；先 wrapper、manifest、metrics，再拆代码。

## 11. 与总重构计划的关系

`docs/tmp_html/project_refactor_plan_20260527.html` 提出的总路线是：

```text
先建立事实源、配置 schema、manifest 和 metrics 统一层，
再迁移脚本和抽库包。
```

YAML 配置管理正好是这个路线的第一块地基。

最小可追溯重构应先完成：

1. YAML schema；
2. local path boundary；
3. resolved config；
4. run manifest；
5. metrics_long；
6. active script index；
7. source-of-truth 文档。

完成这一步后，再开始拆 `synth_online_at_super5.py`、ECGFounder runner、DeepECG adapter 等具体代码，风险会低很多。

## 12. 推荐下一步

建议下一轮直接落地 Phase 0 + Phase 1 的骨架：

- 新建 `configs/README.md`；
- 新建 `configs/local/.gitignore`；
- 新建 `configs/local/linbinhao_server.example.yaml`；
- 新建 `configs/experiments/effnet_direct_k500_v6.yaml`；
- 新建 `configs/experiments/effnet_vae_lhat_k500_v6.yaml`；
- 新建一个只做 dry-run 的 `scripts/run_experiment.py`。

第一版 `run_experiment.py` 不需要真的训练，只要能解析 YAML、校验路径、输出旧脚本命令和 resolved config，就已经能显著降低后续实验出错概率。

## 13. 2026-05-28 当前推进状态

Phase 0/1 已经不再是“下一步”，而是已经落地为可审计 wrapper 层：

- `configs/`、schema、defaults、active experiment YAML、local example 已建立；
- `scripts/run_experiment.py` 已支持 resolved config、manifest、
  `command.sh`、`data_manifest.json`、`k500_ref_ids.json`、`selection.json`、
  input verification、artifact verification、以及受限 `--set` override；
- `scripts/audit_managed_configs.py` 可以 CPU-only 审计全部
  `active_wrapped` configs；
- `scripts/export_metrics_long.py`、`scripts/merge_metrics_long.py`、
  `scripts/export_paper_table.py` 已形成 metrics/table 统一层；
- `ecg_adv_gen` 包内已抽出 labels/data/preprocessing/evaluation/adaptation/
	  models/training 的第一批纯逻辑和路径契约，包括 training loss、full-FT
	  adversarial diagnostics、legacy run naming、feature stream sampling、
	  signal stream loading、signal cache metadata/schema 和 deterministic split
	  helper；EfficientNet direct-K500、EfficientNet VAE-LHAT/AugMix、
	  ECGFounder direct-head、ECGFounder VAE-LHAT 和 ECGFounder full-FT 的
	  legacy runner/manifest child-run 路径已复用同一批轻量 run/path contract，
	  且 config loader import 不加载 torch。

最新 CPU-only 验证：

```text
refactor-focused pytest: 229 passed
compileall: passed
git diff --check: passed
active managed-config audit: 6/6 passed
config loader import: torch_loaded False
```

最新 active managed-config 审计输出：

```text
/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/config_audit/refactor_phase12_closeout_final_20260528/
```

PN2021 eval wrapper 在 execute smoke 中暴露过两个工程问题，均已修复：

- 长 `TMPDIR` 触发 multiprocessing `AF_UNIX path too long`，现在 managed
  runtime env 使用 `/home/linbinhao/tmp_ecg`；
- 结果 JSON 写入时输出父目录不存在，现在 launcher 会提前准备 expected
  artifact parent dirs，`eval_crosscenter.py` 自身也会创建输出父目录。

PN2021 eval wrapper 已完成三层 execute 验证。轻量 smoke 先证明短
`TMPDIR` 和输出目录修复在真实 child runtime 下成立：

```text
config=configs/experiments/pn2021_eval_v6_refexcluded_smoke.yaml
run_id=refactor_smoke_pn2021_eval_runscoped_limit64_20260528
result=status=succeeded, artifact_verification.passed=true
n_checked=10, missing=0, mapping_errors=0, content_errors=0
eval_count=4, all_eval_artifacts_under_run_id=true
```

随后正式 `pn2021_eval_v6_refexcluded` 全量 execute 和 integrated
postprocess execute 也已成功：

```text
run_id=refactor_full_pn2021_eval_runscoped_20260528
result=status=succeeded, artifact_verification.passed=true, missing=0, mapping_errors=0

run_id=refactor_execute_postprocess_pn2021_eval_20260528
result=status=succeeded, artifact_verification.passed=true, n_checked=16,
       postprocess_artifacts_verified=6, missing=0, mapping_errors=0
```

轻量 smoke 只证明工程链路；全量 PN2021 execute 的指标可作为 wrapper
重构等价性证据，但不是新的方法训练结果。

同时，managed configs 已加入 `runtime.run_id` 运行时插值，child 输出路径改为：

```text
${paths.output_root}/${experiment.name}/${runtime.run_id}/...
```

这样同一个实验配置可以安全重复 dry-run、smoke、full-run，不会默认覆盖旧的
child artifact。

## 14. 2026-05-28 ECGFounder Direct Training Wrapper Smoke

为了验证“非 PN2021 eval”的真实训练 wrapper，已跑
`ecgfounder_direct_k500_v6` cached-feature direct-head 小型 execute smoke：

```text
config=configs/experiments/ecgfounder_direct_k500_v6.yaml
run_id=refactor_smoke_ecgfounder_direct_postprocess_ep2_20260528
CUDA_VISIBLE_DEVICES=0
--set training.epochs=2
--set training.batch_size=256
--set training.eval_batch_size=8192
```

第一次 execute 证明 child training 和 artifact verification 已通过，但
managed postprocess 暴露了 exporter 兼容性 bug：

- `export_metrics_long.py` 递归扫描 run dir 时收集了 `training_log.json`；
- ECGFounder direct 的 `training_log.json` 根节点是 list，不是指标对象；
- ECGFounder direct 的 `eval_result.json` 使用 `target_view` 结构，旧 exporter
  没有解析出 metric rows。

已修复：

- directory scan 不再把 `training_log.json` 当作 metrics artifact；
- exporter 增加 ECGFounder direct `target_view` 结构解析；
- 新增 CPU-only 回归测试覆盖 run dir + `training_log.json` list +
  `target_view` 的场景。

修复后用同一 run-id `--resume` 重跑 integrated launcher，结果：

```text
status=succeeded
input_verification.passed=true, n_checked=10
artifact_verification.passed=true, n_checked=22
missing=0, mapping_errors=0, content_errors=0
postprocess_artifacts_verified=4
metrics_long.csv rows=42
paper_table center mean AUROC/AUPRC=0.8632266722 / 0.5105771844
```

这说明当前 wrapper 层已经能覆盖：真实 child training、已有 artifact resume、
child artifact verification、mapping metadata verification、metrics export 和
paper table export。它是 2 epoch 工程 smoke，不作为论文性能结果。

## 15. 2026-05-28 ECGFounder / EfficientNet Training Smoke 覆盖面

截至当前，所有 active training wrapper 都至少有短程 execute smoke 证据：

- `effnet_direct_k500_v6_smoke`：
  `refactor_smoke_effnet_direct_limit64_ep1_20260528`，1 epoch，limited
  PN2021 eval，manifest `succeeded`，artifact verification 通过；
- `effnet_vae_lhat_k500_v6_smoke`：
  `refactor_smoke_effnet_vae_lhat_width2_limit64_ep1_20260528`，四中心
  VAE-LHAT + latent AugMix，manifest `succeeded`，metrics_long 58 行；
- `ecgfounder_direct_k500_v6`：
  `refactor_smoke_ecgfounder_direct_postprocess_ep2_20260528`，cached-feature
  direct-head 2 epoch，manifest `succeeded`，metrics_long 42 行；
- `ecgfounder_vae_lhat_k500_v6_smoke`：
  `refactor_smoke_ecgfounder_vae_lhat_limit64src_ep1_20260528`，四中心
  ECGFounder VAE-LHAT，manifest `succeeded`，metrics_long 84 行，
  all-zero-kept `0.9045766643 / 0.5976623929`，drop-all-zero
  `0.9221097550 / 0.7048084381`；
- `ecgfounder_inithead_fullft_k500_v6_smoke`：
  `refactor_smoke_ecgfounder_inithead_cacheunion_ep1_20260528`，四中心
  ECGFounder init-head full fine-tune，复用 signal-cache union，manifest
  `succeeded`，metrics_long 126 行，all-zero-kept
  `0.9092978816 / 0.6049917842`，drop-all-zero
  `0.9256955173 / 0.7112885361`。

这一步说明 wrapper/YAML/manifest/reporting 的执行链路已经能覆盖当前论文主线
方法形态。它仍不是“完整重构完成”：legacy training loop、WFDB/cache loading、
ECGFounder/DeepECG adapter、以及完整 full-length 多 seed 训练仍留在后续阶段。

当前最新 audit 输出：

```text
/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/config_audit/refactor_phase12_closeout_final_20260528/
managed_experiment_count=6
passed_count=6
failed_count=0
```

## 16. 2026-05-28 一小时收口结论

当前重构已经收口为可交接的 Phase 1/2 状态。短版 handoff 见：

```text
docs/pipelines/refactor_phase12_handoff_20260528.md
```

这次收口确认：

- YAML wrapper、config audit、manifest、run-id-scoped output、postprocess
  reporting、metrics/table export 已经成为当前实验控制层；
- `ecg_adv_gen` 包已经承接第一批稳定 helper 和合同：label metadata、data
  contract、K-shot/ref-id、real-anchor、signal cache、preprocessing shape、
  evaluation view/metric、selection、Latent-Hull helper、ECGFounder path/head/
  inference helper、training loss/stream/split/run naming；
- active EfficientNet 和 ECGFounder 主线可以继续通过 YAML wrapper 跑，不再
  依赖手写大段 bash 参数作为事实源；
- legacy runner 仍然是执行层，不在本轮继续移动 WFDB/cache/runtime、训练 loop
  或外部模型 adapter；
- 当前最后证据为 `229 passed`、`compileall`、`git diff --check`、
  config loader `torch_loaded False` 和
  `refactor_phase12_closeout_final_20260528` active-config audit 6/6 passed。

后续如果继续重构，应作为下一阶段单独启动，而不是在本轮收口里继续扩大范围。
