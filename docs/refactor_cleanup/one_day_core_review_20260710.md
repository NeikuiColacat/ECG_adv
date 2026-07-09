# ECG_rebuild 一天核心代码分层审核包

## 结论与口径

当前 `configs/active_scripts.yaml:latest_mainline` 的 10 个 stage 最终收敛到
6 类运行入口：EfficientNet direct K500、EfficientNet VAE-LHAT、ECGFounder
full-FT、PN2021 clean、EfficientNet PN2021-C、ECGFounder PN2021-C。

为了让一名熟悉 Python/PyTorch 的审核者在一个工作日内完成可信的核心首轮审核，
本包采用两层口径：Tier 1 对高风险机制逐行深审，Tier 2 依靠 golden、typed
adapter 和 invariant tests 做定向核验。发现问题后必须开 follow-up，不能把本包
当作整个仓库的完整形式化证明。

- 核心 Python 文件：**15 个**。
- 15 个文件 whole-file `wc -l`：**11,410 LOC**。
- 相对 Goal 起点 `fc57662` 的同一组 15 文件：**11,609 → 11,410 LOC，净减 199 LOC**。
- Tier 1 逐行深审：**3,008 LOC**。
- Tier 2 guided invariant review：**2,877 LOC**。
- 当天分层覆盖合计：**5,885 LOC**；只有 Tier 1 可称为深审。
- LOC 包含空行和注释，区间首尾均计入；不是语句数或逻辑 LOC。
- YAML、golden contract、typed adapters 和 tests 单列附录；它们是 Tier 2 的证据，
  但不重复计入 5,885 LOC。
- `ecg_adv_gen/evidence/**`、`ecg_adv_gen/reporting/**` 和导出脚本不属于算法核心。
- 不包含外部 `model/*`、历史分支、stabilizer、raw-supervised、oracle 或 selector。

快照：

```text
repo:       /home/linbinhao/ECG_rebuild
branch:     refactor/data-module-20260620
HEAD:       6e4e4cf13810f1ce986b5d4475a1bd7fe432019a
snapshot:   2026-07-10 06:50:27 CST
golden:     sha256=df5d6597167e3eb27b63054ed0e62e823b43c5a48f037c048e8bb6c307b404e1
worktree:   dirty; LOC/line ranges count the current filesystem, not HEAD-only files
GPU:        not used
```

工作树可能被并发任务继续修改。复审前应先执行文末复算命令；若行号漂移，以符号名
为准，并重新计算区间而不是沿用旧行号。

## latest_mainline 的真实调用链

Golden contract 中的 10 个 stage 均经同一个 launcher 和 typed adapter registry：

```text
configs/active_scripts.yaml:latest_mainline (10 stages)
  -> scripts/run_experiment.py
  -> ecg_adv_gen.config.loader
       load_experiment_config
       validate_experiment_config
       build_runner_commands
  -> ecg_adv_gen.config.adapters.registry
       direct_finetune
         -> runner/effnet_direct_finetune.py
       pn2021_eval
         -> runner/pn2021_clean_eval.py
       effnet_vae_lhat
         -> runner/effnet_vae_lhat_augmix.py       [CLI/编排桥]
            -> runner/effnet_vae_lhat.py           [child command contract]
               -> runner/synth_online_at_super5.py [实际 EffNet 在线训练]
               -> runner/pn2021_clean_eval.py      [训练后 clean eval]
       ecgfounder_fullft
         -> runner/ecgfounder_fullft.py
       pn2021c_eval
         -> runner/pn2021c_eval.py
       ecgfounder_pn2021c_eval
         -> runner/ecgfounder_pn2021c_eval.py
```

`runner/effnet_vae_lhat_augmix.py` 是真实调用链的一部分，但其主体是参数声明和
子进程编排，所以放在 adapter 附录做机械核对。核心包选择
`runner/effnet_vae_lhat.py`，因为它集中表达 wrapper 到训练/评测 child command
的精确参数合同。

## 15 文件、5,885 LOC 分层覆盖清单

Tier 1 是逐行深审；Tier 2 只沿 golden/adapter/test 指定的 invariant 做定向阅读，
不把 Tier 2 描述为逐行审计。两个 Tier 的行区间不重叠。

| 顺序 | 文件 | whole LOC | Tier 1 深审区间（LOC） | Tier 2 guided 区间（LOC） | 核心问题 |
|---:|---|---:|---|---|---|
| 1 | `scripts/run_experiment.py` | 196 | — | `1-196`（196） | launcher 的 config、audit、dry-run/execute、run finalize 顺序是否与 golden 一致。 |
| 2 | `ecg_adv_gen/config/loader.py` | 1,377 | — | load/validate `374-476`；matrix/runner command `479-552`（177） | mapping、K500、selection 和 runner 参数是否无损到达 typed adapter。 |
| 3 | `ecg_adv_gen/preprocessing/classifier.py` | 218 | lead/resample/z-score/preprocess `63-189`（127） | — | lead order、filter、100 Hz/1000 点、z-score、pad/crop 顺序是否唯一。 |
| 4 | `ecg_adv_gen/data/pn2021.py` | 451 | — | ref exclusion `13-16`；metadata `58-90`；load/build `93-177`；clean `180-230`；raw/native `233-451`（392） | cache identity、完整 center cache 后排除 ref、raw/native-raw 顺序是否被 invariant tests 锁定。 |
| 5 | `ecg_adv_gen/data/waveform_datasets.py` | 622 | raw/native clean+corrupt `201-412`；ECGFounder bottleneck5000 `494-599`（318） | `PN2021IndexedCenterDataset` `90-123`（34） | corruption、resample、z-score、crop 与两种 backbone 输入是否匹配。 |
| 6 | `ecg_adv_gen/runner/effnet_direct_finetune.py` | 462 | — | dataset/model `61-121`；`train_one` `145-354`；summary `357-392`；`main` `449-458`（317） | direct K500 subset/split/checkpoint 是否构成公平 matched baseline。 |
| 7 | `ecg_adv_gen/runner/effnet_vae_lhat.py` | 296 | — | `1-296` whole file（296） | K500 anchors、last checkpoint、ref exclusion、三链参数和输出路径是否完整转发。 |
| 8 | `ecg_adv_gen/adaptation/lhat.py` | 922 | neighbor/soft labels `440-576`；capped Dirichlet `610-628`；three-chain builder `631-922`（448） | — | NORM/abnormal、anchor dominance、multi-hot/soft label 与三链语义是否正确。 |
| 9 | `methods/augmix/ecg_ops.py` | 212 | 5 个 ECG corruption ops `29-212`（184） | — | 幅度、采样率、lead masking、shape/dtype/finite 和 severity 是否符合 locked 协议。 |
| 10 | `ecg_adv_gen/runner/synth_online_at_super5.py` | 2,147 | consistency `350-495`；diagnostics/PGD/views `544-764`；完整 epoch/payload/checkpoint `1581-2143`（930） | — | VAE adversarial chain 是否真正训练；`1941-2055` 的 ASR、`atk_anchor`、clean/adv BCE、`loss_gain`、invalid 与 epoch payload 是否同源。 |
| 11 | `ecg_adv_gen/runner/ecgfounder_fullft.py` | 2,149 | locked three-chain `317-377`；clean/adv builders `824-1102`；setup `1472-1629`；完整 train/metric/final JSON `1643-2145`（1,001） | — | full trainability、K500 IDs、stream/loss、`1870-1931` epoch metrics 与 `2050-2145` final topology/drop-all-zero JSON 是否一致。 |
| 12 | `ecg_adv_gen/runner/pn2021_clean_eval.py` | 529 | — | include/exclude `67-124`；eval `127-274`；source floor `370-394`；main/output `401-525`（356） | ref-excluded clean 是否明确分离 kept/drop-all-zero，并携带 mapping/source floor。 |
| 13 | `ecg_adv_gen/evaluation/pn2021c.py` | 382 | — | profile resolve、sequence、op build/apply `220-382`（163） | S5/depth23、deterministic seed、operator order 和 native sample rate 是否被测试锁定。 |
| 14 | `ecg_adv_gen/runner/pn2021c_eval.py` | 702 | — | clean metrics `139-206`；`eval_one` `209-435`；protocol/execution `584-698`（410） | EffNet clean/corrupt、ref count/hash、raw-first 和 metadata compatibility 是否 fail-closed。 |
| 15 | `ecg_adv_gen/runner/ecgfounder_pn2021c_eval.py` | 745 | — | exclusion/frontend `120-151`；clean `154-287`；corrupt `290-539`；setup/execution `622-741`（536） | 100→500 Hz/5000 点路径，以及 clean/corrupt 的 z-score、ref、checkpoint、view 是否一致。 |
| **合计** | **15 files** | **11,410** | **3,008** | **2,877** | **当天分层覆盖 5,885 LOC** |

## 一天执行顺序、时间盒与输出

### 09:00–09:20：preflight 与证据钉住

- 运行 golden `--check`，记录 SHA；核对 HEAD、dirty 状态和 15 个路径。
- 输出：一页 `stage -> adapter -> entrypoint -> child runner` 调用链；任何 golden
  stale 或 stage 漂移都直接阻断后续审核。

### 09:20–12:00：Tier 1 数据/机制深审，1,077 LOC

- `preprocessing/classifier.py`：127 LOC。
- `data/waveform_datasets.py`：318 LOC。
- `adaptation/lhat.py`：448 LOC。
- `methods/augmix/ecg_ops.py`：184 LOC。

输出：shape/rate/lead/order 合同表，以及 neighbor、label、hull、corruption 的逐行
结论；每项写 `PASS / finding / follow-up`，不能只记“看过”。

### 13:00–15:45：Tier 1 两套训练 runner 深审，1,931 LOC

- `synth_online_at_super5.py`：930 LOC。
- `ecgfounder_fullft.py`：1,001 LOC。

输出：两套 runner 的 attack→three-chain→loss→metrics→last-checkpoint 对照表。必须
逐项追到 synth `1941-2055` 和 ECGFounder `1870-1931`、`2050-2145` 的 payload；
任一字段无法从生成点追到最终 JSON 就建立 follow-up。

### 15:45–17:30：Tier 2 guided invariant review，2,877 LOC

按正文 Tier 2 区间阅读，但以附录中的 golden、adapter 与测试为索引，不做逐行
深审。优先核对：

- YAML key → golden argv → runner arg → checkpoint/metric；
- K500 ref count/hash、mapping、all-zero view、S5/depth23 和 100→500 Hz；
- direct/三链/ECGFounder 的 init 与 last-checkpoint 匹配。

输出：参数链矩阵和 evaluator 对照表，每条 invariant 附对应测试名或明确的缺口。

### 17:30–18:00：结论与升级

输出按严重度排序的 findings、当天已证实的合同、未证实项和 follow-up 所需文件。
Tier 2 中出现语义分叉、测试覆盖不足或 golden/实现不一致时，升级为下一轮 Tier 1；
不得用“guided 已覆盖”冒充逐行证明。

## 审核必须回答的问题

### 数据合同

- 所有 mainline 输入最终是否严格满足目标 backbone 的 shape/rate/lead order？
- K500 ref IDs 是否只从 evaluation view 排除，而不会破坏完整 center cache？
- corruption 是否发生在 locked protocol 指定的 z-score 前位置？
- ECGFounder clean/corrupt 是否都经相同 100→500 Hz、5000 点和 z-score 路径？

### 方法合同

- 三链是否保持 width=3，VAE-LHAT adversarial waveform 是否确实参与训练？
- 同标签/compatible neighbor 是否避免 NORM 与 abnormal 冲突？
- anchor 是否包含在 hull 中且仍为 anchor-dominant？
- multi-hot/soft labels 是否没有被 argmax 悄然压成单标签？
- `loss_gain`、clean/adv BCE、ASR、invalid decode、source floor 是否来自一致语义？

### 比较与选择合同

- direct、VAE-LHAT 与 ECGFounder 候选是否使用声明的 K500 seed/ref metadata？
- 是否只用锁定的 K500-internal/source-floor 或 last-checkpoint policy，而不读取
  heldout target labels 做选择？
- clean/corrupted 是否使用相同 mapping hash、class order 和明确 all-zero view？
- S5/depth23 是否只改变 corruption composition，没有暗改 preprocessing/frontend？

## 附录 A：golden contract（Tier 2 证据，不重复计入 5,885 LOC）

已核验 digest：

```text
sha256=df5d6597167e3eb27b63054ed0e62e823b43c5a48f037c048e8bb6c307b404e1
```

| 文件 | LOC | 审核作用 |
|---|---:|---|
| `configs/golden/latest_mainline_contract_v1.json` | 1,722 | 10 个 stage 的 host-neutral resolved command、K500 refs、selection、metric views 和 expected artifacts。 |
| `scripts/agent/build_latest_mainline_golden.py` | 200 | 从 active index 和 typed config 重建/校验 golden。 |
| `util/tests/test_latest_mainline_golden_contract.py` | 167 | stage uniqueness、mapping、K500、checkpoint、路径归一化和 stale detection。 |
| **合计** | **2,089** |  |

审核顺序：先执行 generator `--check`，再从 golden 的每个 `runner_adapter` 与
`runner_entrypoint` 回到正文调用链。Golden 证明“配置展开是什么”，不能替代算法
实现审阅或运行时实验复现。

## 附录 B：latest-mainline config 控制面（Tier 2 证据，不重复计入 5,885 LOC）

| 控制面 | 文件数 | LOC |
|---|---:|---:|
| `configs/active_scripts.yaml` | 1 | 455 |
| `latest_mainline.stages` 的 experiment YAML | 10 | 564 |
| 传递继承的 `configs/defaults/*.yaml` | 6 | 352 |
| **合计** | **17** | **1,371** |

只核对：stage 恰好 10 个且 config 唯一、mapping version/hash、class order、四中心、
K/seed/subset seed、selection/checkpoint policy、corruption input、S5/depth23、三链
width/depth/copies。`configs/active_evidence_registry.yaml` 属于证据核验面，不混入
算法 LOC 或本表。

## 附录 C：typed adapters 与薄 wrapper（Tier 2 证据，不重复计入 5,885 LOC）

| 文件 | LOC |
|---|---:|
| `ecg_adv_gen/runner/effnet_vae_lhat_augmix.py` | 306 |
| `ecg_adv_gen/config/adapters/direct_finetune.py` | 121 |
| `ecg_adv_gen/config/adapters/effnet_vae_lhat.py` | 282 |
| `ecg_adv_gen/config/adapters/ecgfounder_fullft.py` | 440 |
| `ecg_adv_gen/config/adapters/pn2021_eval.py` | 138 |
| `ecg_adv_gen/config/adapters/pn2021c_eval.py` | 288 |
| `ecg_adv_gen/config/adapters/ecgfounder_pn2021c_eval.py` | 257 |
| `ecg_adv_gen/config/adapters/registry.py` | 48 |
| **合计** | **1,880** |

机械核对即可：每个 `build_*_argv` 与 `audit_*_command` 成对，adapter 生成的 argv
与 golden 一致，wrapper 只负责 paths、child commands、managed env 和执行顺序。
不要把参数声明重复算作算法 LOC。

## 附录 D：高价值 invariant tests（Tier 2 证据，不重复计入 5,885 LOC）

| 测试文件 | LOC | 主要守护 |
|---|---:|---|
| `util/tests/test_config_loader.py` | 2,762 | 10-stage expansion、typed adapters、three-chain、checkpoint、ref exclusion、dry-run。 |
| `util/tests/test_active_script_index.py` | 279 | latest-mainline/index/managed runner 一致性。 |
| `util/tests/test_latest_mainline_golden_contract.py` | 167 | golden command/manifest 稳定性。 |
| `util/tests/test_preprocessing_contract_behavior.py` | 144 | resample、global z-score、filter 顺序的行为合同。 |
| `util/tests/test_data_contracts.py` | 238 | sample rate、length、lead order、ECGFounder policy。 |
| `util/tests/test_lhat_augmix_ablation.py` | 189 | three-chain base mode、第三链角色与 decoupling。 |
| `util/tests/test_pn2021_corruptions.py` | 176 | official operators、seed、ref hash、clean baseline。 |
| `util/tests/test_pn2021_eval_cache.py` | 232 | cache metadata 与 mmap/NPZ compatibility。 |
| `util/tests/test_pn2021_metric_views.py` | 329 | include/ref exclusion/all-zero views。 |
| `util/tests/test_pn2021c_metadata.py` | 90 | `n_excluded_ref` 和 ref hash mismatch fail-closed。 |
| `util/tests/test_reporting_metric_views.py` | 144 | all-zero-kept/drop-all-zero 导出视图保持分离；仅属输出合同。 |
| `util/tests/test_ecgfounder_pn2021c_evaluator.py` | 453 | full-FT runtime、bottleneck5000、clean/corrupt path。 |
| `util/tests/test_ref_exclusion.py` | 51 | target ref-meta argv 与 order preservation。 |
| `methods/augmix/tests/test_ops.py` | 104 | op shape/dtype/finite/severity 和已删除 conditional mode。 |
| `methods/augmix/tests/test_jsd.py` | 65 | consistency loss 数值与梯度。 |
| `util/tests/test_ecgfounder_preflight.py` | 140 | typed adapter→runner 的 copies/JSD/BCE 跨层约束。 |
| **合计** | **5,563** |  |

按 invariant/测试名抽查，不要求把测试实现再次逐行深审。Reporting/evidence 只在
输出边界核对，不进入 15 文件算法核心。

## 明确排除与高风险 deferred 范围

以下内容不因“一天包未覆盖”而获得删除或修改授权：

- `model/DeepECG`、`model/ECGTwin`、`model/advdiff`、
  `model/ecg_ptbxl_benchmarking`、`model/ecgfounder`：外部模型仓库，只核对公开
  interface、tensor shape 和 checkpoint contract；不得暂存其 dirty 状态。
- 公共/历史 API 的进一步删除、cache/schema 迁移、PN2021/PN2021-C metadata 或
  corruption semantics 裁剪：属于高风险变更，先写 RFC、consumer inventory、迁移
  和 rollback 方案，再单独实施。
- mapping/class order、K500 selection、heldout-label policy、all-zero metric semantics、
  checkpoint selection 的任何改变：会改变论文协议，不能作为代码压缩顺手修改。
- stabilizer frontend、raw-supervised、single-chain shortcut、operator oracle、post-hoc
  selector 和 MIMIC：当前 latest mainline 明确排除或非必需，仅作历史/消融证据。
- `ecg_adv_gen/evidence/**`、`ecg_adv_gen/reporting/**`、registry/run-card/SHA 审计和
  archived reports：属于复现与证据面，必须另审，但不虚增算法核心 LOC。

## 复算与自检命令

复审前在 repo root 执行；均为 CPU-only/read-only（golden `--check` 不重写文件）：

```bash
git rev-parse HEAD
git status --short --branch

wc -l \
  scripts/run_experiment.py \
  ecg_adv_gen/config/loader.py \
  ecg_adv_gen/preprocessing/classifier.py \
  ecg_adv_gen/data/pn2021.py \
  ecg_adv_gen/data/waveform_datasets.py \
  ecg_adv_gen/runner/effnet_direct_finetune.py \
  ecg_adv_gen/runner/effnet_vae_lhat.py \
  ecg_adv_gen/adaptation/lhat.py \
  methods/augmix/ecg_ops.py \
  ecg_adv_gen/runner/synth_online_at_super5.py \
  ecg_adv_gen/runner/ecgfounder_fullft.py \
  ecg_adv_gen/runner/pn2021_clean_eval.py \
  ecg_adv_gen/evaluation/pn2021c.py \
  ecg_adv_gen/runner/pn2021c_eval.py \
  ecg_adv_gen/runner/ecgfounder_pn2021c_eval.py

/home/linbinhao/micromamba/envs/ECGTwin/bin/python \
  scripts/agent/build_latest_mainline_golden.py --check
```

两个 Tier 的机械总和：

```bash
awk 'BEGIN { print 127+318+448+184+930+1001 }'
# Tier 1 expected: 3008

awk 'BEGIN { print 196+177+392+34+317+296+356+163+410+536 }'
# Tier 2 expected: 2877

awk 'BEGIN { print 3008+2877 }'
# combined expected: 5885
```

若修改导致行号漂移，用 AST 重新输出顶层符号边界：

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python - <<'PY'
import ast
from pathlib import Path

for name in [
    "scripts/run_experiment.py",
    "ecg_adv_gen/config/loader.py",
    "ecg_adv_gen/preprocessing/classifier.py",
    "ecg_adv_gen/data/pn2021.py",
    "ecg_adv_gen/data/waveform_datasets.py",
    "ecg_adv_gen/runner/effnet_direct_finetune.py",
    "ecg_adv_gen/runner/effnet_vae_lhat.py",
    "ecg_adv_gen/adaptation/lhat.py",
    "methods/augmix/ecg_ops.py",
    "ecg_adv_gen/runner/synth_online_at_super5.py",
    "ecg_adv_gen/runner/ecgfounder_fullft.py",
    "ecg_adv_gen/runner/pn2021_clean_eval.py",
    "ecg_adv_gen/evaluation/pn2021c.py",
    "ecg_adv_gen/runner/pn2021c_eval.py",
    "ecg_adv_gen/runner/ecgfounder_pn2021c_eval.py",
]:
    tree = ast.parse(Path(name).read_text())
    print(name)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            print(f"  {node.lineno}-{node.end_lineno} {node.name}")
PY
```

本审核包定义审核顺序与边界，不声称替代 CPU tests、managed audit、registry audit、
artifact guard 或实际实验复现。
