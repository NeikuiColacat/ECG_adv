# 另一台 AutoDL 主机同步检查

日期：2026-05-01

## 1. `git pull` 能同步什么

`git pull` 只能同步主仓库中被 git 跟踪的文件，例如：

```text
AGENTS.md
docs/
scripts/
util/
```

当前这轮标签映射修复涉及的 repo 文件：

```text
AGENTS.md
docs/pipelines/dataset_preprocessing_pipeline.md
docs/pipelines/super5_label_mapping_pipeline.md
docs/tmp_md/
scripts/triple_labels/eval_crosscenter.py
scripts/triple_labels/label_schemes.py
scripts/triple_labels/train_ptbxl.py
```

注意：`docs/tmp_md/` 是新增目录，commit/push 之后另一台机器 `git pull` 才会有。

## 2. `git pull` 不会同步什么

以下内容不在主仓库 git 里，另一台机器需要单独准备。

### `/root/autodl-tmp` 大文件

典型必需项：

```text
/root/autodl-tmp/ptbxl/raw100.npy
/root/autodl-tmp/ptbxl/ptbxl_database.csv
/root/autodl-tmp/ptbxl/scp_statements.csv
/root/autodl-tmp/crosscenter_v2/ptbxl_preprocessed.npy
/root/autodl-tmp/physionet2021/training/<center>/
/root/autodl-tmp/triple_labels/super5/best_model.pt
/root/autodl-tmp/triple_labels/super5/training_log.json
/root/autodl-tmp/triple_labels/super5/train_result.json
```

ECGTwin/latent 相关大文件：

```text
/root/autodl-tmp/ECGTwin_Data/Mimic_vae.pt
/root/autodl-tmp/ECGTwin_Data/paired_Mimic_vae_multi_nomic.pt
/root/autodl-tmp/ECGTwin_Data/paired_Mimic_vae_multi_nomic_test.pt
/root/autodl-tmp/ECGTwin_Data/PTBXL_vae_test.pt
/root/autodl-tmp/ptbxl/PTBXL_vae_multi_nomic.pt
```

MIMIC 相关：

```text
/root/autodl-tmp/MIMIC/record_list.csv
/root/autodl-tmp/MIMIC/machine_measurements.csv
/root/autodl-tmp/mimic_tierM/mimic_preprocessed_f16.npy
/root/autodl-tmp/mimic_tierM/mimic_index.npz
```

这些文件很大，不应该直接纳入 git。

### Codex skill

项目 skill 文件在：

```text
/root/.codex/skills/ecg-adv-gen/SKILL.md
```

它不在 repo 里，`git pull` 不会同步。另一台机器如果也要让 Codex 带相同项目记忆，需要手动复制或重新安装/更新该 skill。

## 3. submodule 状态问题

当前 `.gitmodules` 声明了：

```text
model/DeepECG
model/ECGTwin
model/ecg_ptbxl_benchmarking
model/advdiff
```

但是本机检查：

```bash
git ls-files --stage | grep '^160000'
```

没有任何输出。

这说明主仓库索引里没有有效的 gitlink。也就是说，`.gitmodules` 当前像是“孤儿声明”，不是真正生效的 submodule 状态。

当前已采用的实际布局：

```text
model/DeepECG                 -> /root/autodl-tmp/models/DeepECG
model/ECGTwin                 -> /root/autodl-tmp/models/ECGTwin
model/advdiff                 -> /root/autodl-tmp/models/advdiff
model/ecg_ptbxl_benchmarking  -> /root/autodl-tmp/models/ecg_ptbxl_benchmarking
model/ecgfounder              -> /root/autodl-tmp/ecgfounder
```

原因是这些外部仓库可能带有 checkpoint、output、data、benchmark artifacts，
不适合放在系统盘的 repo 工作区里。

本机还存在一些手动 clone 的模型仓库：

```text
model/ECGTwin                 https://github.com/Raiiyf/ECGTwin.git
model/ecg_ptbxl_benchmarking  https://github.com/helme/ecg_ptbxl_benchmarking.git
model/SA-AET                  https://github.com/jiaxiaojunQAQ/SA-AET.git
model/augmix                  https://github.com/google-research/augmix.git
```

但这些目录没有作为主仓库 submodule 被跟踪。另一台机器如果没有这些目录，`git pull` 不会自动创建。

## 4. 另一台机器推荐同步步骤

代码同步：

```bash
cd /root/ECG_adv_Gen
git pull
```

检查模型源码目录是否存在：

```bash
test -f model/DeepECG/notebooks/EfficientNetv2.py && echo OK_DeepECG
test -f model/ECGTwin/config/DiT_ECGTwin.yaml && echo OK_ECGTwin
test -f model/ecg_ptbxl_benchmarking/code/utils/utils.py && echo OK_PTBXL_BENCH
```

如果缺失，需要手动 clone：

```bash
bash scripts/bootstrap_model_repos.sh
```

该脚本会 clone 到：

```text
/root/autodl-tmp/models/
```

然后在 repo 的 `model/` 下建立软链接。

如果只跑当前 super5 EfficientNet/PN2021 评测，最关键的是：

```text
model/DeepECG/notebooks/EfficientNetv2.py
/root/autodl-tmp/ptbxl/
/root/autodl-tmp/crosscenter_v2/ptbxl_preprocessed.npy
/root/autodl-tmp/physionet2021/training/
/root/autodl-tmp/triple_labels/super5/best_model.pt
```

## 5. 当前标签映射修复后的额外注意

PN2021 cache version 已变为：

```text
v3_super5_normsuppress
```

另一台机器如果有旧 cache：

```text
/root/autodl-tmp/triple_labels/pn2021_eval_cache/*v2_normguard.npz
```

不会被新代码复用。新代码会生成：

```text
/root/autodl-tmp/triple_labels/pn2021_eval_cache/<scheme>_<center>_100hz1000_v3_super5_normsuppress.npz
```

因此另一台机器第一次跑 PN2021 eval 会重新构建 cache，需要保证 PhysioNet 2021 原始数据存在。

## 6. 建议后续修复

当前 repo 的 submodule 状态已经按“数据盘外部仓库 + repo 内软链接”方向整理：

1. `.gitignore` 不再整体隐藏 `model/`，允许关键模型入口被 git 看到。
2. `.gitmodules` 增加说明：当前它是外部模型仓库 URL manifest，不是有效 gitlink submodule 状态。
3. 新增 `scripts/bootstrap_model_repos.sh`，用于另一台 AutoDL 复建 `/root/autodl-tmp/models` 和 `model/` 软链接。
4. 对毕业实验，仍建议至少把 `model/DeepECG`、`model/ECGTwin`、`model/ecg_ptbxl_benchmarking` 的确切 commit 写进 `AGENTS.md` 或 `docs/pipelines/`。

不建议现在把 `DeepECG`、`ECGTwin`、`ecg_ptbxl_benchmarking` 直接改成真正 git submodule checkout：

```text
DeepECG 当前目标目录约 2.2G
ECGTwin 当前目标目录约 319M
ecg_ptbxl_benchmarking 当前目标目录约 254M
```

真正 submodule checkout 会把工作树放回系统盘 repo 下，容易和 AutoDL 小系统盘冲突。
