# PN2021 v3 Re-Evaluation Pipeline

本文档定义 PN2021 super5 v3 重新标签映射后，EfficientNet1DV2 模型必须重新执行的 clean 评测流程。

## 背景

当前 PN2021 super5 使用项目自定义语义投影：

```text
SUPER5_PN2021_MAPPING_VERSION = v3_super5_normsuppress_20260501
PN2021_EVAL_CACHE_VERSION     = v3_super5_normsuppress
```

源码：

```text
scripts/triple_labels/label_schemes.py
scripts/triple_labels/eval_crosscenter.py
```

v3 的关键变化是把直接阳性类和 `NORM` suppress-only 异常证据拆开。任何仍引用旧 `eval_result_NORMguard.json`、unversioned `eval_result.json` 或 `v2_normguard` cache 的结果，都不能作为当前主结果。

## 目标

对已有 EfficientNet1DV2 super5 checkpoint 重新跑：

1. PTB-XL fold 10 in-domain AUROC/AUPRC。
2. PN2021 7-center AUROC/AUPRC。
3. 每个 PN2021 center 的 macro AUROC、macro AUPRC。
4. 每个 center 的 per-class AUROC/AUPRC。
5. 输出带 v3 版本名的 JSON，供后续 baseline、center-token、TA-OMAT、PN2021-C 对比使用。

## 固定评测中心

只评估：

```text
chapman_shaoxing
cpsc_2018
cpsc_2018_extra
georgia
ningbo
ptb
st_petersburg_incart
```

硬排除：

```text
ptb-xl
ptbxl
```

`ptb-xl` / `ptbxl` shard 是 PTB-XL 数据泄漏，不能进入平均。

## 输入

默认 baseline checkpoint：

```text
/root/autodl-tmp/triple_labels/super5/best_model.pt
```

PTB-XL：

```text
/root/autodl-tmp/ptbxl/raw100.npy
/root/autodl-tmp/ptbxl/ptbxl_database.csv
/root/autodl-tmp/crosscenter_v2/ptbxl_preprocessed.npy
```

PN2021 raw：

```text
/root/autodl-tmp/physionet2021/training/<center>/*.hea
/root/autodl-tmp/physionet2021/training/<center>/*.mat
```

PN2021 v3 eval cache：

```text
/root/autodl-tmp/triple_labels/pn2021_eval_cache/
  super5_<center>_100hz1000_v3_super5_normsuppress.npz
```

确认的 IO/cache 改进：

```text
保留上述 compressed .npz 作为兼容/历史 cache。
新增 mmap-friendly cache 目录：
  /root/autodl-tmp/triple_labels/pn2021_eval_cache_mmap/
    <scheme>_<center>_100hz1000_v3_super5_normsuppress/
      signals.npy
      labels.npy
      record_ids.npy
      metadata.json
```

原因：

- compressed `.npz` 在 cache hit 时仍要整文件解压，不能 mmap。
- `ningbo` v3 cache 已超过 1GB，后续批量重评测会重复付 CPU 解压成本。
- 新 mmap cache 不覆盖旧 `.npz`，`eval_crosscenter.py` 应优先读取 mmap cache，旧 cache 作为 fallback。

## 标准命令

重新生成/校验标签映射 sanity：

```bash
/root/miniforge3/envs/ECGTwin/bin/python scripts/triple_labels/label_schemes.py \
  --sanity \
  --scheme super5 \
  --mimic_n 2000
```

重跑 clean PN2021 7-center：

```bash
/root/miniforge3/envs/ECGTwin/bin/python scripts/triple_labels/eval_crosscenter.py \
  --scheme super5 \
  --model_dir /root/autodl-tmp/triple_labels/super5 \
  --batch_size 256 \
  --num_workers 8 \
  --skip_mimic \
  --output_path /root/autodl-tmp/triple_labels/super5/eval_result_v3_super5_normsuppress.json
```

如果要评估新的训练 run，把 `--model_dir` 和 `--output_path` 改到该 run 目录。

## 必须保存的产物

每个 run 至少保存：

```text
best_model.pt
training_log.json
train_result.json
eval_result_v3_super5_normsuppress.json
```

评测 JSON 必须包含：

```text
cache_versions.pn2021_eval = v3_super5_normsuppress
label_mapping.pn2021_super5.mapping_version
label_mapping.pn2021_super5.mapping_hash
```

论文汇总时另存：

```text
pn2021_v3_center_summary.csv
pn2021_v3_per_class_summary.csv
```

当前 `train_ptbxl.py` 不会自动生成曲线图或 delta CSV；这些汇总文件需要后处理脚本生成。

## 缺口和 TODO

- `scripts/triple_labels/eval_crosscenter.py` 的 PN2021 center DataLoader 当前有 `num_workers=2` 硬编码风险，应改为使用 CLI `--num_workers`。
- PN2021 cache metadata 当前包含 `crop_len/crop_mode`，但 cache 保存的是 full `(1000,12)` 信号；后续应避免 crop_len 改变导致不必要 rebuild。
- `ptbxl_labels.C5.all.npy` 未编码 mapping/confidence 版本；如修改 PTB-XL super5 规则，必须删除旧 cache 并记录新计数。
- 评测结果不能只看 macro；必须同时看 per-center 和 per-class，尤其 `NORM`、`STTC` 的跨中心漂移。

## 自动执行顺序

1. 检查 baseline checkpoint 是否存在。
2. 跑 `label_schemes.py --sanity`。
3. 删除或忽略旧 v2/unversioned PN2021 eval JSON。
4. 运行 `eval_crosscenter.py` 输出 v3 JSON。
5. 从 v3 JSON 导出 center/per-class CSV。
6. 把结果写回实验汇总文档，旧结果只保留为 historical。

当前实现状态：

- `eval_crosscenter.py` 已支持 `--pn2021_mmap_cache_dir`，默认
  `/root/autodl-tmp/triple_labels/pn2021_eval_cache_mmap`。
- 旧 compressed `.npz` 会自动转换为 mmap cache。
- PN2021 DataLoader 已改为使用 CLI `--num_workers`。
- limit smoke 已通过，7 个中心均可从 mmap cache 读取。

## Current Re-Evaluation Results

Baseline v3 clean result:

```text
model: /root/autodl-tmp/triple_labels/super5/best_model.pt
output: /root/autodl-tmp/triple_labels/super5/eval_result_v3_super5_normsuppress.json
PTB-XL fold10: AUROC=0.9064, AUPRC=0.7754
PN2021 7-center avg: AUROC=0.8344, AUPRC=0.5526
```

P1 comparable run batch re-eval:

```text
input list: /root/autodl-tmp/triple_labels/p1_v3_eval_dirs_20260501.txt
summary: /root/autodl-tmp/triple_labels/p1_v3_eval_summary_20260501.jsonl
status: 12 ok, 15 failed due incompatible 6-class checkpoints
```

CSV export completed:

```text
script: scripts/triple_labels/export_pn2021_v3_summaries.py
outputs:
  /root/autodl-tmp/triple_labels/pn2021_v3_center_summary.csv
  /root/autodl-tmp/triple_labels/pn2021_v3_per_class_summary.csv
included: 50 v3-super5 eval JSONs
rows: center=350, per_class=1750
```

Successful P1 super5 checkpoints ranked by PN2021 AUPRC:

| rank | model | PTB-XL AUROC/AUPRC | PN2021 AUROC/AUPRC |
|---:|---|---:|---:|
| 1 | `synth_anchored_super5_v13b_K400/nin_k400` | 0.9062/0.7765 | 0.8334/0.5555 |
| 2 | `synth_anchored_super5_v14_ctv2/geo_k200` | 0.9064/0.7767 | 0.8340/0.5550 |
| 3 | `synth_anchored_super5_v14_ctv2/extra_k200` | 0.9064/0.7761 | 0.8334/0.5550 |
| 4 | `synth_anchored_super5_v13/geo_k200` | 0.9064/0.7767 | 0.8329/0.5549 |
| 5 | `synth_anchored_super5_v13/nin_k200` | 0.9062/0.7762 | 0.8329/0.5548 |
| 6 | `synth_anchored_super5_v14_ctv2/nin_k200` | 0.9064/0.7770 | 0.8335/0.5548 |
| 7 | `synth_anchored_super5_v13b_K400/geo_k400` | 0.9061/0.7759 | 0.8335/0.5547 |
| 8 | `real_anchored_super5/extra_real_k200` | 0.9061/0.7758 | 0.8334/0.5542 |
| 9 | `real_anchored_super5/nin_real_k200` | 0.9063/0.7763 | 0.8337/0.5541 |
| 10 | `synth_anchored_super5_v13b_K400/extra_k400` | 0.9060/0.7752 | 0.8322/0.5535 |
| 11 | `synth_anchored_super5_v13/extra_k200` | 0.9061/0.7760 | 0.8325/0.5534 |
| 12 | `real_anchored_super5/geo_real_k200` | 0.9064/0.7765 | 0.8334/0.5533 |

Failed P1 items are not comparable super5 checkpoints. Their `best_model.pt`
classifier head has shape `(6, 640)` / `(6,)`, while current super5 eval expects
5 classes. They should be treated as historical Tier-M/old-label runs unless a
separate 6-class eval path is requested.
