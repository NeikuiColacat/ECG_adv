# 500Hz VAE: ECGTwin 原版权重与外部候选方向

Date: 2026-06-02

## 结论

可以继续探索“借助 ECGTwin 原作者 VAE 权重训练更强 500Hz VAE”的方向，但不能直接长跑或默认替换当前主线。

当前事实是：

- ECGTwin 原版 VAE 权重存在，并且可以严格加载到 ECGTwin 原架构的 5000 点版本；
- ECGTwin-init VAE500 和 ECGTwin-teacher DiffuSETSVAE500 都已经实现并跑过 pilot；
- 两条路线技术上可行，但目前没有在下游 VAE-only online AT 中超过 active spectral-r001 VAE500；
- 外部 500Hz ECG VAE 没找到可以直接落地替换的公开 checkpoint，SE-Diff 更适合作为结构/损失参考。

## ECGTwin 原版 VAE

本地权重：

```text
model/ECGTwin/checkpoints/vae_model.pth
/root/autodl-tmp/models/ECGTwin/checkpoints/vae_model.pth
```

协议：

```text
original input/output: (B, 1024, 12)
original latent:       (B, 4, 128)
latent scale:          0.18215
checkpoint keys:       encoder, decoder
```

由于 ECGTwin 原版 VAE 是卷积结构，没有固定长度的位置嵌入，因此复用同一模块到 5000 samples 时，latent 变成：

```text
5000 / 8 = 625
ECGTwin-init VAE500 latent: (B, 4, 625)
```

已有实现：

```text
ecg_adv_gen/vae/ecgtwin_vae500.py
scripts/vae500/train_ptbxl_vae500.py --model_variant ecgtwin_init_vae500
```

## 已有实验事实

| VAE branch | Pearson | first-diff Pearson | MSE | lead residual | decision |
|---|---:|---:|---:|---:|---|
| active DiffuSETSVAE500 | 0.997656 | 0.950460 | 0.007098 | 1.0883 | active baseline |
| ECGTwin-init VAE500 | 0.996431 | 0.947812 | 0.009768 | 1.1255 | 当前不继续长跑 |
| ECGTwin-teacher DiffuSETSVAE500 | 0.997734 | 0.950630 | 0.006939 | 1.1592 | gated candidate only |

下游结论：

```text
ECGTwin-init / teacher 分支没有证明能超过 spectral-r001 VAE500。
原版 ECGTwin 1024 latent -> decode -> resample 到 5000 的诊断也没有修复 direct40-init 更新被 K500 validation 拒绝的问题。
```

因此，当前问题不只是“VAE 训练得不够久”，更可能是：

```text
latent geometry + online-AT objective + sample/selection policy
```

共同限制了下游增益。

## 下一步 gate

只允许先做 CPSC + Ningbo 两中心 gate。

Gate 1: 重建和生理审计

```text
finite decode rate = 100%
invalid/flatline/severe amplitude rate near 0
Pearson 不低于 active DiffuSETSVAE500
MSE/MAE 不比 active DiffuSETSVAE500 差 >5%
lead residual 不劣化
```

Gate 2: latent-hull plausibility

```text
same-label latent interpolation 可正常 decode
small latent perturbation 不产生明显异常波形
decoded_invalid_rate near 0
loss_gain positive
atk_init roughly 0.3-0.7
```

Gate 3: downstream pilot

```text
centers: cpsc_2018, ningbo
matched control: direct40 K500 fine-tune
comparison: active spectral-r001 VAE500 + hard loss-gain
continue only if:
  cpsc_2018 AUPRC improves
  ningbo AUPRC non-decreasing
```

只有这三个 gate 都过，才扩展到四中心。

## 外部 500Hz VAE

| candidate | status | recommendation |
|---|---|---|
| SE-Diff | ICLR 2026 ECG latent diffusion；repo 需要 prerequisites 权重/latent data，但未在仓库里找到可直接使用的 VAE checkpoint | 最高优先级结构参考，不是当前 drop-in replacement |
| DiffuSETS / ECGTwin | 当前项目最接近的可控 VAE family | 保持主线 owned baseline |
| ECGEN | 12-lead 5000-sample VAE 思路接近 | 二级工程 baseline，先确认代码/权重/license |
| TimeVQVAE / ECG tokenizer | 需要重构 latent-hull 方法 | 长期方向，不进入当前冲刺 |

外部模型原则：

```text
如果使用外部 pretrained VAE，必须在论文里标注 external pretraining。
主线 PTB-XL-only claim 不能混入未知外部数据训练的 VAE 权重。
```

## 当前建议

1. 不要直接把 ECGTwin-init VAE500 继续训练很多 epoch。
2. 如果继续 VAE 架构方向，优先做 teacher-distill + SE-Diff-style constraints：
   - spectral loss；
   - first-cycle / beat-level auxiliary decoder；
   - inter-lead / Einthoven consistency；
   - morphology-aware validation。
3. 每次只跑 CPSC + Ningbo gate，避免浪费四中心长跑时间。
4. 如果新 VAE 过不了 gate，回到 online-AT objective 和 selection policy，而不是继续换 VAE。

## Evidence Registry Note

本轮还补了当前 root AutoDL 主机配置：

```text
configs/local/autodl_root.example.yaml
```

使用该配置运行：

```bash
/root/miniforge3/envs/ECGTwin/bin/python scripts/agent/audit_agent_workspace.py \
  --local-config configs/local/autodl_root.example.yaml
```

结果仍然未通过，但原因已经收敛为 artifact lineage 问题：

```text
registry 中声明的 2026-05-28 managed run artifacts、
comparison bundle、
K500 ref meta
没有完整存在于当前 /root/autodl-tmp/runs 和对应旧目录下。
```

这不等同于模型实验失败。它说明若要把当前 500Hz/v7 证据升级为
agent-readable trusted evidence，需要后续补：

```text
run_manifest / run_card
metrics_long
paper_table
comparison_bundle
K500 ref meta 的当前主机路径或重新导出
```

## Links

- SE-Diff code: https://github.com/ignite-abd/SE-Diff
- SE-Diff OpenReview: https://openreview.net/forum?id=95ZV35sBDm
- DiffuSETS code: https://github.com/Raiiyf/DiffuSETS_Exp
- DiffuSETS assets: https://zenodo.org/records/15420698
- ECGEN code: https://github.com/vlbthambawita/ECGEN
