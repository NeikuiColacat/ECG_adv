# AutoDL Tmp Disk Review 2026-05-22

当前磁盘状态：

| mount | size | used | free | use |
|---|---:|---:|---:|---:|
| `/` | 30G | 20G | 11G | 65% |
| `/root/autodl-tmp` | 430G | 369G | 62G | 86% |

当前论文主线是 **VAE-only real-anchor latent-hull 在线对抗训练**。因此清理原则是：保留 PTB-XL、PN2021 原始数据、real-anchor K500 latent/signals、当前 baseline/online-AT 结果；优先清理 MIMIC、旧 ECGTwin 合成池、PN2021-C 物化缓存和迁移包。

## 优先可清理

这些项目与当前 VAE-only 主线无强依赖，删除后主要损失是少数历史复现实验或可再生成缓存。

| path | size | 建议 |
|---|---:|---|
| `/root/autodl-tmp/triple_labels/pn2021_c_cache_minresample_perglobal` | 62G | 若近期不继续 PN2021-C，优先删除。PN2021-C 可以流式重建；已有汇总 CSV/报告应先保留。 |
| `/root/autodl-tmp/mimic_tierM/mimic_preprocessed_f16.npy` | 18G | MIMIC 已不在论文主线；可删或移走。 |
| `/root/autodl-tmp/ECGTwin_Data/paired_Mimic_vae_mix_nomic.pt` | 14G | ECGTwin author paired MIMIC 历史数据；当前 VAE-only 不需要。 |
| `/root/autodl-tmp/ECGTwin_Data/paired_Mimic_vae_multi_nomic.pt` | 9.0G | 同上。 |
| `/root/autodl-tmp/physionet2021/sources/*.zip` | 9.2G | PN2021 已解压到 `physionet2021/training` 后，zip 可删；若担心重新下载麻烦，可先转移。 |
| `/root/autodl-tmp/transfer_artifacts` | 2.8G | 迁移包，确认另一台机器已拿到后可删。 |
| `/root/autodl-tmp/output.zip` | 1.1G | 旧压缩包，若不再需要可删。 |
| `/root/autodl-tmp/crosscenter_v2_smoke/ptbxl_preprocessed.npy` | 998M | smoke 旧缓存，可删。 |
| `/root/autodl-tmp/crosscenter_v2/ptbxl_preprocessed.npy` | 998M | 与当前主缓存重复；当前主缓存是 `/root/autodl-tmp/triple_labels/cache/ptbxl_minimal_resample_per_sample_global_fs100_len1000.npy`。 |

仅清理上面第一项 PN2021-C cache 就可释放约 **62G**；加上 MIMIC paired/cache 和 sources zip，可释放约 **110G+**。

## 可保留

| path | reason |
|---|---|
| `/root/autodl-tmp/ptbxl/raw100.npy` | PTB-XL 训练源数据，当前多模型训练需要。 |
| `/root/autodl-tmp/triple_labels/cache/ptbxl_minimal_resample_per_sample_global_fs100_len1000.npy` | 当前论文输入协议主缓存。 |
| `/root/autodl-tmp/physionet2021/training` | PN2021 原始中心数据，评估和 K500 real-anchor 溯源需要。 |
| `/root/autodl-tmp/ecgtwin_prompt_token_super5/real_anchor_selected_v1` 与 `v2` | 当前 VAE-only K500 target-center anchors 直接依赖。 |
| `/root/autodl-tmp/triple_labels/pn2021_eval_cache_mmap_minresample_perglobal` | 当前 PN2021 快速评估缓存，建议保留。 |
| `/root/autodl-tmp/triple_labels/super5_minresample_full10_perglobal_20260503` | EfficientNet1DV2 主 baseline。 |

## 暂缓清理

| path | size | 原因 |
|---|---:|---|
| `/root/autodl-tmp/ecgtwin_prompt_token_super5` | 14G | center-token ablation 仍可能写论文附录；先保留 real-anchor 和关键 ablation，后续可按报告筛。 |
| `/root/autodl-tmp/graduate_project` | 16G | 毕设阶段历史结果；论文整理完成前先别批量删。 |
| `/root/autodl-tmp/paper_real_anchor_lhat_seed_kfold_20260517` | 13G | 稳定性/K-fold 证据，论文可能引用。 |
| `/root/autodl-tmp/paper_vae_only_lhat_kcurve_20260518` | 5.7G | K 曲线证据，论文可能引用。 |
| `/root/autodl-tmp/models` | 2.7G | 外部 repo 与模型权重，不建议删。 |

## 推荐删除命令

先 dry-run 看路径：

```bash
du -sh /root/autodl-tmp/triple_labels/pn2021_c_cache_minresample_perglobal \
  /root/autodl-tmp/mimic_tierM/mimic_preprocessed_f16.npy \
  /root/autodl-tmp/physionet2021/sources \
  /root/autodl-tmp/transfer_artifacts \
  /root/autodl-tmp/output.zip
```

确认后再删除：

```bash
rm -rf /root/autodl-tmp/triple_labels/pn2021_c_cache_minresample_perglobal
rm -f /root/autodl-tmp/mimic_tierM/mimic_preprocessed_f16.npy
rm -rf /root/autodl-tmp/physionet2021/sources
rm -rf /root/autodl-tmp/transfer_artifacts
rm -f /root/autodl-tmp/output.zip
```

不建议我在未获明确确认前执行这些删除。
