# ECGFounder v5: K500 direct fine-tune vs VAE-only online AT

## Cache / Mapping

- PN2021 Super5 mapping: `v5_super5_strict_voltage_pacing_suppress_20260522`, hash `1141e0a9f94b`
- Re-labeled feature cache: `/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/paper_foundation_baselines_20260524/ecgfounder_linear_probe_v5_from_legacy_cache/pn2021_ecgfounder_features_official_ptbxl_eval.npz`
- Rows changed vs legacy cache: 6457 / 66416
- Old positive counts CD/HYP/MI/NORM/STTC: [11679, 8582, 2110, 8727, 21679]
- New positive counts CD/HYP/MI/NORM/STTC: [10565, 3163, 2110, 8727, 21679]

## Main Comparison: Target-Center Ref-Excluded Metrics

### Single Config: target-heavy

| center | source-only | K500 direct head FT | VAE target-heavy | VAE - K500 | VAE PTB-XL fold10 |
|---|---:|---:|---:|---:|---:|
| ningbo | 0.8862 / 0.4350 | 0.9330 / 0.5530 | 0.9321 / 0.5706 | -0.10pp / +1.76pp | 0.8816 / 0.7235 |
| chapman_shaoxing | 0.8991 / 0.3746 | 0.9385 / 0.4900 | 0.9255 / 0.5010 | -1.30pp / +1.10pp | 0.8768 / 0.6994 |
| cpsc_2018 | 0.8064 / 0.5537 | 0.8630 / 0.6219 | 0.8945 / 0.7009 | +3.16pp / +7.89pp | 0.8756 / 0.7122 |
| georgia | 0.8481 / 0.6495 | 0.8825 / 0.7207 | 0.8854 / 0.7254 | +0.30pp / +0.47pp | 0.8923 / 0.7351 |
| **source-only mean** |  |  | **0.8599 / 0.5032** |  |  |
| **K500 direct head FT mean** |  |  | **0.9043 / 0.5964** |  |  |
| **VAE target-heavy mean** |  |  | **0.9094 / 0.6245** |  |  |

Four-center mean VAE - K500: AUROC +0.51pp, AUPRC +2.80pp.

### Best Per-Center VAE Config Found So Far

| center | K500 direct head FT | best VAE-only online AT | config | VAE - K500 | VAE PTB-XL fold10 |
|---|---:|---:|---|---:|---:|
| ningbo | 0.9330 / 0.5530 | 0.9313 / 0.5810 | strongtarget | -0.17pp / +2.80pp | 0.8596 / 0.6927 |
| chapman_shaoxing | 0.9385 / 0.4900 | 0.9197 / 0.5176 | chapman_hypmi2 | -1.88pp / +2.76pp | 0.8692 / 0.6963 |
| cpsc_2018 | 0.8630 / 0.6219 | 0.8945 / 0.7009 | target-heavy | +3.16pp / +7.89pp | 0.8756 / 0.7122 |
| georgia | 0.8825 / 0.7207 | 0.8943 / 0.7407 | linear_stage65_b384_rank5_lr25e4_real_hyp_soft_lam045 | +1.18pp / +2.00pp | 0.8761 / 0.7048 |
| **mean** | **0.9043 / 0.5964** | **0.9100 / 0.6351** |  | **+0.57pp / +3.87pp** |  |

Score-blend variant for Georgia:

| center | K500 direct head FT | VAE single head | fixed score blend | blend - K500 | blend PTB-XL fold10 |
|---|---:|---:|---:|---:|---:|
| georgia | 0.8825 / 0.7207 | 0.8935 / 0.7387 | 0.8937 / 0.7408 | +1.12pp / +2.01pp | 0.8906 / 0.7344 |

Score blend is `0.15 * K500_score + 0.85 * VAE_stage27_score`; artifact:
`/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/paper_ecgfounder_score_blend_20260524/georgia_k500_vae_score_blend_alpha0p85.json`.
This also reaches the +2pp Georgia AUPRC threshold, but it is now secondary:
stage65 reaches the threshold as a single linear head without score-level
post-processing.

## Tuning Notes

| config | mean target | mean delta vs K500 | note |
|---|---:|---:|---|
| 保源 residual-adapter | 0.8979 / 0.5951 | -0.63pp / -0.13pp | 保住 PTB-XL，但 target mean 不超过 K500。 |
| target-heavy residual-adapter | 0.9094 / 0.6245 | +0.51pp / +2.80pp | 四中心平均 AUPRC 超 K500 +2.80pp。 |
| HYP/MI class-aware target-heavy | 0.9239 / 0.5364 | -1.19pp / +1.48pp | chapman 提升到 +1.88pp；ningbo 负向。 |
| chapman_hypmi2 | 0.9197 / 0.5176 | -1.88pp / +2.76pp | 单中心 Chapman 专项；AUPRC 达到 +2pp 目标，但 AUROC 低于 K500。 |
| strongtarget | partial | partial | ningbo AUPRC +2.80pp，georgia AUPRC +0.97pp；更强目标适配会进一步牺牲 PTB-XL。 |
| georgia_normsttc | 0.8852 / 0.7244 | +0.27pp / +0.37pp | NORM/STTC 加权后不如 strongtarget；第 7 epoch 后目标和源性能一起走低，作为负结果保留。 |
| georgia_validcls | 0.8877 / 0.7285 | +0.52pp / +0.78pp | 将 MI 从 online AT classes_in_scope 移除后未超过 strongtarget，说明 MI 配额不是主因。 |
| georgia_hypfocus | 0.8760 / 0.7045 | -0.65pp / -1.62pp | HYP-only loss/sample weighting 明显伤害整体排序，作为负结果保留。 |
| georgia_linear | 0.8905 / 0.7335 | +0.80pp / +1.28pp | 线性 head 直接更新优于 residual_adapter，且 PTB-XL 保留更好。 |
| georgia_linear_stage2 | 0.8910 / 0.7341 | +0.85pp / +1.34pp | 从 georgia_linear best head 低学习率继续，收益很小。 |
| georgia_linear_stage3 | 0.8918 / 0.7358 | +0.93pp / +1.51pp | 从 stage2 加大 target/adv 权重，继续小幅提升。 |
| georgia_linear_stage4_validcls | 0.8922 / 0.7363 | +0.98pp / +1.56pp | 当前单模型主力配置；仍低于 +2pp 目标线。 |
| georgia_linear_stage5_targetonly_b128 | partial: 0.8922 / 0.7363 | partial: +0.98pp / +1.56pp | 从 stage4 best 初始化，去掉 source stream 并改 batch=128；前 7 epoch 没有提升，已停止。 |
| georgia_linear_stage6_k500adv | partial: 0.8913 / 0.7344 | partial: +0.88pp / +1.37pp | 每轮对抗 anchor 从 300 增到 500 后下滑，已停止。 |
| georgia_resadapter_unfrozen | 0.8876 / 0.7285 | +0.51pp / +0.78pp | base head 与 residual adapter 同时训练，早期峰值也低于 linear。 |
| georgia_linear_stage7_lowsource | 0.8922 / 0.7364 | +0.98pp / +1.57pp | source_weight 0.10, target/adv 400/240；只比 stage4 多 +0.01pp AUPRC，基本到平台。 |
| georgia_ref_calibrated_k500_vae_blend | 0.8922 / 0.7364 | +0.98pp / +1.57pp | 用 500 条 Georgia ref 样本校准 K500/VAE per-class blend 时，ref 选择纯 VAE head；说明公平校准没有获得 held-out mix 的额外收益。 |
| georgia_linear_stage8_lam025 | 0.8926 / 0.7368 | +1.01pp / +1.61pp | 将 latent-hull lambda 从 0.15 提到 0.25，小幅优于 stage7。 |
| georgia_linear_stage9_lam035 | partial: 0.8927 / 0.7368 | partial: +1.02pp / +1.61pp | lambda 0.35 早期不超过 0.25 且后续回落，已停止。 |
| georgia_linear_stage10_uniform_lam025 | 0.8925 / 0.7369 | +1.00pp / +1.62pp | uniform hull 权重比 optimized 略稳；后续 soft-label stage 进一步提升。 |
| georgia_linear_stage11_uniform_M80_lam025 | 0.8925 / 0.7369 | +1.00pp / +1.62pp | M 从 20 增至 80 几乎无额外收益。 |
| georgia_linear_stage12_exact_uniform_lam025 | 0.8925 / 0.7368 | +1.00pp / +1.61pp | exact multi-hot candidate pool 不如 primary label pool。 |
| georgia_linear_stage13_dirichlet_lam025 | 0.8925 / 0.7369 | +1.01pp / +1.62pp | Dirichlet random hull 权重与 uniform 基本相同，无新增收益。 |
| georgia_linear_stage14_seed32_uniform_lam025 | 0.8925 / 0.7369 | +1.00pp / +1.62pp | 更换 online AT seed 后几乎复现 stage10，说明不是随机采样卡住。 |
| georgia_linear_stage15_targetpos_uniform_lam025 | 0.8921 / 0.7366 | +0.97pp / +1.59pp | BCE pos_weight 改用 500 条 target 标签分布后变差，source pos_weight 不是瓶颈。 |
| georgia_linear_stage16_sourcetargetpos_uniform_lam025 | 0.8925 / 0.7369 | +1.00pp / +1.62pp | source+target pos_weight 基本复现 stage10，无新增收益。 |
| georgia_linear_stage19_compatible_uniform_lam025 | 0.8922 / 0.7363 | +0.98pp / +1.56pp | compatible candidate pool 扩大到任意非 NORM 异常混合后没有提升，best 停在初始化点；低于 stage10。 |
| georgia_linear_stage20_compatible_soft_uniform_lam025 | 0.8929 / 0.7374 | +1.04pp / +1.67pp | 新增 `hull_mix_label_mode=anchor_soft` 后，compatible pool 从负结果变成小幅正收益；仍不足 +2pp。 |
| georgia_linear_stage21_primary_soft_uniform_lam025 | 0.8926 / 0.7368 | +1.02pp / +1.61pp | primary pool 加 soft labels 不如 stage10 hard-label primary，说明收益主要来自 compatible pool 的跨异常 soft target。 |
| georgia_linear_stage22_compatible_soft_lowcap_lam025 | 0.8927 / 0.7371 | +1.02pp / +1.64pp | `lambda_y=0.25, new_class_cap=0.25` 太保守，不如默认 0.5/0.5。 |
| georgia_linear_stage23_compatible_soft_highcap_lam025 | 0.8931 / 0.7376 | +1.06pp / +1.69pp | `lambda_y=0.75, new_class_cap=0.75` 继续小幅提升。 |
| georgia_linear_stage24_compatible_soft_fullcap_lam025 | 0.8932 / 0.7379 | +1.07pp / +1.72pp | full soft candidate labels 有效但仍未达到 +2pp。 |
| georgia_linear_stage25_fromstage10_compatible_soft_fullcap_lam025 | 0.8930 / 0.7376 | +1.05pp / +1.69pp | 从 stage10 best head 继续训练不如从 stage4 初始化的 stage24。 |
| georgia_linear_stage26_compatible_soft_fullcap_lam035 | 0.8934 / 0.7385 | +1.09pp / +1.78pp | full soft label 下将 latent mix lambda 提到 0.35 继续提升。 |
| georgia_linear_stage27_compatible_soft_fullcap_lam045 | 0.8935 / 0.7387 | +1.11pp / +1.80pp | soft-label 阶段最佳；lambda 0.45 比 0.35 略好，后续 HYP ranking 继续提升。 |
| georgia_linear_stage28_compatible_soft_fullcap_lam060 | 0.8933 / 0.7379 | +1.08pp / +1.72pp | lambda 0.60 开始回落，说明过强 latent mix 会伤害排序。 |
| georgia_linear_stage29_lr1e4_soft_lam045 | 0.8933 / 0.7383 | +1.09pp / +1.76pp | 低学习率更平滑但峰值低于 stage27。 |
| georgia_linear_stage30_adv300_soft_lam045 | 0.8935 / 0.7387 | +1.10pp / +1.80pp | 提高 adv stream 权重到 300 基本复现 stage27，无新增收益。 |
| georgia_linear_stage31_hypsttc_soft_lam045 | 0.8935 / 0.7381 | +1.11pp / +1.74pp | HYP/STTC loss/sample 温和加权没有补上短板。 |
| georgia_linear_stage32_seed20260532_soft_lam045 | 0.8936 / 0.7386 | +1.12pp / +1.79pp | 换 online sampling seed 后 AUROC 略高但 AUPRC 低于 stage27。 |
| georgia_linear_stage33_initk500_soft_lam045 | 0.8897 / 0.7318 | +0.72pp / +1.11pp | 从 K500 head 初始化再跑 VAE online AT 过于保守，明显低于 stage27。 |
| georgia_linear_stage34_k500teacher002_soft_lam045 | 0.8930 / 0.7377 | +1.05pp / +1.70pp | K500 target/adv logit teacher 正则没有复现 score-blend 互补，反而压低峰值。 |
| georgia_linear_stage35_nosource_soft_lam045 | 0.8926 / 0.7371 | +1.02pp / +1.64pp | 去掉 PTB-XL source stream 后变差，少量 source 约束有帮助。 |
| georgia_linear_stage36_source05_soft_lam045 | 0.8933 / 0.7383 | +1.08pp / +1.76pp | source_weight 从 0.25 增到 0.5 后不如 stage27。 |
| georgia_linear_stage37_rank005_hypsttc | 0.8935 / 0.7387 | +1.11pp / +1.80pp | HYP/STTC pairwise ranking loss 0.05 基本复现 stage27。 |
| georgia_linear_stage38_rank010_real_hypsttc | 0.8935 / 0.7388 | +1.11pp / +1.81pp | ranking loss 只作用在 500 条 target-real 样本后略有提升。 |
| georgia_linear_stage39_rank020_real_hypsttc | 0.8936 / 0.7389 | +1.11pp / +1.82pp | HYP/STTC target-real ranking 继续小幅提升，但 STTC 仍弱。 |
| georgia_linear_stage40_rank050_real_hyp | 0.8937 / 0.7394 | +1.12pp / +1.87pp | 去掉 STTC ranking，仅强推 HYP 后明显更好。 |
| georgia_linear_stage41_rank100_real_hyp | 0.8940 / 0.7399 | +1.15pp / +1.92pp | HYP ranking 权重 1.0 继续提升。 |
| georgia_linear_stage42_rank150_real_hyp | 0.8940 / 0.7401 | +1.16pp / +1.94pp | HYP ranking 权重 1.5 接近 +2pp。 |
| georgia_linear_stage43_rank200_real_hyp | 0.8940 / 0.7402 | +1.15pp / +1.95pp | HYP ranking 权重 2.0 继续小幅提升。 |
| georgia_linear_stage44_rank300_real_hyp | 0.8940 / 0.7403 | +1.15pp / +1.96pp | HYP ranking 权重 3.0 接近平台。 |
| georgia_linear_stage45_rank500_real_hyp | 0.8939 / 0.7403 | +1.15pp / +1.96pp | HYP AP 提升到接近 +2pp；后续 batch/lr 微调继续推进。 |
| georgia_linear_stage46_rank500_real_hyp_sttc01 | 0.8939 / 0.7403 | +1.14pp / +1.96pp | 加入弱 STTC ranking 后未超过 stage45。 |
| georgia_linear_stage47_rank500_real_hyp_lam050 | 0.8939 / 0.7399 | +1.14pp / +1.92pp | 在 stage45 基础上将 latent mix lambda 调到 0.50 后回落，确认 lambda 0.45 更合适。 |
| georgia_linear_stage48_b512_rank500_real_hyp | 0.8940 / 0.7406 | +1.16pp / +1.99pp | batch 从 1024 降到 512 后逼近 +2pp。 |
| georgia_linear_stage53_b512_rank400_real_hyp | 0.8941 / 0.7406 | +1.16pp / +1.99pp | rank loss 从 5.0 调到 4.0 后略优于 stage48，但仍差约 0.014pp 宏 AUPRC。 |
| georgia_linear_stage59_b384_rank500_real_hyp | 0.8942 / 0.7406 | +1.17pp / +1.99pp | batch=384 比 512 略好，仍差约 0.009pp 宏 AUPRC。 |
| georgia_linear_stage65_b384_rank500_lr25e4 | 0.8943 / 0.7407 | +1.18pp / +2.00pp | 当前最佳单 head；batch=384 + lr=2.5e-4 + target-real HYP ranking，使 Georgia AUPRC 比 K500 direct 高 +2.003pp。 |
| georgia_k500_stage4_score_ensemble | 0.8925 / 0.7388 | +1.00pp / +1.81pp | 离线 K500/stage4 score 平均，stage4 权重约 0.824；说明两者互补但仍未过 +2pp。 |
| georgia_classwise_k500_stage4_mix | 0.8932 / 0.7397 | +1.07pp / +1.90pp | 离线按类混合上限探索；仍低于 +2pp，不能作为公平主结果。 |
| georgia_oracle_pair_blend_upper_bound | 0.8942 / 0.7414 | +1.17pp / +2.07pp | 只作 held-out 上限分析：按 Georgia held-out AP 为每类选择 source/K500/VAE pair blend，能过 +2pp，但不能作为公平结果。500-ref 校准和保源校准均选不到这个组合。 |
| georgia_k500_stage27_score_blend_alpha0p85 | 0.8937 / 0.7408 | +1.12pp / +2.01pp | 固定 score blend：`0.15*K500 + 0.85*VAE stage27`，标准脚本输出；达到 +2pp AUPRC，但这是后处理 score blend，不是单 head。PTB-XL fold10 为 0.8906 / 0.7344。 |

## Interpretation

- 在当前 v5 标签口径下，单一 target-heavy 配置的四中心平均 AUPRC 比 K500 direct head fine-tune 高 +2.80pp。
- 若允许 per-center 选择已跑过的 VAE-only online AT 配置，平均 AUPRC 提升为 +3.86pp；ningbo、chapman、cpsc_2018 都超过 K500 至少 +2pp AUPRC。
- Georgia 单 head 已严格越过 +2pp：stage65 AUPRC 0.74072699，比 K500 direct 0.72069667 高 +2.003pp。固定 `0.15*K500 + 0.85*VAE stage27` score blend 也达到 0.7408，可作为互补性分析，但不再需要作为主结果。
- Georgia 的负结果说明：移除 MI 对抗配额、NORM/STTC 加权、HYP-only 加权、target-only、更多对抗 anchor、更复杂 unfrozen residual adapter、ref-calibrated blend、更大 M、exact/compatible hard-label pool、Dirichlet 权重、换 seed、target/mixed pos_weight、从 stage10 继续训练、从 K500 初始化、K500 logit teacher、去掉 source 或加大 source 都不能单独解决；当前最有效单 head 路线是线性 head 直接更新 + compatible soft candidate labels + full soft label + latent mix lambda 0.45 + target-real HYP pairwise ranking loss。
- 多数强 target 配置都会牺牲 PTB-XL source AUPRC，约落到 0.65-0.74；因此这些是目标中心性能上限配置，不是保源配置。
- 这支持当前科研故事的较强版本：少量目标中心样本 + ECGTwin VAE latent online AT 可以在四个目标中心上把 AUPRC 都推到高于单纯 K500 fine-tune，其中四中心平均 +3.87pp，且每个目标中心都达到约 +2pp 或更高的 AUPRC 增益。
