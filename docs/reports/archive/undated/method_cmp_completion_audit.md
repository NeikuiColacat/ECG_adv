# Method Comparison Completion Audit

Date: 2026-05-17

## Objective Restatement

Run prioritized comparison methods against our VAE real-anchor latent-hull AT
route, report AUROC/AUPRC, automate execution, avoid repeated work, control
`/root/autodl-tmp` disk use, and prefer cloning/reusing existing GitHub repos
where practical.

## Requirement-To-Evidence Checklist

| Requirement | Evidence | Status |
|---|---|---|
| Compare against source-only and K-shot target labeled baselines | `/root/autodl-tmp/paper_direct_finetune_k500_20260516/summaries/direct_finetune_k500.csv`; `/root/autodl-tmp/paper_kshot_baseline_cmp_20260516/summaries/kshot_baseline_comparators.csv` | Done |
| Compare Mixup / Manifold Mixup / SAM | Same K-shot comparator summary; includes `mixup`, `manifold_mixup`, `sam` four-center runs | Done |
| Compare ordinary input-space adversarial training | `fgsm_at` and `pgd_at` rows added to `kshot_baseline_comparators.csv`; HTML section `5i` | Done |
| Compare source-domain generalization baselines | `/root/autodl-tmp/paper_source_dg_baselines_20260517/mixstyle_super5/...`; `/root/autodl-tmp/paper_source_dg_baselines_20260517/groupdro_super5/...` | Done |
| Compare TTA / source-free adaptation baselines | AdaBN, TENT, EATA-lite, SHOT-style outputs under `/root/autodl-tmp/paper_tta_baselines_20260517/`; CoTTA-lite added under `/root/autodl-tmp/paper_tta_baselines_20260517/cotta_lite_lr1e-6_aug4_rst001/` | Done |
| Compare UDA baselines | Deep CORAL, DANN, CDAN outputs under `/root/autodl-tmp/paper_uda_baselines_20260517/` | Done |
| Compare ECG foundation / SSL encoders | ECGFounder, ECG-FM, ST-MEM, MERL outputs under `/root/autodl-tmp/paper_foundation_baselines_20260517/` | Done |
| Prefer cloned GitHub repos when practical | Reused or cloned `/root/autodl-tmp/external_repos/ECG-FM`, `ST-MEM`, `MERL-ICML2024`, `cotta`, `CLOCS`, `EATA`, `SHOT-plus`, `Transfer-Learning-Library`; repo code was inspected before local adaptation | Done |
| Avoid wasting disk | Current checked free space: `/root/autodl-tmp` has about 35 GB free; comparison roots are small: foundation baselines about 925 MB, K-shot comparators about 710 MB, external repos about 1.4 GB | Done |
| Report AUROC/AUPRC in one accessible artifact | `docs/reports/archive/undated/method_cmp_four_center_results.html` includes four-center main table and method-specific sections | Done |
| Identify methods not run and why | PCLR requires TensorFlow/Keras and current `ECGTwin` env segfaults on TensorFlow import; running it would require a new heavy isolated TF env. CLOCS official repo has code but no direct official pretrained weights; from-scratch pretraining is not comparable. Auto-TTE/text-to-ECG is lower-priority synthetic augmentation rather than target-center small-K adaptation, and would require separate generation/quality pipeline plus extra disk. | Done |

## Current Strongest Findings

Same EfficientNet backbone:

- Our best LH-AT remains stronger than ordinary K-shot, Mixup, Manifold Mixup,
  SAM, FGSM/PGD input-space AT, source DG, UDA, and TTA baselines.
- Input-space AT is a useful negative control:
  - Ningbo FGSM/PGD around `0.8825 / 0.5027`;
  - Chapman around `0.8891 / 0.4485`;
  - CPSC around `0.8227 / 0.5615`;
  - Georgia around `0.8219 / 0.5997`.
  These are below our LH-AT results.

External foundation encoders:

- ECG-FM, ST-MEM, and MERL do not exceed our LH-AT under frozen LP / K500 head FT.
- ECGFounder K-shot head fine-tuning remains the only external foundation route
  that consistently exceeds our target-center AUPRC, but it changes the backbone
  and should be reported as a strong external representation baseline rather
  than a same-backbone adaptation method.

## Verification

- HTML parse check passed for `docs/reports/archive/undated/method_cmp_four_center_results.html`.
- CoTTA-lite output CSV verified at
  `/root/autodl-tmp/paper_tta_baselines_20260517/cotta_lite_lr1e-6_aug4_rst001/cotta_lite_super5_ref_excluded.csv`.
- No active `scripts/paper` training process was present after the final run.
- Current disk check showed `/root/autodl-tmp` at about `35G` free.
