# Cross-Label Latent-Hull AT Ablation, 2026-05-17

## Objective

Test three VAE latent-hull online adversarial training variants:

1. `exact_multi_hot`: exact same multi-hot latent candidates, full anchor
   multi-hot labels.
2. `compatible_latent_soft`: NORM-only vs abnormal-compatible candidate pool,
   anchor-preserving fractional labels from optimized latent-hull weights.
3. `compatible_teacher_soft`: same compatible fractional labels blended with a
   frozen initial EfficientNet teacher.

All runs used target-center real K=500 ECGTwin VAE anchors, M=20, lambda=0.15,
10 epochs, quality gate disabled, PTB-XL minimal-resample/per-sample-global
baseline initialization, crop length 1000, and full PN2021 v3 evaluation with
the K=500 reference IDs excluded from the target-center test set.

## Artifacts

- Runner: `scripts/paper/run_crosslabel_lhat_ablation_20260517.py`
- Summary CSV:
  `/root/autodl-tmp/paper_crosslabel_lhat_ablation_20260517/summaries/crosslabel_lhat_ablation.csv`
- Summary Markdown:
  `/root/autodl-tmp/paper_crosslabel_lhat_ablation_20260517/summaries/crosslabel_lhat_ablation.md`
- Run root:
  `/root/autodl-tmp/paper_crosslabel_lhat_ablation_20260517/runs/`

Completed artifact count:

```text
12 training_log.json
12 eval_result_v3_super5_normsuppress_exclrefs_crop1000.json
```

## Target-Center Results

Values are AUROC / AUPRC. Deltas are relative to `exact_multi_hot`.

| center | exact_multi_hot | compatible_latent_soft | compatible_teacher_soft |
|---|---:|---:|---:|
| ningbo | 0.8917 / 0.5223 | 0.8917 / 0.5223 (+0.00 / +0.01 pp) | 0.8917 / 0.5223 (+0.00 / +0.00 pp) |
| chapman_shaoxing | 0.8967 / 0.4637 | 0.8967 / 0.4639 (+0.00 / +0.02 pp) | 0.8967 / 0.4639 (+0.00 / +0.01 pp) |
| cpsc_2018 | 0.8545 / 0.5962 | 0.8544 / 0.5962 (-0.00 / -0.00 pp) | 0.8544 / 0.5962 (-0.00 / -0.01 pp) |
| georgia | 0.8256 / 0.6020 | 0.8256 / 0.6020 (-0.00 / -0.00 pp) | 0.8256 / 0.6020 (-0.00 / -0.00 pp) |

## Interpretation

The three variants are effectively tied under this conservative setting. The
compatible cross-label search did not hurt the established real-anchor LH-AT
effect, but it also did not produce a meaningful target-center AUROC/AUPRC gain.

Likely reason: the current setting keeps the perturbation close to the anchor
(`lambda=0.15`) and keeps new soft classes capped (`lambda_y=0.5`,
`new_class_cap=0.5`), while the supervised target-real stream has high weight
(`target_real_weight=40`). In practice, the adversarial samples remain dominated
by the anchor label manifold, so compatible cross-class freedom only changes the
training signal at the second decimal pp level.

Dataset-specific notes:

- `cpsc_2018` K=500 has no MI anchors in this selection, so its cross-label
  soft experiment mainly probes NORM/STTC/CD-compatible structure, not MI.
- `georgia` has very few MI anchors, so MI-related cross-label conclusions are
  weak for this center.
- `chapman_shaoxing` has the richest multi-label structure among these runs,
  yet the compatible-soft improvement over exact is still only about +0.02 pp
  AUPRC.

## Recommendation

Keep `exact_multi_hot` or the existing real-anchor LH-AT route as the main
method. Treat compatible cross-label soft labels as a safe but currently
non-improving ablation.

If this direction is expanded later, use a stronger stress test:

- increase `lambda` to 0.25 or 0.35;
- increase `lambda_y` to 0.75;
- lower `target_real_weight` from 40 to 20;
- include a center/K split with enough MI anchors;
- compare with and without `new_class_cap=0.5`.
