# PN2021 Direct+fixed20 family-balanced baseline

Status: `LOCKED_SINGLE_SEED` on 2026-07-18.

The machine-readable source of truth is
`configs/baselines/pn2021_direct_v1.yaml`. The filename is retained because it
is already part of the refactor whitelist; it is overwritten in place rather
than creating v2/v3 compatibility files.

## Locked protocol

- Four logical centers: Ningbo, Chapman-Shaoxing, CPSC 2018 + Extra, Georgia.
- Each center contributes a fixed K500. Tuning uses deterministic 400 train / 100 validation; refit uses all 500.
- Each base batch produces one clean view and all 20 canonical PN2021-C views: 10 depth-2 and 10 depth-3.
- Loss is `0.5 * clean BCE + 0.5 * mean(20 corrupted BCE)`.
- The 21 weighted backward calls produce exactly one optimizer step per base batch.
- BatchNorm running statistics follow the same family weights: 50% clean and 50% total corrupted.
- Validation corruptions are frozen across epoch, hyperparameter, center, and backbone comparisons.
- Selection score is `0.5 * clean AUPRC + 0.5 * robust AUPRC`, subject to a same-backbone clean floor of historical clean-only AUPRC minus 1 pp.
- Held-out ref-excluded PN2021 and PN2021-C labels are never used for epoch selection.

Super5 order is `CD,HYP,MI,NORM,STTC`; mapping is
`v7_super5_sjr_rgq_review_20260528 / 555ec85d5b51`.

## Locked optimization

| Model | LR | WD | Batch | Cosine horizon | Selected/refit epoch | Validation clean AUPRC | Validation robust AUPRC | Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EfficientNet1DV2 | `5e-5` | `1e-4` | 128 | 30 | 23 | 0.728708 | 0.626459 | 0.677583 |
| ECGFounder full FT | `2e-5` | `1e-4` | 64 | 20 | 20 | 0.820272 | 0.724349 | 0.772310 |

Both use AdamW, bfloat16 AMP, and gradient clipping at 1.0. EfficientNet's
best point is interior at epoch 23. ECGFounder epoch 19 to 20 changes the score
by only +0.0025 pp, so epoch 20 is treated as a converged plateau rather than
extending the horizon using held-out feedback.

## Four-center held-out results

Each cell is macro `AUROC / AUPRC`. PN2021-C first averages the 20 corruption
views within each center, then averages the four centers equally.

| Model | Clean kept | PN2021-C kept | Clean drop-all-zero | PN2021-C drop-all-zero |
|---|---:|---:|---:|---:|
| EfficientNet1DV2 | `0.850009 / 0.505380` | `0.809939 / 0.438257` | `0.871183 / 0.643851` | `0.826919 / 0.566643` |
| ECGFounder | `0.889740 / 0.582508` | `0.830200 / 0.496628` | `0.910291 / 0.697136` | `0.853189 / 0.611176` |

Clean-to-PN2021-C degradation on the primary all-zero-kept view:

| Model | AUROC drop | AUPRC drop |
|---|---:|---:|
| EfficientNet1DV2 | 4.007 pp | 6.712 pp |
| ECGFounder | 5.954 pp | 8.588 pp |

## Matched LR attribution control and latent-threechain candidate

The original `2e-5` Direct baseline above remains frozen. A single-variable
ECGFounder control then changed only the learning rate to `3e-5`; the same
`3e-5`, 20-epoch cosine horizon and globally selected epoch 19 were used for
the latent-threechain candidate. Selection continued to use only the pooled
four-center K500-internal validation400 contract.

| ECGFounder arm | Internal clean AUPRC | Internal robust AUPRC | Score | Selected epoch |
|---|---:|---:|---:|---:|
| Direct, LR `2e-5` | 0.820272 | 0.724349 | 0.772310 | 20 |
| Direct, LR `3e-5` | 0.837184 | 0.750341 | 0.793762 | 19 |
| Latent-threechain, LR `3e-5` | 0.838297 | 0.756702 | 0.797499 | 19 |

Held-out four-center all-zero-kept results:

| ECGFounder arm | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC |
|---|---:|---:|
| Direct, LR `2e-5` | `0.889740 / 0.582508` | `0.830200 / 0.496628` |
| Direct, LR `3e-5` | `0.894629 / 0.599474` | `0.836593 / 0.517625` |
| Latent-threechain, LR `3e-5` | `0.893416 / 0.597813` | `0.839047 / 0.523479` |

The latent candidate therefore improves PN2021-C by
`+0.885 AUROC / +2.685 AUPRC pp` over the original locked Direct baseline, but
only by `+0.245 / +0.585 pp` over the learning-rate-matched Direct control. The
latter is the honest LR-matched full method-bundle delta at this single seed;
it does not isolate latent mixing from every other component in the bundle,
and the larger number must not be presented as entirely caused by the method.

The method is frozen in
`configs/train/methods/exp_paired_augmix_latent_bridge_v1.yaml`: two independent
views each use three depth-2/3 corruption chains, deterministic VAE means,
Dirichlet(1) latent mixing, reconstruction-residual bypass, clean BCE, `0.75`
BCE per augmented view and `3x` multilabel Bernoulli JSD. It is a supervised
latent-AugMix candidate, not LHAT and not an exact reproduction of original
AugMix.

Managed selection evidence is stored at:

```text
/home/linbinhao/ECG_adv_data/runs/manual_refactor/
  managed_pn2021_ecgfounder_direct_lr3e5_control_selection/
  managed_pn2021_ecgfounder_latent_threechain_residual_depth23_aug075_lr3e5_selection/
```

The latent refits actually consumed selection SHA `930566547104...`; the
canonical managed re-selection is SHA `f89e04ebf67...`. Their scientific
content, selected E19 and metrics are identical, and the only byte-level field
difference after JSON parsing is `generated_at_utc`. Both identities are
recorded in the machine-readable registry; future replays must use the managed
selection.

Replay is fail-closed against the managed selection snapshots. Direct refit
uses the byte-identical `configs/train/PN2021_fixed20.yaml`; the latent arm uses
`configs/train/PN2021.yaml` and may differ from its tuning snapshot only in the
profile/status and operational diagnostics, logging and output sections. Seed
namespace/determinism, data/normalization/quality gates, loader policy,
`drop_last`, method resources, the complete pooled-selection rule and selected
E*/scheduler contract must remain identical.

The full-K500 direct-invocation artifact directories are:

```text
managed_pn2021_ecgfounder_direct_lr3e5_control_{refit|eval}_<center>/
managed_pn2021_ecgfounder_latent_threechain_residual_depth23_aug075_lr3e5_{refit|eval}_<center>/
```

Exact checkpoint, selection, train-result, refit-contract and evaluation
SHA256 values are in the machine-readable registry. Despite the historical
`managed_` directory prefix, these eight refit and eight evaluation directories
do not contain execution-time `run_manifest.json`, `run_card.json` or
`run_file_index.json`. Their command/Git/environment metadata cannot be
reconstructed honestly after the fact, so this limitation is recorded rather
than backfilled with current state.

## Frozen artifacts

Canonical refit checkpoints are:

```text
/home/linbinhao/ECG_adv_data/runs/manual_refactor/
  managed_pn2021_<effnet|ecgfounder>_fixed20_refit_<center>/training/checkpoints/last.pt
```

Canonical evaluation records are:

```text
/home/linbinhao/ECG_adv_data/runs/manual_refactor/
  managed_pn2021_<effnet|ecgfounder>_fixed20_eval_<center>/evaluation/evaluation_result.json
```

The registry records every checkpoint, train result, refit contract, evaluation
result SHA256, per-center metrics, selection identity, cache identity, source
checkpoint, and exact config hash. Matched VAE/AugMix comparisons must restart
from the same registered PTB-XL source checkpoint—not from an already adapted
Direct K500 checkpoint—and use the same K500/ref-exclusion/evaluation contract.
Direct-initialized training is a separate ablation and must be labeled as such.

## TensorBoard

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/tensorboard \
  --logdir /home/linbinhao/ECG_adv_data/runs/manual_refactor \
  --host 127.0.0.1 --port 9091
```

Tuning directories end in `_h30` for EfficientNet and `_h20` for ECGFounder;
canonical refit directories have no version suffix.

## Limitation

This is a single-seed engineering control, not a significance claim. Historical
exposure-expanded fixed20 results remain in git history but are not the current
comparison baseline and must not be mixed with this family-balanced protocol.
