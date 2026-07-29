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

## Current effective EfficientNet method

Status: `LOCKED_CURRENT_DEVELOPMENT_SINGLE_SEED` on 2026-07-30.

The current completed winner is L37:

```text
locked PTB-XL EfficientNet1DV2
-> 1024 K500-only two-chain AugMix-SimCLR updates
   + PTB-XL source replay 0.30
   + source-logit anchor 5.0
-> 23 epochs clean + rotating-four supervised adaptation
   + two depth-2 and two depth-3 views per epoch
   + all 20 compositions covered every five epochs
-> one exact-label VAE-LHAT hard view
   + hull lambda 0.6
   + standardized latent L2 epsilon 2.0
   + 5 search steps at learning rate 0.25
   + 1.5 * (0.5 hard BCE + 0.5 clean/hard Bernoulli JSD)
```

Stage 2 keeps the locked EfficientNet optimizer recipe: AdamW, learning rate
`5e-5`, weight decay `1e-4`, batch size 128, bfloat16, epoch 23 under a
30-epoch cosine horizon. Target adaptation consumes only the selected center's
K500. No target-center record outside K500 enters the model; Stage-1 replay is
PTB-XL source data.

The primary view keeps all-zero Super5 records, excludes K500 references,
combines CPSC 2018 and Extra as one logical center, averages 20 PN2021-C views
inside each center, and then averages four centers equally.

| Arm | Clean AUROC | Clean AUPRC | PN2021-C AUROC | PN2021-C AUPRC |
|---|---:|---:|---:|---:|
| Locked Direct+fixed20 | 0.850009 | 0.505380 | 0.809939 | 0.438257 |
| L36, matched pipeline without VAE-LHAT | 0.855507 | 0.518518 | 0.839711 | 0.491366 |
| L37, L36 + VAE-LHAT | 0.858001 | 0.523483 | 0.841685 | 0.494762 |
| L37 minus Direct+fixed20 (pp) | +0.799 | +1.810 | +3.175 | +5.651 |
| L37 minus L36 (pp) | +0.249 | +0.496 | +0.197 | +0.340 |

The full pipeline has the strongest completed EfficientNet development result.
The strictly matched L37-minus-L36 comparison also shows a positive VAE-LHAT
increment in all four centers. Most of the total gain over Direct+fixed20,
however, comes from the AugMix-SimCLR, source-replay and rotating-corruption
base pipeline; only the final matched increment may be attributed to VAE-LHAT.

This is deliberately a development lock, not a paper-final statistical claim.
The family was chosen using held-out PN2021/PN2021-C feedback, only one seed is
complete, and the predeclared absolute PTB-XL source-retention rule fails:
L37 gives `0.880577 / 0.732554` on fold10 versus
`0.901281 / 0.767951` for the locked source checkpoint. Paper-final use
requires an independently seeded replay of this exact frozen recipe without
more held-out tuning.

The executable source and generated config closure are frozen by implementation
commit `d27d5941cdd03db1ef639d46ef5aa4b426f18c63`. Exact checkpoint,
train-manifest, evaluation and summary SHA256 identities are recorded under
`current_effective_method` in the machine-readable registry. The implementation
remains a frozen sandbox replay surface pending a small whitelist extraction;
the lock does not silently promote the large search controller into the clean
runtime architecture.

## Current ECGFounder VAE-LHAT mechanism candidate

Status: `LOCKED_TWO_SEED_MECHANISM_POSITIVE_SOURCE_FLOOR_FAILED` on
2026-07-30.

The compact ECGFounder candidate uses the same main mechanism as L37, adapted
to the locked ECGFounder training budget:

```text
locked PTB-XL ECGFounder
-> 1024 K500-only two-chain AugMix-SimCLR updates
   + PTB-XL source replay 0.30
   + source-logit anchor 5.0
-> 40 epochs clean + rotating-four supervised adaptation
   + AdamW, LR 2e-5, WD 1e-4, batch 64, cosine T40
-> one exact-label VAE-LHAT hard view
   + M=20 from a local pool of 80
   + hull lambda 0.6
   + standardized latent L2 epsilon 2.0
   + 5 search steps at learning rate 0.25
   + 1.5 * (0.5 hard BCE + 0.5 clean/hard Bernoulli JSD)
```

The matched no-VAE arm removes only the final VAE-LHAT auxiliary. Both arms
retain identical source initialization, K500 records, Stage-1 target/source
orders, 320 Stage-2 optimizer updates, corruption traces, optimizer and fixed
last-E40 checkpoint rule. The independent seed-1 replay did not use a held-out
epoch oracle.

| Replay | Clean AUROC | Clean AUPRC | PN2021-C AUROC | PN2021-C AUPRC |
|---|---:|---:|---:|---:|
| Seed 0, L35 minus L34 (pp) | +0.068 | +1.019 | -0.036 | +0.474 |
| Seed 1, VAE-LHAT minus no-VAE (pp) | +0.144 | +0.635 | +0.045 | +0.367 |
| Two-seed mean delta (pp) | +0.106 | +0.827 | +0.005 | +0.420 |

Clean and PN2021-C AUPRC increase in all four centers in both seeds. This is
the current strongest evidence that the online VAE-LHAT branch itself is
useful rather than merely inheriting gain from AugMix-SimCLR or rotating
corruption supervision. The mean AUROC increment is effectively zero, so the
supported mechanism claim is an AUPRC refinement, not a general one-point
improvement in every metric.

The source-domain result prevents paper-final promotion:

| Replay | No-VAE PTB-XL AUROC/AUPRC | VAE-LHAT PTB-XL AUROC/AUPRC | VAE minus no-VAE |
|---|---:|---:|---:|
| Seed 0 | `0.915372 / 0.795770` | `0.908378 / 0.780632` | `-0.699 / -1.514 pp` |
| Seed 1 | `0.915389 / 0.795914` | `0.908806 / 0.780515` | `-0.658 / -1.540 pp` |

The locked ECGFounder source is `0.929407 / 0.824164`; therefore both shared
scaffolds already fail the absolute source floor, and VAE-LHAT adds further
forgetting. This lock means “target-side mechanism-positive and reproducibly
directional,” not “no-trade-off final method.”

Increasing only the VAE auxiliary coefficient from `1.5` to `2.0` was also
rejected. It reached `+0.966 pp` Clean AUPRC and `+0.549 pp` PN2021-C AUPRC
against the seed-1 no-VAE arm, but fell to `0.906104 / 0.774686` on PTB-XL,
which is another `-0.270 / -0.583 pp` below the alpha-1.5 arm. The current
frozen coefficient therefore remains `1.5`.

Exact config, target summaries and source-floor SHA256 identities are recorded
under `current_ecgfounder_mechanism_candidate` in the machine-readable
registry. The replay/search controllers remain isolated sandbox surfaces and
are not promoted into the whitelist runtime by this evidence lock.

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
