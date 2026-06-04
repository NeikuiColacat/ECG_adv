# VAE Online AT Closing Summary - 2026-05-25

This note summarizes the latest closing state after roughly 10+ hours of
ECGFounder and EfficientNet1DV2 VAE-only / latent-hull online adversarial
training exploration.

## Scope

- Mapping: PN2021 Super5 v6 clinician-review mapping.
- Main target centers: `ningbo`, `chapman_shaoxing`, `cpsc_2018`, `georgia`.
- Target data policy: fixed K=500 per center, with held-out reference records
  excluded from final target evaluation.
- Final-method integrity rule: exploration may inspect held-out feedback, but
  the paper-safe recipe must not be manually tuned from full target-center label
  distribution or test labels.

## Machine State At Close

- No active user training process was found.
- One user web server remains: `python -m http.server 9005 --bind 0.0.0.0 --directory docs/tmp_html`.
- GPU state at close: GPU 0 had about 15 GB allocated; GPUs 1-7 were essentially free.
- Disk state: about 454 GB free under the shared filesystem.
- Git worktree is dirty with multiple experimental script edits and new selector scripts. Do not treat the tree as clean.

## EfficientNet1DV2 Results

Direct K500 baseline over the four centers:

```text
mean AUROC/AUPRC = 0.854322 / 0.487952
```

Best current standalone EfficientNet VAE-family result:

```text
compatsoft direct-init latent-mixed-teacher
mean AUROC/AUPRC = 0.872177 / 0.514767
gain vs direct K500 = +1.79 pp AUROC / +2.68 pp AUPRC
drop-all-zero mean = 0.901780 / 0.677552
```

Per-center compatsoft actual eval:

```text
ningbo            0.889928 / 0.439661
chapman_shaoxing  0.895674 / 0.394586
cpsc_2018         0.876968 / 0.622776
georgia           0.826136 / 0.602047
```

Source-distillation EfficientNet branch:

```text
mean AUROC/AUPRC = 0.869834 / 0.513323
```

This was worse than the best latent-mixed-teacher / compatsoft branch, so it
should not be used as a main candidate.

EfficientNet K500-validation selector with compatsoft:

```text
global selector    0.871972 / 0.514041
classwise selector 0.870229 / 0.510367
```

The selector did not improve over the best standalone candidate. For
EfficientNet, the clean claim is therefore the standalone VAE/latent-mixed
teacher improvement, not selector-based improvement.

## ECGFounder Results

Frozen/head-style ECGFounder direct K500 baseline used by the selector:

```text
mean AUROC/AUPRC = 0.904164 / 0.596816
```

Matched full-fine-tuning controls and VAE variants:

```text
fullft_direct                  0.900419 / 0.632736
inithead_direct                0.913540 / 0.630500
inithead_vae_aw8               0.913743 / 0.630812
inithead_vae_aw20              0.914771 / 0.630776
inithead_vae_aw12_lam002_strat 0.914737 / 0.632306
```

The important interpretation is:

- Initializing from the K500-trained head plus low-LR full fine-tuning gives
  most of the ECGFounder AUROC gain.
- VAE online AT still adds a small positive delta over matched init-head direct
  fullFT, but the standalone VAE advantage is modest.
- The best new standalone VAE branch is `inithead_vae_aw12_lam002_strat`:
  about `+0.12 pp AUROC / +0.18 pp AUPRC` over `inithead_direct`, and about
  `+1.06 pp AUROC / +3.55 pp AUPRC` over the frozen/head direct K500 baseline.

`inithead_vae_aw12_lam002_strat` actual four-center eval:

```text
ningbo            0.931638 / 0.566358
chapman_shaoxing  0.927347 / 0.491689
cpsc_2018         0.908323 / 0.734211
georgia           0.891663 / 0.735032
mean              0.914743 / 0.631822
drop-all-zero     0.931725 / 0.739128
```

Best ECGFounder selector result after adding the new VAE candidate:

```text
classwise K500-val selector = 0.919757 / 0.643536
gain vs frozen/head direct  = +1.56 pp AUROC / +4.67 pp AUPRC
gain vs inithead_direct     = +0.62 pp AUROC / +1.30 pp AUPRC
```

This is the strongest ECGFounder result from this round. It is close to the
previous target of `+2 pp AUROC / +5 pp AUPRC` versus direct K500, but still
does not fully reach it.

## Negative Or Incomplete Branches

- EfficientNet source-distillation branch did not improve the mean.
- EfficientNet selector did not beat the best standalone VAE-family model.
- ECGFounder simple stronger adversarial weight sweeps did not create a large
  net VAE advantage over matched direct fullFT.
- ECGFounder full encoder fine-tuning without careful init can improve source
  metrics but hurt target-center performance.
- ECGFounder `abncompat_aw16_lam005` branch is incomplete at close: training
  logs exist, but full `eval_result.json` files were not produced for all
  centers. Do not cite it.

## Current Scientific Interpretation

For EfficientNet1DV2, the VAE/latent-hull online AT family is clearly doing
more than direct K500 fine-tuning: current best gain is about
`+1.79 pp AUROC / +2.68 pp AUPRC`.

For ECGFounder, the story is more nuanced. Most of the improvement over the
frozen/head baseline comes from better target-center K500 supervised adaptation
and head initialization. VAE online AT adds a smaller standalone gain, but it
does provide useful diversity for a K500-internal classwise selector. The best
selector result reaches `0.919757 / 0.643536`, which is the current strongest
ECGFounder number.

## Recommended Stop Point

Stop additional broad sweeps for now. The next useful work, if resumed later,
should be narrow:

1. Implement and test ECGFounder partial-unfreeze with matched direct and VAE
   controls.
2. Clean up the selector scripts and produce paper-ready tables.
3. Add attack-strength diagnostics (`atk_init`, `atk_anchor`, `loss_gain`,
   ASR) to the final cited runs.
4. Keep fixed K=500 as the main protocol; use 10% target-center data only as a
   sensitivity analysis.
