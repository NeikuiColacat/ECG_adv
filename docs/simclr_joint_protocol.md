# Single-stage versus two-stage SimCLR, no VAE

User requested a goal, independent random seeds and GPU 2 on 2026-11-04.
This independent worktree preserves the locked direction and all old results.

Frozen grid: joint / two_stage, seeds 0/1/2, EfficientNet1DV2 / ECGFounder,
Ningbo / Chapman-Shaoxing / CPSC2018(+Extra) / Georgia. 48 training and 48
full evaluations, serially on GPU 2. Fixed source checkpoints and center K500;
last checkpoint; no adaptation outside K500.

Joint objective from first to last update:
0.5 clean BCE + 0.5 mean(rotating four corruption BCE) + 1.0 NT-Xent +
5.0 frozen-source soft-target BCE. Temperature 0.5, two-chain AugMix alpha0.5.
All backbone/head/projector parameters share the supervised AdamW profile.
The auxiliary contrastive forwards restore BatchNorm buffers and RNG after
backward. The source teacher is cached on the selected K500 only. VAE disabled.

Both arms use EffNet 25 epochs/100 supervised steps, Founder 30 epochs/240
supervised steps. The reference additionally uses the locked 1024/256 Stage-1
steps. Joint has seven student forwards per base batch; reference has two per
Stage-1 batch and five per supervised batch. This is explicitly NOT matched
total compute, contrastive exposure or an isolated causal estimate of stage
boundaries. No hyperparameter or checkpoint selection from heldout results.

Primary: PN2021-C macro sklearn average precision (AUPRC field), drop_all_zero,
20 depth2/3 views then four centers equally. Also report clean AP/AUROC and
corrupt AUROC, per-center deltas, three seed mean/sample SD, paired seed deltas
and exploratory t intervals. Practical retention reference: mean clean and
corrupt AP both within 1 pp of reference on each backbone. This is a descriptive
engineering criterion, not a statistical non-inferiority claim or paper-final
test. The fixed centers and source recipes have historical development tuning.

Output root: `/home/linbinhao/ECG_adv_data/runs/simclr_joint_novae_20261104_r0`.
Managed entry: `configs/experiments/simclr_joint/workflow_online_gpu1.yaml`.
The coordinator is CPU-only with `SIMCLR_JOINT_GPU=2`, one worker, source guards,
GPU/RAM/load/disk admission, exact child receipts and refusal to overwrite.
It drains an active child on SIGTERM or when control is paused; no foreign
process is stopped. Frozen configs/results/checkpoints are kept external.

The original cache was retired on 2026-10-17. The user explicitly requested
online corruption and then the existing GPU operator implementation. No new
waveform cache is built. Four completed training children from the drained
`workflow_recovery1` are retained with their file/config/checkpoint hashes.
Training functions and parameters are unchanged. The new queue evaluates and
trains serially through the same launcher on GPU 2.

Evaluation profile: `configs/eval/PN2021_online_gpu.yaml`. Existing Torch
operators, both linear resampling steps, global z-score and model inference
execute on GPU. Clean source waveforms retain their read-only mmap; corruption
uses raw100 -> GPU500 -> GPU100. The original 20 compositions, severity5
parameters, labels, cohorts and K500 exclusion remain fixed. All ten historical
corruption metadata files are verified against the original content ledger;
the deleted waveform is neither required nor certified.

GPU RNG contract `pn2021c_evaluation_gpu_microbatch32_v1` uses fixed 32-record
microbatches and ordered record hashes plus composition/cache-version identity.
It excludes model, arm, training seed and checkpoint. Model batches 128/512
therefore generate identical per-record views. CUDA RNG differs from NumPy;
these are versioned GPU-online results, not byte-identical historical cache
replays. Both comparison arms use this same new evaluation backend; do not
subtract historical NumPy-cache baselines as a matched causal comparison.

Completed readout, 2026-11-04: all 48 training and 48 evaluation children
completed on GPU 2. Three independent seeds, four equally weighted centers,
mean +/- sample SD, percentage scale:

| Backbone | Arm | Clean AUROC | Clean AP | PN2021-C AUROC | PN2021-C AP |
| --- | --- | --- | --- | --- | --- |
| EfficientNet1DV2 | Two-stage, no VAE | 87.44 +/- 0.02 | 65.09 +/- 0.03 | 86.13 +/- 0.08 | 62.70 +/- 0.04 |
| EfficientNet1DV2 | Joint, no VAE | 86.40 +/- 0.02 | 62.60 +/- 0.02 | 81.91 +/- 0.01 | 55.03 +/- 0.02 |
| ECGFounder | Two-stage, no VAE | 91.49 +/- 0.03 | 71.60 +/- 0.12 | 89.38 +/- 0.13 | 67.89 +/- 0.14 |
| ECGFounder | Joint, no VAE | 89.63 +/- 0.02 | 66.64 +/- 0.03 | 85.59 +/- 0.05 | 60.80 +/- 0.05 |

Joint minus two-stage clean/corrupt AP: EfficientNet -2.49/-7.67 pp;
ECGFounder -4.96/-7.09 pp. All four centers and all three seed means decline
on both backbones. The frozen 1 pp retention criterion fails for both models.
This rejects this particular joint recipe/budget as the proposed simplification;
it does not establish that single-stage training is inherently inferior.
The reference is the matched no-VAE two-stage arm, not clean-only fine-tuning.

The external `workflow_online_gpu1/training/comparison.json` contains all 48
center/seed/arm rows, paired deltas and exploratory intervals; its SHA256 is
`a44285fc5ad62adb5b4ced7acfd0198be75260b83ab4b0c276c8e00acbba274b`.
The external `completion_audit.json` independently verifies all 96 child
manifests/file indexes, all 48 checkpoint bytes and 26,016 finite state tensors,
K500 exclusions, paired source/seed/cohort identities, complete 20-view
evaluations and the per-class/view/center/seed metric calculations.

The parent launcher originally raised `KeyError: 5` while projecting the new
queue summary after all work had completed. A post-execution fix registers
schema 5 in the existing run-record owner. The original runtime snapshot,
traceback and pre-recovery parent records are preserved; the parent record is
finalized separately with an explicit recovery receipt. No child or experiment
result is rerun or rewritten by this bookkeeping repair.
