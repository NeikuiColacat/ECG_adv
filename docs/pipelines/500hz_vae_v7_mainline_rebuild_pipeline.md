# 500Hz VAE V7 Mainline Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a paper-safe 500Hz branch: archive cold artifacts to free disk, train a PTB-XL-only 500Hz VAE adapted from DiffuSETS, then rerun v7 Super5 cross-center experiments for EfficientNet1DV2, ECGFounder, and selected `model/ecg_ptbxl_benchmarking` backbones.

**Architecture:** Keep the existing 100Hz ECGTwin VAE / EfficientNet1DV2 result as the stable baseline. Add a separate 500Hz branch with explicit run records, v7 label mapping, PTB-XL fold isolation, PN2021 ref-exclusion, and fair direct-K500 versus VAE-online-AT comparisons. Treat DiffuSETS as a reference implementation; do not edit the external repo directly.

**Tech Stack:** PyTorch, WFDB, NumPy mmap, DiffuSETS VAE reference code, ECG_adv_Gen managed YAML launcher, `zstd`/`tar` for cold archive compression, PTB-XL records500, PN2021 v7 Super5 mapping.

---

## Scope And Non-Negotiable Rules

- Main label mapping: `v7_super5_sjr_rgq_review_20260528`.
- Super5 class order: `CD, HYP, MI, NORM, STTC`.
- Main target centers: `ningbo`, `chapman_shaoxing`, `cpsc_2018`, `georgia`.
- K-shot protocol: fixed `K=500` first; K records must be excluded from final target-center evaluation.
- VAE training data for the clean main claim: PTB-XL folds 1-8 only. Fold 9 is validation. Fold 10 is reconstruction audit and classifier test only.
- The 500Hz EfficientNet branch means the project's existing `EfficientNet1DV2`
  ECG model adapted to `500Hz / 10s / 5000 samples`, not a new image-style 2D
  EfficientNet unless a later plan explicitly adds one.
- PN2021 target-center samples are allowed only as K500 adaptation anchors, not for VAE training, unless a separate external-data ablation is explicitly named.
- Do not write large checkpoints, caches, archives, or generated samples into git.
- Do not overwrite existing experiment directories. Use date/config-named run roots.
- Before GPU jobs, run `nvidia-smi`, choose explicit `CUDA_VISIBLE_DEVICES`, and start with single-GPU smoke runs.

## User-Confirmed Decisions, 2026-05-31

These decisions are no longer open questions for the first execution pass:

1. Use the existing project `EfficientNet1DV2` architecture and adapt it to
   `500Hz / 5000 samples`.
2. After a cold artifact archive passes manifest, checksum, and `tar -tf`
   verification, the original archived directory may be deleted to free space.
3. The clean 500Hz VAE must be trained only on PTB-XL. Do not introduce
   PN2021, MIMIC, SPH, or other datasets into VAE training, so downstream gains
   can be attributed to online adversarial adaptation rather than external VAE
   pretraining data.
4. Run five recommended `model/ecg_ptbxl_benchmarking` backbones after smoke
   validation.
5. Report both PN2021 views: `all-zero-kept` and `drop-all-zero`.

## Current Local Facts To Recheck At Execution Start

Observed on 2026-05-31:

```text
repo root: /root/autodl-tmp/ECG_adv_Gen
data root: /root/autodl-tmp
free disk: about 127G on /root/autodl-tmp
compression tools: zstd, pigz, xz, tar are installed
DiffuSETS reference: model/DiffuSETS -> /root/autodl-tmp/models/DiffuSETS
DiffuSETS Exp reference: /root/autodl-tmp/models/DiffuSETS_Exp
```

If the host is the migrated `/home/linbinhao` machine, use `configs/local/linbinhao_server.example.yaml` as the path template and never assume `/root/autodl-tmp`.

## Phase 1: Cold Artifact Compression And Disk Recovery

**Objective:** Free enough space for 500Hz VAE caches, checkpoints, and multi-backbone experiments without deleting raw PTB-XL/PN2021/MIMIC data or current evidence by accident.

**Target Outcome:**

- `/root/autodl-tmp` free space target: at least `180G` before the 500Hz VAE training phase.
- Every archived directory has:
  - `.tar.zst` archive
  - `.sha256`
  - file count and byte manifest
  - successful `tar -tf` listing check
  - successful `sha256sum -c` verification
  - original directory automatically removed after archive verification, unless
    it is referenced by the active evidence registry or active script index

**Compression Policy:**

- Use `zstd -T0 -19 --long=31` as the default high-ratio multi-core compressor.
- Use `zstd --ultra -22` only for archives below `10G` or after user confirmation; it is slower and more memory-heavy.
- Do not compress already-compressed `.npz`, `.pt`, `.pth`, `.tar.gz`, or `.zip` blindly. First sample-test compression ratio. If ratio is poor, either keep as-is or delete regenerable cache after confirmation.
- Goal-mode deletion policy: after manifest, `.sha256`, `sha256sum -c`, and
  `tar -tf` all pass, delete the original cold directory automatically and
  record the deletion in `docs/tmp_md/cold_archive_manifest_20260531.md`.

**Candidate Buckets:**

| Bucket | Path examples | Default action | Reason |
|---|---|---|---|
| raw data | `/root/autodl-tmp/ptbxl`, `/root/autodl-tmp/physionet2021`, `/root/autodl-tmp/MIMIC` | keep | expensive to restore; needed for future branches |
| active evidence | `/root/autodl-tmp/runs`, current v7 run roots, registry-linked artifacts | keep | needed for paper traceability |
| regenerable PN2021-C cache | `/root/autodl-tmp/triple_labels/pn2021_c_cache_minresample_perglobal` | manifest, ratio-test, then archive-or-delete | 62G cache can be regenerated; often already compressed |
| old graduation artifacts | `/root/autodl-tmp/graduate_project`, `/root/autodl-tmp/streamlit_ecg_demo` | archive then delete original | useful history, not active paper branch |
| old center-token pools | `/root/autodl-tmp/ecgtwin_prompt_token_super5`, `/root/autodl-tmp/center_token_ablation` | archive selected reports/configs; archive full pool only if space permits | not current causal mainline |
| old sweep outputs | `/root/autodl-tmp/paper_latenthull_grid_20260512`, `/root/autodl-tmp/paper_real_anchor_lhat_seed_kfold_20260517`, similar | archive then delete original if not active registry input | old evidence, large enough to save space |
| migration archives | `/root/autodl-tmp/transfer_artifacts/*.tar.gz` | keep until user confirms downloaded | already compressed; deleting gives direct space |

**Execution Steps:**

- [x] **Step 1: Snapshot disk and archive candidates**

Run:

```bash
df -h /root/autodl-tmp /
du -sh /root/autodl-tmp/* 2>/dev/null | sort -hr | sed -n '1,120p'
find /root/autodl-tmp -maxdepth 2 -type f \( -name '*.tar.gz' -o -name '*.tar.zst' -o -name '*.zip' \) -printf '%s %p\n' 2>/dev/null | sort -nr | head -50
```

Expected: identify cold directories whose removal will not break current v7 evidence registry.

- [x] **Step 2: Create archive root**

Run:

```bash
mkdir -p /root/autodl-tmp/archive_cold_20260531/manifests
```

Expected: all archive outputs stay on data disk.

- [x] **Step 3: For each candidate directory, write a manifest**

Example for `graduate_project`:

```bash
target=/root/autodl-tmp/graduate_project
name=graduate_project
find "$target" -xdev -type f -printf '%s\t%p\n' | sort > /root/autodl-tmp/archive_cold_20260531/manifests/${name}.files.tsv
du -sh "$target" > /root/autodl-tmp/archive_cold_20260531/manifests/${name}.du.txt
```

Expected: restore metadata exists before compression.

- [x] **Step 4: Compress with multi-core zstd**

Example:

```bash
target=/root/autodl-tmp/graduate_project
name=graduate_project
tar --xattrs --acls -I 'zstd -T0 -19 --long=31' \
  -cf /root/autodl-tmp/archive_cold_20260531/${name}.tar.zst \
  -C /root/autodl-tmp "$(basename "$target")"
sha256sum /root/autodl-tmp/archive_cold_20260531/${name}.tar.zst \
  > /root/autodl-tmp/archive_cold_20260531/${name}.tar.zst.sha256
sha256sum -c /root/autodl-tmp/archive_cold_20260531/${name}.tar.zst.sha256
tar -tf /root/autodl-tmp/archive_cold_20260531/${name}.tar.zst >/dev/null
```

Expected: archive is readable and has a checksum.

- [x] **Step 5: Auto-remove original after verification**

Run only after confirming the target is not referenced by active evidence files:

```bash
target=/root/autodl-tmp/graduate_project
name=graduate_project
archive=/root/autodl-tmp/archive_cold_20260531/${name}.tar.zst

sha256sum -c ${archive}.sha256
tar -tf "$archive" >/dev/null

if grep -R --fixed-strings "$target" configs/active_evidence_registry.yaml configs/active_scripts.yaml >/dev/null 2>&1; then
  echo "[keep] $target is referenced by active evidence/script config"
  exit 2
fi

rm -rf --one-file-system "$target"
df -h /root/autodl-tmp
```

Expected: free space increases by roughly original size minus archive size.

- [x] **Step 6: Record archive summary**

Create or update:

```text
docs/tmp_md/cold_archive_manifest_20260531.md
```

Expected: it lists archived paths, original sizes, archive sizes, checksums, and deleted originals.

Execution update 2026-06-01:

- Wrote `docs/tmp_md/cold_archive_manifest_20260531.md`.
- Archived `/root/autodl-tmp/graduate_project` to
  `/root/autodl-tmp/archive_cold_20260531/graduate_project.tar.zst`;
  `sha256sum -c` and `tar -I 'zstd -d --long=31' -tf` passed before deleting
  the original.
- Deleted regenerable
  `/root/autodl-tmp/triple_labels/pn2021_c_cache_minresample_perglobal` after
  writing a file manifest and confirming a 128MiB zstd sample ratio of `1.000`.
- `/root/autodl-tmp` free space reached about `191G` before full cache build,
  satisfying the `>=180G` pre-VAE target.

**Stop Conditions:**

- Stop and keep the original if any path is referenced by
  `configs/active_evidence_registry.yaml` or `configs/active_scripts.yaml`.
- Stop if free space drops below `40G` during archive creation.
- Stop if `tar -tf` or `sha256sum -c` fails. Do not delete the original.

## Phase 2: PTB-XL 500Hz VAE Dataset Contract

**Objective:** Build a clean `records500` VAE training dataset from PTB-XL with no fold leakage and explicit preprocessing metadata.

**Target Outcome:**

- Train cache: PTB-XL folds 1-8, shape `(N_train, 5000, 12)`, raw mV, PTB-XL canonical lead order.
- Val cache: fold 9.
- Audit cache: fold 10.
- Manifest records patient/record IDs, fold, labels, unit, lead order, sampling rate, and normalization policy.

**Recommended Input Protocol:**

```text
PTB-XL records500
-> read WFDB physical mV
-> canonical 12-lead order
-> NaN/Inf guard
-> exact 10s = 5000 samples
-> no default notch/bandpass for VAE v1
-> store raw_mV and normalized view separately
```

**Normalization Decision:**

- VAE training v1 should use per-sample global z-score for numerical stability, while preserving raw mV statistics in metadata for reconstruction audit.
- EfficientNet1DV2-500Hz should start with per-sample global z-score to match the current v7 EfficientNet logic.
- `model/ecg_ptbxl_benchmarking` baseline should also run its official dataset-level `StandardScaler` branch, because that is the reference repo protocol.
- ECGFounder should use its official 500Hz preprocessing branch for ECGFounder-specific runs.

**Execution Steps:**

- [x] **Step 1: Add a VAE500 data-prep script**

Create:

```text
scripts/vae500/prepare_ptbxl_500hz_vae_cache.py
```

Responsibilities:

- read PTB-XL metadata and `filename_hr`;
- split folds 1-8/9/10;
- save memmap or shard files under `/root/autodl-tmp/vae500/ptbxl_records500_v7/cache_v1`;
- write `dataset_manifest.json`.

- [x] **Step 2: Smoke-build a tiny cache**

Run:

```bash
/root/miniforge3/envs/ECGTwin/bin/python -u scripts/vae500/prepare_ptbxl_500hz_vae_cache.py \
  --ptbxl_root /root/autodl-tmp/ptbxl \
  --output_dir /root/autodl-tmp/vae500/ptbxl_records500_v7/cache_smoke \
  --limit_per_fold 8 \
  --sampling_rate 500 \
  --length 5000
```

Expected: manifest shows train/val/test folds separated and tensors are finite.

- [x] **Step 3: Full cache build**

Run after smoke passes:

```bash
/root/miniforge3/envs/ECGTwin/bin/python -u scripts/vae500/prepare_ptbxl_500hz_vae_cache.py \
  --ptbxl_root /root/autodl-tmp/ptbxl \
  --output_dir /root/autodl-tmp/vae500/ptbxl_records500_v7/cache_v1 \
  --sampling_rate 500 \
  --length 5000
```

Expected: full cache is under data disk and not tracked by git.

Execution update 2026-06-01:

- Added `scripts/vae500/prepare_ptbxl_500hz_vae_cache.py`.
- Smoke cache passed at
  `/root/autodl-tmp/vae500/ptbxl_records500_v7/cache_smoke`.
- Full cache passed at `/root/autodl-tmp/vae500/ptbxl_records500_v7/cache_v1`.
  Shapes are train `(17418, 5000, 12)`, val `(2183, 5000, 12)`, audit
  `(2198, 5000, 12)`, with no source sampling-rate or length mismatches.

## Phase 3: DiffuSETS-Derived 500Hz VAE

**Objective:** Implement a repo-owned 500Hz VAE that reuses DiffuSETS design ideas but does not depend on editing external model code.

**Target Outcome:**

- A trainable VAE accepts `(B, 5000, 12)` and reconstructs `(B, 5000, 12)`.
- Encoder returns `z`, `mu`, `log_var`.
- Decoder accepts `z` and returns ECG.
- It supports deterministic encode/decode for latent-hull online AT.

**Architecture Choice:**

Start with two model variants:

| Variant | Latent shape | Purpose | Decision rule |
|---|---:|---|---|
| `diffusets500_v1_dynamic` | `(B, 4, 625)` | Minimal change from DiffuSETS 1024-point VAE; same 3 downsample stages | Run first because it is closest to cloned code |
| `diffusets500_v2_compact` | `(B, 8, 313)` | Add a fourth downsample and a wider latent channel count | Run only if v1 is too slow or overfits/under-regularizes |

**Loss v1:**

```text
loss = Huber reconstruction
     + beta * KL
     + 0.05 * first-difference Huber
     + 0.02 * lead-consistency residual
```

Do not add a diffusion model in this phase. The goal is a VAE latent manifold for online AT, not a new full generator.

**Execution Steps:**

- [x] **Step 1: Create model and loss files**

Create:

```text
ecg_adv_gen/vae/diffusets_vae500.py
ecg_adv_gen/vae/losses.py
```

Expected: no import from external `model/DiffuSETS` at runtime; external repo remains reference-only.

- [x] **Step 2: Add shape tests**

Create:

```text
util/tests/test_vae500_shapes.py
```

Test cases:

- `diffusets500_v1_dynamic` encodes `(2, 5000, 12)` into `(2, 4, 625)`;
- decoder returns `(2, 5000, 12)`;
- loss is finite;
- deterministic decode with `noise=0` is stable.

Run:

```bash
/root/miniforge3/envs/ECGTwin/bin/python -m pytest util/tests/test_vae500_shapes.py -q
```

Expected: all tests pass on CPU.

- [x] **Step 3: Add trainer**

Create:

```text
scripts/vae500/train_ptbxl_vae500.py
```

Expected CLI:

```bash
/root/miniforge3/envs/ECGTwin/bin/python -u scripts/vae500/train_ptbxl_vae500.py \
  --cache_dir /root/autodl-tmp/vae500/ptbxl_records500_v7/cache_v1 \
  --output_dir /root/autodl-tmp/vae500/ptbxl_records500_v7/diffusets500_v1_20260531 \
  --model_variant diffusets500_v1_dynamic \
  --epochs 80 \
  --batch_size 16 \
  --lr 1e-4 \
  --kl_warmup_epochs 20 \
  --num_workers 4 \
  --amp \
  --device cuda
```

Expected: `run_config.json`, `metrics.jsonl`, `best.pt`, `latest.pt`, reconstruction samples, and run record.

- [x] **Step 4: Add reconstruction audit**

Create:

```text
scripts/vae500/audit_vae500_reconstruction.py
```

Expected metrics:

- reconstruction MAE/MSE by lead;
- median Pearson correlation by lead;
- first-difference error;
- amplitude p2p distribution drift;
- Einthoven/aVR residual drift;
- invalid output rate: NaN/Inf, flatline, extreme amplitude.

Acceptance for first downstream use:

```text
hard pass before online AT:
  fold9 invalid decode rate < 1%
  fold10 invalid decode rate < 1%
  fold9 median global Pearson correlation >= 0.95
  fold10 median global Pearson correlation >= 0.93
  median lead Pearson correlation >= 0.90 for at least 10/12 leads
  no lead has median Pearson correlation < 0.80
  median first-difference Pearson correlation >= 0.80
  median reconstructed/original p2p amplitude ratio in [0.80, 1.25]
  p5-p95 reconstructed/original p2p amplitude ratio inside [0.50, 2.00]
  lead-consistency residual is not worse than input by more than 20%
  fold10 random visual audit of at least 32 ECGs does not show flatline,
    extreme smoothing, QRS collapse, or obvious lead-order corruption

warning band:
  fold9 global correlation in [0.90, 0.95), or
  fold10 global correlation in [0.88, 0.93), or
  first-difference correlation in [0.70, 0.80)
  -> do not run downstream AT yet; first try more epochs, lower beta/KL,
     or `diffusets500_v2_compact`.

hard fail:
  invalid decode rate >= 1%, NaN/Inf loss, repeated OOM, median global
  correlation < 0.90 on fold9, or visual QRS/lead collapse.
  -> stop the goal and report VAE failure instead of launching classifier AT.
```

Expected training behavior:

```text
first 5-10 epochs:
  reconstruction loss should decrease monotonically on train and fold9 in most
  epochs; no exploding KL or all-zero latent posterior.

by selected checkpoint:
  validation reconstruction should be stable for at least two consecutive
  evaluations; best checkpoint is selected by fold9 reconstruction/audit score,
  never by PN2021 downstream performance.
```

Execution update 2026-06-01:

- Added `ecg_adv_gen/vae/diffusets_vae500.py`,
  `ecg_adv_gen/vae/losses.py`, `scripts/vae500/train_ptbxl_vae500.py`,
  `scripts/vae500/audit_vae500_reconstruction.py`, and
  `util/tests/test_vae500_shapes.py`.
- CPU shape tests passed: `3 passed`.
- GPU smoke run passed at
  `/root/autodl-tmp/vae500/ptbxl_records500_v7/diffusets500_v1_smoke_20260601`.
  This smoke intentionally used tiny data/model settings and is not a VAE
  quality result.
- Full-architecture memory smoke also passed at
  `/root/autodl-tmp/vae500/ptbxl_records500_v7/diffusets500_v1_memsmoke_b16_20260601`
  with `base_channels=64`, attention enabled, `batch_size=16`, bf16 AMP, and
  one mini epoch over 16 train / 8 val records.
- Full 80-epoch optimized training completed at
  `/root/autodl-tmp/vae500/ptbxl_records500_v7/diffusets500_v1_fast_b64_20260601`.
  This used train/val cache in RAM, `batch_size=64`, 8 DataLoader workers,
  `prefetch_factor=4`, bf16 AMP, TF32, and fused AdamW. GPU utilization was
  observed around 99-100% during steady-state training.
- `torch.compile(mode=reduce-overhead)` was smoke-tested but disabled for this
  VAE, because PyTorch 2.1.1 Dynamo failed on the dynamic interpolation path.
- Initial full run failed the strict fold10 lead-consistency residual gate, so
  it was not accepted for downstream online AT.
- Lead-consistency fine-tuning with initialized checkpoints produced the
  accepted VAE:
  `/root/autodl-tmp/vae500/ptbxl_records500_v7/diffusets500_v1_fast_b64_lc025_20260601/checkpoints/best.pt`.
  Numeric reconstruction audit passed:
  fold9 invalid `0.0`, global Pearson `0.9976`, first-diff Pearson `0.9494`,
  12/12 leads with median Pearson `>=0.90`, p2p median `0.9925`, lead residual
  ratio `1.0482`; fold10 invalid `0.0`, global Pearson `0.9977`,
  first-diff Pearson `0.9505`, 12/12 leads `>=0.90`, p2p median `0.9929`,
  lead residual ratio `1.0883`.
- Fold10 random visual audit of 32 overlays passed. The manifest is:
  `/root/autodl-tmp/vae500/ptbxl_records500_v7/diffusets500_v1_fast_b64_lc025_20260601/audit/visual_audit_32/visual_audit_manifest.json`.

## Phase 4: 500Hz EfficientNet1DV2 Baseline

**Objective:** Train a 500Hz version of the current EfficientNet1DV2 Super5 classifier and evaluate PTB-XL fold10 plus PN2021 v7 ref-excluded centers.

**Target Outcome:**

- Direct 500Hz EfficientNet1DV2 baseline trained on PTB-XL folds 1-8.
- PTB-XL fold10 AUROC/AUPRC and per-class metrics.
- PN2021 four-center and seven-center v7 metrics with ref-exclusion support.

**Implementation Choices:**

- Keep the current 100Hz `scripts/triple_labels/train_ptbxl.py` intact.
- Add 500Hz support behind explicit args:
  - `--sampling_rate 500`
  - `--input_len 5000`
  - `--crop_len 5000`
  - `--cache_path ...records500...`
- Classifier training should use the same acceleration defaults as the accepted
  VAE run where safe: RAM-resident NumPy splits, `pin_memory`,
  `persistent_workers`, `prefetch_factor=4`, bf16 AMP, TF32, fused AdamW, and
  `cudnn.benchmark=True`.
- Start the 500Hz EfficientNet smoke/full run with `batch_size=64` on a 24GB
  4090-class GPU; if OOM occurs, retry `32`, then `16`. Do not keep a small
  batch by default if GPU utilization is low.

**Execution Steps:**

- [ ] **Step 1: Add 500Hz cache/preprocess branch**

Modify:

```text
scripts/triple_labels/train_ptbxl.py
scripts/triple_labels/eval_crosscenter.py
scripts/crosscenter_v2/preprocess_utils.py
```

Expected: 100Hz behavior remains unchanged; 500Hz requires explicit CLI args.

Execution update 2026-06-01:

- Added explicit `--sampling_rate` and `--input_len` arguments to
  `scripts/triple_labels/train_ptbxl.py` and
  `scripts/triple_labels/eval_crosscenter.py`.
- 500Hz PTB-XL training cache now reads `filename_hr` records500 instead of
  padding 100Hz arrays to 5000 samples.
- `scripts/crosscenter_v2/preprocess_utils.py` already accepted
  `target_fs/target_len`; no functional edit was needed there. Keep this step
  open until a 500Hz EfficientNet smoke verifies the full classifier path.

- [x] **Step 2: Train 500Hz baseline**

Run:

```bash
CUDA_VISIBLE_DEVICES=<free_gpu> TMPDIR=/root/autodl-tmp/tmp XDG_CACHE_HOME=/root/autodl-tmp/cache \
/root/miniforge3/envs/ECGTwin/bin/python -u scripts/triple_labels/train_ptbxl.py \
  --scheme super5 \
  --output_dir /root/autodl-tmp/triple_labels/super5_v7_effnet1dv2_500hz_full10_20260531 \
  --cache_path /root/autodl-tmp/triple_labels/cache/ptbxl_super5_v7_records500_perglobal_len5000.npy \
  --preprocess_mode minimal_resample \
  --norm_mode per_sample_global \
  --sampling_rate 500 \
  --crop_len 5000 \
  --input_len 5000 \
  --batch_size 64 \
  --epochs 80 \
  --lr 0.001 \
  --weight_decay 0.01 \
  --patience 12 \
  --num_workers 8 \
  --pin_memory true \
  --persistent_workers true \
  --prefetch_factor 4 \
  --drop_last true \
  --amp true \
  --amp_dtype bf16 \
  --allow_tf32 true \
  --matmul_precision high \
  --fused_adamw true \
  --checkpoint_metric auprc \
  --device cuda
```

Expected: `best_model.pt`, PTB-XL fold10 metrics, and saved training log.

Execution update 2026-06-01:

- The first `batch_size=64` run was stopped after three epochs because GPU
  memory use was only about `6.3GB`; it was treated as a throughput smoke, not
  a final baseline.
- The accepted run used `batch_size=128`, bf16 AMP, TF32, fused AdamW,
  `num_workers=8`, and `prefetch_factor=4`. Steady-state epoch time was about
  `19s`, GPU SM utilization sampled at `100%`, and memory use was about
  `12.5GB`.
- Accepted run directory:
  `/root/autodl-tmp/triple_labels/super5_v7_effnet1dv2_500hz_full10_b128_20260601`.
- Early stopping triggered at epoch 47 with best validation AUPRC selected from
  epoch 35. PTB-XL fold10:
  `macro AUROC/AUPRC = 0.9131 / 0.7867`.

- [x] **Step 3: Evaluate PN2021 v7**

Run:

```bash
/root/miniforge3/envs/ECGTwin/bin/python -u scripts/triple_labels/eval_crosscenter.py \
  --scheme super5 \
  --model_dir /root/autodl-tmp/triple_labels/super5_v7_effnet1dv2_500hz_full10_20260531 \
  --crop_len 5000 \
  --batch_size 64 \
  --num_workers 4 \
  --skip_mimic \
  --preprocess_mode minimal_resample \
  --norm_mode per_sample_global \
  --sampling_rate 500 \
  --pn2021_cache_dir /root/autodl-tmp/triple_labels/pn2021_eval_cache_v7_500hz_perglobal \
  --pn2021_mmap_cache_dir /root/autodl-tmp/triple_labels/pn2021_eval_cache_v7_500hz_perglobal_mmap \
  --output_path /root/autodl-tmp/triple_labels/super5_v7_effnet1dv2_500hz_full10_20260531/eval_result_v7_500hz.json
```

Expected: ref-excluded evaluation path supports K500 exclusion before online AT comparison.

Execution update 2026-06-01:

- Evaluation result:
  `/root/autodl-tmp/triple_labels/super5_v7_effnet1dv2_500hz_full10_b128_20260601/eval_result_v7_500hz.json`.
- PTB-XL fold10: `0.9131 / 0.7867`.
- PN2021 7-center all-zero-kept: `0.7812 / 0.4764`.
- PN2021 7-center drop-all-zero: `0.8003 / 0.5804`.
- Main four target centers:

| center | n | all-zero-kept AUROC/AUPRC | drop-all-zero AUROC/AUPRC |
|---|---:|---:|---:|
| ningbo | 34905 | 0.8602 / 0.4675 | 0.8779 / 0.6376 |
| chapman_shaoxing | 10247 | 0.8605 / 0.4261 | 0.8784 / 0.5977 |
| cpsc_2018 | 6877 | 0.7995 / 0.5498 | 0.8471 / 0.6684 |
| georgia | 10344 | 0.8187 / 0.5989 | 0.8264 / 0.6835 |
| mean | - | 0.8347 / 0.5106 | 0.8574 / 0.6468 |

## Phase 5: 500Hz VAE-Only Online AT For EfficientNet1DV2

**Objective:** Replace ECGTwin 1024-point VAE latents with the new PTB-XL-trained 500Hz VAE latents and rerun the v7 real-anchor online adversarial training mainline.

**Target Outcome:**

- Same direct-K500 versus VAE-online-AT comparison as current v7 100Hz mainline.
- Main metric: PN2021 all-zero-kept and drop-all-zero AUROC/AUPRC, ref-excluded.
- Attribution: if 500Hz improves, show whether gain comes from direct 500Hz classifier, VAE500 online AT, or both.

**Implementation Changes:**

- Add VAE backend switch:
  - `ecgtwin1024`
  - `diffusets500_v1`
  - `diffusets500_v2`
- Encode target-center K500 anchors as `(B, 4, 625)` or `(B, 8, 313)`.
- Decode directly to `(B, 5000, 12)`, no 1024-to-1000 interpolation.
- Keep same-label or exact-positive-set latent partner policy.
- Preserve multi-hot labels; do not collapse to top-1.

**Main Recipe First Run:**

```text
K = 500
M = 20
hull_lambda = 0.05
hull_steps = 3
hull_lr = 0.25
epochs = 30
adv_weight warmup = 3 epochs
quality gate = hard invalid signal rejection only
checkpoint selection = K500-internal validation plus PTB-XL/source floor
```

**Runtime Acceleration Defaults:**

For the accepted first-pass runs, keep the target anchors and PTB-XL cache in
RAM where possible and use:

```text
batch_size = 128
num_workers = 8
pin_memory = true
persistent_workers = true
prefetch_factor = 4
allow_tf32 = true
matmul_precision = high
cudnn_benchmark = true
```

Observed GPU use on the 4090D was roughly `88-100%`, with about `22.5GB` VRAM
used. Do not raise batch size blindly; this is already close to the safe
single-card ceiling.

**Execution Steps:**

- [x] **Step 1: Add VAE500 backend to latent-hull scripts**

Modify:

```text
scripts/paper/run_vae_only_latenthull_sweep_20260516.py
scripts/pgd_cross_center/synth_online_at_super5.py
ecg_adv_gen/vae/
```

Expected: old ECGTwin VAE path still works unchanged.

Execution update 2026-06-01:

- Added `VAE500RuntimeWrapper` for loading the accepted PTB-XL-only 500Hz VAE.
- `EfficientNetVictimTierM` now supports explicit latent backends:
  `ecgtwin1024` keeps the old decode/reorder/resample path, while
  `diffusets500_v1` decodes directly to `(B, 5000, 12)` PTB-XL-order signals.
- Latent-hull code no longer assumes `(4,128)` latents; it accepts arbitrary
  `(C,L)` latent shapes such as `(4,625)`.
- Added `scripts/vae500/export_pn2021_vae500_real_anchors.py` to export
  PN2021 K-shot target-center anchors as paired `signals.npz`, `latent.npz`,
  `ref_meta.json`, and `class_trust.json`.
- Smoke result: CPSC K32 one-epoch VAE500 latent-hull training passed with no
  shape/NaN/decode failure.

- [x] **Step 2: Run one-center smoke on CPSC**

Run:

```bash
CUDA_VISIBLE_DEVICES=<free_gpu> /root/miniforge3/envs/ECGTwin/bin/python -u scripts/paper/run_vae_only_latenthull_sweep_20260516.py \
  --mapping_version v7_super5_sjr_rgq_review \
  --center cpsc_2018 \
  --k 500 \
  --vae_backend diffusets500_v1 \
  --vae_ckpt /root/autodl-tmp/vae500/ptbxl_records500_v7/diffusets500_v1_20260531/checkpoints/best.pt \
  --classifier_ckpt /root/autodl-tmp/triple_labels/super5_v7_effnet1dv2_500hz_full10_20260531/best_model.pt \
  --input_len 5000 \
  --sampling_rate 500 \
  --epochs 3 \
  --output_root /root/autodl-tmp/vae500_lhat_smoke_20260531
```

Expected: finite decoded ECG, attack diagnostics logged, no ref leakage.

Execution update 2026-06-01:

- Exported CPSC K500 anchors:
  `/root/autodl-tmp/vae500_lhat_v7/anchors_k500/cpsc_2018/cpsc_2018_real_k500_seed42_vae500.*`.
- Ran CPSC K500 3-epoch VAE500 latent-hull smoke:
  `/root/autodl-tmp/vae500_lhat_v7/smoke_cpsc_k500_ep3_20260601`.
- Attack/online-AT diagnostics were finite. Epoch ASR was about `0.40`, `0.27`,
  `0.26`; decoded invalid samples were not observed in the logged path, and
  the adversarial buffer reached 384 samples.
- CPSC K500-internal validation improved from `0.8642 / 0.7149` to
  `0.8990 / 0.7871` over the 3 smoke epochs.
- Fair ref-excluded held-out CPSC comparison:

| model | PTB-XL fold10 | CPSC ref-excluded | PN2021 7-center |
|---|---:|---:|---:|
| 500Hz EfficientNet baseline | 0.9131 / 0.7867 | 0.7958 / 0.5427 | 0.7807 / 0.4754 |
| CPSC K500 VAE500 LHAT smoke ep3 | 0.9120 / 0.7843 | 0.8444 / 0.5896 | 0.7906 / 0.4791 |
| delta | -0.0011 / -0.0024 | +0.0486 / +0.0469 | +0.0099 / +0.0037 |

- [x] **Step 3: Four-center main run**

Expected output root:

```text
/root/autodl-tmp/runs/effnet1dv2_500hz_vae500_lhat_k500_v7/<run_id>
```

Expected artifacts: run card, resolved config, metrics, per-center eval JSON, attack diagnostics, reconstruction invalid-rate summary.

Execution update 2026-06-01:

- Ran four center-specific K500 VAE500 online-AT runs under:

```text
/root/autodl-tmp/vae500_lhat_v7/<center>_k500_ep30_20260601
```

- All runs used `VAE500RuntimeWrapper`, latent shape `(4,625)`, `M=20`,
  `hull_lambda=0.05`, `hull_steps=3`, `hull_lr=0.25`, `K_anchor=128`,
  `pgd_batch=16`, `target_real_weight=20`, `adv_weight=0.3`,
  `adv_weight_warmup_epochs=3`, and ref-excluded held-out PN2021 evaluation.
- Added online-AT runtime flags for TF32, matmul precision, CuDNN benchmark,
  pinned memory, persistent workers, and DataLoader prefetch.
- Fair all-zero-kept target-center comparison:

| center | 500Hz baseline, K500 excluded | VAE500 online AT, K500 excluded | delta |
|---|---:|---:|---:|
| ningbo | 0.8598 / 0.4650 | 0.8774 / 0.4986 | +1.75pp / +3.35pp |
| chapman_shaoxing | 0.8602 / 0.4150 | 0.8728 / 0.4459 | +1.26pp / +3.09pp |
| cpsc_2018 | 0.7958 / 0.5427 | 0.8482 / 0.5932 | +5.24pp / +5.05pp |
| georgia | 0.8185 / 0.5963 | 0.8327 / 0.6135 | +1.42pp / +1.72pp |
| mean | 0.8336 / 0.5048 | 0.8578 / 0.5378 | +2.42pp / +3.30pp |

- Source PTB-XL fold10 stayed close to the baseline `0.9131 / 0.7867`; adapted
  checkpoints were in the range `0.9105-0.9120 / 0.7792-0.7848`.
- Detailed report:

```text
docs/tmp_md/vae500_lhat_four_center_20260601.md
```

## Phase 6: ECGFounder 500Hz V7 Rebuild

**Objective:** Rerun ECGFounder under the same v7 label protocol and compare direct K500 adaptation against 500Hz VAE-online-AT candidates.

**Target Outcome:**

- ECGFounder direct K500 v7 metrics.
- ECGFounder VAE500-online-AT metrics.
- Clear comparison to EfficientNet1DV2-500Hz and current 100Hz evidence.

**Protocol:**

- Input: official ECGFounder 500Hz, 10s, 12-lead preprocessing branch.
- Labels: v7 Super5.
- Main direct controls:
  - frozen encoder + Super5 head;
  - init-head full fine-tune if current evidence says it is needed;
  - direct K500 adaptation without VAE.
- VAE branch:
  - use decoded 500Hz VAE-online samples as target-adversarial stream;
  - keep target-real supervised stream matched to direct control.

**Execution Steps:**

- [x] **Step 1: Verify ECGFounder preprocessing parity**

Run a small audit that checks:

```text
shape = (B, 12, 5000)
sampling_rate = 500
normalization = ECGFounder official path
label mapping = v7
K500 ref IDs excluded from eval
```

- [x] **Step 2: Run direct ECGFounder v7 control**

Use existing ECGFounder scripts when possible:

```text
scripts/paper/run_ecgfounder_kshot_head_ft_20260517.py
scripts/paper/run_ecgfounder_fullft_super5_pilot_20260523.py
```

Expected output root:

```text
/root/autodl-tmp/runs/ecgfounder_500hz_direct_k500_v7/<run_id>
```

- [x] **Step 3: Run matched VAE500 ECGFounder branch**

Expected: identical K500 split, same model-selection rule, same all-zero handling, only VAE-online-AT stream changes.

Execution update 2026-06-01:

- Reused ECGFounder 500Hz feature cache and rebuilt PN2021 labels to v7:

```text
/root/autodl-tmp/paper_ecgfounder_500hz_v7/feature_cache_official_v7_from_v5
```

- Frozen linear probe v7 run:

```text
/root/autodl-tmp/paper_ecgfounder_500hz_v7/linear_probe_seed42_official_v7
```

PTB-XL fold10: `0.9191 / 0.7952`.

| target view | frozen linear probe |
|---|---:|
| ningbo | 0.8846 / 0.5134 |
| chapman_shaoxing | 0.8871 / 0.4794 |
| cpsc_2018 | 0.8157 / 0.5690 |
| georgia | 0.8564 / 0.6640 |

- Direct K500 head fine-tune:

```text
/root/autodl-tmp/paper_ecgfounder_500hz_v7/kshot_head_ft_seed42_official_v7
```

| center | direct K500 target-center |
|---|---:|
| ningbo | 0.9301 / 0.6444 |
| chapman_shaoxing | 0.9311 / 0.6367 |
| cpsc_2018 | 0.8796 / 0.6618 |
| georgia | 0.8937 / 0.7527 |

- Added VAE500 decoder support to the ECGFounder VAE-only LHAT runner and ran
  a matched 10-epoch branch initialized from the direct K500 heads:

```text
/root/autodl-tmp/paper_ecgfounder_500hz_v7/vae500_lhat_headft_ep10_seed42_official_v7
```

This first ECGFounder VAE500 branch selected epoch 0 for all centers, so it did
not improve over direct K500. Attack diagnostics were finite and active, but
the K500-internal validation did not prefer the adversarially adapted heads.

Detailed report:

```text
docs/tmp_md/ecgfounder_500hz_v7_phase6_20260601.md
```

## Phase 7: `model/ecg_ptbxl_benchmarking` 500Hz Model Set

**Objective:** Add independent 500Hz PTB-XL benchmark-style backbones as comparison models, then evaluate whether VAE500 online AT helps beyond EfficientNet1DV2.

**Target Outcome:**

- Five benchmark backbones trained on PTB-XL 500Hz Super5.
- Their PN2021 v7 direct K500 and VAE-online-AT results are comparable to EfficientNet1DV2.

**Recommended First Backbones:**

| Model | Reason |
|---|---|
| `fastai_inception1d` with `input_size=5000` | strong 1D convolution baseline, already in repo |
| `fastai_resnet1d_wang` with `input_size=5000` | classic ECG/time-series baseline |
| `fastai_xresnet1d50` with `input_size=5000` | stronger capacity comparison |
| `fastai_fcn_wang` with `input_size=5000` | lightweight fully convolutional time-series baseline |
| `fastai_schirrmeister` with `input_size=5000` | different temporal-convolution design, useful as architecture-diversity control |

**Important Difference From Our EfficientNet Pipeline:**

`model/ecg_ptbxl_benchmarking` uses dataset-level `StandardScaler` fit on training data, while our EfficientNet branch uses per-sample global z-score. Keep both protocols explicit; do not silently mix them.

**Execution Steps:**

- [x] **Step 1: Add or wrap 500Hz Super5 configs**

Modify or create:

```text
scripts/triple_labels/train_ptbxl_benchmarking_500hz_super5.py
configs/experiments/ptbxl_benchmarking_500hz_super5_v7.yaml
```

Expected: wrapper trains selected benchmark models and exports predictions/checkpoints in a format usable by our PN2021 evaluator.

- [x] **Step 2: Smoke one model**

Run:

```bash
CUDA_VISIBLE_DEVICES=<free_gpu> /root/miniforge3/envs/ECGTwin/bin/python -u scripts/triple_labels/train_ptbxl_benchmarking_500hz_super5.py \
  --ptbxl_root /root/autodl-tmp/ptbxl \
  --output_dir /root/autodl-tmp/ptbxl_benchmarking_500hz_super5_v7/smoke_inception1d \
  --models fastai_inception1d \
  --sampling_frequency 500 \
  --input_size 5000 \
  --epochs 2 \
  --limit_train 128
```

Expected: data loading, model forward, fold split, and prediction export work.

Execution update 2026-06-01:

- Added:

```text
scripts/triple_labels/train_ptbxl_benchmarking_500hz_super5.py
```

- The wrapper keeps `model/ecg_ptbxl_benchmarking` read-only, loads this
  project's PTB-XL records500 raw cache, fits a dataset-level global
  StandardScaler on train, and supports the five planned benchmark backbones.
- Acceleration flags mirror the EfficientNet branch where safe:
  RAM loading, bf16 AMP, TF32, CuDNN benchmark, fused AdamW, pinned memory,
  persistent workers, and DataLoader prefetch.
- Smoke run:

```text
/root/autodl-tmp/ptbxl_benchmarking_500hz_super5_v7/smoke_inception1d_20260601
```

Smoke settings: `fastai_inception1d`, `limit_train=128`, `limit_val=128`,
`limit_test=128`, `epochs=2`. It completed without model/data shape failure and
reported fold10 smoke metrics `0.5958 / 0.3572`.

- [x] **Step 3: Train selected models**

Run the five-backbone set after the one-model smoke passes:

```bash
CUDA_VISIBLE_DEVICES=<free_gpu> /root/miniforge3/envs/ECGTwin/bin/python -u scripts/triple_labels/train_ptbxl_benchmarking_500hz_super5.py \
  --ptbxl_root /root/autodl-tmp/ptbxl \
  --output_dir /root/autodl-tmp/ptbxl_benchmarking_500hz_super5_v7/five_backbones_20260531 \
  --models fastai_inception1d,fastai_resnet1d_wang,fastai_xresnet1d50,fastai_fcn_wang,fastai_schirrmeister \
  --sampling_frequency 500 \
  --input_size 5000 \
  --epochs 30
```

Expected output root:

```text
/root/autodl-tmp/ptbxl_benchmarking_500hz_super5_v7/<model_name>_<run_id>
```

Acceptance:

- PTB-XL fold10 metrics are not degenerate.
- PN2021 v7 direct metrics are recorded.
- Runtime and memory are acceptable before VAE-online-AT adaptation.

Execution update 2026-06-01:

- Full five-backbone run completed:

```text
/root/autodl-tmp/ptbxl_benchmarking_500hz_super5_v7/five_backbones_full30_20260601
```

- Training settings: records500 RAM loading, `epochs=30`, `batch_size=128`,
  `eval_batch_size=256`, `num_workers=8`, pinned memory, persistent workers,
  `prefetch_factor=4`, bf16 AMP, TF32, CuDNN benchmark, fused AdamW.
- Added PN2021 evaluator:

```text
scripts/triple_labels/eval_ptbxl_benchmarking_500hz_super5.py
```

- PN2021 evaluation uses mmap cache on disk and copies each center into RAM for
  inference. Runtime settings were `batch_size=512`, `num_workers=12`, pinned
  memory, persistent workers, bf16 AMP, TF32, and CuDNN benchmark.
- Redundant compressed benchmark `.npz` PN2021 cache was removed; the retained
  cache is:

```text
/root/autodl-tmp/triple_labels/pn2021_eval_cache_v7_500hz_benchmark_none_mmap
```

Target-center K500-excluded results:

| model | PTB-XL fold10 | PN2021 4-center mean |
|---|---:|---:|
| `fastai_inception1d` | 0.9247 / 0.8138 | 0.8452 / 0.4971 |
| `fastai_resnet1d_wang` | 0.9175 / 0.7984 | 0.8246 / 0.4833 |
| `fastai_xresnet1d50` | 0.9206 / 0.8079 | 0.8511 / 0.5120 |
| `fastai_fcn_wang` | 0.9084 / 0.7869 | 0.8245 / 0.4838 |
| `fastai_schirrmeister` | 0.9070 / 0.7651 | 0.8264 / 0.4671 |

Detailed report:

```text
docs/tmp_md/ptbxl_benchmarking_500hz_v7_phase7_20260601.md
```

Note: VAE500 online AT has not yet been run on these five benchmark backbones.
The current online AT trainer is centered on the project EfficientNet1DV2 victim
path. A generic benchmark victim/model-builder adapter is needed for a fully
matched VAE500-AT comparison on this model set.

## Phase 8: Unified Comparison And Reporting

**Objective:** Produce a single table that separates three effects:

1. 100Hz versus 500Hz input protocol.
2. Backbone choice.
3. 500Hz VAE-online-AT versus matched direct K500 adaptation.

**Target Outcome:**

Create:

```text
docs/tmp_md/500hz_vae_v7_mainline_report_<date>.md
docs/tmp_html/500hz_vae_v7_mainline_report_<date>.html
/root/autodl-tmp/runs/comparison_bundles/500hz_vae_v7_mainline_<date>/
```

Execution update 2026-06-01:

Created the main status reports:

```text
docs/tmp_md/500hz_vae_v7_mainline_report_20260601.md
docs/tmp_html/500hz_vae_v7_mainline_report_20260601.html
```

The report currently covers the accepted VAE audit, EfficientNet1DV2 baseline
and VAE500-online-AT result with both all-zero-kept and drop-all-zero PN2021
views, ECGFounder linear/direct/VAE branches, the five
`ecg_ptbxl_benchmarking` direct backbones, and a historical 100Hz versus current
500Hz effect-size comparison. The comparison bundle directory is still optional
follow-up packaging; all source report data already exists in the run roots
listed above.

**Required Tables:**

- PTB-XL fold10 metrics by backbone.
- PN2021 four-center all-zero-kept metrics by backbone.
- PN2021 four-center drop-all-zero sensitivity metrics.
- Direct K500 vs VAE500-online-AT deltas.
- 100Hz ECGTwin-VAE result vs 500Hz VAE500 result.
- Reconstruction audit summary.
- Attack diagnostics summary: `loss_gain`, positive-hide ASR, negative-add ASR, invalid decode rate.

**Paper-Safe Interpretation Rule:**

- If 500Hz direct baseline improves but VAE-online-AT does not add further gain, claim improved input/backbone protocol, not VAE contribution.
- If VAE-online-AT improves over matched direct K500 across multiple backbones, claim a stronger latent-manifold adaptation contribution.
- If only CPSC improves, report center-specific benefit and diagnose class distribution; do not overclaim universal cross-center improvement.

## Rough Compute Plan

| Stage | Expected time on one 4090-class GPU | Notes |
|---|---:|---|
| compression audit | 10-30 min CPU/IO | no GPU |
| archive selected cold dirs | 1-6 h CPU/IO | depends on selected dirs and compression level |
| PTB-XL 500Hz cache build | 20-60 min CPU/IO | WFDB read can bottleneck |
| VAE500 smoke | 10-30 min GPU | shape/loss only |
| VAE500 full train v1 | 6-18 h GPU | first estimate; stop early if recon collapses |
| EfficientNet1DV2 500Hz train | 2-6 h GPU | batch size may need tuning |
| one-center VAE-online-AT smoke | 20-60 min GPU | catches decode/attack bugs |
| four-center VAE-online-AT | 2-8 h GPU | can parallelize centers if GPUs are free |
| ECGFounder direct/VAE branches | 2-10 h GPU | depends on frozen vs full fine-tune |
| benchmark backbones | 10-60 h total | smoke one model first, then run five backbones; parallelize only when GPUs are safely free |

## Remaining Ambiguities And Execution-Time Checkpoints

No blocking research-design question remains before Phase 1.

During execution, stop and report before these points:

1. Starting the first full VAE500 training after smoke/cache tests, with GPU,
   batch size, estimated runtime, and free disk.
2. Promoting VAE500 outputs into downstream online AT, with reconstruction audit
   metrics and invalid decode rate.
3. Running all five benchmark backbones if the one-model smoke suggests runtime
   above one day or memory instability.

Automatic deletion after verified cold-archive compression does not require an
extra stop. It must still be reported in the archive manifest and final status
message.
