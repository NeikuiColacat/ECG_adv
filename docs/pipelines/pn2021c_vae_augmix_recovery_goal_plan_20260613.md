# PN2021-C Strong-Corruption Recovery With VAE Online AT And Latent AugMix

Date: 2026-06-13

This document is written as a long-running Codex goal handoff plan. The goal is
to test whether ECGTwin VAE online adversarial training plus latent AugMix can
recover a meaningful fraction of AUROC/AUPRC loss when PN2021-C corruptions are
calibrated to cause roughly 10-20 percentage point drops.

## Goal Prompt For Codex

Use this objective if starting a Codex goal:

```text
In /home/linbinhao/ECG_adv_Gen, investigate and implement a reproducible
PN2021-C strong-corruption recovery experiment using EfficientNet1DV2
Direct-K500 initialization plus ECGTwin VAE Latent-Hull online adversarial
training and latent AugMix. Keep the PN2021 Super5 v7 SJR/RGQ mapping, fixed
K500 seed/ref-exclusion policy, and shared-server safety rules. Produce
paper-safe configs, run records, evaluation CSVs, and a concise report showing
whether the method improves corrupted absolute AUROC/AUPRC and how much of the
10-20 pp PN2021-C drop it recovers.
```

## Critical Constraints

- Read the first 100 lines of `AGENTS.md` after every context compaction or
  handoff before writing files, using GPUs, or changing environments.
- Keep all work under `/home/linbinhao`, especially:
  - repo: `/home/linbinhao/ECG_adv_Gen`
  - data/output root: `/home/linbinhao/ECG_adv_data`
  - Python env: `/home/linbinhao/micromamba/envs/ECGTwin`
- Do not use `sudo`.
- Do not install or change CUDA, NVIDIA drivers, kernel, or system
  environments.
- Before any GPU run, check `nvidia-smi` and set explicit
  `CUDA_VISIBLE_DEVICES=<free_gpu_ids>`.
- Prefer one GPU per child run first; only use multi-GPU scheduling if the
  current cluster state is clearly free.
- Do not overwrite existing run directories. Use a new date/config-named
  `run_id`.
- Keep checkpoints, caches, generated signals, PN2021-C outputs, and logs out
  of git.
- Use YAML-managed launcher surfaces when possible:
  `scripts/run_experiment.py` plus `configs/experiments/*.yaml`.

## Existing Source Of Truth

Primary managed surfaces:

- Active script registry: `configs/active_scripts.yaml`
- Active evidence registry: `configs/active_evidence_registry.yaml`
- EfficientNet VAE-LHAT config:
  `configs/experiments/effnet_vae_lhat_k500_v7_sjr_rgq.yaml`
- PN2021-C EfficientNet noAug-vs-AugMix config:
  `configs/experiments/pn2021c_effnet_v7_augmix_vs_noaug.yaml`
- Launcher: `scripts/run_experiment.py`
- Main VAE-LHAT legacy entrypoint:
  `scripts/paper/run_effnet_latent_augmix_stage3_20260524.py`
- PN2021-C evaluator:
  `scripts/triple_labels/eval_pn2021_corruptions.py`
- PN2021-C aggregation helper:
  `ecg_adv_gen/evaluation/pn2021_corruptions.py`

Protocol to preserve:

- Mapping version: `v7_super5_sjr_rgq_review_20260528`
- Mapping hash: `555ec85d5b51`
- Class order: `CD, HYP, MI, NORM, STTC`
- Target centers:
  `ningbo`, `chapman_shaoxing`, `cpsc_2018`, `georgia`
- K-shot policy: fixed `K=500`, seed `20260531`
- Evaluation: ref-exclude the K500 records for each target center.
- Model-selection rule: K500 internal validation plus source/PTB-XL floor.
- Forbidden selection information:
  heldout PN2021 labels, full target-center class distribution, and
  per-center hand-tuned choices.

Current v7 EfficientNet evidence from the active registry:

- Direct K500 all-zero-kept PN2021:
  `0.8521806591 AUROC / 0.5246913957 AUPRC`
- VAE-LHAT all-zero-kept PN2021:
  `0.8711774092 AUROC / 0.5543950864 AUPRC`
- VAE-LHAT gain all-zero-kept:
  `+1.89967501 pp AUROC / +2.97036907 pp AUPRC`
- Direct K500 drop-all-zero PN2021:
  `0.8763066077 AUROC / 0.6666377832 AUPRC`
- VAE-LHAT drop-all-zero PN2021:
  `0.8996374591 AUROC / 0.7176495521 AUPRC`
- VAE-LHAT gain drop-all-zero:
  `+2.33308514 pp AUROC / +5.10117689 pp AUPRC`

## Strong PN2021-C Stress Definition

Standard PN2021-C severity=5 is not enough for this objective. The previous
validated mainline drop was only about:

- Standard severity=5 mean drop:
  `0.021042245869 AUROC / 0.031096588534 AUPRC`

Use the stronger calibrated 10-20 pp single-operator settings as the primary
strong-corruption benchmark:

```text
baseline_shift:
  max_amplitude=2.4
  shift_ratio=0.9
  num_segment=6
  p=1.0

baseline_wander:
  max_amplitude=2.5
  k=6
  max_freq=0.8
  p=1.0

emg_noise:
  max_amplitude=2.3
  p=1.0

powerline_noise:
  max_amplitude=8.0
  p=1.0

random_leads_masking:
  mask_leads_prob=0.57
  p=1.0
```

Previously verified four-center mean drops for those settings:

| corruption | AUROC drop | AUPRC drop |
|---|---:|---:|
| baseline_shift | 0.1022106344 | 0.1375607763 |
| baseline_wander | 0.1304136017 | 0.1353201137 |
| emg_noise | 0.1077170477 | 0.1422639263 |
| powerline_noise | 0.1072432544 | 0.1295567849 |
| random_leads_masking | 0.1039197763 | 0.1522676413 |

Also keep `stress_v2 severity=5` as a smoke/diagnostic profile:

- Previously verified four-center mean drop:
  `0.081395679250 AUROC / 0.099526036377 AUPRC`

Important evaluator rule:

- Do not compare corrupted subset metrics against full clean metrics.
- Always compute clean and corrupted metrics on the same ref-excluded subset
  when measuring drops.

## Main Hypothesis

Direct K500 fine-tuning can fit target-center clean distribution, but under
strong PN2021-C corruptions it remains brittle. ECGTwin VAE Latent-Hull online
AT should improve local on-manifold target-center invariance, while latent
AugMix should expose the classifier to realistic waveform corruptions during
the adversarial branch.

Expected useful signal:

- corrupted absolute AUROC/AUPRC improves over Direct K500;
- PN2021 clean does not collapse;
- drop-all-zero view improves consistently;
- recovery is not only a CPSC/Chapman artifact, with Ningbo/Georgia checked
  explicitly.

## Evaluation Priority

Primary metric:

- strong PN2021-C corrupted absolute macro AUROC/AUPRC over four target centers.

Secondary metrics:

- clean-to-corrupt drop on the same subset;
- PN2021 clean all-zero-kept ref-excluded;
- PN2021 clean drop-all-zero ref-excluded;
- PN2021-C drop-all-zero corrupted view;
- per-center and per-class deltas;
- attack diagnostics from training.

Interpretation rule:

- If a model has higher absolute corrupted AUROC/AUPRC but a larger
  self-clean drop, report it as improved corrupted absolute performance, not as
  fully improved relative robustness.

## Success Criteria

Minimum useful result:

- Four-center strong PN2021-C corrupted absolute performance improves over
  Direct K500 by at least:
  `+2 pp AUROC / +4 pp AUPRC`, with no severe PN2021 clean collapse.

Strong result:

- Recovers at least 30-50% of the calibrated 10-20 pp corruption loss on AUPRC,
  while keeping PN2021 clean and source/PTB-XL floor within acceptable range.

Stop condition:

- If all tested VAE+AugMix candidates improve only standard PN2021-C but not
  the calibrated strong profile, document the negative result and do not keep
  sweeping scalar attack strength.

## Experiment Candidates

Keep the candidate set small. Avoid broad unstructured sweeps.

### Candidate A: Current VAE-LHAT + Latent AugMix Reproduction

Purpose:

- Reproduce the current `effnet_vae_lhat_k500_v7_sjr_rgq` behavior under the
  strong PN2021-C profile.

Expected role:

- baseline VAE+AugMix reference.

Do not tune from this result yet; use it to establish the strong-corruption
delta.

### Candidate B: Corruption-Curriculum Latent AugMix

Purpose:

- Train with mild-to-moderate corruption curriculum instead of directly using
  the strong eval corruption level.

Starting AugMix settings:

```text
latent_augmix.enabled=true
latent_augmix.width=3
latent_augmix.depth=-1
latent_augmix.alpha=1.0
latent_augmix.latent_weight_cap=0.25
latent_augmix.severity curriculum: 1 -> 2 -> 3
ops:
  powerline_noise
  emg_noise
  baseline_wander
  baseline_shift
```

Rationale:

- `random_leads_masking` at strong eval level is too harsh for first-pass
  training and can encourage lead-dropout specialization.
- The four continuous/noise corruptions better match a regularization
  curriculum.

### Candidate C: Curriculum + Mild Lead Masking

Purpose:

- Test whether controlled lead masking helps the harshest PN2021-C masking
  operator without collapsing clean performance.

Starting settings:

```text
random_leads_masking during training:
  mask_leads_prob in {0.15, 0.25, 0.35}
  p <= 0.5
```

Rule:

- Do not train first with `mask_leads_prob=0.57`; reserve that for evaluation.

### Candidate D: Clean-Aug Consistency Branch

Purpose:

- Add or emulate a consistency loss between clean target anchor predictions and
  latent-AugMix/adversarial branch predictions.

Preferred shape:

```text
loss = clean_anchor_bce
     + mix_weight * mixed_latent_bce
     + adv_weight * adversarial_bce_or_bernoulli_kl
     + consistency_weight * bernoulli_kl(sigmoid(clean_logits), sigmoid(aug_logits))
     + optional_source_floor_loss
```

Rule:

- For multilabel Super5, use independent Bernoulli KL or BCE-style consistency,
  not softmax KL.

## VAE-LHAT Defaults To Preserve

Use ECGTwin VAE latent mixing as constrained VRM/manifold mixup.

- Decode path:
  `latent (4,128) -> ECGTwin decode 1024 -> PTB-XL lead reorder -> 1000 samples`
- Lead-order fix:
  `ECGTWIN_TO_PTBXL_INDICES = [0,1,2,3,5,4,6,7,8,9,10,11]`
- Standardize VAE latents before distance search, mixup, or PGD.
- Prefer same-label or exact-positive-set partners.
- Do not mix NORM with abnormal labels in the main recipe.
- Use local kNN or distance-cutoff partners in standardized latent space.
- Include the original anchor in the latent hull.
- Use anchor-dominant coefficients, roughly anchor weight `0.6-0.95`.
- Keep clean K500 anchors in the batch whenever latent mix or PGD is enabled.
- Prefer `latent_mixed_teacher` or `anchor_soft` labels over argmax-collapsed
  hard labels.

Attack/diagnostic targets:

- Training PGD: 3-5 steps.
- Audit PGD: 10-20 steps.
- Practical ASR target: 30-70%.
- If `atk_init < 0.6` or `loss_gain <= 0`, attack is probably too weak.
- If `atk_anchor > 0.8` early and clean/source metrics drop, attack or adv
  stream is probably too strong.
- Do not increase `adv_weight` blindly. Previous high-pressure runs did not
  solve Ningbo/Georgia.

Required training logs:

- clean/adversarial macro AUROC and AUPRC;
- clean BCE, adversarial BCE, `loss_gain`;
- `atk_init`, `atk_anchor`;
- clean-correct conditioned multilabel ASR where available;
- positive-hide, negative-add, sample-anyflip ASR if available;
- latent norm utilization;
- decoded invalid rate, flatline rate, amplitude sanity;
- `latent_augmix_stats`;
- source/PTB-XL floor;
- `best_clean`, `best_robust_val`, and `last`.

## Implementation Phases

### Phase 0: Read-Only Audit

Objective:

- Confirm current config surfaces, required inputs, and existing outputs before
  editing or launching anything.

Commands:

```bash
sed -n '1,100p' AGENTS.md
git status --short
micromamba run -n ECGTwin python scripts/agent/audit_agent_workspace.py
micromamba run -n ECGTwin python scripts/agent/check_external_models.py
```

Expected checks:

- active scripts include `effnet_vae_lhat_k500_v7_sjr_rgq`;
- active scripts include `pn2021c_effnet_v7_augmix_vs_noaug`;
- external model links are valid or known ignored dirty external models;
- no large artifacts are staged;
- K500 ref-meta files exist for all four centers.

### Phase 1: Strong-Corruption Evaluation Dry Run

Objective:

- Verify whether the current PN2021-C launcher can represent the strong
  calibrated profile. If it cannot, add the smallest config/evaluator extension
  needed to express the fixed operator parameters.

Initial dry-run:

```bash
micromamba run -n ECGTwin python scripts/run_experiment.py \
  --config configs/experiments/pn2021c_effnet_v7_augmix_vs_noaug.yaml \
  --local-config configs/local/linbinhao_server.example.yaml \
  --run-id pn2021c_strong_10to20pp_eval_dryrun_20260613 \
  --dry-run \
  --write-plan
```

If `severity_profile=stress_v2` is enough for smoke, use it only as a
diagnostic. For the final strong profile, implement or use a profile that
matches the calibrated parameters above.

Deliverable:

- resolved dry-run manifest;
- note whether calibrated parameters are currently expressible;
- minimal patch if needed.

### Phase 2: Current Candidate Strong PN2021-C Baseline

Objective:

- Evaluate Direct K500, current VAE-LHAT noAug, and current VAE-LHAT+AugMix
  under the same strong PN2021-C profile.

Rules:

- Use the same four target centers and ref-exclusion files.
- Use same-subset clean and corrupted metrics.
- Export both all-zero-kept and drop-all-zero views.
- Record the absolute corrupted metrics and self-clean drops separately.

Deliverables:

- per-center JSONs;
- one aggregate CSV with:
  `method, center, corruption, view, clean_auroc, clean_auprc,
  corrupted_auroc, corrupted_auprc, drop_auroc, drop_auprc`;
- summary markdown with Direct vs VAE noAug vs VAE AugMix.

### Phase 3: Candidate Configs

Objective:

- Add new YAML-managed candidates without changing the existing trusted mainline
  config.

Suggested new files:

```text
configs/experiments/effnet_vae_lhat_augmix_curriculum_k500_v7_sjr_rgq.yaml
configs/experiments/effnet_vae_lhat_augmix_curriculum_mask_k500_v7_sjr_rgq.yaml
configs/experiments/effnet_vae_lhat_augmix_consistency_k500_v7_sjr_rgq.yaml
configs/experiments/pn2021c_effnet_v7_strong_10to20pp.yaml
```

Keep changes small:

- extend existing defaults/adapters if needed;
- do not create a second launcher path;
- preserve `scripts/run_experiment.py`;
- add CPU tests for new config rendering or evaluator profile logic.

Required dry-runs:

```bash
micromamba run -n ECGTwin python scripts/run_experiment.py \
  --config configs/experiments/effnet_vae_lhat_augmix_curriculum_k500_v7_sjr_rgq.yaml \
  --local-config configs/local/linbinhao_server.example.yaml \
  --run-id dryrun_effnet_lhat_curriculum_20260613 \
  --dry-run \
  --write-plan

micromamba run -n ECGTwin python scripts/run_experiment.py \
  --config configs/experiments/pn2021c_effnet_v7_strong_10to20pp.yaml \
  --local-config configs/local/linbinhao_server.example.yaml \
  --run-id dryrun_pn2021c_strong_10to20pp_20260613 \
  --dry-run \
  --write-plan
```

### Phase 4: Short Smoke Runs

Objective:

- Catch broken configs and obviously invalid training before launching full
  four-center runs.

Smoke setup:

- one center first: `cpsc_2018` or `ningbo`;
- small epoch count if the adapter allows it;
- smaller batches only if needed for memory;
- no broad GPU parallelism.

Before running:

```bash
nvidia-smi
```

Launch shape:

```bash
CUDA_VISIBLE_DEVICES=<free_gpu> micromamba run -n ECGTwin python scripts/run_experiment.py \
  --config <candidate_config.yaml> \
  --local-config configs/local/linbinhao_server.example.yaml \
  --run-id <candidate_smoke_run_id> \
  --execute
```

Smoke acceptance:

- run starts and writes manifest/run files;
- decoded invalid rate is low;
- `latent_augmix_stats` present;
- attack diagnostics are present;
- no clean/source collapse in early metrics;
- no runaway disk growth.

### Phase 5: Full Four-Center Runs

Objective:

- Run only candidates that passed smoke.

Recommended order:

1. Candidate A reproduction if missing under the strong profile.
2. Candidate B curriculum no masking.
3. Candidate C curriculum plus mild masking.
4. Candidate D consistency branch only if B/C show a signal or if implementation
   cost is small.

Run discipline:

- check `nvidia-smi` before each batch;
- set explicit `CUDA_VISIBLE_DEVICES`;
- prefer one center per GPU if multiple GPUs are clearly free;
- do not kill unrelated processes;
- monitor disk space before and after each candidate.

### Phase 6: Evaluation And Aggregation

Objective:

- Evaluate each candidate under:
  - PN2021 clean ref-excluded;
  - PN2021 clean drop-all-zero;
  - strong PN2021-C calibrated 10-20 pp profile;
  - optional `stress_v2 severity=5` smoke profile.

Aggregation deliverables:

```text
docs/reports/archive/20260613/pn2021c_vae_augmix_recovery/
  strong_corruption_summary.csv
  strong_corruption_per_center.csv
  strong_corruption_per_class.csv
  clean_vs_corrupt_drop.csv
  candidate_training_diagnostics.csv
  selected_candidate_summary.md
```

Required columns:

```text
run_id
method
candidate
center
corruption
view
mapping_version
mapping_hash
kshot_seed
ref_excluded
clean_macro_auroc
clean_macro_auprc
corrupted_macro_auroc
corrupted_macro_auprc
drop_macro_auroc
drop_macro_auprc
delta_vs_direct_corrupted_auroc_pp
delta_vs_direct_corrupted_auprc_pp
recovery_fraction_auroc
recovery_fraction_auprc
```

Recovery fraction:

```text
recovery_fraction =
  (candidate_corrupted - direct_corrupted)
  / (direct_clean_same_subset - direct_corrupted)
```

Report both AUROC and AUPRC fractions.

### Phase 7: Selection And Final Report

Selection rule:

- Pick the candidate with the best K500-internal validation/source-safe rule,
  then report heldout PN2021/PN2021-C once.
- Do not pick by heldout PN2021-C oracle.

Final report must include:

- what was tested;
- exact configs and run IDs;
- source/target mapping version and hash;
- K500 seed and ref-exclusion status;
- corrupted absolute metrics;
- self-clean drops;
- drop-all-zero metrics;
- per-center failure analysis;
- training diagnostics, especially ASR and decoded invalid rates;
- explicit statement whether the result is:
  - positive mainline candidate;
  - partial robustness result;
  - negative result.

## Failure Branches

If attack diagnostics are weak:

- `atk_init < 0.6` or `loss_gain <= 0`:
  increase PGD steps or small epsilon/hull radius first;
  do not immediately increase adv stream weight.

If attack is too strong:

- `atk_anchor > 0.8` early plus clean/source drop:
  reduce adv pressure, warm up more slowly, or reduce `hull_lambda`.

If CPSC improves but Ningbo/Georgia do not:

- stop scalar sweeps;
- inspect per-class HYP/MI ranking and target-center latent geometry;
- consider source-aware or cross-center-invariant constraints.

If random lead masking dominates:

- separate masking robustness from noise/baseline robustness;
- report masking as a distinct harsh operator;
- do not train with eval-level `mask_leads_prob=0.57` unless a mild curriculum
  has already shown benefit.

If current evaluator cannot express calibrated parameters:

- implement a named profile such as `calibrated_10to20pp_20260606`;
- add tests for profile construction;
- keep `standard` and `stress_v2` behavior unchanged.

If multiprocessing hits `AF_UNIX path too long`:

```bash
mkdir -p /home/linbinhao/tmp
export TMPDIR=/home/linbinhao/tmp
```

Then rerun with `num_workers=0` if needed.

## Code Quality And Tests

Before finishing any code/config change:

```bash
micromamba run -n ECGTwin python scripts/agent/audit_agent_workspace.py
micromamba run -n ECGTwin python -m pytest util/tests/test_pn2021_corruptions.py
```

If config adapters or launcher rendering changed, also run relevant config tests:

```bash
micromamba run -n ECGTwin python -m pytest util/tests/test_config_loader.py
```

Do not stage or commit:

- checkpoints;
- generated PN2021-C caches;
- large CSV/JSON output if not intended as lightweight report data;
- external model directories or symlink target changes;
- credentials or local-only machine config.

## Expected Final Handoff

At the end, write a short handoff summary with:

- best candidate and run ID;
- exact strong-corruption profile used;
- Direct K500 corrupted metrics;
- best candidate corrupted metrics;
- absolute delta in pp;
- recovery fraction;
- PN2021 clean impact;
- source/PTB-XL floor impact;
- whether drop-all-zero agrees with all-zero-kept;
- list of artifacts and paths;
- remaining risks or blockers.

If the result is negative, the final handoff should say that plainly and
recommend the next mechanism-level step, not another broad parameter sweep.
