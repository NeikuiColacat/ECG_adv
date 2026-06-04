# ECG_adv_Gen Agent Notes

This file is durable project memory for coding agents working in this
ECG_adv_Gen repository. The original AutoDL path was `/root/ECG_adv_Gen`; on
the current migrated host, use the `/home/linbinhao` paths below.

Critical startup rule for this shared server:

- After every context compaction, resume, or new Codex handoff, read the first
  100 lines of this `AGENTS.md` before running any command that would use GPU,
  write files, or modify any environment.
- Treat this machine as a multi-user shared server at all times. Do not damage
  other users' files, system environments, experiment outputs, running
  processes, ports, or GPU jobs.
- Keep all operations for this project inside the current user's home tree.
  For this migrated host, that means paths under `/home/linbinhao`, especially
  `/home/linbinhao/ECG_adv_Gen` and the migrated data root below.

Current migrated host override, initialized 2026-05-23:

```text
repo root:      /home/linbinhao/ECG_adv_Gen
migrated data:  /home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp
python env:     /home/linbinhao/micromamba/envs/ECGTwin/bin/python
runtime skill:  /home/linbinhao/.codex/skills/ecg-adv-gen/SKILL.md
```

Current agent operating layer, initialized 2026-05-29:

```text
active evidence registry: configs/active_evidence_registry.yaml
CPU-only agent audit:     micromamba run -n ECGTwin python scripts/agent/audit_agent_workspace.py
                         covers active evidence, active scripts, and dirty layers
handoff readiness:        inspect handoff_contract.handoff_readiness
source-of-truth summary:  inspect handoff_contract.source_of_truth_summary
source-of-truth status:   inspect handoff_contract.source_of_truth_status
intent-to-add paths:      inspect handoff_contract.source_of_truth_summary.intent_to_add_paths
managed YAML git status:  inspect active_scripts.config_git_summary
artifact risk checklist:  inspect git.blocking_artifact_risks
dirty handoff gate:       inspect git.dirty_summary.handoff_gate in the audit JSON
dirty layer details:      inspect git.dirty_summary.by_layer for per-layer actions
current handoff note:     docs/codex-handoffs/current_workspace_handoff.md
legacy VAE manifest:      micromamba run -n ECGTwin python scripts/agent/backfill_vae_lhat_manifest.py
comparison bundle build:  micromamba run -n ECGTwin python scripts/agent/build_comparison_bundle.py --force
run finalizer:            micromamba run -n ECGTwin python scripts/agent/finalize_run.py --run-dir <run_dir>
run registry update:      micromamba run -n ECGTwin python scripts/agent/register_run.py --run-dir <run_dir> --status provisional
details:                  docs/pipelines/agent_operating_layer_20260529.md
```

Research and uncertainty handling:

- When a method/design choice is uncertain or external precedent would help,
  use installed MCP tools, relevant skills, live web search, and parallel
  subagents as needed before committing to a direction. Treat that research as
  guidance for the ECG_adv_Gen mainline, not as permission to drift from the
  active evidence registry or project constraints.

Current repository navigation, initialized 2026-05-29:

```text
agent entrypoint:       AGENTS.md
human project overview: README.md
stable package code:    ecg_adv_gen/
experiment configs:     configs/defaults/, configs/experiments/
active run index:       configs/active_scripts.yaml
active evidence facts:  configs/active_evidence_registry.yaml
label mapping evidence: configs/label_mappings/, docs/labeling/
managed CLIs:           scripts/run_experiment.py, scripts/agent/
legacy experiment CLIs: scripts/paper/, scripts/triple_labels/, scripts/pgd_cross_center/
long-term docs:         docs/pipelines/
archived reports:       docs/reports/archive/
external model handles: model/
tests:                  util/tests/  # project-wide tests; future target is tests/
```

AGENTS.md maintenance:

- Keep the first 100 lines focused on shared-server safety, current host
  overrides, and agent entrypoints.
- Move historical reports and dated exploration notes to
  `docs/reports/archive/YYYYMMDD/`; cite them from pipeline docs or the active
  evidence registry instead of expanding this file.
- Treat `/root/...` commands below as original AutoDL historical examples unless
  the current migrated host override above gives an explicit `/home/linbinhao`
  replacement.
- Detailed cleanup suggestions are tracked in
  `docs/pipelines/agents_md_organization_suggestions_20260529.md`.

Do not move active legacy entrypoints unless `configs/active_scripts.yaml` and
the tests are updated in the same change. Prefer extracting pure logic into
`ecg_adv_gen/` while keeping old script paths as thin wrappers.

Every new managed experiment should leave an agent-readable run record:
`run_card.json`, `run_file_index.json`, `summary.md`, and logical category
directories (`configs/`, `manifests/`, `logs/`, `checkpoints/`, `eval/`,
`diagnostics/`, `reports/`, `artifacts/`). Record the experiment purpose and
result summary before treating the run as handoff-ready.

This host is not running as root. Do not assume `/root/autodl-tmp` or
`/root/miniforge3/envs/ECGTwin/bin/python` are accessible here unless a later
setup step creates those paths. Use the migrated data path above for current
commands, or pass explicit CLI paths.

Shared server cluster constraints:

- Do not use `sudo`.
- Do not update the Linux kernel.
- Do not install, replace, or upgrade CUDA.
- Do not install, replace, or upgrade NVIDIA/GPU drivers.
- Do not make system-level environment changes.
- Use only user-level or project-level environments.
- Keep all files touched by this project under the current user's home tree,
  especially `/home/linbinhao/ECG_adv_Gen` and
  `/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp`.
  Do not create, edit, delete, chmod, chown, or relink files outside the user's
  home tree unless the user explicitly requests it.
- Before starting GPU training or long inference, check current GPU usage with
  `nvidia-smi` and explicitly select intended free GPU(s), for example with
  `CUDA_VISIBLE_DEVICES=...`. Do not assume all 8 GPUs are available.
- Prefer single-GPU runs first. Use multi-GPU only when the user asks for it or
  when the available cluster state clearly makes it safe.
- Avoid broad process commands such as `pkill python`, `killall`, or unscoped
  `kill`. Only stop PIDs that have been confirmed to belong to this user's
  current experiment.
- Do not overwrite existing experiment directories by default. Write new runs to
  date/config-named output directories, and use `--force` only for clearly owned
  temporary or failed runs.
- Bind local web servers to `127.0.0.1`; if the requested port is occupied, do
  not kill the existing process unless it is confirmed to be this user's server.
- Keep large checkpoints, datasets, feature caches, logs, and generated samples
  out of git. Store them under user-owned data/output directories.
- CPU, memory, and disk IO can be used more aggressively when the machine is not
  under pressure, but first check the current load/free memory/disk state for
  long or heavy jobs. Scale `num_workers`, batching, and parallel preprocessing
  back down if shared resources become tight.
- Avoid launching multiple heavy WFDB/PN2021 feature-extraction or cache-build
  jobs in parallel unless current CPU, memory, and IO load are clearly low.

Repo-tracked Codex skill copy:

```text
.codex/skills/ecg-adv-gen/SKILL.md
```

The active runtime skill lives outside git. On the original root AutoDL host it
was `/root/.codex/skills/ecg-adv-gen/SKILL.md`; on this migrated user host it is
`/home/linbinhao/.codex/skills/ecg-adv-gen/SKILL.md`. When moving hosts, copy
the repo-tracked skill into the active Codex user's runtime skill directory.

## Environment

- Repo root: `/root/ECG_adv_Gen`
- Python env: `/root/miniforge3/envs/ECGTwin/bin/python`
- Hardware target: RTX 4090D 24GB VRAM, 15 CPU cores, 80GB RAM.
- Current migrated host observation: 8x NVIDIA RTX 4090 24GB GPUs are visible,
  with the `ECGTwin` environment under `/home/linbinhao/micromamba/envs/ECGTwin`.
- Optimize future training/preprocessing for this hardware profile:
  - prefer AMP/bf16 where numerically safe on the 4090D;
  - keep large arrays/checkpoints/caches under `/root/autodl-tmp/`;
  - use mmap or streaming for PN2021/PN2021-C instead of repeatedly
    decompressing large `.npz` files;
  - set DataLoader `num_workers` from CLI and tune around 4-8 before using all
    15 CPU cores;
  - use `pin_memory`, `persistent_workers`, and `prefetch_factor` for long
    training jobs when `num_workers > 0`;
  - avoid materializing every corrupted PN2021-C copy unless disk has been
    expanded substantially.
- System disk is small. Put large outputs, checkpoints, samples, caches, and logs under `/root/autodl-tmp/`, not the repo.
- Disk hygiene as of 2026-05-03 after user expansion and PN2021-C materialization: `/` has about 9.6 GB free and `/root/autodl-tmp` is about 350 GB total with about 54 GB free. Keep long-run temp/cache paths on the data disk, for example `TMPDIR=/root/autodl-tmp/tmp` and `XDG_CACHE_HOME=/root/autodl-tmp/cache` when safe for a command. Do not materialize another full PN2021-C copy without checking free space first.
- Do not move/delete repo historical artifacts or Git objects just to free disk unless the user explicitly approves; prefer package caches, bytecode caches, and new experiment outputs under `/root/autodl-tmp`.
- `rg` may be unavailable in this environment. Use `find`, `grep`, `sed`, `nl`, and `wc` when needed.
- Use `apply_patch` for manual file edits. Do not overwrite unrelated dirty worktree changes.

## Current Graduation Project Story

Mainline method:

```text
PTB-XL super5 real ECG
-> EfficientNet1DV2 super5 baseline
-> ECGTwin author training reproduction (IBE + DiT) with thesis-grade logs
-> ECGTwin VAE latent manifold for target-center augmentation
-> TA-OMAT / synth-anchor ablations
-> PhysioNet/CinC 2021 7-center AUROC/AUPRC evaluation
```

Current preferred thesis route:

```text
reproduce ECGTwin author pipeline
-> use ECGTwin VAE latent space for target-anchored on-manifold augmentation
-> keep direct ECGTwin synthetic augmentation as an ablation, not the main claim
```

Latest 2026-05-23 project state after migration:

```text
Most stable mainline:
  target-center real ECG anchors
  -> ECGTwin VAE latent space
  -> same-label / real_all_present Latent-Hull online adversarial training
  -> PN2021 held-out ref-excluded + PN2021-C evaluation

Center prompt token:
  useful for ECGTwin generation control and ablations, but matched no-token
  controls often equal or exceed it downstream. Do not present center-token
  synthetic candidates as the main causal source of improvement unless new
  matched controls prove it.

Direct ECGTwin DiT synthetic augmentation:
  keep as reproduced generative framework and ablation evidence, not the main
  classifier-improvement claim.

ECGFounder branch:
  docs/pipelines/ecgfounder_frozen_linear_probe_pipeline.md contains a new
  2026-05-23 strong branch. ECGFounder residual-adapter + VAE-only Latent-Hull
  online AT with source-logit anchor improves four target centers strongly while
  preserving PTB-XL source performance. Treat this as a promising new mainline
  candidate that still needs multi-seed replication and clear comparison
  against EfficientNet1DV2.
```

Latest VAE-online AT lessons, frozen 2026-05-27:

```text
Primary evidence docs:
  docs/reports/archive/20260525/vae_online_at_closing_summary_20260525.md
  docs/reports/archive/20260523/vae_only_24h_summary_20260523.html
  .codex/skills/ecg-vae-online-at/SKILL.md

Historical snapshot protocol from 2026-05-27:
  PN2021 Super5 v6 clinician-review mapping, four target centers
  (ningbo, chapman_shaoxing, cpsc_2018, georgia), fixed K=500 target
  ECGs per center, K500 ref ids excluded from final target evaluation.
  Use 10% target data only as sensitivity analysis unless explicitly changed.
  Current managed runs should use the v7 SJR/RGQ mapping recorded in
  configs/active_scripts.yaml and configs/active_evidence_registry.yaml.

Most effective current method for a clean VAE contribution:
  EfficientNet1DV2 direct-K500 checkpoint
  -> target-center real ECGTwin VAE latents
  -> local same-label / compatible latent-hull online AT
  -> multi-hot preserving latent labels.
  Best current standalone family:
    compatsoft / direct-init / latent_mixed_teacher / anchor-soft.
  Four-center mean:
    direct K500                 0.854322 / 0.487952
    best VAE-family             0.872177 / 0.514767
    net gain over direct K500   +1.79 pp AUROC / +2.68 pp AUPRC
    drop-all-zero mean          0.901780 / 0.677552

Strongest ECGFounder number, but with more complex attribution:
  K500-internal classwise selector over matched direct/fullFT/VAE candidates
  reached 0.919757 / 0.643536.
  Gain vs frozen/head direct K500: +1.56 pp AUROC / +4.67 pp AUPRC.
  Gain vs init-head direct fullFT: +0.62 pp AUROC / +1.30 pp AUPRC.
  Do not claim most ECGFounder gain comes from VAE. Most gain comes from
  target-center K500 supervised adaptation, head initialization, and fullFT
  recipe; VAE is a smaller refinement and useful selector-diversity source.

Technical details to preserve as main-method defaults:
  - Treat ECGTwin VAE as an on-manifold adversarial regularizer, not as proof
    that direct ECGTwin DiT synthetic samples are the causal source of gain.
  - Use real target-center ECG anchors. Decode from ECGTwin VAE, reorder leads
    with ECGTWIN_TO_PTBXL_INDICES, then resample 1024 -> 1000 for 100Hz
    EfficientNet-style classifiers.
  - Standardize VAE latents before distance search, mixup, or PGD.
  - Prefer local same-label / exact-positive-set partners; compatible-label
    soft BCE is an ablation. Do not mix NORM with abnormal labels in the main
    recipe.
  - Include the original anchor in the latent hull and keep anchor-dominant
    coefficients. Keep hull_lambda low-to-mid, usually 0.05-0.15.
  - Keep clean K500 anchors in the batch whenever latent mix or PGD is enabled.
  - Prefer multi-hot preserving labels such as latent_mixed_teacher or
    anchor-soft over argmax-collapsed hard labels for mixed latents.
  - Use K500-internal validation plus a PTB-XL/source clean floor for checkpoint
    selection. Do not select with full target-center heldout labels or known
    target-center test class distributions.
  - Log attack diagnostics for cited runs: atk_init, atk_anchor, loss_gain,
    clean/adversarial BCE, ASR, decoded invalid rates, source floor, and
    per-center/per-class target deltas.
  - Practical ASR target is roughly 30-70%. Use attack success as a control
    signal, not as the objective.

Mistakes and near-null routes already explored:
  - Do not attribute ECGFounder K500/fullFT gains to VAE alone.
  - Do not tune the final method from full target-center class distribution,
    heldout test labels, or heldout-oracle checkpointing. Exploration may look
    at heldout feedback, but final claims need a frozen global recipe.
  - Do not blindly increase adversarial weight, PGD steps, epsilon, or
    hull_lambda. Too-strong attacks hurt Ningbo/Georgia or source performance;
    too-weak attacks collapse atk_init/atk_anchor and add no useful pressure.
  - Do not revive heavy victim-score quality gates as the main mechanism.
    Keep only hard rejection for non-finite, flatline, severe amplitude, or
    physically invalid decoded signals.
  - Do not rely on last-epoch or heldout-oracle selection to make a candidate
    look better; robust overfitting and per-center tradeoffs were observed.
  - Do not expect selector-grid expansion alone to unlock the result. Adding
    more historical candidates, source-partner-only variants, label-mode-only
    variants, or compatible-neighbor soft-label smoothing did not beat the
    current EfficientNet ceiling.
  - Do not treat raw ECG AugMix/corruption consistency as solved. On
    2026-05-30, target-only raw corruption consistency gave only about
    +0.04 to +0.10 PN2021-C AUPRC pp, and PTB-XL source-scope raw corruption
    hurt clean/corrupted four-center means. The best tested AugMix variant
    remains the original mild latent-branch s2, but its gain over VAE noAug is
    only about +0.05 AUROC pp / +0.10 AUPRC pp on PN2021-C.
  - Do not treat ECGFounder feature adapters, last-block-only EfficientNet
    updates, blind full-encoder fine-tuning, stronger rare-ranking alone, or
    source-logit anchoring alone as solved routes. They were stable in parts
    but did not improve the four-center mean enough.
  - Do not overclaim low-K results. K=20/50 were unstable; K=100 has signal in
    some ECGFounder short-horizon runs; fixed K=500 is the clean main protocol.
```

For current experimental truth, prefer `configs/active_evidence_registry.yaml`,
`configs/active_scripts.yaml`, and the latest `docs/pipelines/*.md` over older
command snippets in this file. The current managed EfficientNet/ECGFounder
mainline uses the SJR/RGQ-reviewed v7 PN2021 Super5 mapping:

```text
SUPER5_PN2021_MAPPING_VERSION = v7_super5_sjr_rgq_review_20260528
SUPER5_PN2021_MAPPING_HASH = 555ec85d5b51
```

Older v3/v5/v6 command names and result files are historical unless they are
explicitly listed as active or smoke/superseded in `configs/active_scripts.yaml`.
Report the mapping version/hash with all final metrics.

Do not make no-IBE ECGTwin self-training mandatory for the thesis mainline. The
previous Scheme B/no-IBE implementation is archived under
`trash/cleanup_20260430/deprecated_no_ibe/` and should only be used as historical
reference unless the user explicitly revives it.

Do not make MIMIC mandatory for the thesis mainline. PTB-XL + PN2021 gives the
clean closed loop; MIMIC is an optional noisy OOD evaluation/pretraining source.

Primary plan document:

```text
docs/experiment_summary_for_advisor.md
docs/module_ablation_1_real_vs_synth.md
trash/docs_cleanup_20260501/historical_no_ibe/ecgtwin_no_ibe_diffusion_augmenter_recommended_plan.md  # historical Scheme B plan
```

## Key Data And Artifact Paths

- ECGTwin original repo: `model/ECGTwin/`
- ECGTwin author reproduction scripts: `scripts/ecgtwin_author_repro/`
- Archived no-IBE Scheme B implementation: `trash/cleanup_20260430/deprecated_no_ibe/`
- PTB-XL raw/preprocessed: `datasets/PTBXL/`, `/root/autodl-tmp/ptbxl/`
- PTB-XL VAE/nomic cache: `datasets/PTBXL/PTBXL_vae_multi_nomic.pt`
- PN2021 centers: `/root/autodl-tmp/physionet2021/training/<center>/`
- MIMIC raw/cache root: `/root/autodl-tmp/MIMIC/`
- ECGTwin latent MIMIC cache: `/root/autodl-tmp/ECGTwin_Data/Mimic_vae.pt`
- ECGTwin paired MIMIC cache: `/root/autodl-tmp/ECGTwin_Data/paired_Mimic_vae_multi_nomic.pt`
- ECGTwin author repro outputs: `/root/autodl-tmp/ecgtwin_author_repro/<run_name>/`
- EfficientNet super5 outputs: `/root/autodl-tmp/triple_labels/<run_name>/`

External model repos are expected to live on the data disk and be linked into
`model/`:

```text
model/DeepECG                 -> /root/autodl-tmp/models/DeepECG
model/ECGTwin                 -> /root/autodl-tmp/models/ECGTwin
model/advdiff                 -> /root/autodl-tmp/models/advdiff
model/ecg_ptbxl_benchmarking  -> /root/autodl-tmp/models/ecg_ptbxl_benchmarking
```

On a new AutoDL host, run:

```bash
bash scripts/bootstrap_model_repos.sh
```

`.gitmodules` is currently an external-model URL manifest, not active gitlink
submodules; `git submodule update --init` should not be relied on unless
`git ls-files --stage | grep '^160000'` shows real gitlinks.

Previously trained baseline:

```text
/root/autodl-tmp/triple_labels/super5/best_model.pt
/root/autodl-tmp/triple_labels/super5/training_log.json
/root/autodl-tmp/triple_labels/super5/train_result.json
/root/autodl-tmp/triple_labels/super5/eval_result_NORMguard.json
```

Baseline metrics remembered:

```text
PTB-XL fold10 AUROC/AUPRC ~= 0.9064 / 0.7754
PN2021 7-center NORMguard avg AUROC/AUPRC ~= 0.8390 / 0.5889
```

Archived first Scheme B PTB-XL diffusion run:

```text
/root/autodl-tmp/ecgtwin_class_super5/ptbxl_scheme_b_init/
best val loss ~= 0.03080
samples_n300/synth_waveforms.npz: 1500 ECGs, (N,1000,12), PTB-XL lead order
```

First real+synth classifier run:

```text
/root/autodl-tmp/triple_labels/super5_scheme_b_n300_r025/
PTB-XL fold10 AUROC/AUPRC ~= 0.9059 / 0.7671
PN2021 avg AUROC/AUPRC ~= 0.8337 / 0.5756
```

That first augmentation run did not beat the baseline overall. Treat it as
engineering closure and ablation evidence, not final evidence of benefit.

Current main experimental conclusion:

```text
TA-OMAT real-anchor latent PGD > direct synth-anchor diffusion for stability/speed.
ECGTwin diffusion is still useful as the reproduced generative framework and as
an ablation, but the final EfficientNetV2 enhancement claim should rely on the
ECGTwin VAE latent manifold plus target-center anchors.
```

## ECGTwin Facts To Preserve

- Original ECGTwin VAE input/output: `(B, 1024, 12)`, channels-last, raw mV, ECGTwin/MIMIC lead order.
- VAE latent: `(B, 4, 128)`.
- PTB-XL latent cache is already scaled by `0.18215`.
- ECGTwin/MIMIC lead order differs from PTB-XL by aVL/aVF swap.
- Convert decoded ECGTwin output to PTB-XL order before classifier training:

```python
ECGTWIN_TO_PTBXL_INDICES = [0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11]
```

- Classifier input format should be `(N, 1000, 12)`, PTB-XL lead order, 100Hz, 10 seconds.
- Original ECGTwin DiT text path uses:

```text
text_embed:      (B, L, 768)
text_embed_mask: (B, L)
pat_info:        (hr, age, sex)
text_projector + CrossAttention
```

- Original IBE provides `base_vector` for AdaLN modulation. Preserve this fact
  for ECGTwin reproduction. Do not introduce a no-IBE replacement unless the
  user explicitly revives the archived Scheme B branch.
- Never make `text_embed_mask` all zeros. Use `null_text_embed + mask=1` for unconditional text dropout; all-zero masks can produce softmax NaN.

## Super5 Labels

Class order:

```text
CD, HYP, MI, NORM, STTC
```

Source of truth:

```text
ecg_adv_gen/labels/super5_mapping.py
```

`scripts/triple_labels/label_schemes.py` is now a legacy compatibility wrapper
for Super5. It re-exports the package-owned Super5 API while keeping sub23/pn26
logic local for older callers.

PN2021 super5 mapping:

```text
SUPER5_PN2021_MAPPING_VERSION = v7_super5_sjr_rgq_review_20260528
SUPER5_PN2021_MAPPING_HASH = 555ec85d5b51
PN2021_EVAL_CACHE_VERSION = v7_super5_sjr_rgq_review
```

- PN2021 has no official `SNOMED -> PTB-XL super5` crosswalk. The mapping is a
  project-defined semantic projection for external-center evaluation.
- Current PN2021 v7 splits direct positives and NORM suppression:
  `SNOMED_TO_SUPER5_POSITIVE`, `NORM_POSITIVE_SNOMEDS`,
  `NORM_SUPPRESS_SNOMEDS`.
- `Q wave abnormal` and `early repolarization` are suppress-only by default, not
  direct STTC positives.
- `sinus bradycardia`, `sinus tachycardia`, and `sinus arrhythmia` are not
  treated as PTB-XL-normal equivalents by default; they suppress NORM.
- If PN2021 mapping, parser, preprocessing, or class order changes, bump cache
  version and rebuild the PN2021 eval cache through the managed local paths in
  `configs/local/*.yaml`.

Historical Scheme B prompt fragments:

| class | prompt |
|---|---|
| NORM | `normal ecg|sinus rhythm` |
| MI | `myocardial infarction|pathological q wave` |
| STTC | `st-t change|st segment abnormality|t wave abnormality` |
| CD | `conduction disturbance|bundle branch block` |
| HYP | `ventricular hypertrophy|left ventricular hypertrophy` |

For multi-label samples, concatenate fragments with `|`.

## Implementation Layout

Keep original `model/ECGTwin/` intact. Current active new code should live in:

```text
scripts/ecgtwin_author_repro/      # ECGTwin author IBE + DiT reproduction
scripts/pgd_cross_center/          # TA-OMAT / synth-anchor ablations
scripts/triple_labels/             # EfficientNet1DV2 training/eval
util/ecg_digital_features.py       # digital ECG validation
```

Archived no-IBE implementation:

```text
trash/cleanup_20260430/deprecated_no_ibe/methods/ecgtwin_class_super5/
trash/cleanup_20260430/deprecated_no_ibe/scripts/ecgtwin_class_super5/
```

## Current Training And Sampling Commands

ECGTwin author reproduction, stage 1 IBE:

```bash
/root/miniforge3/envs/ECGTwin/bin/python -u scripts/ecgtwin_author_repro/train_ibe_repro.py \
  --output_dir /root/autodl-tmp/ecgtwin_author_repro/<run>/ibe_stage1 \
  --epochs 40 --batch_size 65536 --mini_batch_size 512 \
  --num_workers 4 --pin_memory --persistent_workers --prefetch_factor 2 \
  --drop_last --amp --amp_dtype bf16 --device cuda
```

ECGTwin author reproduction, stage 2 DiT:

```bash
/root/miniforge3/envs/ECGTwin/bin/python -u scripts/ecgtwin_author_repro/train_dit_repro.py \
  --output_dir /root/autodl-tmp/ecgtwin_author_repro/<run>/dit_stage2 \
  --ibe_path /root/autodl-tmp/ecgtwin_author_repro/<run>/ibe_stage1/checkpoints/IBE_best.pth \
  --epochs 30 --batch_size 512 --num_workers 4 \
  --pin_memory --persistent_workers --prefetch_factor 2 \
  --amp --amp_dtype bf16 --device cuda
```

One-shot pipeline:

```bash
bash scripts/ecgtwin_author_repro/run_author_repro_pipeline.sh
```

Synthetic classifier input:

```text
signals: (N, 1000, 12) float32, PTB-XL lead order, 100Hz
labels:  (N, 5) float32, class order CD/HYP/MI/NORM/STTC
```

Classifier augmentation is implemented in `scripts/triple_labels/train_ptbxl.py` with:

```text
--synth_npz
--synth_ratio
```

PN2021 eval:

```bash
/root/miniforge3/envs/ECGTwin/bin/python scripts/triple_labels/eval_crosscenter.py \
  --scheme super5 \
  --model_dir /path/to/model_dir \
  --skip_mimic
```

## PN2021 Evaluation Rules

Use these 7 centers:

```text
chapman_shaoxing
cpsc_2018
cpsc_2018_extra
georgia
ningbo
ptb
st_petersburg_incart
```

Hard-exclude `ptb-xl` / `ptbxl` shards because they leak PTB-XL training data.

Main metrics:

```text
macro AUROC
macro AUPRC
per-center delta
per-class delta
```

## MIMIC Strategy

MIMIC should be optional pretraining:

```text
MIMIC single-sample latent pretrain
-> PTB-XL fine-tune
-> sample only from PTB-XL fine-tuned checkpoint
-> EfficientNetV2 real+synth evaluation
```

Use `/root/autodl-tmp/ECGTwin_Data/Mimic_vae.pt` first. It has 744,372 single ECG latents, each with label fields:

```text
subject_id, ecg_time, text, hr, age, sex
```

Do not use full paired MIMIC as a default single-sample/no-IBE route. The paired file has about 6.4M pairs and exists for original IBE reference-target training. After removing IBE, the reference ECG has no clean role and the pair expansion wastes compute.

MIMIC report labels:

```python
from scripts.triple_labels.label_schemes import mimic_report_to_super5
```

Apply PTB-XL NORM semantics:

```text
if any abnormal class CD/HYP/MI/STTC is positive, force NORM=0
```

MIMIC rough guarded distribution:

```text
N = 744,372
mapped nonzero ~= 737,554
CD   ~= 178,311
HYP  ~=  86,608
MI   ~= 167,472
NORM ~= 257,351
STTC ~= 298,721
```

Use patient-level split by `subject_id`, not random ECG split.

## Generation Quality Validation

ECGTwin author's validation stack:

- Signal level: FID, improved Precision, Recall, F1 in ECG feature space.
- Feature/physiology level: HR-MAE between generated ECG HR and target condition HR.
- Diagnostic/semantic level: ECG-text CLIP Score.
- Personal consistency: base-vector t-SNE, similarity score, silhouette coefficient.
- Downstream utility: auto-diagnosis improvement.
- Qualitative/case analysis: 12-lead figures, attention maps, prompt-to-prompt editing cases.

For current super5 downstream utility and any historical no-IBE/synthetic ablation, use a task-adapted stack:

1. Basic signal sanity using `util/ecg_viz.sanity_check`:
   - NaN/Inf
   - flatline/saturation
   - DC offset
   - amplitude p2p
   - Einthoven and aVR residuals
   - HR estimate and HR range
2. Digital ECG criteria using `util/ecg_digital_features.py` and `docs/ecg_digital_thresholds.md`:
   - HR, RR-CV, P wave, PR, QRS, ST/T, voltage criteria.
   - Existing prior digital validation suggests NORM/MI/STTC are stronger; HYP/CD may fail absolute voltage or detailed conduction criteria.
3. Class/prompt consistency:
   - Use the frozen EfficientNet1DV2 super5 baseline as a victim classifier.
   - Generated class should raise the intended class probability.
   - For NORM, abnormal probabilities should stay low.
   - Use this as a filter, not as the only proof.
4. Feature distribution:
   - Compute real-vs-synth feature FID/rFID and precision/recall/F1 using EfficientNet penultimate features or ECGTwin CLIP features.
   - Compare synthetic samples to PTB-XL fold 1-8 and fold 9.
5. Downstream utility:
   - PTB-XL fold10 macro AUROC/AUPRC.
   - PN2021 7-center macro AUROC/AUPRC.
   - per-class and per-center deltas.

Expected `eval_synth.py` artifacts:

```text
synth_quality_summary.json
per_sample_quality.csv
per_class_quality.csv
victim_scores.csv
feature_distribution.json
digital_criteria_report.md
figures/*.png
```

Do not claim generated ECGs are clinically valid solely because they improve a classifier. The thesis argument should combine visual quality, physiology sanity, prompt/class consistency, feature distribution, and downstream external evaluation.

## Visualization

Use `util/ecg_viz.py`.

Important functions:

```text
plot_ecg
plot_with_report
sanity_check
plot_ecg_ecgtwin_style
plot_ecg_ecgtwin_gallery
```

`plot_ecg_ecgtwin_gallery` was added to show all 12 leads in one large figure with enough vertical row spacing to avoid peaks overlapping adjacent leads.

Gallery-safe output previously generated under:

```text
/root/autodl-tmp/ecgtwin_class_super5/ptbxl_scheme_b_init/ecgtwin_gallery_safe/
```

## Logging Requirements

Every diffusion run should save:

```text
run_config.json or config.yaml
train.log
metrics.jsonl
loss_curve.csv
loss_curve.png
checkpoints/latest.pt
checkpoints/best.pt
samples/*.npz
figures/*.png
```

Every downstream classifier run should save:

```text
training_log.json
train_result.json
eval_result.json
baseline_vs_synth_metrics.csv
pn2021_per_center_delta.csv
```

## Fallbacks

- If `class_text` is unstable or hurts fold9, use `class_only` as the thesis main result and keep text prompt as compatibility/demo.
- If HYP/CD synthetic samples fail quality gates, only use trusted classes such as NORM/MI/STTC for augmentation and report HYP/CD as limitations.
- If MIMIC pretraining does not improve downstream metrics, write it as an ablation: large-scale MIMIC improves/changes latent modeling but noisy report labels and single-center domain shift limit augmentation benefits.
- If PN2021 average does not improve, analyze per-center/per-class deltas and make the claim narrower.
