# ECG Manual Refactor Agent Instructions

Worktree focus: ECG LLM: PULSE / ECG-R1 image evaluation and PULSE fine-tuning.
Branch: `direction/ecg-llm`. Confirm the live branch before editing; this is a
navigation hint, not authority to run or delete experiments in any direction.

## Shared Server Safety

This machine is a multi-user shared server. Other users' files, environments,
processes, ports, GPU jobs, and outputs are off-limits unless the user
explicitly says otherwise.

- No `sudo`, kernel updates, CUDA/NVIDIA driver installation, replacement or upgrades,
  or system-level environment changes.
- Use only user-level or project-level environments.
- Keep operations under `/home/linbinhao`.
- Do not create, edit, delete, chmod, chown, or relink files outside that home
  tree unless the user explicitly requests it.
- Do not use broad process commands such as `pkill python` or `killall`.
- Stop only PIDs confirmed to belong to this user's current task.
- Do not overwrite an existing experiment directory by default.
- Do not edit source used by running jobs or bypass their source/hash guards.
- Bind local web servers to `127.0.0.1`.
- If a requested port is occupied, do not kill the listener unless it is
  confirmed to belong to this user and task.
- Keep checkpoints, datasets, waveform caches, generated samples, and verbose
  run logs outside Git.

Before GPU training, long inference, or a cache build:

1. Re-read the first 100 lines of this file after every compaction, resume, or
   handoff.
2. Check `nvidia-smi`.
3. Select only confirmed free devices through `CUDA_VISIBLE_DEVICES`.
4. Prefer one GPU unless the user requests more or the live state makes
   parallel use clearly safe.
5. Check load, free memory, and disk capacity before heavy CPU, RAM, or IO work.
6. Scale workers and concurrency down if the shared host is under pressure.

After a long job starts, verify real progress and hand it back to the background
queue. Do not keep an agent polling unless the user asks for monitoring.

## Current Host

```text
repo root:    current checkout (`git rev-parse --show-toplevel`)
data root:    /home/linbinhao/ECG_adv_data
python:       /home/linbinhao/miniforge3/envs/ECGTwin/bin/python
cli tools:    /home/linbinhao/miniforge3/envs/cli-tools/bin
keep policy:  docs/refactor_cleanup/manual_refactor_keep_manifest.md
launcher:     boot_scripts/run_experiment.py
active index: configs/active_scripts.yaml
evidence:     configs/active_evidence_registry.yaml
agent skills: .codex/skills/README.md
```

Legacy Git history is read-only provenance, never a runtime dependency.

## Clean-Room Boundary

The keep manifest is the default build and review boundary.

- Reuse a retained owner; prefer direct functions and explicit loops over new
  engines, plugins or one-use abstraction layers. Do not restore compatibility
  wrappers just to keep retired entrypoints alive.
- Keep data/model, training, evaluation and scheduling responsibilities clear.
  Remove internal duplicate checks only when one owner still enforces the same
  contract; preserve data identity, finite values, gradients, BN/RNG and hashes.
- Share fixes through reviewed Git commits, never cross-worktree imports.
  One writer owns each index; preserve historical experiments and identities.
- Runtime dependencies must be retained files or their artifacts. No imports
  from `agent_workspace` or legacy `ecg_adv_gen`, `methods`, or `scripts`.
- Exploratory code belongs in `agent_workspace/`; only its two explicitly
  retained performance summaries are active evidence. Keep large outputs external.
- Use `apply_patch`; preserve unrelated dirty worktree changes.
- Do not delete legacy files until a generated candidate list has been reviewed
  and the user explicitly confirms deletion. Retiring worktrees also requires
  a reviewed exact list and separate deletion confirmation.

## Single Launch Surface

All retained experiments launch from tracked YAML through:

```bash
/home/linbinhao/miniforge3/envs/ECGTwin/bin/python \
  boot_scripts/run_experiment.py \
  --config configs/experiments/<experiment>.yaml \
  --dry-run
```

Dry-run before execution: resolve the full YAML closure within the selected
bundle, accept only code-owned entrypoints, and reject dynamic imports or
launcher-owned overrides. Dry-run loads no data, model or GPU and has no writes.
Executed runs write outside the worktree and retain a run card, file index,
summary, resolved config snapshots and hashes.

Do not add long Bash launchers or dated one-off Python entrypoints.

## Locked Research Contract

Use `docs/directions.md` and `configs/directions.yaml` to select the direction.
The catalog is navigation, not another launcher or config generator.

- **ECG LLM:** native-500-Hz waveform -> image -> model-specific processor.
  ECG-R1's retained evaluation is image-only; do not claim waveform+image parity.
- **SimCLR:** fixed PTB-XL source -> target K500 -> two-chain AugMix + SimCLR
  -> clean/rotating corruption supervision + contracted VAE-LHAT.
  Selector: `configs/train/methods/augmix_simclr_lhat.yaml`.
- **JSD:** existing single-stage R18, JSD weight 1.5 and LHAT supervised mass
  0.20 replacing clean loss, not an extra 20% loss. No SimCLR, source replay,
  source logit anchor or stage boundary. Do not substitute two-stage JSD-12.
  Selector: `configs/train/methods/a1_rot4_two_chain_balanced_jsd1p5_vae_lhat_replace0p2.yaml`.

Shared Super5 data identity (including current LLM adaptation/evaluation):

- Super5 order: `CD, HYP, MI, NORM, STTC`; mapping
  `v7_super5_sjr_rgq_review_20260528` / `555ec85d5b51`.
- Centers: `ningbo`, `chapman_shaoxing`, `cpsc_2018` (including Extra), `georgia`.
- Adapt only on the selected center's fixed K500; final evaluation excludes
  those identities. Outside-K500 records are development assessment only,
  never adaptation data.
- Traditional primary metrics: macro AUROC and sklearn average precision, `drop_all_zero`,
  equal views then equal centers. `all_zero_kept` is secondary audit evidence.
  LLM generated-label metrics are not interchangeable with raw-logit AUROC/AP.

Traditional waveform contract (LLM uses its native500 image protocol above):

- Canonical input: raw physical mV, `(B,1000,12)`, 100 Hz, PTB-XL lead order.
  Corrupt before per-sample global z-score; PN2021-C runs in 500 Hz then returns
  to canonical 100 Hz. EfficientNet consumes 100 Hz; Founder linearly interpolates
  `1000 -> 5000` on device before the same global z-score.
- VAE I/O: `(B,1024,12)` in ECGTwin lead order; latent `(B,4,128)`.
  Reorder decoded leads with `[0,1,2,3,5,4,6,7,8,9,10,11]` before classification.

Locked SimCLR method details (not defaults for JSD or LLM):

- Stage 1: clean vs one two-chain strong view; retain frozen PTB-XL source-logit
  anchor. No VAE tail, VICReg, PTB-XL replay or residual head.
- Stage 2: clean/corruption weights 0.5/0.5; rotate two depth-2 and two depth-3
  views. Post-Stage-1 logit-anchor teacher disabled.
- LHAT: nearest exact-label non-self M=20, hull lambda=1.0, standardized L2
  epsilon=12, one attack step; attack-then-contract with linear clean/hard
  endpoint residual correction. Add base/auxiliary gradients directly; no PCGrad.
- Recipes remain heldout-tuned development evidence, not final paper claims.

## Evidence Discipline

Start from `configs/active_evidence_registry.yaml`.

- Configs prove replayability, not performance; reports are not independent validation.
- Keep development, historical trusted, and paper-final evidence separate.
- Do not select checkpoints or hyperparameters from heldout target labels or
  the full target-center class distribution.
- Report source checkpoint, config closure, seed identity, model/backbone,
  optimizer-step budget, mapping hash, K500 exclusion, and metric view.
- For VAE-LHAT, report raw-search ASR, contracted-view ASR, acceptance rate,
  BCE gain, selected contraction coefficient, and invalid-decode rate together.
- Do not attribute the full pipeline gain to VAE; use the matched no-VAE
  ablation for that claim.
- Paper promotion requires a recipe frozen without further heldout feedback,
  at least three independent repeats per backbone, mean/standard deviation,
  per-center deltas, and registered run/checkpoint hashes.

Historical `ecg_adv_gen` metrics are embedded as a provenance snapshot in the
active evidence registry. Recover historical code from commit
`3a39a516420b52c219a782a7b7440f82746f4b90` only when explicitly requested.

## Required Verification

Use the project Python directly:

```bash
/home/linbinhao/miniforge3/envs/ECGTwin/bin/python -m pytest -q
```

`pytest.ini` limits discovery to the retained `util/tests/test_*.py` contract
suite. Keep its inventory synchronized with A9 of the keep manifest.

Docs-only changes need link/scope checks and `git diff --check`, not GPU jobs.
Before staging, committing or pushing, use the artifact Git guard, run
`git diff --check`, inspect `git status --short` and the exact staged diff.
Exclude artifacts, checkpoints,
caches, secrets, model-link changes and unrelated edits; never stage broadly.
