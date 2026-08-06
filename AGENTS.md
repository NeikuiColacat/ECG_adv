# ECG Manual Refactor Agent Instructions

## Shared Server Safety

This machine is a multi-user shared server. Other users' files, environments,
processes, ports, GPU jobs, and outputs are off-limits unless the user
explicitly says otherwise.

- Do not use `sudo`.
- Do not update the Linux kernel.
- Do not install, replace, or upgrade CUDA or NVIDIA drivers.
- Do not make system-level environment changes.
- Use only user-level or project-level environments.
- Keep operations under `/home/linbinhao`.
- Do not create, edit, delete, chmod, chown, or relink files outside that home
  tree unless the user explicitly requests it.
- Do not use broad process commands such as `pkill python` or `killall`.
- Stop only PIDs confirmed to belong to this user's current task.
- Do not overwrite an existing experiment directory by default.
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

## Current Host

```text
repo root:    /home/linbinhao/ECG_manual_refactor_clean
data root:    /home/linbinhao/ECG_adv_data
python:       /home/linbinhao/micromamba/envs/ECGTwin/bin/python
keep policy:  docs/refactor_cleanup/manual_refactor_keep_manifest.md
launcher:     boot_scripts/run_experiment.py
active index: configs/active_scripts.yaml
evidence:     configs/active_evidence_registry.yaml
agent skills: .codex/skills/README.md
```

The old `ECG_adv_Gen` repository and Git history are read-only provenance
oracles. They are not active runtime dependencies of this clean-room rebuild.

## Clean-Room Boundary

Use
`docs/refactor_cleanup/manual_refactor_keep_manifest.md`
as the default build and review boundary.

- Reuse a retained module before creating a parallel implementation.
- New runtime code may depend only on retained files or artifacts produced by
  retained files.
- Do not import from `agent_workspace`.
- Do not import from the legacy `ecg_adv_gen`, `methods`, or `scripts` trees.
- Do not restore compatibility wrappers merely to keep old entrypoints alive.
- Keep exploratory scripts and intermediate outputs under `agent_workspace/`.
- Only the two explicitly retained performance-summary artifacts may remain
  active evidence there.
- Use `apply_patch` for manual file edits.
- Preserve unrelated dirty worktree changes.
- Do not delete legacy files until a generated candidate list has been reviewed
  and the user explicitly confirms deletion.

## Single Launch Surface

All retained experiments launch from tracked YAML through:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python \
  boot_scripts/run_experiment.py \
  --config configs/experiments/<experiment>.yaml \
  --dry-run
```

Run a dry-run before an actual launch. The launcher must:

- resolve the entire YAML closure inside the selected config bundle;
- accept only code-owned entrypoint names;
- reject dynamic imports and launcher-owned argument overrides;
- write outside the worktree;
- never load data, a model, or a GPU during dry-run;
- leave a run card, file index, summary, resolved config snapshots, and hashes
  for executed runs.

Do not add long Bash launchers or dated one-off Python entrypoints.

## Locked Research Contract

Current prospective method:

```text
PTB-XL source checkpoint
-> one target center's fixed K500 records
-> two-chain AugMix + SimCLR
-> clean plus rotating depth2/depth3 supervised adaptation
-> exact-label attack-then-contract VAE-LHAT BCE
-> K500-ref-excluded PN2021 Clean and PN2021-C evaluation
```

Data and evaluation:

- Super5 class order: `CD, HYP, MI, NORM, STTC`.
- PN2021 mapping version:
  `v7_super5_sjr_rgq_review_20260528`.
- Mapping hash: `555ec85d5b51`.
- Logical centers: `ningbo`, `chapman_shaoxing`, `cpsc_2018`, `georgia`.
- `cpsc_2018` includes CPSC 2018 Extra.
- Model adaptation sees only that logical center's fixed K500.
- Records outside K500 may be used only for development parameter assessment,
  never for model adaptation.
- Final evaluation excludes the K500 record identities.
- Primary table metrics are macro AUROC and sklearn average precision using
  `drop_all_zero`.
- `all_zero_kept` is secondary audit evidence.

Waveform contract:

- Canonical classifier bottleneck: raw physical mV, `(B,1000,12)`, 100 Hz,
  PTB-XL lead order.
- Corruptions happen before per-sample global z-score.
- PN2021-C operators run in the 500 Hz domain and return to canonical 100 Hz.
- EfficientNet consumes canonical 100 Hz.
- ECGFounder uses linear `1000 -> 5000` interpolation on device, then the same
  per-sample global z-score policy.
- ECGTwin VAE input/output is `(B,1024,12)` in ECGTwin lead order.
- VAE latent shape is `(B,4,128)`.
- Decoded ECGTwin output must be reordered with
  `[0,1,2,3,5,4,6,7,8,9,10,11]` before the canonical classifier path.

Method contract:

- Stage 1 uses clean versus one two-chain AugMix strong view with SimCLR.
- No Stage-1 VAE tail, VICReg, PTB-XL replay, or residual head.
- Stage 2 weights clean and corruption families `0.5 / 0.5`.
- The corruption schedule rotates two depth-2 and two depth-3 views.
- The logit-anchor teacher is a frozen post-Stage-1/pre-Stage-2 snapshot.
- VAE-LHAT uses exact-label non-self neighbors, `M=20`, hull lambda `0.6`,
  standardized L2 epsilon `12`, ten steps, and attack-then-contract selection.
- Base and VAE auxiliary gradients are added directly; PCGrad is excluded.
- The current recipe is heldout-tuned, single-seed development evidence, not a
  final paper claim.

## Evidence Discipline

Start from `configs/active_evidence_registry.yaml`.

- A tracked config proves replayability, not performance.
- A report proves presentation, not independent validation.
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
/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest -q \
  util/tests/test_augmentations.py \
  util/tests/test_torch_augmentations.py \
  util/tests/test_data_contracts.py \
  util/tests/test_labels_super5.py \
  util/tests/test_pn2021_corruptions.py
```

Before committing or pushing:

- run `git diff --check`;
- inspect `git status --short`;
- confirm no large artifacts, checkpoints, caches, secrets, host-specific
  model-link changes, or unrelated user edits are staged;
- use the artifact Git guard skill;
- never stage broad directories blindly.
