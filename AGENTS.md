# Traditional JSD Agent Instructions

This checkout owns `direction/traditional-jsd`. Confirm the live branch and dirty
state before edits. Read [METHOD](docs/METHOD.md) for the scientific contract
and [EVIDENCE](docs/EVIDENCE.md) for historical provenance and recovery.

## Shared server safety

- Other users' files, jobs, processes, environments and ports are off-limits.
- No sudo, kernel updates, CUDA/NVIDIA-driver changes or system environment edits.
- Work under `/home/linbinhao`; use user/project environments.
- Do not write, delete, chmod, chown or relink outside that home without explicit scope.
- No broad process kills; stop only PIDs confirmed to belong to this task.
- Never overwrite an existing experiment output by default.
- Keep datasets, checkpoints, caches, generated samples and logs outside Git.
- Bind local servers to `127.0.0.1`; do not displace an unrelated listener.
- Before GPU work check `nvidia-smi`, select free GPUs through
  `CUDA_VISIBLE_DEVICES`, and prefer one device.
- Before heavy CPU/RAM/IO work check load, memory and disk; limit workers.
- Do not edit source used by running jobs or bypass source/hash guards.

## Code and configuration

Use `apply_patch`; preserve unrelated edits. Reuse existing owners, direct
functions and explicit loops; avoid compatibility wrappers and generic engines.
Keep data/model, training and evaluation responsibilities clear.
No imports from another worktree, historical code or `agent_workspace`.
Source recovery belongs to a separate review, never a runtime dependency.
Use one writer per file and separate branches/worktrees for conflicting work.

Launch from `boot_scripts/run_experiment.py --config <experiment> --dry-run`
before execution. YAML chooses only code-owned entrypoints and explicit config
references. Do not add long shell launchers, legacy queues or dated entrypoints.
Use fresh external run directories and retain run cards, file indexes, config
snapshots and hashes. Prefer the ten examples in `configs/experiments/`.
The project Python is `/home/linbinhao/miniforge3/envs/ECGTwin/bin/python`.

## Locked method

- Main: `a1_rot4_two_chain_balanced_jsd1p5_vae_lhat_replace0p2`.
- Profile: `configs/train/PN2021_a1_joint_long_budget_r13.yaml`.
- Existing R18 is single-stage, JSD 1.5; LHAT supervised mass 0.20 replaces
  clean loss. No SimCLR, source replay, source anchor or stage boundary.
- Do not substitute the historical two-stage JSD-12 recipe.
- Keep `a0_clean_v1` and `a1_corrupt_ft_rot4_v1` reference baselines.
- Preserve source checkpoint, budget, mapping, class order, seeds and K500 IDs.
- New evaluation uses `configs/eval/PN2021_online_gpu.yaml`; no disk corruption cache.
- Adapt only on the chosen center's K500. Exclude those identities in evaluation.
- Outside-K500 target labels are development assessment only; never use them
  for adaptation or checkpoint selection.
- Primary: drop-all-zero macro AUROC / sklearn AP, equal views then centers.
- Recipes are heldout-tuned development evidence, not paper-final claims.

## Verification and reporting

Preserve units, shapes, leads, finite values, gradients, freeze/eval state,
BatchNorm, RNG, optimizer budget, data identity and artifact hashes.
Deleting an apparently unused initializer can change RNG; duplicate checks may
be removed only if one owner still enforces the full contract.
For VAE claims report matched no-VAE comparisons, raw/contracted ASR, acceptance,
BCE gain, selected contraction coefficient and invalid-decode rate.
Do not compare online corruption metrics causally with historical NumPy cache metrics.

Run focused checks and the retained CPU suite for source/config changes:
`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 <python> -m pytest -q`.
Numerical training changes also need tiny old/new state and RNG/BN comparisons;
GPU or real-model claims require actual scoped runs.
Docs-only edits need scope/link checks and `git diff --check`.
Use [artifact-git-guard](.codex/skills/artifact-git-guard/SKILL.md) before staging,
committing or pushing; inspect exact paths and diff. No broad staging.
The authorized slim-down removes redundant tracked code/config/docs after an
exact deletion plan; it does not authorize deleting external results or worktrees.
