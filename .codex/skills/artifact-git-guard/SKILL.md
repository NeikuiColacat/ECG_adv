---
name: artifact-git-guard
description: Use before staging, committing, or pushing ECG_adv_Gen changes to prevent large artifacts, external model links, credentials, and generated runs from entering Git.
---

# Artifact Git Guard

Use this before any `git add`, commit, or push.

## Checks

1. Run `micromamba run -n cli-tools git status --short --branch`.
2. Inspect staged and unstaged files.
3. Do not stage external model link/payload changes under `model/*` unless the
   user explicitly asks.
4. Do not stage raw ECG data, datasets, checkpoints, generated samples, feature
   caches, run directories, `mlruns/`, `wandb/`, `.dvc/cache/`, or large binary
   artifacts.
5. Do not stage credentials, private keys, tokens, `.env`, or local machine
   secrets.
6. Run `git diff --check` and `git diff --cached --check` before commit.
7. Run the narrow relevant tests or smoke checks for touched code.

## Large Artifact Heuristic

Treat these as suspicious unless they are tiny fixtures intentionally tracked:

- `*.pt`, `*.pth`, `*.ckpt`, `*.npz`, `*.npy`, `*.h5`, `*.pkl`
- `runs/`, `outputs/`, `checkpoints/`, `mlruns/`, `wandb/`, `.dvc/cache/`

## Final Commit Summary

Report what was committed, what was intentionally left unstaged, and what
verification passed or could not be run.
