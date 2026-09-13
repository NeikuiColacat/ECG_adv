---
name: artifact-git-guard
description: Check exact ECG staged changes before git add, commit or push to exclude artifacts, credentials, external model handles and unrelated dirty work.
---

# Artifact Git Guard

Use before staging, committing or pushing. An edit/review request alone does not
authorize these actions; permission to commit is not permission to push.

## Inspect the Exact Change

1. Check `git status --short --branch`, the unstaged diff and staged diff.
2. Stage only selected paths/hunks owned by the requested task; preserve other
   people's or unrelated dirty edits. Do not stage broad directories blindly.
3. Exclude datasets, checkpoints, waveform/features caches, generated samples,
   run outputs and large binaries; tiny intentional test fixtures need review.
4. Keep `model/` external links/payloads, local configs, secrets, tokens, keys
   and `.env` out unless their specific non-secret change is explicitly scoped.
   Never expose credentials in diagnostic output.
5. Run `git diff --check` and `git diff --cached --check`; verify touched behavior
   according to AGENTS. Docs-only edits need no training or full model load.

Suspicious extensions: `.pt .pth .ckpt .npz .npy .h5 .pkl`.
Suspicious directories: `runs/ outputs/ checkpoints/ mlruns/ wandb/ .dvc/cache/`.
These are inspection hints, not permission to delete matching files.

Report the actual commit/push outcome, validation, and what stayed unstaged.
Do not claim a commit was made merely because the working diff is ready.
