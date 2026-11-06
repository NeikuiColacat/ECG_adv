---
name: artifact-git-guard
description: Inspect exact ECG staged changes to exclude artifacts, credentials and unrelated work.
---

# Artifact Git Guard

Use before staging, committing or pushing. An editing request does not itself
authorize a commit or push; use existing session authorization when supplied.

1. Inspect git status --short --branch, unstaged diff and staged diff.
2. Stage selected task-owned paths/hunks only; preserve unrelated dirty work.
3. Exclude data, checkpoints, waveform/feature caches, generated samples,
   run outputs, environment directories and large binaries.
4. Exclude model/ local links, secrets, tokens, private keys and .env files.
   Do not expose credentials in diagnostics.
5. Run git diff --check and git diff --cached --check; inspect the exact staged
   diff and complete behavior-appropriate checks from AGENTS.md.

Inspect suspicious extensions (.pt .pth .ckpt .npy .npz .h5 .pkl) and directories
(runs/ outputs/ checkpoints/ mlruns/ wandb/). These hints never authorize deletion.
Tiny intentional fixtures and environment lock/manifest files need normal review.
Report actual commit/push outcomes and what remains unstaged; do not claim success
from a prepared working diff alone.
