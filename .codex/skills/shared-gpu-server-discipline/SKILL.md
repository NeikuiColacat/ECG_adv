---
name: shared-gpu-server-discipline
description: Use before launching any ECG_manual_refactor GPU training, inference, cache build, long evaluation, local web server, or other shared-server resource-heavy job.
---

# Shared GPU Server Discipline

This project runs on a multi-user shared server. Treat safety and traceability as
part of the experiment contract.

## Non-negotiables

- Do not use `sudo`.
- Do not update the Linux kernel, CUDA, NVIDIA drivers, or system packages.
- Do not modify system-level environments.
- Keep project operations under `/home/linbinhao`.
- Before any GPU job, read the first 100 lines of `AGENTS.md` after context
  compaction, resume, or a new Codex handoff.
- Before GPU training or long inference, run `nvidia-smi` and explicitly choose
  free GPUs with `CUDA_VISIBLE_DEVICES=...`.
- Default to one GPU. Use more only when the user explicitly allows it or the
  machine is clearly idle.
- Never use broad process commands such as `pkill python`, `killall`, or
  unscoped `kill`.
- Bind local web servers to `127.0.0.1`. If a port is occupied, do not kill the
  process unless it is confirmed to belong to the current user and task.

## Preflight

1. Confirm working directory is inside `/home/linbinhao`.
2. Check `git status --short --branch`.
3. Check GPU state with `nvidia-smi` for GPU jobs.
4. For long CPU or IO jobs, check load, memory, and disk space.
5. Choose a date/config-named output directory that does not overwrite previous
   runs.
6. Record GPU ids, command, run id, log path, checkpoint path, and manifest path.

## Stop Conditions

- The command would write outside `/home/linbinhao`.
- The command would install or upgrade system packages, CUDA, drivers, or kernel.
- The command would use all GPUs without explicit user approval.
- The command would overwrite a non-temporary experiment directory.
- The command would expose raw ECG data, credentials, or private artifacts to an
  external service without explicit approval.
