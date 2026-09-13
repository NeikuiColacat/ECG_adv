---
name: shared-gpu-server-discipline
description: Check resources and job ownership before ECG GPU/heavy CPU/IO work or local web serving, and safely hand off, pause or resume authorized jobs.
---

# Shared Server Execution

Use the first 100 lines of `AGENTS.md` for host and safety rules; reread after
compaction/resume/handoff before heavy work. This skill grants no new resources.

## Before Actual Launch or Resume

1. Confirm checkout, dirty state, exact task/output scope and existing jobs.
2. For GPU work inspect `nvidia-smi` and compute-process ownership; choose only
   confirmed free devices with explicit `CUDA_VISIBLE_DEVICES` (UUID preferred).
   Preserve the user's GPU/concurrency ceiling and the workflow allowlist.
3. Check CPU load, available RAM, disk and requested tmpfs capacity. Scale
   workers/concurrency down under pressure; do not run parallel heavy cache
   builds merely because GPUs are free.
4. Use the retained launcher and dry-run, fresh external outputs, exact command,
   environment and task-owned logs/receipts. Never overwrite a previous run.

Prefer one GPU unless more are authorized or live state clearly permits it;
using all GPUs still requires explicit approval. A historical idle-context
exception is not current permission to share someone else's occupied GPU.

## Temporary Storage and Long Jobs

If RAM intermediates are explicitly requested, use a private task-scoped RAM
directory (for example under `/dev/shm`) and check capacity. This is a scoped
exception to the home-only write rule, not general permission outside home.
Keep necessary checkpoints, predictions and final provenance in the authorized
external data root. Do not silently spill temporary outputs to the SSD.
If all disk writes are forbidden, clarify the final-output destination before
launch instead of overriding that restriction for durable artifacts.

Verify real task progress and actual compute, not only reserved VRAM. Then let
the managed queue run in the background and hand off its status/control path;
avoid agent polling unless monitoring is requested. Do not change live source
or remove source/hash guards to make a handoff pass.

For pause/recovery, verify PID/UID/start-time identity and use the task's own
pause/drain controls. Stop only exact authorized task-owned PIDs, never broad
process names. Do not reuse stale PID or GPU-number assumptions.

## Stop and Clarify

Stop before an unauthorized outside-home write, system/kernel/CUDA/driver
change, foreign process/port mutation, overwrite, or external exposure of ECG
data/secrets. Local servers bind to 127.0.0.1; keep unrelated listeners intact.
Resource pressure means reducing or waiting for capacity, not taking other jobs.
