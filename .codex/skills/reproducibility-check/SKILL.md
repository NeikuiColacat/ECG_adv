---
name: reproducibility-check
description: Audit ECG experiment config closure, run and checkpoint identity, replay evidence and paper-readiness without treating an audit as authorization to rerun.
---

# ECG Reproducibility Check

Separate config replayability, completed execution and validated performance.
A snapshot status is not a live-job check; a report is not independent evidence.

## Audit Existing Evidence First

- Locate the experiment YAML, complete resolved closure, exact command/cwd,
  environment versions, Git SHA and dirty/source snapshot identity.
- Bind source/final checkpoints and actual artifact bytes to their hashes,
  run card/file index, result schema and selection policy.
- Verify dataset/cache/mapping, class order, centers, K500 IDs/ref-exclusion,
  seed namespace/replicate, budgets and metric aggregation.
- Record missing checkpoints or fields explicitly. Metadata cannot prove that
  a weight still exists or can load; do not rewrite an old hash to fit new bytes.
- Distinguish ledger quick membership/size checks from full content verification;
  bound expensive hashing to what the claim requires and check resource pressure.

For matched comparisons, use `model-eval`. A scientific config hash, a run hash
and a source-code hash have different roles; do not treat them as interchangeable.

## Replay and Promotion

Read-only verification is the default for audit/report requests. Use a small
replay when it is within the requested task and needed to resolve a real gap;
heavy/full reruns require scoped authorization and shared-server preflight.
A failed audit does not authorize deleting evidence or automatically retraining.

Source refactors can invalidate historical/current replay guards even when
numerically equivalent. Preserve old snapshots and establish an explicit new
validated identity; never bypass the guard or edit source used by a running job.

Paper promotion requires a recipe frozen without further heldout feedback,
at least three independent repeats per backbone, mean/SD, per-center deltas
and registered run/checkpoint hashes. Dirty, unmatched or heldout-tuned runs
remain development evidence; repeats alone do not remove those limitations.

Report what was checked, integrity/replay/performance status separately, and
the smallest remaining verification needed. Do not stage or publish by default.
