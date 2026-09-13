---
name: ecg-code-simplifier
description: Audit or simplify ECG_manual_refactor code for human review while preserving numerical behavior, data identity, reproducibility and shared-server safety.
---

# ECG Code Simplifier

Reduce cognitive load and production complexity, not merely physical line count.
Follow `AGENTS.md` and the relevant keep-manifest entries.

## Scope

- Default to the requested changed-code slice: inspect diff, callers and tests;
  use adjacent files only to establish a clearer owner.
- A repository-wide audit is read-only unless implementation is also requested.
  Report larger opportunities instead of expanding the patch silently.
- Check live-job and historical replay source locks before editing. Even an
  unused-import removal can break a source hash; never bypass the guard.

## Simplify in This Order

1. Reuse a retained owner; remove contract-free forwarding and one-use machinery.
2. Consolidate genuinely equivalent parsing/validation/transformations.
3. Prefer direct functions and explicit loops over deeply nested expressions,
   generic engines or compressed one-line code.
4. Remove dead branches only after checking all consumers below.

Moving the same complexity to a new module is not a simplification. Small
duplication may be cheaper than introducing dependency coupling.

## Before Merging or Removing

Search direct calls/imports, strings/YAML/registries/entrypoints, tests/mocks,
serialization/reflection and generated consumers. Distinguish historical-only
provenance from active replay code. No direct caller does not prove dead code.

- `SAFE`: local, behavior-equivalent and covered.
- `CAREFUL`: shared owner, config/schema or numerics; needs equivalence tests.
- `RISKY`: data/metric/model or launch/evidence identity; report first unless
  explicitly scoped. File deletion still follows the manifest approval gate.

## Preserve Non-Obvious Behavior

- Units, shape, rate, leads, labels, K500/ref-exclusion and mapping/metric identity.
- Config/seed/checkpoint/artifact provenance, finite values and gradient boundaries.
- Freeze/eval state, BatchNorm and RNG. Unused module initialization can consume
  RNG; an apparently redundant forward can change BN or the exposure budget.
- JSON writers are not interchangeable without checking serialization, failure
  cleanup, atomic replacement and fsync guarantees.
- Resource controls, output isolation and Git artifact guards.

Remove duplicate defensive checks only when one canonical owner enforces the
same contract and tests prove equivalence.

## Verify and Report

Run focused tests; for training-loop/numerical changes compare tiny old/new
parameters, optimizer/scheduler state, BN, RNG, view order and artifact identity.
Run the retained CPU suite for core/data/evaluation/launcher changes, and
`git diff --check`. Docs-only edits need scope/link/diff checks, not training.
Real-model/GPU parity requires a scoped, resource-checked run when applicable.

Report the reviewed slice, net complexity/line changes, verification and deferred
risks. Do not stage, commit, delete experiments or launch heavy jobs without
the corresponding authorization.
