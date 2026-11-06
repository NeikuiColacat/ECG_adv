---
name: ecg-code-simplifier
description: Simplify the selected traditional ECG checkout while preserving scientific and runtime contracts.
---

# ECG Code Simplifier

Read AGENTS.md and docs/METHOD.md. Inspect the requested slice, callers, YAML,
entrypoints and tests before changing ownership or deleting files.
Check live-job source locks; even an import removal can invalidate a run hash.

Prefer existing owners, direct functions and explicit loops. Remove dead wrappers,
duplicate work and unused branches after checking string-based consumers and
serialized artifacts. Moving complexity to a new module is not simplification.
Small duplication may cost less than a new shared abstraction.

Preserve units, shapes, leads, labels, K500 exclusion, mapping, metric identity,
config/seed/checkpoint provenance, finite values and gradient boundaries.
Preserve freeze/eval state, BatchNorm, RNG and optimizer/exposure budgets.
Initializer removal can consume different RNG; extra forwards can change BN.
Do not substitute JSON writers without checking atomicity and failure behavior.
Keep one owner for every required validation and maintain focused tests.

User-authorized simplification may remove redundant tracked files after recording
the exact plan. Never infer authority to delete external results or worktrees.
Keep historical recovery in docs/EVIDENCE.md; historical code is not a dependency.
Use apply_patch and preserve unrelated dirty work.

For source/config edits run relevant focused checks and the retained CPU suite.
For numerical changes compare tiny old/new parameters, optimizer/scheduler state,
BN, RNG and view ordering; test real models/GPU only within authorized scope.
Docs-only edits need links, scope and git diff --check.
Report the removed complexity, validation and remaining limits.
Staging, committing, pushing and heavy jobs require their own applicable scope.
