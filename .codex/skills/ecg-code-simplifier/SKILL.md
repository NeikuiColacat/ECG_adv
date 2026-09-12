---
name: ecg-code-simplifier
description: Simplify recently changed ECG_manual_refactor code without changing behavior or weakening data, evaluation, reproducibility, and safety contracts. Use after implementation or when asked to simplify, deduplicate, reduce overengineering, shrink code, or audit code complexity in this repository.
---

# ECG Code Simplifier

Reduce production code and cognitive load while preserving observable behavior
and the locked research contract.

## Scope

Read `AGENTS.md` and the keep manifest first. Specialized data, evaluation,
reproducibility, GPU, and Git Skills take precedence in their domains.

Default to changed-code mode:

1. Inspect `git status`, the relevant diff, callers, and tests.
2. Edit only touched files plus the minimum adjacent code needed to simplify
   ownership.
3. Report larger opportunities instead of widening the patch silently.

Use repository-audit mode only when the user explicitly requests a broad audit.
That mode is read-only until the user chooses a slice to implement.

## Simplification Order

1. Reuse an existing retained helper or canonical owner.
2. Remove pass-through wrappers and one-use abstractions that add no contract.
3. Consolidate duplicate parsing, validation, or transformations under one
   owner.
4. Flatten control flow and replace custom machinery with a clear standard
   operation.
5. Remove truly dead branches only after the one-way-door check.

Keep a change only when total complexity decreases; moving the same complexity
to a new abstraction is not simplification.

## One-Way-Door Check

Before deleting or merging a symbol, path, config key, or branch, search for:

- direct calls and imports;
- string references, YAML closures, registries, and launch entrypoints;
- tests, mocks, serialization fields, reflection, and generated consumers;
- historical-only references that must remain provenance rather than runtime.

Classify candidates:

- `SAFE`: local, behavior-equivalent, covered by existing tests.
- `CAREFUL`: shared owner, config surface, serialization, or numerics; require a
  focused equivalence test.
- `RISKY`: data identity, metric identity, model semantics, launch/evidence
  identity, or external API; report first unless the user explicitly scopes it.

## Protected Contracts

Never simplify away:

- waveform units, shape, sampling rate, lead order, labels, K500 identity, or
  ref-exclusion checks;
- mapping/hash and metric-view checks;
- config closure, seed, checkpoint, artifact, and Git provenance;
- frozen/eval state restoration, finite-value checks, and gradient boundaries;
- shared-server resource controls, output isolation, or Git artifact guards.

Defensive code is removable only when another canonical owner enforces the same
contract and verification proves equivalence.

## Verify and Report

- Run focused tests for every changed behavior boundary.
- Run the retained pytest suite when core, data, evaluation, or launch code
  changes.
- Run `git diff --check` and inspect the final diff and status.
- Report scope, production-line delta, verification, and deferred risky items.

Do not stage, commit, push, delete experiments, or launch resource-heavy jobs
unless the user separately requests them.
