---
name: ecg-adv-gen
description: Route implementation, protocol, evaluation, and evidence work for the ECG_manual_refactor PTB-XL Super5, PN2021/PN2021-C, EfficientNet1DV2, ECGFounder, AugMix, and VAE-LHAT pipeline. Use for work in this clean-room repository; treat legacy ECG_adv_Gen material as read-only provenance.
---

# ECG Manual Refactor Project Router

Use this skill as the project entrypoint, not as a second copy of every project
contract.

## Authoritative Sources

Read these in order:

1. `AGENTS.md` for safety and the current research contract.
2. `docs/refactor_cleanup/manual_refactor_keep_manifest.md` for the retained
   code and config boundary.
3. `configs/active_scripts.yaml` for active launch surfaces.
4. `configs/active_evidence_registry.yaml` for evidence status.
5. The resolved YAML and implementation for any value that may have changed.

Do not treat a Skill, report, historical command, or tracked config alone as
proof that an experiment ran or improved performance.

## Current Mainline

Route the selected direction using `docs/directions.md` and the non-executable
`configs/directions.yaml` catalog. ECG LLM owns PULSE/ECG-R1. The traditional
JSD direction uses the existing single-stage R18 recipe (JSD 1.5, LHAT 0.20),
not the two-stage JSD-12 ablation. The pipeline below is the SimCLR direction.
Keep shared code in this repository; never import from another worktree.

```text
PTB-XL source checkpoint
-> one target center's fixed K500 records
-> two-chain AugMix + SimCLR
-> clean plus rotating depth2/depth3 supervision
-> exact-label attack-then-contract VAE-LHAT BCE
-> K500-ref-excluded PN2021 Clean and PN2021-C evaluation
```

The current metric identity is Super5 `CD,HYP,MI,NORM,STTC`, mapping
`v7_super5_sjr_rgq_review_20260528` / `555ec85d5b51`, with
`drop_all_zero` as the primary view. Current evidence is development evidence
unless the active registry says otherwise.

## Hard Boundaries

- New runtime code may depend only on keep-manifest files or artifacts produced
  by them.
- Do not import legacy `ecg_adv_gen`, `methods`, `scripts`, or exploratory
  `agent_workspace` code.
- Adapt on one logical center's fixed K500 only and exclude those identities
  from target evaluation.
- Keep CPSC 2018 and Extra as one logical center.
- Keep datasets, caches, checkpoints, generated signals, and run outputs outside
  Git.
- Do not promote heldout-tuned, unmatched, dirty-Git, or single-seed results to
  paper evidence.

## Route Specialized Work

- Loading, resampling, leads, labels, K500, and exclusion:
  `data-prep-validator`.
- VAE-LHAT geometry, diagnostics, or ablations: `ecg-vae-online-at`.
- Metrics and matched model comparisons: `model-eval`.
- Config closure, run identity, replay, or paper promotion:
  `reproducibility-check`.
- GPU, long inference, cache builds, CPU/IO, or web serving:
  `shared-gpu-server-discipline`.
- Behavior-preserving code reduction: `ecg-code-simplifier`.
- Commit or push: `artifact-git-guard`.
- Durable-agent-process review: `ecg-agent-retrospective`.

Use only the smallest set needed for the task; the specialized Skill owns its
detailed checklist.

## Historical Recovery

Read `references/legacy_provenance.md` only when the user explicitly asks for
legacy ECG_adv_Gen code, commands, results, or prompt-token history. Historical
paths and parameter values are never current defaults.
