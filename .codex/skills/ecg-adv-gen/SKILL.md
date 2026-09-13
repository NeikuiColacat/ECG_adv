---
name: ecg-adv-gen
description: Route ECG_manual_refactor work across PULSE/ECG-R1 image models, traditional SimCLR, and single-stage JSD; locate the relevant project contracts, configs, and evidence.
---

# ECG Project Router

Use the selected checkout's `AGENTS.md` and project-local skills. Same-named
global skills are separate definitions, not extra instructions to merge.

## Find Current Truth

1. Read `AGENTS.md`; confirm checkout, branch, dirty state and task scope.
2. Use `docs/directions.md` / `configs/directions.yaml` to select the direction.
3. Locate the relevant entries in the keep manifest, `configs/active_scripts.yaml`
   and `configs/active_evidence_registry.yaml` with `rg`. Read the applicable
   rules and entries; unrelated historical batches are not a startup checklist.
4. Inspect the selected YAML closure and implementation for mutable values;
   inspect actual run artifacts before claiming execution or performance.

Repository paths are relative to this checkout. Skill reference paths are
relative to this skill directory. Do not import code from another worktree.

## Keep Directions Distinct

- ECG LLM: PULSE/ECG-R1 use native500 images and model-specific processors.
  Current R1 evaluation is image-only; generated labels are not class scores.
- Traditional SimCLR: locked two-stage AugMix/SimCLR then supervised adaptation
  with VAE-LHAT; see `configs/train/methods/augmix_simclr_lhat.yaml`.
- Traditional JSD: single-stage R18, JSD1.5 and LHAT clean-loss replacement0.20;
  follow its selector in the direction catalog, not the two-stage JSD-12 ablation.

The AGENTS data, K500/ref-exclusion, evidence and safety rules apply throughout.
A config, skill or report alone proves neither performance nor paper readiness.

## Route Only Relevant Work

| Task | Skill |
|---|---|
| Data, resampling, labels, K500, image-input identity | `data-prep-validator` |
| VAE-LHAT geometry, contraction, diagnostics | `ecg-vae-online-at` |
| Metrics, uncertainty, matched comparisons | `model-eval` |
| Config/run/checkpoint identity and replay | `reproducibility-check` |
| GPU, heavy CPU/IO, cache builds, web serving | `shared-gpu-server-discipline` |
| Behavior-preserving code reduction or code audit | `ecg-code-simplifier` |
| Staging, committing, pushing | `artifact-git-guard` |
| Project instructions, skills, handoffs | `ecg-agent-retrospective` |

Use the smallest applicable set. An explanation/status request is read-only;
do not turn skill routing into authorization to launch, delete, commit or publish.

## Historical Recovery

Read [legacy provenance](references/legacy_provenance.md) only for explicitly
requested legacy code/history. Its commits, old paths and parameters are
provenance, never current defaults.
