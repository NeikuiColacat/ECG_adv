# Agent Operating Layer, 2026-05-29

This refactor makes the workspace easier for a future AI coding agent to
resume without replaying old chats.

## External Practice Check

The design follows a lightweight subset of established practice:

- AGENTS.md is a predictable agent entry point for setup, tests, and safety
  rules. Sources: https://agents.md/ and
  https://developers.openai.com/codex/guides/agents-md.
- Experiment tracking should record params, code versions, metrics, and
  artifacts per run. Source: https://mlflow.org/docs/latest/ml/tracking.
- Run directories should keep resolved config and command overrides. Source:
  https://hydra.cc/docs/1.3/tutorials/basic/running_your_app/working_directory/.
- Resume-capable checkpoints need more than weights: model state, optimizer
  state, epoch, and training state must be saved. Sources:
  https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html and
  https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html.

## Implemented Layer

Tracked entry point:

```text
configs/active_evidence_registry.yaml
```

It records the current trusted mainline:

- Direct baseline: `effnet_direct_k500_v7_sjr_rgq`
- VAE method: `effnet_vae_lhat_k500_v7_sjr_rgq`
- Mapping: `v7_super5_sjr_rgq_review_20260528`
- Mapping hash: `555ec85d5b51`
- Fixed K500 seed: `20260531`
- Target centers: `ningbo`, `chapman_shaoxing`, `cpsc_2018`, `georgia`
- Evaluation views: all-zero-kept and drop-all-zero, both ref-excluded
- Trusted/provisional/deprecated run status and no-commit artifact policy

CPU-only audit:

```bash
micromamba run -n ECGTwin python scripts/agent/audit_agent_workspace.py
```

The audit checks:

- active configs resolve and match registry protocol;
- Super5 mapping version/hash/class order are consistent;
- K500 ref-meta files exist and contain 500 ref IDs per target center;
- Direct and VAE configs use the same backbone, seed, mapping, and centers;
- metrics_long and paper tables can be traced to source eval JSONs;
- the managed comparison bundle exists when the registry marks it built;
- blocked checkpoint/dataset/run artifacts are not tracked or staged;
- dirty external model handles are warnings, not failures, unless staged.

Managed comparison bundle:

```bash
micromamba run -n ECGTwin python scripts/agent/build_comparison_bundle.py --force
```

Current output:

```text
${paths.output_root}/comparison_bundles/v7_sjr_rgq_effnet_direct_vs_vae_lhat_20260528/
```

Artifacts:

- `comparison_manifest.json`
- `metrics_long.csv`
- `comparison_delta.csv`
- `paper_table_pn2021_all_zero_kept_refexcluded.csv`
- `paper_table_pn2021_drop_all_zero_refexcluded.csv`

The bundle records source metrics hashes, baseline/candidate run IDs, mapping
metadata, per-center deltas, and center-mean deltas. The current center-mean
VAE gains are:

| View | AUROC gain | AUPRC gain |
|---|---:|---:|
| all-zero kept | +1.90 pp | +2.97 pp |
| drop-all-zero | +2.33 pp | +5.10 pp |

## VAE-AT Checkpoint Contract

`scripts/pgd_cross_center/synth_online_at_super5.py` now writes epoch-boundary
state for future interrupted runs:

```text
checkpoints/checkpoint_latest.pt
checkpoints/checkpoint_best.pt
checkpoints/checkpoint_index.jsonl
diagnostics_epoch.jsonl
agent_decision.json
```

`checkpoint_latest.pt` contains:

- model, optimizer, scheduler state;
- epoch, best metric, best epoch, early-stop counters;
- quality buffer state;
- stratified walker state;
- Python, NumPy, torch CPU, and torch CUDA RNG state;
- EWA anchor tensors;
- full training log and argument snapshot.

Resume is epoch-boundary only:

```bash
... scripts/pgd_cross_center/synth_online_at_super5.py ... --resume latest
```

or through the v7 wrapper:

```bash
... scripts/paper/run_effnet_latent_augmix_stage3_20260524.py ... --resume latest
```

`diagnostics_epoch.jsonl` records ASR, `atk_anchor`, clean/adversarial BCE,
loss gain, decoded invalid rate, buffer size, quick eval, and the agent
decision. `atk_init` is currently logged as unavailable for this latent-hull
generator because the script does not retain the random-init decoded ECG; this
is explicit rather than silently absent.

## Legacy Traceability Closure

The v7 Direct run has a managed launcher manifest:

```text
${paths.output_root}/20260528/v7_sjr_rgq_main_20260528_direct_plan/run_manifest.json
```

The v7 VAE output predates the managed launcher manifest path, so it is tracked
with a legacy backfilled manifest:

```text
${paths.output_root}/effnet_vae_lhat_k500_v7_sjr_rgq/v7_sjr_rgq_main_20260528/run_manifest.backfilled.json
```

Build or refresh it with:

```bash
micromamba run -n ECGTwin python scripts/agent/backfill_vae_lhat_manifest.py
```

This manifest records the per-center legacy launch configs, train/eval
commands, selected checkpoints, and artifact hashes. Future reruns should still
launch VAE through `scripts/run_experiment.py --execute` so the original managed
manifest exists at run time rather than being reconstructed later.

## Next Step

Before the next GPU experiment:

1. Run `scripts/agent/audit_agent_workspace.py`.
2. Check GPU state with `nvidia-smi`.
3. Launch from a YAML config or write an equivalent manifest.
4. After training, rebuild the comparison bundle and rerun the audit.
