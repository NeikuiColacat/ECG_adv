# ECG_adv_Gen

ECG_adv_Gen is the working repo for the PTB-XL Super5 to PN2021 cross-center
ECG adaptation project. The current mainline is:

```text
PTB-XL Super5 EfficientNet1DV2 baseline
-> fixed K=500 target-center PN2021 adaptation
-> ECGTwin VAE latent-hull online adversarial training
-> PN2021 held-out, ref-excluded AUROC/AUPRC evaluation
```

For the latest trusted experiment facts, start from
[`configs/active_evidence_registry.yaml`](configs/active_evidence_registry.yaml).
For agent safety and shared-server rules, start from [`AGENTS.md`](AGENTS.md).

## Quick Start

```bash
cd /home/linbinhao/ECG_adv_Gen
micromamba run -n cli-tools git status --short --branch
micromamba run -n ECGTwin python scripts/agent/audit_agent_workspace.py
```

Before any GPU run on the shared server:

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=<free_gpu_ids> micromamba run -n ECGTwin python ...
```

Do not use `sudo`, do not touch CUDA/NVIDIA drivers/kernel/system env, and keep
project work under `/home/linbinhao`.

## Navigation For Agents

| Path | Purpose |
|---|---|
| `AGENTS.md` | Durable agent memory, shared-server rules, current mainline facts |
| `configs/active_evidence_registry.yaml` | Current trusted claim, run lineage, metrics, artifact policy |
| `configs/active_scripts.yaml` | Which YAML configs wrap which legacy scripts and which paths must not move |
| `configs/defaults/` | Shared YAML defaults for mapping, model, VAE-LHAT, ECGFounder, EfficientNet |
| `configs/experiments/` | Reproducible managed experiment configs |
| `configs/local/*.example.yaml` | Host-local path examples; real local YAML is ignored |
| `configs/label_mappings/` | Structured label-mapping evidence, including PN2021 Super5 review JSONL |
| `ecg_adv_gen/` | Stable Python package code for config, data, labels, adaptation, evaluation, evidence, reporting, training |
| `scripts/run_experiment.py` | Managed YAML launcher |
| `scripts/agent/` | CPU-only audit, run finalization/registration, manifest backfill, comparison bundle, retrospective inventory tools |
| `scripts/paper/` | Dated experiment wrappers kept for evidence and reproducibility |
| `scripts/triple_labels/` | Legacy Super5 train/eval entrypoints and current label-scheme source |
| `scripts/pgd_cross_center/` | Legacy online AT / latent-hull orchestrators |
| `docs/pipelines/` | Long-term pipeline, refactor, reproduction, and evidence docs |
| `docs/labeling/` | Human-readable label mapping review and clinician audit material |
| `docs/reports/archive/` | Archived HTML/MD reports that are useful context but not canonical facts |
| `docs/tmp_md/`, `docs/tmp_html/` | Legacy temporary reports; do not treat as canonical unless referenced by pipelines |
| `model/` | External model repo handles and submodules; do not stage host-specific symlink changes |
| `util/tests/` | Project-wide tests. Future target layout is top-level `tests/` |
| `trash/` | Ignored cold archive; do not import active code from here without migrating and testing it |

## Current Evidence

The active EfficientNet1DV2 v7 claim is registered in
[`configs/active_evidence_registry.yaml`](configs/active_evidence_registry.yaml):

| Comparison | AUROC | AUPRC |
|---|---:|---:|
| Direct K500, all-zero kept | 0.8522 | 0.5247 |
| VAE-LHAT K500, all-zero kept | 0.8712 | 0.5544 |
| Delta | +1.90 pp | +2.97 pp |
| Direct K500, drop-all-zero | 0.8763 | 0.6666 |
| VAE-LHAT K500, drop-all-zero | 0.8996 | 0.7176 |
| Delta | +2.33 pp | +5.10 pp |

Protocol facts:

- mapping: `v7_super5_sjr_rgq_review_20260528`, hash `555ec85d5b51`;
- class order: `CD, HYP, MI, NORM, STTC`;
- target centers: `ningbo`, `chapman_shaoxing`, `cpsc_2018`, `georgia`;
- K-shot protocol: fixed `K=500`, seed `20260531`, ref ids excluded from eval;
- selection policy: K500-internal validation plus source-performance floor.

## Managed Commands

CPU-only workspace audit:

```bash
micromamba run -n ECGTwin python scripts/agent/audit_agent_workspace.py
```

Managed dry-run example:

```bash
micromamba run -n ECGTwin python scripts/run_experiment.py \
  --config configs/experiments/effnet_direct_k500_v7_sjr_rgq.yaml \
  --local-config configs/local/linbinhao_server.example.yaml \
  --run-id dryrun_effnet_direct_k500_v7_sjr_rgq \
  --dry-run
```

Build the registered direct-vs-VAE comparison bundle:

```bash
micromamba run -n ECGTwin python scripts/agent/build_comparison_bundle.py --force
```

Backfill the legacy VAE-LHAT run manifest:

```bash
micromamba run -n ECGTwin python scripts/agent/backfill_vae_lhat_manifest.py
```

Finalize and register a completed run for future agent handoff:

```bash
micromamba run -n ECGTwin python scripts/agent/finalize_run.py \
  --run-dir /path/to/run_dir \
  --purpose "Why this experiment was run" \
  --result-summary "What happened and how to interpret it" \
  --outcome provisional

micromamba run -n ECGTwin python scripts/agent/register_run.py \
  --run-dir /path/to/run_dir \
  --status provisional
```

Managed `scripts/run_experiment.py --write-plan` and `--execute` now create a
standard per-run record automatically: `run_card.json`, `run_file_index.json`,
`summary.md`, and category directories such as `configs/`, `manifests/`,
`logs/`, `checkpoints/`, `eval/`, `diagnostics/`, and `reports/`.

## What Not To Move

The following are still active or evidence-sensitive paths. Extract reusable
logic into `ecg_adv_gen/`, but keep these entrypoint paths stable unless the
YAML index, docs, and tests are updated in the same change:

- `scripts/paper/`
- `scripts/triple_labels/train_ptbxl.py`
- `scripts/triple_labels/eval_crosscenter.py`
- `scripts/pgd_cross_center/synth_online_at_super5.py`
- `scripts/ecgtwin_author_repro/`
- `model/DeepECG`
- `model/ECGTwin`
- `model/advdiff`
- `model/ecg_ptbxl_benchmarking`
- `model/ecgfounder`

## Documentation

- Pipeline index: [`docs/pipelines/README.md`](docs/pipelines/README.md)
- Agent operating layer: [`docs/pipelines/agent_operating_layer_20260529.md`](docs/pipelines/agent_operating_layer_20260529.md)
- Run record management: [`docs/pipelines/run_record_management_20260529.md`](docs/pipelines/run_record_management_20260529.md)
- v7 EfficientNet mainline repro: [`docs/pipelines/v7_sjr_rgq_effnet_mainline_repro_20260528.md`](docs/pipelines/v7_sjr_rgq_effnet_mainline_repro_20260528.md)
- Super5 label mapping pipeline: [`docs/pipelines/super5_label_mapping_pipeline.md`](docs/pipelines/super5_label_mapping_pipeline.md)
- Config guide: [`configs/README.md`](configs/README.md)
