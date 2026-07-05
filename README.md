# ECG_adv_Gen

ECG_adv_Gen is the working repo for the PTB-XL Super5 to PN2021 cross-center
ECG adaptation project. The current mainline is:

```text
PN2021/PN2021-C VAE-LHAT + three-chain AugMix
chain1/chain2: official corruption chains
chain3: ECGTwin VAE-LHAT adversarial waveform
-> clean PN2021 and PN2021-C ref-excluded AUROC/AUPRC evaluation
```

For the latest reproducible experiment path, start from
`latest_mainline` in
[`configs/active_scripts.yaml`](configs/active_scripts.yaml). It declares the
`vae_lhat_threechain_augmix_pn2021c` method as 10 stages, all launched through
[`scripts/run_experiment.py`](scripts/run_experiment.py).
For the latest trusted experiment facts, use
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
| `configs/active_scripts.yaml` | Which YAML configs map to managed package runners and which paths must not move |
| `configs/defaults/` | Shared YAML defaults for mapping, model, VAE-LHAT, ECGFounder, EfficientNet |
| `configs/experiments/` | Reproducible managed experiment configs |
| `configs/local/*.example.yaml` | Host-local path examples; real local YAML is ignored |
| `configs/label_mappings/` | Structured label-mapping evidence, including PN2021 Super5 review JSONL |
| `ecg_adv_gen/` | Stable Python package code for config, data, labels, adaptation, evaluation, evidence, reporting, training |
| `ecg_adv_gen/runner/` | Package-owned experiment runner modules used by managed YAML configs |
| `scripts/run_experiment.py` | Managed YAML launcher |
| `scripts/agent/` | CPU-only audit, run finalization/registration, manifest backfill, comparison bundle, retrospective inventory tools |
| `docs/pipelines/` | Long-term pipeline, refactor, reproduction, and evidence docs |
| `docs/labeling/` | Human-readable label mapping review and clinician audit material |
| `docs/refactor_cleanup/` | Cleanup manifests and public-tree reduction reports |
| `model/` | External model repo handles and submodules; do not stage host-specific symlink changes |
| `util/tests/` | Project-wide tests. Future target layout is top-level `tests/` |

## Current Evidence

The active EfficientNet1DV2 v7 claim is registered in
[`configs/active_evidence_registry.yaml`](configs/active_evidence_registry.yaml):

| View | Direct K500 | VAE-LHAT three-chain | Delta |
|---|---:|---:|---:|
| `pn2021_all_zero_kept_refexcluded` AUROC / AUPRC | 0.8492 / 0.5271 | 0.8735 / 0.5637 | +2.43 pp / +3.66 pp |
| `pn2021_drop_all_zero_refexcluded` AUROC / AUPRC | 0.8732 / 0.6703 | 0.9015 / 0.7307 | +2.83 pp / +6.03 pp |
| `pn2021c_all_zero_kept_corrupted_refexcluded` AUROC / AUPRC | 0.8100 / 0.4729 | 0.8286 / 0.4940 | +1.87 pp / +2.12 pp |
| `pn2021c_drop_all_zero_corrupted_refexcluded` AUROC / AUPRC | 0.8312 / 0.6087 | 0.8547 / 0.6439 | +2.35 pp / +3.52 pp |

Protocol facts:

- mapping: `v7_super5_sjr_rgq_review_20260528`, hash `555ec85d5b51`;
- class order: `CD, HYP, MI, NORM, STTC`;
- target centers: `ningbo`, `chapman_shaoxing`, `cpsc_2018`, `georgia`;
- K-shot protocol: fixed `K=500`, seed `20260601`, ref ids excluded from eval;
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

Dry-run every `latest_mainline` stage before launching GPU work:

```bash
micromamba run -n ECGTwin python -m pytest \
  util/tests/test_config_loader.py::test_latest_mainline_configs_dry_run_through_run_experiment_cli -q
```

Build the registered direct-vs-VAE comparison bundle:

```bash
micromamba run -n ECGTwin python scripts/agent/build_comparison_bundle.py --force
```

The legacy VAE-LHAT manifest backfiller is provenance-only for old v7 evidence;
it is not part of the `latest_mainline` replay path.

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

The following external model handles are evidence-sensitive host-local paths.
Keep package logic in `ecg_adv_gen/` and experiment launch through
`scripts/run_experiment.py` plus tracked YAML. Legacy script archives were
removed from the public tree; do not restore them as callable entrypoints.

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
