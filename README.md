# ECG research workspace

One repository, three research directions, shared data and implementation.
Start with [the direction guide](docs/directions.md); its small
[configuration catalog](configs/directions.yaml) points to existing experiments.
For the PULSE single-chain / three-chain comparison, start with the compact
[code map and evaluation protocol](docs/pulse_evaluation_architecture.md).

| Direction | Main method | Long-lived branch |
|---|---|---|
| ECG LLM | PULSE / ECG-R1 image evaluation and PULSE fine-tuning | `direction/ecg-llm` |
| Traditional + SimCLR | ECGFounder / EfficientNet, locked two-stage AugMix + SimCLR + LHAT | `direction/traditional-simclr` |
| Traditional + JSD | Both backbones, single-stage Joint, JSD 1.5 and LHAT supervised mass 0.20 | `direction/traditional-jsd` |

The JSD direction uses the existing R18 recipe. The September 11 two-stage
Founder JSD/width experiment remains an auxiliary comparison, not its default.
These are development recipes, not paper-final claims. Different stage/step
budgets do not establish a matched causal estimate of SimCLR's benefit.
Each direction has a long-lived branch; temporary review worktrees may also
exist. Use the live Git worktree list for current checkouts. Historical
consolidation and recovery decisions are recorded separately.

## PULSE review path

The current PULSE comparison is a fixed recipe: eight completed adapters,
four K500-excluded centers, and original/single/three predictions on six views.
Read [the architecture map](docs/pulse_evaluation_architecture.md) first, then
follow its explicit training, inference, validation and scheduling owners.
The [full evaluation guide](docs/pulse_full_lora32_live_eval_20260928.md) lists
the initial and reboot-recovery entrypoints. Inspect the selected run's state
files for progress; a dated index entry is a snapshot.

## Run one experiment

Read [AGENTS.md](AGENTS.md), choose an existing experiment from the direction
guide, then dry-run from the current checkout:

```bash
/home/linbinhao/miniforge3/envs/ECGTwin/bin/python \
  boot_scripts/run_experiment.py \
  --config configs/experiments/manual_refactor_pn2021_effnet_augmix_simclr_lhat_ningbo.yaml \
  --dry-run
```

A dry-run loads no data, model, or GPU. Existing output directories are reported
as collisions, not overwritten. For a new execution, choose a fresh external
`--run-dir`, check `nvidia-smi`, and explicitly set `CUDA_VISIBLE_DEVICES`.
LLM execution uses the model-specific environment declared by its workflow;
the CPU project environment is not a replacement for the author dependencies.

## Where to work

| Location | Owner |
|---|---|
| `data_preprocess/` | Shared preprocessing, cache/split/ledger, data loading |
| `util/augmentations/` | Shared ECG corruption kernels and profile |
| `core/`, `models/` | Traditional training/methods/models; PULSE-specific files own PULSE training |
| `util/evaluation/` | Explicit traditional and LLM evaluators/backends |
| `util/config_bundle.py`, `util/random_seed.py`, `util/run_record.py` | Shared configuration, randomness, provenance |
| `boot_scripts/` | One managed launcher and thin task entrypoints |
| `configs/directions.yaml` | Small human navigation catalog; not executable configuration |
| `configs/active_scripts.yaml` | Full historical and current executable index |
| `configs/active_evidence_registry.yaml` | Evidence identities and limitations |
| `util/tests/` | Contract and focused numerical regression tests |

Shared fixes are reviewed as small commits and integrated through the common
baseline. Do not import source from another worktree. One owner controls each
worktree's branch/index; collaborators use their own clone/branch, or agree on
non-overlapping file ownership. Do not change source used by a running job.

## Scientific boundaries

- Super5 order: `CD, HYP, MI, NORM, STTC`.
- PN2021 mapping: `v7_super5_sjr_rgq_review_20260528` / `555ec85d5b51`.
- Centers: Ningbo, Chapman-Shaoxing, CPSC 2018 including Extra, Georgia.
- Adaptation uses one center's fixed K500; target evaluation excludes its IDs.
- Traditional primary metrics: macro AUROC and sklearn average precision,
  `drop_all_zero`, equal corruption views then equal centers.
- Traditional input: raw mV `(B,1000,12)`, 100 Hz. Corrupt before global
  per-sample z-score; Founder interpolates `1000 -> 5000` on device first.
- LLM input: native-500-Hz ECG rendered to images, then the model's own processor.
  Its generated-label metrics are not interchangeable with raw-logit AUROC/AP.
- Seeds, loss weights, label mapping and old artifact identities are not changed
  by code organization. Data/checkpoints/results stay outside Git.

The [config guide](configs/README.md) describes YAML closure and evaluation
subjects. The tracked data ledger is the source of truth for current members
and bytes; quick runtime checks do not constitute full content verification.

## Verify and preserve

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/linbinhao/miniforge3/envs/ECGTwin/bin/python -m pytest -q
git diff --check
```

CPU tests do not replace opt-in GPU or real-model parity checks. Preserve
unrelated dirty files, source checkpoints and final results. No `sudo`,
system/CUDA/driver changes, unscoped process kills, or broad Git staging.
When running the complete suite on this host, give pytest a fresh home-owned
directory with `--basetemp`; path-contract fixtures intentionally reject `/tmp`.

The [keep manifest](docs/refactor_cleanup/manual_refactor_keep_manifest.md)
owns retained-file and deletion boundaries. The
[consolidation record](docs/refactor_cleanup/three_direction_consolidation.md)
tracks recovery, verification and retirement approval. Old worktrees and
unique experiments are not disposable just because their Git status is clean.
