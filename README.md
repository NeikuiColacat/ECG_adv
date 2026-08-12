# ECG Manual Refactor

This repository is the clean-room execution surface for the PTB-XL Super5 to
PN2021 cross-center ECG adaptation study. The old project remains provenance in
Git history; active code must stay inside the keep manifest.

Current prospective method:

```text
PTB-XL source checkpoint
-> one target center's fixed K500 records
-> two-chain AugMix + SimCLR representation adaptation
-> 50% clean + 50% rotating depth2/depth3 supervised adaptation
-> one exact-label attack-then-contract VAE-LHAT view
-> ref-excluded PN2021 Clean and PN2021-C evaluation
```

The current scores are heldout-tuned, single-seed development evidence. They
are not paper-final results. See
[`configs/active_evidence_registry.yaml`](configs/active_evidence_registry.yaml)
for the exact boundary and
[`configs/active_scripts.yaml`](configs/active_scripts.yaml) for the executable
surface.

## Start Here

1. Read [`AGENTS.md`](AGENTS.md) for shared-server safety.
2. Read the
   [`manual refactor keep manifest`](docs/refactor_cleanup/manual_refactor_keep_manifest.md).
3. For Codex-assisted work, use the repo-tracked
   [project skills](.codex/skills/README.md).
4. Select a tracked YAML under `configs/experiments/`.
5. Dry-run it through the single launcher before using data, models, or GPUs.

```bash
cd /home/linbinhao/ECG_manual_refactor_paper_kernel_v2

/home/linbinhao/miniforge3/envs/ECGTwin/bin/python \
  boot_scripts/run_experiment.py \
  --config configs/experiments/manual_refactor_pn2021_effnet_augmix_simclr_lhat_ningbo.yaml \
  --dry-run
```

Before an actual GPU run:

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=<confirmed_free_gpu> \
  /home/linbinhao/miniforge3/envs/ECGTwin/bin/python \
  boot_scripts/run_experiment.py \
  --config <tracked-experiment.yaml>
```

Do not use `sudo`, change CUDA/drivers, overwrite an existing run, or write
large artifacts into the repository.

If the tracked output already exists, `--dry-run` remains read-only and reports
`run_dir_collision: true` plus `would_fail_execution: true`. A real launch still
fails closed; pass `--run-dir <new-external-directory>` to start a new run.

## Active Layout

| Path | Responsibility |
|---|---|
| `configs/` | Data, training, finite-recipe, evaluation, seed, baseline, and experiment contracts |
| `data_preprocess/` | PTB-XL/PN2021 preprocessing, cache loading, splitting, runtime datasets, and content-ledger verification |
| `models/` | Model contracts, input adaptation, factories, checkpoints, EfficientNet, ECGFounder, and VAE interfaces |
| `core/` | Five finite RecipeSpec variants, AugMix/VAE-LHAT execution, supervised training, and online adaptation |
| `boot_scripts/` | Thin managed CLI entrypoints; no experiment business logic |
| `util/` | Augmentations, metrics, evaluation, random identity, and run records |
| `util/tests/` | CPU contract tests for the retained execution surface |
| `.codex/skills/` | Repo-tracked, public project procedures for agents; no private session memory |
| `docs/refactor_cleanup/manual_refactor_keep_manifest.md` | Authoritative keep/delete boundary |
| `agent_workspace/performance_summary_20260727/` | Two explicitly retained development evidence artifacts only |

Anything outside the keep manifest is legacy, temporary, or pending review. It
must not become a new runtime dependency.

Managed data-consuming runs are bound to a tracked 121-member derived-cache
and split ledger. Execution performs an inventory-and-size gate before creating
the run directory, snapshots the YAML closure and ledger, and launches from
that immutable bundle. Two offline full-content passes established the seal;
it does not cover raw WFDB inputs or prove preprocessing correctness, and the
runtime quick gate does not rehash all 405 GB.

## Locked Contracts

- Class order: `CD, HYP, MI, NORM, STTC`.
- PN2021 mapping: `v7_super5_sjr_rgq_review_20260528`,
  hash `555ec85d5b51`.
- Logical target centers: Ningbo, Chapman-Shaoxing, CPSC 2018 plus Extra, and
  Georgia.
- Adaptation data: fixed K500 from one logical center only.
- Evaluation: exclude that center's K500 identities.
- Primary metrics: macro AUROC and sklearn average precision after
  `drop_all_zero`.
- Canonical waveform: raw physical mV, 100 Hz, 1000 points, 12 leads,
  time-channel layout.
- EfficientNet input: 100 Hz, then per-sample global z-score.
- ECGFounder input: linear 100 Hz to 500 Hz adaptation, then per-sample global
  z-score.
- PN2021-C: locked five-operator profile, 500 Hz operator domain, all ten
  depth-2 and ten depth-3 compositions.

## Necessary CPU Verification

```bash
/home/linbinhao/miniforge3/envs/ECGTwin/bin/python -m pytest -q
```

`pytest.ini` restricts discovery to the retained `util/tests/test_*.py` public
contract suite; the keep-manifest inventory test rejects stale or ghost entries.

Run outputs belong under `/home/linbinhao/ECG_adv_data/runs/`, not in Git.
Every managed run records its resolved config closure, command, Git state,
seeds, RecipeSpec identity, checkpoints, metrics, and file hashes through
`util/run_record.py`. The PN2021 CLI keeps the explicit `--method-config`
selector; that file now chooses one of five code-owned finite recipes rather
than describing a dynamic graph.

Prospective PN2021 adaptation emits a schema-v1 `pn2021_train_result`; its
selected final checkpoint is schema v3, and both carry the same immutable
training lineage. Formal evaluation emits a schema-v3
`pn2021_evaluation_result` whose `subject` selects exactly one of three modes:

- `prospective_train_result`: one target center, with `--train-result` and the
  matching tracked `--method-config`;
- `legacy_center_adapted`: one target center, with an explicit schema-v2 A0 or
  Direct checkpoint;
- `source_registry`: the canonical four centers, with a locked PTB-XL source
  registry.

`--train-result` is an output artifact reference, not a YAML configuration
dependency, so it is intentionally excluded from the copied config closure.
The evaluation subject still locks that artifact by resolved path and SHA256;
the method selector and source registry remain ordinary hashed config-closure
members.

After all four prospective center evaluations exist, the tracked EfficientNet
or ECGFounder matrix experiment aggregates their diagonal only. This is a
CPU/JSON step: it reads no ECG data and never deserializes a checkpoint or
constructs a model/GPU runtime. It does stream checkpoint bytes to verify their
SHA256, rejects lineage/cohort/center drift, and recomputes equal-view then
equal-center metrics instead of trusting member aggregate fields. Its schema-v1
`pn2021_diagonal_four_center_evaluation` remains prospective evidence until the
underlying registered repetitions satisfy the paper-promotion gate.

## Evidence Boundary

The current simplified recipe was chosen after a 94-candidate development
search. Neither selected backbone candidate passed every original promotion
gate. The locked numbers may guide prospective replication, but final thesis
claims still require a frozen recipe, at least three independent repeats per
backbone, registered run records, and mean/standard-deviation reporting.

The 2026-08-06 `r2` run remains legacy typed-method-graph metric evidence. It
is not a post-migration RecipeSpec replay: its run-scoped snapshots did not
include the Stage-1 AugMix config/resource identity. The 16 current mainline
train/eval configs therefore target a fresh
`manual_refactor_paper_kernel_v2_recipe_v1_r1` output root, which has not yet
been used for a full replication. A separate one-epoch Ningbo diagnostic smoke
at commit `dfd00ec` completed Direct EfficientNet and both mainline backbones;
it verifies the managed CUDA seams only and is not performance or paper
evidence. Its locked artifact identity and limitations are recorded in
`configs/active_scripts.yaml`.

Historical `ecg_adv_gen` launchers and package modules are not active here.
Recover them from commit
`3a39a516420b52c219a782a7b7440f82746f4b90` only when historical provenance is
explicitly requested.
