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
cd /home/linbinhao/ECG_manual_refactor_clean

/home/linbinhao/micromamba/envs/ECGTwin/bin/python \
  boot_scripts/run_experiment.py \
  --config configs/experiments/manual_refactor_pn2021_effnet_augmix_simclr_lhat_ningbo.yaml \
  --dry-run
```

Before an actual GPU run:

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=<confirmed_free_gpu> \
  /home/linbinhao/micromamba/envs/ECGTwin/bin/python \
  boot_scripts/run_experiment.py \
  --config <tracked-experiment.yaml>
```

Do not use `sudo`, change CUDA/drivers, overwrite an existing run, or write
large artifacts into the repository.

## Active Layout

| Path | Responsibility |
|---|---|
| `configs/` | Data, training, method, evaluation, seed, baseline, and experiment contracts |
| `data_preprocess/` | PTB-XL/PN2021 preprocessing, cache loading, splitting, and runtime datasets |
| `models/` | Model contracts, input adaptation, factories, checkpoints, EfficientNet, ECGFounder, and VAE interfaces |
| `core/` | Typed method graph, AugMix/VAE-LHAT execution, supervised training, and online adaptation |
| `boot_scripts/` | Thin managed CLI entrypoints; no experiment business logic |
| `util/` | Augmentations, metrics, evaluation, random identity, run records, TensorBoard, and visualization |
| `util/tests/` | CPU contract tests for the retained execution surface |
| `.codex/skills/` | Repo-tracked, public project procedures for agents; no private session memory |
| `docs/refactor_cleanup/manual_refactor_keep_manifest.md` | Authoritative keep/delete boundary |
| `agent_workspace/performance_summary_20260727/` | Two explicitly retained development evidence artifacts only |

Anything outside the keep manifest is legacy, temporary, or pending review. It
must not become a new runtime dependency.

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
/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest -q \
  util/tests/test_augmentations.py \
  util/tests/test_torch_augmentations.py \
  util/tests/test_data_contracts.py \
  util/tests/test_labels_super5.py \
  util/tests/test_pn2021_corruptions.py
```

Run outputs belong under `/home/linbinhao/ECG_adv_data/runs/`, not in Git.
Every managed run records its resolved config closure, command, Git state,
seeds, checkpoints, metrics, and file hashes through `util/run_record.py`.

## Evidence Boundary

The current simplified recipe was chosen after a 94-candidate development
search. Neither selected backbone candidate passed every original promotion
gate. The locked numbers may guide prospective replication, but final thesis
claims still require a frozen recipe, at least three independent repeats per
backbone, registered run records, and mean/standard-deviation reporting.

Historical `ecg_adv_gen` launchers and package modules are not active here.
Recover them from commit
`3a39a516420b52c219a782a7b7440f82746f4b90` only when historical provenance is
explicitly requested.
