# Final Cleanup Summary 2026-07-05

## Scope

This cleanup converted the repository from a broad research-history workspace into a public latest-mainline reproduction tree.

Public mainline:

- Source of truth: `configs/active_scripts.yaml:latest_mainline`
- Method: PN2021/PN2021-C VAE-LHAT + three-chain AugMix
- Mapping: `v7_super5_sjr_rgq_review_20260528 / 555ec85d5b51`
- Class order: `CD,HYP,MI,NORM,STTC`
- Launcher: `scripts/run_experiment.py`

No GPU training, sudo, or system environment changes were used.

## Removed From Public Tree

- Deleted historical docs/reports/plans from the first cleanup pass: 240 files.
- Deleted transient untracked experiment YAMLs from the first cleanup pass: 24 files.
- Deleted historical `configs/experiments/*.yaml` not in latest mainline or SOTA replay: 491 files.
- Deleted `trash/`: 251 files.
- Deleted non-core tests outside the requested public-mainline regression set: 62 files.
- Deleted old `docs/tmp_html/` files: 48 files.
- Deleted non-mainline demo/prototype code: 39 files, including `apps/streamlit_ecg_demo/`, `methods/ecgtwin_gen/`, `sub_experiment/`, and AugMix visualization-only files.

Kept because they are still in the latest-mainline execution closure:

- `ecg_adv_gen/runner/synth_online_at_super5.py`
- `ecg_adv_gen/training/tierm_online_adv.py`
- `methods/augmix/ecg_ops.py`
- `methods/augmix/severity.py`
- `methods/augmix/jsd_loss.py`
- `methods/augmix/latent_viz/latent_augmix.py`
- `adversarial/latent_hull_pgd.py`

## Current Public Surface

- Tracked text/config/code lines: 68,256.
- `configs/experiments/*.yaml`: 18 files.
- Latest mainline configs: 10 files.
- SOTA replay configs: 8 files, marked `sota_replay_reference`.
- Core test files under `util/tests/`: 11 files.
- `trash/` files remaining: 0.

Largest remaining tracked line groups:

| Area | Lines |
|---|---:|
| `ecg_adv_gen/` | 45,728 |
| `util/` | 6,882 |
| `configs/` | 4,306 |
| `adversarial/` | 3,310 |
| `scripts/` | 2,077 |
| `.codex/` | 1,572 |
| `docs/` | 1,464 |
| `methods/` | 1,272 |
| `data/` | 382 |

Core human audit surface:

| Area | Lines |
|---|---:|
| `ecg_adv_gen/runner` | 19,077 |
| `ecg_adv_gen/config` | 5,563 |
| `ecg_adv_gen/evaluation` | 5,159 |
| `ecg_adv_gen/data` | 4,707 |
| `adversarial` | 3,310 |
| `ecg_adv_gen/training` | 2,471 |
| `ecg_adv_gen/adaptation` | 1,733 |
| `ecg_adv_gen/models` | 1,219 |
| `methods/augmix` | 1,081 |
| `ecg_adv_gen/labels` | 837 |

## Reproduction Entry

Use `configs/active_scripts.yaml:latest_mainline` as the ordered stage list.

Dry-run any stage:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python scripts/run_experiment.py \
  --config configs/experiments/<stage>.yaml \
  --local-config configs/local/linbinhao_server.example.yaml \
  --run-id <run_id> \
  --dry-run
```

The public config folder now contains only latest-mainline YAMLs plus the 8 preserved SOTA replay YAMLs.

## Verification

Passed:

- `git diff --check`
- Latest-mainline dry-run: 10/10 configs passed through `scripts/run_experiment.py --dry-run`
- Core pytest:

```text
189 passed in 37.87s
```

Core pytest command:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest -q \
  util/tests/test_config_loader.py \
  util/tests/test_active_script_index.py \
  util/tests/test_pn2021c_eval_adapter.py \
  util/tests/test_ecgfounder_pn2021c_evaluator.py \
  util/tests/test_data_contracts.py \
  util/tests/test_labels_super5.py \
  util/tests/test_pn2021_eval_cache.py \
  util/tests/test_ref_exclusion.py \
  util/tests/test_pn2021c_protocol.py \
  util/tests/test_pn2021_corruptions.py \
  util/tests/test_pn2021_metric_views.py
```

CPU-only agent audit:

```text
passed: true
warning_count: 10
```

Warnings are expected for the current dirty cleanup branch: guarded external model handles are dirty, 8 SOTA replay YAMLs are currently untracked, and source-of-truth files are dirty/untracked until this cleanup is staged and committed.

## Remaining Risks

- The repo still has a large dirty worktree from the broader refactor. Review before commit.
- Do not stage guarded external model handles under `model/`.
- Keep `ecg_adv_gen/runner/synth_online_at_super5.py` for now. It is large, but it remains the actual EfficientNet VAE-LHAT training command target.
- A future reduction pass can split runner training code after the current mainline is reproduced from the cleaned tree.
