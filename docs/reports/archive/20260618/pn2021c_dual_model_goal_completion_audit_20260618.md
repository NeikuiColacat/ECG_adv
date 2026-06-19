# PN2021-C Dual-Model Calibration Goal Completion Audit

Date: 2026-06-18 14:28 +08:00

## Goal

Execute `docs/pipelines/pn2021c_dual_model_corruption_calibration_plan_20260618.md`
to find a `dual_model_10to15pp_v1` profile where EfficientNet1DV2
direct-K500/fullFT and ECGFounder direct-K500/fullFT both drop roughly 10-15 pp
AUROC on the five PN2021-C operators, with same-operator inter-model AUROC-drop
gap preferably <= 3 pp.

## Current Verdict

Not complete. The implementation and launch scaffolding are ready, but the
required GPU sweep has not run because all visible GPUs were occupied at every
preflight check through 2026-06-18 14:29:48 +08:00.

## Requirement Status

| Requirement | Status | Current evidence |
|---|---|---|
| Read `AGENTS.md` first 100 lines and obey shared-server rules | Done | Re-read before GPU checks in this session. No GPU job was launched because `nvidia-smi` showed no free GPU. |
| Write outputs under `/home/linbinhao/ECG_adv_data/runs/` without overwriting old runs | Done so far | Active run root: `/home/linbinhao/ECG_adv_data/runs/pn2021c_dual_model_10to15pp_sweep_20260618/`. |
| Custom YAML/JSON corruption profile support in EfficientNet evaluator | Done | `scripts/triple_labels/eval_pn2021_corruptions.py` supports `--severity_profile custom`, profile file/name, stream-only custom mode, and resolved params in JSON. |
| Custom YAML/JSON corruption profile support in ECGFounder evaluator | Done | `scripts/triple_labels/eval_ecgfounder_pn2021_corruptions.py` supports the same custom profile surface and resolved params. |
| Direct-K500/fullFT baselines only | Prepared | Baseline inventory: `docs/reports/archive/20260618/pn2021c_dual_model_calibration_baseline_inventory_20260618.md`. Sweep helper points to direct EffNet and ECGFounder fullFT paths only. |
| Raw ECG corruption before model preprocessing, no extra stabilizer/oracle repair | Prepared | Sweep commands use `raw_first`, `bottleneck5000`, or `native_raw_first` explicitly; no stabilizer/oracle module is added. |
| Sweep five operators | Prepared | Candidate file includes `baseline_shift`, `baseline_wander`, `emg_noise`, `powerline_noise`, `random_leads_masking`. |
| Compare powerline locked path and native raw-first path | Prepared | Candidate grid includes `power_locked_*` and `power_native_*`; aggregate emits `powerline_branch`. |
| CPSC 2018 + Georgia two-center calibration | Not run | Full two-center manifest has 96 tasks, but no `two_center_calibration/candidate_scores.csv` or result JSONs exist. |
| Freeze `dual_model_10to15pp_v1` after two-center calibration | Not run | No frozen profile file exists yet. This must wait for two-center `candidate_scores.csv`. |
| Four-center confirmation after freeze | Not run | `plan-frozen` command exists, but cannot be run until frozen profile exists. |
| Realism diagnostics with visual samples and RMS/PSD checks | Tool ready, not run | `scripts/agent/pn2021c_realism_diagnostics.py` exists and is tested; no diagnostics outputs exist yet because there is no frozen profile. |
| Final CSV/Markdown summaries | Tool ready, not run | `aggregate`, `selection-report`, and `confirmation-summary` exist and are tested; final score files are absent. |

## Existing Artifacts

Run root:

```text
/home/linbinhao/ECG_adv_data/runs/pn2021c_dual_model_10to15pp_sweep_20260618/
```

Prepared manifests:

```text
smoke_cpsc2018_limit2000/manifest.csv              # 48 tasks + header
two_center_calibration/manifest.csv                # 96 tasks + header
```

Prepared shard scripts:

```text
smoke_cpsc2018_limit2000/commands_shard{0,1,2,3}.sh
two_center_calibration/commands_shard{0,1,2,3}.sh
```

CPU smoke outputs:

```text
smoke/effnet_cpsc_emg_amp2p3_limit16_cpu.json
smoke/ecgfounder_cpsc_emg_amp2p3_limit16_cpu.json
```

CPU smoke is pipeline evidence only. It must not be interpreted as calibration
evidence because it uses `limit=16`.

## Missing Required Artifacts

These are the concrete files that still need to be produced before the goal can
be considered complete:

```text
two_center_calibration/candidate_scores.csv
configs/corruption_profiles/pn2021c_dual_model_10to15pp_v1.yaml
docs/reports/archive/20260618/pn2021c_dual_model_10to15pp_v1_selection_report.md
diagnostics/diagnostics.csv
diagnostics/diagnostics.json
diagnostics/**/clean_gallery.png
diagnostics/**/corrupted_gallery.png
diagnostics/**/comparison_overlay.png
four_center_confirmation/manifest.csv
four_center_confirmation/candidate_scores.csv
docs/reports/archive/20260618/pn2021c_dual_model_10to15pp_v1_confirmation.csv
docs/reports/archive/20260618/pn2021c_dual_model_10to15pp_v1_confirmation.md
```

## Current Blocker

GPU availability. At 2026-06-18 14:29:48 +08:00:

```text
GPU 0-1: openvla python jobs, about 20 GB each, active utilization.
GPU 2-7: VLLM workers, about 22-23 GB each.
```

No GPU had enough free memory for the planned sweep. No process was killed.

## Next Commands Once GPUs Are Free

Run the full two-center calibration shards only after a fresh `nvidia-smi`
shows genuinely free GPUs:

```bash
cd /home/linbinhao/ECG_adv_Gen
GPU_ID=<gpu0> bash /home/linbinhao/ECG_adv_data/runs/pn2021c_dual_model_10to15pp_sweep_20260618/two_center_calibration/commands_shard0.sh
GPU_ID=<gpu1> bash /home/linbinhao/ECG_adv_data/runs/pn2021c_dual_model_10to15pp_sweep_20260618/two_center_calibration/commands_shard1.sh
GPU_ID=<gpu2> bash /home/linbinhao/ECG_adv_data/runs/pn2021c_dual_model_10to15pp_sweep_20260618/two_center_calibration/commands_shard2.sh
GPU_ID=<gpu3> bash /home/linbinhao/ECG_adv_data/runs/pn2021c_dual_model_10to15pp_sweep_20260618/two_center_calibration/commands_shard3.sh
```

Then continue with `aggregate`, `select`, `selection-report`, realism
diagnostics, `plan-frozen`, four-center confirmation, and `confirmation-summary`
as listed in:

```text
/home/linbinhao/ECG_adv_data/runs/pn2021c_dual_model_10to15pp_sweep_20260618/summary.md
```
