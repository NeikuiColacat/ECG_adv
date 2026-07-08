# Latest Mainline Dry-Run Check

Generated: 2026-07-05

Command shape: `micromamba run -n ECGTwin python scripts/run_experiment.py --config <config> --local-config configs/local/linbinhao_server.example.yaml --run-id cleanup_manifest_dryrun_<stage> --dry-run`

| stage | role | config | status | log |
|---|---|---|---|---|
| `effnet_direct_k500_v7` | `efficientnet_direct_baseline` | `configs/experiments/effnet_direct_k500_v7_sjr_rgq.yaml` | PASS | `/dev/shm/ecg_cleanup_dryrun_20260705/effnet_direct_k500_v7.log` |
| `effnet_direct_k500_v7_pn2021_clean` | `efficientnet_direct_baseline_pn2021_clean_refexcluded_eval` | `configs/experiments/pn2021_eval_v7_sjr_rgq_refexcluded.yaml` | PASS | `/dev/shm/ecg_cleanup_dryrun_20260705/effnet_direct_k500_v7_pn2021_clean.log` |
| `effnet_threechain_locked_train` | `efficientnet_main_method_train` | `configs/experiments/effnet_vae_lhat_augmix_threechain_locked_k500.yaml` | PASS | `/dev/shm/ecg_cleanup_dryrun_20260705/effnet_threechain_locked_train.log` |
| `effnet_threechain_locked_pn2021c_s5` | `efficientnet_main_method_pn2021c_eval` | `configs/experiments/pn2021c_effnet_threechain_locked_official_s5.yaml` | PASS | `/dev/shm/ecg_cleanup_dryrun_20260705/effnet_threechain_locked_pn2021c_s5.log` |
| `effnet_threechain_locked_pn2021c_depth23` | `efficientnet_main_method_depth23_composite_eval` | `configs/experiments/pn2021c_effnet_official_s5_depth23_composite.yaml` | PASS | `/dev/shm/ecg_cleanup_dryrun_20260705/effnet_threechain_locked_pn2021c_depth23.log` |
| `ecgfounder_ptbxl_locked_fullft` | `ecgfounder_source_fullft_prerequisite` | `configs/experiments/ecgfounder_ptbxl_super5_fullft_locked.yaml` | PASS | `/dev/shm/ecg_cleanup_dryrun_20260705/ecgfounder_ptbxl_locked_fullft.log` |
| `ecgfounder_k500_locked_fullft` | `ecgfounder_k500_fullft_prerequisite` | `configs/experiments/ecgfounder_k500_fullft_locked.yaml` | PASS | `/dev/shm/ecg_cleanup_dryrun_20260705/ecgfounder_k500_locked_fullft.log` |
| `ecgfounder_threechain_locked_train` | `ecgfounder_main_method_train` | `configs/experiments/ecgfounder_vae_lhat_augmix_threechain_locked_k500.yaml` | PASS | `/dev/shm/ecg_cleanup_dryrun_20260705/ecgfounder_threechain_locked_train.log` |
| `ecgfounder_threechain_locked_pn2021c_s5` | `ecgfounder_main_method_pn2021c_eval` | `configs/experiments/pn2021c_ecgfounder_threechain_locked_official_s5.yaml` | PASS | `/dev/shm/ecg_cleanup_dryrun_20260705/ecgfounder_threechain_locked_pn2021c_s5.log` |
| `ecgfounder_threechain_locked_pn2021c_depth23` | `ecgfounder_main_method_depth23_composite_eval` | `configs/experiments/pn2021c_ecgfounder_official_s5_depth23_composite.yaml` | PASS | `/dev/shm/ecg_cleanup_dryrun_20260705/ecgfounder_threechain_locked_pn2021c_depth23.log` |

Overall: 10/10 passed.
