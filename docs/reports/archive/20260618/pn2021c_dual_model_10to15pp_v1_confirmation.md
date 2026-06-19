# dual_model_10to15pp_v1 Four-Center Confirmation

Source scores: `/home/linbinhao/ECG_adv_data/runs/pn2021c_dual_model_10to15pp_sweep_20260618/four_center_confirmation/candidate_scores.csv`
CSV: `docs/reports/archive/20260618/pn2021c_dual_model_10to15pp_v1_confirmation.csv`

| Operator | Candidate | Input modes | Powerline branch | EffNet AUROC drop pp | ECGFounder AUROC drop pp | Gap pp | Status |
|---|---|---|---|---:|---:|---:|---|
| baseline_shift | dual_model_10to15pp_v1 | bottleneck5000,raw_first |  | 12.7556 | 6.3555 | 6.4001 | miss_low |
| baseline_wander | dual_model_10to15pp_v1 | bottleneck5000,raw_first |  | 12.0069 | 12.4982 | 0.4913 | in_band |
| emg_noise | dual_model_10to15pp_v1 | bottleneck5000,raw_first |  | 15.1512 | 15.3773 | 0.2261 | miss_high |
| powerline_noise | dual_model_10to15pp_v1 | native_raw_first | native_raw_first | 11.2029 | 16.2444 | 5.0415 | miss_high |
| random_leads_masking | dual_model_10to15pp_v1 | bottleneck5000,raw_first |  | 12.5482 | 11.9601 | 0.5882 | in_band |
