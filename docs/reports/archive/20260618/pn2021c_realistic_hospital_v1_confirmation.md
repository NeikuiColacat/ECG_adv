# realistic_hospital Four-Center Confirmation

Source scores: `/home/linbinhao/ECG_adv_data/runs/pn2021c_realistic_hospital_sweep_20260618/four_center_confirmation/candidate_scores.csv`
CSV: `docs/reports/archive/20260618/pn2021c_realistic_hospital_v1_confirmation.csv`
Score profile: `realistic_hospital`

| Operator | Candidate | Input modes | Powerline branch | EffNet AUROC drop pp | ECGFounder AUROC drop pp | Gap pp | Status |
|---|---|---|---|---:|---:|---:|---|
| baseline_shift | realistic_hospital_v1 | bottleneck5000,raw_first |  | 7.3258 | 4.7249 | 2.6009 | miss_low |
| baseline_wander | realistic_hospital_v1 | bottleneck5000,raw_first |  | 8.1459 | 6.4821 | 1.6637 | realistic_main |
| emg_noise | realistic_hospital_v1 | bottleneck5000,raw_first |  | 6.9545 | 9.5273 | 2.5728 | realistic_main |
| powerline_noise | realistic_hospital_v1 | native_raw_first | native_raw_first | 5.3235 | 9.5725 | 4.2491 | gap_high |
| random_leads_masking | realistic_hospital_v1 | bottleneck5000,raw_first |  | 6.2346 | 6.6149 | 0.3802 | realistic_main |
