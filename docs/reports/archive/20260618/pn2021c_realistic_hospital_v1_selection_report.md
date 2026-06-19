# realistic_hospital_v1 Selection Report

Candidate scores: `/home/linbinhao/ECG_adv_data/runs/pn2021c_realistic_hospital_sweep_20260618/two_center_combined_candidate_scores.csv`
Frozen profile: `configs/corruption_profiles/pn2021c_realistic_hospital_v1.yaml`

| Operator | Selected candidate | Model | Centers | Mean AUROC drop pp | Mean AUPRC drop pp | Gap pp | Accepted | Branch | Params |
|---|---|---|---|---:|---:|---:|---|---|---|
| baseline_shift | shift_amp0p6_r0p35_seg2 | ecgfounder | cpsc_2018,georgia | 5.1485 | 9.2289 | 2.4387 | true |  | {"dependency":false,"freq":100,"max_amplitude":0.6,"min_amplitude":0.0,"num_segment":2,"p":1.0,"shift_ratio":0.35} |
| baseline_shift | shift_amp0p6_r0p35_seg2 | effnet | cpsc_2018,georgia | 7.5872 | 9.7539 | 2.4387 | true |  | {"dependency":false,"freq":100,"max_amplitude":0.6,"min_amplitude":0.0,"num_segment":2,"p":1.0,"shift_ratio":0.35} |
| baseline_wander | wander_amp1p2_k4_f0p4 | ecgfounder | cpsc_2018,georgia | 7.3397 | 12.1342 | 0.2553 | true |  | {"dependency":false,"freq":100,"k":4,"max_amplitude":1.2,"max_freq":0.4,"min_amplitude":0.0,"min_freq":0.03,"p":1.0} |
| baseline_wander | wander_amp1p2_k4_f0p4 | effnet | cpsc_2018,georgia | 7.5950 | 10.0022 | 0.2553 | true |  | {"dependency":false,"freq":100,"k":4,"max_amplitude":1.2,"max_freq":0.4,"min_amplitude":0.0,"min_freq":0.03,"p":1.0} |
| emg_noise | emg_amp0p32 | ecgfounder | cpsc_2018,georgia | 9.5332 | 16.8828 | 3.5995 | false |  | {"dependency":false,"max_amplitude":0.32,"min_amplitude":0.0,"p":1.0} |
| emg_noise | emg_amp0p32 | effnet | cpsc_2018,georgia | 5.9337 | 8.6573 | 3.5995 | false |  | {"dependency":false,"max_amplitude":0.32,"min_amplitude":0.0,"p":1.0} |
| powerline_noise | power_native_amp3p5 | ecgfounder | cpsc_2018,georgia | 8.3443 | 11.8921 | 2.9161 | true | native_raw_first | {"dependency":false,"freq":100,"max_amplitude":3.5,"min_amplitude":0.0,"p":1.0} |
| powerline_noise | power_native_amp3p5 | effnet | cpsc_2018,georgia | 5.4282 | 6.0612 | 2.9161 | true | native_raw_first | {"dependency":false,"freq":100,"max_amplitude":3.5,"min_amplitude":0.0,"p":1.0} |
| random_leads_masking | mask_p0p40 | ecgfounder | cpsc_2018,georgia | 6.0754 | 11.5608 | 1.0880 | false |  | {"mask_leads_prob":0.4,"mask_leads_selection":"random","p":1.0} |
| random_leads_masking | mask_p0p40 | effnet | cpsc_2018,georgia | 4.9873 | 7.5098 | 1.0880 | false |  | {"mask_leads_prob":0.4,"mask_leads_selection":"random","p":1.0} |
