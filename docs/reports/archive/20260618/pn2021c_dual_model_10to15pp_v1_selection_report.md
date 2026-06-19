# dual_model_10to15pp_v1 Selection Report

Candidate scores: `/home/linbinhao/ECG_adv_data/runs/pn2021c_dual_model_10to15pp_sweep_20260618/two_center_combined_candidate_scores.csv`
Frozen profile: `configs/corruption_profiles/pn2021c_dual_model_10to15pp_v1.yaml`

| Operator | Selected candidate | Model | Centers | Mean AUROC drop pp | Mean AUPRC drop pp | Gap pp | Accepted | Branch | Params |
|---|---|---|---|---:|---:|---:|---|---|---|
| baseline_shift | shift_amp1p0_r0p55_seg2 | ecgfounder | cpsc_2018,georgia | 6.9208 | 12.0799 | 5.2711 | false |  | {"dependency":false,"freq":100,"max_amplitude":1.0,"min_amplitude":0.0,"num_segment":2,"p":1.0,"shift_ratio":0.55} |
| baseline_shift | shift_amp1p0_r0p55_seg2 | effnet | cpsc_2018,georgia | 12.1920 | 15.1886 | 5.2711 | false |  | {"dependency":false,"freq":100,"max_amplitude":1.0,"min_amplitude":0.0,"num_segment":2,"p":1.0,"shift_ratio":0.55} |
| baseline_wander | wander_amp1p6_k5_f0p6 | ecgfounder | cpsc_2018,georgia | 13.3093 | 19.3621 | 2.4428 | true |  | {"dependency":false,"freq":100,"k":5,"max_amplitude":1.6,"max_freq":0.6,"min_amplitude":0.0,"min_freq":0.03,"p":1.0} |
| baseline_wander | wander_amp1p6_k5_f0p6 | effnet | cpsc_2018,georgia | 10.8665 | 14.2862 | 2.4428 | true |  | {"dependency":false,"freq":100,"k":5,"max_amplitude":1.6,"max_freq":0.6,"min_amplitude":0.0,"min_freq":0.03,"p":1.0} |
| emg_noise | emg_amp0p6 | ecgfounder | cpsc_2018,georgia | 15.9236 | 24.1791 | 1.8147 | false |  | {"dependency":false,"max_amplitude":0.6,"min_amplitude":0.0,"p":1.0} |
| emg_noise | emg_amp0p6 | effnet | cpsc_2018,georgia | 14.1090 | 17.3492 | 1.8147 | false |  | {"dependency":false,"max_amplitude":0.6,"min_amplitude":0.0,"p":1.0} |
| powerline_noise | power_native_amp32 | ecgfounder | cpsc_2018,georgia | 14.9784 | 18.0037 | 4.9055 | false | native_raw_first | {"dependency":false,"freq":100,"max_amplitude":32.0,"min_amplitude":0.0,"p":1.0} |
| powerline_noise | power_native_amp32 | effnet | cpsc_2018,georgia | 10.0729 | 10.6250 | 4.9055 | false | native_raw_first | {"dependency":false,"freq":100,"max_amplitude":32.0,"min_amplitude":0.0,"p":1.0} |
| random_leads_masking | mask_p0p65 | ecgfounder | cpsc_2018,georgia | 10.9133 | 18.1402 | 0.4613 | true |  | {"mask_leads_prob":0.65,"mask_leads_selection":"random","p":1.0} |
| random_leads_masking | mask_p0p65 | effnet | cpsc_2018,georgia | 10.4521 | 13.6562 | 0.4613 | true |  | {"mask_leads_prob":0.65,"mask_leads_selection":"random","p":1.0} |
