# PN2021-C Locked Official Severity-5 Summary 2026-06-18

Verdict: **accept official severity=5; ECGFounder AUPRC mean drop is within the 8-12 pp target band; no custom severity profile needed**.

Protocol: K500 train-only, ref IDs excluded from PN2021 evaluation, last checkpoint only, standard official severity=5, raw/interpolated waveform corruption before z-score, no stabilizer. ECGFounder uses the bottleneck5000 path: raw ECG -> 100Hz/1000 -> 500Hz/5000 -> corruption -> per-sample global z-score -> model.

## Overall

|model|n_units|clean_auroc|clean_auprc|corrupted_auroc|corrupted_auprc|drop_auroc_pp|drop_auprc_pp|
|---|---|---|---|---|---|---|---|
|ECGFounder|20|0.882|0.599|0.825|0.509|5.72|8.98|
|EfficientNet1DV2|20|0.877|0.568|0.829|0.494|4.83|7.43|

## By Operator

|model|corruption|n_units|clean_auroc|clean_auprc|corrupted_auroc|corrupted_auprc|drop_auroc_pp|drop_auprc_pp|
|---|---|---|---|---|---|---|---|---|
|ECGFounder|baseline_shift|4|0.882|0.599|0.847|0.547|3.49|5.24|
|ECGFounder|baseline_wander|4|0.882|0.599|0.867|0.577|1.53|2.24|
|ECGFounder|emg_noise|4|0.882|0.599|0.822|0.501|5.98|9.84|
|ECGFounder|powerline_noise|4|0.882|0.599|0.795|0.470|8.65|12.92|
|ECGFounder|random_leads_masking|4|0.882|0.599|0.792|0.453|8.95|14.66|
|EfficientNet1DV2|baseline_shift|4|0.877|0.568|0.818|0.484|5.92|8.41|
|EfficientNet1DV2|baseline_wander|4|0.877|0.568|0.847|0.522|2.98|4.60|
|EfficientNet1DV2|emg_noise|4|0.877|0.568|0.828|0.487|4.86|8.12|
|EfficientNet1DV2|powerline_noise|4|0.877|0.568|0.867|0.554|1.01|1.47|
|EfficientNet1DV2|random_leads_masking|4|0.877|0.568|0.783|0.423|9.40|14.54|

## By Center

|model|center|n_units|clean_auroc|clean_auprc|corrupted_auroc|corrupted_auprc|drop_auroc_pp|drop_auprc_pp|
|---|---|---|---|---|---|---|---|---|
|ECGFounder|chapman_shaoxing|5|0.905|0.520|0.851|0.442|5.38|7.80|
|ECGFounder|cpsc_2018|5|0.883|0.706|0.819|0.576|6.38|13.02|
|ECGFounder|georgia|5|0.845|0.643|0.789|0.567|5.55|7.61|
|ECGFounder|ningbo|5|0.895|0.528|0.840|0.453|5.57|7.50|
|EfficientNet1DV2|chapman_shaoxing|5|0.899|0.513|0.851|0.427|4.86|8.61|
|EfficientNet1DV2|cpsc_2018|5|0.883|0.622|0.826|0.538|5.67|8.38|
|EfficientNet1DV2|georgia|5|0.834|0.614|0.789|0.558|4.42|5.55|
|EfficientNet1DV2|ningbo|5|0.892|0.525|0.848|0.453|4.39|7.18|

## Source Artifacts

- EfficientNet1DV2 eval root: `/home/linbinhao/ECG_adv_data/runs/pn2021c_effnet_threechain_locked_official_s5/locked_rawfirst_20260618_effnet_copies2/`
- ECGFounder eval root: `/home/linbinhao/ECG_adv_data/runs/pn2021c_ecgfounder_threechain_locked_official_s5/locked_rawfirst_20260618/`
- ECGFounder launch manifest: `/home/linbinhao/ECG_adv_data/runs/managed_launches/pn2021c_ecgfounder_threechain_locked_official_s5_v1/run_manifest.json`
- Protocol trace CSV: `pn2021c_locked_official_s5_protocol_trace_20260618.csv`
