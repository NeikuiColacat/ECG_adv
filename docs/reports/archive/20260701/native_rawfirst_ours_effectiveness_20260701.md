# Native Raw-First PN2021-C Ours Effectiveness Check

Date: 2026-07-01

Scope: official severity=5 depth2+depth3 composite PN2021-C, four centers
(`cpsc_2018`, `chapman_shaoxing`, `georgia`, `ningbo`), all-zero-kept macro
AUROC/AUPRC. This run moves corruption to the earliest raw ECG position:

`WFDB native fs/native length -> lead reorder + NaN guard -> corruption -> model preprocessing -> z-score -> model`.

## Ours Under Native Raw-First

| Backbone | Method | Depth | AUROC/AUPRC | Drop vs own clean |
|---|---|---:|---:|---:|
| EfficientNet1DV2 | c117 VAE-LH + locked three-chain AugMix | 2 | 0.8358 / 0.5222 | 4.47 / 6.21 pp |
| EfficientNet1DV2 | c117 VAE-LH + locked three-chain AugMix | 3 | 0.8131 / 0.4941 | 6.74 / 9.02 pp |
| EfficientNet1DV2 | c117 VAE-LH + locked three-chain AugMix | 2+3 | 0.8244 / 0.5082 | 5.60 / 7.61 pp |
| ECGFounder | c110 VAE-LH + locked three-chain AugMix | 2 | 0.8246 / 0.5383 | 6.19 / 9.76 pp |
| ECGFounder | c110 VAE-LH + locked three-chain AugMix | 3 | 0.7989 / 0.4972 | 8.76 / 13.87 pp |
| ECGFounder | c110 VAE-LH + locked three-chain AugMix | 2+3 | 0.8117 / 0.5178 | 7.47 / 11.81 pp |

## Readout

The method still works under native raw-first corruption. Absolute PN2021-C
performance does not collapse; it is higher than the earlier report's
model-facing corruption placement for both backbones.

Strict same-position effectiveness against Direct K500 requires rerunning the
Direct K500 baseline with `corruption_input=native_raw_first`. Do not claim a
same-position pp gain until that baseline is filled.

## Artifacts

- Summary CSV:
  `/home/linbinhao/ECG_adv_data/runs/pn2021c_native_rawfirst_official_s5_depth23_ours_20260701/native_rawfirst_official_s5_depth23_ours_summary.csv`
- Long CSV:
  `/home/linbinhao/ECG_adv_data/runs/pn2021c_native_rawfirst_official_s5_depth23_ours_20260701/native_rawfirst_official_s5_depth23_ours_long.csv`
- EffNet JSONs:
  `/home/linbinhao/ECG_adv_data/runs/pn2021c_native_rawfirst_official_s5_depth23_ours_20260701/effnet_c117/`
- ECGFounder JSONs:
  `/home/linbinhao/ECG_adv_data/runs/pn2021c_native_rawfirst_official_s5_depth23_ours_20260701/ecgfounder_c110/`
