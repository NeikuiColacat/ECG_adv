# Feature Distribution Analysis

Feature extractor: EfficientNet1DV2 penultimate global pooled feature.

Interpretation: C2ST balanced accuracy close to 0.5 means two feature sets are hard to separate; higher means stronger distribution gap. FID/MMD are relative distances and should be compared against the real-train vs real-test baseline.

## Summary

| comparison | class | n_a | n_b | FID | MMD-RBF | C2ST bal. acc. |
|---|---|---:|---:|---:|---:|---:|
| real_train2000_vs_real_test | CD | 477 | 477 | 0.1160 | 0.0033 | 0.4976 |
| synth_vs_real_test | CD | 477 | 477 | 2.5789 | 0.0933 | 0.8572 |
| synth_vs_real_train2000 | CD | 477 | 477 | 2.3367 | 0.0911 | 0.8711 |
| real_train2000_vs_real_test | HYP | 255 | 255 | 0.1495 | 0.0000 | 0.5136 |
| synth_vs_real_test | HYP | 255 | 255 | 11.5878 | 0.2550 | 0.9268 |
| synth_vs_real_train2000 | HYP | 255 | 255 | 11.6066 | 0.2468 | 0.9203 |
| real_train2000_vs_real_test | MI | 560 | 560 | 0.0273 | 0.0003 | 0.5315 |
| synth_vs_real_test | MI | 560 | 560 | 5.5934 | 0.2879 | 0.8387 |
| synth_vs_real_train2000 | MI | 560 | 560 | 5.3585 | 0.2996 | 0.8369 |
| real_train2000_vs_real_test | NORM | 600 | 600 | 0.0358 | 0.0014 | 0.4989 |
| synth_vs_real_test | NORM | 600 | 600 | 9.5948 | 0.4055 | 0.9461 |
| synth_vs_real_train2000 | NORM | 600 | 600 | 8.8570 | 0.3833 | 0.9411 |
| real_train2000_vs_real_test | STTC | 499 | 499 | 0.0647 | 0.0000 | 0.5053 |
| synth_vs_real_test | STTC | 499 | 499 | 5.7420 | 0.2001 | 0.9040 |
| synth_vs_real_train2000 | STTC | 499 | 499 | 6.2867 | 0.2184 | 0.9160 |

## Config

```json
{
  "model_ckpt": "/root/autodl-tmp/triple_labels/super5_minresample_full10_perglobal_20260503/best_model.pt",
  "real_cache": "/root/autodl-tmp/triple_labels/cache/ptbxl_minimal_resample_per_sample_global_fs100_len1000.npy",
  "labels": "/root/autodl-tmp/graduate_project/method_a_real2000_seed42/ptbxl_labels.C5.all.npy",
  "split_json": "/root/autodl-tmp/graduate_project/splits/ptbxl_super5_seed42_train2000_val2000.json",
  "synth_npz": "/root/autodl-tmp/graduate_project/self_distill_v2_filtered_v46_ptbxl_contrast_seed42/synth_v2_filtered_top4000_gamma03.npz",
  "out_dir": "/root/autodl-tmp/final_round_ablation_20260504/feature_distribution",
  "max_per_class": 600,
  "pca_per_class": 160,
  "batch_size": 256,
  "seed": 42,
  "device": "cuda",
  "class_names": [
    "CD",
    "HYP",
    "MI",
    "NORM",
    "STTC"
  ],
  "selection": {
    "CD": {
      "n": 477,
      "train_indices_preview": [
        3048,
        18318,
        9974,
        15464,
        1590,
        9086,
        3111,
        10035,
        20516,
        2236
      ],
      "test_indices_preview": [
        15399,
        7084,
        4756,
        14782,
        12738,
        18641,
        10096,
        20689,
        9526,
        3471
      ],
      "synth_indices_preview": [
        141,
        781,
        573,
        665,
        248,
        416,
        445,
        301,
        791,
        484
      ],
      "available": {
        "synth": 800,
        "train": 477,
        "test": 3997
      }
    },
    "HYP": {
      "n": 255,
      "train_indices_preview": [
        16001,
        3135,
        14202,
        17830,
        19497,
        1768,
        3301,
        10227,
        4426,
        3749
      ],
      "test_indices_preview": [
        1850,
        21339,
        19989,
        17406,
        9530,
        10575,
        14223,
        2441,
        20451,
        16849
      ],
      "synth_indices_preview": [
        1464,
        1364,
        1195,
        1579,
        986,
        810,
        1218,
        1239,
        1039,
        1274
      ],
      "available": {
        "synth": 800,
        "train": 255,
        "test": 2136
      }
    },
    "MI": {
      "n": 560,
      "train_indices_preview": [
        7929,
        19065,
        17355,
        20392,
        14604,
        6207,
        2606,
        5023,
        15811,
        11616
      ],
      "test_indices_preview": [
        8645,
        17034,
        3936,
        3146,
        8675,
        19898,
        6259,
        10799,
        15475,
        6741
      ],
      "synth_indices_preview": [
        1808,
        1960,
        2276,
        2164,
        1607,
        1855,
        1635,
        2001,
        2132,
        2122
      ],
      "available": {
        "synth": 800,
        "train": 560,
        "test": 4415
      }
    },
    "NORM": {
      "n": 600,
      "train_indices_preview": [
        284,
        7363,
        17848,
        10555,
        5406,
        20585,
        155,
        18919,
        2109,
        1260
      ],
      "test_indices_preview": [
        19257,
        11573,
        12101,
        18368,
        20493,
        19946,
        14186,
        14570,
        8403,
        924
      ],
      "synth_indices_preview": [
        2697,
        3019,
        2768,
        2602,
        2677,
        3113,
        2547,
        2503,
        2570,
        3098
      ],
      "available": {
        "synth": 800,
        "train": 839,
        "test": 7808
      }
    },
    "STTC": {
      "n": 499,
      "train_indices_preview": [
        12747,
        8337,
        21126,
        15577,
        531,
        12472,
        11385,
        4717,
        2156,
        287
      ],
      "test_indices_preview": [
        19033,
        6017,
        10283,
        10277,
        21008,
        5321,
        12013,
        3914,
        20828,
        12476
      ],
      "synth_indices_preview": [
        3323,
        3612,
        3464,
        3529,
        3675,
        3443,
        3545,
        3610,
        3523,
        3325
      ],
      "available": {
        "synth": 800,
        "train": 499,
        "test": 4244
      }
    }
  }
}
```
