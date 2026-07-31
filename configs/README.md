# Handoff configuration bundle

This directory contains only the data and model contracts needed by the K500
comparison handoff.

| Path | Purpose |
|---|---|
| `data/PTBXL.yaml` | PTB-XL cache-generation contract |
| `data/PN2021.yaml` | PN2021 cache-generation contract |
| `data/PN2021_super5_v7.yaml` | Reviewed PN2021-to-Super5 projection |
| `data/splits.yaml` | Frozen PTB-XL and PN2021 split-generation policy |
| `data/data_load.yaml` | Shared mmap/runtime loader defaults |
| `data/k500_handoff.yaml` | Exact K500, mapping, module, and artifact identities |
| `augmentation/*.yaml` | PN2021-C corruption-cache contract |
| `eval/PN2021.yaml` | Ref-exclusion and evaluation population contract |
| `train/PTBXL.yaml` | Source-model input/training contract |
| `train/vae.yaml` | ECGTwin VAE architecture/checkpoint contract |
| `baselines/ptbxl_source_v1.yaml` | Locked source-model weight identities |
| `random_seed.yaml` | Base seed and named deterministic streams |

Tracked paths point only to Git-ignored local directories such as `data/`,
`cache/`, `runs/`, and `handoff_artifacts/`. Runtime consumers should still
pass explicit read-only `cache_dir` and `split_dir` arguments to
`get_dataloader`. Changing paths does not authorize changing protocol fields,
mapping, split membership, or random identities.

Large data, split arrays, and model checkpoints stay outside Git. Their
transport paths and SHA256 identities are declared by the two handoff
registries above.
