---
name: data-prep-validator
description: Review ECG preprocessing, splits, mapping and input identity, keeping traditional raw100 classifiers separate from native500 PULSE/ECG-R1 image paths.
---

# Data Prep Validator

Start with the selected YAML and the data contracts in `AGENTS.md`.
A status/explanation request calls for inspection, not rebuilding caches.

## Shared Super5 Identity

- Mapping owner: `configs/data/PN2021_super5_v7.yaml`; version
  `v7_super5_sjr_rgq_review_20260528`, hash `555ec85d5b51`,
  class order `CD,HYP,MI,NORM,STTC`.
- One center's fixed K500 is adaptation data; exclude those record identities
  from target evaluation. Keep CPSC2018 + Extra one logical center.
- PTB-XL source records must not enter the locked PN2021 target evaluation.
  Check record identity, not merely array position or display filenames.

## Select the Input Path

- Traditional: raw physical mV `(B,1000,12)`, 100 Hz, PTB-XL lead order.
  Corrupt in 500 Hz then return to 100 Hz; global per-sample z-score follows.
  Founder interpolates 1000 -> 5000 on device before normalization.
- VAE: ECGTwin `(B,1024,12)`; bridge through `models/vae.py` / `core/lhat.py`
  to canonical 100 Hz, with lead reorder `[0,1,2,3,5,4,6,7,8,9,10,11]`.
- PULSE/ECG-R1 images: follow the native500 waveform, lead/paper layout, renderer
  and model-specific processor configuration. Do not impose raw100 downsampling
  or classifier z-score on this path. R1's current evaluation is image-only.
  Renderer/processor edits need real-input pixel/tensor and model parity;
  correct tensor shape alone is insufficient.

## Validate Proportionately

1. Identify source/cache/split versions, units, layout, normalization and IDs.
2. Check the relevant mapping, cohort, K500-exclusion and clean/corrupt alignment.
3. Distinguish ledger quick inventory/size checks from full content verification;
   neither alone proves preprocessing correctness.
4. For an authorized change, run the smallest relevant fixture/smoke and record
   lineage. Cache rebuilds or heavy GPU checks require resource preflight.
5. Flag unexplained rate/lead/split/mapping/metric drift; do not silently repair
   metadata to reuse incompatible cache bytes.

Missing dataset/mapping/K500 identity blocks a trustworthy comparison, not all
read-only diagnosis. Do not overwrite caches or create a new split by default.
