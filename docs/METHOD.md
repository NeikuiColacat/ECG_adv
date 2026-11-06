# Single-stage R18 JSD 1.5 + LHAT replacement 0.20

The main selector is
`configs/train/methods/a1_rot4_two_chain_balanced_jsd1p5_vae_lhat_replace0p2.yaml`.
`configs/train/PN2021_a1_joint_long_budget_r13.yaml` owns backbone budgets.
R18 is the existing single-stage joint recipe: no SimCLR/projector, source
replay, source-logit anchor, Stage-2 teacher or stage boundary.
Historical two-stage JSD weight 12 is a different method.

## Locked method

Clean and rotating corruption supervision run jointly with two-chain AugMix.
Multilabel Bernoulli JSD weight is 1.5; AugMix supervised view policy is
`chains_mean`. The base family masses are clean 0.25, corruption total 0.25
(four views at 0.0625), and AugMix auxiliary 0.5.
LHAT's supervised mass 0.20 replaces clean supervision; it is not an additional
20% loss. Anchors without eligible hard views retain clean supervision.

| Backbone | Epochs / cosine horizon | Batch / lr |
| --- | --- | --- |
| EfficientNet1DV2 | 50 / 60 | 128 / 5e-5 |
| ECGFounder | 60 / 60 | 64 / 3e-5 |

Use AdamW, weight decay 1e-4, clip norm 1 and bf16. Use all K500 each epoch,
no validation split, and the final checkpoint.

The selector references `configs/train/lhat_pure_delta_nondecreasing.yaml`:
nearest exact-label non-self M=20 from local pool 80, hull lambda 1,
standardized latent L2 epsilon 12, one Adam attack step at lr 0.25.
Contraction `nondecreasing_pure_delta_grid_v3` searches t=[0,0.25,0.5,0.75,1],
keeps BCE at least clean with no new clean-correct flips, margin retention 0.5,
and uses the constant clean-anchor residual correction.
Do not replace this with SimCLR's linear endpoint correction.

Retain the matched JSD no-VAE selector
`a1_rot4_two_chain_balanced_jsd1p5` and both reference baselines:
`a0_clean_v1` (clean-only), `a1_corrupt_ft_rot4_v1` (direct corruption).
Compare matched budgets and resources when making a component claim.

## Data and model contract

Super5 order is `CD,HYP,MI,NORM,STTC`; mapping is
`v7_super5_sjr_rgq_review_20260528` / `555ec85d5b51`.
Logical centers are `ningbo`, `chapman_shaoxing`, `cpsc_2018` (including Extra)
and `georgia`. Adapt only on the selected fixed K500. Evaluate after excluding
those IDs; outside-K500 target records are development assessment only.

PTB-XL source training uses official folds 1–8, fold 9 selects best macro AP,
and fold 10 is evaluated after restoring that checkpoint.
Use the registered source `best.pt` and SHA256 in
`configs/baselines/ptbxl_source_v1.yaml`; a new source run does not update this lock.

Canonical waveform is float32 physical mV, `(B,1000,12)`, 100 Hz, PTB-XL lead order.
Select the ten-second center window before aligned-corner interpolation for PN2021;
PTB-XL records100 is already 100 Hz. Store only raw100 waveforms.
Corrupt before per-sample global z-score. EfficientNet retains 1000 points;
ECGFounder linearly interpolates `1000 -> 5000` on device, then normalizes.
Models receive channel-time tensors through the code-owned input adapter.

VAE I/O is `(B,1024,12)` in ECGTwin lead order, latent `(B,4,128)`.
Decoded leads reorder as `[0,1,2,3,5,4,6,7,8,9,10,11]` before classification.
Fit latent standardization on eligible K500 training records only.
Exact-positive-label, non-self, distinct candidates prevent NORM/abnormal mixing;
anchors without eligible candidates fall back to clean supervision.
No heldout record or target-center class frequency enters adaptation.

## Online evaluation and metrics

Use `configs/eval/PN2021_online_gpu.yaml`, backend `online_torch_gpu_v1`.
Read clean raw100 mmap and corruption metadata; execute raw100 -> GPU500
corruption -> GPU100, model-domain adaptation and inference without persisting
corrupted waveform arrays. Historical metadata paths/cache-version keys can remain
for identity and RNG; they do not require a disk corruption-waveform cache.

Evaluate clean and severity-5 PN2021-C: ten depth-2 plus ten depth-3 compositions.
Randomness is keyed by ordered record IDs, composition and cache version,
with fixed corruption microbatches independent of model inference batch.
Keep backend, cohort and metric view fixed for paired comparisons.
Historical NumPy cache corruption and online Torch corruption are different
random implementations; their metrics are not interchangeable causal controls.

Primary metrics are `drop_all_zero` macro AUROC and macro sklearn average precision.
The result key `macro_auprc` denotes AP, not trapezoidal PR area.
`all_zero_kept` is secondary audit evidence.
Average the twenty corruption views equally within each center, then the four
centers equally. Independent training repeats use mean and sample SD;
K500 resampling SD measures sampling sensitivity, not training-seed variability.
Raw scores are 0–1; report 0.01 absolute difference as one percentage point.

For LHAT report raw-search and contracted ASR, acceptance, BCE gain, selected t
and invalid-decode rate together. A low contracted ASR alone does not establish
an effective attack. Match source, K500, seed, budget and backend when attributing
a VAE increment; the full method's gain over clean-only is not all a VAE effect.

## Code ownership

| Location | Responsibility |
| --- | --- |
| `data_preprocess/`, `configs/data/` | Raw preprocessing, caches, immutable splits, ledger and finite loader plans |
| `models/` | EfficientNet, ECGFounder, VAE and model input adapters |
| `core/online_trainer.py`, `core/methods/` | Training, scientific method graph and recipe identity |
| `core/augmix.py`, `core/corruption.py`, `util/augmentations/` | AugMix and ECG corruption |
| `core/lhat.py`, `core/latent_pool.py` | Train-only latent candidate pool, attack and contraction |
| `util/evaluation/` | Clean/corruption inference and metric aggregation |
| `util/config_bundle.py`, `util/random_seed.py`, `util/run_record.py` | Config closure, RNG and provenance |
| `boot_scripts/` | Managed launcher and finite task entrypoints |
| `util/tests/` | Data, numerical, launcher and artifact contracts |

Runtime quick ledger validation checks exact membership and byte sizes; it does
not rehash every waveform array. Full content sealing requires a separate full
hash check of quiescent data. Do not call quick verification a full integrity audit.
