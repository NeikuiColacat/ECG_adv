# Data configs

This directory stores tracked configuration files for dataset loading,
validation, preprocessing, splitting, and cache generation.

Runtime data and generated caches must remain outside Git.

`PN2021.yaml` points to `PN2021_super5_v7.yaml`, which is the runtime source
for the PN2021 SNOMED-to-Super5 mapping used during cache generation.

Both dataset caches contain `signals.npy` at 100 Hz and a derived
`signals_500hz.npy` created from the 100 Hz waveform by fixed, aligned-corner
linear interpolation for models that require a 5000-point input grid.

PN2021 native-rate waveforms are also converted to 100 Hz with aligned-corner
linear interpolation after the ten-second center window is selected. PTB-XL
`records100` waveforms are already native 100 Hz and therefore require no
interpolation for the base cache.

`splits.yaml` defines the immutable ID-only split layer over those caches.
PTB-XL uses official folds 1-8/9/10 with patient-disjoint verification. PN2021
uses four logical K500 centers; logical `cpsc_2018` combines the physical
`cpsc_2018` and `cpsc_2018_extra` sources while retaining both source names in
the split manifests. Split artifacts contain indices, record IDs and hash IDs,
never duplicated waveforms.

`data_load.yaml` binds the tracked `data_content_ledger_v1.jsonl` by SHA256 and
defines only the shared mmap engine policy consumed by
`data_preprocess/data_runtime.py`. Batch, worker, residency, validation and
model-input policy live in the three finite loader plans and their tracked
train/evaluation profiles; they are not generic defaults.

## Data content ledger tool status

`data_preprocess/data_ledger.py` deterministically generates and verifies a
relocatable JSONL content ledger for the four active cache/split roots. Quick
verification checks the exact member inventory and byte sizes; full
verification additionally streams every member through SHA256 and rejects
metadata-detectable changes during the scan. This is not filesystem snapshot
isolation: full sealing requires quiescent, read-only roots and an independent
second full verification pass. The live seal was generated once and
independently full-verified a second time over 121 files / 404,751,718,887
bytes; both passes produced SHA256
`d3bf1f18046a695b9c6089f2866cdb854042c3b528434018ecab8f56d36af665`.
Managed execution checks exact inventory and sizes before creating the run
directory, snapshots the same ledger bytes under `configs/` and `manifests/`,
and launches the delegate from the immutable config snapshot. Dry-run only
reports the expected SHA and required roots; aggregate-only jobs bypass the
data gate.

The seal covers derived cache/split bytes, not raw WFDB inputs or preprocessing
correctness. Runtime quick verification intentionally does not rehash 405 GB.

## K500 comparison handoff

`k500_handoff.yaml` is the machine-readable contract for giving another method
implementation access to the same source data, target-center K500 partitions,
internal K400/K100 tuning partitions, and ref-excluded evaluation population.
It records the required tracked YAML/config SHA256 values plus each logical
center's exact `hash_id_set_sha256`.

The base seed is provenance for how the split was created; it is not sufficient
by itself to identify the selected records. Consumers must use the supplied
read-only split artifact directory and verify the source-manifest,
split-manifest, and hash-set identities exposed by
`loader.dataset.selection.describe()`. They must not re-sample K500 at runtime.

The minimum tracked config bundle is:

- `data/PTBXL.yaml`
- `data/PN2021.yaml`
- `data/PN2021_super5_v7.yaml`
- `data/splits.yaml`
- `data/data_load.yaml`
- `random_seed.yaml`
- `eval/PN2021.yaml`
- `augmentation/cache.yaml`
- `augmentation/operators.yaml`

The public data surface consists of three finite plans:

- `PTBXLLoaderPlan.open_train/open_validation/open_test` owns the official
  folds, training-only shuffle, and complete fold-10 holdback policy.
- `PN2021K500LoaderPlan.open_ordered/open_training` accepts exactly one of the
  four logical centers, verifies the exact 500-record selection, and forbids
  `drop_last`.
- `PN2021EvaluationLoaderPlan.open_clean/open_corrupted/open_session` accepts
  only canonical ref-excluded clean/corrupted aliases and sequential views.

The managed train/evaluation adapters derive these plan objects from the
tracked YAML closure. There is no public free-form loader entrypoint and YAML
cannot choose a runtime callable.

The returned signal is float32 raw mV with shape `(1000, 12)` and the label
order is `CD,HYP,MI,NORM,STTC`. Only K500-derived partitions may be used for
target-center adaptation or tuning. Ref-excluded evaluation partitions are
read-only and cannot be used for model adaptation or checkpoint selection.

Both PTB-XL source backbones receive this same canonical raw 100 Hz
time-channel batch. EfficientNet keeps the 1000-point grid; ECGFounder applies
device-local linear `1000 -> 5000` interpolation with `align_corners=True`.
Per-sample global z-score is applied after any interpolation, then the batch is
converted to channel-time for the model.

Runtime caches, split arrays, local paths and model handles remain outside Git.
Only the portable contract and SHA256 identities are tracked here.
