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

`data_load.yaml` defines the shared runtime defaults consumed by
`data_preprocess/data_runtime.py`: batch size, worker/pinned-memory settings,
mmap policy, deterministic shuffle partitions, and the model-input transform.
Explicit `get_dataloader(...)` arguments take precedence over these defaults.
