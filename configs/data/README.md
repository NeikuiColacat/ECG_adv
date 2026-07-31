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

The public data entry point is `data_preprocess.data_runtime.get_dataloader`.
For a method-owned preprocessing pipeline, request canonical raw mV data:

```python
from data_preprocess.data_runtime import get_dataloader

loader = get_dataloader(
    dataset="pn2021",
    partition="k500",
    logical_center="ningbo",
    cache_dir="/path/to/read_only/pn2021_cache",
    split_dir="/path/to/read_only/pn2021_super5_k500_cpsc_combined_v1",
    sampling_rate_hz=100,
    prepare_for_model=False,
    sanitize=False,
    global_zscore=False,
    output_layout="time_channel",
    shuffle=False,
)
identity = loader.dataset.selection.describe()
```

The returned signal is float32 raw mV with shape `(1000, 12)` and the label
order is `CD,HYP,MI,NORM,STTC`. Only K500-derived partitions may be used for
target-center adaptation or tuning. Ref-excluded evaluation partitions are
read-only and cannot be used for model adaptation or checkpoint selection.

Runtime caches, split arrays, local paths and model handles remain outside Git.
Only the portable contract and SHA256 identities are tracked here.
