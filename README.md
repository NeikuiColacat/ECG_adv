# K500 Data and Model Handoff

This branch is a deliberately small integration surface for academic-method
comparisons. It contains only:

- PTB-XL and PN2021 preprocessing, cache, and frozen-split loading;
- the reviewed PN2021-to-Super5 mapping and deterministic random-seed contract;
- the fixed four-center K500/K400/K100/ref-exclusion identities;
- PN2021-C corruption-cache construction;
- EfficientNet1DV2, ECGFounder, and ECGTwin VAE model definitions;
- strict checkpoint loaders, canonical input adapters, YAML contracts, and
  focused tests.

It intentionally does not contain training mainlines, method search code,
experiment outputs, reports, external repositories, datasets, or weights.
Those exclusions keep a comparison method from silently inheriting our method
implementation.

## Locked comparison identity

```text
split_id:       pn2021_super5_k500_cpsc_combined_v1
base_seed:      20260501
seed_namespace: ecg_manual_refactor_split_v1
mapping:        v7_super5_sjr_rgq_review_20260528
mapping_hash:   555ec85d5b51
class_order:    CD,HYP,MI,NORM,STTC
centers:        ningbo, chapman_shaoxing, cpsc_2018, georgia
```

The seed alone does not identify K500. Exact reproducibility also requires the
same candidate population, mapping, candidate ordering, and source-cache
manifest. Use the supplied frozen split artifacts and verify the identities in
`configs/data/k500_handoff.yaml`; do not resample K500 at method runtime.

## External handoff archive

The separately distributed archive is named:

```text
ecg_k500_data_models_handoff_20260731.tar.zst
```

After extraction its root contains:

```text
ecg_k500_data_models_handoff_20260731/
  MANIFEST.yaml
  README.md
  licenses/
  splits/manual_refactor_super5_v2_k500_tune_seed20260501/
  weights/ecgtwin/vae_model.pth
  weights/ecgfounder/12_lead_ECGFounder.pth
  weights/ptbxl_source/efficientnet1dv2_best.pt
  weights/ptbxl_source/ecgfounder_best.pt
```

The archive includes split identities and model weights, but not raw ECG data
or waveform caches. Build those caches from legally obtained PTB-XL/PN2021
data, or receive the existing read-only caches through a separate data channel.

## Data interface

Install Python 3.10+ with `numpy`, `pandas`, `pyarrow`, `PyYAML`, `torch`,
`wfdb`, and `pytest`. From the repository root:

```python
from pathlib import Path

from data_preprocess.data_runtime import get_dataloader

bundle = Path("/path/to/ecg_k500_data_models_handoff_20260731")
loader = get_dataloader(
    dataset="pn2021",
    partition="k500",
    logical_center="ningbo",
    cache_dir="/path/to/read_only/pn2021_100hz_cache",
    split_dir=(
        bundle
        / "splits"
        / "manual_refactor_super5_v2_k500_tune_seed20260501"
        / "pn2021_super5_k500_cpsc_combined_v1"
    ),
    sampling_rate_hz=100,
    prepare_for_model=False,
    sanitize=False,
    global_zscore=False,
    output_layout="time_channel",
    shuffle=False,
)

print(loader.dataset.selection.describe())
batch = next(iter(loader))
```

Canonical raw output is float32 mV with shape `(B, 1000, 12)` and labels with
shape `(B, 5)` in `CD,HYP,MI,NORM,STTC` order. A method may own its input
preprocessing, but it must not change record membership, labels, or split
identity.

For tuning, use only `k500_tune_train` (K400) and
`k500_tune_validation` (K100). The evaluation partitions are read-only and
must not update weights, batch-normalization state, teachers, prototypes,
normalization statistics, or checkpoint selection.

## Model definitions and weights

```python
from pathlib import Path

from models import (
    build_ecgfounder,
    build_ecgtwin_vae,
    build_efficientnet1dv2,
)

bundle = Path("/path/to/ecg_k500_data_models_handoff_20260731")

efficientnet = build_efficientnet1dv2(
    checkpoint_path=(
        bundle / "weights/ptbxl_source/efficientnet1dv2_best.pt"
    )
)
ecgfounder = build_ecgfounder(
    pretrained_checkpoint_path=(
        bundle / "weights/ecgfounder/12_lead_ECGFounder.pth"
    ),
    task_checkpoint_path=(
        bundle / "weights/ptbxl_source/ecgfounder_best.pt"
    ),
    trainable_scope="full",
)
vae_encoder, vae_decoder = build_ecgtwin_vae(
    config_path="configs/train/vae.yaml",
    checkpoint_path=bundle / "weights/ecgtwin/vae_model.pth",
)
```

Use `models.input_adapter.prepare_canonical_model_input` as the single bridge
from raw `(B,1000,12)` mV waveforms to each classifier's declared input grid.
Every classifier must return finite raw logits with shape `(B,5)`.

Exact checkpoint hashes and sizes are recorded in
`configs/baselines/ptbxl_source_v1.yaml` and
`configs/data/k500_handoff.yaml`. Verify the archive manifest before loading
any weight.

## Validation

```bash
python -m pytest util/tests -q
```

The focused suite checks cache and split integrity, deterministic seed streams,
K500 handoff identities, preprocessing behavior, canonical input adapters, and
model construction. No GPU training or dataset build is required for the unit
suite.
