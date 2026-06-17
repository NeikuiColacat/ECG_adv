# ECGFounder Raw1000 Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit ECGFounder fullFT `raw1000` supervised input mode so the clean supervised stream, VAE decoded adversarial stream, raw-AugMix corruption stream, and PN2021-C evaluator can all enter ECGFounder through the same 100 Hz stabilizer path.

**Architecture:** Preserve the current `cached5000` path as the default. Add a small dataset-aware stream-loader helper, then let `scripts/paper/run_ecgfounder_fullft_super5_pilot_20260523.py` select either cached ECGFounder `12 x 5000` supervised tensors or raw/minimal-resample `12 x 1000` supervised tensors. The first experiment is CPSC-only; expand only if the formal PN2021-C five-operator CPSC AUPRC beats the current matched-stabilizer reference `0.6405`.

**Tech Stack:** Python, PyTorch, numpy mmap caches, existing ECGFounder fullFT runner, existing PN2021-C ECGFounder evaluator, pytest CPU plumbing tests.

---

## File Structure

- Modify `ecg_adv_gen/training/signal_streams.py`: add a generic tagged dataset wrapper and a dataset-aware weighted stream-loader builder; keep `build_weighted_signal_stream_loader(...)` backward-compatible.
- Modify `ecg_adv_gen/training/__init__.py`: export the new helper and wrapper.
- Modify `scripts/paper/run_ecgfounder_fullft_super5_pilot_20260523.py`: add `--supervised_input_mode {cached5000,raw1000}`, default cache paths for PTB-XL raw1000, supervised input dispatch, and raw1000 train-loader construction.
- Modify `util/tests/test_signal_streams.py`: cover the dataset-aware stream-loader helper.
- Modify `util/tests/test_ecgfounder_fullft_raw_augmix.py`: cover parser defaults, supervised input dispatch, raw1000 train-loader dataset selection, and run-record config exposure.
- Update `docs/reports/archive/20260616/pn2021c_augmix_gap_diagnostic_20260616.md`: append the implemented command and result once the CPSC smoke/eval completes.

---

### Task 1: Add Dataset-Aware Weighted Signal Loader

**Files:**
- Modify: `ecg_adv_gen/training/signal_streams.py`
- Modify: `ecg_adv_gen/training/__init__.py`
- Test: `util/tests/test_signal_streams.py`

- [ ] **Step 1: Write the failing test**

Add this import in `util/tests/test_signal_streams.py`:

```python
from torch.utils.data import TensorDataset
```

Extend the `from ecg_adv_gen.training import (...)` block with:

```python
    TaggedSignalDataset,
    build_weighted_signal_stream_loader_from_datasets,
```

Add this test after `test_build_weighted_signal_stream_loader_preserves_weights_and_four_tensors`:

```python
def test_build_weighted_signal_stream_loader_from_datasets_accepts_custom_source_target():
    source_ds = TensorDataset(
        torch.ones((2, 12, 1000), dtype=torch.float32),
        torch.zeros((2, 5), dtype=torch.float32),
    )
    target_ds = TensorDataset(
        torch.full((1, 12, 1000), 2.0, dtype=torch.float32),
        torch.ones((1, 5), dtype=torch.float32),
    )
    adv_x = np.full((1, 12, 1000), 3.0, dtype=np.float32)
    adv_y = np.ones((1, 5), dtype=np.float32)

    loader = build_weighted_signal_stream_loader_from_datasets(
        source_dataset=source_ds,
        target_dataset=target_ds,
        source_weight=1.0,
        target_real_weight=2.0,
        adv_weight=3.0,
        batch_size=4,
        num_workers=0,
        num_classes=5,
        adv_signals=adv_x,
        adv_labels=adv_y,
        pin_memory=False,
    )

    assert len(loader.dataset) == 4
    assert [float(w) for w in loader.sampler.weights] == pytest.approx([1.0, 1.0, 2.0, 3.0])
    x, y, stream, teacher = next(iter(loader))
    assert x.shape[1:] == torch.Size([12, 1000])
    assert y.shape[1:] == torch.Size([5])
    assert teacher.shape[1:] == torch.Size([5])
    assert set(stream.tolist()).issubset({0, 1, 2})
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest util/tests/test_signal_streams.py::test_build_weighted_signal_stream_loader_from_datasets_accepts_custom_source_target -q
```

Expected: fail with an import error for `TaggedSignalDataset` or `build_weighted_signal_stream_loader_from_datasets`.

- [ ] **Step 3: Implement the generic wrapper and loader**

In `ecg_adv_gen/training/signal_streams.py`, add this class after `TaggedCachedSignalDataset`:

```python
class TaggedSignalDataset(Dataset):
    """Tag any two-tensor ECG dataset with a stream id and zero teacher logits."""

    def __init__(self, base: Dataset, stream_id: int, num_classes: int) -> None:
        self.base = base
        self.stream_id = int(stream_id)
        self.teacher_logits = torch.zeros((int(num_classes),), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = self.base[idx]
        return (
            torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32),
            torch.tensor(self.stream_id, dtype=torch.long),
            self.teacher_logits.clone(),
        )
```

In the same file, add this function above `build_weighted_signal_stream_loader(...)`:

```python
def build_weighted_signal_stream_loader_from_datasets(
    *,
    source_dataset: Dataset,
    target_dataset: Dataset,
    source_weight: float,
    target_real_weight: float,
    adv_weight: float,
    batch_size: int,
    num_workers: int,
    num_classes: int,
    adv_signals: np.ndarray | None = None,
    adv_labels: np.ndarray | None = None,
    adv_teacher_logits: np.ndarray | None = None,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """Build source/target/adversarial ECG stream loader from prepared datasets."""
    source_ds = TaggedSignalDataset(source_dataset, stream_id=0, num_classes=num_classes)
    target_ds = TaggedSignalDataset(target_dataset, stream_id=1, num_classes=num_classes)

    datasets: list[Dataset] = []
    weights: list[float] = []
    if float(source_weight) > 0.0:
        datasets.append(source_ds)
        weights.extend([float(source_weight)] * len(source_ds))
    if float(target_real_weight) > 0.0:
        datasets.append(target_ds)
        weights.extend([float(target_real_weight)] * len(target_ds))
    if (
        adv_signals is not None
        and adv_labels is not None
        and len(adv_signals) > 0
        and float(adv_weight) > 0.0
    ):
        adv_ds = TaggedMemorySignalDataset(
            adv_signals,
            adv_labels,
            stream_id=2,
            teacher_logits=adv_teacher_logits,
        )
        datasets.append(adv_ds)
        weights.extend([float(adv_weight)] * len(adv_ds))
    if not datasets:
        raise RuntimeError("no active training streams; check source/target/adv weights")

    combined = ConcatDataset(datasets)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return DataLoader(
        combined,
        batch_size=int(batch_size),
        sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=int(num_workers) > 0,
        drop_last=bool(drop_last),
    )
```

Refactor the body of `build_weighted_signal_stream_loader(...)` to construct cached datasets and call the new function:

```python
    source_ds = CachedSignalDataset(source_signals, source_labels, source_indices)
    target_ds = CachedSignalDataset(target_signals, target_labels, target_indices)
    return build_weighted_signal_stream_loader_from_datasets(
        source_dataset=source_ds,
        target_dataset=target_ds,
        source_weight=source_weight,
        target_real_weight=target_real_weight,
        adv_weight=adv_weight,
        batch_size=batch_size,
        num_workers=num_workers,
        num_classes=num_classes,
        adv_signals=adv_signals,
        adv_labels=adv_labels,
        adv_teacher_logits=adv_teacher_logits,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
```

Add these names to `__all__`:

```python
    "TaggedSignalDataset",
    "build_weighted_signal_stream_loader_from_datasets",
```

In `ecg_adv_gen/training/__init__.py`, import and export:

```python
    TaggedSignalDataset,
    build_weighted_signal_stream_loader_from_datasets,
```

- [ ] **Step 4: Run signal stream tests**

Run:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest util/tests/test_signal_streams.py -q
```

Expected: all tests pass.

### Task 2: Add ECGFounder Supervised Input Dispatch

**Files:**
- Modify: `scripts/paper/run_ecgfounder_fullft_super5_pilot_20260523.py`
- Test: `util/tests/test_ecgfounder_fullft_raw_augmix.py`

- [ ] **Step 1: Write parser and dispatch tests**

Add these tests after `test_input_stabilizer_kwargs_from_args_omits_disabled_values`:

```python
def test_fullft_parser_defaults_to_cached5000_supervised_input_mode():
    parser = fullft.build_arg_parser()
    args = parser.parse_args(["--ref_meta_json", "/tmp/ref.json"])
    assert args.supervised_input_mode == "cached5000"


def test_fullft_parser_accepts_raw1000_supervised_input_mode():
    parser = fullft.build_arg_parser()
    args = parser.parse_args(
        [
            "--ref_meta_json",
            "/tmp/ref.json",
            "--supervised_input_mode",
            "raw1000",
            "--source_raw1000_signal_cache",
            "/tmp/source.npy",
            "--source_raw1000_label_cache",
            "/tmp/labels.npy",
        ]
    )
    assert args.supervised_input_mode == "raw1000"
    assert args.source_raw1000_signal_cache == "/tmp/source.npy"
    assert args.source_raw1000_label_cache == "/tmp/labels.npy"


def test_prepare_supervised_ecgfounder_input_dispatches_by_mode(monkeypatch):
    raw = torch.zeros((2, 12, 1000), dtype=torch.float32)
    cached = torch.zeros((2, 12, 5000), dtype=torch.float32)
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_raw(x, kwargs):
        calls.append(("raw", dict(kwargs)))
        return x + 1.0

    def fake_cached(x, kwargs):
        calls.append(("cached", dict(kwargs)))
        return x + 2.0

    monkeypatch.setattr(fullft, "prepare_raw_ecgfounder_input", fake_raw)
    monkeypatch.setattr(fullft, "prepare_cached_ecgfounder_input", fake_cached)

    stabilizer = {"bandpass_low_hz": 0.5, "bandpass_high_hz": 35.0}
    raw_out = fullft.prepare_supervised_ecgfounder_input(raw, "raw1000", stabilizer)
    cached_out = fullft.prepare_supervised_ecgfounder_input(cached, "cached5000", stabilizer)

    torch.testing.assert_close(raw_out, raw + 1.0)
    torch.testing.assert_close(cached_out, cached + 2.0)
    assert calls == [("raw", stabilizer), ("cached", stabilizer)]
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest \
  util/tests/test_ecgfounder_fullft_raw_augmix.py::test_fullft_parser_defaults_to_cached5000_supervised_input_mode \
  util/tests/test_ecgfounder_fullft_raw_augmix.py::test_fullft_parser_accepts_raw1000_supervised_input_mode \
  util/tests/test_ecgfounder_fullft_raw_augmix.py::test_prepare_supervised_ecgfounder_input_dispatches_by_mode \
  -q
```

Expected: fail because the CLI flags and helper do not exist.

- [ ] **Step 3: Implement parser flags and dispatch helper**

In `scripts/paper/run_ecgfounder_fullft_super5_pilot_20260523.py`, add these constants near `DEFAULT_OUT_DIR`:

```python
SUPERVISED_INPUT_MODES = ("cached5000", "raw1000")
DEFAULT_SOURCE_RAW1000_SIGNAL_CACHE = (
    DATA_ROOT / "triple_labels/cache/ptbxl_minimal_resample_per_sample_global_fs100_len1000.npy"
)
DEFAULT_SOURCE_RAW1000_LABEL_CACHE = (
    DATA_ROOT / "triple_labels/super5_minresample_full10_perglobal_20260503/ptbxl_labels.C5.all.npy"
)
```

Add this helper after `prepare_cached_ecgfounder_input(...)`:

```python
def prepare_supervised_ecgfounder_input(
    x: torch.Tensor,
    supervised_input_mode: str,
    input_stabilizer_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    mode = str(supervised_input_mode)
    if mode == "raw1000":
        return prepare_raw_ecgfounder_input(x, input_stabilizer_kwargs)
    if mode == "cached5000":
        return prepare_cached_ecgfounder_input(x, input_stabilizer_kwargs)
    raise ValueError(f"unknown supervised_input_mode: {mode!r}")
```

Add parser options after the input stabilizer flags:

```python
    ap.add_argument(
        "--supervised_input_mode",
        choices=SUPERVISED_INPUT_MODES,
        default="cached5000",
        help=(
            "cached5000 keeps the historical ECGFounder 12x5000 supervised stream; "
            "raw1000 feeds source/target supervised ECGs through the same 100Hz "
            "ecg1000_to_ecgfounder_input path used by raw-AugMix and PN2021-C eval."
        ),
    )
    ap.add_argument("--source_raw1000_signal_cache", default=str(DEFAULT_SOURCE_RAW1000_SIGNAL_CACHE))
    ap.add_argument("--source_raw1000_label_cache", default=str(DEFAULT_SOURCE_RAW1000_LABEL_CACHE))
```

In the training loop, replace:

```python
                logits = model(prepare_cached_ecgfounder_input(x, input_stabilizer_kwargs))
```

with:

```python
                logits = model(
                    prepare_supervised_ecgfounder_input(
                        x,
                        args.supervised_input_mode,
                        input_stabilizer_kwargs,
                    )
                )
```

Add `supervised_input_mode` to the top-level result payload:

```python
        "supervised_input_mode": str(args.supervised_input_mode),
```

- [ ] **Step 4: Run the targeted tests**

Run:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest \
  util/tests/test_ecgfounder_fullft_raw_augmix.py::test_fullft_parser_defaults_to_cached5000_supervised_input_mode \
  util/tests/test_ecgfounder_fullft_raw_augmix.py::test_fullft_parser_accepts_raw1000_supervised_input_mode \
  util/tests/test_ecgfounder_fullft_raw_augmix.py::test_prepare_supervised_ecgfounder_input_dispatches_by_mode \
  -q
```

Expected: all pass.

### Task 3: Wire Raw1000 Source/Target Supervised Datasets

**Files:**
- Modify: `scripts/paper/run_ecgfounder_fullft_super5_pilot_20260523.py`
- Test: `util/tests/test_ecgfounder_fullft_raw_augmix.py`

- [ ] **Step 1: Write loader tests**

Add this test after `test_prepare_supervised_ecgfounder_input_dispatches_by_mode`:

```python
def test_make_train_loader_raw1000_uses_source_cache_and_target_anchor(monkeypatch, tmp_path):
    source_signals = np.zeros((3, 1000, 12), dtype=np.float32)
    source_labels = np.eye(5, dtype=np.float32)[:3]
    source_signal_path = tmp_path / "ptbxl_raw1000.npy"
    source_label_path = tmp_path / "ptbxl_labels.npy"
    np.save(source_signal_path, source_signals)
    np.save(source_label_path, source_labels)

    target_signal_path = tmp_path / "cpsc_2018_real_k500_seed20260531.signals.npz"
    np.savez_compressed(
        target_signal_path,
        signals=np.ones((2, 1000, 12), dtype=np.float32),
        labels=np.ones((2, 5), dtype=np.float32),
        record_ids=np.asarray(["r0", "r1"], dtype=str),
    )

    def fake_anchor_signal_npz_path(center, args):
        assert center == "cpsc_2018"
        return target_signal_path

    captured = {}

    def fake_build_loader(**kwargs):
        captured.update(kwargs)
        return "loader"

    monkeypatch.setattr(fullft, "anchor_signal_npz_path", fake_anchor_signal_npz_path)
    monkeypatch.setattr(fullft, "build_weighted_signal_stream_loader_from_datasets", fake_build_loader)

    args = fullft.build_arg_parser().parse_args(
        [
            "--ref_meta_json",
            "/tmp/ref.json",
            "--center",
            "cpsc_2018",
            "--supervised_input_mode",
            "raw1000",
            "--source_raw1000_signal_cache",
            str(source_signal_path),
            "--source_raw1000_label_cache",
            str(source_label_path),
            "--batch_size",
            "4",
            "--num_workers",
            "0",
        ]
    )
    ptbxl_payload = {
        "signals": np.zeros((3, 12, 5000), dtype=np.float32),
        "labels": source_labels,
        "folds": np.asarray([1, 2, 9], dtype=np.int64),
    }
    pn_payload = {
        "signals": np.zeros((2, 12, 5000), dtype=np.float32),
        "labels": np.ones((2, 5), dtype=np.float32),
    }

    loader = fullft.make_train_loader(
        ptbxl_payload,
        pn_payload,
        np.asarray([0], dtype=np.int64),
        args,
        target_train_record_ids={"r1"},
    )

    assert loader == "loader"
    assert len(captured["source_dataset"]) == 2
    assert len(captured["target_dataset"]) == 1
    src_x, _ = captured["source_dataset"][0]
    tgt_x, _ = captured["target_dataset"][0]
    assert tuple(src_x.shape) == (12, 1000)
    assert tuple(tgt_x.shape) == (12, 1000)
```

- [ ] **Step 2: Run the failing loader test**

Run:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest \
  util/tests/test_ecgfounder_fullft_raw_augmix.py::test_make_train_loader_raw1000_uses_source_cache_and_target_anchor \
  -q
```

Expected: fail because `make_train_loader(...)` does not accept `target_train_record_ids` and does not call the dataset-aware loader.

- [ ] **Step 3: Import the new stream helper**

In `scripts/paper/run_ecgfounder_fullft_super5_pilot_20260523.py`, extend the `ecg_adv_gen.training` import with:

```python
    CachedSignalDataset,
    build_weighted_signal_stream_loader_from_datasets,
```

- [ ] **Step 4: Add source raw1000 loader helper**

Add this helper after `load_ptbxl_partner_pool(...)`:

```python
def load_source_raw1000_dataset(args: argparse.Namespace, source_indices: np.ndarray, expected_labels: np.ndarray) -> RawSignalDataset:
    signal_path = Path(args.source_raw1000_signal_cache)
    label_path = Path(args.source_raw1000_label_cache)
    if not signal_path.exists():
        raise FileNotFoundError(signal_path)
    if not label_path.exists():
        raise FileNotFoundError(label_path)
    signals = np.load(signal_path, mmap_mode="r")
    labels = np.load(label_path, mmap_mode="r").astype(np.float32, copy=False)
    if labels.shape[0] != signals.shape[0]:
        raise ValueError(f"source raw1000 signals/labels length mismatch: {signals.shape} vs {labels.shape}")
    source_indices = np.asarray(source_indices, dtype=np.int64)
    if expected_labels.shape[0] != source_indices.shape[0]:
        raise ValueError(
            f"expected source label rows mismatch: {expected_labels.shape[0]} vs {source_indices.shape[0]}"
        )
    if not np.allclose(labels[source_indices], expected_labels.astype(np.float32), atol=1e-6):
        raise ValueError("source raw1000 labels do not align with ECGFounder supervised cache labels")
    return RawSignalDataset(signals, labels, indices=source_indices)
```

- [ ] **Step 5: Modify `make_train_loader(...)` for raw1000 mode**

Change the signature:

```python
def make_train_loader(
    ptbxl_payload: dict[str, np.ndarray],
    pn_payload: dict[str, np.ndarray],
    target_indices: np.ndarray,
    args: argparse.Namespace,
    adv_signals: np.ndarray | None = None,
    adv_labels: np.ndarray | None = None,
    adv_teacher_logits: np.ndarray | None = None,
    adv_weight: float | None = None,
    target_train_record_ids: set[str] | None = None,
) -> DataLoader:
```

Replace the return statement with:

```python
    if str(args.supervised_input_mode) == "raw1000":
        if target_train_record_ids is None:
            raise ValueError("raw1000 supervised input mode requires target_train_record_ids")
        source_ds = load_source_raw1000_dataset(
            args,
            source_idx,
            ptbxl_payload["labels"][source_idx],
        )
        target_ds = load_target_raw_dataset(
            args.center,
            args,
            sorted(str(x) for x in target_train_record_ids),
        )
        return build_weighted_signal_stream_loader_from_datasets(
            source_dataset=source_ds,
            target_dataset=target_ds,
            source_weight=float(args.source_weight),
            target_real_weight=float(args.target_real_weight),
            adv_weight=effective_adv_weight,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_classes=len(CLASS_NAMES_SUPER5),
            adv_signals=adv_signals,
            adv_labels=adv_labels,
            adv_teacher_logits=adv_teacher_logits,
        )

    return build_weighted_signal_stream_loader(
        source_signals=ptbxl_payload["signals"],
        source_labels=ptbxl_payload["labels"],
        source_indices=source_idx,
        target_signals=pn_payload["signals"],
        target_labels=pn_payload["labels"],
        target_indices=target_indices,
        source_weight=float(args.source_weight),
        target_real_weight=float(args.target_real_weight),
        adv_weight=effective_adv_weight,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_classes=len(CLASS_NAMES_SUPER5),
        adv_signals=adv_signals,
        adv_labels=adv_labels,
        adv_teacher_logits=adv_teacher_logits,
    )
```

In the epoch loop, pass the selected K500 training IDs:

```python
            target_train_record_ids=target_train_ids,
```

- [ ] **Step 6: Run the loader test**

Run:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest \
  util/tests/test_ecgfounder_fullft_raw_augmix.py::test_make_train_loader_raw1000_uses_source_cache_and_target_anchor \
  -q
```

Expected: pass.

### Task 4: Verify Existing Paths Still Pass

**Files:**
- No new files beyond Tasks 1-3.

- [ ] **Step 1: Run focused CPU tests**

Run:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest \
  util/tests/test_signal_streams.py \
  util/tests/test_ecgfounder_fullft_raw_augmix.py \
  util/tests/test_ecgfounder_pn2021c_evaluator.py \
  -q
```

Expected: all tests pass.

- [ ] **Step 2: Run whitespace check**

Run:

```bash
git diff --check
```

Expected: no output.

### Task 5: Run CPSC Raw1000 Parity Smoke

**Files:**
- Writes run artifacts under `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_teacherdrift_raw1000_cpsc_20260616`.

- [ ] **Step 1: Check GPU state**

Run:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits
```

Expected: choose one free GPU. Do not use GPUs already occupied by other users.

- [ ] **Step 2: Launch the CPSC training smoke on one free GPU**

If GPU 0 is free, run:

```bash
export ECG_ADV_GEN_DATA_ROOT=/home/linbinhao/ECG_adv_data
export ECGFOUNDER_ROOT=/home/linbinhao/ECG_adv_data/ecgfounder
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
/home/linbinhao/micromamba/envs/ECGTwin/bin/python -u scripts/paper/run_ecgfounder_fullft_super5_pilot_20260523.py \
  --out_dir /home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_teacherdrift_raw1000_cpsc_20260616 \
  --run_name cpsc_fullft_vae_aw20_rawaugmix_noisemask_teacherdrift_raw1000_stabilizer35_ep10_seed20260531 \
  --center cpsc_2018 \
  --ref_meta_json /home/linbinhao/ECG_adv_data/paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets/cpsc_2018/k500_seed20260531/cpsc_2018_real_k500_seed20260531.ref_meta.json \
  --anchor_base_root /home/linbinhao/ECG_adv_data/paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets \
  --cache_dir /home/linbinhao/ECG_adv_data/runs/ecgfounder_inithead_fullft_k500_v6/main_repro_20260528_1355_ecgfounder_inithead/cache \
  --k 500 \
  --epochs 10 \
  --batch_size 128 \
  --eval_batch_size 256 \
  --num_workers 0 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --source_weight 1.0 \
  --target_real_weight 120.0 \
  --target_val_count 100 \
  --target_val_seed 20260531 \
  --target_val_split_mode stratified \
  --selection_metric source_plus_target_val_auprc \
  --target_val_score_weight 0.5 \
  --init_head_path /home/linbinhao/ECG_adv_data/runs/ecgfounder_direct_k500_v7_sjr_rgq/mainline_v7_k500_4gpu_20260605/cpsc_2018/runs/cpsc_2018_k500_seed20260531/best_head.pt \
  --enable_vae_adv_stream \
  --adv_weight 20.0 \
  --k_anchor 300 \
  --anchor_sample_mode hard_bce \
  --hull_m 20 \
  --hull_lambda 0.15 \
  --hull_steps 5 \
  --hull_lr 0.25 \
  --hull_include_anchor \
  --pgd_eps 2.0 \
  --pgd_batch 4 \
  --enable_raw_corrupt_consistency \
  --raw_corrupt_scope source_target \
  --raw_corrupt_batch_size 128 \
  --raw_corrupt_copies 2 \
  --raw_corrupt_prob 1.0 \
  --raw_corrupt_severity 5 \
  --raw_corrupt_severity_profile calibrated_10to20pp \
  --raw_corrupt_ops powerline_noise emg_noise random_leads_masking \
  --raw_corrupt_view_mode augmix \
  --raw_corrupt_augmix_width 1 \
  --raw_corrupt_augmix_depth 1 \
  --raw_corrupt_augmix_mixture_mode fixed \
  --raw_corrupt_augmix_mixture_prob 1.0 \
  --raw_corrupt_no_renorm \
  --raw_corrupt_consistency_weight 2.0 \
  --raw_corrupt_consistency_loss jsd \
  --raw_corrupt_bce_weight 1.0 \
  --raw_corrupt_max_batches 64 \
  --enable_raw_corrupt_aux_consistency \
  --raw_corrupt_aux_ops baseline_wander baseline_shift \
  --raw_corrupt_aux_consistency_weight 0.0 \
  --raw_corrupt_aux_bce_weight 0.0 \
  --raw_corrupt_aux_teacher_weight 1.0 \
  --raw_corrupt_aux_teacher_loss mse_logits \
  --raw_corrupt_aux_teacher_view corrupt \
  --raw_corrupt_aux_teacher_model_path /home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_pn2021c_calibrated_10to20pp_20260616/vae_inithead_aw20/cpsc_2018/best_model.pt \
  --ecgfounder_input_bandpass_low_hz 0.5 \
  --ecgfounder_input_bandpass_high_hz 35.0 \
  --ecgfounder_input_repair_flat_leads \
  --supervised_input_mode raw1000 \
  --source_raw1000_signal_cache /home/linbinhao/ECG_adv_data/triple_labels/cache/ptbxl_minimal_resample_per_sample_global_fs100_len1000.npy \
  --source_raw1000_label_cache /home/linbinhao/ECG_adv_data/triple_labels/super5_minresample_full10_perglobal_20260503/ptbxl_labels.C5.all.npy \
  --device cuda \
  --seed 20260531 2>&1 | tee /home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_teacherdrift_raw1000_cpsc_20260616/train.log
```

Expected: run directory contains `best_model.pt` and `eval_result.json`; `eval_result.json` contains `"supervised_input_mode": "raw1000"`.

### Task 6: Run Formal PN2021-C CPSC Evaluation

**Files:**
- Writes JSON under `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_teacherdrift_raw1000_cpsc_pn2021c_20260616/cpsc_2018`.

- [ ] **Step 1: Launch formal five-operator PN2021-C eval**

Run on the same free GPU:

```bash
export ECG_ADV_GEN_DATA_ROOT=/home/linbinhao/ECG_adv_data
export ECGFOUNDER_ROOT=/home/linbinhao/ECG_adv_data/ecgfounder
export CUDA_VISIBLE_DEVICES=0
/home/linbinhao/micromamba/envs/ECGTwin/bin/python scripts/triple_labels/eval_ecgfounder_pn2021_corruptions.py \
  --run_dir /home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_teacherdrift_raw1000_cpsc_20260616/runs/cpsc_fullft_vae_aw20_rawaugmix_noisemask_teacherdrift_raw1000_stabilizer35_ep10_seed20260531 \
  --centers cpsc_2018 \
  --corruptions powerline_noise emg_noise baseline_wander baseline_shift random_leads_masking \
  --severities 5 \
  --severity_profile calibrated_10to20pp \
  --batch_size 128 \
  --num_workers 0 \
  --device cuda \
  --ecgfounder_input_bandpass_low_hz 0.5 \
  --ecgfounder_input_bandpass_high_hz 35.0 \
  --ecgfounder_input_repair_flat_leads \
  --recompute_clean_with_input_stabilizer \
  --output_path /home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_teacherdrift_raw1000_cpsc_pn2021c_20260616/cpsc_2018/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json
```

Expected: JSON contains clean and five severity-5 corruption rows.

- [ ] **Step 2: Compare against the CPSC matched-stabilizer threshold**

Run:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python - <<'PY'
import json
from pathlib import Path
p = Path('/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_teacherdrift_raw1000_cpsc_pn2021c_20260616/cpsc_2018/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json')
d = json.load(open(p))
agg = d['aggregate_by_corruption_severity']['5']
print(json.dumps(agg, indent=2))
print('pass_threshold_auprc_gt_0p6405=', float(agg['macro_auprc']) > 0.6405)
PY
```

Expected: if `pass_threshold_auprc_gt_0p6405=True`, expand to Chapman/Georgia/Ningbo. If false, do not spend four-center GPU time on raw1000 parity.

### Task 7: Document Result and Next Decision

**Files:**
- Modify: `docs/reports/archive/20260616/pn2021c_augmix_gap_diagnostic_20260616.md`

- [ ] **Step 1: Generate the Markdown result table**

Run:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python - <<'PY'
import json
from pathlib import Path
p = Path('/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_teacherdrift_raw1000_cpsc_pn2021c_20260616/cpsc_2018/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json')
d = json.load(open(p))
agg = d['aggregate_by_corruption_severity']['5']
clean = d['clean_metric_overrides']['cpsc_2018']
drop_auroc = 100.0 * (float(clean['macro_auroc']) - float(agg['macro_auroc']))
drop_auprc = 100.0 * (float(clean['macro_auprc']) - float(agg['macro_auprc']))
verdict = 'expand to four centers' if float(agg['macro_auprc']) > 0.6405 else 'stop raw1000 parity'
print('Raw1000 parity CPSC implementation result:')
print()
print('| Variant | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Drop AUROC / AUPRC | Verdict |')
print('|---|---:|---:|---:|---|')
print('| cached5000 matched stabilizer reference | 0.8944 / 0.6992 | 0.8641 / 0.6405 | 3.03 / 5.88 pp | reference |')
print(
    '| raw1000 supervised parity | '
    f"{float(clean['macro_auroc']):.4f} / {float(clean['macro_auprc']):.4f} | "
    f"{float(agg['macro_auroc']):.4f} / {float(agg['macro_auprc']):.4f} | "
    f"{drop_auroc:.2f} / {drop_auprc:.2f} pp | {verdict} |"
)
PY
```

Expected: the command prints a complete Markdown table with no manual metric
substitution required.

- [ ] **Step 2: Add the generated result section**

Append the generated Markdown table under `### ECGFounder 100 Hz Train/Eval
Parity Audit` in
`docs/reports/archive/20260616/pn2021c_augmix_gap_diagnostic_20260616.md`.

- [ ] **Step 3: Run report whitespace check**

Run:

```bash
git diff --check -- docs/reports/archive/20260616/pn2021c_augmix_gap_diagnostic_20260616.md
```

Expected: no output.

## Self-Review

- Spec coverage: The plan covers the raw1000 supervised input mode, default backward compatibility, stream sampling preservation, CPU tests, CPSC-only smoke, formal PN2021-C evaluation, and the decision threshold for four-center expansion.
- Placeholder scan: The plan contains concrete paths, commands, thresholds, and generated-report commands; no metric placeholders are left for manual substitution.
- Type consistency: The new loader accepts PyTorch `Dataset` instances returning `(x, y)`, while the fullFT runner keeps emitting `(x, y, stream, teacher_logits)` after tagging. The supervised input dispatch receives a tensor plus mode string and returns an ECGFounder-ready tensor.
