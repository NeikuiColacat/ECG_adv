# Paper Safety And Typed Adapter Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the current ECG_adv_Gen mainline safer for paper use by enforcing ref-excluded PN2021 evaluation, preventing PN2021-C metadata drift, upgrading EffNet VAE-LHAT evidence from legacy backfill toward registered run records, and reducing long argv / large-function launch debt with typed adapters.

**Architecture:** Keep legacy script entrypoints stable, but move paper-critical policy checks and command construction into CPU-testable `ecg_adv_gen/` modules. Add typed adapters for high-risk launch surfaces first, then thin `configs/experiments/*.yaml` around structured fields instead of long hand-written argv. Defer invasive GPU training-loop moves until evaluation, metadata, and evidence contracts are locked.

**Tech Stack:** Python 3.10 in `/home/linbinhao/micromamba/envs/ECGTwin/bin/python`, PyYAML, pytest, existing `ecg_adv_gen` package, managed launcher `scripts/run_experiment.py`, CPU-only smoke tests.

---

## Scope And Constraints

This plan fixes the refactor issues identified by the latest multi-agent scan:

- `eval_crosscenter.py` does not yet force paper-safe K500 ref exclusion.
- PN2021-C evaluation can mix clean/corrupt metadata, mapping versions, or ref-exclusion counts.
- `eval_crosscenter.py` defaults still allow legacy preprocessing behavior without an explicit diagnostic mode.
- K-shot metadata parsing has a legacy `items` shape that can include records from the wrong center.
- EffNet v7 VAE-LHAT mainline evidence is traceable by metrics, but still weaker than a full registered run-record chain.
- Most experiment YAML files still contain long `runner.argv` blocks; only `effnet_vae_lhat` and `ecgfounder_vae_lhat` are currently typed adapters.
- `ecg_adv_gen/config/loader.py::audit_runner_commands` and `scripts/pgd_cross_center/synth_online_at_super5.py::main` are oversized coordination functions.

Shared-server constraints:

- Do not run GPU training, long inference, or cache builds during this refactor unless explicitly requested.
- Do not touch `model/*` handles.
- Do not stage or commit generated runs, checkpoints, datasets, feature caches, or `model/*`.
- Use `/home/linbinhao/micromamba/envs/ECGTwin/bin/python` for tests and smoke commands.
- Before any git staging/commit/push work, use the `artifact-git-guard` skill.

## File Structure

Create:

- `ecg_adv_gen/evaluation/protocols.py`
  Owns paper-safe PN2021 protocol checks: ref-exclusion requirements, center coverage, preprocessing mode assertions, and diagnostic-mode escape hatches.

- `util/tests/test_evaluation_protocols.py`
  CPU-only tests for ref-exclusion, center coverage, preprocessing mode, and diagnostic exceptions.

- `ecg_adv_gen/evaluation/pn2021c_metadata.py`
  Owns clean/corrupt metadata compatibility checks for PN2021-C.

- `util/tests/test_pn2021c_metadata.py`
  CPU-only tests for mapping hash, cache version, preprocess contract, and ref-exclusion count compatibility.

- `ecg_adv_gen/config/adapters/pn2021_eval.py`
  Typed adapter for clean PN2021 ref-excluded evaluation commands.

- `ecg_adv_gen/config/adapters/pn2021c_eval.py`
  Typed adapter for PN2021-C evaluation commands.

- `ecg_adv_gen/config/adapters/prompt_token_online_at.py`
  Typed adapter for the prompt-token online AT minimal surface after eval safety is stable.

- `ecg_adv_gen/config/adapters/direct_finetune.py`
  Typed adapter for direct K500 / percent direct finetune matrix commands.

- `ecg_adv_gen/config/runner_audit.py`
  Dispatcher for command audit helpers moved out of `ecg_adv_gen/config/loader.py`.

- `util/tests/test_pn2021_eval_adapter.py`
  CPU-only tests for typed clean eval adapter output and audit rules.

- `util/tests/test_pn2021c_eval_adapter.py`
  CPU-only tests for typed PN2021-C adapter output and audit rules.

- `util/tests/test_prompt_token_adapter.py`
  CPU-only tests for prompt-token online AT adapter output and required artifact refs.

- `util/tests/test_direct_finetune_adapter.py`
  CPU-only tests for direct finetune typed adapter output.

Modify:

- `scripts/triple_labels/eval_crosscenter.py`
  Add `--eval_protocol`, `--diagnostic_legacy_eval`, `--allow_missing_centers`, and paper-safe protocol checks before writing final JSON.

- `scripts/triple_labels/eval_pn2021_corruptions.py`
  Require clean eval JSON for paper mode, load clean metadata, validate corrupt metadata, and record compatibility output.

- `scripts/triple_labels/build_pn2021_corruptions.py`
  Write metadata fields required by PN2021-C validation.

- `ecg_adv_gen/evaluation/pn2021_metric_views.py`
  Add center coverage helper or call `evaluation.protocols` from script-level code.

- `ecg_adv_gen/evaluation/__init__.py`
  Export the new protocol and PN2021-C metadata helpers.

- `ecg_adv_gen/data/kshot.py`
  Add strict center-filtered selected-record parser while preserving legacy parsing for explicitly requested legacy mode.

- `util/tests/test_pn2021_index_kshot.py`
  Add wrong-center `items` tests.

- `ecg_adv_gen/evidence/backfill.py`
  Upgrade VAE-LHAT backfill output with selection and run-record compatibility fields.

- `ecg_adv_gen/evidence/run_record.py`
  Enforce selection, K500 ref ids, resolved config, command, and metrics-source records for trusted mainline registration.

- `ecg_adv_gen/evidence/registry.py`
  Make trusted mainline claims with only legacy backfill fail or emit a blocking audit issue until a registered run record is present.

- `ecg_adv_gen/evidence/comparison.py`
  Require method records for trusted comparison bundles, using already-supported `method_records`.

- `scripts/agent/backfill_vae_lhat_manifest.py`
  Write upgraded VAE-LHAT lineage fields.

- `scripts/agent/register_run.py`
  Keep `--status auto`, and reject trusted registration when run-card contract is incomplete.

- `configs/schemas/experiment_config.schema.json`
  Add typed field schemas for `eval_protocol`, `pn2021c`, `prompt_token_online_at`, and direct finetune adapter inputs.

- `ecg_adv_gen/config/adapters/registry.py`
  Register `pn2021_eval`, `pn2021c_eval`, `prompt_token_online_at`, and `direct_finetune`.

- `ecg_adv_gen/config/loader.py`
  Replace inlined audit logic with calls into `runner_audit.py` and adapter-specific audit functions.

- `configs/experiments/pn2021_eval_v7_sjr_rgq_refexcluded.yaml`
  Convert from long argv to `runner.adapter: pn2021_eval`.

- `configs/experiments/pn2021c_effnet_v7_augmix_vs_noaug.yaml`
  Convert from long argv to `runner.adapter: pn2021c_eval`.

- `configs/experiments/ecgtwin_prompt_token_online_at_minimal.yaml`
  Convert from long argv to `runner.adapter: prompt_token_online_at` after Tasks 1-4 pass.

- `configs/experiments/effnet_direct_k500_v7_sjr_rgq_matrix.yaml` and direct benchmark matrix configs
  Convert repeated direct finetune argv blocks to `runner.adapter: direct_finetune`.

- `configs/active_scripts.yaml`
  Update source-of-truth entries and notes for the new adapters and paper-safe eval policy.

- `configs/README.md` and `docs/codex-handoffs/current_workspace_handoff.md`
  Document the new paper-safe eval mode, typed adapter migration order, and run-record requirements.

Do not modify in the first pass:

- `scripts/pgd_cross_center/synth_online_at_super5.py::main` training loop body, except for small helper imports needed by prompt-token adapter dry-run tests.
- GPU model training behavior.
- External model handles under `model/*`.

---

### Task 1: Add Paper-Safe PN2021 Evaluation Protocol

**Files:**
- Create: `ecg_adv_gen/evaluation/protocols.py`
- Create: `util/tests/test_evaluation_protocols.py`
- Modify: `scripts/triple_labels/eval_crosscenter.py`
- Modify: `ecg_adv_gen/evaluation/__init__.py`
- Modify: `configs/README.md`

- [ ] **Step 1: Write failing tests for protocol validation**

Create `util/tests/test_evaluation_protocols.py` with these tests:

```python
import pytest

from ecg_adv_gen.evaluation.protocols import (
    PN2021ProtocolError,
    validate_pn2021_eval_protocol,
)


def _per_center(n_excluded_ref=500):
    return {
        "ningbo": {"n_excluded_ref": n_excluded_ref, "effective_n": 10},
        "chapman_shaoxing": {"n_excluded_ref": n_excluded_ref, "effective_n": 10},
        "cpsc_2018": {"n_excluded_ref": n_excluded_ref, "effective_n": 10},
        "cpsc_2018_extra": {"n_excluded_ref": 0, "effective_n": 10},
        "georgia": {"n_excluded_ref": n_excluded_ref, "effective_n": 10},
        "ptb": {"n_excluded_ref": 0, "effective_n": 10},
        "st_petersburg_incart": {"n_excluded_ref": 0, "effective_n": 10},
    }


def test_paper_refexcluded_requires_target_ref_exclusion():
    with pytest.raises(PN2021ProtocolError, match="ningbo"):
        validate_pn2021_eval_protocol(
            eval_protocol="paper_refexcluded",
            per_center=_per_center(n_excluded_ref=0),
            target_centers=["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"],
            eval_centers=[
                "chapman_shaoxing",
                "cpsc_2018",
                "cpsc_2018_extra",
                "georgia",
                "ningbo",
                "ptb",
                "st_petersburg_incart",
            ],
            preprocess_mode="paper",
            crop_len=1000,
        )


def test_paper_refexcluded_requires_all_eval_centers():
    per_center = _per_center()
    per_center.pop("ptb")
    with pytest.raises(PN2021ProtocolError, match="missing eval centers"):
        validate_pn2021_eval_protocol(
            eval_protocol="paper_refexcluded",
            per_center=per_center,
            target_centers=["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"],
            eval_centers=[
                "chapman_shaoxing",
                "cpsc_2018",
                "cpsc_2018_extra",
                "georgia",
                "ningbo",
                "ptb",
                "st_petersburg_incart",
            ],
            preprocess_mode="paper",
            crop_len=1000,
        )


def test_diagnostic_unrefexcluded_allows_zero_ref_exclusion():
    result = validate_pn2021_eval_protocol(
        eval_protocol="diagnostic_unrefexcluded",
        per_center=_per_center(n_excluded_ref=0),
        target_centers=["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"],
        eval_centers=[
            "chapman_shaoxing",
            "cpsc_2018",
            "cpsc_2018_extra",
            "georgia",
            "ningbo",
            "ptb",
            "st_petersburg_incart",
        ],
        preprocess_mode="legacy_ecgfounder_filter",
        crop_len=250,
        diagnostic_legacy_eval=True,
        allow_missing_centers=False,
    )
    assert result["status"] == "diagnostic"
```

- [ ] **Step 2: Run the new tests and confirm failure**

Run:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest -q util/tests/test_evaluation_protocols.py
```

Expected: import failure for `ecg_adv_gen.evaluation.protocols`.

- [ ] **Step 3: Implement `evaluation.protocols`**

Create `ecg_adv_gen/evaluation/protocols.py` with:

```python
"""Paper-safe PN2021 evaluation protocol checks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


PAPER_EVAL_PROTOCOL = "paper_refexcluded"
DIAGNOSTIC_EVAL_PROTOCOL = "diagnostic_unrefexcluded"
ALLOWED_EVAL_PROTOCOLS = (PAPER_EVAL_PROTOCOL, DIAGNOSTIC_EVAL_PROTOCOL)


class PN2021ProtocolError(ValueError):
    """Raised when a PN2021 eval result is not paper-safe."""


def _missing_centers(per_center: Mapping[str, Any], eval_centers: Sequence[str]) -> list[str]:
    return [str(center) for center in eval_centers if str(center) not in per_center]


def validate_pn2021_eval_protocol(
    *,
    eval_protocol: str,
    per_center: Mapping[str, Mapping[str, Any]],
    target_centers: Sequence[str],
    eval_centers: Sequence[str],
    preprocess_mode: str,
    crop_len: int,
    diagnostic_legacy_eval: bool = False,
    allow_missing_centers: bool = False,
    min_target_ref_excluded: int = 1,
) -> dict[str, Any]:
    """Validate that PN2021 metrics satisfy the requested protocol."""

    if eval_protocol not in ALLOWED_EVAL_PROTOCOLS:
        allowed = ", ".join(ALLOWED_EVAL_PROTOCOLS)
        raise PN2021ProtocolError(f"eval_protocol must be one of {allowed}, got {eval_protocol!r}")

    missing = _missing_centers(per_center, eval_centers)
    if missing and not allow_missing_centers:
        raise PN2021ProtocolError(f"missing eval centers: {missing}")

    if eval_protocol == DIAGNOSTIC_EVAL_PROTOCOL:
        if preprocess_mode != "paper" and not diagnostic_legacy_eval:
            raise PN2021ProtocolError(
                "legacy preprocessing requires --diagnostic_legacy_eval in diagnostic_unrefexcluded mode"
            )
        return {
            "status": "diagnostic",
            "eval_protocol": eval_protocol,
            "missing_eval_centers": missing,
            "target_ref_exclusion_required": False,
        }

    if preprocess_mode != "paper":
        raise PN2021ProtocolError(
            f"paper_refexcluded requires preprocess_mode='paper', got {preprocess_mode!r}"
        )
    if int(crop_len) != 1000:
        raise PN2021ProtocolError(f"paper_refexcluded requires crop_len=1000, got {crop_len}")

    ref_failures = []
    for center in target_centers:
        row = per_center.get(str(center))
        if row is None:
            ref_failures.append(f"{center}: no metrics row")
            continue
        if int(row.get("n_excluded_ref", 0)) < int(min_target_ref_excluded):
            ref_failures.append(f"{center}: n_excluded_ref={row.get('n_excluded_ref', 0)}")
    if ref_failures:
        raise PN2021ProtocolError("paper_refexcluded requires target ref exclusion: " + "; ".join(ref_failures))

    return {
        "status": "paper_safe",
        "eval_protocol": eval_protocol,
        "missing_eval_centers": missing,
        "target_ref_exclusion_required": True,
        "min_target_ref_excluded": int(min_target_ref_excluded),
    }
```

- [ ] **Step 4: Export the helper**

Modify `ecg_adv_gen/evaluation/__init__.py`:

```python
from .protocols import (
    ALLOWED_EVAL_PROTOCOLS,
    DIAGNOSTIC_EVAL_PROTOCOL,
    PAPER_EVAL_PROTOCOL,
    PN2021ProtocolError,
    validate_pn2021_eval_protocol,
)
```

Add these names to `__all__` if the file uses `__all__`.

- [ ] **Step 5: Wire `eval_crosscenter.py` arguments**

In `scripts/triple_labels/eval_crosscenter.py`, add parser args:

```python
parser.add_argument(
    "--eval_protocol",
    choices=["paper_refexcluded", "diagnostic_unrefexcluded"],
    default="paper_refexcluded",
)
parser.add_argument("--diagnostic_legacy_eval", action="store_true")
parser.add_argument("--allow_missing_centers", action="store_true")
parser.add_argument("--min_target_ref_excluded", type=int, default=1)
```

Ensure the script maps its current paper preprocessing defaults to:

```python
preprocess_mode = "paper"
```

Only allow legacy ECGFounder-style filtering when both conditions are true:

```python
args.eval_protocol == "diagnostic_unrefexcluded" and args.diagnostic_legacy_eval
```

- [ ] **Step 6: Validate before writing final JSON**

After `pn2021` metrics are assembled and before writing the eval result JSON, call:

```python
from ecg_adv_gen.evaluation import validate_pn2021_eval_protocol

protocol_summary = validate_pn2021_eval_protocol(
    eval_protocol=args.eval_protocol,
    per_center=pn2021_result["per_center"],
    target_centers=PN2021_TARGET_CENTERS_4,
    eval_centers=eval_centers,
    preprocess_mode=preprocess_mode,
    crop_len=args.crop_len,
    diagnostic_legacy_eval=args.diagnostic_legacy_eval,
    allow_missing_centers=args.allow_missing_centers,
    min_target_ref_excluded=args.min_target_ref_excluded,
)
pn2021_result["eval_protocol"] = protocol_summary
```

- [ ] **Step 7: Run focused tests**

Run:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest -q \
  util/tests/test_evaluation_protocols.py \
  util/tests/test_pn2021_metric_views.py \
  util/tests/test_run_record_management.py
```

Expected: all selected tests pass.

---

### Task 2: Enforce PN2021-C Clean/Corrupt Metadata Compatibility

**Files:**
- Create: `ecg_adv_gen/evaluation/pn2021c_metadata.py`
- Create: `util/tests/test_pn2021c_metadata.py`
- Modify: `scripts/triple_labels/eval_pn2021_corruptions.py`
- Modify: `scripts/triple_labels/build_pn2021_corruptions.py`
- Modify: `ecg_adv_gen/evaluation/__init__.py`

- [ ] **Step 1: Write failing metadata compatibility tests**

Create `util/tests/test_pn2021c_metadata.py`:

```python
import pytest

from ecg_adv_gen.evaluation.pn2021c_metadata import (
    PN2021CMetadataError,
    validate_pn2021c_metadata_compatibility,
)


def _clean():
    return {
        "label_mapping": {"version": "v7", "hash": "hash-v7"},
        "preprocess": {"contract_id": "ptbxl_pn2021_super5_ecgtwin_decode_v1"},
        "pn2021": {"per_center": {"ningbo": {"n_excluded_ref": 500}}},
    }


def _corrupt():
    return {
        "label_mapping": {"version": "v7", "hash": "hash-v7"},
        "preprocess": {"contract_id": "ptbxl_pn2021_super5_ecgtwin_decode_v1"},
        "pn2021c": {
            "cache_version": "v7_refexcluded_100hz1000",
            "center": "ningbo",
            "n_excluded_ref": 500,
        },
    }


def test_pn2021c_metadata_accepts_matching_clean_and_corrupt():
    result = validate_pn2021c_metadata_compatibility(
        clean_eval=_clean(),
        corrupt_metadata=_corrupt(),
        center="ningbo",
        required_cache_version="v7_refexcluded_100hz1000",
    )
    assert result["compatible"] is True


def test_pn2021c_metadata_rejects_mapping_hash_drift():
    corrupt = _corrupt()
    corrupt["label_mapping"]["hash"] = "old-hash"
    with pytest.raises(PN2021CMetadataError, match="label_mapping.hash"):
        validate_pn2021c_metadata_compatibility(
            clean_eval=_clean(),
            corrupt_metadata=corrupt,
            center="ningbo",
            required_cache_version="v7_refexcluded_100hz1000",
        )


def test_pn2021c_metadata_rejects_ref_exclusion_drift():
    corrupt = _corrupt()
    corrupt["pn2021c"]["n_excluded_ref"] = 0
    with pytest.raises(PN2021CMetadataError, match="n_excluded_ref"):
        validate_pn2021c_metadata_compatibility(
            clean_eval=_clean(),
            corrupt_metadata=corrupt,
            center="ningbo",
            required_cache_version="v7_refexcluded_100hz1000",
        )
```

- [ ] **Step 2: Run the metadata tests and confirm failure**

Run:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest -q util/tests/test_pn2021c_metadata.py
```

Expected: import failure for `ecg_adv_gen.evaluation.pn2021c_metadata`.

- [ ] **Step 3: Implement metadata checker**

Create `ecg_adv_gen/evaluation/pn2021c_metadata.py`:

```python
"""PN2021-C clean/corrupt metadata compatibility checks."""

from __future__ import annotations

from typing import Any, Mapping


class PN2021CMetadataError(ValueError):
    """Raised when PN2021-C metadata is incompatible with the clean baseline."""


def _get(mapping: Mapping[str, Any], dotted: str) -> Any:
    cur: Any = mapping
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return None
        cur = cur[part]
    return cur


def validate_pn2021c_metadata_compatibility(
    *,
    clean_eval: Mapping[str, Any],
    corrupt_metadata: Mapping[str, Any],
    center: str,
    required_cache_version: str,
) -> dict[str, Any]:
    """Validate that one corrupted cache matches the clean eval baseline."""

    comparisons = [
        ("label_mapping.hash", _get(clean_eval, "label_mapping.hash"), _get(corrupt_metadata, "label_mapping.hash")),
        ("label_mapping.version", _get(clean_eval, "label_mapping.version"), _get(corrupt_metadata, "label_mapping.version")),
        ("preprocess.contract_id", _get(clean_eval, "preprocess.contract_id"), _get(corrupt_metadata, "preprocess.contract_id")),
    ]
    for name, clean_value, corrupt_value in comparisons:
        if clean_value != corrupt_value:
            raise PN2021CMetadataError(f"{name} mismatch: clean={clean_value!r}, corrupt={corrupt_value!r}")

    corrupt_cache_version = _get(corrupt_metadata, "pn2021c.cache_version")
    if corrupt_cache_version != required_cache_version:
        raise PN2021CMetadataError(
            f"pn2021c.cache_version mismatch: expected={required_cache_version!r}, got={corrupt_cache_version!r}"
        )

    clean_ref = _get(clean_eval, f"pn2021.per_center.{center}.n_excluded_ref")
    corrupt_ref = _get(corrupt_metadata, "pn2021c.n_excluded_ref")
    if clean_ref != corrupt_ref:
        raise PN2021CMetadataError(f"n_excluded_ref mismatch for {center}: clean={clean_ref!r}, corrupt={corrupt_ref!r}")

    return {
        "compatible": True,
        "center": str(center),
        "cache_version": str(required_cache_version),
        "n_excluded_ref": int(clean_ref or 0),
    }
```

- [ ] **Step 4: Update PN2021-C build metadata**

In `scripts/triple_labels/build_pn2021_corruptions.py`, ensure every cache NPZ writes `metadata_json` containing:

```python
metadata = {
    "label_mapping": {
        "version": mapping_version,
        "hash": mapping_hash,
    },
    "preprocess": {
        "contract_id": preprocess_contract_id,
        "input_fs": 100,
        "input_len": 1000,
    },
    "pn2021c": {
        "cache_version": cache_version,
        "center": center,
        "corruption": corruption,
        "severity": int(severity),
        "n_excluded_ref": int(n_excluded_ref),
    },
}
np.savez_compressed(..., metadata_json=json.dumps(metadata, sort_keys=True))
```

Use existing local variable names where available; if names differ, keep the payload keys exactly as shown.

- [ ] **Step 5: Require clean eval JSON in paper mode**

In `scripts/triple_labels/eval_pn2021_corruptions.py`, make paper mode fail when `--clean_eval_json` is absent:

```python
if not args.clean_eval_json and not args.diagnostic_without_clean:
    raise SystemExit("--clean_eval_json is required for paper PN2021-C evaluation")
```

Add:

```python
parser.add_argument("--diagnostic_without_clean", action="store_true")
parser.add_argument("--required_cache_version", default="v7_refexcluded_100hz1000")
```

- [ ] **Step 6: Validate every loaded corruption cache**

After loading each corruption NPZ metadata, call:

```python
compatibility = validate_pn2021c_metadata_compatibility(
    clean_eval=clean_eval_payload,
    corrupt_metadata=metadata,
    center=center,
    required_cache_version=args.required_cache_version,
)
```

Record `compatibility` in the output JSON under each center/corruption/severity row.

- [ ] **Step 7: Run focused tests**

Run:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest -q \
  util/tests/test_pn2021c_metadata.py \
  util/tests/test_pn2021_corruptions.py \
  util/tests/test_pn2021_eval_cache.py
```

Expected: all selected tests pass.

---

### Task 3: Make K-Shot Meta Parsing Strict For Paper Runners

**Files:**
- Modify: `ecg_adv_gen/data/kshot.py`
- Modify: `util/tests/test_pn2021_index_kshot.py`
- Modify: paper runners that call `load_selected_record_ids_from_meta`

- [ ] **Step 1: Add failing strict parser test**

Append to `util/tests/test_pn2021_index_kshot.py`:

```python
import json

from ecg_adv_gen.data.kshot import load_selected_record_ids_from_meta


def test_selected_record_ids_items_shape_filters_wrong_center_in_strict_mode(tmp_path):
    path = tmp_path / "multi_center_ref_meta.json"
    path.write_text(
        json.dumps(
            {
                "center": "ningbo",
                "items": [
                    {"center": "ningbo", "record_id": "N1"},
                    {"center": "georgia", "record_id": "G1"},
                ],
            }
        ),
        encoding="utf-8",
    )
    assert load_selected_record_ids_from_meta(path, "ningbo", strict_center=True) == {"N1"}


def test_selected_record_ids_items_shape_can_preserve_legacy_mode(tmp_path):
    path = tmp_path / "multi_center_ref_meta.json"
    path.write_text(
        json.dumps(
            {
                "center": "ningbo",
                "items": [
                    {"center": "ningbo", "record_id": "N1"},
                    {"center": "georgia", "record_id": "G1"},
                ],
            }
        ),
        encoding="utf-8",
    )
    assert load_selected_record_ids_from_meta(path, "ningbo", strict_center=False) == {"N1", "G1"}
```

- [ ] **Step 2: Run the strict parser tests and confirm failure**

Run:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest -q util/tests/test_pn2021_index_kshot.py -k selected_record_ids
```

Expected: `strict_center` is an unexpected keyword or wrong-center record is retained.

- [ ] **Step 3: Implement strict center filtering**

Modify `load_selected_record_ids_from_meta` in `ecg_adv_gen/data/kshot.py`:

```python
def load_selected_record_ids_from_meta(
    path: str | Path,
    center: str,
    *,
    id_keys: Sequence[str] = ("record_ids", "ref_record_ids", "selected_ref_record_ids"),
    strict_center: bool = True,
) -> set[str]:
    ...
    elif isinstance(payload, dict):
        for key in id_keys:
            if key in payload:
                selected.update(str(x) for x in payload[key])
        if not selected and "items" in payload:
            for row in payload["items"]:
                if not isinstance(row, dict):
                    continue
                if strict_center and str(row.get("center", center)) != center:
                    continue
                rid = row.get("record_id") or row.get("record")
                if rid is not None:
                    selected.add(str(rid))
```

Keep the top-level list behavior center-filtered.

- [ ] **Step 4: Update paper runners to pass strict mode explicitly**

For every current paper runner that reads selected K-shot refs, pass:

```python
strict_center=True
```

For historical diagnostic code that depends on old behavior, pass:

```python
strict_center=False
```

The expected paper-safe default is strict center filtering.

- [ ] **Step 5: Run focused tests**

Run:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest -q \
  util/tests/test_pn2021_index_kshot.py \
  util/tests/test_kshot_artifacts.py
```

Expected: all selected tests pass.

---

### Task 4: Upgrade EffNet VAE-LHAT Evidence To Registered Run-Record Chain

**Files:**
- Modify: `ecg_adv_gen/evidence/backfill.py`
- Modify: `ecg_adv_gen/evidence/run_record.py`
- Modify: `ecg_adv_gen/evidence/registry.py`
- Modify: `ecg_adv_gen/evidence/comparison.py`
- Modify: `scripts/agent/backfill_vae_lhat_manifest.py`
- Modify: `scripts/agent/register_run.py`
- Modify: `util/tests/test_agent_operating_layer.py`
- Modify: `util/tests/test_run_record_management.py`
- Modify: `configs/active_evidence_registry.yaml`
- Modify: `docs/pipelines/run_record_management_20260529.md`

- [ ] **Step 1: Add tests for trusted legacy backfill rejection**

In `util/tests/test_agent_operating_layer.py`, add a registry audit test:

```python
def test_trusted_mainline_legacy_backfill_without_registered_run_is_blocking(tmp_path):
    local_config = tmp_path / "local.yaml"
    local_config.write_text(
        "paths:\n"
        f"  output_root: {tmp_path / 'outputs'}\n"
        f"  data_root: {tmp_path / 'data'}\n"
        f"  write_boundary: {tmp_path}\n",
        encoding="utf-8",
    )
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        """
schema_version: 1
claims:
  - claim_id: effnet_v7_vae_lhat_improves_direct_k500
    status: trusted
    traceability: legacy_backfilled_manifest
    manifest: ${paths.output_root}/missing/run_manifest.backfilled.json
""",
        encoding="utf-8",
    )
    report = audit_active_evidence_registry(
        repo_root=tmp_path,
        registry_path=registry,
        local_config_path=local_config,
        require_existing_artifacts=False,
        check_git=False,
    )
    issue_codes = {issue["code"] for issue in report["issues"]}
    assert "trusted_legacy_backfill_missing_registered_run" in issue_codes
```

- [ ] **Step 2: Add tests for upgraded backfill fields**

In `util/tests/test_run_record_management.py`, add:

```python
def test_finalize_rejects_backfilled_vae_lhat_manifest_without_selection_json(tmp_path):
    run_dir = _make_minimal_run_dir(tmp_path)
    (run_dir / "selection.json").unlink()
    with pytest.raises(RunRecordError, match="selection.json is required"):
        finalize_run_record(
            run_dir,
            purpose="EffNet v7 VAE-LHAT K500 traceability check.",
            result_summary="Backfilled VAE-LHAT run should fail without explicit selection evidence.",
            outcome="succeeded",
        )


def test_finalize_accepts_backfilled_vae_lhat_selection_safety(tmp_path):
    run_dir = _make_minimal_run_dir(tmp_path)
    (run_dir / "selection.json").write_text(
        json.dumps(
            {
                "policy": "target_real_val_internal",
                "selection_source": "target_real_val",
                "heldout_target_labels_used_for_selection": False,
                "full_target_distribution_used_for_selection": False,
            }
        ),
        encoding="utf-8",
    )
    card = finalize_run_record(
        run_dir,
        purpose="EffNet v7 VAE-LHAT K500 traceability check.",
        result_summary="Backfilled VAE-LHAT run records target-real validation selection without heldout target labels.",
        outcome="succeeded",
    )
    assert card["result"]["outcome"] == "succeeded"
```

- [ ] **Step 3: Upgrade backfill output**

Modify `ecg_adv_gen/evidence/backfill.py` so backfilled VAE-LHAT manifests include:

```python
"selection": {
    "policy": "target_real_val_internal",
    "selection_source": "target_real_val",
    "heldout_target_labels_used_for_selection": False,
    "full_target_distribution_used_for_selection": False,
    "source_floor_used": False,
    "notes": "Backfilled from legacy early_stop_info.json and command records.",
},
"run_record": {
    "registration_status": "provisional",
    "status": "succeeded",
    "purpose": "EffNet v7 VAE-LHAT K500 target-center adaptation evidence backfill.",
    "result_summary": "Legacy VAE-LHAT run backfilled for traceability; cite only after registry registration.",
},
```

If center-level `early_stop_info.json` exists, include per-center:

```python
"checkpoint_selection": {
    "center": center,
    "best_epoch": best_epoch,
    "metric": metric_name,
    "metric_value": metric_value,
    "selection_source": selection_source,
}
```

- [ ] **Step 4: Enforce trusted registration contract**

In `ecg_adv_gen/evidence/run_record.py`, ensure trusted registration requires:

```python
required_roles = {
    "run_manifest",
    "resolved_config",
    "command",
    "data_manifest",
    "selection",
    "k500_refs",
    "metrics_source",
}
```

When any role is missing, raise `RunRecordError` with the missing role names.

- [ ] **Step 5: Require method records for trusted comparison bundles**

In `ecg_adv_gen/evidence/comparison.py`, after building `method_records`, add:

```python
if trusted and not all(record.get("registered_run") for record in method_records.values()):
    raise ComparisonBundleError("trusted comparison requires registered_run metadata for every method")
```

Define the module-level error class before `build_comparison_bundle` if the module does not already expose one:

```python
class ComparisonBundleError(ValueError):
    """Raised when a comparison bundle cannot satisfy the requested evidence contract."""
```

- [ ] **Step 6: Run evidence tests**

Run:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest -q \
  util/tests/test_run_record_management.py \
  util/tests/test_agent_operating_layer.py
```

Expected: all selected tests pass.

- [ ] **Step 7: Run CPU-only audit smoke**

Run:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python scripts/agent/audit_agent_workspace.py --skip-existing-artifacts
```

Expected: no blocking artifact-risk failures; guarded `model/*` dirty handles may remain warnings only.

---

### Task 5: Convert High-Risk Long Argv Surfaces To Typed Adapters

**Files:**
- Create: `ecg_adv_gen/config/adapters/pn2021_eval.py`
- Create: `ecg_adv_gen/config/adapters/pn2021c_eval.py`
- Create: `ecg_adv_gen/config/adapters/prompt_token_online_at.py`
- Create: `ecg_adv_gen/config/adapters/direct_finetune.py`
- Create: `util/tests/test_pn2021_eval_adapter.py`
- Create: `util/tests/test_pn2021c_eval_adapter.py`
- Create: `util/tests/test_prompt_token_adapter.py`
- Create: `util/tests/test_direct_finetune_adapter.py`
- Modify: `ecg_adv_gen/config/adapters/registry.py`
- Modify: `configs/schemas/experiment_config.schema.json`
- Modify: selected experiment YAML files listed in the file structure section
- Modify: `configs/active_scripts.yaml`
- Modify: `configs/README.md`

- [ ] **Step 1: Add adapter output tests**

For `util/tests/test_pn2021_eval_adapter.py`, assert the adapter produces required paper flags:

```python
from ecg_adv_gen.config.adapters.pn2021_eval import build_pn2021_eval_argv


def test_pn2021_eval_adapter_forces_paper_refexcluded_flags():
    config = {
        "paths": {"data_root": "/data", "output_root": "/out"},
        "paper_protocol": {"kshot": {"k": 500, "seed": 20260531}},
        "evaluation": {"mapping_version": "v7", "mapping_hash": "hash"},
        "runtime": {"run_id": "run1"},
        "experiment": {"name": "pn2021_eval"},
        "runner": {"script": "scripts/triple_labels/eval_crosscenter.py"},
    }
    argv = [str(x) for x in build_pn2021_eval_argv(config, {"matrix": {"center": "ningbo"}})]
    assert "--eval_protocol" in argv
    assert argv[argv.index("--eval_protocol") + 1] == "paper_refexcluded"
    assert "--exclude_ref_ids" in argv
    assert "--report_drop_all_zero_pn2021" in argv
```

For the other adapter tests, assert:

- `pn2021c_eval` requires `--clean_eval_json` and `--required_cache_version`.
- `prompt_token_online_at` emits `--quick_eval_source target_real_val` and run-id-scoped outputs.
- `direct_finetune` emits K, seed, center, ref meta, output root, and eval postprocess hooks.

- [ ] **Step 2: Run adapter tests and confirm failure**

Run:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest -q \
  util/tests/test_pn2021_eval_adapter.py \
  util/tests/test_pn2021c_eval_adapter.py \
  util/tests/test_prompt_token_adapter.py \
  util/tests/test_direct_finetune_adapter.py
```

Expected: import failures for new adapter modules.

- [ ] **Step 3: Implement adapter modules**

Each adapter exposes one builder:

```python
def build_pn2021_eval_argv(config: Mapping[str, Any], context: Mapping[str, Any]) -> list[Any]:
    ...


def build_pn2021c_eval_argv(config: Mapping[str, Any], context: Mapping[str, Any]) -> list[Any]:
    ...


def build_prompt_token_online_at_argv(config: Mapping[str, Any], context: Mapping[str, Any]) -> list[Any]:
    ...


def build_direct_finetune_argv(config: Mapping[str, Any], context: Mapping[str, Any]) -> list[Any]:
    ...
```

Rules:

- Builders only construct argv; they do not touch disk.
- Builders use `context["matrix"]` for center/backbone/protocol expansion.
- Builders hard-code paper-safe policy flags where the protocol requires them.
- Builders derive output paths from `paths.output_root`, `experiment.name`, and `runtime.run_id`.

- [ ] **Step 4: Register adapters**

Modify `ecg_adv_gen/config/adapters/registry.py`:

```python
from .direct_finetune import build_direct_finetune_argv
from .pn2021_eval import build_pn2021_eval_argv
from .pn2021c_eval import build_pn2021c_eval_argv
from .prompt_token_online_at import build_prompt_token_online_at_argv

RUNNER_ADAPTERS.update(
    {
        "direct_finetune": build_direct_finetune_argv,
        "pn2021_eval": build_pn2021_eval_argv,
        "pn2021c_eval": build_pn2021c_eval_argv,
        "prompt_token_online_at": build_prompt_token_online_at_argv,
    }
)
```

If `RUNNER_ADAPTERS` is a literal dict, add the entries in that literal instead of calling `update`.

- [ ] **Step 5: Convert clean PN2021 eval YAML first**

Modify `configs/experiments/pn2021_eval_v7_sjr_rgq_refexcluded.yaml` to use:

```yaml
runner:
  adapter: pn2021_eval
  script: scripts/triple_labels/eval_crosscenter.py
  matrix:
    center:
      - ningbo
      - chapman_shaoxing
      - cpsc_2018
      - georgia
```

Keep existing postprocess commands, run record metadata, mapping version, and paths.

- [ ] **Step 6: Convert PN2021-C YAML second**

Modify `configs/experiments/pn2021c_effnet_v7_augmix_vs_noaug.yaml` to use:

```yaml
runner:
  adapter: pn2021c_eval
  script: scripts/triple_labels/eval_pn2021_corruptions.py
```

Keep the matrix dimensions for center and method variant. Ensure typed fields include:

```yaml
pn2021c:
  required_cache_version: v7_refexcluded_100hz1000
  clean_eval_json: ${paths.output_root}/...
```

Use the existing clean eval output path from the current YAML.

- [ ] **Step 7: Convert prompt-token online AT YAML third**

Modify `configs/experiments/ecgtwin_prompt_token_online_at_minimal.yaml` to use:

```yaml
runner:
  adapter: prompt_token_online_at
  script: scripts/pgd_cross_center/synth_online_at_super5.py
```

Keep existing artifact references for gated samples, latent pool, class trust, target-real signals, and ref meta.

- [ ] **Step 8: Convert direct finetune matrices fourth**

Convert direct matrix configs after eval adapters pass:

```yaml
runner:
  adapter: direct_finetune
  script: scripts/paper/run_direct_finetune_k500_20260516.py
```

For benchmark configs, set:

```yaml
model:
  backbone: benchmark_resnet1d_wang
```

or the existing backbone value from that YAML.

- [ ] **Step 9: Run config tests and dry-run smoke**

Run:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest -q \
  util/tests/test_config_loader.py \
  util/tests/test_active_script_index.py \
  util/tests/test_pn2021_eval_adapter.py \
  util/tests/test_pn2021c_eval_adapter.py \
  util/tests/test_prompt_token_adapter.py \
  util/tests/test_direct_finetune_adapter.py
```

Run clean eval dry-run:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python scripts/run_experiment.py \
  --config configs/experiments/pn2021_eval_v7_sjr_rgq_refexcluded.yaml \
  --local-config configs/local/linbinhao_server.example.yaml \
  --run-id smoke_pn2021_eval_adapter_20260605 \
  --dry-run
```

Expected dry-run output contains `--eval_protocol paper_refexcluded` and `--exclude_ref_ids`.

---

### Task 6: Split Loader Command Audit Into A Dispatcher

**Files:**
- Create: `ecg_adv_gen/config/runner_audit.py`
- Modify: `ecg_adv_gen/config/loader.py`
- Modify: adapter modules with audit helpers
- Modify: `util/tests/test_config_loader.py`
- Modify: `util/tests/test_active_script_index.py`

- [ ] **Step 1: Add regression test for audit dispatcher coverage**

In `util/tests/test_config_loader.py`, add:

```python
from ecg_adv_gen.config.runner_audit import audit_runner_command


def test_runner_audit_dispatches_eval_crosscenter_command():
    command = {
        "argv": [
            "python",
            "scripts/triple_labels/eval_crosscenter.py",
            "--scheme",
            "super5",
            "--model_dir",
            "/out/model",
            "--exclude_ref_ids",
            "/out/ref.json",
            "--eval_protocol",
            "paper_refexcluded",
        ],
        "matrix": {"center": "ningbo"},
    }
    report = audit_runner_command(command, config={"paper_protocol": {"kshot": {"k": 500, "seed": 1}}})
    assert report["errors"] == []
```

- [ ] **Step 2: Run dispatcher test and confirm failure**

Run:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest -q util/tests/test_config_loader.py -k runner_audit_dispatches
```

Expected: import failure for `ecg_adv_gen.config.runner_audit`.

- [ ] **Step 3: Implement `runner_audit.py`**

Create:

```python
"""Command audit dispatcher for YAML-managed runner commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def script_name(command: Mapping[str, Any]) -> str:
    argv = [str(x) for x in command.get("argv", [])]
    if len(argv) < 2:
        return ""
    return Path(argv[1]).name


def audit_runner_command(command: Mapping[str, Any], *, config: Mapping[str, Any]) -> dict[str, list[str]]:
    name = script_name(command)
    if name == "eval_crosscenter.py":
        from ecg_adv_gen.config.adapters.pn2021_eval import audit_pn2021_eval_command

        return audit_pn2021_eval_command(command, config=config)
    if name == "eval_pn2021_corruptions.py":
        from ecg_adv_gen.config.adapters.pn2021c_eval import audit_pn2021c_eval_command

        return audit_pn2021c_eval_command(command, config=config)
    if name in {"run_direct_finetune_k500_20260516.py", "run_benchmark_direct_finetune_v7_20260530.py"}:
        from ecg_adv_gen.config.adapters.direct_finetune import audit_direct_finetune_command

        return audit_direct_finetune_command(command, config=config)
    return {"errors": [], "warnings": [f"no dedicated audit for {name or '<empty>'}"]}
```

- [ ] **Step 4: Move audit branches out of loader**

In `ecg_adv_gen/config/loader.py::audit_runner_commands`, replace per-script audit blocks with:

```python
from ecg_adv_gen.config.runner_audit import audit_runner_command

for command in commands:
    report = audit_runner_command(command, config=config)
    errors.extend(report.get("errors", []))
    warnings.extend(report.get("warnings", []))
```

Keep the public `audit_runner_commands` signature unchanged.

- [ ] **Step 5: Run full config audit tests**

Run:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest -q \
  util/tests/test_config_loader.py \
  util/tests/test_active_script_index.py
```

Expected: all selected tests pass, and `audit_runner_commands` no longer owns per-script audit details.

---

### Task 7: Extract Low-Risk Online-AT Helpers From The Large Training Script

**Files:**
- Create: `ecg_adv_gen/training/online_at_records.py`
- Create: `util/tests/test_online_at_records.py`
- Modify: `scripts/pgd_cross_center/synth_online_at_super5.py`
- Modify: `ecg_adv_gen/training/__init__.py`

- [ ] **Step 1: Write tests for checkpoint selection payloads**

Create `util/tests/test_online_at_records.py`:

```python
from ecg_adv_gen.training.online_at_records import build_checkpoint_selection_record


def test_build_checkpoint_selection_record_marks_target_val_as_internal():
    record = build_checkpoint_selection_record(
        center="ningbo",
        best_epoch=4,
        metric_name="target_macro_auprc",
        metric_value=0.51,
        selection_source="target_real_val",
        heldout_target_labels_used=False,
    )
    assert record["center"] == "ningbo"
    assert record["selection_source"] == "target_real_val"
    assert record["heldout_target_labels_used_for_selection"] is False
    assert record["paper_safe_selection"] is True
```

- [ ] **Step 2: Run helper tests and confirm failure**

Run:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest -q util/tests/test_online_at_records.py
```

Expected: import failure for `ecg_adv_gen.training.online_at_records`.

- [ ] **Step 3: Implement helper**

Create `ecg_adv_gen/training/online_at_records.py`:

```python
"""Small CPU-safe record builders for online AT runs."""

from __future__ import annotations

from typing import Any


def build_checkpoint_selection_record(
    *,
    center: str,
    best_epoch: int,
    metric_name: str,
    metric_value: float,
    selection_source: str,
    heldout_target_labels_used: bool,
) -> dict[str, Any]:
    paper_safe = selection_source == "target_real_val" and not heldout_target_labels_used
    return {
        "center": str(center),
        "best_epoch": int(best_epoch),
        "metric_name": str(metric_name),
        "metric_value": float(metric_value),
        "selection_source": str(selection_source),
        "heldout_target_labels_used_for_selection": bool(heldout_target_labels_used),
        "paper_safe_selection": bool(paper_safe),
    }
```

- [ ] **Step 4: Replace inline selection payload construction**

In `scripts/pgd_cross_center/synth_online_at_super5.py`, import:

```python
from ecg_adv_gen.training.online_at_records import build_checkpoint_selection_record
```

Where the script writes early-stop or selection metadata, replace the inline dict with:

```python
selection_record = build_checkpoint_selection_record(
    center=args.center_name,
    best_epoch=best_epoch,
    metric_name=args.es_metric,
    metric_value=float(best_metric),
    selection_source=args.quick_eval_source,
    heldout_target_labels_used=False,
)
```

Keep the old output keys by nesting `selection_record` under the existing JSON field instead of changing file names.

- [ ] **Step 5: Run online AT helper tests**

Run:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest -q \
  util/tests/test_online_at_helpers.py \
  util/tests/test_online_at_records.py
```

Expected: all selected tests pass.

---

## Full Verification

After all tasks complete, run:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest -q \
  util/tests/test_config_loader.py \
  util/tests/test_active_script_index.py \
  util/tests/test_agent_operating_layer.py \
  util/tests/test_run_record_management.py \
  util/tests/test_data_contracts.py \
  util/tests/test_evaluation_selection.py \
  util/tests/test_pn2021_index_kshot.py \
  util/tests/test_pn2021_metric_views.py \
  util/tests/test_pn2021_corruptions.py \
  util/tests/test_pn2021_eval_cache.py \
  util/tests/test_online_at_helpers.py
```

Run import/compile smoke:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m py_compile \
  ecg_adv_gen/evaluation/protocols.py \
  ecg_adv_gen/evaluation/pn2021c_metadata.py \
  ecg_adv_gen/config/adapters/pn2021_eval.py \
  ecg_adv_gen/config/adapters/pn2021c_eval.py \
  ecg_adv_gen/config/adapters/prompt_token_online_at.py \
  ecg_adv_gen/config/adapters/direct_finetune.py \
  ecg_adv_gen/config/runner_audit.py \
  scripts/triple_labels/eval_crosscenter.py \
  scripts/triple_labels/eval_pn2021_corruptions.py \
  scripts/pgd_cross_center/synth_online_at_super5.py
```

Run managed dry-runs:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python scripts/run_experiment.py \
  --config configs/experiments/pn2021_eval_v7_sjr_rgq_refexcluded.yaml \
  --local-config configs/local/linbinhao_server.example.yaml \
  --run-id smoke_pn2021_eval_adapter_20260605 \
  --dry-run

/home/linbinhao/micromamba/envs/ECGTwin/bin/python scripts/run_experiment.py \
  --config configs/experiments/pn2021c_effnet_v7_augmix_vs_noaug.yaml \
  --local-config configs/local/linbinhao_server.example.yaml \
  --run-id smoke_pn2021c_eval_adapter_20260605 \
  --dry-run

/home/linbinhao/micromamba/envs/ECGTwin/bin/python scripts/run_experiment.py \
  --config configs/experiments/ecgtwin_prompt_token_online_at_minimal.yaml \
  --local-config configs/local/linbinhao_server.example.yaml \
  --run-id smoke_prompt_token_adapter_20260605 \
  --dry-run
```

Run workspace audit:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python scripts/agent/audit_agent_workspace.py --skip-existing-artifacts
```

Expected final state:

- Clean PN2021 eval dry-run includes `--eval_protocol paper_refexcluded`.
- PN2021-C dry-run includes `--clean_eval_json` and `--required_cache_version`.
- Prompt-token online AT dry-run no longer has a 100-plus item hand-written YAML argv block.
- Trusted comparison bundle records registered run metadata for both direct and VAE methods.
- Audit has no blocking artifact risks.
- Dirty `model/*` handles remain unstaged and ignored as external handles.

## Stop Conditions

Stop and report before continuing if any of these occur:

- A required smoke test would need GPU execution.
- A fix requires changing final metric values instead of only enforcing protocol or metadata checks.
- A paper-safe check contradicts existing trusted registry data.
- A generated artifact or checkpoint appears in `git status`.
- `model/*` would need staging, deletion, relinking, or content edits.

## Suggested Goal Prompt

```text
开启 goal 模式，执行 docs/superpowers/plans/2026-06-05-paper-safety-typed-adapter-refactor.md。

目标：把当前 ECG_adv_Gen 主线重构到 paper-safe + typed-adapter + registered-evidence 更稳定的 handoff 状态。按计划逐项执行，优先完成 Task 1-5：PN2021 ref-excluded 评估强制门、PN2021-C metadata 校验、K-shot strict parser、EffNet v7 VAE-LHAT run-record/evidence 升级、pn2021_eval/pn2021c_eval/prompt_token_online_at/direct_finetune typed adapters。Task 6-7 在前面测试通过后再做。

约束：不跑 GPU 训练或长任务；不碰 model/*；不提交大 artifact；所有改动保持旧 legacy entrypoint 可用；优先写 CPU-only 测试再实现；每个阶段运行对应 pytest、py_compile、run_experiment --dry-run 和 audit_agent_workspace.py --skip-existing-artifacts。完成后给出已改文件、测试结果、仍未解决的风险、以及是否达到 handoff/commit 状态。
```
