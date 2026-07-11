"""PN2021-C clean/corrupt metadata compatibility checks."""

from __future__ import annotations

from typing import Any, Mapping

from ecg_adv_gen.matched_ecgfounder import MATCHED_ECGFOUNDER_CONTRACT_VERSION

from .pn2021_corruptions import ref_ids_sha256


class PN2021CMetadataError(ValueError):
    """Raised when PN2021-C metadata is incompatible with the clean baseline."""


def _run_k500_identity(payload: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    stage = str(payload.get("stage") or "")
    config = payload.get("config") if isinstance(payload.get("config"), Mapping) else {}
    config_stage = str(config.get("stage") or "")
    if stage == "ptbxl_source":
        required = ("center", "K", "target_train_K", "selected_ref_record_ids", "target_train_record_ids")
        missing = [field for field in required if field not in payload]
        if missing or "stage" not in config:
            raise PN2021CMetadataError(
                f"{label} ptbxl_source contract is missing fields: {missing + ([] if 'stage' in config else ['config.stage'])}"
            )
        if config_stage != stage:
            raise PN2021CMetadataError(
                f"{label} stage={stage!r} conflicts with config.stage={config_stage!r}"
            )
        if payload["center"] not in (None, ""):
            raise PN2021CMetadataError(f"{label} marked ptbxl_source but center is not empty")
        for field in ("K", "target_train_K"):
            if type(payload[field]) is not int or payload[field] != 0:
                raise PN2021CMetadataError(f"{label} marked ptbxl_source but {field} is not zero")
        for field in ("selected_ref_record_ids", "target_train_record_ids"):
            if not isinstance(payload[field], list) or payload[field]:
                raise PN2021CMetadataError(f"{label} marked ptbxl_source but {field} is not an empty list")
        return {"source_only": True, "stage": stage}

    required = ("stage", "center", "K", "target_train_K", "selected_ref_record_ids", "target_train_record_ids")
    missing = [field for field in required if field not in payload]
    if missing or "stage" not in config:
        raise PN2021CMetadataError(
            f"{label} K500 contract is missing fields: {missing + ([] if 'stage' in config else ['config.stage'])}"
        )
    if stage != "k500" or config_stage != stage:
        raise PN2021CMetadataError(
            f"{label} stage={stage!r} conflicts with required config.stage='k500'"
        )
    center = str(payload["center"] or "")
    if not center:
        raise PN2021CMetadataError(f"{label} K500 center is empty")
    if type(payload["K"]) is not int or payload["K"] != 500:
        raise PN2021CMetadataError(f"{label} K must be integer 500")
    matched = (
        str(payload.get("matched_contract") or config.get("matched_contract") or "")
        == MATCHED_ECGFOUNDER_CONTRACT_VERSION
    )
    if type(payload["target_train_K"]) is not int:
        raise PN2021CMetadataError(f"{label} target_train_K must be an integer")
    if matched:
        if not 0 < payload["target_train_K"] < payload["K"]:
            raise PN2021CMetadataError(f"{label} matched target_train_K must be in (0, 500)")
    elif payload["target_train_K"] != 500:
        raise PN2021CMetadataError(f"{label} target_train_K must be integer 500")
    k = payload["K"]
    for field in ("selected_ref_record_ids",):
        values = payload[field]
        if not isinstance(values, list) or len(values) != k:
            raise PN2021CMetadataError(f"{label} {field} must be a list with exactly {k} ids")
        normalized = [str(item) for item in values]
        if len(set(normalized)) != k:
            raise PN2021CMetadataError(f"{label} {field} must contain {k} unique ids")
    selected = [str(item) for item in payload["selected_ref_record_ids"]]
    target_train = [str(item) for item in payload["target_train_record_ids"]]
    if len(target_train) != payload["target_train_K"] or len(set(target_train)) != len(target_train):
        raise PN2021CMetadataError(f"{label} target_train_record_ids do not match target_train_K")
    if matched:
        split = payload.get("k500_split") if isinstance(payload.get("k500_split"), Mapping) else {}
        split_train = [str(item) for item in split.get("train_record_ids") or []]
        split_val = [str(item) for item in split.get("val_record_ids") or []]
        if set(split_train) != set(target_train):
            raise PN2021CMetadataError(f"{label} matched train IDs do not match k500_split")
        if set(split_train).intersection(split_val) or set(split_train).union(split_val) != set(selected):
            raise PN2021CMetadataError(f"{label} matched K500 split does not partition all 500 refs")
    elif set(target_train) != set(selected):
        raise PN2021CMetadataError(f"{label} target_train_record_ids do not match selected_ref_record_ids")
    selected_hash = ref_ids_sha256(selected)
    seed = config.get("subset_seed", config.get("seed"))
    return {
        "source_only": False,
        "stage": stage,
        "center": center,
        "k": k,
        "seed": int(seed) if seed is not None else None,
        "ref_record_ids_sha256": selected_hash,
    }


def validate_target_init_k500_identity(
    current_run: Mapping[str, Any],
    init_run: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject a target-adapted initialization trained on another K500 subset."""

    current = _run_k500_identity(current_run, label="current run")
    init = _run_k500_identity(init_run, label="initialization run")
    if init["source_only"]:
        return init
    fields = ("center", "k", "ref_record_ids_sha256")
    mismatches = [field for field in fields if current.get(field) != init.get(field)]
    if current.get("seed") is not None and init.get("seed") is not None and current["seed"] != init["seed"]:
        mismatches.append("seed")
    if mismatches:
        raise PN2021CMetadataError(
            "target-adapted initialization K500 identity mismatch: "
            + ", ".join(
                f"{field} current={current.get(field)!r} init={init.get(field)!r}"
                for field in mismatches
            )
        )
    return init


def evaluation_k500_identities(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Extract actual per-center ref-exclusion hashes from an evaluation result."""

    found: dict[str, dict[str, set[Any]]] = {}

    def add(center: object, ref_hash: object = None, count: object = None) -> None:
        if ref_hash in (None, "") and count is None:
            return
        values = found.setdefault(str(center), {"hashes": set(), "counts": set()})
        if ref_hash not in (None, ""):
            values["hashes"].add(str(ref_hash))
        if count is not None:
            if type(count) is not int:
                raise PN2021CMetadataError(
                    f"invalid evaluation K500 count for {center}: {count!r}"
                )
            values["counts"].add(count)

    protocol_hashes = _get(payload, "pn2021.eval_protocol.target_ref_id_hashes")
    if isinstance(protocol_hashes, Mapping):
        for center, value in protocol_hashes.items():
            add(center, value)
    pn2021_rows = _get(payload, "pn2021.per_center")
    if isinstance(pn2021_rows, Mapping):
        for center, row in pn2021_rows.items():
            if isinstance(row, Mapping):
                add(center, count=row.get("n_excluded_ref"))

    per_center = payload.get("per_center")
    if isinstance(per_center, Mapping):
        for center, corruptions in per_center.items():
            if not isinstance(corruptions, Mapping):
                continue
            for severities in corruptions.values():
                if not isinstance(severities, Mapping):
                    continue
                for severity, row in severities.items():
                    if not str(severity).isdigit() or not isinstance(row, Mapping):
                        continue
                    compatibility = row.get("metadata_compatibility")
                    metadata = row.get("metadata")
                    pn2021c = metadata.get("pn2021c") if isinstance(metadata, Mapping) else None
                    if isinstance(pn2021c, Mapping):
                        embedded_center = str(pn2021c.get("center") or "")
                        if embedded_center and embedded_center != str(center):
                            raise PN2021CMetadataError(
                                f"embedded PN2021-C center {embedded_center!r} does not match outer center {center!r}"
                            )
                    for source in (row, compatibility, pn2021c):
                        if isinstance(source, Mapping):
                            add(
                                center,
                                source.get("ref_record_ids_sha256"),
                                source.get("n_excluded_ref", source.get("n_excluded_ref_ids_for_center")),
                            )

    selected = payload.get("selected_ref_record_ids")
    center = str(payload.get("center") or "")
    if center and isinstance(selected, list) and selected:
        add(center, ref_ids_sha256(selected), len(selected))

    out: dict[str, dict[str, Any]] = {}
    for center, values in sorted(found.items()):
        if len(values["hashes"]) > 1 or len(values["counts"]) > 1:
            raise PN2021CMetadataError(
                f"multiple evaluation K500 identities for {center}: "
                f"hashes={sorted(values['hashes'])!r}, counts={sorted(values['counts'])!r}"
            )
        out[center] = {
            "k": next(iter(values["counts"]), None),
            "ref_record_ids_sha256": next(iter(values["hashes"]), None),
        }
    return out


def _get(mapping: Mapping[str, Any], dotted: str) -> Any:
    cur: Any = mapping
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _canonical_label_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    label_mapping = payload.get("label_mapping")
    if isinstance(label_mapping, Mapping):
        if "hash" in label_mapping or "version" in label_mapping:
            return {
                "version": label_mapping.get("version"),
                "hash": label_mapping.get("hash"),
            }
        nested = label_mapping.get("pn2021_super5")
        if isinstance(nested, Mapping):
            return {
                "version": nested.get("mapping_version") or nested.get("version"),
                "hash": nested.get("mapping_hash") or nested.get("hash"),
            }
    pn2021_mapping = payload.get("pn2021_mapping")
    if isinstance(pn2021_mapping, Mapping):
        return {
            "version": pn2021_mapping.get("mapping_version") or pn2021_mapping.get("version"),
            "hash": pn2021_mapping.get("mapping_hash") or pn2021_mapping.get("hash"),
        }
    return {"version": None, "hash": None}


def _canonical_preprocess(payload: Mapping[str, Any]) -> dict[str, Any]:
    preprocess = payload.get("preprocess")
    if isinstance(preprocess, Mapping):
        return {
            "contract_id": preprocess.get("contract_id"),
            "preprocess_mode": preprocess.get("preprocess_mode"),
            "norm_mode": preprocess.get("norm_mode"),
            "crop_len": preprocess.get("crop_len"),
            "target_len": preprocess.get("target_len"),
        }
    preprocess_config = payload.get("preprocess_config")
    if isinstance(preprocess_config, Mapping):
        return {
            "contract_id": preprocess_config.get("contract_id"),
            "preprocess_mode": preprocess_config.get("preprocess_mode"),
            "norm_mode": preprocess_config.get("norm_mode"),
            "crop_len": preprocess_config.get("crop_len"),
            "target_len": preprocess_config.get("target_len"),
        }
    return {
        "contract_id": None,
        "preprocess_mode": None,
        "norm_mode": None,
        "crop_len": None,
        "target_len": None,
    }


def canonical_pn2021c_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical metadata fields used for clean/corrupt comparison."""

    return {
        "label_mapping": _canonical_label_mapping(payload),
        "preprocess": _canonical_preprocess(payload),
    }


def build_pn2021c_metadata_payload(
    *,
    clean_metadata: Mapping[str, Any],
    center: str,
    corruption: str,
    public_severity: int,
    internal_severity: int,
    cache_version: str,
    n_excluded_ref: int,
    preprocess_contract_id: str,
    ref_record_ids_sha256: str | None = None,
    source_clean_cache: str | None = None,
    seed: int | None = None,
    limit: int | None = None,
    severity_profile: str = "standard",
    ops_source: str = "methods/augmix/ecg_ops.py",
    protocol_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the paper metadata payload for one PN2021-C corruption cache."""

    canonical = canonical_pn2021c_metadata(clean_metadata)
    label_mapping = dict(canonical["label_mapping"])
    preprocess = dict(canonical["preprocess"])
    preprocess["contract_id"] = preprocess.get("contract_id") or preprocess_contract_id
    pn2021c = {
        "cache_version": str(cache_version),
        "center": str(center),
        "source_clean_cache": source_clean_cache,
        "corruption": str(corruption),
        "severity": int(public_severity),
        "public_severity": int(public_severity),
        "internal_severity": int(internal_severity),
        "severity_profile": str(severity_profile),
        "n_excluded_ref": int(n_excluded_ref),
        "ref_record_ids_sha256": ref_record_ids_sha256,
        "seed": int(seed) if seed is not None else None,
        "limit": int(limit) if limit else None,
        "ops_source": ops_source,
    }
    protocol = dict(protocol_metadata or {})
    if protocol:
        if protocol.get("input_order_id"):
            pn2021c["input_order_id"] = str(protocol["input_order_id"])
        if protocol.get("zscore_timing"):
            pn2021c["zscore_timing"] = str(protocol["zscore_timing"])
    payload = dict(clean_metadata)
    payload["label_mapping"] = label_mapping
    payload["preprocess"] = preprocess
    payload["pn2021c"] = pn2021c
    payload["pn2021_c"] = dict(pn2021c)
    if protocol:
        payload["pn2021c_protocol"] = protocol
    return payload


def build_center_scoped_clean_eval_payload(
    clean_metadata: Mapping[str, Any],
    *,
    center: str,
    n_excluded_ref: int,
    ref_record_ids_sha256: str,
    preprocess_contract_id: str,
    preprocess_mode: str,
    norm_mode: str,
    crop_len: int,
    target_len: int = 1000,
    clean_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a PN2021-C clean baseline payload for the exact evaluated center.

    Legacy clean PN2021 JSONs may predate the paper-safe ``eval_protocol``
    block. PN2021-C evaluates one center at a time, so this payload freezes the
    current center's ref-exclusion hash and same-subset clean metrics without
    rewriting the original clean eval artifact.
    """

    payload = dict(clean_metadata)
    canonical = canonical_pn2021c_metadata(payload)
    preprocess = dict(canonical["preprocess"])
    preprocess["contract_id"] = preprocess.get("contract_id") or preprocess_contract_id
    preprocess["preprocess_mode"] = preprocess.get("preprocess_mode") or str(preprocess_mode)
    preprocess["norm_mode"] = preprocess.get("norm_mode") or str(norm_mode)
    preprocess["crop_len"] = preprocess.get("crop_len") or int(crop_len)
    preprocess["target_len"] = preprocess.get("target_len") or int(target_len)
    payload["preprocess"] = preprocess

    pn2021 = dict(payload.get("pn2021") or {})
    per_center = dict(pn2021.get("per_center") or {})
    row = dict(per_center.get(center) or {})
    row["n_excluded_ref"] = int(n_excluded_ref)
    if clean_metrics:
        for key in (
            "macro_auroc",
            "macro_auprc",
            "n_classes_used",
            "per_class",
            "drop_all_zero_macro_auroc",
            "drop_all_zero_macro_auprc",
            "drop_all_zero_n_classes_used",
            "drop_all_zero_per_class",
            "drop_all_zero_n_records",
        ):
            if key in clean_metrics:
                row[key] = clean_metrics[key]
    per_center[str(center)] = row

    protocol = dict(pn2021.get("eval_protocol") or {})
    target_hashes = dict(protocol.get("target_ref_id_hashes") or {})
    existing_hash = target_hashes.get(str(center))
    if existing_hash not in (None, "", ref_record_ids_sha256):
        raise PN2021CMetadataError(
            f"clean_eval ref hash mismatch for {center}: "
            f"existing={existing_hash!r}, current={ref_record_ids_sha256!r}"
        )
    target_hashes[str(center)] = str(ref_record_ids_sha256)
    protocol.update(
        {
            "status": "paper_safe",
            "eval_protocol": "paper_refexcluded",
            "target_ref_exclusion_required": True,
            "center_scoped_for_pn2021c": True,
            "target_ref_id_hashes": target_hashes,
            "preprocess_mode": preprocess["preprocess_mode"],
            "crop_len": int(preprocess["crop_len"]),
        }
    )
    pn2021["eval_protocol"] = protocol
    pn2021["per_center"] = per_center
    payload["pn2021"] = pn2021
    return payload


def validate_pn2021c_metadata_compatibility(
    *,
    clean_eval: Mapping[str, Any],
    corrupt_metadata: Mapping[str, Any],
    center: str,
    required_cache_version: str,
    required_input_order_id: str | None = None,
) -> dict[str, Any]:
    """Validate that one corrupted cache matches the clean eval baseline."""

    clean = canonical_pn2021c_metadata(clean_eval)
    corrupt = canonical_pn2021c_metadata(corrupt_metadata)
    clean_protocol = _get(clean_eval, "pn2021.eval_protocol")
    if not isinstance(clean_protocol, Mapping):
        raise PN2021CMetadataError("clean_eval pn2021.eval_protocol is required for PN2021-C paper evaluation")
    if clean_protocol.get("status") != "paper_safe" or clean_protocol.get("eval_protocol") != "paper_refexcluded":
        raise PN2021CMetadataError(
            "clean_eval pn2021.eval_protocol must be paper_safe paper_refexcluded "
            f"for PN2021-C paper evaluation, got {clean_protocol!r}"
        )
    comparisons = [
        ("label_mapping.hash", clean["label_mapping"]["hash"], corrupt["label_mapping"]["hash"]),
        ("label_mapping.version", clean["label_mapping"]["version"], corrupt["label_mapping"]["version"]),
        ("preprocess.contract_id", clean["preprocess"]["contract_id"], corrupt["preprocess"]["contract_id"]),
        ("preprocess.preprocess_mode", clean["preprocess"]["preprocess_mode"], corrupt["preprocess"]["preprocess_mode"]),
        ("preprocess.norm_mode", clean["preprocess"]["norm_mode"], corrupt["preprocess"]["norm_mode"]),
        ("preprocess.crop_len", clean["preprocess"]["crop_len"], corrupt["preprocess"]["crop_len"]),
        ("preprocess.target_len", clean["preprocess"]["target_len"], corrupt["preprocess"]["target_len"]),
    ]
    for name, clean_value, corrupt_value in comparisons:
        if clean_value != corrupt_value:
            raise PN2021CMetadataError(f"{name} mismatch: clean={clean_value!r}, corrupt={corrupt_value!r}")
        if clean_value in (None, ""):
            raise PN2021CMetadataError(f"{name} is required for PN2021-C paper evaluation")

    corrupt_cache_version = _get(corrupt_metadata, "pn2021c.cache_version")
    if corrupt_cache_version != required_cache_version:
        raise PN2021CMetadataError(
            f"pn2021c.cache_version mismatch: expected={required_cache_version!r}, got={corrupt_cache_version!r}"
        )

    clean_ref = _get(clean_eval, f"pn2021.per_center.{center}.n_excluded_ref")
    corrupt_ref = _get(corrupt_metadata, "pn2021c.n_excluded_ref")
    if clean_ref != corrupt_ref:
        raise PN2021CMetadataError(f"n_excluded_ref mismatch for {center}: clean={clean_ref!r}, corrupt={corrupt_ref!r}")
    clean_ref_hash = _get(clean_eval, f"pn2021.eval_protocol.target_ref_id_hashes.{center}")
    corrupt_ref_hash = _get(corrupt_metadata, "pn2021c.ref_record_ids_sha256")
    if clean_ref_hash != corrupt_ref_hash:
        raise PN2021CMetadataError(
            f"ref_record_ids_sha256 mismatch for {center}: "
            f"clean={clean_ref_hash!r}, corrupt={corrupt_ref_hash!r}"
        )
    if clean_ref_hash in (None, ""):
        raise PN2021CMetadataError(f"ref_record_ids_sha256 is required for {center}")

    if required_input_order_id:
        protocol_input_order = _get(corrupt_metadata, "pn2021c_protocol.input_order_id")
        pn2021c_input_order = _get(corrupt_metadata, "pn2021c.input_order_id")
        actual_input_order = protocol_input_order or pn2021c_input_order
        if actual_input_order != required_input_order_id:
            raise PN2021CMetadataError(
                f"input_order_id mismatch: expected={required_input_order_id!r}, "
                f"got={actual_input_order!r}"
            )
        zscore_timing = (
            _get(corrupt_metadata, "pn2021c_protocol.zscore_timing")
            or _get(corrupt_metadata, "pn2021c.zscore_timing")
        )
        if zscore_timing != "after_corruption_before_model":
            raise PN2021CMetadataError(
                "zscore_timing mismatch: expected='after_corruption_before_model', "
                f"got={zscore_timing!r}"
            )

    return {
        "compatible": True,
        "center": str(center),
        "cache_version": str(required_cache_version),
        "n_excluded_ref": int(clean_ref or 0),
        "ref_record_ids_sha256": str(clean_ref_hash),
        "label_mapping_hash": str(clean["label_mapping"]["hash"]),
        "preprocess_contract_id": str(clean["preprocess"]["contract_id"]),
        "input_order_id": str(required_input_order_id) if required_input_order_id else None,
    }
