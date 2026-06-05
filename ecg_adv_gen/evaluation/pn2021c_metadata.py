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
    payload = dict(clean_metadata)
    payload["label_mapping"] = label_mapping
    payload["preprocess"] = preprocess
    payload["pn2021c"] = pn2021c
    payload["pn2021_c"] = dict(pn2021c)
    return payload


def validate_pn2021c_metadata_compatibility(
    *,
    clean_eval: Mapping[str, Any],
    corrupt_metadata: Mapping[str, Any],
    center: str,
    required_cache_version: str,
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

    return {
        "compatible": True,
        "center": str(center),
        "cache_version": str(required_cache_version),
        "n_excluded_ref": int(clean_ref or 0),
        "ref_record_ids_sha256": str(clean_ref_hash),
        "label_mapping_hash": str(clean["label_mapping"]["hash"]),
        "preprocess_contract_id": str(clean["preprocess"]["contract_id"]),
    }
