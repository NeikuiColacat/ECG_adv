"""CPU-only tests for the default PN2021 -> PTB-XL Super5 mapping JSONL."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

from ecg_adv_gen.labels import super5_mapping


REPO = Path(__file__).resolve().parents[2]
DATA_PROTOCOL = REPO / "configs" / "data" / "data_protocol_default.yaml"


def _load_mapping_jsonl(
    path: Path,
    *,
    metadata_record_type: str,
    label_record_type: str,
) -> tuple[list[dict], list[dict]]:
    metadata_records: list[dict] = []
    label_records: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            record = json.loads(line)
            record_type = record.get("record_type")
            if record_type == metadata_record_type:
                metadata_records.append(record)
            elif record_type == label_record_type:
                label_records.append(record)
            else:
                raise AssertionError(f"Unexpected record_type at line {line_no}: {record_type!r}")
    return metadata_records, label_records


def test_data_protocol_default_label_mapping_jsonl_matches_code_policy():
    protocol = yaml.safe_load(DATA_PROTOCOL.read_text(encoding="utf-8"))
    labels = protocol["labels"]
    mapping_file = labels["mapping_file"]
    mapping_path = REPO / mapping_file["path"]
    evidence_source_path = REPO / mapping_file["evidence_source_path"]

    payload = mapping_path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == mapping_file["sha256"]
    assert payload == evidence_source_path.read_bytes()

    metadata_records, label_records = _load_mapping_jsonl(
        mapping_path,
        metadata_record_type=mapping_file["metadata_record_type"],
        label_record_type=mapping_file["label_record_type"],
    )
    assert len(metadata_records) == mapping_file["metadata_record_count"]
    assert len(label_records) == mapping_file["label_record_count"]
    assert len(metadata_records) + len(label_records) == mapping_file["record_count"]

    metadata = metadata_records[0]
    assert metadata["mapping_version_expected_in_code"] == labels["mapping_version"]
    assert metadata["super5_class_order"] == labels["class_order"]
    assert labels["mapping_version"] == super5_mapping.SUPER5_PN2021_MAPPING_VERSION
    assert labels["mapping_hash"] == super5_mapping.SUPER5_PN2021_MAPPING_HASH
    assert labels["class_order"] == list(super5_mapping.CLASS_NAMES_SUPER5)
    decision_counts = Counter(record["normalized_decision"] for record in label_records)
    assert decision_counts == {
        "direct_positive": 64,
        "norm_candidate": 1,
        "suppress_only": 63,
        "ignored": 9,
    }

    seen_snomed_codes: set[int] = set()
    for record in label_records:
        snomed_code = record["snomed_code"]
        assert isinstance(snomed_code, int)
        assert snomed_code not in seen_snomed_codes
        seen_snomed_codes.add(snomed_code)

        assert record["record_type"] == mapping_file["label_record_type"]
        assert record["pn2021_label"].strip()
        decision = record[mapping_file["decision_field"]]
        target_class = record.get("super5_target_class")
        if decision == "direct_positive":
            assert target_class in {"CD", "HYP", "MI", "STTC"}
        elif decision == "norm_candidate":
            assert target_class == "NORM"
        elif decision in {"suppress_only", "ignored"}:
            assert target_class is None
        else:
            raise AssertionError(f"Unexpected mapping decision: {decision!r}")

    positive = {
        record["snomed_code"]: record["super5_target_class"]
        for record in label_records
        if record["normalized_decision"] == "direct_positive"
    }
    norm_positive = {
        record["snomed_code"]
        for record in label_records
        if record["normalized_decision"] == "norm_candidate"
    }
    norm_suppress = {
        record["snomed_code"]
        for record in label_records
        if record["normalized_decision"] == "suppress_only"
    }
    ignored = {
        record["snomed_code"]
        for record in label_records
        if record["normalized_decision"] == "ignored"
    }

    assert positive == dict(super5_mapping.SNOMED_TO_SUPER5_POSITIVE)
    assert norm_positive == set(super5_mapping.NORM_POSITIVE_SNOMEDS)
    assert norm_suppress == set(super5_mapping.NORM_SUPPRESS_SNOMEDS)
    assert ignored == set(super5_mapping.SUPER5_PN2021_IGNORED_SNOMEDS)
