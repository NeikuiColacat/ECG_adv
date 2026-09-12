"""Stdlib-only validation for frozen image-LLM shard evidence."""
from __future__ import annotations

import json
from pathlib import Path
import re

from util.pn2021_artifact_contract import sha256_file

_CLASS_ORDER = ("CD", "HYP", "MI", "NORM", "STTC")
LABEL_PATTERN = re.compile(r"(?<![A-Z])(?:NORM|MI|STTC|CD|HYP)(?![A-Z])")


def parse_response(response: str) -> dict:
    answer = response
    incomplete_reasoning = "<think>" in answer and "</think>" not in answer
    if "</think>" in answer:
        answer = answer.rsplit("</think>", 1)[1]
    elif incomplete_reasoning:
        answer = ""
    match = re.search(r"<answer>(.*?)</answer>", answer, flags=re.DOTALL)
    if match:
        answer = match.group(1)
    labels = [label for label in _CLASS_ORDER if label in set(LABEL_PATTERN.findall(answer))]
    stripped = answer.strip().strip("` \n\r\t")
    residual = LABEL_PATTERN.sub("", stripped)
    strict = bool(labels) and not residual.strip(" ;,[]\"'\n\r\t.:")
    return {"predicted_labels": labels, "parse_valid": bool(labels), "incomplete_reasoning": incomplete_reasoning,
            "strict_label_list": strict, "normal_abnormal_conflict": "NORM" in labels and len(labels) > 1}


def validate_result(payload: dict, result_path: Path) -> None:
    if payload.get("schema_version") not in {1, 2} or payload.get("artifact_type") != "ecg_image_evaluation_result":
        raise ValueError("invalid image-LLM artifact identity")
    if payload.get("status") != "complete":
        raise ValueError("image-LLM shard is incomplete")
    count = payload.get("expected_predictions")
    if type(count) is not int or count < 1 or payload.get("completed_predictions") != count:
        raise ValueError("image-LLM prediction count mismatch")
    reference = payload["prediction_artifact"]
    path = (result_path.parent / reference["path"]).resolve()
    path.relative_to(result_path.parent.resolve())
    if path.is_symlink() or sha256_file(path) != reference["sha256"]:
        raise ValueError("image-LLM prediction artifact mismatch")
    protocol = payload["protocol"]
    if protocol.get("source_snapshot"):
        for relative, digest in protocol["implementation_sha256"].items():
            source = (result_path.parent / "source_snapshot" / relative).resolve()
            source.relative_to((result_path.parent / "source_snapshot").resolve())
            if sha256_file(source) != digest:
                raise ValueError("frozen inference source snapshot mismatch")
    cohort_path = result_path.parent / "cohort.jsonl"
    if sha256_file(cohort_path) != protocol["cohort_sha256"]:
        raise ValueError("image-LLM cohort artifact mismatch")
    with cohort_path.open() as handle:
        cohort = [json.loads(line) for line in handle]
    if payload["schema_version"] == 2:
        if protocol.get("scheduler") != "fixed_task_queue_v1" or protocol.get("cohort_mode") not in {"full_pulse", "pilot_smoke"}:
            raise ValueError("invalid elastic result scheduler or cohort")
        if sha256_file(result_path.parent / "task_index.json") != protocol["task_index_sha256"]:
            raise ValueError("elastic task index mismatch")
        index = json.loads((result_path.parent / "task_index.json").read_text())
        if (len({task["task_id"] for task in index}) != len(index)
                or sum(task["n_predictions"] for task in index) != count):
            raise ValueError("elastic task index is incomplete or duplicated")
        for task in index:
            task_path = Path(task["path"]).resolve()
            task_path.relative_to(Path("/home/linbinhao/ECG_adv_data"))
            if sha256_file(task_path) != task["sha256"]:
                raise ValueError("elastic committed task changed")
        selected = cohort
    else:
        rank, world = protocol["rank"], protocol["world_size"]
        if type(rank) is not int or type(world) is not int or not 0 <= rank < world <= 4:
            raise ValueError("invalid image-LLM shard topology")
        selected = [r for i, r in enumerate(cohort) if (i // 4) % world == rank]
    expected = {(r["sample_key"], c["condition_id"]) for r in selected for c in protocol["conditions"]}
    if len(expected) != count:
        raise ValueError("image-LLM expected grid count mismatch")
    truth = {r["sample_key"]: r["label_names"] for r in selected}
    seen = set()
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["sample_key"], row["condition_id"])
            if key in seen:
                raise ValueError("duplicate image-LLM prediction")
            if key not in expected or row.get("true_labels") != truth[key[0]]:
                raise ValueError("image-LLM prediction identity or labels mismatch")
            for field in ("clean_waveform_sha256", "input_waveform_sha256"):
                digest = row.get(field, "")
                if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                    raise ValueError("missing per-prediction input waveform fingerprint")
            seen.add(key)
    if seen != expected:
        raise ValueError("image-LLM prediction file count mismatch")
    if protocol.get("mapping_hash") != "555ec85d5b51" or protocol.get("metric_view") != "drop_all_zero":
        raise ValueError("image-LLM metric identity mismatch")
    if not protocol.get("k500_ref_excluded"):
        raise ValueError("image-LLM ref exclusion not verified")
