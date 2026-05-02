"""Translate PTB-XL split records into ECGTwin-style English prompts.

This script is a safe local-only converter:
1) Prefer PTB-XL `scp_codes` mapped through
   ``scp_statements.csv`` descriptions (English) as the primary source.
2) If SCP descriptions are not available, fall back to local rule-based German→English
   report translation (no external API).
3) If both are unavailable, use class fallback prompts derived from super5.

Output is deterministic JSONL (and optional CSV), suitable for text_embed pipeline.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from scripts.triple_labels.label_schemes import CLASS_NAMES_SUPER5, ptbxl_scp_to_super5  # noqa: E402

DEFAULT_CSV = "/root/autodl-tmp/ptbxl/ptbxl_database.csv"
DEFAULT_SCP = "/root/autodl-tmp/ptbxl/scp_statements.csv"
DEFAULT_SPLIT = (
    "/root/autodl-tmp/graduate_project/splits/"
    "ptbxl_super5_seed42_train2000_val2000.json"
)
DEFAULT_OUT_JSONL = "/root/autodl-tmp/graduate_project/ptbxl_train2000_ecgtwin_prompts_seed42.jsonl"
DEFAULT_OUT_CSV = "/root/autodl-tmp/graduate_project/ptbxl_train2000_ecgtwin_prompts_seed42.csv"
DEFAULT_THRESHOLD = 50.0
MAX_FRAGMENTS = 8


REPORT_RULES: list[tuple[str, str]] = [
    # rhythm / normal variants
    (r"\bsinusrhythmus\b|\bsinus rhythm\b|\bsinusrytm\b", "sinus rhythm"),
    (r"\bsinusbradykardie\b|\bsinus bradycardia\b", "sinus bradycardia"),
    (r"\bsinustachykardie\b|\bsinus tachycardia\b", "sinus tachycardia"),
    (r"\bsinus arrhythmie\b|\bsinus arrhythmia\b", "sinus arrhythmia"),
    (r"\bnormales ekg\b|\bnormal ecg\b|\bnormalt ekg\b", "normal ecg"),
    (r"\bsonst normales ekg\b", "normal ecg"),
    (r"\batrial fibrillation\b|\bvorhofflimmern\b|\bafib\b", "atrial fibrillation"),
    (r"\batrial flutter\b|\bvorhofflattern\b", "atrial flutter"),
    # conduction / rhythm
    (r"\bventrikul(?:aere|äre|are|ale)\s+extrasystole|\bventrikulale extrasystole", "premature ventricular contractions"),
    (r"\bsupraventrikul(?:aere|äre|are|ale)\s+extrasystole|\bpremature atrial contraction", "premature atrial contractions"),
    (r"\bextrasystole\(n\)|\bextrasystolen?\b|\bextrasystole\b", "premature beats"),
    (r"\blinksschenkelblock\b|\bleft bundle branch block\b|\blbbb\b", "left bundle branch block"),
    (r"\brechtsschenkelblock\b|\brechtschenkelblock\b|\bright bundle branch block\b|\brbbb\b", "right bundle branch block"),
    (r"\bunvollstaendiger rechtsschenkelblock\b|\bincomplete right bundle branch block\b", "incomplete right bundle branch block"),
    (r"\bunvollstaendiger linksschenkelblock\b|\bincomplete left bundle branch block\b", "incomplete left bundle branch block"),
    (r"\blinksanteriorer hemiblock\b|\bleft anterior fascicular block\b", "left anterior fascicular block"),
    (r"\blinksposteriorer hemiblock\b|\bleft posterior fascicular block\b", "left posterior fascicular block"),
    (r"\bav[- ]?block\b|\batrioventrikul[aä]rer block\b|\batrioventricular block\b", "atrioventricular block"),
    (r"\bwpw\b|\bwolff parkinson white\b", "wolff parkinson white"),
    # hypertrophy / axis / volume
    (r"\blinkshypertrophie\b|\blinksventrikulaere hypertrophie\b|\bleft ventricular hypertrophy\b|\blvh\b", "left ventricular hypertrophy"),
    (r"\brechtshypertrophie\b|\bright ventricular hypertrophy\b|\brvh\b", "right ventricular hypertrophy"),
    (r"\bright atrial overload/enlargement\b|\bright atrial enlargement\b|\bright atrial over(?:ae|ä)lterung\b|\bp-dextrocardiale\b", "right atrial overload/enlargement"),
    (r"\bleft atrial overload/enlargement\b|\bleft atrial enlargement\b|\bleft atrial over(?:ae|ä)lterung\b|\bp-sinistrocardiale\b", "left atrial overload/enlargement"),
    (r"\bhochvoltage\b|\bhigh voltage\b|\bamplitudenkriterien\b|\bvoltages are high\b", "high voltage"),
    (r"\bniederspannung\b|\blow voltage\b|\blow limb lead voltage\b", "low voltage"),
    # MI / ischemia / ST-T
    (r"\binferiorer infarkt\b|\binferior myocardial infarction\b", "inferior myocardial infarction"),
    (r"\banteroseptaler infarkt\b|\banteroseptal myocardial infarction\b", "anteroseptal myocardial infarction"),
    (r"\bvorderwandinfarkt\b|\banterior myocardial infarction\b", "anterior myocardial infarction"),
    (r"\binferolateraler infarkt\b|\binferolateral myocardial infarction\b", "inferolateral myocardial infarction"),
    (r"\blateraler infarkt\b|\blateral myocardial infarction\b", "lateral myocardial infarction"),
    (r"\bmyokardinfarkt\b|\bmyocardial infarction\b|\binfarkt\b", "myocardial infarction"),
    (r"\bmyokardschaden\b|\bmyocardial injury\b", "myocardial injury"),
    (r"\bst[ -]?elevation\b|\bst elevation\b", "st elevation"),
    (r"\bst[ -]?senkung\b|\bst depression\b", "st depression"),
    (r"\bischem(?:ia|ie)\b|\bischaemie\b|\bischemic\b|\bischaemic\b", "ischemic st-t changes"),
    (r"\bt(?:-)?wave(?:\s+inversion)?\b|\bt inversions?\b|\bt abnormalities?\b|\babnorm(?:es|ales)? t\b|\bt abnorm", "t wave abnormality"),
    (r"\bst ?& ?t abnorm|\bst-t changes|\bst-t abnormal", "st-t abnormality"),
    (r"\bunspezifische st changes\b|\bnon-specific st changes\b|\bnon specific st changes\b", "non-specific ST changes"),
    (r"\blngqt\b|\blong qt(?: interval)?\b", "long qt interval"),
    (r"\blongerte\w+\s+pr\b|\bprolonged pr\b", "prolonged pr interval"),
    # general fallbacks captured from mixed-language phrases
    (r"\bpatholog(?:isches|e)\s+q\b|\bq waves?\b", "pathological q waves"),
    (r"\bno definite pathology\b", "no definite pathology"),
]

CLASS_FALLBACK = {
    "NORM": "sinus rhythm|normal ecg",
    "MI": "myocardial infarction",
    "STTC": "t wave abnormality|st-t abnormality",
    "HYP": "left ventricular hypertrophy|high voltage",
    "CD": "conduction disturbance",
}


def parse_scp_codes(value: str | dict[str, Any] | float | None) -> dict[str, float]:
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items()}
    try:
        parsed = ast.literal_eval(str(value))
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in parsed.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    return out


def clean_german_text(text: str) -> str:
    """Normalize odd encodings and keep simple punctuation boundaries."""
    text = str(text or "").lower()
    replacements = {
        "ä": "ae",
        "ö": "oe",
        "ü": "ue",
        "ß": "ss",
        "Ä": "ae",
        "Ö": "oe",
        "Ü": "ue",
    }
    for a, b in replacements.items():
        text = text.replace(a, b).replace(a.upper(), b.upper() if b else b)
    text = re.sub(r"\b\d+(?:\.\d+)?\s*unbest[a-z]*\s+bericht\b", " ", text)
    text = re.sub(r"\bunbest[a-z]*\s+bericht\b", " ", text)
    text = re.sub(r"\bedit:.*$", " ", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip(" .;")
    return text


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        val = re.sub(r"\s+", " ", str(item).strip().lower())
        val = val.strip(" .;")
        if not val or val in seen:
            continue
        seen.add(val)
        out.append(val)
    return out


def report_rule_fragments(report: str) -> list[str]:
    clean = clean_german_text(report)
    hits: list[str] = []
    for pattern, fragment in REPORT_RULES:
        if re.search(pattern, clean):
            hits.append(fragment)
    return dedupe_keep_order(hits)


def build_scp_maps(scp_statements_path: str) -> tuple[dict[str, str], dict[str, str]]:
    df = pd.read_csv(scp_statements_path).set_index("Unnamed: 0")
    desc: dict[str, str] = {}
    diag: dict[str, str] = {}
    for code, row in df.iterrows():
        description = row.get("description")
        diagnostic = row.get("diagnostic_class")
        if isinstance(description, str) and description.strip():
            desc[str(code)] = description.strip().lower()
        if isinstance(diagnostic, str) and diagnostic.strip():
            diag[str(code)] = diagnostic.strip()
    return desc, diag


def scp_fragments(
    scp_codes: dict[str, float],
    scp_desc: dict[str, str],
    threshold: float,
    fallback_if_empty: bool = True,
) -> tuple[list[str], float, list[tuple[str, float, str]]]:
    items = sorted(scp_codes.items(), key=lambda kv: float(kv[1]), reverse=True)
    used: list[tuple[str, float, str]] = []
    for code, conf in items:
        if conf < threshold:
            continue
        desc = scp_desc.get(code)
        if desc:
            used.append((code, float(conf), desc))
    if not used and fallback_if_empty:
        for code, conf in items:
            desc = scp_desc.get(code)
            if desc:
                used.append((code, float(conf), desc))
                break
    if not used:
        return [], 0.0, []
    fragments = [x[2] for x in used]
    conf = max(x[1] for x in used)
    return dedupe_keep_order(fragments), conf, used


def super5_from_scp(scp_codes: dict[str, float]) -> list[str]:
    y = ptbxl_scp_to_super5(scp_codes)
    return [name for name, val in zip(CLASS_NAMES_SUPER5, y) if float(val) == 1.0]


def build_prompt(
    report: str,
    scp_codes: dict[str, float],
    scp_desc: dict[str, str],
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[str, list[str], str, float, list[str], list[str]]:
    classes = super5_from_scp(scp_codes)
    report_frags = report_rule_fragments(report)
    scp_frags, conf, _ = scp_fragments(scp_codes, scp_desc, threshold, fallback_if_empty=True)

    # ECGTwin consumes concise English fragments separated by '|'. For this
    # conversion we preserve original-report information first, then append SCP
    # descriptions so the prompt remains label-faithful even when the German
    # rule dictionary misses a phrase.
    combined = dedupe_keep_order(report_frags + scp_frags)
    if combined:
        source_parts = []
        if report_frags:
            source_parts.append("german_report_rules")
        if scp_frags:
            source_parts.append("scp_codes")
        source = "+".join(source_parts)
        prompt = "|".join(combined[:MAX_FRAGMENTS])
        return prompt, classes, source, conf, report_frags, scp_frags

    # Last resort: class-level fallback.
    class_fallback = []
    for cls in classes:
        if cls in CLASS_FALLBACK:
            class_fallback.append(CLASS_FALLBACK[cls])
    source = "class_fallback"
    if not class_fallback:
        class_fallback = [CLASS_FALLBACK["NORM"]]
    prompt = "|".join(dedupe_keep_order("|".join(class_fallback).split("|"))[:MAX_FRAGMENTS])
    return prompt, classes, source, 0.0, [], []


def load_split_indices(split_path: str, split_key: str, limit: int = 0) -> list[int]:
    with open(split_path) as f:
        split = json.load(f)
    if split_key not in split:
        raise KeyError(f"split_json missing key: {split_key}")
    indices = [int(i) for i in split[split_key]]
    if limit > 0:
        indices = indices[:limit]
    return indices


def write_csv_records(out_path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    cols = [
        "ptbxl_index",
        "ecg_id",
        "original_report",
        "scp_codes",
        "super5",
        "ecgtwin_prompt_en",
        "confidence",
        "source",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in records:
            row = dict(r)
            row["scp_codes"] = json.dumps(row["scp_codes"], ensure_ascii=False)
            row["super5"] = "|".join(row["super5"])
            w.writerow({c: row.get(c, "") for c in cols})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", default=DEFAULT_CSV)
    ap.add_argument("--scp_statements", default=DEFAULT_SCP)
    ap.add_argument("--split_json", default=DEFAULT_SPLIT)
    ap.add_argument("--split_key", default="train_indices")
    ap.add_argument("--out_jsonl", default=DEFAULT_OUT_JSONL)
    ap.add_argument("--out_csv", default=DEFAULT_OUT_CSV)
    ap.add_argument("--write_csv", action="store_true", help="Also write CSV output.")
    ap.add_argument("--limit", type=int, default=0, help="Only process first N split indices.")
    ap.add_argument("--scp_threshold", type=float, default=DEFAULT_THRESHOLD)
    args = ap.parse_args()

    df = pd.read_csv(args.csv_path)
    indices = load_split_indices(args.split_json, args.split_key, args.limit)
    scp_desc, _ = build_scp_maps(args.scp_statements)

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    source_counter = Counter()
    for idx in indices:
        row = df.iloc[idx]
        scp_codes = parse_scp_codes(row["scp_codes"])
        prompt, classes, source, conf, report_frags, scp_frags = build_prompt(
            row.get("report", ""),
            scp_codes,
            scp_desc,
            threshold=args.scp_threshold,
        )
        source_counter[source] += 1
        records.append(
            {
                "ptbxl_index": int(idx),
                "ecg_id": int(row["ecg_id"]),
                "original_report": "" if pd.isna(row.get("report")) else str(row.get("report")),
                "scp_codes": scp_codes,
                "super5": classes,
                "ecgtwin_prompt_en": prompt,
                "clean_report": clean_german_text(row.get("report", "")),
                "german_rule_fragments": report_frags,
                "scp_fragments": scp_frags,
                "confidence": float(conf),
                "source": source,
            }
        )

    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if args.write_csv:
        write_csv_records(Path(args.out_csv), records)

    print(f"[done] wrote {len(records)} records to {out_jsonl}")
    for key, value in sorted(source_counter.items()):
        print(f"[source] {key}: {value}")


if __name__ == "__main__":
    main()
