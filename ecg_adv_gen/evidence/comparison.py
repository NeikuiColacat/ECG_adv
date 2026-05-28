"""Build managed Direct-vs-candidate comparison bundles from metrics_long CSVs."""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ecg_adv_gen.evidence.registry import load_evidence_registry
from ecg_adv_gen.reporting import export_paper_table, merge_metrics_long


class ComparisonBundleError(ValueError):
    """Raised when a comparison bundle cannot be built safely."""


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_lookup(rows: list[dict[str, str]], view: str) -> dict[tuple[str, str, str], float]:
    out: dict[tuple[str, str, str], float] = {}
    for row in rows:
        if row.get("canonical_view") != view:
            continue
        if row.get("scope") != "center":
            continue
        if row.get("class_name"):
            continue
        metric = row.get("metric", "")
        if metric not in {"macro_auroc", "macro_auprc", "drop_all_zero_macro_auroc", "drop_all_zero_macro_auprc"}:
            continue
        canonical_metric = "macro_auroc" if metric.endswith("macro_auroc") else "macro_auprc"
        value = _float(row.get("value"))
        if value is not None:
            out[(row["run_id"], row["center"], canonical_metric)] = value
    return out


def _write_delta_table(
    *,
    merged_metrics: Path,
    output_path: Path,
    views: list[str],
    centers: list[str],
    baseline_run_id: str,
    candidate_run_id: str,
) -> list[dict[str, Any]]:
    rows = _read_metrics(merged_metrics)
    out_rows: list[dict[str, Any]] = []
    for view in views:
        lookup = _metric_lookup(rows, view)
        for center in centers:
            direct_auroc = lookup.get((baseline_run_id, center, "macro_auroc"))
            direct_auprc = lookup.get((baseline_run_id, center, "macro_auprc"))
            candidate_auroc = lookup.get((candidate_run_id, center, "macro_auroc"))
            candidate_auprc = lookup.get((candidate_run_id, center, "macro_auprc"))
            out_rows.append(
                {
                    "view": view,
                    "scope": "center",
                    "center": center,
                    "baseline_run_id": baseline_run_id,
                    "candidate_run_id": candidate_run_id,
                    "baseline_macro_auroc": direct_auroc,
                    "baseline_macro_auprc": direct_auprc,
                    "candidate_macro_auroc": candidate_auroc,
                    "candidate_macro_auprc": candidate_auprc,
                    "delta_macro_auroc": None if direct_auroc is None or candidate_auroc is None else candidate_auroc - direct_auroc,
                    "delta_macro_auprc": None if direct_auprc is None or candidate_auprc is None else candidate_auprc - direct_auprc,
                    "delta_macro_auroc_pp": None if direct_auroc is None or candidate_auroc is None else 100.0 * (candidate_auroc - direct_auroc),
                    "delta_macro_auprc_pp": None if direct_auprc is None or candidate_auprc is None else 100.0 * (candidate_auprc - direct_auprc),
                }
            )
        for metric in ("macro_auroc", "macro_auprc"):
            missing = [
                c for c in centers
                if (baseline_run_id, c, metric) not in lookup
                or (candidate_run_id, c, metric) not in lookup
            ]
            if missing:
                raise ComparisonBundleError(f"Missing {metric} rows for view={view}: {missing}")
        direct_mean_auroc = sum(lookup[(baseline_run_id, c, "macro_auroc")] for c in centers) / len(centers)
        direct_mean_auprc = sum(lookup[(baseline_run_id, c, "macro_auprc")] for c in centers) / len(centers)
        candidate_mean_auroc = sum(lookup[(candidate_run_id, c, "macro_auroc")] for c in centers) / len(centers)
        candidate_mean_auprc = sum(lookup[(candidate_run_id, c, "macro_auprc")] for c in centers) / len(centers)
        out_rows.append(
            {
                "view": view,
                "scope": "center_mean",
                "center": "|".join(centers),
                "baseline_run_id": baseline_run_id,
                "candidate_run_id": candidate_run_id,
                "baseline_macro_auroc": direct_mean_auroc,
                "baseline_macro_auprc": direct_mean_auprc,
                "candidate_macro_auroc": candidate_mean_auroc,
                "candidate_macro_auprc": candidate_mean_auprc,
                "delta_macro_auroc": candidate_mean_auroc - direct_mean_auroc,
                "delta_macro_auprc": candidate_mean_auprc - direct_mean_auprc,
                "delta_macro_auroc_pp": 100.0 * (candidate_mean_auroc - direct_mean_auroc),
                "delta_macro_auprc_pp": 100.0 * (candidate_mean_auprc - direct_mean_auprc),
            }
        )

    fieldnames = [
        "view",
        "scope",
        "center",
        "baseline_run_id",
        "candidate_run_id",
        "baseline_macro_auroc",
        "baseline_macro_auprc",
        "candidate_macro_auroc",
        "candidate_macro_auprc",
        "delta_macro_auroc",
        "delta_macro_auprc",
        "delta_macro_auroc_pp",
        "delta_macro_auprc_pp",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow({
                key: "" if row.get(key) is None else row.get(key)
                for key in fieldnames
            })
    return out_rows


def _claim_by_id(registry: dict[str, Any], claim_id: str | None) -> dict[str, Any]:
    claims = registry.get("active_claims", [])
    if claim_id is None:
        if len(claims) != 1:
            raise ComparisonBundleError("Registry has multiple claims; pass --claim-id")
        return claims[0]
    for claim in claims:
        if claim.get("claim_id") == claim_id:
            return claim
    raise ComparisonBundleError(f"Unknown claim_id: {claim_id}")


def build_comparison_bundle(
    *,
    registry_path: Path,
    local_config_path: Path,
    claim_id: str | None = None,
    output_dir: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build a managed comparison artifact from registry-declared metrics."""
    registry = load_evidence_registry(registry_path, local_config_path)
    claim = _claim_by_id(registry, claim_id)
    comparison = claim["comparison_bundle"]
    baseline_key = comparison["baseline_method"]
    candidate_key = comparison["candidate_method"]
    baseline = claim["methods"][baseline_key]
    candidate = claim["methods"][candidate_key]
    out_dir = Path(output_dir or comparison["output_dir"]).expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        if not force:
            raise ComparisonBundleError(f"Output directory is non-empty; pass --force: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics = Path(baseline["metrics_long"]).expanduser().resolve()
    candidate_metrics = Path(candidate["metrics_long"]).expanduser().resolve()
    merge_manifest = merge_metrics_long([baseline_metrics, candidate_metrics], out_dir)
    merged_metrics = Path(merge_manifest["metrics_long"])

    paper_table_paths: dict[str, str] = {}
    for view in claim["protocol"]["evaluation_views"]:
        table_dir = out_dir / f"paper_table_{view}"
        table_manifest = export_paper_table(
            merged_metrics,
            table_dir,
            view=view,
            dataset="pn2021",
            centers=list(claim["protocol"]["target_centers"]),
            include_dataset_rows=False,
            include_center_mean=True,
            baseline_run_id=comparison["baseline_run_id"],
        )
        table_path = table_dir / "paper_table.csv"
        final_table = out_dir / f"paper_table_{view}.csv"
        shutil.copy2(table_path, final_table)
        paper_table_paths[view] = str(final_table)
        shutil.copy2(table_dir / "paper_table_manifest.json", out_dir / f"paper_table_{view}_manifest.json")
        table_manifest["copied_to"] = str(final_table)

    delta_rows = _write_delta_table(
        merged_metrics=merged_metrics,
        output_path=out_dir / "comparison_delta.csv",
        views=list(claim["protocol"]["evaluation_views"]),
        centers=list(claim["protocol"]["target_centers"]),
        baseline_run_id=comparison["baseline_run_id"],
        candidate_run_id=comparison["candidate_run_id"],
    )
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_id": comparison["comparison_id"],
        "claim_id": claim["claim_id"],
        "baseline_method": baseline_key,
        "candidate_method": candidate_key,
        "baseline_run_id": comparison["baseline_run_id"],
        "candidate_run_id": comparison["candidate_run_id"],
        "protocol": claim["protocol"],
        "registry_path": str(Path(registry_path).expanduser().resolve()),
        "local_config_path": str(Path(local_config_path).expanduser().resolve()),
        "output_dir": str(out_dir),
        "source_metrics": {
            baseline_key: {
                "path": str(baseline_metrics),
                "sha256": _sha256_file(baseline_metrics),
            },
            candidate_key: {
                "path": str(candidate_metrics),
                "sha256": _sha256_file(candidate_metrics),
            },
        },
        "artifacts": {
            "metrics_long": str(merged_metrics),
            "merge_manifest": str(out_dir / "merge_manifest.json"),
            "comparison_delta": str(out_dir / "comparison_delta.csv"),
            "paper_tables": paper_table_paths,
        },
        "delta_rows": delta_rows,
    }
    manifest_path = out_dir / "comparison_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n",
        encoding="utf-8",
    )
    return manifest
