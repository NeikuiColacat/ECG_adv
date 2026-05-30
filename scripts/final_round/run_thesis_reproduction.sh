#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE="${1:-help}"

DATA_ROOT="${ECG_ADV_DATA_ROOT:-${HOME}/autodl-tmp}"
PTBXL_ROOT="${ECG_ADV_PTBXL_ROOT:-${DATA_ROOT}/ptbxl}"
GRAD_ROOT="${ECG_ADV_GRAD_ROOT:-${DATA_ROOT}/graduate_project}"
TRIPLE_ROOT="${ECG_ADV_TRIPLE_ROOT:-${DATA_ROOT}/triple_labels}"
FINAL_ROUND_ROOT="${ECG_ADV_FINAL_ROUND_ROOT:-${DATA_ROOT}/final_round_ablation_20260504}"
APP_DATA_ROOT="${ECG_ADV_APP_DATA_ROOT:-${DATA_ROOT}/streamlit_ecg_demo}"

export ECG_ADV_DATA_ROOT="${DATA_ROOT}"
export ECG_ADV_PTBXL_ROOT="${PTBXL_ROOT}"
export ECG_ADV_GRAD_ROOT="${GRAD_ROOT}"
export ECG_ADV_TRIPLE_ROOT="${TRIPLE_ROOT}"
export ECG_ADV_FINAL_ROUND_ROOT="${FINAL_ROUND_ROOT}"
export ECG_ADV_APP_DATA_ROOT="${APP_DATA_ROOT}"
export TMPDIR="${TMPDIR:-${DATA_ROOT}/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${DATA_ROOT}/cache}"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_CMD=("${PYTHON}")
elif command -v uv >/dev/null 2>&1; then
  PYTHON_CMD=(uv run python)
else
  PYTHON_CMD=(python)
fi

usage() {
  cat <<EOF
Usage: bash scripts/final_round/run_thesis_reproduction.sh <stage>

Stages:
  env              Print resolved paths and Python command.
  restore_artifacts Restore migrate_files/*.tar.gz into ECG_ADV_DATA_ROOT.
  thesis_assets    Check local figure/image links referenced by thesis.md.
  preflight        Check archive/package-required artifact files from docs/artifact_manifest.json.
  preflight_full   Check all required full-rerun artifact/data dependencies.
  split            Create the PTB-XL super5 train2000/val2000/test17799 split.
  low_sample       Summarize Table 6.5-6.7 archived custom-split results.
  low_sample_rerun Re-run Table 6.5-6.7 training jobs; requires init checkpoints.
  ablation_6_8     Run Table 6.8 ablation commands.
  author_repro     Run ECGTwin IBE + DiT author-style reproduction.
  medical_validity Run Table 6.4 proxy quality summary for generated ECG pools.
  figures          Export five-class generated ECG visualization examples.
  feature_dist     Run optional real-vs-synthetic feature distribution analysis.
  export_onnx      Export the default EfficientNetV2 checkpoint to ONNX.
  build_trt        Build the default TensorRT FP16 engine from ONNX.
  benchmark        Run PyTorch/ONNX/TensorRT inference benchmark.
  evidence         Summarize thesis tables and key evidence into JSON/Markdown.
  package_artifacts Copy packageable files from docs/artifact_manifest.json into an artifact bundle.
  streamlit        Start the Streamlit demo.
  all              Run split, preflight, low_sample, medical_validity, figures, benchmark, evidence.

Important:
  Large datasets, checkpoints, ONNX, TensorRT engines, and generated pools are
  read from ECG_ADV_DATA_ROOT and are not committed to git.
EOF
}

print_env() {
  echo "ROOT=${ROOT}"
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "PTBXL_ROOT=${PTBXL_ROOT}"
  echo "GRAD_ROOT=${GRAD_ROOT}"
  echo "TRIPLE_ROOT=${TRIPLE_ROOT}"
  echo "FINAL_ROUND_ROOT=${FINAL_ROUND_ROOT}"
  echo "APP_DATA_ROOT=${APP_DATA_ROOT}"
  echo "PYTHON_CMD=${PYTHON_CMD[*]}"
  echo "[versions]"
  "${PYTHON_CMD[@]}" "${ROOT}/scripts/final_round/environment_snapshot.py"
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[missing] ${path}" >&2
    return 1
  fi
}

stage_split() {
  mkdir -p "${GRAD_ROOT}/splits"
  "${PYTHON_CMD[@]}" "${ROOT}/scripts/triple_labels/create_ptbxl_super5_split.py" \
    --csv_path "${PTBXL_ROOT}/ptbxl_database.csv" \
    --out_path "${GRAD_ROOT}/splits/ptbxl_super5_seed42_train2000_val2000.json"
}

stage_preflight() {
  "${PYTHON_CMD[@]}" "${ROOT}/scripts/final_round/preflight_thesis_archive.py" --verify-sha --scope archive
}

stage_preflight_full() {
  "${PYTHON_CMD[@]}" "${ROOT}/scripts/final_round/preflight_thesis_archive.py" \
    --verify-sha \
    --scope full \
    --report-json "${DATA_ROOT}/thesis_archive_artifacts/full_preflight_missing_artifacts.json" \
    --report-md "${DATA_ROOT}/thesis_archive_artifacts/full_preflight_missing_artifacts.md"
}

stage_thesis_assets() {
  "${PYTHON_CMD[@]}" "${ROOT}/scripts/final_round/check_thesis_assets.py"
}

stage_restore_artifacts() {
  "${PYTHON_CMD[@]}" "${ROOT}/scripts/final_round/restore_migrate_artifacts.py" \
    --migrate_dir "${ROOT}/migrate_files" \
    --out_dir "${DATA_ROOT}" \
    ${RESTORE_DRY_RUN:+--dry-run}
}

stage_author_repro() {
  bash "${ROOT}/scripts/ecgtwin_author_repro/run_author_repro_pipeline.sh"
}

stage_low_sample() {
  require_file "${GRAD_ROOT}/splits/ptbxl_super5_seed42_train2000_val2000.json"
  "${PYTHON_CMD[@]}" "${ROOT}/scripts/final_round/summarize_low_sample_results.py" \
    --out_dir "${FINAL_ROUND_ROOT}/low_sample_summary"
}

stage_low_sample_rerun() {
  require_file "${GRAD_ROOT}/splits/ptbxl_super5_seed42_train2000_val2000.json"
  LOW_SAMPLE_RERUN=1 bash "${ROOT}/scripts/final_round/run_low_sample_three_methods.sh"
}

stage_ablation_6_8() {
  bash "${ROOT}/scripts/final_round/run_table_6_8_ablations.sh"
}

stage_medical_validity() {
  "${PYTHON_CMD[@]}" "${ROOT}/scripts/final_round/run_medical_validity_ablation.py" \
    --out_dir "${FINAL_ROUND_ROOT}/medical_validity"
}

stage_figures() {
  "${PYTHON_CMD[@]}" "${ROOT}/scripts/final_round/curate_thesis_ecg_examples.py" \
    --out_dir "${FINAL_ROUND_ROOT}/thesis_selected_ecg_examples"
}

stage_feature_dist() {
  "${PYTHON_CMD[@]}" "${ROOT}/scripts/final_round/run_feature_distribution_analysis.py" \
    --out_dir "${FINAL_ROUND_ROOT}/feature_distribution"
}

stage_export_onnx() {
  "${PYTHON_CMD[@]}" "${ROOT}/scripts/deploy/export_efficientnetv2_onnx.py"
}

stage_build_trt() {
  "${PYTHON_CMD[@]}" "${ROOT}/scripts/deploy/build_tensorrt_engine.py" \
    --fp16 --workspace_mib 1024 --min_batch 1 --opt_batch 4 --max_batch 32
}

stage_benchmark() {
  "${PYTHON_CMD[@]}" "${ROOT}/scripts/deploy/benchmark_inference_backends.py" \
    --batch_sizes 1 8 32 --repeats "${BENCHMARK_REPEATS:-100}"
}

stage_evidence() {
  "${PYTHON_CMD[@]}" "${ROOT}/scripts/final_round/summarize_final_thesis_evidence.py" \
    --out_dir "${FINAL_ROUND_ROOT}/final_evidence"
}

stage_package_artifacts() {
  "${PYTHON_CMD[@]}" "${ROOT}/scripts/final_round/package_thesis_artifacts.py" \
    --include-optional \
    --skip-missing \
    --out_dir "${DATA_ROOT}/thesis_archive_artifacts"
}

stage_streamlit() {
  if command -v uv >/dev/null 2>&1; then
    uv run streamlit run "${ROOT}/apps/streamlit_ecg_demo/app.py" \
      --server.address "${STREAMLIT_ADDRESS:-0.0.0.0}" \
      --server.port "${STREAMLIT_PORT:-8501}" \
      --server.headless true
  else
    streamlit run "${ROOT}/apps/streamlit_ecg_demo/app.py" \
      --server.address "${STREAMLIT_ADDRESS:-0.0.0.0}" \
      --server.port "${STREAMLIT_PORT:-8501}" \
      --server.headless true
  fi
}

case "${STAGE}" in
  help|-h|--help) usage ;;
  env) print_env ;;
  restore_artifacts) stage_restore_artifacts ;;
  thesis_assets) stage_thesis_assets ;;
  preflight) stage_preflight ;;
  preflight_full) stage_preflight_full ;;
  split) stage_split ;;
  author_repro) stage_author_repro ;;
  low_sample) stage_low_sample ;;
  low_sample_rerun) stage_low_sample_rerun ;;
  ablation_6_8) stage_ablation_6_8 ;;
  medical_validity) stage_medical_validity ;;
  figures) stage_figures ;;
  feature_dist) stage_feature_dist ;;
  export_onnx) stage_export_onnx ;;
  build_trt) stage_build_trt ;;
  benchmark) stage_benchmark ;;
  evidence) stage_evidence ;;
  package_artifacts) stage_package_artifacts ;;
  streamlit) stage_streamlit ;;
  all)
    stage_thesis_assets
    stage_split
    stage_preflight
    stage_low_sample
    stage_medical_validity
    stage_figures
    stage_benchmark
    stage_evidence
    ;;
  *)
    usage
    echo "[error] unknown stage: ${STAGE}" >&2
    exit 2
    ;;
esac
