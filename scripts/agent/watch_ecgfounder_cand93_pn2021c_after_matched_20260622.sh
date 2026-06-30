#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "usage: $0 <gpu_id> <wait_eval_json> <log_path> <config=run_id> [<config=run_id> ...]" >&2
  exit 2
fi

GPU_ID="$1"
WAIT_EVAL="$2"
LOG_PATH="$3"
shift 3

PYTHON_BIN="/home/linbinhao/micromamba/envs/ECGTwin/bin/python"
PROJECT_ROOT="/home/linbinhao/ECG_adv_Gen"
LOCAL_CONFIG="configs/local/linbinhao_server.example.yaml"

mkdir -p "$(dirname "$LOG_PATH")"

{
  echo "[$(date -Is)] post_eval_watcher_start gpu=${GPU_ID} wait_eval=${WAIT_EVAL}"
  while [ ! -f "$WAIT_EVAL" ]; do
    echo "[$(date -Is)] waiting_matched_eval"
    sleep 120
  done
  echo "[$(date -Is)] matched_eval_found; sleep_for_cleanup"
  sleep 180
} >> "$LOG_PATH" 2>&1

cd "$PROJECT_ROOT"

for spec in "$@"; do
  config="${spec%%=*}"
  run_id="${spec#*=}"
  {
    echo "[$(date -Is)] launch_eval config=${config} run_id=${run_id} gpu=${GPU_ID}"
    nvidia-smi
  } >> "$LOG_PATH" 2>&1
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" scripts/run_experiment.py \
    --config "$config" \
    --local-config "$LOCAL_CONFIG" \
    --run-id "$run_id" \
    --execute \
    --write-plan >> "$LOG_PATH" 2>&1
  echo "[$(date -Is)] eval_done config=${config} run_id=${run_id}" >> "$LOG_PATH" 2>&1
done

echo "[$(date -Is)] post_eval_watcher_done" >> "$LOG_PATH" 2>&1
