#!/usr/bin/env bash
set -euo pipefail

WAIT_PID="${1:?usage: watch_then_run_selected_pn2021_c_queue.sh <pid> <pipeline_log>}"
PIPELINE_LOG="${2:?usage: watch_then_run_selected_pn2021_c_queue.sh <pid> <pipeline_log>}"
QUEUE_LOG="${QUEUE_LOG:-/root/autodl-tmp/triple_labels/selected_pn2021_c_queue_after_author_repro.log}"
SLEEP_SEC="${SLEEP_SEC:-300}"

echo "$(date -Is) [watch] waiting for pid=${WAIT_PID}" | tee -a "${QUEUE_LOG}"
while kill -0 "${WAIT_PID}" 2>/dev/null; do
  sleep "${SLEEP_SEC}"
done

if grep -q "\[pipeline\] done" "${PIPELINE_LOG}"; then
  echo "$(date -Is) [watch] upstream pipeline done; starting selected PN2021-C queue" | tee -a "${QUEUE_LOG}"
  bash scripts/triple_labels/run_selected_pn2021_c_queue.sh 2>&1 | tee -a "${QUEUE_LOG}"
else
  echo "$(date -Is) [watch] upstream pipeline did not finish cleanly; queue skipped" | tee -a "${QUEUE_LOG}"
  exit 1
fi
