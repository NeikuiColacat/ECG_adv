#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_ROOT="${ECG_ADV_DATA_ROOT:-${HOME}/autodl-tmp}"
MODEL_ROOT="${MODEL_ROOT:-${ECG_ADV_MODEL_ROOT:-${DATA_ROOT}/models}}"

mkdir -p "${MODEL_ROOT}" "${REPO_ROOT}/model"

clone_or_report() {
  local name="$1"
  local url="$2"
  local dst="${MODEL_ROOT}/${name}"

  if [ -e "${dst}" ]; then
    if git -C "${dst}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      echo "[ok] ${dst} exists"
    else
      echo "[warn] ${dst} exists but is not a valid git checkout; inspect before replacing"
    fi
  else
    echo "[clone] ${url} -> ${dst}"
    git clone "${url}" "${dst}"
  fi

  ln -sfn "${dst}" "${REPO_ROOT}/model/${name}"
}

clone_or_report "DeepECG" "https://github.com/HeartWise-AI/DeepECG_Docker.git"
clone_or_report "ECGTwin" "https://github.com/Raiiyf/ECGTwin.git"
clone_or_report "ecg_ptbxl_benchmarking" "https://github.com/helme/ecg_ptbxl_benchmarking.git"
clone_or_report "advdiff" "https://github.com/EricDai0/advdiff.git"

echo "[done] model links:"
ls -la "${REPO_ROOT}/model"
