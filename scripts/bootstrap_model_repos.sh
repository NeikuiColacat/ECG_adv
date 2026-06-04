#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if DEFAULT_REPO_ROOT="$(git -C "${SCRIPT_DIR}/.." rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

REPO_ROOT="${REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
MODEL_ROOT_EXPLICIT=0
ECGFOUNDER_ROOT_EXPLICIT=0
if [[ -n "${MODEL_ROOT+x}" ]]; then
  MODEL_ROOT_EXPLICIT=1
fi
if [[ -n "${ECGFOUNDER_ROOT+x}" ]]; then
  ECGFOUNDER_ROOT_EXPLICIT=1
fi
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
MODEL_ROOT="${MODEL_ROOT:-${DATA_ROOT}/models}"
ECGFOUNDER_ROOT="${ECGFOUNDER_ROOT:-${DATA_ROOT}/ecgfounder}"
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON="${PYTHON}"
elif [[ -x "/home/linbinhao/micromamba/envs/ECGTwin/bin/python" ]]; then
  PYTHON="/home/linbinhao/micromamba/envs/ECGTwin/bin/python"
elif [[ -x "/root/miniforge3/envs/ECGTwin/bin/python" ]]; then
  PYTHON="/root/miniforge3/envs/ECGTwin/bin/python"
else
  PYTHON="python3"
fi
LOCAL_CONFIG=""
CHECK_ONLY=0

usage() {
  cat <<'USAGE'
Usage: bash scripts/bootstrap_model_repos.sh [options]

Options:
  --check-only                 Validate configured model links without cloning or relinking.
  --repo-root PATH             Repository root. Defaults to the git root containing this script.
  --data-root PATH             Data root used for default MODEL_ROOT. Env: DATA_ROOT.
  --model-root PATH            Root for DeepECG/ECGTwin/advdiff/ecg_ptbxl_benchmarking. Env: MODEL_ROOT.
  --ecgfounder-root PATH       ECGFounder target root. Env: ECGFOUNDER_ROOT.
  --local-config PATH          Read external_models targets from configs/local YAML.
  -h, --help                   Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-only)
      CHECK_ONLY=1
      shift
      ;;
    --repo-root)
      REPO_ROOT="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      if [[ "${MODEL_ROOT_EXPLICIT}" -eq 0 ]]; then
        MODEL_ROOT="${DATA_ROOT}/models"
      fi
      if [[ "${ECGFOUNDER_ROOT_EXPLICIT}" -eq 0 ]]; then
        ECGFOUNDER_ROOT="${DATA_ROOT}/ecgfounder"
      fi
      shift 2
      ;;
    --model-root)
      MODEL_ROOT="$2"
      MODEL_ROOT_EXPLICIT=1
      shift 2
      ;;
    --ecgfounder-root)
      ECGFOUNDER_ROOT="$2"
      ECGFOUNDER_ROOT_EXPLICIT=1
      shift 2
      ;;
    --local-config)
      LOCAL_CONFIG="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[error] unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

REPO_ROOT="$(realpath -m "${REPO_ROOT}")"
MODEL_ROOT="$(realpath -m "${MODEL_ROOT}")"
ECGFOUNDER_ROOT="$(realpath -m "${ECGFOUNDER_ROOT}")"

default_plan() {
  printf 'DeepECG\t%s\t%s\n' "https://github.com/HeartWise-AI/DeepECG_Docker.git" "${MODEL_ROOT}/DeepECG"
  printf 'ECGTwin\t%s\t%s\n' "https://github.com/Raiiyf/ECGTwin.git" "${MODEL_ROOT}/ECGTwin"
  printf 'advdiff\t%s\t%s\n' "https://github.com/EricDai0/advdiff.git" "${MODEL_ROOT}/advdiff"
  printf 'ecg_ptbxl_benchmarking\t%s\t%s\n' "https://github.com/helme/ecg_ptbxl_benchmarking.git" "${MODEL_ROOT}/ecg_ptbxl_benchmarking"
  printf 'ecgfounder\t%s\t%s\n' "https://github.com/PKUDigitalHealth/ECGFounder.git" "${ECGFOUNDER_ROOT}"
}

bootstrap_plan() {
  if [[ -n "${LOCAL_CONFIG}" ]]; then
    local config_path="${LOCAL_CONFIG}"
    if [[ "${config_path}" != /* ]]; then
      config_path="${REPO_ROOT}/${config_path}"
    fi
    "${PYTHON}" "${REPO_ROOT}/scripts/agent/check_external_models.py" \
      --repo-root "${REPO_ROOT}" \
      --local-config "${config_path}" \
      --emit-bootstrap-plan
  else
    default_plan
  fi
}

resolve_target() {
  local raw="$1"
  realpath -m "${raw}"
}

check_link() {
  local name="$1"
  local expected="$2"
  local handle="${REPO_ROOT}/model/${name}"
  local expected_resolved
  expected_resolved="$(resolve_target "${expected}")"

  if [[ ! -L "${handle}" ]]; then
    echo "[error] model/${name} is missing or is not a symlink: ${handle}" >&2
    return 1
  fi

  local raw_target
  raw_target="$(readlink "${handle}")"
  local actual_resolved
  if [[ "${raw_target}" == /* ]]; then
    actual_resolved="$(resolve_target "${raw_target}")"
  else
    actual_resolved="$(cd "$(dirname "${handle}")" && resolve_target "${raw_target}")"
  fi

  if [[ "${actual_resolved}" != "${expected_resolved}" ]]; then
    echo "[error] model/${name} points to ${actual_resolved}; expected ${expected_resolved}" >&2
    return 1
  fi
  if [[ ! -e "${expected_resolved}" ]]; then
    echo "[error] model/${name} target is missing: ${expected_resolved}" >&2
    return 1
  fi
  if ! git -C "${expected_resolved}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "[warn] model/${name} target exists but is not a git checkout: ${expected_resolved}" >&2
  fi

  echo "[ok] model/${name} -> ${expected_resolved}"
}

clone_or_report() {
  local name="$1"
  local url="$2"
  local dst="$3"
  dst="$(resolve_target "${dst}")"

  if [[ "${CHECK_ONLY}" -eq 1 ]]; then
    check_link "${name}" "${dst}"
    return
  fi

  mkdir -p "$(dirname "${dst}")" "${REPO_ROOT}/model"

  if [[ -e "${dst}" ]]; then
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

mkdir -p "${REPO_ROOT}/model"
mapfile -t PLAN_ROWS < <(bootstrap_plan)
if [[ "${#PLAN_ROWS[@]}" -eq 0 ]]; then
  echo "[error] empty external model bootstrap plan" >&2
  exit 2
fi
for row in "${PLAN_ROWS[@]}"; do
  IFS=$'\t' read -r name url target <<<"${row}"
  [[ -z "${name}" ]] && continue
  clone_or_report "${name}" "${url}" "${target}"
done

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
  echo "[done] external model links validated"
else
  echo "[done] model links:"
  ls -la "${REPO_ROOT}/model"
fi
