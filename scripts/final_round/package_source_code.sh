#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${ECG_ADV_DATA_ROOT:-${HOME}/autodl-tmp}"
OUT_DIR="${SOURCE_OUT_DIR:-${DATA_ROOT}/thesis_archive_artifacts/source_code}"
STAMP="${SOURCE_PACKAGE_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
PACKAGE_NAME="${SOURCE_PACKAGE_NAME:-ECG_adv_source_${STAMP}.tar.gz}"
OUT_PATH="${OUT_DIR}/${PACKAGE_NAME}"
LIST_PATH="$(mktemp)"

cleanup() {
  rm -f "${LIST_PATH}"
}
trap cleanup EXIT

mkdir -p "${OUT_DIR}"

# Include tracked files plus untracked files that are not ignored. This lets a
# final local archive include newly added docs before they are committed, while
# still excluding .venv/, migrate_files/, caches, raw data, and large artifacts.
git -C "${ROOT}" ls-files -z --cached --others --exclude-standard > "${LIST_PATH}"

if [[ ! -s "${LIST_PATH}" ]]; then
  echo "[error] no source files selected for packaging" >&2
  exit 1
fi

tar -C "${ROOT}" \
  --null \
  --files-from "${LIST_PATH}" \
  --transform 's,^,ECG_adv/,' \
  -czf "${OUT_PATH}"

sha256sum "${OUT_PATH}" > "${OUT_PATH}.sha256"

echo "[package_source] wrote ${OUT_PATH}"
echo "[package_source] wrote ${OUT_PATH}.sha256"
