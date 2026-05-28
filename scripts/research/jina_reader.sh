#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/research/jina_reader.sh read <url>
  scripts/research/jina_reader.sh search <query...>

Examples:
  scripts/research/jina_reader.sh read https://example.com/article
  scripts/research/jina_reader.sh search ECG foundation model domain adaptation
EOF
}

urlencode() {
  /home/linbinhao/micromamba/envs/ECGTwin/bin/python - "$1" <<'PY'
import sys
from urllib.parse import quote_plus

print(quote_plus(sys.argv[1]))
PY
}

if [[ $# -lt 2 ]]; then
  usage >&2
  exit 2
fi

mode="$1"
shift

case "$mode" in
  read)
    url="$1"
    exec curl -fsSL "https://r.jina.ai/${url}"
    ;;
  search)
    query="$*"
    encoded="$(urlencode "$query")"
    exec curl -fsSL "https://s.jina.ai/?q=${encoded}"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
