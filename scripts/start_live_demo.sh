#!/usr/bin/env bash

set -euo pipefail

HOST_ADDRESS="${HOST_ADDRESS:-127.0.0.1}"
PORT="${PORT:-4173}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
VENV_PYTHON="${VENV_DIR}/bin/python"
ENV_FILE="${REPO_ROOT}/.env"
ACTIVE_INDEX_POINTER="${REPO_ROOT}/datasets/index/active_index.json"

resolve_bootstrap_python() {
  if [[ -x "${VENV_PYTHON}" ]]; then
    echo "${VENV_PYTHON}"
    return
  fi

  local candidates=(python3.11 python3 python)
  local candidate
  for candidate in "${candidates[@]}"; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      "${candidate}" -c 'import sys; print(sys.executable)'
      return
    fi
  done

  echo "No usable Python 3.11+ interpreter was found. Install Python 3.11 or newer, then rerun this script." >&2
  exit 1
}

ensure_venv() {
  local bootstrap_python
  bootstrap_python="$(resolve_bootstrap_python)"
  if [[ ! -x "${VENV_PYTHON}" ]]; then
    echo "Creating repo-local virtual environment at ${VENV_DIR}"
    "${bootstrap_python}" -m venv "${VENV_DIR}"
  fi
}

ensure_editable_install() {
  if ! "${VENV_PYTHON}" -c 'import bgrag' >/dev/null 2>&1; then
    echo "Installing repo dependencies into .venv"
    (
      cd "${REPO_ROOT}"
      "${VENV_PYTHON}" -m pip install -e '.[dev]'
    )
  fi
}

ensure_env_file() {
  if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Missing .env file at ${ENV_FILE}. Add COHERE_API_KEY before starting the live demo." >&2
    exit 1
  fi
  if ! grep -Eq '^COHERE_API_KEY=\S+' "${ENV_FILE}"; then
    echo "COHERE_API_KEY is missing or empty in ${ENV_FILE}." >&2
    exit 1
  fi
}

ensure_elasticsearch() {
  if curl --silent --fail --max-time 2 http://127.0.0.1:9200 >/dev/null 2>&1; then
    echo "Elasticsearch already responding at http://127.0.0.1:9200"
    return
  fi
  echo "Starting local Elasticsearch"
  "${SCRIPT_DIR}/start_elasticsearch.sh"
}

test_active_index_ready() {
  if [[ ! -f "${ACTIVE_INDEX_POINTER}" ]]; then
    return 1
  fi

  local namespace
  namespace="$("${VENV_PYTHON}" - <<'PY'
import json
from pathlib import Path
path = Path("datasets/index/active_index.json")
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print("")
else:
    print(str(payload.get("namespace", "")))
PY
)"

  if [[ -z "${namespace}" ]]; then
    return 1
  fi

  [[ -f "${REPO_ROOT}/datasets/index/${namespace}/index_manifest.json" ]] &&
    [[ -f "${REPO_ROOT}/datasets/index/${namespace}/chunk_embeddings.json" ]]
}

ensure_baseline_index() {
  if test_active_index_ready; then
    echo "Active baseline index already present"
    return
  fi

  echo "Building baseline index for the live demo"
  (
    cd "${REPO_ROOT}"
    "${VENV_PYTHON}" -c "from bgrag.demo_server import build_demo_settings; from bgrag.pipeline import run_build_index; import json; settings = build_demo_settings(); print(json.dumps(run_build_index(settings, 'baseline'), indent=2))"
  )
}

ensure_venv
ensure_editable_install
ensure_env_file
ensure_elasticsearch

(
  cd "${REPO_ROOT}"
  ensure_baseline_index
  echo "Launching PolicyAI live demo at http://${HOST_ADDRESS}:${PORT}"
  exec "${VENV_PYTHON}" -m bgrag.demo_server --host "${HOST_ADDRESS}" --port "${PORT}"
)
