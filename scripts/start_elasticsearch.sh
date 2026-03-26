#!/usr/bin/env bash

set -euo pipefail

VERSION="${1:-9.3.1}"
ELASTIC_URL="${ELASTIC_URL:-http://127.0.0.1:9200}"
HEAP_MB="${HEAP_MB:-1024}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CACHE_ROOT="${REPO_ROOT}/.cache/elasticsearch"
INSTALL_ROOT="${CACHE_ROOT}/install"
DATA_ROOT="${CACHE_ROOT}/data"
LOGS_ROOT="${CACHE_ROOT}/logs"
PID_PATH="${CACHE_ROOT}/elasticsearch.pid"

mkdir -p "${INSTALL_ROOT}" "${DATA_ROOT}" "${LOGS_ROOT}"

if curl --silent --fail --max-time 2 "${ELASTIC_URL}" >/dev/null 2>&1; then
  echo "Elasticsearch already responding at ${ELASTIC_URL}"
  exit 0
fi

uname_s="$(uname -s)"
uname_m="$(uname -m)"

case "${uname_s}" in
  Darwin)
    platform="darwin"
    archive_ext="tar.gz"
    ;;
  Linux)
    platform="linux"
    archive_ext="tar.gz"
    ;;
  *)
    echo "Unsupported operating system: ${uname_s}" >&2
    exit 1
    ;;
esac

case "${uname_m}" in
  arm64|aarch64)
    arch="aarch64"
    ;;
  x86_64)
    arch="x86_64"
    ;;
  *)
    echo "Unsupported CPU architecture: ${uname_m}" >&2
    exit 1
    ;;
esac

archive_name="elasticsearch-${VERSION}-${platform}-${arch}.${archive_ext}"
download_url="https://artifacts.elastic.co/downloads/elasticsearch/${archive_name}"
archive_path="${CACHE_ROOT}/${archive_name}"
es_home="${INSTALL_ROOT}/elasticsearch-${VERSION}"
stdout_path="${LOGS_ROOT}/stdout.log"
stderr_path="${LOGS_ROOT}/stderr.log"

if [[ ! -d "${es_home}" ]]; then
  if [[ ! -f "${archive_path}" ]]; then
    echo "Downloading Elasticsearch ${VERSION} from ${download_url}"
    curl -L --fail --retry 8 --retry-all-errors --output "${archive_path}" "${download_url}"
  fi
  echo "Extracting Elasticsearch to ${INSTALL_ROOT}"
  tar -xzf "${archive_path}" -C "${INSTALL_ROOT}"
fi

export ES_JAVA_OPTS="-Xms${HEAP_MB}m -Xmx${HEAP_MB}m"

nohup "${es_home}/bin/elasticsearch" \
  -Ediscovery.type=single-node \
  -Expack.security.enabled=false \
  -Ecluster.routing.allocation.disk.threshold_enabled=false \
  -Ehttp.host=127.0.0.1 \
  -Etransport.host=127.0.0.1 \
  -Epath.data="${DATA_ROOT}" \
  -Epath.logs="${LOGS_ROOT}" \
  >"${stdout_path}" 2>"${stderr_path}" &

es_pid=$!
echo "${es_pid}" > "${PID_PATH}"

ready=0
for _ in $(seq 1 60); do
  sleep 2
  if curl --silent --fail --max-time 3 "${ELASTIC_URL}" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "${es_pid}" >/dev/null 2>&1; then
    break
  fi
done

if [[ "${ready}" -ne 1 ]]; then
  if ! kill -0 "${es_pid}" >/dev/null 2>&1; then
    echo "Elasticsearch exited during startup. Check ${stdout_path} and ${stderr_path}" >&2
  else
    echo "Elasticsearch did not become ready at ${ELASTIC_URL}. Check ${stdout_path} and ${stderr_path}" >&2
  fi
  exit 1
fi

echo "Elasticsearch started. PID=${es_pid} URL=${ELASTIC_URL}"
