#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
CONFIG_FILE="${1:?Usage: run_dflash_distribution_colab.sh CONFIG.json}"
SESSION="${COLAB_SESSION:-vdflash-distribution}"
GPU="${COLAB_GPU:-A100}"
CREATE_SESSION="${COLAB_CREATE_SESSION:-1}"
MOUNT_DRIVE="${COLAB_MOUNT_DRIVE:-1}"
HIGH_MEM="${COLAB_HIGH_MEM:-1}"
TIMEOUT="${COLAB_TIMEOUT:-3600}"

if ! command -v colab >/dev/null 2>&1; then
  echo "Google Colab CLI is not installed or not on PATH." >&2
  echo "Install it with: uv tool install google-colab-cli" >&2
  exit 2
fi
if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Run config does not exist: ${CONFIG_FILE}" >&2
  exit 2
fi

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/vdflash-colab-distribution.XXXXXX")"
trap 'rm -rf "${TMP_ROOT}"' EXIT
SOURCE_ARCHIVE="${TMP_ROOT}/vdflash_distribution_source.tar.gz"

tar -czf "${SOURCE_ARCHIVE}" \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  --exclude='*/tests' \
  -C "${REPO_ROOT}" \
  src configs/dflash_configs scripts/requirements_colab_dflash_distribution.txt

if [[ "${CREATE_SESSION}" == "1" ]]; then
  CREATE_ARGS=(colab new -s "${SESSION}" --gpu "${GPU}")
  if [[ "${HIGH_MEM}" == "1" ]]; then
    CREATE_ARGS+=(--high-mem)
  fi
  "${CREATE_ARGS[@]}"
fi

if [[ "${MOUNT_DRIVE}" == "1" ]]; then
  colab drivemount -s "${SESSION}" /content/drive
fi

colab upload -s "${SESSION}" "${SOURCE_ARCHIVE}" /content/vdflash_distribution_source.tar.gz
colab upload -s "${SESSION}" "${CONFIG_FILE}" /content/dflash_distribution_config.json
colab exec -s "${SESSION}" --timeout "${TIMEOUT}" -f \
  "${REPO_ROOT}/scripts/colab_dflash_distribution_remote.py"
