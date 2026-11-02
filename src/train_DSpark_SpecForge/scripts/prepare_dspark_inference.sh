#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
ARTIFACT_ROOT="${DSPARK_ARTIFACT_ROOT:-$REPO_ROOT/artifacts/ihbkaiser_qwen25vl-3b-msd-checkpoints}"
ARCHIVE_PATH="${DSPARK_ARCHIVE_PATH:-$ARTIFACT_ROOT/dspark_3layer.zst}"
PARTIAL_PATH="${ARCHIVE_PATH}.partial"
EXTRACT_ROOT="${DSPARK_EXTRACT_ROOT:-$ARTIFACT_ROOT/extracted_dspark}"
ARCHIVE_URL="https://huggingface.co/datasets/ihbkaiser/qwen25vl-3b-msd-checkpoints/resolve/main/dspark_3layer.zst"
EXPECTED_SIZE=51534964941
EXPECTED_SHA256="52ea9887ec90c69fb2dc2f9381f0cb4e3ccd5e48b046043700680dc257ad0ec5"

for binary in curl sha256sum stat tar zstd; do
  if ! command -v "$binary" >/dev/null 2>&1; then
    echo "Required command not found: $binary" >&2
    exit 1
  fi
done

mkdir -p "$(dirname -- "$ARCHIVE_PATH")"

if [[ -e "$ARCHIVE_PATH" ]]; then
  actual_size="$(stat -c '%s' "$ARCHIVE_PATH")"
  if [[ "$actual_size" != "$EXPECTED_SIZE" ]]; then
    echo "Existing archive has wrong size: $actual_size (expected $EXPECTED_SIZE)" >&2
    exit 1
  fi
  actual_sha="$(sha256sum "$ARCHIVE_PATH" | cut -d ' ' -f 1)"
  if [[ "$actual_sha" != "$EXPECTED_SHA256" ]]; then
    echo "Existing archive SHA-256 does not match the Hugging Face LFS digest" >&2
    exit 1
  fi
else
  partial_size=0
  if [[ -f "$PARTIAL_PATH" ]]; then
    partial_size="$(stat -c '%s' "$PARTIAL_PATH")"
  fi
  if ((partial_size > EXPECTED_SIZE)); then
    echo "Partial archive is larger than expected: $partial_size bytes" >&2
    exit 1
  fi
  if ((partial_size < EXPECTED_SIZE)); then
    echo "Downloading $ARCHIVE_URL"
    echo "Partial download is resumable at $PARTIAL_PATH"
    curl --fail --location --retry 10 --retry-delay 5 --continue-at - \
      --progress-bar "$ARCHIVE_URL" --output "$PARTIAL_PATH"
  fi
  actual_size="$(stat -c '%s' "$PARTIAL_PATH")"
  if [[ "$actual_size" != "$EXPECTED_SIZE" ]]; then
    echo "Downloaded archive has wrong size: $actual_size (expected $EXPECTED_SIZE)" >&2
    exit 1
  fi
  actual_sha="$(sha256sum "$PARTIAL_PATH" | cut -d ' ' -f 1)"
  if [[ "$actual_sha" != "$EXPECTED_SHA256" ]]; then
    echo "Downloaded archive SHA-256 does not match the Hugging Face LFS digest" >&2
    exit 1
  fi
  mv -- "$PARTIAL_PATH" "$ARCHIVE_PATH"
  echo "Downloaded and verified $ARCHIVE_PATH"
fi

python_bin="${DSPARK_PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  echo "Python environment not found: $python_bin" >&2
  exit 1
fi

prepare_args=(
  --archive "$ARCHIVE_PATH"
  --output-root "$EXTRACT_ROOT"
  --phase "${DSPARK_PHASE:-auto}"
  --step "${DSPARK_STEP:-latest}"
  --run-id "${DSPARK_RUN_ID:-qwen25vl-3b-dspark-llava68k-dspark_targets_26_28_30_32_34}"
)
"$python_bin" "$SCRIPT_DIR/prepare_dspark_checkpoint.py" "${prepare_args[@]}"
