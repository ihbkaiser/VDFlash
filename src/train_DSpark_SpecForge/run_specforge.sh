#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../scripts/resolve_workspace.sh"
cd "${SCRIPT_DIR}"

# SpecForge imports its package by the top-level name ``specforge``.  Keeping
# this checkout on PYTHONPATH makes the copied training tree runnable without
# installing it into the repository's existing VLM environment.
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
else
  export PYTHONPATH="${SCRIPT_DIR}"
fi

exec "${PYTHON_BIN}" -m specforge.cli "$@"
