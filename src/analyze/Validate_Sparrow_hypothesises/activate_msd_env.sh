#!/usr/bin/env bash
# Source this compatibility file before running the MSD/Sparrow commands.
# The historical filename is retained, but all code now uses the single
# repository environment resolved by scripts/resolve_workspace.sh.

MSD_ACTIVATION_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${MSD_ACTIVATION_DIR}/../../../scripts/resolve_workspace.sh"
MSD_PROJECT_ROOT="${REPO_ROOT}"
MSD_VENV="${VENV_ROOT}"

export PATH="${VENV_ROOT}/bin:${PATH}"

echo "Workspace environment active: ${MSD_VENV}"
echo "HF_HOME=${HF_HOME}"
