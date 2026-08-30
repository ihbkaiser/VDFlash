#!/usr/bin/env bash
# Source this file from a repository launcher.
#
# The resolver is intentionally independent of the caller's current directory
# and PATH.  It derives the repository from this file's location, or accepts
# VD_FLASH_WORKSPACE when a mounted/shared workspace needs to be selected.
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "resolve_workspace.sh must be sourced, not executed" >&2
    exit 2
fi

_VD_FLASH_RESOLVER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${VD_FLASH_WORKSPACE:-}" ]]; then
    if [[ ! -d "${VD_FLASH_WORKSPACE}" ]]; then
        echo "VD_FLASH_WORKSPACE is not a directory: ${VD_FLASH_WORKSPACE}" >&2
        return 1 2>/dev/null || exit 1
    fi
    REPO_ROOT="$(cd -- "${VD_FLASH_WORKSPACE}" && pwd)"
else
    if REPO_ROOT_FROM_GIT="$(git -C "${_VD_FLASH_RESOLVER_DIR}" rev-parse --show-toplevel 2>/dev/null)"; then
        REPO_ROOT="$(cd -- "${REPO_ROOT_FROM_GIT}" && pwd)"
    else
        REPO_ROOT="$(cd -- "${_VD_FLASH_RESOLVER_DIR}/.." && pwd)"
    fi
fi

if [[ ! -d "${REPO_ROOT}/src" || ! -d "${REPO_ROOT}/scripts" ]]; then
    echo "Unable to identify VDFlash workspace root: ${REPO_ROOT}" >&2
    return 1 2>/dev/null || exit 1
fi

VENV_ROOT="${REPO_ROOT}/.venv"
PYTHON_BIN="${VENV_ROOT}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Required workspace environment is missing: ${PYTHON_BIN}" >&2
    echo "Set VD_FLASH_WORKSPACE to the checkout that contains .venv, or mount that workspace." >&2
    return 1 2>/dev/null || exit 1
fi

export REPO_ROOT
export VENV_ROOT
export PYTHON_BIN
export PYTHON="${PYTHON_BIN}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.matplotlib}"

# Keep CUDA runtime wheels in the single workspace environment discoverable.
# The glob is intentionally Python-version agnostic because the environment is
# supplied by the host and may be Python 3.10, 3.11, or 3.12.
for VD_FLASH_CUDA_PACKAGE in cusparse nvjitlink; do
    for VD_FLASH_CUDA_LIB in "${VENV_ROOT}"/lib/python*/site-packages/nvidia/${VD_FLASH_CUDA_PACKAGE}/lib; do
        if [[ -d "${VD_FLASH_CUDA_LIB}" ]]; then
            export LD_LIBRARY_PATH="${VD_FLASH_CUDA_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        fi
    done
done

unset _VD_FLASH_RESOLVER_DIR REPO_ROOT_FROM_GIT VD_FLASH_CUDA_PACKAGE VD_FLASH_CUDA_LIB
