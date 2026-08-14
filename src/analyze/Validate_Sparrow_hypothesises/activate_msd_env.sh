#!/usr/bin/env bash
# Source this file before running the MSD/Sparrow commands.
# It exposes the repository venv and the CUDA runtime wheels used by
# bitsandbytes.  The latter is needed when the host has an NVIDIA driver but
# no system-wide CUDA 12 toolkit installation.

MSD_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MSD_VENV="${MSD_PROJECT_ROOT}/.venv-msd"
if [[ ! -x "${MSD_VENV}/bin/python" ]]; then
    echo "Missing environment: ${MSD_VENV}" >&2
    return 1 2>/dev/null || exit 1
fi

export PATH="${MSD_VENV}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-${MSD_PROJECT_ROOT}/.hf_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${MSD_PROJECT_ROOT}/.matplotlib}"

MSD_NVIDIA_ROOT="${MSD_VENV}/lib/python3.10/site-packages/nvidia"
MSD_CUDA_LIBS=(
    "${MSD_NVIDIA_ROOT}/cusparse/lib"
    "${MSD_NVIDIA_ROOT}/nvjitlink/lib"
)
for MSD_CUDA_LIB in "${MSD_CUDA_LIBS[@]}"; do
    if [[ -d "${MSD_CUDA_LIB}" ]]; then
        export LD_LIBRARY_PATH="${MSD_CUDA_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
done

echo "MSD environment active: ${MSD_VENV}"
echo "HF_HOME=${HF_HOME}"
