#!/usr/bin/env bash

# Resumable Phase-1 H3.1 collection.  The MSD runner itself resumes complete
# sample/retention jobs from the JSONL; this outer loop also restarts it after
# an unexpected process termination until all 50 samples are recorded.

set -u

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_PATH="${ROOT_DIR}/results/phase1_raw_20260923/h31/msd_retention_vdc50_25k_final.jsonl"
EXPECTED_ROWS=200

cd -- "${ROOT_DIR}" || exit 1

while true; do
    if [[ -f "${OUTPUT_PATH}" ]]; then
        row_count="$(wc -l < "${OUTPUT_PATH}")"
    else
        row_count=0
    fi

    if (( row_count >= EXPECTED_ROWS )); then
        echo "H3.1 complete: ${row_count} rows"
        exit 0
    fi

    echo "Starting/resuming H3.1: ${row_count}/${EXPECTED_ROWS} rows"
    HF_HOME=/path/to/local-resource \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    MSD_VISION_CHUNK_FRAMES=4 \
    MSD_MODEL_PARALLEL_VISUAL_DEVICE=1 \
    MSD_MODEL_PARALLEL_SPLIT_LAYER=26 \
    .venv-msd/bin/python -u -m src.analyze.Validate_Sparrow_hypothesises.run_msd \
        --base-model /path/to/local-resource \
        --msd-model /path/to/local-resource \
        --manifest dataset/VideoDetailCaption/subset_manifest.jsonl \
        --dataset-root dataset/VideoDetailCaption \
        --calibration dataset/VideoDetailCaption/calibration_complete_20260823_currentenv.jsonl \
        --visual-targets 25000 \
        --condition retention \
        --retention-percentages 100 50 25 0 \
        --selection uniform \
        --allow-out-of-tolerance \
        --device-map model_parallel \
        --max-memory 0:22GiB,1:14GiB \
        --max-new-tokens 512 \
        --output "${OUTPUT_PATH}"
    runner_status=$?

    if [[ -f "${OUTPUT_PATH}" ]]; then
        row_count="$(wc -l < "${OUTPUT_PATH}")"
    else
        row_count=0
    fi
    if (( row_count >= EXPECTED_ROWS )); then
        echo "H3.1 complete: ${row_count} rows"
        exit 0
    fi

    echo "MSD runner stopped with status ${runner_status}; ${row_count}/${EXPECTED_ROWS} rows remain. Retrying after 10 seconds."
    sleep 10
done
