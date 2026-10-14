"""Modal launcher for the independent Qwen2.5-VL 3B -> 7B VDC study."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import modal


MODULE_PATH = Path(__file__).resolve()
REPO_ROOT = next((parent for parent in MODULE_PATH.parents if (parent / "src").is_dir()), Path("/root/VDFlash"))
SRC_ROOT = REPO_ROOT / "src"
APP_NAME = "vdflash-standard-sd-vdc-retention"
HF_VOLUME_NAME = os.environ.get("MODAL_HF_VOLUME", "qresafe-awq-hf-cache")
DATA_VOLUME_NAME = os.environ.get("MODAL_VDC_VOLUME", "vdflash-vdc-subset-data")
RESULTS_VOLUME_NAME = os.environ.get("MODAL_RESULTS_VOLUME", "standard-sd-retention-results")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        "transformers==4.57.1",
        "huggingface_hub<1.0",
        "accelerate>=1.2",
        "bitsandbytes",
        "safetensors",
        "qwen-vl-utils[decord]",
        "pandas",
        "pyarrow",
        "matplotlib",
    )
    .add_local_dir(SRC_ROOT, "/root/VDFlash/src", copy=True)
    .env(
        {
            "PYTHONPATH": "/root/VDFlash",
            "HF_HOME": "/root/.cache/huggingface",
            "TRANSFORMERS_CACHE": "/root/.cache/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

hf_volume = modal.Volume.from_name(HF_VOLUME_NAME, create_if_missing=True)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="L40S",
    timeout=12 * 60 * 60,
    volumes={"/root/.cache/huggingface": hf_volume, "/data": data_volume},
)
def prepare_vdc_remote(subset_size: int = 10, seed: int = 42) -> int:
    command = [
        sys.executable,
        "-m",
        "src.analyze.Validate_Sparrow_hypothesises.prepare_vdc_subset_fast",
        "--output-dir",
        "/data/vdc_subset",
        "--subset-size",
        str(subset_size),
        "--seed",
        str(seed),
    ]
    completed = subprocess.run(command, cwd="/root/VDFlash", check=False)
    if completed.returncode:
        raise RuntimeError(f"VDC preparation failed: {completed.returncode}")
    data_volume.commit()
    return 0


@app.function(
    image=image,
    gpu="L40S",
    timeout=12 * 60 * 60,
    volumes={
        "/root/.cache/huggingface": hf_volume,
        "/data": data_volume,
        "/results": results_volume,
    },
)
def run_remote(argv: list[str]) -> int:
    command = [
        sys.executable,
        "-m",
        "src.analyze.Validate_Sparrow_hypothesises.standard_sd_retention",
        *argv,
    ]
    completed = subprocess.run(command, cwd="/root/VDFlash", check=False)
    if completed.returncode:
        raise RuntimeError(f"standard SD retention runner failed: {completed.returncode}")
    results_volume.commit()
    return 0


@app.local_entrypoint()
def main(args: str = "") -> None:
    argv = shlex.split(args)
    if not argv:
        raise SystemExit(
            "Pass runner flags with --args, e.g. "
            "--args '--manifest /data/vdc_subset/subset_manifest.jsonl "
            "--video-root /data/vdc_subset --output /results/vdc10/raw.jsonl --limit 10 --fresh'"
        )
    run_remote.remote(argv)
