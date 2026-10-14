"""Run the trained-MSD reverse-transfer experiment on Modal.

This launcher mounts model/data/results separately and copies both the
experiment package and the vendored MSD/EAGLE implementation into the image.
It never falls back to target-only logits.

Example::

    modal run src/analyze/Validate_Sparrow_hypothesises/modal_reverse_transfer.py --args "--dataset mvbench --dataset-path /data/.../manifest.jsonl --video-root /data/.../videos --num-samples 1 --output-dir /results/reverse_transfer/mvbench_smoke"
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import modal


MODULE_PATH = Path(__file__).resolve()
REPO_ROOT = next(
    (parent for parent in MODULE_PATH.parents if (parent / "src").is_dir()),
    Path("/root/VDFlash"),
)
SRC_ROOT = REPO_ROOT / "src"
MSD_ROOT = REPO_ROOT / "externals" / "MSD"
APP_NAME = "vdflash-msd-reverse-transfer"
HF_VOLUME_NAME = os.environ.get("MODAL_HF_VOLUME", "qresafe-awq-hf-cache")
DATA_VOLUME_NAME = os.environ.get(
    "MODAL_DATA_VOLUME", "visual-probing-sparrow-mvbench-hard-l4-data"
)
RESULTS_VOLUME_NAME = os.environ.get(
    "MODAL_RESULTS_VOLUME", "reverse-text-visual-results"
)


def _local_git_provenance() -> tuple[str, str]:
    commit = os.environ.get("GIT_COMMIT", "")
    dirty = os.environ.get("GIT_DIRTY", "")
    try:
        if not commit:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        if not dirty:
            dirty = "true" if subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip() else "false"
    except (OSError, subprocess.CalledProcessError):
        pass
    return commit, dirty


GIT_COMMIT, GIT_DIRTY = _local_git_provenance()


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
        "matplotlib",
    )
    .add_local_dir(SRC_ROOT, "/root/VDFlash/src", copy=True)
    .add_local_dir(MSD_ROOT, "/root/VDFlash/externals/MSD", copy=True)
    .env(
        {
            "PYTHONPATH": "/root/VDFlash",
            "HF_HOME": "/root/.cache/huggingface",
            "TRANSFORMERS_CACHE": "/root/.cache/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
            "GIT_COMMIT": GIT_COMMIT,
            "GIT_DIRTY": GIT_DIRTY,
        }
    )
)

hf_volume = modal.Volume.from_name(HF_VOLUME_NAME, create_if_missing=True)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=False)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)

app = modal.App(APP_NAME)


def _output_dir(argv: list[str]) -> str | None:
    try:
        index = argv.index("--output-dir")
    except ValueError:
        return None
    return argv[index + 1] if index + 1 < len(argv) else None


@app.function(
    image=image,
    gpu="L4",
    timeout=12 * 60 * 60,
    volumes={
        "/root/.cache/huggingface": hf_volume,
        "/data": data_volume,
        "/results": results_volume,
    },
)
def run_remote(argv: list[str]) -> int:
    """Run actual MSD draft scoring and build its report in the container."""

    command = [
        sys.executable,
        "-m",
        "src.analyze.Validate_Sparrow_hypothesises.reverse_transfer_cli",
        *argv,
    ]
    completed = subprocess.run(command, cwd="/root/VDFlash", check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"MSD reverse-transfer runner failed with {completed.returncode}")

    output_dir = _output_dir(argv)
    if output_dir:
        raw_results = Path(output_dir) / "raw_results.jsonl"
        if raw_results.exists():
            report = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.analyze.Validate_Sparrow_hypothesises.reverse_transfer_report",
                    "--input",
                    str(raw_results),
                    "--output-dir",
                    str(raw_results.parent / "report"),
                ],
                cwd="/root/VDFlash",
                check=False,
            )
            if report.returncode != 0:
                raise RuntimeError(f"MSD reverse-transfer report failed with {report.returncode}")
    results_volume.commit()
    return 0


@app.local_entrypoint()
def main(args: str = "") -> None:
    argv = shlex.split(args)
    if not argv:
        raise SystemExit(
            "Pass runner flags with --args, for example: "
            "--args '--dataset mvbench --dataset-path /data/.../manifest.jsonl "
            "--video-root /data/.../videos --num-samples 1 "
            "--output-dir /results/reverse_transfer/mvbench_smoke'"
        )
    run_remote.remote(argv)
