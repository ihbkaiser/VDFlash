#!/usr/bin/env python3
"""Prepare the pinned MSD visual-cut source/data/models inside Colab."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path


ROOT = Path("/content/vdflash")
SOURCE_ARCHIVE = Path("/content/vdflash_msd_colab_source.tar.gz")
DATA_ARCHIVE = Path("/content/vdflash_msd_colab_vdc25.tar.gz")
SOURCE_SHA256 = "86084cfc6f78d3a6c0de491d420cc635bc2b83403d922542806713cd6297602e"
DATA_SHA256 = "2c8f1ca846cae4a87bfcf175ac9d9096a032e91ff336248b004f2fe44d06b500"
DATA_PARTS = [Path(f"/content/vdflash_msd_vdc25_part_{index:03d}") for index in range(8)]
TARGET_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
TARGET_MODEL_REVISION = "eed13092ef92e448dd6875b2a00151bd3f7db0ac"
MSD_MODEL_ID = "lucylyn/MSD-Qwen2VL-7B-Instruct"
MSD_MODEL_REVISION = "c764d2ebd9b45d86dbf7960bffc918806c015234"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_and_extract(archive: Path, expected_hash: str) -> None:
    actual_hash = sha256(archive)
    if actual_hash != expected_hash:
        raise RuntimeError(f"SHA256 mismatch for {archive.name}: {actual_hash}")
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(ROOT)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    verify_and_extract(SOURCE_ARCHIVE, SOURCE_SHA256)

    if any(not part.is_file() for part in DATA_PARTS):
        missing = [str(part) for part in DATA_PARTS if not part.is_file()]
        raise FileNotFoundError(f"missing VDC transfer parts: {missing}")
    with DATA_ARCHIVE.open("wb") as output:
        for part in DATA_PARTS:
            with part.open("rb") as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    output.write(chunk)
    verify_and_extract(DATA_ARCHIVE, DATA_SHA256)

    environment = os.environ.copy()
    environment["HF_HOME"] = "/content/hf"
    environment["MSD_VISION_CHUNK_FRAMES"] = "8"
    os.environ.update(environment)
    packages = [
        "transformers==4.49.0",
        "accelerate==1.3.0",
        "bitsandbytes>=0.45.5",
        "qwen-vl-utils==0.0.14",
        "av>=14",
        "safetensors>=0.4.5",
    ]
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-input", *packages],
        check=True,
        env=environment,
    )

    from huggingface_hub import snapshot_download

    target_path = snapshot_download(
        repo_id=TARGET_MODEL_ID,
        revision=TARGET_MODEL_REVISION,
        cache_dir=environment["HF_HOME"],
    )
    msd_path = snapshot_download(
        repo_id=MSD_MODEL_ID,
        revision=MSD_MODEL_REVISION,
        cache_dir=environment["HF_HOME"],
    )

    import accelerate
    import bitsandbytes
    import qwen_vl_utils
    import torch
    import transformers

    if not torch.cuda.is_available():
        raise RuntimeError("Colab runtime has no visible CUDA device")
    if torch.cuda.get_device_name(0) != "NVIDIA L4":
        raise RuntimeError(f"expected Colab L4, found {torch.cuda.get_device_name(0)}")
    model_config = json.loads((Path(target_path) / "config.json").read_text())
    checkpoint_config = json.loads((Path(msd_path) / "config.json").read_text())
    if model_config.get("model_type") != "qwen2_vl":
        raise RuntimeError(f"unexpected target config: {model_config.get('model_type')}")
    if checkpoint_config.get("model_type") != "qwen2_vl":
        raise RuntimeError(f"unexpected MSD config: {checkpoint_config.get('model_type')}")

    report = {
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "qwen_vl_utils": getattr(qwen_vl_utils, "__version__", "unknown"),
        "target_model_id": TARGET_MODEL_ID,
        "target_model_revision": TARGET_MODEL_REVISION,
        "target_model_path": target_path,
        "target_model_type": model_config.get("model_type"),
        "target_layers": model_config.get("num_hidden_layers"),
        "msd_model_id": MSD_MODEL_ID,
        "msd_model_revision": MSD_MODEL_REVISION,
        "msd_model_path": msd_path,
        "msd_model_type": checkpoint_config.get("model_type"),
        "video_samples": len((ROOT / "dataset/VideoDetailCaption/subset_manifest.jsonl").read_text().splitlines()),
        "source_sha256": SOURCE_SHA256,
        "dataset_sha256": DATA_SHA256,
    }
    output_path = ROOT / "results/22_10_26/msd_visual_cut_colab_environment.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


main()
