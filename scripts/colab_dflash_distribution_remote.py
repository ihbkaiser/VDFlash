"""Bootstrap the uploaded VDFlash source bundle and run the configured replay."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile


ROOT = Path("/content/vdflash")
SOURCE_ARCHIVE = Path("/content/vdflash_distribution_source.tar.gz")
CONFIG_PATH = Path("/content/dflash_distribution_config.json")


def _safe_extract(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            resolved = (destination / member.name).resolve()
            if resolved != root and root not in resolved.parents:
                raise RuntimeError(f"unsafe source archive path: {member.name}")
        archive.extractall(destination, members=members)


def _install_dependencies() -> None:
    requirements = ROOT / "scripts/requirements_colab_dflash_distribution.txt"
    marker = ROOT / ".colab_distribution_requirements_sha256"
    fingerprint = hashlib.sha256(requirements.read_bytes()).hexdigest()
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == fingerprint:
        print("[colab-bootstrap] dependency fingerprint already installed", flush=True)
        return
    print("[colab-bootstrap] installing pinned inference dependencies", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-input", "-r", str(requirements)],
        check=True,
    )
    marker.write_text(fingerprint + "\n", encoding="utf-8")


def _run_module(module: str, *arguments: str, env: dict[str, str]) -> None:
    command = [sys.executable, "-u", "-m", module, *arguments]
    print(f"[colab-bootstrap] command={' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    if not SOURCE_ARCHIVE.is_file():
        raise FileNotFoundError(SOURCE_ARCHIVE)
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(CONFIG_PATH)
    _safe_extract(SOURCE_ARCHIVE, ROOT)
    # The workspace path helper recognizes either a checkout's .git directory
    # or its vendored externals directory as a repository marker. The Colab
    # source bundle intentionally omits both, so provide the empty marker.
    (ROOT / "externals").mkdir(parents=True, exist_ok=True)
    _install_dependencies()

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT), str(ROOT / "src/train_Dflash_SpecForge"), env.get("PYTHONPATH", ""))
    ).rstrip(os.pathsep)
    env.setdefault("VD_FLASH_WORKSPACE", str(ROOT))
    env.setdefault("HF_HOME", "/content/hf")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")

    _run_module(
        "src.analyze.Validate_Sparrow_hypothesises.run_dflash_distribution_replay",
        "--config",
        str(CONFIG_PATH),
        env=env,
    )
    if config.get("analyze_after_run", False):
        _run_module(
            "src.analyze.Validate_Sparrow_hypothesises.analyze_dflash_distribution_replay",
            "--run-dir",
            str(Path(config["output_dir"])),
            "--bootstrap-reps",
            str(int(config.get("bootstrap_reps", 2000))),
            "--seed",
            str(int(config.get("bootstrap_seed", 20261102))),
            env=env,
        )


main()
