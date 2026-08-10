"""Environment and data checks executed before any GPU inference."""

from __future__ import annotations

import importlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

from .dataset import load_vdc_manifest
from .paper_contract import PaperContract, validate_contract


def _module_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except ImportError:
        return None
    return str(getattr(module, "__version__", "installed"))


def run_preflight(
    contract: PaperContract,
    repo_root: str | Path,
    require_gpu: bool = False,
    require_models: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root)
    checks: list[dict[str, Any]] = []

    def add(name: str, status: str, detail: str, **extra: Any) -> None:
        checks.append({"name": name, "status": status, "detail": detail, **extra})

    contract_errors = validate_contract(contract)
    add("paper_contract", "ok" if not contract_errors else "error", "; ".join(contract_errors) or "valid")
    pdf = root / contract.paper_pdf
    add("paper_pdf", "ok" if pdf.exists() else "error", str(pdf))
    manifest = root / contract.dataset
    if manifest.exists():
        try:
            samples = load_vdc_manifest(manifest, manifest.parent)
            add("vdc_manifest", "ok", f"{len(samples)} samples", sample_count=len(samples))
        except Exception as exc:  # pragma: no cover - exact exception is part of CLI output
            add("vdc_manifest", "error", str(exc))
    else:
        add("vdc_manifest", "error", f"missing: {manifest}")

    add("python", "ok" if sys.version_info[:2] == (3, 10) else "warning", platform.python_version())
    torch_version = _module_version("torch")
    if torch_version is None:
        add("torch", "error", "PyTorch is not installed")
    else:
        try:
            import torch

            cuda = bool(torch.cuda.is_available())
            detail = f"torch={torch_version}, cuda={cuda}"
            add("torch_cuda", "ok" if cuda or not require_gpu else "error", detail, cuda_available=cuda)
            if cuda:
                add("gpu", "ok", torch.cuda.get_device_name(0), device_count=torch.cuda.device_count())
        except Exception as exc:  # pragma: no cover
            add("torch_cuda", "error", str(exc))

    for module_name in ("transformers", "accelerate", "av", "matplotlib", "qwen_vl_utils"):
        version = _module_version(module_name)
        add(module_name, "ok" if version else "error", version or "not installed", version=version)
    bitsandbytes = _module_version("bitsandbytes")
    add("bitsandbytes", "ok" if bitsandbytes else ("error" if require_gpu else "warning"), bitsandbytes or "not installed")

    if require_models:
        add(
            "msd_weights",
            "manual",
            "Set a local checkpoint path or authenticate Hugging Face before the GPU run",
            model=contract.msd_weights,
        )
        add(
            "layer_model",
            "manual",
            "Set a local checkpoint path or authenticate Hugging Face before the GPU run",
            model=contract.layer_target_model,
        )

    errors = [check for check in checks if check["status"] == "error"]
    return {
        "valid": not errors,
        "repo_root": str(root),
        "python": platform.python_version(),
        "checks": checks,
        "errors": errors,
    }


def write_preflight(path: str | Path, result: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
