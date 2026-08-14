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
            device_count = int(torch.cuda.device_count()) if cuda else 0
            detail = f"torch={torch_version}, cuda={cuda}, devices={device_count}"
            add(
                "torch_cuda",
                "ok" if cuda or not require_gpu else "error",
                detail,
                cuda_available=cuda,
                device_count=device_count,
            )
            if cuda:
                names = [torch.cuda.get_device_name(index) for index in range(device_count)]
                add("gpu", "ok", "; ".join(names), device_count=device_count, devices=names)
        except Exception as exc:  # pragma: no cover
            add("torch_cuda", "error", str(exc))

    for module_name in ("transformers", "accelerate", "av", "matplotlib", "qwen_vl_utils"):
        version = _module_version(module_name)
        add(module_name, "ok" if version else "error", version or "not installed", version=version)

    try:
        import transformers

        qwen25_available = hasattr(transformers, "Qwen2_5_VLForConditionalGeneration")
        add(
            "qwen25_vl",
            "ok" if qwen25_available else "error",
            "Qwen2.5-VL Transformers integration available"
            if qwen25_available
            else "Qwen2.5-VL is unavailable; install Transformers 4.49.0 or newer",
        )
    except Exception as exc:  # pragma: no cover
        add("qwen25_vl", "error", str(exc))

    eagle_root = root / "externals" / "MSD" / "EAGLE"
    if eagle_root.exists():
        try:
            import sys as _sys

            if str(eagle_root) not in _sys.path:
                _sys.path.insert(0, str(eagle_root))
            import eagle.model.ea_model  # noqa: F401

            add("msd_eagle", "ok", "vendored MSD/EAGLE imports successfully")
        except Exception as exc:  # pragma: no cover
            add("msd_eagle", "error", str(exc))
    else:
        add("msd_eagle", "error", f"missing: {eagle_root}")

    bitsandbytes = _module_version("bitsandbytes")
    add("bitsandbytes", "ok" if bitsandbytes else ("error" if require_gpu else "warning"), bitsandbytes or "not installed")
    if bitsandbytes:
        try:
            from bitsandbytes.cextension import CudaBNBNativeLibrary, lib

            bnb_cuda = isinstance(lib, CudaBNBNativeLibrary)
            add(
                "bitsandbytes_cuda",
                "ok" if bnb_cuda or not require_gpu else "error",
                "CUDA backend loaded" if bnb_cuda else "CUDA backend unavailable",
                cuda_available=bnb_cuda,
            )
        except Exception as exc:  # pragma: no cover
            add("bitsandbytes_cuda", "error" if require_gpu else "warning", str(exc))

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
