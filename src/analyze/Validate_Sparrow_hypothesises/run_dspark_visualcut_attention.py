"""Run the shared VisualCut/attention campaign against DSpark checkpoints.

The paired inference and attention instrumentation are shared with the DFlash
runner. This adapter supplies DSpark checkpoint/config names and rejects a
checkpoint that materializes as a different draft architecture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.infer.qwen25vl_dflash_compare import (
    _load_draft as _inference_loader,
)

from . import run_dflash_visualcut_attention as _runner


CHECKPOINTS = (
    "dspark_3layer_26_28_30_32_34",
    "dspark_3layer_1_9_17_25_33",
)
CONFIGS = {
    "dspark_3layer_26_28_30_32_34": "high_only",
    "dspark_3layer_1_9_17_25_33": "balanced",
}
def _checkpoint_path(root: Path, checkpoint_name: str) -> Path:
    """Resolve the stripped Phase 2 state dict staged for Colab inference."""

    if checkpoint_name not in CONFIGS:
        raise ValueError(f"unknown DSpark checkpoint {checkpoint_name!r}")
    target_layers = checkpoint_name.removeprefix("dspark_3layer_")
    model_tag = (
        "qwen25vl-3b-dspark-llava68k-"
        f"dspark_targets_{target_layers}"
    )
    return (
        root
        / f"dspark_depth3_layers_{target_layers}"
        / f"dspark_targets_{target_layers}"
        / "phase2"
        / model_tag
        / "training_state.pt"
    )


def _load_dspark_draft(
    checkpoint: str | Path,
    config_path: str | Path,
    *,
    device,
    dtype,
) -> tuple[Any, float]:
    # Call the inference module's loader directly. The shared runner's module
    # attribute is replaced below with this adapter; resolving through that
    # attribute would recurse back into this function.
    draft, load_s = _inference_loader(
        checkpoint, config_path, device=device, dtype=dtype
    )
    if type(draft).__name__ != "DSparkDraftModel":
        raise TypeError(
            "DSpark checkpoint/config materialized as "
            f"{type(draft).__name__}, expected DSparkDraftModel"
        )
    return draft, load_s


_runner.CHECKPOINTS = CHECKPOINTS
_runner.CONFIGS = CONFIGS
_runner._checkpoint_path = _checkpoint_path
_runner._load_draft = _load_dspark_draft


def main(argv: list[str] | None = None) -> int:
    return _runner.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
