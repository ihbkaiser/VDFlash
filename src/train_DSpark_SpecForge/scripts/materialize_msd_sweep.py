#!/usr/bin/env python3
"""Materialize immutable per-depth MSD JSON/YAML run configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def _write_if_same_or_new(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(
                f"refusing to overwrite a different generated config: {path}"
            )
        return
    path.write_text(content, encoding="utf-8")


def materialize_run(
    *,
    base_draft: Path,
    base_recipe: Path,
    generated_root: Path,
    output_root: Path,
    depth: int,
    target_model_path: str,
    text_feature_root: str,
    visual_feature_root: str,
    learning_rate: float,
    micro_batch_size: int,
    accumulation_steps: int,
    gpu_count: int,
) -> tuple[Path, Path]:
    """Create one isolated depth run without mutating shared feature roots."""

    if depth not in {1, 3, 5}:
        raise ValueError("MSD depth must be 1, 3, or 5")
    required_paths = {
        "target_model_path": target_model_path,
        "text_feature_root": text_feature_root,
        "visual_feature_root": visual_feature_root,
    }
    missing = [name for name, value in required_paths.items() if not str(value).strip()]
    if missing:
        raise ValueError(f"MSD run paths must not be empty: {missing}")
    if micro_batch_size < 1 or accumulation_steps < 1 or gpu_count < 1:
        raise ValueError("batch sizes and gpu_count must be positive")

    draft: dict[str, Any] = json.loads(base_draft.read_text(encoding="utf-8"))
    recipe: dict[str, Any] = yaml.safe_load(base_recipe.read_text(encoding="utf-8"))
    draft["num_hidden_layers"] = depth
    draft["architectures"] = ["MSDDraftModel"]

    depth_root = Path(output_root) / f"depth{depth}"
    draft_path = Path(generated_root) / f"depth{depth}" / "draft_config.json"
    recipe_path = Path(generated_root) / f"depth{depth}" / "train.yaml"
    recipe["model"].update(
        target_model_path=target_model_path,
        draft_model_config=str(draft_path),
        draft_num_hidden_layers=depth,
    )
    recipe["data"].update(
        hidden_states_path=text_feature_root,
        msd_visual_hidden_states_path=visual_feature_root,
    )
    recipe["training"].update(
        num_epochs=40,
        msd_total_epochs=40,
        learning_rate=float(learning_rate),
        batch_size=micro_batch_size,
        accumulation_steps=accumulation_steps,
    )
    recipe["deployment"]["trainer"]["nproc_per_node"] = gpu_count
    recipe["run_id"] = f"qwen25vl-3b-msd-depth{depth}"
    recipe["output_dir"] = str(depth_root / "output")

    draft_text = json.dumps(draft, indent=2, sort_keys=True) + "\n"
    recipe_text = yaml.safe_dump(recipe, sort_keys=False)
    _write_if_same_or_new(draft_path, draft_text)
    _write_if_same_or_new(recipe_path, recipe_text)
    return draft_path, recipe_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-draft", type=Path, required=True)
    parser.add_argument("--base-recipe", type=Path, required=True)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--target-model-path", required=True)
    parser.add_argument("--text-feature-root", required=True)
    parser.add_argument("--visual-feature-root", required=True)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--gpu-count", type=int, default=4)
    args = parser.parse_args()
    draft, recipe = materialize_run(**vars(args))
    print(f"DRAFT_CONFIG={draft}")
    print(f"TRAIN_CONFIG={recipe}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
