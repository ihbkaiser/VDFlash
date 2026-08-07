from __future__ import annotations

import argparse
from dataclasses import fields
from typing import Any

from .config import DFlashTrainConfig


def _csv_ints(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integer layer IDs") from exc
    if not result:
        raise argparse.ArgumentTypeError("selected target layer list cannot be empty")
    return result


def add_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    """Add config-backed overrides shared by all three pipeline entrypoints."""

    parser.add_argument("--config", default="", help="Optional JSON/YAML pipeline config")
    parser.add_argument("--target-model", default=None)
    parser.add_argument("--target-revision", default=None)
    parser.add_argument("--stage", choices=["text", "multimodal"], default=None)
    parser.add_argument("--dataset-repo", default=None)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--image-archive", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--prepared-manifest", default=None)
    parser.add_argument("--teacher-cache-dir", default=None)
    parser.add_argument("--cache-shard-size", type=int, default=None)
    parser.add_argument("--image-min-pixels", type=int, default=None)
    parser.add_argument("--image-max-pixels", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", dest="learning_rate", type=float, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--dtype", dest="mixed_precision", choices=["no", "bf16", "fp16"], default=None)
    parser.add_argument("--max-seq-len", dest="max_seq_length", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--num-anchors", type=int, default=None)
    parser.add_argument("--anchor-chunk-size", type=int, default=None)
    parser.add_argument("--min-anchor-chunk-size", type=int, default=None)
    parser.add_argument("--num-draft-layers", type=int, default=None)
    parser.add_argument("--num-target-features", type=int, default=None)
    parser.add_argument("--selected-target-layers", type=_csv_ints, default=None)
    parser.add_argument("--context-mode", choices=["full", "text_only"], default=None)
    parser.add_argument(
        "--teacher-response-mode",
        choices=["target_generate", "dataset"],
        default=None,
        help="Use raw-greedy target responses (paper-aligned) or source-dataset responses",
    )
    parser.add_argument("--response-max-new-tokens", type=int, default=None)
    parser.add_argument("--teacher-require-eos", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint", default=None, help="Weights-only initialization checkpoint")
    parser.add_argument(
        "--resume",
        "--resume-from-checkpoint",
        dest="resume_from_checkpoint",
        default=None,
        help="Checkpoint containing optimizer/scheduler/progress state",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--eval-cache-samples", type=int, default=None)
    parser.add_argument("--save-every-steps", type=int, default=None)
    parser.add_argument("--keep-last-checkpoints", type=int, default=None)
    parser.add_argument(
        "--selective-image-download",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-flex-attention", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--compile-flex-attention", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--require-loss-decrease", action=argparse.BooleanOptionalAction, default=None)


def config_from_args(args: argparse.Namespace) -> DFlashTrainConfig:
    if args.config:
        base = DFlashTrainConfig.from_file(args.config).to_dict()
    else:
        base = DFlashTrainConfig().to_dict()
    valid = {field.name for field in fields(DFlashTrainConfig)}
    for key, value in vars(args).items():
        if key in valid and value is not None:
            base[key] = value
    if args.resume_from_checkpoint is not None and args.checkpoint is None:
        base["checkpoint"] = ""
    if args.checkpoint is not None and args.resume_from_checkpoint is None:
        base["resume_from_checkpoint"] = ""
    # If only --stage changes the default config, choose that stage's official
    # repository unless the caller explicitly supplied --dataset-repo.
    if args.stage is not None and args.dataset_repo is None and not args.config:
        base["dataset_repo"] = ""
    return DFlashTrainConfig(**base)


def config_summary(config: DFlashTrainConfig, keys: list[str]) -> dict[str, Any]:
    values = config.to_dict()
    return {key: values[key] for key in keys}
