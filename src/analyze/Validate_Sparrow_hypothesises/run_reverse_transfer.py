"""Run the trained MSD draft-hidden reverse-transfer experiment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from .reverse_transfer import (
    EXPERIMENTS,
    ReverseTransferSample,
    deterministic_permutations,
    load_reverse_manifest,
    sample_to_json,
)
from .reverse_transfer_runtime import (
    MSDReverseTransferError,
    architecture_trace,
    load_msd_runtime,
    prepare_reverse_input,
    score_draft_candidates,
    target_prefill,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _row_key(sample_id: str, permutation_id: str, condition: str) -> str:
    return f"{sample_id}|{permutation_id}|{condition}"


def _variants(sample: ReverseTransferSample, args: argparse.Namespace):
    if args.permutations:
        return deterministic_permutations(sample, args.seed)
    return [("identity", sample)]


def _selected_conditions(args: argparse.Namespace) -> list[str]:
    conditions = list(args.conditions or EXPERIMENTS)
    if args.input_order:
        conditions = [condition for condition in conditions if EXPERIMENTS[condition][0] == args.input_order]
    if args.conditioning:
        conditions = [condition for condition in conditions if EXPERIMENTS[condition][1] == args.conditioning]
    if not conditions:
        raise SystemExit("the --input-order/--conditioning filters selected no E1-E6 conditions")
    return conditions


def _base_row(sample: ReverseTransferSample, condition: str, permutation_id: str, args: argparse.Namespace) -> dict[str, Any]:
    input_order, conditioning = EXPERIMENTS[condition]
    return {
        "status": "started",
        "dataset": args.dataset,
        "sample_id": sample.sample_id,
        "video_id": sample.video_id,
        "task_type": sample.task_type,
        "sample_fingerprint": sample.fingerprint(),
        "permutation_id": permutation_id,
        "condition": condition,
        "input_order": input_order,
        "msd_conditioning": conditioning,
        "question": sample.question,
        "options": list(sample.options),
        "ground_truth": sample.correct_label,
        "ground_truth_content": sample.correct_answer,
        "sample": sample_to_json(sample),
        "target_checkpoint": args.target_model,
        "msd_checkpoint": args.msd_checkpoint,
        "prediction_source": "MSD draft hidden -> frozen target LM head",
        "target_logits_used": False,
    }


def _run_one(
    model: Any,
    processor: Any,
    device: torch.device,
    sample: ReverseTransferSample,
    condition: str,
    permutation_id: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    row = _base_row(sample, condition, permutation_id, args)
    started = time.perf_counter()
    input_order, conditioning = EXPERIMENTS[condition]
    try:
        prepared = prepare_reverse_input(
            processor,
            model,
            sample,
            input_order=input_order,
            device=device,
            fps=args.fps,
            num_frames=args.num_frames,
            max_pixels=args.max_pixels,
        )
        row["position_metadata"] = prepared.position_metadata()
        position_metadata = prepared.position_metadata()
        if input_order == "video_text":
            if position_metadata["visual_last_position"] >= position_metadata["task_text_first_position"]:
                raise MSDReverseTransferError("Video->Text prompt did not place all video positions before task text")
        else:
            if position_metadata["task_text_last_position"] >= position_metadata["visual_first_position"]:
                raise MSDReverseTransferError("Text->Video prompt did not place task text before video positions")
        target_hidden = target_prefill(model, prepared)
        row["target_hidden_full_shape"] = list(target_hidden.shape)
        row["architecture_trace"] = architecture_trace(model, args.target_model, args.msd_checkpoint)
        metrics = score_draft_candidates(model, prepared, target_hidden, conditioning=conditioning)
        row.update(metrics)
        row["target_hidden_full_fingerprint"] = _fingerprint(target_hidden)
        row["no_future_answer_token"] = True
        row["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - keep one bad video from hiding coverage
        row.update({
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "error_traceback": traceback.format_exc(),
            "evidence_eligible": False,
        })
    row["runtime_seconds"] = time.perf_counter() - started
    if torch.cuda.is_available():
        row["peak_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        torch.cuda.empty_cache()
    if row["status"] == "ok":
        metadata = row.get("feature_mask_metadata", {})
        row["evidence_eligible"] = bool(
            metadata.get("target_logits_used") is False
            and metadata.get("draft_hidden_source") == "model.ea_layer.output_after_draft_decoder"
            and metadata.get("lm_head_source") == "model.base_model.lm_head(draft_hidden)"
            and metadata.get("future_answer_token_provided") is False
        )
        if not row["evidence_eligible"]:
            row["status"] = "gate_failed"
    return row


def _fingerprint(value: torch.Tensor) -> str:
    import hashlib

    return hashlib.sha256(value.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def _provenance() -> dict[str, Any]:
    versions: dict[str, str | None] = {"python": sys.version.split()[0], "torch": torch.__version__}
    for package in ("transformers", "qwen-vl-utils", "huggingface-hub", "accelerate"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    commit = os.environ.get("GIT_COMMIT")
    if not commit:
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).parents[3],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None
    dirty = os.environ.get("GIT_DIRTY")
    return {
        "software_versions": versions,
        "git_commit": commit or None,
        "git_dirty": None if dirty is None else dirty.lower() in {"1", "true", "yes"},
    }


def run(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise SystemExit("MSD reverse-transfer runner requires CUDA; run unit tests locally and use Modal for GPU")
    if not args.msd_checkpoint:
        raise SystemExit("--msd-checkpoint is required; refusing to substitute target-only or random draft logits")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "raw_results.jsonl"
    existing = _read_jsonl(raw_path) if args.resume else []
    completed = {
        _row_key(str(row.get("sample_id")), str(row.get("permutation_id")), str(row.get("condition")))
        for row in existing
        if row.get("status") in {"ok", "gate_failed", "error"}
    }
    samples = load_reverse_manifest(
        args.dataset_path,
        dataset=args.dataset,
        video_root=args.video_root,
        num_samples=args.num_samples,
        require_videos=True,
    )
    conditions = _selected_conditions(args)
    model, processor, device = load_msd_runtime(
        target_model=args.target_model,
        msd_checkpoint=args.msd_checkpoint,
        device_map=args.device_map,
        max_memory=args.max_memory,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    config = {
        "target_model": args.target_model,
        "msd_checkpoint": args.msd_checkpoint,
        "dataset": args.dataset,
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "video_root": str(Path(args.video_root).resolve()) if args.video_root else None,
        "conditions": conditions,
        "permutations": args.permutations,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "seed": args.seed,
        "device": str(device),
        "backend": "vendored MSD/EAGLE draft hidden states",
        "target_logits_used": False,
        **_provenance(),
    }
    (output / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    total = 0
    for sample in samples:
        for permutation_id, variant in _variants(sample, args):
            for condition in conditions:
                key = _row_key(variant.sample_id, permutation_id, condition)
                if key in completed:
                    continue
                total += 1
                print(f"[{total}] {key}", flush=True)
                _append_jsonl(raw_path, _run_one(model, processor, device, variant, condition, permutation_id, args))
                completed.add(key)
    if args.include_d4:
        diagnostic_path = output / "d4_video_only.jsonl"
        for sample in samples:
            diagnostic_done = False
            if args.resume and diagnostic_path.exists():
                diagnostic_done = any(
                    json.loads(line).get("sample_id") == sample.sample_id
                    for line in diagnostic_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            if diagnostic_done:
                continue
            row = _base_row(sample, "E3", "identity", args)
            row.update({"condition": "D4", "input_order": "video_only", "msd_conditioning": "full"})
            started = time.perf_counter()
            try:
                prepared = prepare_reverse_input(processor, model, sample, input_order="video_text", device=device, fps=args.fps, num_frames=args.num_frames, max_pixels=args.max_pixels, include_text=False)
                target_hidden = target_prefill(model, prepared)
                row.update(score_draft_candidates(model, prepared, target_hidden, conditioning="full"))
                row["status"] = "diagnostic_only"
            except Exception as exc:  # noqa: BLE001
                row.update({"status": "error", "error_type": type(exc).__name__, "error": str(exc)})
            row["runtime_seconds"] = time.perf_counter() - started
            _append_jsonl(diagnostic_path, row)
    print(f"Wrote MSD rows to {raw_path}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--msd-checkpoint", default="lucylyn/MSD-Qwen2VL-7B-Instruct")
    parser.add_argument("--dataset", required=True, choices=("mvbench", "vidhalluc"))
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--conditions", nargs="+", choices=list(EXPERIMENTS), default=list(EXPERIMENTS))
    parser.add_argument("--input-order", choices=("video_text", "text_video"))
    parser.add_argument("--conditioning", choices=("full", "visual", "text"))
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--min-pixels", type=int, default=128 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--permutations", action="store_true")
    parser.add_argument("--include-d4", action="store_true")
    parser.add_argument("--device-map", choices=("cuda", "auto", "model_parallel"), default="cuda")
    parser.add_argument("--max-memory")
    return parser


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run(build_parser().parse_args()))
