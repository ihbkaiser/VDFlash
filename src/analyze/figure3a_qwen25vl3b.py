"""Measure Sparrow Figure 3(a) with Qwen2.5-VL-3B on MVBench.

The intervention follows the paper's layer-wise visual-flow test: the native
model and prompt are held fixed, while visual-token key columns are masked in
all decoder layers from a selected cutoff onward.  The native no-mask output
is the baseline; layer cutoffs are the plotted variants.

The module keeps prompt construction, answer scoring, and aggregation free of
Torch so they can be tested on a CPU-only environment.  Model and video
processing imports are intentionally lazy and only occur in ``run``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import string
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_MANIFEST = "dataset/MVBench/classified/selected.jsonl"
DEFAULT_OUTPUT = "results/figure3a_qwen25vl3b.jsonl"
DEFAULT_TASKS = (
    "action_prediction",
    "action_sequence",
    "moving_attribute",
    "moving_direction",
    "object_interaction",
)
POST_PROMPT = "Only give the best option.\n"


def default_layer_cut_points(layer_count: int, stride: int = 4) -> list[int]:
    """Return paper-spaced visual-flow cutoffs, excluding native baseline."""

    if layer_count <= 0:
        raise ValueError(f"layer_count must be positive, got {layer_count}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    return list(range(0, layer_count, stride))


def build_mvbench_prompt(row: Mapping[str, Any], post_prompt: str = POST_PROMPT) -> str:
    """Build the canonical local MVBench multiple-choice prompt."""

    question = str(row["question"])
    candidates = list(row["candidates"])
    if not candidates:
        raise ValueError("MVBench record has no candidates")
    letters = string.ascii_uppercase
    if len(candidates) > len(letters):
        raise ValueError(f"MVBench record has too many candidates: {len(candidates)}")
    options = "".join(
        f"({letters[index]}) {candidate}\n"
        for index, candidate in enumerate(candidates)
    )
    return f"Question:{question}\nOption:\n{options}{post_prompt}"


def _option_for_answer(answer: Any, candidates: Sequence[Any]) -> str | None:
    for index, candidate in enumerate(candidates):
        # MVBench's reference scorer maps the answer by exact candidate
        # equality. Do not normalize here: that could collapse two distinct
        # candidates and would change the benchmark label.
        if candidate == answer:
            return string.ascii_uppercase[index]
    return None


def _option_for_prediction(prediction: Any) -> str | None:
    text = str(prediction).strip()
    # Match the same option-letter behavior as the local lmms-eval MVBench
    # scorer. In particular, candidate text such as ``blue`` is not promoted
    # to option B; the benchmark expects an option-letter prediction.
    match = re.match(r"^\s*([A-E])\.\s*.+$", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([A-E])\b", text, re.IGNORECASE)
    return match.group(1).upper() if match else None


def score_mvbench_prediction(row: Mapping[str, Any], prediction: str) -> dict[str, Any]:
    """Map a generated answer to an MVBench option and report exact accuracy."""

    candidates = list(row["candidates"])
    target_option = _option_for_answer(row["answer"], candidates)
    predicted_option = _option_for_prediction(prediction)
    return {
        "correct": bool(target_option is not None and target_option == predicted_option),
        "target_option": target_option,
        "predicted_option": predicted_option,
    }


def aggregate_figure3a_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-record results into task-by-cutoff Figure 3(a) points."""

    grouped: dict[tuple[str, int | None], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["task"]), row.get("layer_cut"))].append(row)

    def sort_key(item: tuple[str, int | None]) -> tuple[str, int]:
        task, cutoff = item
        return task, (10**9 if cutoff is None else int(cutoff))

    summary: list[dict[str, Any]] = []
    for (task, cutoff) in sorted(grouped, key=sort_key):
        group = grouped[(task, cutoff)]
        prefix_values = [float(row.get("prefix_agreement", 0.0)) for row in group]
        lossless_values = [
            bool(row.get("lossless", value >= 1.0))
            for row, value in zip(group, prefix_values)
        ]
        summary.append(
            {
                "task": task,
                "layer_cut": cutoff,
                "num_samples": len(group),
                "num_correct": sum(bool(row.get("correct", False)) for row in group),
                "accuracy": sum(bool(row.get("correct", False)) for row in group) / len(group),
                "mean_prefix_agreement": sum(prefix_values) / len(group),
                "lossless_rate": sum(lossless_values) / len(group),
            }
        )
    return summary


def plot_figure3a(summary: Iterable[Mapping[str, Any]], output: str | Path) -> None:
    """Render task accuracy versus visual-flow cutoff, with native baselines."""

    import matplotlib.pyplot as plt

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for point in summary:
        grouped[str(point["task"])].append(point)
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    for task in sorted(grouped):
        points = grouped[task]
        variants = [point for point in points if point.get("layer_cut") is not None]
        variants.sort(key=lambda point: int(point["layer_cut"]))
        if not variants:
            continue
        x_values = [int(point["layer_cut"]) for point in variants]
        y_values = [float(point["accuracy"]) for point in variants]
        (line,) = axis.plot(x_values, y_values, marker="o", label=task)
        baselines = [point for point in points if point.get("layer_cut") is None]
        if baselines:
            axis.axhline(
                float(baselines[0]["accuracy"]),
                color=line.get_color(),
                linestyle="--",
                alpha=0.35,
                label=f"{task} native",
            )
    axis.set_xlabel("Visual KV masked from decoder layer x")
    axis.set_ylabel("MVBench accuracy")
    axis.set_title("Sparrow Figure 3(a): Qwen2.5-VL-3B")
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.25)
    axis.legend(loc="best", fontsize="small")
    figure.tight_layout()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def read_mvbench_records(
    manifest: str | Path,
    *,
    tasks: Sequence[str] = DEFAULT_TASKS,
    limit_per_task: int | None = None,
) -> list[dict[str, Any]]:
    """Read deterministic task records and optionally cap each task equally."""

    requested = set(tasks)
    if not requested:
        raise ValueError("at least one task is required")
    rows_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with Path(manifest).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"manifest line {line_number} is not an object")
            task = str(row.get("task", ""))
            if task in requested:
                rows_by_task[task].append(row)

    selected: list[dict[str, Any]] = []
    for task in tasks:
        task_rows = rows_by_task.get(task, [])
        if limit_per_task is not None:
            if limit_per_task <= 0:
                raise ValueError("limit_per_task must be positive")
            task_rows = task_rows[:limit_per_task]
        selected.extend(task_rows)
    if not selected:
        raise ValueError(f"manifest has no records for requested tasks: {list(tasks)}")
    return selected


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_ids(values: Sequence[int]) -> str:
    payload = json.dumps([int(value) for value in values], separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    common = 0
    for expected, actual in zip(left, right):
        if int(expected) != int(actual):
            break
        common += 1
    return common


def _decode(processor: Any, tokens: Sequence[int]) -> str:
    try:
        return processor.batch_decode(
            [list(tokens)], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    except TypeError:
        return processor.batch_decode([list(tokens)], skip_special_tokens=True)[0]


def _layer_count(model: Any) -> int:
    candidates = [
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(model, "language_model", None),
        getattr(model, "model", None),
    ]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "layers"):
            return len(candidate.layers)
    config = getattr(model, "config", None)
    value = getattr(config, "num_hidden_layers", None)
    if value is not None:
        return int(value)
    raise RuntimeError("could not locate Qwen decoder layers")


def _batch_value(batch: Any, key: str) -> Any:
    return batch.get(key) if isinstance(batch, Mapping) else getattr(batch, key, None)


def _row_result(
    *,
    record: Mapping[str, Any],
    args: argparse.Namespace,
    layer_count: int,
    video_sha256: str,
    input_ids: Sequence[int],
    target_tokens: Sequence[int],
    candidate_tokens: Sequence[int],
    target_text: str,
    candidate_text: str,
    layer_cut: int | None,
    visual_token_count: int,
) -> dict[str, Any]:
    score = score_mvbench_prediction(record, candidate_text)
    common = _common_prefix_length(target_tokens, candidate_tokens)
    target_length = max(1, len(target_tokens))
    return {
        "row_id": f"{record['sample_id']}:layer-cut-{layer_cut if layer_cut is not None else 'native'}",
        "paper_figure": "Figure 3(a)",
        "condition": "native_no_mask" if layer_cut is None else "layer_visual_kv_ablation",
        "task": record["task"],
        "sample_id": record["sample_id"],
        "video": record["video"],
        "video_path": record["video_path"],
        "video_sha256": video_sha256,
        "target_model": args.model,
        "layer_count": layer_count,
        "layer_cut": layer_cut,
        "visual_kv_masked_from": layer_cut,
        "cutoff_convention": "mask visual key columns in decoder layers i >= layer_cut; native baseline has no mask",
        "prompt": build_mvbench_prompt(record),
        "fps": args.fps,
        "max_frames": args.max_frames,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.0,
        "visual_token_count": visual_token_count,
        "input_fingerprint": _sha256_ids(input_ids),
        "target_output_ids": [int(value) for value in target_tokens],
        "candidate_output_ids": [int(value) for value in candidate_tokens],
        "target_output_hash": _sha256_ids(target_tokens),
        "candidate_output_hash": _sha256_ids(candidate_tokens),
        "target_text": target_text,
        "candidate_text": candidate_text,
        "reference_answer": record["answer"],
        "target_option": score["target_option"],
        "predicted_option": score["predicted_option"],
        "correct": score["correct"],
        "prefix_length": common,
        "prefix_agreement": common / target_length,
        "lossless": list(target_tokens) == list(candidate_tokens),
    }


def _generate(model: Any, batch: Any, *, max_new_tokens: int) -> list[int]:
    import torch

    input_ids = _batch_value(batch, "input_ids")
    input_length = int(input_ids.shape[1])
    with torch.inference_mode():
        output = model.generate(
            **batch,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    return output[0, input_length:].detach().to("cpu").tolist()


def _run_record(
    model: Any,
    processor: Any,
    record: Mapping[str, Any],
    args: argparse.Namespace,
    layer_count: int,
    cutoffs: Sequence[int],
) -> list[dict[str, Any]]:
    from .Validate_Sparrow_hypothesises.model_analysis import mask_visual_keys
    from .Validate_Sparrow_hypothesises.runtime import move_batch_to_device, model_device, process_video

    path = Path(str(record["video_path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    video_sha256 = _sha256_file(path)
    batch = process_video(
        processor,
        path,
        build_mvbench_prompt(record),
        fps=args.fps,
        max_pixels=args.max_pixels,
        max_frames=args.max_frames,
    )
    batch = move_batch_to_device(batch, model_device(model))
    input_ids_tensor = _batch_value(batch, "input_ids")
    input_ids = input_ids_tensor[0].detach().to("cpu").tolist()
    video_token_id = int(getattr(model.config, "video_token_id"))
    visual_positions = (input_ids_tensor[0] == video_token_id).nonzero(as_tuple=False).flatten()
    visual_positions_list = visual_positions.detach().to("cpu").tolist()
    if not visual_positions_list:
        raise RuntimeError(
            f"no video-token positions found for {record['sample_id']}; refusing a no-op ablation"
        )

    target_tokens = _generate(model, batch, max_new_tokens=args.max_new_tokens)
    target_text = _decode(processor, target_tokens)
    rows = [
        _row_result(
            record=record,
            args=args,
            layer_count=layer_count,
            video_sha256=video_sha256,
            input_ids=input_ids,
            target_tokens=target_tokens,
            candidate_tokens=target_tokens,
            target_text=target_text,
            candidate_text=target_text,
            layer_cut=None,
            visual_token_count=len(visual_positions_list),
        )
    ]
    for cutoff in cutoffs:
        with mask_visual_keys(model, visual_positions_list, int(cutoff)):
            candidate_tokens = _generate(model, batch, max_new_tokens=args.max_new_tokens)
        candidate_text = _decode(processor, candidate_tokens)
        rows.append(
            _row_result(
                record=record,
                args=args,
                layer_count=layer_count,
                video_sha256=video_sha256,
                input_ids=input_ids,
                target_tokens=target_tokens,
                candidate_tokens=candidate_tokens,
                target_text=target_text,
                candidate_text=candidate_text,
                layer_cut=int(cutoff),
                visual_token_count=len(visual_positions_list),
            )
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--tasks", nargs="+", choices=DEFAULT_TASKS, default=list(DEFAULT_TASKS))
    parser.add_argument("--limit-per-task", type=int, default=None)
    parser.add_argument("--layer-cut-points", type=int, nargs="+", default=None)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=360 * 420)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--quantized", action="store_true")
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--plot-output", default=None)
    return parser


def run(args: argparse.Namespace) -> int:
    import torch

    from .Validate_Sparrow_hypothesises.model_analysis import load_qwen_model
    from .Validate_Sparrow_hypothesises.runtime import (
        build_qwen2vl_video_processor,
        require_cuda,
    )

    require_cuda()
    records = read_mvbench_records(
        args.manifest,
        tasks=args.tasks,
        limit_per_task=args.limit_per_task,
    )
    model = load_qwen_model(
        args.model,
        device_map=args.device_map,
        dtype=args.dtype,
        quantized=args.quantized,
        attn_implementation="eager",
    )
    processor = build_qwen2vl_video_processor(args.model, args.min_pixels, args.max_pixels)
    layer_count = _layer_count(model)
    cutoffs = (
        default_layer_cut_points(layer_count)
        if args.layer_cut_points is None
        else [int(value) for value in args.layer_cut_points]
    )
    if any(cutoff < 0 or cutoff >= layer_count for cutoff in cutoffs):
        raise ValueError(f"layer cutoffs must be in [0, {layer_count - 1}], got {cutoffs}")
    if len(set(cutoffs)) != len(cutoffs):
        raise ValueError(f"layer cutoffs must be unique, got {cutoffs}")

    rows: list[dict[str, Any]] = []
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_handle:
        for index, record in enumerate(records, start=1):
            try:
                record_rows = _run_record(model, processor, record, args, layer_count, cutoffs)
                rows.extend(record_rows)
                for row in record_rows:
                    output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                output_handle.flush()
                print(
                    f"[{index}/{len(records)}] {record['sample_id']} "
                    f"baseline={record_rows[0]['predicted_option']} "
                    f"accuracy={int(record_rows[0]['correct'])}",
                    flush=True,
                )
            except Exception as exc:  # keep long experiments auditable and resumable by inspection
                error_video_path = Path(str(record["video_path"]))
                error_row = {
                    "row_id": f"{record['sample_id']}:error",
                    "paper_figure": "Figure 3(a)",
                    "condition": "error",
                    "task": record["task"],
                    "sample_id": record["sample_id"],
                    "video_path": record["video_path"],
                    "video_sha256": _sha256_file(error_video_path) if error_video_path.is_file() else None,
                    "target_model": args.model,
                    "layer_count": layer_count,
                    "layer_cut_points": list(cutoffs),
                    "prompt": build_mvbench_prompt(record),
                    "fps": args.fps,
                    "max_frames": args.max_frames,
                    "min_pixels": args.min_pixels,
                    "max_pixels": args.max_pixels,
                    "max_new_tokens": args.max_new_tokens,
                    "dtype": args.dtype,
                    "device_map": args.device_map,
                    "quantized": args.quantized,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                rows.append(error_row)
                output_handle.write(json.dumps(error_row, ensure_ascii=False) + "\n")
                output_handle.flush()
                print(f"[{index}/{len(records)}] {record['sample_id']} ERROR {exc}", flush=True)
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    scored_rows = [row for row in rows if "correct" in row]
    error_rows = [row for row in rows if row.get("condition") == "error"]
    errors_by_task: dict[str, int] = defaultdict(int)
    for row in error_rows:
        errors_by_task[str(row["task"])] += 1
    summary = aggregate_figure3a_rows(scored_rows)
    try:
        import transformers

        transformers_version = transformers.__version__
    except ImportError:
        transformers_version = None
    expected_rows = len(records) * (len(cutoffs) + 1)
    summary_path = Path(args.summary_output) if args.summary_output else output_path.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "paper_figure": "Figure 3(a)",
                "model": args.model,
                "manifest": str(Path(args.manifest).resolve()),
                "manifest_sha256": _sha256_file(Path(args.manifest)),
                "tasks": list(args.tasks),
                "num_records": len(records),
                "num_scored_rows": len(scored_rows),
                "num_error_records": len(error_rows),
                "errors_by_task": dict(sorted(errors_by_task.items())),
                "expected_rows_if_complete": expected_rows,
                "coverage": len(scored_rows) / max(1, expected_rows),
                "layer_count": layer_count,
                "layer_cut_points": cutoffs,
                "baseline": "native_no_mask",
                "dtype": args.dtype,
                "device_map": args.device_map,
                "quantized": args.quantized,
                "fps": args.fps,
                "max_frames": args.max_frames,
                "min_pixels": args.min_pixels,
                "max_pixels": args.max_pixels,
                "max_new_tokens": args.max_new_tokens,
                "torch_version": torch.__version__,
                "transformers_version": transformers_version,
                "summary": summary,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.plot_output:
        plot_figure3a(summary, args.plot_output)
        print(f"wrote plot to {args.plot_output}")
    print(f"wrote {len(rows)} rows to {output_path}")
    print(f"wrote summary to {summary_path}")
    if error_rows:
        print(f"completed with {len(error_rows)} error record(s); see summary coverage", flush=True)
        return 2
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
