#!/usr/bin/env python3
"""Run paired AR, linear DFlash, and DDTree video retention sweeps."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from qwen_vl_utils import process_vision_info

repo_root = Path(__file__).resolve().parents[1]
specforge_root = repo_root / "src" / "train_Dflash_SpecForge"
for path in (repo_root, specforge_root):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.analyze.Validate_Sparrow_hypothesises.dflash_runtime import find_visual_positions
from src.infer.qwen25vl_dflash_compare import (
    InstrumentedDFlashDecoder,
    PreparedVideoPrompt,
    _compute_position_ids,
    _eos_token_ids,
    _load_draft,
    _load_target,
    _target_greedy,
    score_caption,
)
from src.infer.qwen25vl_ddtree import InstrumentedDDTreeDecoder
from scripts.compare_dflash_vs_msd import extract_objects


DEFAULT_TARGET = "/path/to/local-resource"
DEFAULT_CHECKPOINT = (
    "dataset/qwen25vl-3b-depth1-depth3/"
    "qwen25vl-3b-dflash-llava68k-h32-depth3-latest/training_state.pt"
)
DEFAULT_DRAFT_CONFIG = "src/train_Dflash_SpecForge/configs/qwen2.5-vl-3b-dflash-h32-depth3.json"
RETENTION_LEVELS = (100.0, 50.0, 25.0, 0.0)
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "him", "himself",
    "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "me", "more", "most", "my",
    "myself", "no", "nor", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "out",
    "over", "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "them", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "why", "with",
    "would", "you", "your", "video", "clip", "scene", "someone", "something",
}


def token_hash(token_ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(token_ids).encode("utf-8")).hexdigest()[:16]


def sync_visible_gpus() -> None:
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device_index)


def reset_peak_memory_all_gpus() -> None:
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device_index)


def peak_memory_by_device() -> dict[str, int]:
    return {
        f"cuda:{device_index}": int(torch.cuda.max_memory_allocated(device_index))
        for device_index in range(torch.cuda.device_count())
    }


def load_records(manifest_path: Path, video_root: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        sample_id = item.get("source_id") or item.get("video_name") or item.get("sample_id")
        video_rel = item.get("local_video_path") or item.get("video_path") or f"{item.get('video_name')}.mp4"
        video_path = video_root / video_rel
        if not video_path.is_file():
            alternatives = [
                video_root / "Test_Videos" / f"{item.get('video_name')}.mp4",
                video_root / f"{item.get('video_name')}.mp4",
            ]
            video_path = next((candidate for candidate in alternatives if candidate.is_file()), video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"manifest row {line_number}: video missing for {sample_id}: {video_path}")

        question = item.get("question") or item.get("prompt") or "Describe this video in detail."
        choices = item.get("choices")
        if isinstance(choices, dict) and choices:
            options = "\n".join(f"{key}. {value}" for key, value in choices.items())
            question = f"{question}\nOptions:\n{options}\nPlease choose the best option directly."
        elif isinstance(item.get("candidates"), list) and item["candidates"]:
            options = "\n".join(
                f"{chr(65 + index)}. {candidate}" for index, candidate in enumerate(item["candidates"])
            )
            question = f"{question}\nOptions:\n{options}\nPlease choose the best option directly."
        records.append(
            {
                "sample_id": str(sample_id),
                "video_name": str(item.get("video_name") or sample_id),
                "video_path": video_path,
                "question": str(question),
                "metadata": item,
            }
        )
    return records


def prepare_prompt(
    processor: Any,
    target: Any,
    sample: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[PreparedVideoPrompt, int]:
    media = [
        {
            "type": "video",
            "video": str(sample["video_path"]),
            "max_pixels": 256 * 28 * 28,
            "fps": 1.0,
            "max_frames": 16,
        },
        {"type": "text", "text": sample["question"]},
    ]
    messages = [{"role": "user", "content": media}]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    call_kwargs: dict[str, Any] = {"text": [rendered], "padding": True, "return_tensors": "pt"}
    if image_inputs:
        call_kwargs["images"] = image_inputs
    if video_inputs:
        call_kwargs["videos"] = video_inputs
        call_kwargs.update(dict(video_kwargs or {}))
    batch = processor(**call_kwargs)
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    position_ids = _compute_position_ids(target, batch)
    target_kwargs = {key: value for key, value in batch.items() if key not in {"input_ids", "position_ids"}}
    prompt = PreparedVideoPrompt(
        record=sample["metadata"],
        video_path=sample["video_path"],
        inputs=batch,
        position_ids=position_ids,
        target_kwargs=target_kwargs,
        frame_counts=(),
        video_grid_thw=(),
    )
    visual_positions = find_visual_positions(batch["input_ids"], target=target, processor=processor)
    return prompt, len(visual_positions)


def _normalized_yes_no(text: str) -> str | None:
    match = re.search(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
    return match.group(1).lower() if match else None


def _choice_letter(text: str, choices: dict[str, Any], answer: Any) -> tuple[str | None, str | None]:
    answer_text = str(answer).strip()
    target = answer_text.upper() if answer_text.upper() in choices else None
    if target is None:
        for letter, choice in choices.items():
            if str(choice).strip().casefold() == answer_text.casefold():
                target = str(letter).upper()
                break
    match = re.match(r"\s*\(?([A-Z])\)?(?:[.)\s,:-]|$)", text, flags=re.IGNORECASE)
    predicted = match.group(1).upper() if match else None
    if predicted not in {str(key).upper() for key in choices}:
        folded = text.casefold()
        exact = [str(key).upper() for key, value in choices.items() if str(value).casefold() in folded]
        predicted = exact[0] if len(exact) == 1 else None
    return predicted, target


def score_vidhalluc(text: str, record: dict[str, Any]) -> dict[str, Any]:
    task = str(record.get("task_type", ""))
    answer = record.get("answer", "")
    if task in {"BinaryQA", "Scene_Transition"}:
        predicted = _normalized_yes_no(text)
        expected = _normalized_yes_no(str(answer))
        return {
            "task_type": task,
            "predicted_label": predicted,
            "reference_label": expected,
            "task_correct": predicted is not None and predicted == expected,
        }
    if task == "MCQ":
        predicted, expected = _choice_letter(text, record.get("choices") or {}, answer)
        return {
            "task_type": task,
            "predicted_label": predicted,
            "reference_label": expected,
            "task_correct": predicted is not None and predicted == expected,
        }
    if task == "Sequence_Ranking":
        # VidHalluc answers often explain the order in prose ("Action A ...
        # followed by Action B") instead of emitting the compact label "AB".
        # Read the first two explicit, distinct action labels in textual order.
        action_labels = re.findall(r"\bAction\s+([AB])\b", text, flags=re.IGNORECASE)
        ordered_labels: list[str] = []
        for label in action_labels:
            label = label.upper()
            if not ordered_labels or label != ordered_labels[-1]:
                ordered_labels.append(label)
            if len(ordered_labels) == 2:
                break
        predicted = "".join(ordered_labels) if len(ordered_labels) == 2 else None
        if predicted is None:
            predicted_match = re.search(r"\b(A\s*B|B\s*A)\b", text.upper())
            predicted = re.sub(r"\s+", "", predicted_match.group(1)) if predicted_match else None
        expected = re.sub(r"\s+", "", str(answer).upper())
        return {
            "task_type": task,
            "predicted_label": predicted,
            "reference_label": expected,
            "task_correct": predicted is not None and predicted == expected,
        }
    return {"task_type": task, "predicted_label": None, "reference_label": str(answer), "task_correct": False}


def score_videochair(text: str, record: dict[str, Any]) -> dict[str, Any]:
    generated = extract_objects(text)
    gt_objects = {str(value).casefold() for value in record.get("ground_truth_objects", [])}
    hallucinated = [word for word in generated if word.casefold() not in gt_objects]
    grounded = [word for word in generated if word.casefold() in gt_objects]
    gt_count = len(gt_objects)
    refs = list(record.get("ground_truth_captions") or [])
    reference_metrics = [score_caption(text, str(reference)) for reference in refs] or [score_caption(text, "")]
    metric_keys = reference_metrics[0].keys()
    averaged_metrics = {
        f"mean_reference_{key}": sum(float(metric[key]) for metric in reference_metrics) / len(reference_metrics)
        for key in metric_keys
    }
    return {
        **averaged_metrics,
        "chair_i_reference_vocab": len(hallucinated) / len(generated) if generated else 0.0,
        "chair_s_reference_vocab": bool(hallucinated),
        "object_precision_reference_vocab": len(grounded) / len(generated) if generated else 0.0,
        "object_recall_reference_vocab": len(set(grounded)) / gt_count if gt_count else 0.0,
        "generated_lexical_items": len(generated),
        "reference_vocab_items": gt_count,
    }


def task_metrics(dataset: str, text: str, record: dict[str, Any]) -> dict[str, Any]:
    if dataset == "vidhalluc":
        return score_vidhalluc(text, record)
    if dataset == "videochair":
        return score_videochair(text, record)
    reference = str(record.get("answer", ""))
    return score_caption(text, reference)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_existing(path: Path, key: str) -> dict[str, dict[str, Any]]:
    existing = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status", "ok") == "ok" and row.get(key):
                existing[str(row[key])] = row
    return existing


def _quality_fields(dataset: str, text: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return task_metrics(dataset, text, metadata)


def run_dataset(
    *,
    target: Any,
    draft: Any,
    processor: Any,
    dataset: str,
    manifest: Path,
    video_root: Path,
    output_dir: Path,
    limit: int | None,
    max_new_tokens: int,
    tree_budget: int,
    retention_levels: tuple[float, ...],
    methods: tuple[str, ...],
    debug_compare_single_token: bool = False,
) -> dict[str, Any]:
    records = load_records(manifest, video_root)
    if limit is not None:
        records = records[:limit]
    out_dir = output_dir / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    ar_path = out_dir / "target_ar.jsonl"
    spec_path = out_dir / "speculative.jsonl"
    trace_path = out_dir / "round_traces.jsonl"
    ar_rows = load_existing(ar_path, "sample_id")
    spec_rows = load_existing(spec_path, "row_id")
    rounds = []
    for line in trace_path.read_text(encoding="utf-8").splitlines() if trace_path.exists() else []:
        if line.strip():
            try:
                rounds.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    traced_rows = {row.get("row_id") for row in rounds}

    dflash_decoder = InstrumentedDFlashDecoder(
        target, draft, device=torch.device("cuda:0"), draft_device=torch.device("cuda:1"),
        token_decoder=lambda ids: processor.tokenizer.decode(ids, skip_special_tokens=True).strip(),
    )
    ddtree_decoder = InstrumentedDDTreeDecoder(
        target, draft, device=torch.device("cuda:0"), draft_device=torch.device("cuda:1"),
        token_decoder=lambda ids: processor.tokenizer.decode(ids, skip_special_tokens=True).strip(),
    )
    stop_token_ids = _eos_token_ids(processor, target)
    for sample_index, sample in enumerate(records, 1):
        sample_id = sample["sample_id"]
        print(f"[{dataset} {sample_index}/{len(records)}] {sample_id}", flush=True)
        try:
            prompt, visual_token_count = prepare_prompt(
                processor, target, sample, device=torch.device("cuda:0")
            )
            input_ids = prompt.inputs["input_ids"]
            prompt_length = int(input_ids.shape[1])
            if sample_id not in ar_rows:
                reset_peak_memory_all_gpus()
                sync_visible_gpus()
                ar_started = time.perf_counter()
                target_output, ar_timing = _target_greedy(
                    target,
                    prompt,
                    max_new_tokens=max_new_tokens,
                    stop_token_ids=stop_token_ids,
                    device=torch.device("cuda:0"),
                )
                sync_visible_gpus()
                ar_timing["end_to_end_s"] = time.perf_counter() - ar_started
                ar_timing["runner_wall_seconds"] = ar_timing["end_to_end_s"]
                ar_memory = peak_memory_by_device()
                ar_timing["peak_memory_by_device_bytes"] = ar_memory
                ar_timing["peak_memory_bytes"] = sum(ar_memory.values())
                target_tokens = target_output[0, prompt_length:].detach().cpu().tolist()
                ar_timing["tokens_per_second"] = len(target_tokens) / max(ar_timing["end_to_end_s"], 1e-12)
                target_text = processor.tokenizer.decode(target_tokens, skip_special_tokens=True).strip()
                ar_row = {
                    "sample_id": sample_id,
                    "video_name": sample["video_name"],
                    "dataset": dataset,
                    "status": "ok",
                    "num_input_tokens": prompt_length,
                    "num_output_tokens": len(target_tokens),
                    "prompt_token_hash": token_hash(input_ids[0].detach().cpu().tolist()),
                    "visual_token_count": visual_token_count,
                    "output_text": target_text,
                    "output_token_ids": target_tokens,
                    "output_hash": token_hash(target_tokens),
                    "timing": ar_timing,
                    "task_metrics": _quality_fields(dataset, target_text, sample["metadata"]),
                }
                append_jsonl(ar_path, ar_row)
                ar_rows[sample_id] = ar_row
            else:
                ar_row = ar_rows[sample_id]
                target_tokens = list(ar_row["output_token_ids"])
                target_text = str(ar_row["output_text"])

            visual_positions = find_visual_positions(
                input_ids, target=target, processor=processor
            )
            for retention in retention_levels:
                if retention >= 100.0:
                    keep_mask = None
                    retained_visual = len(visual_positions)
                else:
                    keep_count = int(round(len(visual_positions) * retention / 100.0))
                    keep_positions = set(visual_positions[:keep_count])
                    keep_mask = torch.ones(prompt_length, dtype=torch.bool, device="cuda:0")
                    for position in visual_positions:
                        if position not in keep_positions:
                            keep_mask[position] = False
                    retained_visual = len(keep_positions)

                for method in methods:
                    row_id = f"{sample_id}:{method}:depth3:retention-{int(retention)}:budget-{tree_budget if method == 'ddtree' else 'na'}"
                    if row_id in spec_rows:
                        continue
                    try:
                        reset_peak_memory_all_gpus()
                        sync_visible_gpus()
                        started = time.perf_counter()
                        if method == "dflash":
                            result = dflash_decoder.decode(
                                input_ids=input_ids,
                                position_ids=prompt.position_ids,
                                target_kwargs=prompt.target_kwargs,
                                max_new_tokens=max_new_tokens,
                                stop_token_ids=stop_token_ids,
                                prefill_target_context_keep_mask=keep_mask,
                            )
                        elif method == "ddtree":
                            result = ddtree_decoder.decode(
                                input_ids=input_ids,
                                position_ids=prompt.position_ids,
                                target_kwargs=prompt.target_kwargs,
                                max_new_tokens=max_new_tokens,
                                stop_token_ids=stop_token_ids,
                                tree_budget=tree_budget,
                                prefill_target_context_keep_mask=keep_mask,
                                debug_compare_single_token=debug_compare_single_token,
                            )
                        else:
                            raise ValueError(f"unsupported method: {method}")
                        sync_visible_gpus()
                        wall_elapsed = time.perf_counter() - started
                        result.end_to_end_s = wall_elapsed
                        memory_by_device = peak_memory_by_device()
                        result.peak_memory_bytes = sum(memory_by_device.values())
                        output_tokens = result.output_ids[0, prompt_length:].detach().cpu().tolist()
                        output_text = processor.tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
                        output_digest = token_hash(output_tokens)
                        prefix = 0
                        for predicted, expected in zip(output_tokens, target_tokens):
                            if predicted != expected:
                                break
                            prefix += 1
                        lossless = output_tokens == target_tokens
                        speedup = (ar_row["timing"]["end_to_end_s"] / result.end_to_end_s) if result.end_to_end_s else 0.0
                        decode_speedup = (ar_row["timing"]["decode_s"] / result.decode_s) if result.decode_s else 0.0
                        row = {
                            "row_id": row_id,
                            "sample_id": sample_id,
                            "video_name": sample["video_name"],
                            "dataset": dataset,
                            "method": method,
                            "draft_depth": 3,
                            "tree_budget": tree_budget if method == "ddtree" else None,
                            "visual_retention_percentage": retention,
                            "visual_token_count": visual_token_count,
                            "retained_visual_tokens": retained_visual,
                            "status": "ok",
                            "target_output_hash": ar_row["output_hash"],
                            "prompt_token_hash": ar_row["prompt_token_hash"],
                            "speculative_output_hash": output_digest,
                            "speculative_output_token_ids": output_tokens,
                            "lossless": lossless,
                            "lossless_prefix_length": prefix,
                            "target_num_tokens": len(target_tokens),
                            "spec_num_tokens": len(output_tokens),
                            "target_output_text": target_text,
                            "speculative_output_text": output_text,
                            "target_task_metrics": ar_row["task_metrics"],
                            "task_metrics": _quality_fields(dataset, output_text, sample["metadata"]),
                            "tau_proposal": result.tau_proposal,
                            "tau_effective": result.tau_effective,
                            "acceptance_rounds": result.acceptance_rounds,
                            "verification_steps": len(result.acceptance_rounds),
                            "target_forward_calls": result.target_forward_calls,
                            "prefill_seconds": result.prefill_s,
                            "draft_seconds": result.draft_s,
                            "tree_build_seconds": getattr(result, "tree_build_s", 0.0),
                            "tree_compile_seconds": getattr(result, "tree_compile_s", 0.0),
                            "verification_seconds": result.verify_s,
                            "commit_seconds": getattr(result, "commit_s", 0.0),
                            "decode_seconds": result.decode_s,
                            "end_to_end_seconds": result.end_to_end_s,
                            "runner_wall_seconds": wall_elapsed,
                            "ar_prefill_seconds": ar_row["timing"]["prefill_s"],
                            "ar_decode_seconds": ar_row["timing"]["decode_s"],
                            "ar_end_to_end_seconds": ar_row["timing"]["end_to_end_s"],
                            "decode_speedup": decode_speedup,
                            "end_to_end_speedup": speedup,
                            "tokens_per_second": len(output_tokens) / max(result.end_to_end_s, 1e-12),
                            "peak_memory_bytes": result.peak_memory_bytes,
                            "peak_memory_by_device_bytes": memory_by_device,
                            "tree_traces": getattr(result, "tree_traces", []),
                        }
                        append_jsonl(spec_path, row)
                        spec_rows[row_id] = row
                        if row_id not in traced_rows:
                            for round_index, round_row in enumerate(result.acceptance_rounds):
                                append_jsonl(
                                    trace_path,
                                    {
                                        "row_id": row_id,
                                        "sample_id": sample_id,
                                        "dataset": dataset,
                                        "method": method,
                                        "retention_percentage": retention,
                                        "round_index": round_index,
                                        **round_row,
                                        "tree_trace": (
                                            result.tree_traces[round_index]
                                            if hasattr(result, "tree_traces") and round_index < len(result.tree_traces)
                                            else None
                                        ),
                                    },
                                )
                            traced_rows.add(row_id)
                        print(
                            f"  {method} retain={int(retention):3d}% n={len(output_tokens):3d} "
                            f"tau={result.tau_effective} e2e={speedup:.3f}x lossless={lossless}",
                            flush=True,
                        )
                    except Exception as exc:
                        error_row = {
                            "row_id": row_id,
                            "sample_id": sample_id,
                            "video_name": sample["video_name"],
                            "dataset": dataset,
                            "method": method,
                            "visual_retention_percentage": retention,
                            "tree_budget": tree_budget if method == "ddtree" else None,
                            "status": "error",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                        append_jsonl(spec_path, error_row)
                        print(f"  ERROR {method} retain={int(retention)}: {type(exc).__name__}: {exc}", flush=True)
                        raise
        except Exception as exc:
            print(f"[ERROR] {dataset}/{sample_id}: {type(exc).__name__}: {exc}", flush=True)
            raise

    summary = {
        "dataset": dataset,
        "manifest": str(manifest),
        "records_requested": len(records),
        "target_rows": len(ar_rows),
        "speculative_success_rows": sum(row.get("status") == "ok" for row in spec_rows.values()),
        "speculative_failed_rows": sum(row.get("status") != "ok" for row in spec_rows.values()),
        "retention_levels": list(retention_levels),
        "methods": list(methods),
        "tree_budget": tree_budget,
        "attention_implementation": getattr(target.config, "_attn_implementation", None),
        "files": {"target_ar": str(ar_path), "speculative": str(spec_path), "round_traces": str(trace_path)},
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default=DEFAULT_TARGET)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--draft-config", default=DEFAULT_DRAFT_CONFIG)
    parser.add_argument("--dataset", choices=["vdc", "vidhalluc", "videochair", "all"], default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("results/ddtree_eval_20261019"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--tree-budget", type=int, default=16)
    parser.add_argument("--retention-levels", default="100,50,25,0")
    parser.add_argument("--methods", default="dflash,ddtree")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--attention-implementation",
        choices=("sdpa", "eager"),
        default="sdpa",
        help="Qwen target attention backend; eager is useful to diagnose block-vs-token numerical drift.",
    )
    parser.add_argument(
        "--debug-compare-single-token",
        action="store_true",
        help="Replay the accepted DDTree path token-by-token for a diagnostic run.",
    )
    args = parser.parse_args()
    run_started_utc = datetime.now(timezone.utc).isoformat()
    run_started_monotonic = time.perf_counter()

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError(
            "This runner requires two visible CUDA devices: target on cuda:0 and drafter on cuda:1."
        )
    retention_levels = tuple(float(value) for value in args.retention_levels.split(","))
    if not retention_levels or any(value < 0 or value > 100 for value in retention_levels):
        raise ValueError("retention levels must be comma-separated percentages in [0, 100]")
    methods = tuple(value.strip().lower() for value in args.methods.split(",") if value.strip())
    if not methods or set(methods) - {"dflash", "ddtree"}:
        raise ValueError("methods must be selected from dflash,ddtree")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = torch.device("cuda:0")
    processor, target, target_load_s = _load_target(
        args.target_model,
        device=device,
        dtype=torch.bfloat16,
        attention=args.attention_implementation,
        device_map=args.device_map,
        max_memory={0: "18GiB", 1: "12GiB"},
    )
    draft, draft_load_s = _load_draft(
        args.checkpoint,
        args.draft_config,
        device=torch.device("cuda:1"),
        dtype=torch.bfloat16,
    )
    print(
        f"Loaded target in {target_load_s:.2f}s and DFlash drafter in {draft_load_s:.2f}s; "
        f"GPU0={torch.cuda.get_device_name(0)}, GPU1={torch.cuda.get_device_name(1)}",
        flush=True,
    )

    dataset_configs = {
        "vdc": (Path("dataset/VideoDetailCaption/test.jsonl"), Path("dataset/VideoDetailCaption")),
        "vidhalluc": (Path("dataset/VidHalluc/test.jsonl"), Path("dataset/VidHalluc")),
        "videochair": (Path("dataset/VideoChair/test.jsonl"), Path("dataset/VideoChair")),
    }
    requested = ["vdc", "vidhalluc", "videochair"] if args.dataset == "all" else [args.dataset]
    summaries = []
    for dataset in requested:
        manifest, video_root = dataset_configs[dataset]
        summaries.append(
            run_dataset(
                target=target,
                draft=draft,
                processor=processor,
                dataset=dataset,
                manifest=manifest,
                video_root=video_root,
                output_dir=args.output_dir,
                limit=args.limit,
                max_new_tokens=args.max_new_tokens,
                tree_budget=args.tree_budget,
                retention_levels=retention_levels,
                methods=methods,
                debug_compare_single_token=args.debug_compare_single_token,
            )
        )
    all_summary = {
        "run_started_utc": run_started_utc,
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "run_wall_seconds": time.perf_counter() - run_started_monotonic,
        "target_model": args.target_model,
        "checkpoint": args.checkpoint,
        "draft_config": args.draft_config,
        "checkpoint_size_bytes": Path(args.checkpoint).stat().st_size if Path(args.checkpoint).is_file() else None,
        "draft_config_sha256": hashlib.sha256(Path(args.draft_config).read_bytes()).hexdigest(),
        "target_load_seconds": target_load_s,
        "draft_load_seconds": draft_load_s,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "devices": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": int(torch.cuda.get_device_properties(index).total_memory),
            }
            for index in range(torch.cuda.device_count())
        ],
        "seed": 0,
        "target_dtype": "bfloat16",
        "video_preprocessing": {
            "max_frames": 16,
            "fps": 1.0,
            "max_pixels": 256 * 28 * 28,
            "attention_implementation": args.attention_implementation,
        },
        "max_new_tokens": args.max_new_tokens,
        "tree_budget": args.tree_budget,
        "retention_levels": list(retention_levels),
        "methods": list(methods),
        "attention_implementation": args.attention_implementation,
        "dataset_summaries": summaries,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_summary.json").write_text(json.dumps(all_summary, indent=2), encoding="utf-8")
    print(json.dumps(all_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
