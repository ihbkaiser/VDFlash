"""Run DFlash follow-ups for H2.1 (E7), H3.1 and H3.2.

The runner keeps the target model on the full visual prompt.  E7 changes only
the draft-side target-hidden context and crosses Natural/Answer-hint prompts
with Full/Reduced/Deleted visual context.  H3.1 compares Full, visual-value
Zero and visual-context Cut while recording proposal acceptance at each answer
position. H3.2 reuses the same intervention grid for VDC and MVBench while
comparing independently trained depth-1 and depth-3 draft checkpoints.

VDC uses the existing per-sample 3000-token calibration. MVBench deliberately
uses a fixed processor configuration and records the realized visual-token
count, because a VDC calibration file must not be reused for another corpus.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from src.analyze.Validate_Sparrow_hypothesises.dataset import (
    build_prompt_question,
    load_mvbench_manifest,
    load_vdc_manifest,
)
from src.analyze.Validate_Sparrow_hypothesises.model_analysis import (
    find_instruction_masks,
)
from src.analyze.Validate_Sparrow_hypothesises.dflash_runtime import (
    find_visual_positions,
    input_fingerprint,
)
from src.analyze.Validate_Sparrow_hypothesises.run_dflash_experiments import (
    _calibration_settings,
    _prepare_prompt,
    _read_jsonl,
)
from src.infer.qwen25vl_dflash_compare import (
    InstrumentedDFlashDecoder,
    _eos_token_ids,
    _sha256_tokens,
    _load_draft,
    _load_target,
    _resolve_dtype,
    capture_dflash_attention,
)
from src.workspace import resolve_workspace_path, workspace_path


DEFAULT_TARGET_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_CHECKPOINT = (
    "dataset/qwen25vl-3b-dflash-20e-llava68k-latest/training_state.pt"
)
DEFAULT_DRAFT_CONFIG = workspace_path(
    "src", "train_Dflash_SpecForge", "configs", "qwen2.5-vl-3b-dflash.json"
)
DEFAULT_MANIFEST = workspace_path("dataset", "VideoDetailCaption", "test.jsonl")
DEFAULT_VIDEO_ROOT = workspace_path("dataset", "VideoDetailCaption")
DEFAULT_CALIBRATION = workspace_path(
    "dataset", "VideoDetailCaption", "calibration_complete_20260823_currentenv.jsonl"
)


def build_prompt_record(sample: Any, prompt_variant: str) -> dict[str, Any]:
    """Copy one manifest sample and change only its controlled question text."""

    try:
        record = asdict(sample)
    except TypeError:
        record = dict(vars(sample))
    record["question"] = build_prompt_question(sample, prompt_variant)
    record["video_name"] = str(record.get("video_name") or sample.sample_id)
    record["id"] = record["video_name"]
    record["sample_id"] = record["video_name"]
    return record


def build_visual_keep_mask(
    sequence_length: int,
    visual_positions: Sequence[int],
    retention_percentage: float,
) -> torch.Tensor:
    """Keep all text and a deterministic prefix of visual context positions."""

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if not 0.0 <= float(retention_percentage) <= 100.0:
        raise ValueError("retention_percentage must be between 0 and 100")
    positions = [int(value) for value in visual_positions]
    if any(value < 0 or value >= sequence_length for value in positions):
        raise ValueError("visual_positions must be inside the sequence")
    keep = torch.ones(sequence_length, dtype=torch.bool)
    retained = int(math.ceil(len(positions) * float(retention_percentage) / 100.0))
    keep[torch.as_tensor(positions[retained:], dtype=torch.long)] = False
    return keep


def find_answer_hint_positions(
    input_ids: Sequence[int], hint_token_ids: Sequence[int]
) -> list[int]:
    """Return token positions of one contiguous answer-hint span."""

    values = [int(value) for value in input_ids]
    needle = [int(value) for value in hint_token_ids]
    if not needle or len(needle) > len(values):
        return []
    for start in range(len(values) - len(needle) + 1):
        if values[start : start + len(needle)] == needle:
            return list(range(start, start + len(needle)))
    return []


def _tokenize_without_special(processor: Any, text: str) -> list[int]:
    tokenizer = getattr(processor, "tokenizer", processor)
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if torch.is_tensor(encoded):
        encoded = encoded.detach().cpu().tolist()
    return [int(value) for value in encoded]


def build_prompt_regions(
    input_ids: torch.Tensor,
    processor: Any,
    visual_positions: Sequence[int],
    sample: Any,
    prompt_variant: str,
) -> dict[str, list[int]]:
    """Split prompt positions into visual, question and answer-hint regions."""

    masks = find_instruction_masks(input_ids, processor, visual_positions)
    ids = input_ids[0].detach().cpu().tolist()
    answer_hint: list[int] = []
    if prompt_variant == "answer_hint":
        hint_text = f"Reference information (oracle hint): {sample.answer}"
        # The first BPE token can absorb the preceding newline(s), so the
        # isolated hint string is not always byte-equivalent to the same span
        # inside the rendered chat template. Try the boundary variants and
        # retain the longest match.
        hint_candidates = (
            hint_text,
            f"\n{hint_text}",
            f"\n\n{hint_text}",
            # Matching the answer independently handles BPE boundary changes
            # around the label/colon while still identifying the informative
            # part of the oracle hint.
            str(sample.answer),
            f" {sample.answer}",
            f"\n{sample.answer}",
            f"\n\n{sample.answer}",
        )
        answer_hint = max(
            (
                find_answer_hint_positions(ids, _tokenize_without_special(processor, candidate))
                for candidate in hint_candidates
            ),
            key=len,
            default=[],
        )
    instruction = set(int(value) for value in masks["instruction_positions"])
    if prompt_variant == "answer_hint" and not answer_hint:
        # The answer is the suffix of the user message immediately before the
        # assistant marker.  If matching the serialized hint fails because of
        # a chat-template/BPE boundary, recover the answer-only span from the
        # known suffix length.  This is deliberately limited to answer tokens;
        # the fixed label is not treated as evidence.
        answer_length = len(_tokenize_without_special(processor, str(sample.answer)))
        ordered_instruction = sorted(instruction)
        if 0 < answer_length <= len(ordered_instruction):
            answer_hint = ordered_instruction[-answer_length:]
    answer_hint_set = set(answer_hint).intersection(instruction)
    visual = set(int(value) for value in visual_positions)
    question_text = sorted(instruction.difference(answer_hint_set))
    other_text = sorted(int(value) for value in masks["text_positions"])
    classified = visual | set(question_text) | answer_hint_set | set(other_text)
    other_context = sorted(set(range(len(ids))).difference(classified))
    return {
        "visual": sorted(visual),
        "question_text": question_text,
        "answer_hint": sorted(answer_hint_set),
        "other_text": other_text,
        "other_context": other_context,
    }


def remap_prompt_regions(
    regions: Mapping[str, Sequence[int]], keep_mask: Sequence[bool] | torch.Tensor
) -> dict[str, list[int]]:
    """Map original prompt positions into the compact DFlash context."""

    keep = torch.as_tensor(keep_mask, dtype=torch.bool).flatten()
    kept = torch.nonzero(keep, as_tuple=False).flatten().tolist()
    position_map = {int(old): index for index, old in enumerate(kept)}
    return {
        name: [position_map[int(value)] for value in positions if int(value) in position_map]
        for name, positions in regions.items()
    }


def summarize_dflash_attention_regions(
    records: Iterable[Mapping[str, Any]],
    regions: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    """Aggregate first-prefill DFlash attention mass and per-key density."""

    selected = [
        record
        for record in records
        if int(record.get("forward_index", 0)) == 0
        and torch.is_tensor(record.get("weights"))
    ]
    if not selected:
        return {
            "status": "unsupported",
            "captured_records": 0,
            "captured_layers": 0,
        }

    region_names = tuple(regions.keys())
    mass_values: dict[str, list[float]] = {name: [] for name in region_names}
    density_values: dict[str, list[float]] = {name: [] for name in region_names}
    context_masses: list[float] = []
    noise_masses: list[float] = []
    context_lengths: list[int] = []
    for record in selected:
        weights = record["weights"]
        if weights.ndim == 4:
            attention = weights[..., : int(record["context_length"])].mean(dim=(0, 1, 2))
        elif weights.ndim == 3:
            attention = weights[..., : int(record["context_length"])].mean(dim=(0, 1))
        else:
            raise ValueError("DFlash attention weights must have rank 3 or 4")
        context_length = int(attention.numel())
        context_lengths.append(context_length)
        context_masses.append(float(attention.sum().item()))
        noise = weights[..., int(record["context_length"]) :]
        noise_masses.append(float(noise.mean(dim=tuple(range(noise.ndim - 1))).sum().item()))
        for name in region_names:
            positions = [
                int(value)
                for value in regions[name]
                if 0 <= int(value) < context_length
            ]
            mass = float(attention[positions].sum().item()) if positions else 0.0
            mass_values[name].append(mass)
            density_values[name].append(mass / len(positions) if positions else 0.0)

    summary: dict[str, Any] = {
        "status": "ok",
        "captured_records": len(selected),
        "captured_layers": len({int(record["layer_index"]) for record in selected}),
        "context_length_mean": sum(context_lengths) / len(context_lengths),
        "context_attention_mass": sum(context_masses) / len(context_masses),
        "noise_attention_mass": sum(noise_masses) / len(noise_masses),
    }
    for name in region_names:
        key_count = sum(len(regions[name]) for _ in selected) / len(selected)
        summary[f"{name}_key_count"] = key_count
        summary[f"{name}_mass"] = sum(mass_values[name]) / len(selected)
        summary[f"{name}_density"] = sum(density_values[name]) / len(selected)
        context_uniform_density = (
            summary["context_attention_mass"] / summary["context_length_mean"]
            if summary["context_length_mean"] > 0
            else 0.0
        )
        summary[f"{name}_density_ratio_vs_context_uniform"] = (
            summary[f"{name}_density"] / context_uniform_density
            if context_uniform_density > 0
            else 0.0
        )
    question_density = summary.get("question_text_density", 0.0)
    answer_density = summary.get("answer_hint_density", 0.0)
    question_mass = summary.get("question_text_mass", 0.0)
    answer_mass = summary.get("answer_hint_mass", 0.0)
    summary["answer_hint_density_vs_question_text"] = (
        answer_density / question_density if question_density > 0 else None
    )
    summary["answer_hint_mass_share_of_question_plus_hint"] = (
        answer_mass / (answer_mass + question_mass)
        if answer_mass + question_mass > 0
        else None
    )
    return summary


def acceptance_by_position(
    rounds: Sequence[Mapping[str, Any]], output_length: int
) -> list[dict[str, Any]]:
    """Build per-answer-position proposal and acceptance rates."""

    if output_length < 0:
        raise ValueError("output_length must be non-negative")
    proposed = [0] * output_length
    accepted = [0] * output_length
    cursor = 0
    for item in rounds:
        # Older DFlash telemetry did not persist the absolute answer offset.
        # In that case, reconstruct it from the emitted block lengths so that
        # later rounds are not incorrectly counted at position zero.
        start_value = item.get("answer_position_start")
        start = int(start_value) if start_value is not None else cursor
        proposal_positions = item.get("proposal_positions")
        if proposal_positions is None:
            count = int(item.get("proposal_count", 0))
            proposal_positions = list(range(start, start + count))
        accepted_positions = set(
            int(value) for value in item.get("accepted_proposal_positions", [])
        )
        for value in proposal_positions:
            position = int(value)
            if 0 <= position < output_length:
                proposed[position] += 1
                if position in accepted_positions:
                    accepted[position] += 1
        emitted = int(item.get("effective_emitted_tokens", 0))
        if emitted <= 0:
            emitted = len(proposal_positions) + 1
        cursor = max(cursor, start + emitted)
    return [
        {
            "position": position,
            "proposed": proposed[position],
            "accepted": accepted[position],
            "rate": (
                accepted[position] / proposed[position]
                if proposed[position]
                else None
            ),
        }
        for position in range(output_length)
    ]


def _zero_visual_transform(visual_positions: Sequence[int]):
    positions = torch.as_tensor(list(visual_positions), dtype=torch.long)

    def transform(hidden: torch.Tensor) -> torch.Tensor:
        result = hidden.clone()
        if positions.numel():
            result[:, positions.to(device=hidden.device), :] = 0
        return result

    return transform


def _common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    common = 0
    for first, second in zip(left, right):
        if int(first) != int(second):
            break
        common += 1
    return common


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()


def _load_records(
    manifest: Path,
    video_root: Path,
    calibration: Path | None,
    dataset: str,
) -> list[dict[str, Any]]:
    from src.analyze.Validate_Sparrow_hypothesises.run_dflash_experiments import _sample_record

    if dataset == "vdc":
        samples = load_vdc_manifest(manifest, video_root)
    elif dataset == "mvbench":
        samples = load_mvbench_manifest(manifest, video_root)
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    calibration_rows = _read_jsonl(calibration) if calibration is not None else []
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in calibration_rows:
        by_sample.setdefault(str(row.get("sample_id")), []).append(row)
    records = [_sample_record(sample, by_sample.get(sample.sample_id, [])) for sample in samples]
    if dataset == "mvbench":
        raw_by_id = {
            str(row["sample_id"]): row for row in _read_jsonl(manifest)
            if str(row.get("sample_id")) in {sample.sample_id for sample in samples}
        }
        for record in records:
            raw = raw_by_id[str(record["sample_id"])]
            record["candidates"] = list(raw["candidates"])
            record["dataset_format"] = "mvbench"
            record["task"] = str(raw.get("task", ""))
    return records


def _condition_specs(experiment: str) -> list[dict[str, Any]]:
    if experiment == "e7":
        return [
            {"visual_condition": "full", "retention_percentage": 100.0},
            {"visual_condition": "reduced", "retention_percentage": 25.0},
            {"visual_condition": "deleted", "retention_percentage": 0.0},
        ]
    if experiment in {"h3_1", "h3_2"}:
        return [
            {"visual_condition": "full", "retention_percentage": 100.0},
            {"visual_condition": "zero", "retention_percentage": 0.0},
            {"visual_condition": "cut", "retention_percentage": 0.0},
        ]
    raise ValueError(f"unknown experiment: {experiment}")


def _run_one(
    *,
    args: argparse.Namespace,
    sample_record: dict[str, Any],
    prompt_variant: str,
    condition: Mapping[str, Any],
    target: Any,
    draft: Any,
    processor: Any,
    device: torch.device,
    draft_device: torch.device,
    target_cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    from src.infer.qwen25vl_dflash_compare import prepare_video_prompt

    sample_proxy = type("SampleProxy", (), sample_record)()
    prompt_record = build_prompt_record(sample_proxy, prompt_variant)
    # ``build_prompt_record`` intentionally emits the compact VDC contract.
    # MVBench additionally needs candidates so that the canonical multiple-
    # choice prompt can be rebuilt by prepare_video_prompt.
    prompt_record.update(
        {
            "answer": sample_record.get("answer"),
            "local_video_path": sample_record.get("local_video_path"),
            "video_path": sample_record.get("local_video_path"),
            "candidates": sample_record.get("candidates"),
            "dataset_format": args.dataset,
        }
    )
    prompt_record["calibration_by_target"] = sample_record.get("calibration_by_target", {})
    if args.dataset == "vdc":
        prompt, visual_positions, fingerprint, settings = _prepare_prompt(
            processor=processor,
            target=target,
            sample=prompt_record,
            video_root=str(args.video_root),
            device=device,
            target_visual_tokens=int(args.target_visual_tokens),
            allow_out_of_tolerance=bool(args.allow_out_of_tolerance),
        )
        visual_budget_policy = "vdc_per_sample_calibration"
    else:
        prompt = prepare_video_prompt(
            processor,
            target,
            prompt_record,
            video_root=str(args.video_root),
            device=device,
            num_frames=int(args.mvbench_num_frames),
            video_min_pixels=int(args.mvbench_min_pixels),
            video_max_pixels=int(args.mvbench_max_pixels),
            video_reader="decord",
            dataset_format="mvbench",
        )
        visual_positions = find_visual_positions(
            prompt.inputs["input_ids"], target=target, processor=processor
        )
        fingerprint = input_fingerprint(prompt.inputs)
        settings = {
            "frames": int(args.mvbench_num_frames),
            "min_pixels": int(args.mvbench_min_pixels),
            "max_pixels": int(args.mvbench_max_pixels),
            "source": "fixed_mvbench_processor_config",
        }
        visual_budget_policy = "mvbench_fixed_processor_config"

    regions = build_prompt_regions(
        prompt.inputs["input_ids"], processor, visual_positions, sample_proxy, prompt_variant
    )
    sequence_length = int(prompt.inputs["input_ids"].shape[1])
    visual_condition = str(condition["visual_condition"])
    retention = float(condition["retention_percentage"])
    keep_mask = build_visual_keep_mask(sequence_length, visual_positions, retention)
    prefill_transform = None
    context_keep_mask = None
    if visual_condition == "zero":
        prefill_transform = _zero_visual_transform(visual_positions)
        # Value ablation keeps the visual positions in the DFlash context;
        # only their hidden values are replaced by zero.
        keep_mask = torch.ones(sequence_length, dtype=torch.bool)
    elif visual_condition in {"reduced", "deleted", "cut"}:
        context_keep_mask = keep_mask
    if visual_condition == "full":
        keep_mask = torch.ones(sequence_length, dtype=torch.bool)

    capture_attention = bool(args.experiment == "e7" or args.capture_attention)
    old_attention = getattr(getattr(draft, "config", None), "_attn_implementation", None)
    if capture_attention and getattr(draft, "config", None) is not None:
        draft.config._attn_implementation = "eager"
    try:
        from src.analyze.Validate_Sparrow_hypothesises.run_dflash_experiments import _decode_prompt

        cached_target = target_cache.get(fingerprint) if target_cache is not None else None
        result = _decode_prompt(
            target=target,
            draft=draft,
            processor=processor,
            prompt=prompt,
            device=device,
            draft_device=draft_device,
            max_new_tokens=int(args.max_new_tokens),
            prefill_transform=prefill_transform,
            prefill_target_context_keep_mask=context_keep_mask,
            capture_attention=capture_attention,
            capture_first_attention_only=capture_attention,
            target_output_ids=(cached_target or {}).get("target_output_ids"),
            target_timing=(cached_target or {}).get("target_timing"),
        )
    finally:
        if capture_attention and getattr(draft, "config", None) is not None:
            draft.config._attn_implementation = old_attention

    if target_cache is not None and fingerprint not in target_cache:
        target_cache[fingerprint] = {
            "target_output_ids": list(result["target_output_ids"]),
            "target_timing": dict(result.get("timing", {}).get("target", {})),
        }

    target_tokens = [int(value) for value in result["target_output_ids"]]
    speculative_tokens = [int(value) for value in result["speculative_output_ids"]]
    common = _common_prefix(target_tokens, speculative_tokens)
    mapped_regions = remap_prompt_regions(regions, keep_mask)
    row: dict[str, Any] = {
        "backend": "dflash",
        "experiment": (
            "e7_h21"
            if args.experiment == "e7"
            else "h3_1_position"
            if args.experiment == "h3_1"
            else "h3_2_depth_comparison"
        ),
        "semantic_status": "adapted",
        "dataset": args.dataset,
        "training_corpus": args.training_corpus,
        "draft_depth": int(args.draft_depth) if args.draft_depth is not None else None,
        "visual_budget_policy": visual_budget_policy,
        "target_model": args.target_model,
        "draft_checkpoint": str(args.checkpoint),
        "draft_config": str(args.draft_config),
        "sample_id": str(sample_record["sample_id"]),
        "prompt_variant": prompt_variant,
        "visual_condition": visual_condition,
        "retention_percentage": retention,
        "target_visual_tokens": (
            int(args.target_visual_tokens) if args.dataset == "vdc" else None
        ),
        "actual_visual_tokens": len(visual_positions),
        "draft_context_visual_tokens": len(mapped_regions["visual"]),
        "target_input_fingerprint": fingerprint,
        "full_target_input_fingerprint": fingerprint,
        "draft_input_fingerprint": f"{fingerprint}:{visual_condition}",
        "calibration_settings": settings,
        "calibration_status": sample_record.get("calibration_by_target", {})
        .get(int(args.target_visual_tokens), {})
        .get("status"),
        "calibration_actual_visual_tokens": sample_record.get("calibration_by_target", {})
        .get(int(args.target_visual_tokens), {})
        .get("actual_visual_tokens"),
        "calibration_relative_error": sample_record.get("calibration_by_target", {})
        .get(int(args.target_visual_tokens), {})
        .get("relative_error"),
        "visual_positions": mapped_regions["visual"],
        "prompt_regions": mapped_regions,
        "target_output_ids": target_tokens,
        "speculative_output_ids": speculative_tokens,
        "target_output_hash": _sha256_tokens(target_tokens),
        "speculative_output_hash": _sha256_tokens(speculative_tokens),
        "lossless": common == len(target_tokens),
        "lossless_prefix_length": common,
        "acceptance_rounds": result.get("acceptance", []),
        "acceptance_by_position": acceptance_by_position(
            result.get("acceptance", []), len(target_tokens)
        ),
        "attention_summary": summarize_dflash_attention_regions(
            result.get("attention_records", []), mapped_regions
        )
        if capture_attention
        else None,
        "metrics": result.get("metrics", {}),
        "timing": result.get("timing", {}),
        "speedup": result.get("speedup", {}),
        "status": "ok",
    }
    metrics = result.get("metrics", {})
    row["accepted_prefix_tokens"] = metrics.get("tau_proposal")
    row["accepted_effective_tokens"] = metrics.get("tau_effective")
    row["mean_accepted_proposals"] = metrics.get("mean_accepted_proposals")
    return row


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    target_model_path = Path(str(args.target_model))
    if target_model_path.exists():
        args.target_model = str(resolve_workspace_path(target_model_path))
    args.checkpoint = resolve_workspace_path(args.checkpoint)
    args.draft_config = resolve_workspace_path(args.draft_config)
    args.manifest = resolve_workspace_path(args.manifest)
    args.video_root = resolve_workspace_path(args.video_root)
    if args.calibration is not None:
        args.calibration = resolve_workspace_path(args.calibration)
    args.output_dir = resolve_workspace_path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run DFlash follow-up on a GPU host")
    records = _load_records(args.manifest, args.video_root, args.calibration, args.dataset)
    if args.sample_id:
        wanted = set(args.sample_id)
        records = [row for row in records if str(row["sample_id"]) in wanted]
    if args.limit is not None:
        records = records[: int(args.limit)]
    if not records:
        raise ValueError("no samples selected")

    dtype = _resolve_dtype(args.dtype, torch.device(args.device))
    processor, target, _ = _load_target(
        args.target_model,
        device=torch.device(args.device),
        dtype=dtype,
        attention=args.target_attention,
        device_map=args.device_map,
        max_memory=args.max_memory,
    )
    draft, _ = _load_draft(
        args.checkpoint,
        args.draft_config,
        device=torch.device(args.draft_device),
        dtype=dtype,
    )
    output_files: dict[str, str] = {}
    try:
        experiments = ("e7", "h3_1") if args.experiment == "all" else (args.experiment,)
        for experiment in experiments:
            output_path = args.output_dir / (
                "e7_dflash.jsonl"
                if experiment == "e7"
                else "h3_1_dflash.jsonl"
                if experiment == "h3_1"
                else "h3_2_depth_dflash.jsonl"
            )
            existing = set()
            if args.resume and output_path.is_file():
                for row in _read_jsonl(output_path):
                    existing.add(str(row.get("row_id", "")))
            if experiment == "e7":
                variants = tuple(args.prompt_variants or ("natural", "answer_hint"))
            else:
                variants = ("natural",)
            rows_written = 0
            target_cache: dict[str, dict[str, Any]] = {}
            for sample_record in records:
                for variant in variants:
                    for condition in _condition_specs(experiment):
                        row_id = (
                            f"{sample_record['sample_id']}:{variant}:"
                            f"{condition['visual_condition']}"
                        )
                        if row_id in existing:
                            continue
                        print(
                            f"[{experiment}] {sample_record['sample_id']} "
                            f"{variant} {condition['visual_condition']}",
                            flush=True,
                        )
                        try:
                            row = _run_one(
                                args=args,
                                sample_record=sample_record,
                                prompt_variant=variant,
                                condition=condition,
                                target=target,
                                draft=draft,
                                processor=processor,
                                device=torch.device(args.device),
                                draft_device=torch.device(args.draft_device),
                                target_cache=target_cache,
                            )
                        except Exception as exc:
                            row = {
                                "backend": "dflash",
                                "experiment": (
                                    "e7_h21"
                                    if experiment == "e7"
                                    else "h3_1_position"
                                    if experiment == "h3_1"
                                    else "h3_2_depth_comparison"
                                ),
                                "semantic_status": "adapted",
                                "dataset": args.dataset,
                                "training_corpus": args.training_corpus,
                                "draft_depth": int(args.draft_depth) if args.draft_depth is not None else None,
                                "target_model": args.target_model,
                                "draft_checkpoint": str(args.checkpoint),
                                "draft_config": str(args.draft_config),
                                "sample_id": str(sample_record["sample_id"]),
                                "prompt_variant": variant,
                                "visual_condition": condition["visual_condition"],
                                "row_id": row_id,
                                "status": "error",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                            print(f"  ERROR: {row['error']}", flush=True)
                        row.setdefault("row_id", row_id)
                        _write_jsonl(output_path, [row])
                        rows_written += 1
                        gc.collect()
                        torch.cuda.empty_cache()
            output_files[experiment] = str(output_path)
            print(f"[{experiment}] wrote {rows_written} rows to {output_path}", flush=True)
    finally:
        del draft, target, processor
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "backend": "dflash",
        "experiments": list(output_files),
        "output_files": output_files,
        "sample_count": len(records),
        "checkpoint": str(args.checkpoint),
        "target_model": args.target_model,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("e7", "h3_1", "h3_2", "all"))
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--draft-config", default=DEFAULT_DRAFT_CONFIG)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--video-root", default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--calibration", default=DEFAULT_CALIBRATION)
    parser.add_argument("--dataset", choices=("vdc", "mvbench"), default="vdc")
    parser.add_argument(
        "--training-corpus",
        choices=("llava68k", "sharegpt68k", "unknown"),
        default="unknown",
        help="Training corpus label stored in every output row.",
    )
    parser.add_argument(
        "--draft-depth",
        type=int,
        choices=(1, 3, 5),
        default=None,
        help="Number of draft layers; required as metadata for H3.2.",
    )
    parser.add_argument("--output-dir", default=workspace_path("results", "e7_dflash_h3_20260911"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-id", action="append", default=None)
    parser.add_argument(
        "--prompt-variants",
        nargs="+",
        choices=("natural", "answer_hint"),
        default=None,
        help="Optional E7 prompt subset; defaults to both variants.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--draft-device", default="cuda:1")
    parser.add_argument("--device-map", choices=("cuda", "auto", "model_parallel"), default="model_parallel")
    parser.add_argument("--max-memory", default="0:22GiB,1:14GiB")
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "no"), default="auto")
    parser.add_argument("--target-attention", default="sdpa")
    parser.add_argument("--target-visual-tokens", type=int, default=3000)
    parser.add_argument("--mvbench-num-frames", type=int, default=8)
    parser.add_argument("--mvbench-min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--mvbench-max-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--allow-out-of-tolerance", action="store_true")
    parser.add_argument(
        "--capture-attention",
        action="store_true",
        help="Capture first-prefill DFlash attention summaries in addition to acceptance.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_experiment(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
