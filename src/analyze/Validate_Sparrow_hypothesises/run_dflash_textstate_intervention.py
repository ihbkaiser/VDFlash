"""Run the locked DFlash text-state intervention pilot."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping

import torch

from src.analyze.Validate_Sparrow_hypothesises.run_dflash_distribution_replay import (
    _build_prompt_record,
)
from src.analyze.Validate_Sparrow_hypothesises.run_dflash_experiments import (
    _prepare_prompt,
)
from src.analyze.Validate_Sparrow_hypothesises.run_dflash_h21_h31 import _load_records
from src.infer.qwen25vl_dflash_compare import (
    InstrumentedDFlashDecoder,
    _capture_target_hidden_states,
    _concat_captured_target_hidden,
    _compute_position_ids,
    _eos_token_ids,
    _load_draft,
    _load_target,
    _resolve_dtype,
    _sha256_tokens,
    _visual_token_mask,
)


ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = ROOT / "results/dflash_textstate_intervention/pilot12_3k_20261106"
SELECTION_PATH = RUN_DIR / "design/sample_selection.json"
SCHEDULE_PATH = RUN_DIR / "design/block_schedule.json"
CARD_PATH = RUN_DIR / "design/experiment_card.json"
DEFAULT_TARGET = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
DEFAULT_CHECKPOINT = ROOT / "artifacts/Tphuc15_qwen25vl-3b-depth1-depth3/qwen25vl-3b-dflash-llava68k-h32-depth3-latest/training_state.pt"
DEFAULT_CONFIG = ROOT / "src/train_Dflash_SpecForge/configs/qwen2.5-vl-3b-dflash-h32-depth3.json"
DEFAULT_MANIFEST = ROOT / "dataset/VideoDetailCaption/test.jsonl"
DEFAULT_VIDEO_ROOT = ROOT / "dataset/VideoDetailCaption"
DEFAULT_CALIBRATION = ROOT / "results/dflash_distribution_replay/vdc50_balanced_distribution_replay_full_forced_trajectory_20261102/calibration/calibration.jsonl"
EXPECTED_CHECKPOINT_SHA256 = "77cdbc44dea0e08cc1ee958e3efcef61318e756a77366f4fcef799435d5c3d41"
EXPECTED_CONFIG_SHA256 = "a0d3f6e1cce43bc0b8b5e636b8295908afcc60bca38bf4fd9d57c843e5ed1484"
EXPECTED_TARGET_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
CONDITIONS = ("G", "A", "N")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(values: Any) -> str:
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_source_pair(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[tuple[int, int], dict[str, Any]]]:
    pair_meta: dict[str, Any] | None = None
    pair_status: dict[str, Any] | None = None
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            kind = record.get("record_type")
            if kind == "pair_meta":
                pair_meta = record
            elif kind == "pair_status":
                pair_status = record
            elif kind == "position":
                key = (
                    int(record["block_anchor_position"]),
                    int(record["target_position"]),
                )
                rows[key] = {
                    "target_id": int(record["target_greedy_token_id"]),
                    "full_top1": int(record["full_metrics"]["top1_id"]),
                    "cut_top1": int(record["cut_metrics"]["top1_id"]),
                }
    if pair_meta is None or pair_status is None:
        raise ValueError(f"source pair is incomplete: {path}")
    if pair_status.get("status") != "ok" or not pair_status.get("full_matches_target_greedy"):
        raise ValueError(f"source pair failed its replay audit: {path}")
    return pair_meta, pair_status, rows


def build_fixed_schedule(
    sample_design: dict[str, Any],
    source_rows: Mapping[tuple[int, int], dict[str, Any]],
    target_ids: list[int],
) -> dict[int, dict[str, int]]:
    anchors = sorted({anchor for anchor, _ in source_rows})
    schedule: dict[int, dict[str, int]] = {}
    for block in sample_design["blocks"]:
        anchor = int(block["anchor_position"])
        next_anchor = next((value for value in anchors if value > anchor), None)
        if next_anchor is None:
            raise ValueError(f"source schedule lacks next anchor after {anchor}")
        proposal_count = sum(1 for a, _ in source_rows if a == anchor)
        if proposal_count != int(block["proposal_count"]):
            raise ValueError(
                f"source proposal coverage differs at anchor {anchor}: "
                f"{proposal_count} != {block['proposal_count']}"
            )
        if anchor < 0 or anchor >= len(target_ids):
            raise ValueError(f"source anchor {anchor} is outside target trajectory")
        if int(target_ids[anchor]) != int(block["anchor_token_id"]):
            raise ValueError(f"source anchor token differs from target at {anchor}")
        emitted = next_anchor - anchor
        if emitted < 1 or emitted > proposal_count + 1:
            raise ValueError(
                f"invalid source emission at anchor {anchor}: {emitted} tokens"
            )
        schedule[anchor] = {
            "anchor_token_id": int(block["anchor_token_id"]),
            "proposal_count": proposal_count,
            "emitted_tokens": emitted,
        }
    return schedule


class FeatureBankCollector:
    """Store one causal feature row per absolute position, retaining the first."""

    def __init__(self, sequence_length: int, device: torch.device):
        self.sequence_length = int(sequence_length)
        self.device = device
        self.bank: torch.Tensor | None = None
        self.seen = torch.zeros(self.sequence_length, dtype=torch.bool)
        self.duplicate_count = 0
        self.max_duplicate_abs_error = 0.0

    def __call__(self, hidden: torch.Tensor, positions: torch.Tensor) -> None:
        if hidden.ndim != 3 or hidden.shape[0] != 1:
            raise ValueError("feature capture expects [1, sequence, feature]")
        position_values = positions.detach().cpu().tolist()
        if len(position_values) != hidden.shape[1]:
            raise ValueError("feature rows and absolute positions do not align")
        if self.bank is None:
            self.bank = torch.zeros(
                (1, self.sequence_length, hidden.shape[-1]),
                dtype=hidden.dtype,
                device=self.device,
            )
        source = hidden.detach().to(device=self.device)
        for source_index, raw_position in enumerate(position_values):
            position = int(raw_position)
            if position < 0 or position >= self.sequence_length:
                continue
            if self.seen[position]:
                self.duplicate_count += 1
                error = float(
                    (self.bank[0, position].float() - source[0, source_index].float())
                    .abs()
                    .max()
                    .item()
                )
                self.max_duplicate_abs_error = max(self.max_duplicate_abs_error, error)
            else:
                self.bank[0, position].copy_(source[0, source_index])
                self.seen[position] = True

    def finish(self, required_length: int) -> torch.Tensor:
        if self.bank is None:
            raise RuntimeError("target context capture did not run")
        missing = torch.nonzero(~self.seen[:required_length], as_tuple=False).flatten()
        if missing.numel():
            raise RuntimeError(
                "target context feature bank has uncovered positions: "
                f"first={missing[:12].tolist()}, count={missing.numel()}"
            )
        return self.bank


def capture_ablation_bank(
    *,
    target: Any,
    prompt: Any,
    gold_ids: list[int],
    layer_ids: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Capture target states with merged video features zeroed before scatter."""

    input_ids = prompt.inputs["input_ids"]
    prompt_length = int(input_ids.shape[1])
    answer = torch.as_tensor(gold_ids, device=device, dtype=input_ids.dtype).view(1, -1)
    full_input_ids = torch.cat([input_ids.to(device), answer], dim=1)
    total_length = int(full_input_ids.shape[1])
    inputs_for_positions = dict(prompt.inputs)
    inputs_for_positions["input_ids"] = full_input_ids
    attention = inputs_for_positions.get("attention_mask")
    if attention is None:
        attention = torch.ones_like(full_input_ids)
    else:
        attention = attention.to(device=device)
        if attention.shape[1] != total_length:
            extension = torch.ones(
                (attention.shape[0], total_length - attention.shape[1]),
                dtype=attention.dtype,
                device=device,
            )
            attention = torch.cat([attention, extension], dim=1)
    inputs_for_positions["attention_mask"] = attention
    position_ids = _compute_position_ids(target, inputs_for_positions)
    forward_kwargs = {
        key: value
        for key, value in dict(prompt.target_kwargs).items()
        if key not in {"input_ids", "position_ids", "past_key_values", "cache_position"}
    }
    forward_kwargs.update(
        {
            "input_ids": full_input_ids,
            "attention_mask": attention,
            "position_ids": position_ids,
            "cache_position": torch.arange(total_length, device=device),
            "past_key_values": None,
            "use_cache": False,
            "output_hidden_states": False,
            "return_dict": True,
            "logits_to_keep": 1,
        }
    )
    model_core = getattr(target, "model", None)
    if model_core is None or not callable(getattr(model_core, "get_video_features", None)):
        raise RuntimeError("cannot locate Qwen video-feature merger before language-model scatter")
    had_override = "get_video_features" in getattr(model_core, "__dict__", {})
    old_override = model_core.__dict__.get("get_video_features") if had_override else None
    original_get_video_features = model_core.get_video_features
    counter = {"calls": 0, "rows": 0, "nonzero_rows": 0}

    def zero_video_features(pixel_values_videos, video_grid_thw=None):
        outputs = original_get_video_features(pixel_values_videos, video_grid_thw)
        zeros = tuple(torch.zeros_like(value) for value in outputs)
        counter["calls"] += 1
        counter["rows"] += sum(int(value.shape[0]) for value in outputs)
        counter["nonzero_rows"] += sum(
            int(torch.count_nonzero(value).item()) for value in zeros
        )
        return zeros

    model_core.get_video_features = zero_video_features
    try:
        with torch.inference_mode(), _capture_target_hidden_states(
            target, layer_ids, device=device
        ) as captured:
            target(**forward_kwargs)
        bank = _concat_captured_target_hidden(captured, layer_ids)
    finally:
        if had_override:
            model_core.get_video_features = old_override
        else:
            delattr(model_core, "get_video_features")
    if counter["calls"] < 1 or counter["rows"] < 1 or counter["nonzero_rows"] != 0:
        raise RuntimeError(f"visual feature zeroing hook audit failed: {counter}")
    if tuple(bank.shape[:2]) != (1, total_length):
        raise RuntimeError(f"ablation feature bank shape mismatch: {tuple(bank.shape)}")
    return bank, {
        "intervention_entry": f"{type(model_core).__name__}.get_video_features",
        "hook_point": "after video visual encoder/merger; before placeholder scatter",
        "zero_feature_calls": counter["calls"],
        "zero_feature_rows": counter["rows"],
        "nonzero_values_after_intervention": counter["nonzero_rows"],
        "visual_rows": int(counter["rows"]),
        "text_embeddings_or_token_ids_changed": False,
        "prompt_length": prompt_length,
        "sequence_length": total_length,
    }


def perturbation_bank(
    *,
    sample_id: str,
    g_bank: torch.Tensor,
    a_bank: torch.Tensor,
    visual_positions: set[int],
    hidden_size: int,
    seed: int,
) -> torch.Tensor:
    """Create deterministic row-wise Gaussian perturbations matched per layer."""

    g_cpu = g_bank.detach().to(device="cpu", dtype=torch.float32)[0]
    a_cpu = a_bank.detach().to(device="cpu", dtype=torch.float32)[0]
    output = g_cpu.clone()
    sequence_length, feature_size = g_cpu.shape
    layer_count = feature_size // hidden_size
    if layer_count * hidden_size != feature_size:
        raise ValueError("captured feature width is not divisible by target hidden size")
    for position in range(sequence_length):
        if position in visual_positions:
            continue
        for layer in range(layer_count):
            start = layer * hidden_size
            end = start + hidden_size
            delta = a_cpu[position, start:end] - g_cpu[position, start:end]
            magnitude = torch.linalg.vector_norm(delta)
            if float(magnitude.item()) == 0.0:
                continue
            key = f"{seed}\0{sample_id}\0{layer}\0{position}".encode("utf-8")
            row_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
            generator = torch.Generator(device="cpu").manual_seed(row_seed)
            direction = torch.randn(hidden_size, generator=generator, dtype=torch.float32)
            direction_norm = torch.linalg.vector_norm(direction)
            output[position, start:end] = (
                g_cpu[position, start:end] + direction * (magnitude / direction_norm)
            )
    return output.unsqueeze(0).to(device=g_bank.device, dtype=g_bank.dtype)


class BankTransform:
    def __init__(
        self,
        *,
        source: str,
        g_bank: torch.Tensor,
        a_bank: torch.Tensor,
        n_bank: torch.Tensor,
        visual_positions: set[int],
        device: torch.device,
    ):
        self.source = source
        self.g_bank = g_bank
        self.a_bank = a_bank
        self.n_bank = n_bank
        self.visual_positions = visual_positions
        self.device = device

    def __call__(self, _live_hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        indices_cpu = positions.detach().to(device="cpu", dtype=torch.long)
        if indices_cpu.numel() and (
            int(indices_cpu.min()) < 0 or int(indices_cpu.max()) >= self.g_bank.shape[1]
        ):
            raise RuntimeError("feature-bank lookup is outside the captured trajectory")
        indices = indices_cpu.to(device=self.device)
        output = self.g_bank.index_select(1, indices).clone()
        text_mask = torch.tensor(
            [int(value) not in self.visual_positions for value in indices_cpu.tolist()],
            dtype=torch.bool,
            device=self.device,
        )
        if text_mask.any():
            replacement_bank = self.a_bank if self.source == "A" else self.n_bank
            output[:, text_mask, :] = replacement_bank.index_select(
                1, indices[text_mask]
            )
        return output


def prefix_length(proposal_ids: list[int], target_ids: list[int]) -> int:
    count = 0
    for proposed, target in zip(proposal_ids, target_ids, strict=True):
        if proposed != target:
            break
        count += 1
    return count


def write_stream_row(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    handle.write("\n")
    handle.flush()


def execute_replay(
    *,
    sample_id: str,
    source: str,
    prompt: Any,
    processor: Any,
    target: Any,
    draft: Any,
    device: torch.device,
    gold_ids: list[int],
    cut_keep_mask: torch.Tensor,
    fixed_schedule: dict[int, dict[str, int]] | None,
    max_rounds: int | None,
    eligible_anchors: set[int] | None,
    capture_schedule: bool,
    transform: BankTransform | None,
    feature_sink=None,
    position_handle,
    block_handle,
    video_started: float,
    per_video_cap_s: float,
    deadline: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    decoder = InstrumentedDFlashDecoder(
        target,
        draft,
        device=device,
        draft_device=device,
    )
    prompt_length = int(prompt.inputs["input_ids"].shape[1])
    block_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    schedule_rows: list[dict[str, int]] = []
    last_progress = time.monotonic()

    def collect_pair(full_logits, cut_logits, block_target_ids, metadata):
        nonlocal last_progress
        if not bool(torch.isfinite(full_logits).all()) or not bool(torch.isfinite(cut_logits).all()):
            raise RuntimeError(f"non-finite draft logits for {sample_id}/{source}")
        full_ids = full_logits.argmax(dim=-1).detach().cpu().tolist()
        cut_ids = cut_logits.argmax(dim=-1).detach().cpu().tolist()
        target_values = [int(value) for value in block_target_ids]
        full_prefix = prefix_length(full_ids, target_values)
        cut_prefix = prefix_length(cut_ids, target_values)
        base = {
            "sample_id": sample_id,
            "source": source,
            "block_index": int(metadata["round_index"]),
            "anchor_position": int(metadata["anchor_position"]),
            "anchor_token_id": int(metadata["anchor_token_id"]),
            "proposal_position_start": int(metadata["proposal_position_start"]),
            "proposal_count": int(metadata["proposal_count"]),
            "eligible": (
                None
                if eligible_anchors is None
                else int(metadata["anchor_position"]) in eligible_anchors
            ),
            "full_prefix": full_prefix,
            "cut_prefix": cut_prefix,
            "target_ids": target_values,
            "full_top1_ids": full_ids,
            "cut_top1_ids": cut_ids,
        }
        block_rows.append(base)
        for slot, target_id in enumerate(target_values):
            position = int(metadata["proposal_position_start"]) + slot
            position_rows.append(
                {
                    "sample_id": sample_id,
                    "source": source,
                    "block_index": int(metadata["round_index"]),
                    "anchor_position": int(metadata["anchor_position"]),
                    "answer_position": position,
                    "slot": slot + 1,
                    "target_id": target_id,
                    "full_top1_id": full_ids[slot],
                    "cut_top1_id": cut_ids[slot],
                    "full_hit": full_ids[slot] == target_id,
                    "cut_hit": cut_ids[slot] == target_id,
                    "eligible": base["eligible"],
                },
            )
        now = time.monotonic()
        if now - last_progress >= 30:
            print(
                f"[replay] sample={sample_id} source={source} "
                f"blocks={len(block_rows)}/{max_rounds} elapsed={now-started:.1f}s",
                flush=True,
            )
            last_progress = now

    def record_schedule(entry: dict[str, int]) -> None:
        schedule_rows.append({key: int(value) for key, value in entry.items()})

    torch.cuda.synchronize(device)
    started = time.monotonic()
    try:
        result = decoder.decode(
            input_ids=prompt.inputs["input_ids"],
            position_ids=prompt.position_ids,
            target_kwargs=prompt.target_kwargs,
            max_new_tokens=256,
            stop_token_ids=_eos_token_ids(processor, target),
            distribution_cut_keep_mask=cut_keep_mask,
            distribution_target_ids=gold_ids,
            distribution_sink=collect_pair,
            distribution_fixed_schedule=fixed_schedule,
            distribution_max_rounds=max_rounds,
            distribution_deadline=deadline,
            distribution_schedule_sink=record_schedule if capture_schedule else None,
            context_feature_transform=transform,
            context_feature_sink=feature_sink,
        )
    except TimeoutError:
        seconds = time.monotonic() - started
        effective_eligible = eligible_anchors or set()
        for row in block_rows:
            if row["eligible"] is None:
                row["eligible"] = int(row["anchor_position"]) in effective_eligible
        for row in position_rows:
            if row["eligible"] is None:
                row["eligible"] = int(row["anchor_position"]) in effective_eligible
        for row in block_rows:
            write_stream_row(block_handle, row)
        for row in position_rows:
            write_stream_row(position_handle, row)
        return block_rows, {
            "condition_source": source,
            "block_count": len(block_rows),
            "elapsed_s": seconds,
            "elapsed_per_block_s": seconds / max(1, len(block_rows)),
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "complete": False,
            "deadline_expired": True,
            "schedule_entries": schedule_rows,
        }
    seconds = time.monotonic() - started
    torch.cuda.synchronize(device)
    memory = int(torch.cuda.max_memory_allocated(device))
    if max_rounds is not None and len(block_rows) != max_rounds:
        raise RuntimeError(
            f"block coverage mismatch for {sample_id}/{source}: "
            f"{len(block_rows)} != {max_rounds}"
        )
    if eligible_anchors is None and capture_schedule:
        eligible_anchors = {
            int(schedule_rows[index]["answer_offset"])
            for index in range(max(0, len(schedule_rows) - 1))
            if int(schedule_rows[index]["proposal_count"]) == 15
            and int(schedule_rows[index + 1]["answer_offset"])
            > int(schedule_rows[index]["answer_offset"])
        }
    for row in block_rows:
        if row["eligible"] is None:
            row["eligible"] = int(row["anchor_position"]) in (eligible_anchors or set())
    for row in position_rows:
        if row["eligible"] is None:
            row["eligible"] = int(row["anchor_position"]) in (eligible_anchors or set())
    for row in block_rows:
        write_stream_row(block_handle, row)
    for row in position_rows:
        write_stream_row(position_handle, row)
    output_tokens = result.output_ids[0, prompt_length : prompt_length + len(gold_ids)].detach().cpu().tolist()
    output_prefix_matches = [int(value) for value in output_tokens] == [int(value) for value in gold_ids]
    if not output_prefix_matches:
        raise RuntimeError(f"forced target trajectory output mismatch for {sample_id}/{source}")
    return block_rows, {
        "condition_source": source,
        "block_count": len(block_rows),
        "complete": True,
        "elapsed_s": seconds,
        "elapsed_per_block_s": seconds / max(1, len(block_rows)),
        "peak_memory_bytes": memory,
        "target_forward_calls": result.target_forward_calls,
        "target_prefix_reproduced": True,
        "output_prefix_matches": output_prefix_matches,
        "eligible_anchors": sorted(eligible_anchors or []),
        "schedule_entries": schedule_rows,
    }


def generate_g_reference(
    *,
    prompt: Any,
    processor: Any,
    target: Any,
    draft: Any,
    device: torch.device,
    max_new_tokens: int,
    deadline: float,
) -> tuple[list[int], dict[int, dict[str, int]], list[dict[str, int]], dict[str, Any]]:
    """Generate the local reference through DFlash's actual target verifier."""

    decoder = InstrumentedDFlashDecoder(
        target,
        draft,
        device=device,
        draft_device=device,
    )
    prompt_length = int(prompt.inputs["input_ids"].shape[1])
    torch.cuda.synchronize(device)
    started = time.monotonic()
    result = decoder.decode(
        input_ids=prompt.inputs["input_ids"],
        position_ids=prompt.position_ids,
        target_kwargs=prompt.target_kwargs,
        max_new_tokens=max_new_tokens,
        stop_token_ids=_eos_token_ids(processor, target),
        distribution_deadline=deadline,
    )
    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    output_ids = [
        int(value)
        for value in result.output_ids[0, prompt_length:].detach().cpu().tolist()
    ]
    if len(output_ids) < 16:
        raise RuntimeError("native DFlash reference output is too short for replay")

    schedule_entries: list[dict[str, int]] = []
    fixed_schedule: dict[int, dict[str, int]] = {}
    answer_offset = 0
    max_proposals = int(draft.block_size) - 1
    for round_index, round_info in enumerate(result.acceptance_rounds):
        if answer_offset >= len(output_ids):
            break
        raw_emitted = int(round_info["effective_emitted_tokens"])
        if raw_emitted < 1:
            raise RuntimeError(f"invalid DFlash emission length in round {round_index}")
        available_output = len(output_ids) - answer_offset
        emitted = min(raw_emitted, available_output)
        proposal_count = min(
            int(round_info["proposal_count"]),
            max(0, available_output - 1),
            max_proposals,
        )
        if int(round_info.get("answer_position_start", answer_offset + 1)) != answer_offset + 1:
            raise RuntimeError(
                "native DFlash round offsets are not contiguous: "
                f"round={round_index}, expected={answer_offset + 1}, "
                f"actual={round_info.get('answer_position_start')}"
            )
        entry = {
            "answer_offset": answer_offset,
            "anchor_token_id": output_ids[answer_offset],
            "proposal_count": proposal_count,
            "emitted_tokens": emitted,
        }
        schedule_entries.append(entry)
        if proposal_count > 0:
            fixed_schedule[answer_offset] = {
                "anchor_token_id": entry["anchor_token_id"],
                "proposal_count": proposal_count,
                "emitted_tokens": emitted,
            }
        answer_offset += emitted

    if not fixed_schedule:
        raise RuntimeError("native DFlash reference produced no proposal-bearing rounds")
    timing = {
        "elapsed_s": elapsed,
        "target_forward_calls": int(result.target_forward_calls),
        "round_count": len(result.acceptance_rounds),
        "output_length": len(output_ids),
        "output_sha256": _sha256_tokens(output_ids),
        "schedule_round_count": len(schedule_entries),
        "proposal_bearing_round_count": len(fixed_schedule),
        "effective_emitted_tokens": [
            int(item["effective_emitted_tokens"])
            for item in result.acceptance_rounds
        ],
    }
    return output_ids, fixed_schedule, schedule_entries, timing


def paired_stratified_bootstrap(
    sample_rows: list[dict[str, Any]],
    groups: list[list[str]],
    *,
    reps: int = 2000,
    seed: int = 20261106,
) -> dict[str, Any]:
    by_id = {row["sample_id"]: row for row in sample_rows}
    rng = random.Random(seed)
    i_values: list[float] = []
    j_values: list[float] = []
    sampled_ids: list[list[list[str]]] = []
    for _ in range(reps):
        draw = [
            [rng.choice(group) for _ in range(len(group))]
            for group in groups
        ]
        ids = [sample_id for quartile in draw for sample_id in quartile]
        sampled_ids.append(draw)
        i_values.append(sum(by_id[sample_id]["I"] for sample_id in ids) / len(ids))
        j_values.append(sum(by_id[sample_id]["J"] for sample_id in ids) / len(ids))
    i_sorted = sorted(i_values)
    j_sorted = sorted(j_values)

    def percentile(sorted_values: list[float], probability: float) -> float:
        position = probability * (len(sorted_values) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        weight = position - lower
        return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight

    return {
        "method": "paired stratified percentile bootstrap; resample 3 sample IDs with replacement within each duration quartile",
        "reps": reps,
        "seed": seed,
        "sampled_ids": sampled_ids,
        "I_samples": i_values,
        "J_samples": j_values,
        "I_ci95": [percentile(i_sorted, 0.025), percentile(i_sorted, 0.975)],
        "J_ci95": [percentile(j_sorted, 0.025), percentile(j_sorted, 0.975)],
    }


def run(args: argparse.Namespace) -> int:
    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    schedule_doc = json.loads(Path(args.schedule).read_text(encoding="utf-8"))
    card = json.loads(Path(args.card).read_text(encoding="utf-8"))
    selected_ids = list(selection.get("sample_ids", []))
    if selection.get("sample_count") != len(selected_ids) or len(selected_ids) not in (4, 8, 12):
        raise ValueError("locked sample selection is invalid")
    phase = args.phase
    protocol_revision = int(
        card.get("runtime_revision", selection.get("runtime_revision", 3))
    )
    smoke_ids = list(selection["smoke_sample_ids"])
    sample_ids = selected_ids
    if not set(smoke_ids).issubset(sample_ids):
        raise ValueError("smoke samples must belong to the selected pilot cohort")
    manifest = Path(args.manifest)
    video_root = Path(args.video_root)
    calibration = Path(args.calibration)
    checkpoint = Path(args.checkpoint)
    draft_config = Path(args.draft_config)
    target_model = str(args.target_model)
    for path in (manifest, video_root, calibration, checkpoint, draft_config, Path(target_model)):
        if not path.exists():
            raise FileNotFoundError(path)
    checkpoint_sha = sha256_file(checkpoint)
    config_sha = sha256_file(draft_config)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"draft checkpoint SHA mismatch: {checkpoint_sha}")
    if config_sha != EXPECTED_CONFIG_SHA256:
        raise ValueError(f"draft config SHA mismatch: {config_sha}")
    if EXPECTED_TARGET_REVISION not in target_model:
        raise ValueError(f"target path is not the locked snapshot: {target_model}")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this locked pilot requires a CUDA device")
    dtype = _resolve_dtype("bf16", device)
    started_wall = time.monotonic()
    cap_s = float(args.max_gpu_hours) * 3600
    per_video_cap_s = float(args.per_video_minutes) * 60
    print(f"[load] device={torch.cuda.get_device_name(device)} dtype={dtype}", flush=True)
    processor, target, target_load_s = _load_target(
        target_model,
        device=device,
        dtype=dtype,
        attention="sdpa",
        device_map="cuda",
    )
    draft, draft_load_s = _load_draft(
        checkpoint,
        draft_config,
        device=device,
        dtype=dtype,
    )
    records = _load_records(manifest, video_root, calibration, "vdc")
    record_by_id = {str(row["sample_id"]): row for row in records}
    missing_samples = sorted(set(sample_ids) - set(record_by_id))
    if missing_samples:
        raise ValueError(f"selected samples missing from manifest: {missing_samples}")
    layer_ids = [int(value) for value in draft.target_layer_ids]
    if layer_ids != [1, 9, 17, 25, 33]:
        raise ValueError(f"draft selected target layers differ from protocol: {layer_ids}")
    hidden_size = int(getattr(getattr(target.config, "text_config", target.config), "hidden_size"))
    audit = {
        "status": "running",
        "phase": phase,
        "target_model": target_model,
        "target_revision": EXPECTED_TARGET_REVISION,
        "draft_checkpoint": str(checkpoint),
        "draft_checkpoint_sha256": checkpoint_sha,
        "draft_config": str(draft_config),
        "draft_config_sha256": config_sha,
        "target_layers": layer_ids,
        "dtype": str(dtype),
        "target_attention": "sdpa",
        "draft_attention": "eager",
        "device": torch.cuda.get_device_name(device),
        "target_load_s": target_load_s,
        "draft_load_s": draft_load_s,
        "gpu_hours_cap": args.max_gpu_hours,
        "per_video_minutes_cap": args.per_video_minutes,
        "planned_sample_count": len(sample_ids),
        "started_epoch": time.time(),
        "completed_samples": [],
        "sample_audits": {},
        "protocol_status": f"revision {protocol_revision}: G-native DFlash target-verifier output defines the local reference and emission schedule; forced G/A/N replays must reproduce its emitted tokens",
    }
    audit_path = run_dir / f"{phase}_audit.json"
    write_json(run_dir / "audit.json", audit)
    write_json(
        run_dir / f"run_config_{phase}.json",
        {
            "phase": phase,
            "protocol_revision": protocol_revision,
            "target_model": target_model,
            "target_revision": EXPECTED_TARGET_REVISION,
            "target_attention": "sdpa",
            "target_dtype": str(dtype),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "draft_config": str(draft_config),
            "draft_config_sha256": config_sha,
            "sample_ids": sample_ids,
            "smoke_sample_ids": smoke_ids,
            "conditions": ["G_full", "G_cut", "A_full", "A_cut", "N_full", "N_cut"],
            "max_gpu_hours": args.max_gpu_hours,
            "per_video_minutes": args.per_video_minutes,
            "bootstrap_reps": card["bootstrap_reps"],
            "bootstrap_seed": card["bootstrap_seed"],
            "noise_seed": card["noise_seed"],
            "selection_sha256": sha256_file(Path(args.selection)),
            "schedule_sha256": sha256_file(Path(args.schedule)),
        },
    )
    import transformers

    snapshot_dir = run_dir / "source_snapshot"
    write_json(
        snapshot_dir / f"runtime_{phase}.json",
        {
            "python": os.sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "source_sha256": {
                relative: sha256_file(ROOT / relative)
                for relative in (
                    "src/analyze/Validate_Sparrow_hypothesises/run_dflash_textstate_intervention.py",
                    "src/analyze/Validate_Sparrow_hypothesises/run_dflash_experiments.py",
                    "src/infer/qwen25vl_dflash_compare.py",
                    "src/train_Dflash_SpecForge/specforge/modeling/draft/dflash.py",
                )
            },
        },
    )

    data_dir = run_dir / ("smoke" if phase == "smoke" else "pilot")
    data_dir.mkdir(parents=True, exist_ok=True)
    position_path = data_dir / "positions.jsonl.gz"
    block_path = data_dir / "blocks.jsonl"
    if position_path.exists() or block_path.exists():
        raise FileExistsError(
            f"output already exists under {data_dir}; choose a fresh --output-dir"
        )
    position_mode = "wt"
    block_mode = "w"
    sample_iter = smoke_ids if phase == "smoke" else sample_ids
    quartile_by_id = {
        str(sample["sample_id"]): int(group["duration_quartile"])
        for group in selection["groups"]
        for sample in group["selected"]
    }
    with gzip.open(position_path, position_mode, encoding="utf-8", compresslevel=1) as position_handle, block_path.open(block_mode, encoding="utf-8") as block_handle:
        for sample_index, sample_id in enumerate(sample_iter, start=1):
            if time.monotonic() - started_wall >= cap_s:
                audit["status"] = "incomplete_gpu_hour_cap"
                break
            sample_started = time.monotonic()
            record = _build_prompt_record(record_by_id[sample_id])
            prompt, visual_positions, prompt_fingerprint, settings = _prepare_prompt(
                processor=processor,
                target=target,
                sample=record,
                video_root=str(video_root),
                device=device,
                target_visual_tokens=3000,
                allow_out_of_tolerance=False,
                video_reader="decord",
            )
            prompt_length = int(prompt.inputs["input_ids"].shape[1])
            if len(visual_positions) != int(schedule_doc["samples"][sample_id]["actual_visual_tokens"]):
                raise RuntimeError(f"visual-token count changed for {sample_id}")
            source_pair_path = ROOT / schedule_doc["samples"][sample_id]["source_pair"]
            if sha256_file(source_pair_path) != schedule_doc["samples"][sample_id]["source_sha256"]:
                raise RuntimeError(f"source pair SHA mismatch for {sample_id}")
            pair_meta, pair_status, source_rows = read_source_pair(source_pair_path)
            local_video_sha256 = sha256_file(prompt.video_path)
            source_gold_ids = [int(value) for value in pair_meta["target_greedy_ids"]]
            if _sha256_tokens(source_gold_ids) != schedule_doc["samples"][sample_id]["target_greedy_ids_sha256"]:
                raise RuntimeError(f"locked target-token hash mismatch for {sample_id}")
            if pair_meta["target_input_fingerprint"] != prompt_fingerprint:
                raise RuntimeError(f"target prompt fingerprint changed for {sample_id}")
            if int(pair_meta["prompt_length"]) != prompt_length:
                raise RuntimeError(f"prompt length changed for {sample_id}")
            deadline = min(started_wall + cap_s, sample_started + per_video_cap_s)
            gold_ids, reference_schedule, reference_schedule_entries, reference_timing = (
                generate_g_reference(
                    prompt=prompt,
                    processor=processor,
                    target=target,
                    draft=draft,
                    device=device,
                    max_new_tokens=int(card["max_new_tokens"]),
                    deadline=deadline,
                )
            )
            source_trajectory_match = gold_ids == source_gold_ids
            source_common = min(len(gold_ids), len(source_gold_ids))
            source_token_agreement = (
                sum(a == b for a, b in zip(gold_ids[:source_common], source_gold_ids[:source_common]))
                / max(1, source_common)
            )
            source_first_mismatch = next(
                (index for index, (a, b) in enumerate(zip(gold_ids, source_gold_ids)) if a != b),
                None,
            )
            smoke_scope = phase == "smoke"
            if smoke_scope:
                replay_gold_ids = gold_ids[:64]
                if len(replay_gold_ids) < 16:
                    raise RuntimeError(f"G-native reference is too short for smoke: {sample_id}")
            else:
                replay_gold_ids = gold_ids
            active_schedule: dict[int, dict[str, int]] = {}
            for entry in reference_schedule_entries:
                answer_offset = int(entry["answer_offset"])
                if answer_offset >= len(replay_gold_ids):
                    break
                available_output = len(replay_gold_ids) - answer_offset
                proposal_count = min(
                    int(entry["proposal_count"]), max(0, available_output - 1)
                )
                emitted_tokens = min(
                    int(entry["emitted_tokens"]), available_output, proposal_count + 1
                )
                if proposal_count > 0:
                    active_schedule[answer_offset] = {
                        "anchor_token_id": int(replay_gold_ids[answer_offset]),
                        "proposal_count": proposal_count,
                        "emitted_tokens": emitted_tokens,
                    }
            if not active_schedule:
                raise RuntimeError(f"G-native reference has no replayable rounds for {sample_id}")
            visual_mask = _visual_token_mask(prompt.inputs["input_ids"], target)
            cut_keep_mask = ~visual_mask
            if int(cut_keep_mask.sum()) != prompt_length - len(visual_positions):
                raise RuntimeError(f"cut mask mismatch for {sample_id}")
            total_length = prompt_length + len(replay_gold_ids)
            g_collector = FeatureBankCollector(total_length, device)
            print(
                f"[{phase}] sample {sample_index}/{len(sample_iter)} {sample_id}: "
                f"prompt={prompt_length} visual={len(visual_positions)} "
                f"output_prefix={len(replay_gold_ids)} reference=G-native", 
                flush=True,
            )
            g_rows, g_timing = execute_replay(
                sample_id=sample_id,
                source="G",
                prompt=prompt,
                processor=processor,
                target=target,
                draft=draft,
                device=device,
                gold_ids=replay_gold_ids,
                cut_keep_mask=cut_keep_mask,
                fixed_schedule=active_schedule,
                max_rounds=len(active_schedule),
                eligible_anchors=None,
                capture_schedule=True,
                transform=None,
                feature_sink=g_collector,
                position_handle=position_handle,
                block_handle=block_handle,
                video_started=sample_started,
                per_video_cap_s=per_video_cap_s,
                deadline=deadline,
            )
            if not g_timing.get("complete", False):
                audit["status"] = "incomplete_deadline_during_G"
                audit["sample_audits"][sample_id] = {
                    "partial_blocks_G": len(g_rows),
                    "condition_timing": g_timing,
                }
                write_json(audit_path, audit)
                break
            schedule_entries = g_timing["schedule_entries"]
            if not schedule_entries or not g_rows:
                raise RuntimeError(f"G reference produced no replay blocks for {sample_id}")
            replay_schedule = {
                int(entry["answer_offset"]): {
                    "anchor_token_id": int(entry["anchor_token_id"]),
                    "proposal_count": int(entry["proposal_count"]),
                    "emitted_tokens": int(entry["emitted_tokens"]),
                }
                for entry in schedule_entries
                if int(entry["proposal_count"]) > 0
            }
            if replay_schedule != active_schedule:
                raise RuntimeError(
                    f"G forced replay did not reproduce native emission schedule for {sample_id}"
                )
            max_rounds = len(replay_schedule)
            eligible_anchors = {int(value) for value in g_timing["eligible_anchors"]}
            if smoke_scope and len(eligible_anchors) < 3:
                raise RuntimeError(
                    f"first 64 output tokens contain fewer than 3 complete nonterminal blocks for {sample_id}"
                )
            if not g_timing.get("output_prefix_matches"):
                raise RuntimeError(
                    f"G Full replay did not reproduce current target-greedy trajectory for {sample_id}"
                )
            exact_slot_agreement = True
            common_slots = 0
            for row in g_rows:
                anchor = int(row["anchor_position"])
                start_position = int(row["proposal_position_start"])
                for slot, target_id in enumerate(row["target_ids"]):
                    source_item = source_rows.get((anchor, start_position + slot))
                    if source_item is not None:
                        common_slots += 1
                        exact_slot_agreement &= (
                            int(row["full_top1_ids"][slot]) == source_item["full_top1"]
                            and int(row["cut_top1_ids"][slot]) == source_item["cut_top1"]
                            and int(target_id) == source_item["target_id"]
                        )
            required_end = prompt_length + max(
                int(entry["answer_offset"]) + int(entry["emitted_tokens"])
                for entry in schedule_entries
                if int(entry["proposal_count"]) > 0
            )
            g_bank = g_collector.finish(min(total_length, required_end))
            g_prefix_parity = bool(g_timing.get("output_prefix_matches"))
            a_bank, intervention_audit = capture_ablation_bank(
                target=target,
                prompt=prompt,
                gold_ids=gold_ids,
                layer_ids=layer_ids,
                device=device,
            )
            if time.monotonic() >= deadline:
                audit["status"] = "incomplete_deadline_during_A_feature_capture"
                audit["sample_audits"][sample_id] = {
                    "current_G_output_parity": g_prefix_parity,
                    "condition_timings": {"G": g_timing},
                }
                write_json(audit_path, audit)
                return 2
            if int(intervention_audit["zero_feature_rows"]) != len(visual_positions):
                raise RuntimeError(
                    f"visual intervention row count differs from placeholders for {sample_id}: "
                    f"{intervention_audit['zero_feature_rows']} != {len(visual_positions)}"
                )
            if smoke_scope:
                leakage_anchor = max(active_schedule)
            else:
                leakage_anchor = min(max(active_schedule), 31)
            altered_ids = list(gold_ids)
            vocab_size = int(getattr(target.config, "vocab_size", 152064))
            altered_ids[-1] = (altered_ids[-1] + 1) % vocab_size
            altered_bank, _ = capture_ablation_bank(
                target=target,
                prompt=prompt,
                gold_ids=altered_ids,
                layer_ids=layer_ids,
                device=device,
            )
            if time.monotonic() >= deadline:
                audit["status"] = "incomplete_deadline_during_leakage_audit"
                audit["sample_audits"][sample_id] = {
                    "current_G_output_parity": g_prefix_parity,
                    "condition_timings": {"G": g_timing},
                    "a_intervention": intervention_audit,
                }
                write_json(audit_path, audit)
                return 2
            leakage_prefix_length = prompt_length + leakage_anchor
            leakage_error = float(
                (
                    a_bank[:, :leakage_prefix_length].float()
                    - altered_bank[:, :leakage_prefix_length].float()
                )
                .abs()
                .max()
                .item()
            )
            if leakage_error > 1e-5:
                raise RuntimeError(
                    f"causal suffix perturbation changed prefix text features for {sample_id}: "
                    f"max_abs={leakage_error}"
                )
            del altered_bank
            g_features = g_bank[:, :total_length]
            a_features = a_bank[:, :total_length]
            visual_set = {int(value) for value in visual_positions}
            n_bank = perturbation_bank(
                sample_id=sample_id,
                g_bank=g_features,
                a_bank=a_features,
                visual_positions=visual_set,
                hidden_size=hidden_size,
                seed=int(card["noise_seed"]),
            )
            first_visual = min(visual_set)
            prior_visual = max(0, first_visual)
            prior_errors = (g_features[:, :prior_visual].float() - a_features[:, :prior_visual].float()).abs()
            text_positions = torch.tensor(
                [position for position in range(total_length) if position not in visual_set],
                dtype=torch.long,
                device=device,
            )
            layer_diagnostics = []
            for layer_index, layer_id in enumerate(layer_ids):
                start = layer_index * hidden_size
                end = start + hidden_size
                g_text = g_features.index_select(1, text_positions)[..., start:end].float()
                a_text = a_features.index_select(1, text_positions)[..., start:end].float()
                diff = a_text - g_text
                noise_text = n_bank.index_select(1, text_positions)[..., start:end].float()
                matched_norm_error = torch.linalg.vector_norm(noise_text - g_text, dim=-1) - torch.linalg.vector_norm(diff, dim=-1)
                layer_diagnostics.append(
                    {
                        "layer_id": layer_id,
                        "mean_text_delta_l2": float(torch.linalg.vector_norm(diff, dim=-1).mean().item()),
                        "mean_noise_delta_l2": float(torch.linalg.vector_norm(noise_text - g_text, dim=-1).mean().item()),
                        "max_noise_norm_match_error": float(matched_norm_error.abs().max().item()),
                    }
                )
            sample_rows: list[dict[str, Any]] = list(g_rows)
            transforms = {
                source: BankTransform(
                    source=source,
                    g_bank=g_features,
                    a_bank=a_features,
                    n_bank=n_bank,
                    visual_positions=visual_set,
                    device=device,
                )
                for source in ("A", "N")
            }
            source_timings = {"G": g_timing}
            for source in ("A", "N"):
                if time.monotonic() - started_wall >= cap_s:
                    audit["status"] = "incomplete_gpu_hour_cap"
                    break
                rows, timing = execute_replay(
                    sample_id=sample_id,
                    source=source,
                    prompt=prompt,
                    processor=processor,
                    target=target,
                    draft=draft,
                    device=device,
                    gold_ids=replay_gold_ids,
                    cut_keep_mask=cut_keep_mask,
                    fixed_schedule=active_schedule,
                    max_rounds=max_rounds,
                    eligible_anchors=eligible_anchors,
                    capture_schedule=False,
                    transform=transforms[source],
                    feature_sink=None,
                    position_handle=position_handle,
                    block_handle=block_handle,
                    video_started=sample_started,
                    per_video_cap_s=per_video_cap_s,
                    deadline=deadline,
                )
                sample_rows.extend(rows)
                source_timings[source] = timing
                print(
                    f"[{phase}] {sample_id} {source}: {len(rows)} blocks "
                    f"{timing['elapsed_s']:.1f}s, {timing['elapsed_per_block_s']:.3f}s/block; "
                    f"elapsed={time.monotonic()-started_wall:.1f}s", flush=True
                )
                if not timing.get("complete", False):
                    audit["status"] = f"incomplete_deadline_during_{source}"
                    break
            if audit["status"] != "running":
                audit["sample_audits"][sample_id] = {
                    "partial_blocks_by_source": {
                        source: len([row for row in sample_rows if row["source"] == source])
                        for source in CONDITIONS
                    },
                    "condition_timings": source_timings,
                    "current_G_output_parity": g_prefix_parity,
                }
                write_json(audit_path, audit)
                break
            audit["sample_audits"][sample_id] = {
                "duration_quartile": quartile_by_id[sample_id],
                "reference_kind": "native_DFlash_target_verifier_output",
                "prompt_fingerprint": prompt_fingerprint,
                "video_path": str(prompt.video_path),
                "video_sha256": local_video_sha256,
                "video_bytes": prompt.video_path.stat().st_size,
                "prompt_length": prompt_length,
                "target_output_length": len(gold_ids),
                "target_output_sha256": _sha256_tokens(gold_ids),
                "replay_output_length": len(replay_gold_ids),
                "replay_output_sha256": _sha256_tokens(replay_gold_ids),
                "source_trajectory_match": source_trajectory_match,
                "source_target_token_agreement_fraction": source_token_agreement,
                "source_first_mismatch_answer_position": source_first_mismatch,
                "reference_generation_timing": reference_timing,
                "reference_schedule": reference_schedule_entries,
                "visual_positions": len(visual_positions),
                "calibration_settings": settings,
                "block_count": len(eligible_anchors),
                "g_schedule_rounds": len(schedule_entries),
                "current_G_output_parity": g_prefix_parity,
                "source_all_slot_top1_agreement": exact_slot_agreement,
                "source_common_slots_compared": common_slots,
                "g_schedule": schedule_entries,
                "g_bank_captured_positions": int(g_collector.seen.sum().item()),
                "g_bank_duplicate_count": g_collector.duplicate_count,
                "g_bank_max_duplicate_abs_error": g_collector.max_duplicate_abs_error,
                "a_intervention": intervention_audit,
                "pre_visual_text_max_abs_error": float(prior_errors.max().item()) if prior_errors.numel() else 0.0,
                "suffix_perturbation_anchor": leakage_anchor,
                "suffix_perturbation_prefix_max_abs_error": leakage_error,
                "feature_layer_diagnostics": layer_diagnostics,
                "condition_timings": source_timings,
                "elapsed_video_s": time.monotonic() - sample_started,
            }
            audit["completed_samples"].append(sample_id)
            write_json(audit_path, audit)
            print(
                f"[{phase}] sample {sample_id} complete; video_elapsed="
                f"{time.monotonic()-sample_started:.1f}s, total_elapsed="
                f"{time.monotonic()-started_wall:.1f}s", flush=True
            )
            del g_bank, g_features, a_bank, a_features, n_bank
            torch.cuda.empty_cache()

    audit["elapsed_s"] = time.monotonic() - started_wall
    audit["gpu_hours_used_wall_estimate"] = audit["elapsed_s"] / 3600
    if phase == "smoke" and audit["status"] == "running":
        audit["status"] = "smoke_passed" if len(audit["completed_samples"]) == len(smoke_ids) else "smoke_incomplete"
    elif phase == "full" and audit["status"] == "running":
        audit["status"] = "completed" if len(audit["completed_samples"]) == len(sample_ids) else "incomplete"
    write_json(audit_path, audit)
    write_json(run_dir / "audit.json", audit)
    if phase == "full" and audit["status"] == "completed":
        analyze_full_results(run_dir, selection, sample_ids)
    print(f"[{phase}] status={audit['status']} elapsed={audit['elapsed_s']:.1f}s", flush=True)
    return 0 if audit["status"] in {"smoke_passed", "completed"} else 2


def analyze_full_results(run_dir: Path, selection: dict[str, Any], sample_ids: list[str]) -> None:
    data_dir = run_dir / "pilot"
    block_rows = [json.loads(line) for line in (data_dir / "blocks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    by_sample_source: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in block_rows:
        by_sample_source.setdefault((row["sample_id"], row["source"]), []).append(row)
    sample_results: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        row: dict[str, Any] = {"sample_id": sample_id}
        for source in CONDITIONS:
            all_rows = by_sample_source.get((sample_id, source), [])
            rows = [item for item in all_rows if item.get("eligible") is True]
            if not rows:
                raise RuntimeError(f"missing block metrics for {sample_id}/{source}")
            anchor_set = {int(item["anchor_position"]) for item in rows}
            if source == "G":
                expected_anchors = anchor_set
            elif anchor_set != expected_anchors:
                raise RuntimeError(
                    f"fixed block schedule differs for {sample_id}/{source}"
                )
            row[f"tau_{source}_full"] = sum(item["full_prefix"] for item in rows) / len(rows)
            row[f"tau_{source}_cut"] = sum(item["cut_prefix"] for item in rows) / len(rows)
            row[f"B_{source}"] = row[f"tau_{source}_full"] - row[f"tau_{source}_cut"]
        row["I"] = row["B_A"] - row["B_G"]
        row["J"] = row["B_A"] - row["B_N"]
        sample_results.append(row)
    groups = [
        [str(sample["sample_id"]) for sample in group["selected"]]
        for group in selection["groups"]
    ]
    bootstrap = paired_stratified_bootstrap(sample_results, groups)
    mean_i = sum(row["I"] for row in sample_results) / len(sample_results)
    mean_j = sum(row["J"] for row in sample_results) / len(sample_results)
    if mean_i >= 0.05 and bootstrap["I_ci95"][0] > 0 and bootstrap["J_ci95"][0] > 0:
        decision = "supports_text_visual_substitutability_under_this_intervention_and_noise_control"
    elif mean_i >= 0.05 and bootstrap["I_ci95"][0] > 0:
        decision = "supports_interaction_but_specificity_vs_noise_control_is_unclear"
    elif bootstrap["I_ci95"][1] < 0:
        decision = "opposite_to_predicted_interaction"
    else:
        decision = "inconclusive"
    with (data_dir / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_results[0]))
        writer.writeheader()
        writer.writerows(sample_results)
    with (data_dir / "blocks.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [key for key in block_rows[0] if key not in {"target_ids", "full_top1_ids", "cut_top1_ids"}]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in block_rows)
    write_json(data_dir / "bootstrap.json", bootstrap)
    summary = {
        "status": f"complete_{len(sample_results)}_video_pilot",
        "n_videos": len(sample_results),
        "n_eligible_blocks_per_source": sum(
            1 for row in block_rows if row.get("source") == "G" and row.get("eligible") is True
        ),
        "I_mean_video_macro": mean_i,
        "I_ci95": bootstrap["I_ci95"],
        "J_mean_video_macro": mean_j,
        "J_ci95": bootstrap["J_ci95"],
        "decision": decision,
        "samples": sample_results,
        "tau_macro_mean": {
            f"tau_{source}_{visual}": sum(
                row[f"tau_{source}_{visual}"] for row in sample_results
            ) / len(sample_results)
            for source in CONDITIONS
            for visual in ("full", "cut")
        },
    }
    aggregate_manifest_path = run_dir / "aggregate_manifest.json"
    aggregate_manifest = (
        json.loads(aggregate_manifest_path.read_text(encoding="utf-8"))
        if aggregate_manifest_path.is_file()
        else None
    )
    if aggregate_manifest is not None:
        summary["execution_batches"] = aggregate_manifest.get("execution_batches", [])
        summary["audit_gate"] = aggregate_manifest.get("audit_gate", {})
    write_json(data_dir / "summary.json", summary)
    quartile_sizes = [len(group["selected"]) for group in selection["groups"]]
    if len(set(quartile_sizes)) == 1:
        cohort_description = (
            f"a stratified {len(sample_results)}-video cohort "
            f"({quartile_sizes[0]} videos per duration quartile)"
        )
    else:
        cohort_description = (
            f"a {len(sample_results)}-video cohort with quartile sizes {quartile_sizes}"
        )
    batch_details = ""
    if aggregate_manifest is not None:
        batch_names = [
            f"{batch['sample_count']} videos from `{batch['run_dir']}`"
            for batch in aggregate_manifest.get("execution_batches", [])
        ]
        if batch_names:
            batch_details = " Execution was split across " + " and ".join(batch_names) + "."
    audit_gate = (
        aggregate_manifest.get("audit_gate", {})
        if aggregate_manifest is not None
        else {}
    )
    audit_statement = (
        f" Audit review: `{audit_gate.get('status', 'not separately recorded')}`. "
        f"{audit_gate.get('summary', '')}"
        if audit_gate
        else ""
    )
    report = [
        "# DFlash text-state intervention pilot: result",
        "",
        f"- Status: `{summary['status']}`; videos: {summary['n_videos']}; eligible G blocks: {summary['n_eligible_blocks_per_source']}",
        f"- Primary interaction I: {mean_i:.4f} tokens/round; stratified paired bootstrap 95% CI [{bootstrap['I_ci95'][0]:.4f}, {bootstrap['I_ci95'][1]:.4f}]",
        f"- Noise-control contrast J: {mean_j:.4f} tokens/round; 95% CI [{bootstrap['J_ci95'][0]:.4f}, {bootstrap['J_ci95'][1]:.4f}]",
        f"- Decision: `{decision}`.",
        "",
        f"All outcomes are forced-trajectory proposal-prefix agreement, not answer accuracy, free-running acceptance, or latency. This exploratory pilot uses one checkpoint and one noise seed on {cohort_description}.{batch_details} The reference output and emission schedule come from native DFlash G target verification; forced G/A/N replay must reproduce that schedule. The earlier A100 source trajectory did not reproduce on the local T4 stack and remains diagnostic only.{audit_statement}",
        "",
        "See `samples.csv`, `blocks.csv`, `bootstrap.json`, and `audit.json` for paired results and execution provenance.",
    ]
    (data_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    parser.add_argument("--target-model", default=str(DEFAULT_TARGET))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--draft-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--video-root", default=str(DEFAULT_VIDEO_ROOT))
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    parser.add_argument("--selection", default=str(SELECTION_PATH))
    parser.add_argument("--schedule", default=str(SCHEDULE_PATH))
    parser.add_argument("--card", default=str(CARD_PATH))
    parser.add_argument("--output-dir", default=str(RUN_DIR))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-gpu-hours", type=float, default=2.0)
    parser.add_argument("--per-video-minutes", type=float, default=10.0)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    try:
        raise SystemExit(run(arguments))
    except Exception as exc:
        phase_audit_path = Path(arguments.output_dir) / f"{arguments.phase}_audit.json"
        root_audit_path = Path(arguments.output_dir) / "audit.json"
        audit_path = phase_audit_path if phase_audit_path.is_file() else root_audit_path
        if audit_path.is_file():
            try:
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                audit["status"] = "failed"
                audit["failure"] = f"{type(exc).__name__}: {exc}"
                audit["failed_epoch"] = time.time()
                write_json(phase_audit_path, audit)
                write_json(root_audit_path, audit)
            except Exception:
                pass
        raise
