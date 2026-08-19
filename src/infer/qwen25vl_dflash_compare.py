"""Compare two SpecForge Qwen2.5-VL DFlash checkpoints on one VDC sample."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any

import torch

from src.analyze.Whether_they_are_appliable_for_dDrafter.compare_dflash_reference import (
    score_pair,
)


def _now(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _extract_context_feature(hidden_states: Any, layer_ids: list[int]) -> torch.Tensor:
    if hidden_states is None:
        raise RuntimeError("target forward did not return hidden_states")
    try:
        selected = [hidden_states[layer_id + 1] for layer_id in layer_ids]
    except (IndexError, TypeError) as exc:
        raise RuntimeError(
            "target hidden_states do not contain the configured DFlash layers"
        ) from exc
    return torch.cat(selected, dim=-1)


def _cache_length(cache: Any) -> int:
    getter = getattr(cache, "get_seq_length", None)
    if not callable(getter):
        return 0
    return int(getter())


def _crop_cache(cache: Any, length: int) -> None:
    crop = getattr(cache, "crop", None)
    if callable(crop):
        crop(int(length))


def _as_stop_set(stop_token_ids: list[int] | None) -> set[int]:
    return {int(token_id) for token_id in (stop_token_ids or [])}


@dataclass
class SpeculativeDecodeResult:
    """Instrumented output from one greedy DFlash decode."""

    output_ids: torch.Tensor
    num_input_tokens: int
    acceptance_rounds: list[dict[str, Any]] = field(default_factory=list)
    target_forward_calls: int = 0
    prefill_s: float = 0.0
    draft_s: float = 0.0
    verify_s: float = 0.0
    decode_s: float = 0.0
    end_to_end_s: float = 0.0
    peak_memory_bytes: int = 0

    @property
    def num_output_tokens(self) -> int:
        return max(0, int(self.output_ids.shape[1]) - self.num_input_tokens)

    @property
    def matched_proposals(self) -> list[int]:
        return [int(item["matched_proposals"]) for item in self.acceptance_rounds]

    @property
    def effective_emitted_tokens(self) -> list[int]:
        return [int(item["effective_emitted_tokens"]) for item in self.acceptance_rounds]

    @property
    def valid_acceptance_rounds(self) -> list[dict[str, Any]]:
        """Full, non-terminal rounds used for paper-style tau metrics."""

        return [
            item
            for item in self.acceptance_rounds
            if not item.get("is_partial_block", False)
            and not item.get("is_terminal", False)
        ]

    @property
    def tau_proposal(self) -> float | None:
        """Mean number of draft proposals accepted on valid rounds."""

        rounds = self.valid_acceptance_rounds
        if not rounds:
            return None
        return sum(int(item["matched_proposals"]) for item in rounds) / len(rounds)

    @property
    def tau_effective(self) -> float | None:
        """Mean emitted length, including the target's bonus token."""

        rounds = self.valid_acceptance_rounds
        if not rounds:
            return None
        return sum(int(item["effective_emitted_tokens"]) for item in rounds) / len(rounds)

    def as_dict(self) -> dict[str, Any]:
        rounds = self.acceptance_rounds
        return {
            "num_output_tokens": self.num_output_tokens,
            "target_forward_calls": int(self.target_forward_calls),
            "rounds": len(rounds),
            "mean_accepted_proposals": (
                sum(self.matched_proposals) / len(rounds) if rounds else 0.0
            ),
            "mean_effective_emitted_tokens": (
                sum(self.effective_emitted_tokens) / len(rounds) if rounds else 0.0
            ),
            "tau": self.tau_effective,
            "tau_proposal": self.tau_proposal,
            "tau_effective": self.tau_effective,
            "acceptance_rounds": rounds,
            "timing": {
                "prefill_s": float(self.prefill_s),
                "draft_s": float(self.draft_s),
                "verify_s": float(self.verify_s),
                "decode_s": float(self.decode_s),
                "end_to_end_s": float(self.end_to_end_s),
                "tokens_per_second": self.num_output_tokens / max(self.end_to_end_s, 1e-12),
            },
            "peak_memory_bytes": int(self.peak_memory_bytes),
        }


class InstrumentedDFlashDecoder:
    """Run SpecForge DFlash while exposing acceptance and timing telemetry."""

    def __init__(self, target: Any, draft: Any, *, device: torch.device):
        self.target = target
        self.draft = draft
        self.device = torch.device(device)

    def _target_forward(self, kwargs: dict[str, Any]) -> Any:
        try:
            return self.target(**kwargs)
        except TypeError as exc:
            if "logits_to_keep" not in str(exc):
                raise
            fallback = dict(kwargs)
            fallback.pop("logits_to_keep", None)
            return self.target(**fallback)

    def _extend_positions(self, position_ids: torch.Tensor, total_length: int) -> torch.Tensor:
        return _extend_position_ids(position_ids, total_length, self.device)

    @torch.inference_mode()
    def decode(
        self,
        *,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        target_kwargs: dict[str, Any] | None,
        max_new_tokens: int,
        stop_token_ids: list[int] | None,
    ) -> SpeculativeDecodeResult:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("DFlash inference currently supports batch size one")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")

        input_ids = input_ids.to(device=self.device, dtype=torch.long)
        prompt_length = int(input_ids.shape[1])
        max_length = prompt_length + int(max_new_tokens)
        block_size = int(self.draft.block_size)
        mask_token_id = int(self.draft.mask_token_id)
        all_positions = self._extend_positions(position_ids, max_length + block_size)
        target_cache = _new_cache()
        draft_cache = _new_cache()
        stop_ids = _as_stop_set(stop_token_ids)
        target_kwargs = dict(target_kwargs or {})
        output_ids = torch.full(
            (1, max_length + block_size),
            mask_token_id,
            dtype=torch.long,
            device=self.device,
        )
        output_ids[:, :prompt_length] = input_ids

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        started = _now(self.device)
        prefill_kwargs = dict(target_kwargs)
        prefill_kwargs.update(
            {
                "input_ids": input_ids,
                "position_ids": all_positions[..., :prompt_length],
                "past_key_values": target_cache,
                "cache_position": torch.arange(prompt_length, device=self.device),
                "use_cache": True,
                "logits_to_keep": 1,
                "output_hidden_states": True,
                "return_dict": True,
            }
        )
        prefill_started = _now(self.device)
        target_output = self._target_forward(prefill_kwargs)
        prefill_s = _now(self.device) - prefill_started
        target_calls = 1
        if getattr(target_output, "past_key_values", None) is not None:
            target_cache = target_output.past_key_values
        target_embed = self.target.get_input_embeddings()
        target_hidden = _extract_context_feature(
            target_output.hidden_states,
            [int(value) for value in self.draft.target_layer_ids],
        )
        pending_anchor = target_output.logits[:, -1:, :].argmax(dim=-1)
        output_ids[:, prompt_length : prompt_length + 1] = pending_anchor
        decode_started = _now(self.device)

        if max_new_tokens == 0 or int(pending_anchor.item()) in stop_ids:
            end_to_end_s = _now(self.device) - started
            decode_s = _now(self.device) - decode_started
            result = SpeculativeDecodeResult(
                output_ids=output_ids[:, :max_length],
                num_input_tokens=prompt_length,
                target_forward_calls=target_calls,
                prefill_s=prefill_s,
                decode_s=decode_s,
                end_to_end_s=end_to_end_s,
            )
            return self._finish_result(result, stop_ids)

        start = prompt_length
        rounds: list[dict[str, Any]] = []
        draft_s = 0.0
        verify_s = 0.0
        while start < max_length:
            block_output_ids = output_ids[:, start : start + block_size].clone()
            block_position_ids = all_positions[..., start : start + block_size]
            draft_started = _now(self.device)
            noise_embedding = target_embed(block_output_ids)
            draft_hidden = self.draft(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=all_positions[
                    ...,
                    _cache_length(draft_cache) : start + block_size,
                ],
                past_key_values=draft_cache,
                use_cache=True,
                is_causal=False,
            )
            _crop_cache(draft_cache, start)
            proposals = self.draft._sample_draft_tokens(
                self.target,
                draft_hidden,
                block_output_ids,
            )
            draft_s += _now(self.device) - draft_started
            block_output_ids[:, 1:] = proposals

            verify_started = _now(self.device)
            target_output = self._target_forward(
                {
                    "input_ids": block_output_ids,
                    "position_ids": block_position_ids,
                    "past_key_values": target_cache,
                    "cache_position": torch.arange(
                        start, start + block_output_ids.shape[1], device=self.device
                    ),
                    "use_cache": True,
                    "output_hidden_states": True,
                    "return_dict": True,
                }
            )
            verify_s += _now(self.device) - verify_started
            target_calls += 1
            if getattr(target_output, "past_key_values", None) is not None:
                target_cache = target_output.past_key_values
            posterior = target_output.logits.argmax(dim=-1)
            proposal_matches = block_output_ids[:, 1:] == posterior[:, :-1]
            accepted = int(proposal_matches.cumprod(dim=1).sum(dim=1)[0].item())
            emitted = accepted + 1
            output_ids[:, start : start + emitted] = block_output_ids[:, :emitted]
            output_ids[:, start + emitted] = posterior[:, accepted]
            start += emitted
            _crop_cache(target_cache, start)
            target_hidden = _extract_context_feature(
                target_output.hidden_states,
                [int(value) for value in self.draft.target_layer_ids],
            )[:, :emitted, :]
            stop_hit = any(
                int(token_id) in stop_ids
                for token_id in output_ids[0, prompt_length : min(start + 1, max_length)].tolist()
            )
            rounds.append(
                {
                    "proposal_count": int(proposals.shape[1]),
                    "matched_proposals": accepted,
                    "effective_emitted_tokens": emitted,
                    "is_partial_block": int(proposals.shape[1]) < block_size - 1,
                    "is_terminal": bool(stop_hit or start >= max_length - 1),
                }
            )
            if stop_hit:
                break

        decode_s = _now(self.device) - decode_started
        end_to_end_s = _now(self.device) - started
        result = SpeculativeDecodeResult(
            output_ids=output_ids[:, :max_length],
            num_input_tokens=prompt_length,
            acceptance_rounds=rounds,
            target_forward_calls=target_calls,
            prefill_s=prefill_s,
            draft_s=draft_s,
            verify_s=verify_s,
            decode_s=decode_s,
            end_to_end_s=end_to_end_s,
        )
        return self._finish_result(result, stop_ids)

    def _finish_result(
        self,
        result: SpeculativeDecodeResult,
        stop_token_ids: set[int] | None = None,
    ) -> SpeculativeDecodeResult:
        if self.device.type == "cuda":
            result.peak_memory_bytes = int(torch.cuda.max_memory_allocated(self.device))
        result.output_ids = result.output_ids[
            :, result.output_ids[0] != int(self.draft.mask_token_id)
        ]
        if stop_token_ids:
            generated = result.output_ids[0, result.num_input_tokens :]
            stop_positions = torch.where(
                torch.isin(
                    generated,
                    torch.tensor(sorted(stop_token_ids), device=generated.device),
                )
            )[0]
            if stop_positions.numel() > 0:
                result.output_ids = result.output_ids[
                    :, : result.num_input_tokens + int(stop_positions[0].item()) + 1
                ]
        return result


def _new_cache() -> Any:
    from transformers import DynamicCache

    return DynamicCache()


def load_manifest_sample(manifest: str | Path, sample_index: int) -> dict[str, Any]:
    """Load one JSONL record by zero-based index with a useful bounds error."""

    if sample_index < 0:
        raise IndexError(f"sample-index {sample_index} must be non-negative")
    path = Path(manifest).expanduser().resolve(strict=True)
    record_index = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if record_index == sample_index:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(
                    f"manifest record {record_index} must contain a JSON object"
                )
            return record
        record_index += 1
    raise IndexError(f"sample-index {sample_index} is outside manifest {path}")


def _with_video_suffix(path: Path) -> list[Path]:
    if path.suffix:
        return [path, path.with_suffix(".mp4"), path.with_suffix(".mkv")]
    return [path.with_suffix(suffix) for suffix in (".mp4", ".mkv", ".mov", ".avi")]


def resolve_video_path(record: dict[str, Any], video_root: str | Path) -> Path:
    """Resolve common VideoDetailCaption path/name fields to a local video."""

    root = Path(video_root).expanduser().resolve(strict=True)
    candidates: list[Path] = []
    for key in ("video_path", "path", "video_file", "local_video_path"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            raw = Path(value)
            candidates.extend(_with_video_suffix(raw if raw.is_absolute() else root / raw))

    for key in ("video_name", "video", "video_id", "id"):
        value = record.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        name = Path(value)
        for base in (root, root / "Test_Videos", root / "test_videos"):
            candidates.extend(_with_video_suffix(base / name))

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    shown = ", ".join(str(path) for path in candidates[:8]) or "<no video field>"
    raise FileNotFoundError(f"Video not found; tried: {shown}")


def score_caption(prediction: str, reference: str) -> dict[str, float]:
    """Return dependency-light caption overlap metrics."""

    return {
        key: float(value) for key, value in score_pair(prediction, reference).items()
    }


_TIMING_KEYS = (
    "prefill_s",
    "draft_s",
    "verify_s",
    "decode_s",
    "end_to_end_s",
    "tokens_per_second",
    "checkpoint_load_s",
    "target_prefill_s",
    "target_decode_s",
    "target_greedy_s",
    "speedup_vs_target",
)
_METRIC_KEYS = (
    "exact_match",
    "bleu1",
    "bleu2",
    "bleu3",
    "bleu4",
    "bleu",
    "rouge_l",
    "coverage",
    "unigram_precision",
    "unigram_recall",
    "unigram_f1",
)


def _bounded_metric(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and 0.0 <= number <= 1.0


def _finite_nonnegative(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and value >= 0


def validate_report_success(report: dict[str, Any]) -> bool:
    """Return whether a comparison report meets the lossless smoke gate."""

    baseline = report.get("target_baseline")
    checkpoints = report.get("checkpoints")
    if not isinstance(baseline, dict) or not str(baseline.get("prediction", "")).strip():
        return False
    if not isinstance(checkpoints, list) or len(checkpoints) != 2:
        return False
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, dict):
            return False
        if checkpoint.get("status") != "ok":
            return False
        if checkpoint.get("outputs_match") is not True:
            return False
        if not str(checkpoint.get("prediction", "")).strip():
            return False
        metrics = checkpoint.get("text_metrics")
        timing = checkpoint.get("timing")
        speedup = checkpoint.get("speedup")
        acceptance = checkpoint.get("acceptance")
        if (
            not isinstance(metrics, dict)
            or not isinstance(timing, dict)
            or not isinstance(speedup, dict)
            or not isinstance(acceptance, dict)
        ):
            return False
        if any(not _bounded_metric(metrics.get(key)) for key in _METRIC_KEYS):
            return False
        if any(not _finite_nonnegative(timing.get(key)) for key in _TIMING_KEYS):
            return False
        if any(
            not _finite_nonnegative(speedup.get(key)) for key in ("esr", "dsr")
        ):
            return False
        for key in ("tau", "tau_proposal", "tau_effective"):
            value = acceptance.get(key)
            if value is not None and not _finite_nonnegative(value):
                return False
    return True


__all__ = [
    "InstrumentedDFlashDecoder",
    "PreparedVideoPrompt",
    "SpeculativeDecodeResult",
    "load_manifest_sample",
    "resolve_video_path",
    "score_caption",
    "validate_report_success",
]


DEFAULT_TARGET_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_DRAFT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "train_Dflash_SpecForge"
    / "configs"
    / "qwen2.5-vl-3b-dflash.json"
)
DEFAULT_CHECKPOINTS = (
    Path("dataset/qwen25vl-3b-dflash-llava68k-latest"),
    Path("dataset/qwen25vl-3b-dflash-sharegpt68k-latest"),
)
DEFAULT_MANIFEST = Path("dataset/VideoDetailCaption/test.jsonl")
DEFAULT_VIDEO_ROOT = Path("dataset/VideoDetailCaption")
DEFAULT_OUTPUT = Path("results/infer/qwen25vl_3b_dflash_vdc_sample0.json")


@dataclass(frozen=True)
class PreparedVideoPrompt:
    record: dict[str, Any]
    video_path: Path
    inputs: dict[str, Any]
    position_ids: torch.Tensor
    target_kwargs: dict[str, Any]
    frame_counts: tuple[int, ...]
    video_grid_thw: tuple[tuple[int, int, int], ...]


def _ensure_specforge_importable() -> None:
    specforge_root = Path(__file__).resolve().parents[1] / "train_Dflash_SpecForge"
    if str(specforge_root) not in sys.path:
        sys.path.insert(0, str(specforge_root))


def _resolve_dtype(value: str, device: torch.device) -> torch.dtype:
    if value == "no":
        return torch.float32
    if value == "fp16":
        return torch.float16
    if value == "bf16":
        return torch.bfloat16
    if value != "auto":
        raise ValueError(f"unsupported dtype: {value}")
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] < 8:
        return torch.float16
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def _load_target(
    model_path: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    attention: str,
) -> tuple[Any, Any, float]:
    from transformers import AutoProcessor

    started = _now(device)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    try:
        from transformers import AutoModelForImageTextToText

        model_cls = AutoModelForImageTextToText
    except ImportError:
        from transformers import AutoModelForVision2Seq as model_cls
    try:
        target = model_cls.from_pretrained(
            model_path,
            dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attention,
            trust_remote_code=True,
        )
    except (AttributeError, TypeError, ValueError):
        try:
            from transformers import AutoModelForVision2Seq as fallback_cls
        except ImportError:
            from transformers import AutoModelForCausalLM as fallback_cls
        target = fallback_cls.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attention,
            trust_remote_code=True,
        )
    target.to(device=device, dtype=dtype).eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    return processor, target, _now(device) - started


def _compute_position_ids(target: Any, inputs: dict[str, Any]) -> torch.Tensor:
    kwargs = {
        key: inputs[key]
        for key in (
            "input_ids",
            "image_grid_thw",
            "video_grid_thw",
            "second_per_grid_ts",
            "attention_mask",
        )
        if key in inputs
    }
    candidates = [
        target,
        getattr(target, "model", None),
        getattr(getattr(target, "model", None), "language_model", None),
    ]
    failures = []
    for candidate in candidates:
        if candidate is None:
            continue
        method = getattr(candidate, "get_rope_index", None)
        if method is None:
            continue
        try:
            positions = method(**kwargs)
            if isinstance(positions, tuple):
                positions = positions[0]
            if positions.ndim == 2:
                positions = positions.unsqueeze(0).expand(3, -1, -1)
            if positions.ndim == 3 and positions.shape[0] == 3:
                return positions.to(dtype=torch.long)
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            failures.append(f"{type(candidate).__name__}.get_rope_index: {exc}")
    detail = "; ".join(failures) or "no compatible get_rope_index API"
    raise RuntimeError(f"Unable to compute Qwen2.5-VL 3-axis M-RoPE positions: {detail}")


def prepare_video_prompt(
    processor: Any,
    target: Any,
    record: dict[str, Any],
    *,
    video_root: str | Path,
    device: torch.device,
    num_frames: int,
    video_min_pixels: int,
    video_max_pixels: int,
    video_reader: str,
) -> PreparedVideoPrompt:
    from src.train_VLM.video import prepare_qwen_messages

    video_path = resolve_video_path(record, video_root)
    question = str(
        record.get("question")
        or (
            "Please provide a detailed description of the video, focusing on the "
            "main subjects, their actions, and the background scenes."
        )
    ).strip()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path.as_uri()},
                {"type": "text", "text": question},
            ],
        }
    ]
    inputs, metadata = prepare_qwen_messages(
        processor,
        messages,
        device=device,
        video_reader=video_reader,
        video_num_frames=num_frames,
        video_min_pixels=video_min_pixels,
        video_max_pixels=video_max_pixels,
    )
    input_ids = inputs.get("input_ids")
    if not torch.is_tensor(input_ids) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Qwen processor must return batch-size-one input_ids")
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
        inputs["attention_mask"] = attention_mask
    position_ids = _compute_position_ids(target, inputs)
    target_kwargs = {
        key: value
        for key, value in inputs.items()
        if key not in {"input_ids", "position_ids"}
    }
    return PreparedVideoPrompt(
        record=record,
        video_path=video_path,
        inputs=inputs,
        position_ids=position_ids,
        target_kwargs=target_kwargs,
        frame_counts=metadata.frame_counts,
        video_grid_thw=metadata.video_grid_thw,
    )


def _sha256_tokens(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(8, "big", signed=False))
    return digest.hexdigest()


def _decode_text(processor: Any, token_ids: torch.Tensor) -> str:
    tokenizer = getattr(processor, "tokenizer", processor)
    return tokenizer.decode(token_ids.detach().cpu().tolist(), skip_special_tokens=True).strip()


def _extend_position_ids(
    position_ids: torch.Tensor,
    total_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Extend 2-D or Qwen M-RoPE position IDs for cached decoding."""

    position_ids = position_ids.to(device=device, dtype=torch.long)
    if total_length < position_ids.shape[-1]:
        raise ValueError("total_length cannot be shorter than position_ids")
    if position_ids.ndim == 2:
        if position_ids.shape[0] != 1:
            raise ValueError("2-D position_ids must have batch size one")
        starts = position_ids[:, -1:] + 1
        offsets = torch.arange(
            total_length - position_ids.shape[1], device=device
        ).view(1, -1)
        return torch.cat([position_ids, starts + offsets], dim=-1)
    if position_ids.ndim == 3 and position_ids.shape[0] == 3:
        if position_ids.shape[1] != 1:
            raise ValueError("3-D position_ids must have batch size one")
        starts = position_ids[:, :, -1:] + 1
        offsets = torch.arange(
            total_length - position_ids.shape[-1], device=device
        ).view(1, 1, -1)
        return torch.cat([position_ids, starts + offsets], dim=-1)
    raise ValueError(
        "position_ids must have shape [1, sequence] or [3, 1, sequence]"
    )


def _eos_token_ids(processor: Any, target: Any) -> list[int]:
    values = (
        getattr(processor.tokenizer, "eos_token_id", None),
        getattr(processor.tokenizer, "im_end_id", None),
        getattr(getattr(target, "generation_config", None), "eos_token_id", None),
    )
    result: set[int] = set()
    for value in values:
        if isinstance(value, (list, tuple, set)):
            result.update(int(item) for item in value)
        elif value is not None:
            result.add(int(value))
    return sorted(result)


def _target_greedy(
    target: Any,
    prompt: PreparedVideoPrompt,
    *,
    max_new_tokens: int,
    stop_token_ids: list[int] | None,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    input_ids = prompt.inputs["input_ids"].to(device=device, dtype=torch.long)
    prompt_length = int(input_ids.shape[1])
    all_position_ids = _extend_position_ids(
        prompt.position_ids,
        prompt_length + max_new_tokens,
        device,
    )
    stop_ids = _as_stop_set(stop_token_ids)
    target_cache = _new_cache()
    prefill_kwargs = dict(prompt.target_kwargs)
    prefill_kwargs.update(
        {
            "input_ids": input_ids,
            "position_ids": all_position_ids[..., :prompt_length],
            "past_key_values": target_cache,
            "cache_position": torch.arange(prompt_length, device=device),
            "use_cache": True,
            "logits_to_keep": 1,
            "return_dict": True,
        }
    )

    started = _now(device)
    prefill_started = _now(device)
    with torch.inference_mode():
        try:
            target_output = target(**prefill_kwargs)
        except TypeError as exc:
            if "logits_to_keep" not in str(exc):
                raise
            prefill_kwargs.pop("logits_to_keep", None)
            target_output = target(**prefill_kwargs)
    prefill_s = _now(device) - prefill_started

    target_cache = getattr(target_output, "past_key_values", None)
    if target_cache is None:
        raise RuntimeError("Target model did not return a KV cache during prefill")

    output = input_ids.clone()
    decode_started = _now(device)
    if max_new_tokens > 0:
        next_token = target_output.logits[:, -1:, :].argmax(dim=-1)
        output = torch.cat([output, next_token], dim=1)
        stop_hit = int(next_token[0, 0].item()) in stop_ids
        while output.shape[1] - prompt_length < max_new_tokens and not stop_hit:
            cache_length = _cache_length(target_cache)
            if cache_length >= all_position_ids.shape[-1]:
                raise RuntimeError(
                    "Target KV cache exceeded the prepared position-id sequence"
                )
            step_kwargs = {
                "input_ids": output[:, -1:],
                "position_ids": all_position_ids[..., cache_length : cache_length + 1],
                "past_key_values": target_cache,
                "cache_position": torch.tensor([cache_length], device=device),
                "use_cache": True,
                "logits_to_keep": 1,
                "return_dict": True,
            }
            with torch.inference_mode():
                try:
                    target_output = target(**step_kwargs)
                except TypeError as exc:
                    if "logits_to_keep" not in str(exc):
                        raise
                    step_kwargs.pop("logits_to_keep", None)
                    target_output = target(**step_kwargs)
            target_cache = getattr(target_output, "past_key_values", None)
            if target_cache is None:
                raise RuntimeError("Target model did not return a KV cache during decode")
            next_token = target_output.logits[:, -1:, :].argmax(dim=-1)
            output = torch.cat([output, next_token], dim=1)
            stop_hit = int(next_token[0, 0].item()) in stop_ids
    decode_s = _now(device) - decode_started
    end_to_end_s = _now(device) - started
    new_tokens = output[:, prompt_length:]
    memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return output, {
        "num_output_tokens": int(new_tokens.shape[1]),
        "prefill_s": float(prefill_s),
        "decode_s": float(decode_s),
        "end_to_end_s": float(end_to_end_s),
        "tokens_per_second": int(new_tokens.shape[1]) / max(end_to_end_s, 1e-12),
        "peak_memory_bytes": memory,
    }


def _load_draft(
    checkpoint: str | Path,
    config_path: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
):
    _ensure_specforge_importable()
    from specforge.export.checkpoint_io import materialize_draft, resolve_training_state

    started = _now(device)
    state = resolve_training_state(str(checkpoint))
    draft = materialize_draft(state, str(config_path))
    del state
    gc.collect()
    draft.to(device=device, dtype=dtype).eval()
    load_s = _now(device) - started
    return draft, load_s


def _checkpoint_result(
    *,
    label: str,
    checkpoint: str,
    reference: str,
    target_ids: torch.Tensor,
    speculative: SpeculativeDecodeResult,
    processor: Any,
    target_timing: dict[str, Any],
    load_s: float,
) -> dict[str, Any]:
    prompt_length = speculative.num_input_tokens
    target_new = target_ids[:, prompt_length:]
    speculative_new = speculative.output_ids[:, prompt_length:]
    target_tokens = target_new[0].detach().cpu().tolist()
    speculative_tokens = speculative_new[0].detach().cpu().tolist()
    outputs_match = target_tokens == speculative_tokens
    text = _decode_text(processor, speculative_new[0])
    metrics = speculative.as_dict()
    timing = dict(metrics.pop("timing"))
    timing.update(
        {
            "checkpoint_load_s": float(load_s),
            "target_prefill_s": float(target_timing["prefill_s"]),
            "target_decode_s": float(target_timing["decode_s"]),
            "target_greedy_s": float(target_timing["end_to_end_s"]),
            "speedup_vs_target": target_timing["end_to_end_s"]
            / max(timing["end_to_end_s"], 1e-12),
        }
    )
    esr = target_timing["end_to_end_s"] / max(timing["end_to_end_s"], 1e-12)
    dsr = target_timing["decode_s"] / max(timing["decode_s"], 1e-12)
    return {
        "label": label,
        "checkpoint": checkpoint,
        "status": "ok" if outputs_match and text else "mismatch",
        "prediction": text,
        "outputs_match": outputs_match,
        "target_output_hash": _sha256_tokens(target_tokens),
        "speculative_output_hash": _sha256_tokens(speculative_tokens),
        "target_output_tokens": target_tokens,
        "speculative_output_tokens": speculative_tokens,
        "text_metrics": score_caption(text, reference),
        "timing": timing,
        "acceptance": {
            "rounds": metrics.pop("rounds"),
            "mean_accepted_proposals": metrics.pop("mean_accepted_proposals"),
            "mean_effective_emitted_tokens": metrics.pop("mean_effective_emitted_tokens"),
            "target_forward_calls": metrics.pop("target_forward_calls"),
            "acceptance_rounds": metrics.pop("acceptance_rounds"),
            "tau": metrics.pop("tau"),
            "tau_proposal": metrics.pop("tau_proposal"),
            "tau_effective": metrics.pop("tau_effective"),
        },
        "speedup": {"esr": float(esr), "dsr": float(dsr)},
        "num_output_tokens": metrics.pop("num_output_tokens"),
        "peak_memory_bytes": metrics.pop("peak_memory_bytes"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--checkpoint", action="append", type=Path, default=None)
    parser.add_argument("--draft-config", type=Path, default=DEFAULT_DRAFT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "no"), default="auto")
    parser.add_argument("--target-attention", default="sdpa")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--video-min-pixels", type=int, default=50176)
    parser.add_argument("--video-max-pixels", type=int, default=50176)
    parser.add_argument("--video-reader", default="torchvision")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this comparison on a GPU host")
    dtype = _resolve_dtype(args.dtype, device)
    checkpoints = tuple(args.checkpoint or DEFAULT_CHECKPOINTS)
    if len(checkpoints) != 2:
        raise ValueError("exactly two --checkpoint values are required")

    record = load_manifest_sample(args.manifest, args.sample_index)
    processor, target, target_load_s = _load_target(
        args.target_model,
        device=device,
        dtype=dtype,
        attention=args.target_attention,
    )
    prepare_started = _now(device)
    prompt = prepare_video_prompt(
        processor,
        target,
        record,
        video_root=args.video_root,
        device=device,
        num_frames=args.num_frames,
        video_min_pixels=args.video_min_pixels,
        video_max_pixels=args.video_max_pixels,
        video_reader=args.video_reader,
    )
    prepare_s = _now(device) - prepare_started
    target_output, target_timing = _target_greedy(
        target,
        prompt,
        max_new_tokens=args.max_new_tokens,
        stop_token_ids=_eos_token_ids(processor, target),
        device=device,
    )
    prompt_length = int(prompt.inputs["input_ids"].shape[1])
    target_prediction = _decode_text(processor, target_output[0, prompt_length:])
    reference = str(record.get("answer") or record.get("reference") or "")
    report: dict[str, Any] = {
        "target_model": args.target_model,
        "draft_config": str(args.draft_config),
        "device": str(device),
        "dtype": str(dtype),
        "sample_index": args.sample_index,
        "sample_id": str(
            record.get("video_name") or record.get("id") or args.sample_index
        ),
        "video": str(prompt.video_path),
        "question": str(record.get("question") or ""),
        "reference": reference,
        "preprocessing": {
            "num_frames": args.num_frames,
            "video_min_pixels": args.video_min_pixels,
            "video_max_pixels": args.video_max_pixels,
            "video_reader": args.video_reader,
            "frame_counts": list(prompt.frame_counts),
            "video_grid_thw": [list(row) for row in prompt.video_grid_thw],
        },
        "timing": {
            "target_load_s": float(target_load_s),
            "video_prepare_s": float(prepare_s),
        },
        "target_baseline": {
            "prediction": target_prediction,
            "output_tokens": target_output[0, prompt_length:].detach().cpu().tolist(),
            "output_hash": _sha256_tokens(
                target_output[0, prompt_length:].detach().cpu().tolist()
            ),
            "text_metrics": score_caption(target_prediction, reference),
            **target_timing,
        },
        "checkpoints": [],
    }

    for checkpoint in checkpoints:
        checkpoint = Path(checkpoint)
        label = checkpoint.name
        try:
            draft, load_s = _load_draft(
                checkpoint,
                args.draft_config,
                device=device,
                dtype=dtype,
            )
            decoder = InstrumentedDFlashDecoder(target, draft, device=device)
            speculative = decoder.decode(
                input_ids=prompt.inputs["input_ids"],
                position_ids=prompt.position_ids,
                target_kwargs=prompt.target_kwargs,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=_eos_token_ids(processor, target),
            )
            result = _checkpoint_result(
                label=label,
                checkpoint=str(checkpoint),
                reference=reference,
                target_ids=target_output,
                speculative=speculative,
                processor=processor,
                target_timing=target_timing,
                load_s=load_s,
            )
            report["checkpoints"].append(result)
            del decoder, draft, speculative
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as exc:
            report["checkpoints"].append(
                {
                    "label": label,
                    "checkpoint": str(checkpoint),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    report["success"] = validate_report_success(report)
    return report


def _print_report(report: dict[str, Any]) -> None:
    print(f"[sample] {report['sample_id']} video={report['video']}")
    target = report["target_baseline"]
    print(f"[target] {target['prediction']}")
    target_metrics = target["text_metrics"]
    print(
        "target metrics: "
        f"exact={target_metrics['exact_match']:.3f} "
        f"rouge_l={target_metrics['rouge_l']:.3f} "
        f"bleu={target_metrics['bleu']:.3f} "
        f"coverage={target_metrics['coverage']:.3f}"
    )
    for result in report["checkpoints"]:
        print(f"\n[{result['label']}] status={result['status']}")
        if result["status"] == "error":
            print(f"error: {result['error']}")
            continue
        print(result["prediction"])
        timing = result["timing"]
        acceptance = result["acceptance"]
        tau = acceptance["tau"]
        tau_text = f"{tau:.2f}" if tau is not None else "n/a"
        print(
            "metrics: "
            f"match={result['outputs_match']} "
            f"rouge_l={result['text_metrics']['rouge_l']:.3f} "
            f"bleu={result['text_metrics']['bleu']:.3f} "
            f"tau={tau_text} "
            f"e2e={timing['end_to_end_s']:.3f}s "
            f"tok/s={timing['tokens_per_second']:.2f} "
            f"ESR={result['speedup']['esr']:.2f}x "
            f"DSR={result['speedup']['dsr']:.2f}x"
        )
    print(f"\n[success] {report['success']}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_comparison(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    _print_report(report)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
