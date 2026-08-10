"""Online speculative decoder for Qwen3.5-4B + DFlash with exact hybrid-cache
verification.

The Qwen3.5-4B target uses a hybrid cache: 28 gated-delta-net linear-attention
layers plus 4 full-attention layers.  In Transformers 5.3.0 a cached forward
with ``seq_len > 1`` sends the linear layers through the chunked path with a
zeroed initial state, so a multi-token verification block is *not* equivalent
to sequential greedy decoding.  To keep the experiment lossless (the plan's
mandatory invariant), the decoder therefore verifies draft proposals with the
target's native one-token recurrent path, which is exactly the path used by
greedy decoding.  Drafting itself is unchanged and fully parallel (one DFlash
block of 16 tokens per round).

For reference/research the plan's original parallel block verification is also
implemented (``verify_mode="block"``) with pre-block cache cloning and accepted-
prefix replay.  It is exact for plain attention targets but is *not* exact for
the Qwen3.5 hybrid cache in this Transformers release; the runner's output-
equality gate therefore rejects it for the main experiment.
"""

from __future__ import annotations

import copy
import hashlib
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import torch

from transformers import DynamicCache


def load_qwen35_dflash_models(
    *,
    target_model: str,
    draft_model: str,
    device: torch.device | str,
) -> tuple[Any, Any, Any]:
    """Load processor, Qwen3.5 target, and DFlash draft lazily.

    Kept beside the decoder so benchmark runners do not depend on the
    Video-MME-specific acceptance CLI just to load the models.
    """

    from transformers import AutoModelForImageTextToText, AutoProcessor

    try:
        import dflash  # noqa: F401
    except ImportError:
        repo_root = Path(__file__).resolve().parents[3]
        dflash_path = repo_root / "externals" / "dflash"
        if not dflash_path.exists():
            raise
        sys.path.insert(0, str(dflash_path))
    from dflash import DFlashDraftModel

    # T4 (SM 7.5) does not provide native bfloat16 arithmetic.  Loading the
    # Qwen3.5 target and DFlash draft in float16 keeps the CUDA path usable;
    # newer GPUs retain bfloat16 for better numerical behavior.
    dtype = torch.bfloat16
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] < 8:
        dtype = torch.float16

    processor = AutoProcessor.from_pretrained(target_model, trust_remote_code=True)
    target = AutoModelForImageTextToText.from_pretrained(
        target_model, dtype=dtype, trust_remote_code=True
    )
    draft = DFlashDraftModel.from_pretrained(
        draft_model, dtype=dtype, trust_remote_code=True
    )
    target.to(device).eval()
    draft.to(device).eval()
    return processor, target, draft


def _now(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _sample_argmax(logits: torch.Tensor) -> torch.Tensor:
    """Greedy sampling (temperature == 0)."""
    return torch.argmax(logits, dim=-1)


def extract_context_feature(
    hidden_states: list[torch.Tensor] | tuple[torch.Tensor, ...],
    layer_ids: list[int],
) -> torch.Tensor:
    """Concatenate the target layer outputs used by the DFlash checkpoint.

    ``hidden_states[0]`` is the embedding output, so DFlash layer ids are
    shifted by one, exactly like ``externals/dflash/dflash/model.py``.
    """

    offset = 1
    selected = [hidden_states[layer_id + offset] for layer_id in layer_ids]
    return torch.cat(selected, dim=-1)


def clone_hybrid_cache(cache: Any) -> Any:
    """Clone a Qwen3.5 hybrid cache (attention KV + linear states)."""

    cloned = copy.copy(cache)
    for name in ("conv_states", "recurrent_states", "key_cache", "value_cache"):
        values = getattr(cache, name, None)
        if values is not None:
            setattr(
                cloned,
                name,
                [value.clone() if torch.is_tensor(value) else value for value in values],
            )
    return cloned


def sha256_tokens(ids: Iterable[int]) -> str:
    digest = hashlib.sha256()
    for token_id in ids:
        digest.update(int(token_id).to_bytes(8, "big"))
    return digest.hexdigest()


def text_anchor_positions(
    input_ids: torch.Tensor,
    vision_start_id: int,
    vision_end_id: int,
) -> torch.Tensor:
    """Positions kept by ``text_anchor``: everything outside the vision span.

    The span from the first ``vision_start`` token to the last ``vision_end``
    token is removed (boundaries, timestamps and ``video_pad`` tokens
    included), leaving only text hidden states before and after the video.
    """

    if input_ids.ndim == 2:
        input_ids = input_ids[0]
    starts = (input_ids == vision_start_id).nonzero(as_tuple=False).flatten()
    ends = (input_ids == vision_end_id).nonzero(as_tuple=False).flatten()
    keep = torch.ones_like(input_ids, dtype=torch.bool)
    if starts.numel() and ends.numel():
        first = int(starts[0])
        last = int(ends[-1])
        keep[first : last + 1] = False
    return keep.nonzero(as_tuple=False).flatten()


def _uniform_positions(positions: torch.Tensor, ratio: float) -> torch.Tensor:
    """Keep a deterministic, uniformly spread subset of positions."""

    if positions.numel() == 0 or ratio <= 0.0:
        return positions[:0]
    if ratio >= 1.0:
        return positions
    count = max(1, min(int(round(positions.numel() * ratio)), positions.numel()))
    offsets = torch.linspace(
        0,
        positions.numel() - 1,
        steps=count,
        device=positions.device,
    ).round().long()
    return positions[offsets]


def draft_context_positions(
    input_ids: torch.Tensor,
    *,
    vision_start_id: int,
    vision_end_id: int,
    video_token_id: int,
    visual_ratio: float,
    mm_token_type_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return prompt positions retained by the draft context.

    ``visual_ratio`` controls only visual positions inside the vision span;
    surrounding text is always retained.  At 1.0 the complete prompt is
    retained.  At 0.0 the complete vision span (boundaries, timestamps and
    video tokens) is removed, matching the old ``text_anchor`` behavior.
    Intermediate ratios keep a uniformly spread subset of visual tokens.
    """

    if not math.isfinite(float(visual_ratio)) or not 0.0 <= visual_ratio <= 1.0:
        raise ValueError("visual_ratio must be between 0.0 and 1.0")
    if input_ids.ndim == 2:
        input_ids = input_ids[0]
    if input_ids.ndim != 1:
        raise ValueError("input_ids must have shape [sequence] or [1, sequence]")

    prompt_len = int(input_ids.numel())
    all_positions = torch.arange(prompt_len, device=input_ids.device)
    if visual_ratio >= 1.0:
        return all_positions

    starts = (input_ids == vision_start_id).nonzero(as_tuple=False).flatten()
    ends = (input_ids == vision_end_id).nonzero(as_tuple=False).flatten()
    if not starts.numel() or not ends.numel():
        return all_positions

    first = int(starts[0])
    last = int(ends[-1])
    span = torch.arange(first, last + 1, device=input_ids.device)
    visual_mask = input_ids[span] == video_token_id
    if mm_token_type_ids is not None:
        mm_token_type_ids = mm_token_type_ids.reshape(-1).to(input_ids.device)
        if mm_token_type_ids.numel() == prompt_len:
            # Qwen's video modality is type 2.  The token-id check remains
            # useful for processors/configurations that omit modality ids.
            visual_mask |= mm_token_type_ids[span] == 2
    visual_positions = span[visual_mask]
    selected_visual = _uniform_positions(visual_positions, float(visual_ratio))

    keep = torch.ones(prompt_len, dtype=torch.bool, device=input_ids.device)
    keep[first : last + 1] = False
    keep[selected_visual] = True
    return keep.nonzero(as_tuple=False).flatten()


def _count_mm_groups(mm_token_type_ids: torch.Tensor, modality: int) -> int:
    values = mm_token_type_ids.flatten().tolist()
    groups = 0
    prev = None
    for value in values:
        if value == modality and prev != modality:
            groups += 1
        prev = value
    return groups


def split_video_grid_for_groups(
    mm_token_type_ids: torch.Tensor,
    video_grid_thw: torch.Tensor,
) -> torch.Tensor:
    """Expand ``video_grid_thw`` to one row per video span in the prompt.

    The Qwen3.5-4B processor emits one vision span (and one type-2 group) per
    temporal patch, but returns a single grid row with the full temporal
    dimension.  ``get_rope_index`` iterates one grid row per type-2 group, so
    multi-patch videos must be split into per-span rows (e.g. ``[2,14,14]`` ->
    ``[[1,14,14],[1,14,14]]``) or the model raises ``StopIteration``.
    """

    groups = _count_mm_groups(mm_token_type_ids, 2)
    rows = int(video_grid_thw.shape[0])
    if groups == rows:
        return video_grid_thw
    if rows != 1:
        raise ValueError(
            f"Cannot expand video grids: {rows} grid rows for {groups} video groups"
        )
    total_t, height, width = (int(video_grid_thw[0, i]) for i in range(3))
    if total_t % groups != 0:
        raise ValueError(f"Temporal grid {total_t} not divisible by {groups} groups")
    per_group = total_t // groups
    return torch.stack(
        [
            torch.tensor([per_group, height, width], dtype=video_grid_thw.dtype)
            for _ in range(groups)
        ]
    )


def normalize_video_inputs(prompt_inputs: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the processor outputs with a consistent video grid."""

    normalized = dict(prompt_inputs)
    mm_type_ids = normalized.get("mm_token_type_ids")
    video_grid_thw = normalized.get("video_grid_thw")
    if mm_type_ids is not None and video_grid_thw is not None:
        normalized["video_grid_thw"] = split_video_grid_for_groups(
            mm_type_ids, video_grid_thw
        )
    return normalized


@dataclass
class AcceptanceRound:
    round_index: int
    proposal_count: int
    matched_proposals: int
    effective_emitted_tokens: int
    is_partial_block: bool
    is_terminal: bool
    is_eos_truncated: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "proposal_count": self.proposal_count,
            "matched_proposals": self.matched_proposals,
            "effective_emitted_tokens": self.effective_emitted_tokens,
            "is_partial_block": self.is_partial_block,
            "is_terminal": self.is_terminal,
            "is_eos_truncated": self.is_eos_truncated,
        }


@dataclass
class DecodeResult:
    output_ids: torch.Tensor
    acceptance_rounds: list[AcceptanceRound] = field(default_factory=list)
    target_forward_calls: int = 0
    prefill_latency_s: float = 0.0
    draft_latency_s: float = 0.0
    verify_latency_s: float = 0.0
    decode_latency_s: float = 0.0
    end_to_end_latency_s: float = 0.0
    peak_memory_bytes: int = 0
    num_input_tokens: int = 0

    @property
    def num_output_tokens(self) -> int:
        return int(self.output_ids.shape[1] - self.num_input_tokens)

    def tau_proposal(self) -> Optional[float]:
        values = [
            round_.matched_proposals
            for round_ in self.acceptance_rounds
            if not round_.is_partial_block and not round_.is_terminal
        ]
        if not values:
            return None
        return sum(values) / len(values)

    def tau_effective(self) -> Optional[float]:
        values = [
            round_.effective_emitted_tokens
            for round_ in self.acceptance_rounds
            if not round_.is_partial_block and not round_.is_terminal
        ]
        if not values:
            return None
        return sum(values) / len(values)

    def full_block_rate(self) -> Optional[float]:
        full = [r for r in self.acceptance_rounds if not r.is_terminal]
        if not full:
            return None
        return sum(not r.is_partial_block for r in full) / len(full)


class Qwen35DFlashDecoder:
    """VLM-aware online DFlash decoder for Qwen3.5-4B.

    Parameters
    ----------
    target:
        ``Qwen3_5ForConditionalGeneration`` in eval mode.
    draft:
        ``DFlashDraftModel`` loaded from the DFlash checkpoint.
    device:
        Torch device used by both models.
    visual_ratio:
        Fraction of visual positions retained in the draft context.  The
        surrounding text is always retained; 1.0 keeps the full prompt and
        0.0 keeps only text outside the vision span.
    verify_mode:
        ``"exact"`` (default) verifies proposals one token at a time with the
        native recurrent path (lossless by construction).  ``"block"`` uses
        the plan's parallel block verification with cache clone/replay.
    """

    def __init__(
        self,
        target: torch.nn.Module,
        draft: torch.nn.Module,
        *,
        device: torch.device,
        visual_ratio: float = 1.0,
        context_mode: Optional[str] = None,
        verify_mode: str = "exact",
        block_size: Optional[int] = None,
        stop_token_ids: Iterable[int] = (),
    ) -> None:
        if context_mode is not None:
            # Backward-compatible bridge for callers using the old API.
            if visual_ratio != 1.0:
                raise ValueError("pass either visual_ratio or context_mode, not both")
            legacy_ratios = {"full": 1.0, "text_anchor": 0.0}
            if context_mode not in legacy_ratios:
                raise ValueError("context_mode must be 'full' or 'text_anchor'")
            visual_ratio = legacy_ratios[context_mode]
        if not math.isfinite(float(visual_ratio)) or not 0.0 <= visual_ratio <= 1.0:
            raise ValueError("visual_ratio must be between 0.0 and 1.0")
        if verify_mode not in ("exact", "block"):
            raise ValueError("verify_mode must be 'exact' or 'block'")
        self.target = target
        self.draft = draft
        self.device = device
        self.visual_ratio = float(visual_ratio)
        self.verify_mode = verify_mode
        self.block_size = int(block_size or draft.block_size)
        self.stop_token_ids = set(int(x) for x in stop_token_ids)
        self.mask_token_id = int(draft.mask_token_id)
        self.target_layer_ids = list(draft.target_layer_ids)
        self.config = target.config
        self.video_token_id = int(getattr(target.config, "video_token_id", 248057))
        self.vision_start_id = int(getattr(target.config, "vision_start_token_id", 248053))
        self.vision_end_id = int(getattr(target.config, "vision_end_token_id", 248054))
        self.input_embeddings = target.get_input_embeddings()
        self.lm_head = target.get_output_embeddings()
        if self.lm_head is None:
            self.lm_head = target.lm_head

    # ------------------------------------------------------------------ utils

    def _feature_dim(self) -> int:
        return len(self.target_layer_ids) * int(self.target.config.text_config.hidden_size)

    def _decode_position_ids(self, start: int, length: int, delta: torch.Tensor) -> torch.Tensor:
        """4D M-RoPE position ids for generated text positions."""

        text_row = torch.arange(start, start + length, device=self.device).view(1, 1, length)
        vision_rows = (text_row + delta.view(1, 1, 1)).expand(3, 1, length)
        return torch.cat([text_row, vision_rows], dim=0)

    def _target_forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cache: Any,
        cache_position: torch.Tensor,
        *,
        logits_to_keep: int,
        output_hidden_states: bool,
        total_len: int,
    ):
        # ``generate`` extends the attention mask to cover all past positions
        # at every step; the full-attention layers build their 4D causal mask
        # from it, so the same growing mask is required for exactness.
        attention_mask = torch.ones(1, total_len, device=self.device, dtype=torch.long)
        return self.target(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
            logits_to_keep=logits_to_keep,
            output_hidden_states=output_hidden_states,
            output_attentions=False,
            return_dict=True,
        )

    def _greedy_reference(
        self,
        prompt_inputs: dict[str, Any],
        *,
        max_new_tokens: int,
        stop_token_ids: Iterable[int],
    ) -> tuple[torch.Tensor, float]:
        stop = list(int(x) for x in stop_token_ids)
        inputs = {
            key: (value.to(self.device) if torch.is_tensor(value) else value)
            for key, value in prompt_inputs.items()
        }
        started = _now(self.device)
        outputs = self.target.generate(
            **inputs,
            do_sample=False,
            temperature=None,
            max_new_tokens=max_new_tokens,
            eos_token_id=stop or None,
            pad_token_id=stop[0] if stop else None,
            return_dict_in_generate=True,
            output_hidden_states=False,
            output_attentions=False,
        )
        greedy_ids = outputs.sequences[0]
        prompt_len = int(inputs["input_ids"].shape[1])
        generated = greedy_ids[prompt_len:]
        for idx, token in enumerate(generated.tolist()):
            if int(token) in self.stop_token_ids:
                generated = generated[: idx + 1]
                break
        return generated.unsqueeze(0), _now(self.device) - started

    # ------------------------------------------------------------- draft loop

    def _draft_proposals(
        self,
        *,
        anchor: torch.Tensor,
        target_hidden: torch.Tensor,
        draft_cache: Any,
        draft_position_ids: torch.Tensor,
        draft_start: int,
    ) -> tuple[torch.Tensor, Any, float]:
        block_size = self.block_size
        block_ids = torch.full(
            (1, block_size), self.mask_token_id, device=self.device, dtype=torch.long
        )
        block_ids[:, 0] = anchor
        noise_embedding = self.input_embeddings(block_ids)
        cache_len = draft_cache.get_seq_length()
        positions = draft_position_ids[:, cache_len : draft_start + block_size]
        started = _now(self.device)
        draft_hidden = self.draft(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=positions,
            past_key_values=draft_cache,
            use_cache=True,
            is_causal=False,
        )
        draft_latency = _now(self.device) - started
        draft_cache.crop(draft_start)
        logits = self.lm_head(draft_hidden)[:, 1 - block_size :, :]
        proposals = _sample_argmax(logits)[0]
        return proposals, draft_cache, draft_latency

    # ----------------------------------------------------------- verification

    def _verify_exact(
        self,
        *,
        proposals: torch.Tensor,
        anchor: torch.Tensor,
        anchor_hidden: Optional[torch.Tensor],
        current_prediction: Optional[torch.Tensor],
        cache: Any,
        anchor_pos: int,
        delta: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, Any, int, float]:
        """Verify proposals one token at a time (lossless).

        The accepted prefix (anchor + matched proposals) hidden states are
        returned for the next draft block.  A forward on the bonus token is
        also executed so that its hidden states are available as the next
        round's anchor (mirroring what the parallel block forward provides for
        free in ``verify_mode="block"``).
        """

        if anchor_hidden is None or current_prediction is None:
            # Round 1: the prefill predicted the anchor but never computed its
            # hidden states; run the exact one-token forward on the anchor.
            pos_ids = self._decode_position_ids(anchor_pos, 1, delta)
            cache_position = torch.tensor([anchor_pos], device=self.device)
            started = _now(self.device)
            output = self._target_forward(
                anchor.view(1, 1),
                pos_ids,
                cache,
                cache_position,
                logits_to_keep=1,
                output_hidden_states=True,
                total_len=anchor_pos + 1,
            )
            anchor_forward_latency = _now(self.device) - started
            anchor_hidden = extract_context_feature(output.hidden_states, self.target_layer_ids)[
                :, -1:, :
            ]
            current_prediction = _sample_argmax(output.logits[:, -1])
            anchor_calls = 1
        else:
            anchor_forward_latency = 0.0
            anchor_calls = 0

        accepted_hidden: list[torch.Tensor] = [anchor_hidden]
        verify_latency = 0.0
        calls = anchor_calls
        accepted = 0
        proposal_count = int(proposals.numel())
        while accepted < proposal_count:
            token = proposals[accepted]
            if int(current_prediction) != int(token):
                break
            position = anchor_pos + accepted + 1
            pos_ids = self._decode_position_ids(position, 1, delta)
            cache_position = torch.tensor([position], device=self.device)
            started = _now(self.device)
            output = self._target_forward(
                token.view(1, 1),
                pos_ids,
                cache,
                cache_position,
                logits_to_keep=1,
                output_hidden_states=True,
                total_len=position + 1,
            )
            verify_latency += _now(self.device) - started
            calls += 1
            current_prediction = _sample_argmax(output.logits[:, -1])
            accepted_hidden.append(
                extract_context_feature(
                    output.hidden_states, self.target_layer_ids
                )[:, -1:, :]
            )
            accepted += 1
            if int(token) in self.stop_token_ids:
                # EOS accepted: nothing may follow it, so stop this round and
                # do not run a bonus forward.
                return (
                    accepted_hidden,
                    None,
                    None,
                    None,
                    cache,
                    calls,
                    verify_latency + anchor_forward_latency,
                )

        # Bonus = the target prediction after the last accepted token.  Its
        # forward provides the next round's anchor hidden states.
        bonus = current_prediction
        if int(bonus) in self.stop_token_ids:
            return (
                accepted_hidden,
                bonus,
                None,
                None,
                cache,
                calls,
                verify_latency + anchor_forward_latency,
            )
        position = anchor_pos + accepted + 1
        pos_ids = self._decode_position_ids(position, 1, delta)
        cache_position = torch.tensor([position], device=self.device)
        started = _now(self.device)
        output = self._target_forward(
            bonus.view(1, 1),
            pos_ids,
            cache,
            cache_position,
            logits_to_keep=1,
            output_hidden_states=True,
            total_len=position + 1,
        )
        verify_latency += _now(self.device) - started
        calls += 1
        bonus_hidden = extract_context_feature(
            output.hidden_states, self.target_layer_ids
        )[:, -1:, :]
        next_prediction = _sample_argmax(output.logits[:, -1])
        return (
            accepted_hidden,
            bonus,
            bonus_hidden,
            next_prediction,
            cache,
            calls,
            verify_latency + anchor_forward_latency,
        )

    def _verify_block(
        self,
        *,
        block_ids: torch.Tensor,
        cache: Any,
        anchor_pos: int,
        delta: torch.Tensor,
        proposal_count: int,
    ) -> tuple[list[torch.Tensor], torch.Tensor, Any, int, float]:
        """Plan's parallel block verification with cache clone + replay.

        NOTE: for the Qwen3.5 hybrid cache this is not lossless in
        Transformers 5.3.0 (linear-attention state restarts per call).  Kept
        for research and tests.
        """

        length = block_ids.shape[1]
        pos_ids = self._decode_position_ids(anchor_pos, length, delta)
        cache_position = torch.arange(anchor_pos, anchor_pos + length, device=self.device)
        cache_before = clone_hybrid_cache(cache)
        started = _now(self.device)
        output = self._target_forward(
            block_ids,
            pos_ids,
            cache,
            cache_position,
            logits_to_keep=length,
            output_hidden_states=True,
            total_len=anchor_pos + length,
        )
        verify_latency = _now(self.device) - started
        calls = 1
        posterior = _sample_argmax(output.logits[0])
        # block_ids: [anchor, p1, ..., pN]; posterior[i] predicts block_ids[i+1].
        matches = (block_ids[0, 1:] == posterior[:-1]).tolist()
        accepted = 0
        while accepted < proposal_count and matches[accepted]:
            accepted += 1
        accepted_count = accepted + 1  # anchor + accepted proposals
        # Stop at the first EOS inside the accepted proposals: nothing may be
        # accepted after it.
        eos_in_accepted = None
        for idx in range(1, accepted_count):
            if int(block_ids[0, idx]) in self.stop_token_ids:
                eos_in_accepted = idx
                break
        if eos_in_accepted is not None:
            accepted_count = eos_in_accepted + 1
        if accepted_count == length:
            cache = output.past_key_values
        else:
            replay_ids = block_ids[:, :accepted_count]
            replay_pos = pos_ids[:, :, :accepted_count]
            replay_position = cache_position[:accepted_count]
            replay = self._target_forward(
                replay_ids,
                replay_pos,
                cache_before,
                replay_position,
                logits_to_keep=accepted_count,
                output_hidden_states=False,
                total_len=anchor_pos + accepted_count,
            )
            calls += 1
            cache = replay.past_key_values
        hidden = extract_context_feature(output.hidden_states, self.target_layer_ids)[
            :, :accepted_count, :
        ]
        accepted_hidden = [hidden[:, i : i + 1, :] for i in range(accepted_count)]
        if eos_in_accepted is not None:
            bonus = None
        else:
            bonus = posterior[accepted]
        return accepted_hidden, bonus, cache, calls, verify_latency

    # -------------------------------------------------------------- main loop

    @torch.inference_mode()
    def decode(
        self,
        prompt_inputs: dict[str, Any],
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
    ) -> DecodeResult:
        prompt_inputs = normalize_video_inputs(prompt_inputs)
        if temperature >= 1e-5:
            raise NotImplementedError("This decoder is currently greedy-only (temperature 0)")
        if prompt_inputs["input_ids"].shape[0] != 1:
            raise ValueError("The decoder supports batch size one")
        self.target.eval()
        self.draft.eval()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        started_at = _now(self.device)
        input_ids = prompt_inputs["input_ids"].to(self.device)
        prompt_len = int(input_ids.shape[1])
        attention_mask = prompt_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = attention_mask.to(self.device)
        mm_type_ids = prompt_inputs.get("mm_token_type_ids")
        if mm_type_ids is not None:
            mm_type_ids = mm_type_ids.to(self.device)
        video_grid_thw = prompt_inputs.get("video_grid_thw")
        if video_grid_thw is not None:
            video_grid_thw = video_grid_thw.to(self.device)

        # ---- prefill
        prefill_started = _now(self.device)
        pos3, delta = self.target.model.get_rope_index(
            input_ids,
            mm_token_type_ids=mm_type_ids if mm_type_ids is not None else torch.zeros_like(input_ids),
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        text_row = torch.arange(prompt_len, device=self.device).view(1, 1, prompt_len)
        prefill_positions = torch.cat([text_row, pos3], dim=0)
        vision_inputs = {
            key: prompt_inputs[key]
            for key in (
                "pixel_values",
                "pixel_values_videos",
                "image_grid_thw",
                "video_grid_thw",
                "video_pad",
            )
            if prompt_inputs.get(key) is not None
        }
        vision_inputs = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in vision_inputs.items()
        }
        prefill = self.target(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=prefill_positions,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True,
            **vision_inputs,
        )
        prefill_latency_s = _now(self.device) - prefill_started
        cache = prefill.past_key_values
        if cache is None:
            raise RuntimeError("Target did not return a cache")
        target_calls = 1

        # ---- draft context for the first block
        kept = draft_context_positions(
            input_ids,
            vision_start_id=self.vision_start_id,
            vision_end_id=self.vision_end_id,
            video_token_id=self.video_token_id,
            visual_ratio=self.visual_ratio,
            mm_token_type_ids=mm_type_ids,
        )
        draft_context_len = int(kept.numel())
        target_hidden = extract_context_feature(prefill.hidden_states, self.target_layer_ids)[
            :, kept, :
        ]
        if target_hidden.shape[1] != draft_context_len:
            raise RuntimeError("Draft context length mismatch")
        draft_position_ids = torch.arange(
            draft_context_len + max_new_tokens + self.block_size, device=self.device
        ).unsqueeze(0)
        draft_cache = DynamicCache()

        anchor = _sample_argmax(prefill.logits[:, -1])
        output_ids = torch.cat([input_ids, anchor.view(1, 1)], dim=1)
        stop_ids = self.stop_token_ids
        generated = 1
        draft_start = draft_context_len
        acceptance_rounds: list[AcceptanceRound] = []
        draft_latency_s = 0.0
        verify_latency_s = 0.0
        anchor_pos = prompt_len
        anchor_hidden: Optional[torch.Tensor] = None
        current_prediction: Optional[torch.Tensor] = None
        stopped = bool(int(anchor) in stop_ids)

        while not stopped and generated < max_new_tokens:
            remaining = max_new_tokens - generated
            proposal_count = min(self.block_size - 1, max(0, remaining - 1))
            round_index = len(acceptance_rounds)
            proposals = torch.empty(0, device=self.device, dtype=torch.long)
            if proposal_count > 0:
                proposals, draft_cache, latency = self._draft_proposals(
                    anchor=anchor,
                    target_hidden=target_hidden,
                    draft_cache=draft_cache,
                    draft_position_ids=draft_position_ids,
                    draft_start=draft_start,
                )
                draft_latency_s += latency
                proposals = proposals[:proposal_count]

            if self.verify_mode == "exact":
                accepted_hidden, bonus, bonus_hidden, next_prediction, cache, calls, latency = (
                    self._verify_exact(
                    proposals=proposals,
                    anchor=anchor,
                    anchor_hidden=anchor_hidden,
                    current_prediction=current_prediction,
                    cache=cache,
                    anchor_pos=anchor_pos,
                    delta=delta,
                )
                )
                matched = len(accepted_hidden) - 1
                emitted = (
                    torch.cat([proposals[:matched], bonus.view(1)], dim=0)
                    if bonus is not None
                    else proposals[:matched]
                )
                anchor = bonus
                anchor_hidden = bonus_hidden
                current_prediction = next_prediction
            else:
                block_ids = torch.cat([anchor.view(1, 1), proposals.view(1, -1)], dim=1)
                accepted_hidden, bonus, cache, calls, latency = self._verify_block(
                    block_ids=block_ids,
                    cache=cache,
                    anchor_pos=anchor_pos,
                    delta=delta,
                    proposal_count=proposal_count,
                )
                matched = len(accepted_hidden) - 1
                emitted = (
                    torch.cat([proposals[:matched], bonus.view(1)], dim=0)
                    if bonus is not None
                    else proposals[:matched]
                )
                anchor = bonus
                anchor_hidden = None
                current_prediction = None
            target_calls += calls
            verify_latency_s += latency

            # accepted prefix hidden states feed the next draft block
            target_hidden = torch.cat(accepted_hidden, dim=1)
            eos_index = next(
                (idx for idx, token in enumerate(emitted.tolist()) if int(token) in stop_ids),
                None,
            )
            eos_truncated = eos_index is not None
            if eos_truncated:
                emitted = emitted[: eos_index + 1]
            output_ids = torch.cat([output_ids, emitted.view(1, -1)], dim=1)
            generated += int(emitted.numel())
            # ``anchor`` is the last emitted bonus token and its absolute
            # position advances with every token emitted in this round.  The
            # cache has already consumed the same prefix; keeping the old
            # prompt position here misaligns M-RoPE on the next round and can
            # make exact verification diverge from greedy decoding.
            anchor_pos += int(emitted.numel())
            draft_start += matched + 1

            is_partial = proposal_count < self.block_size - 1
            is_terminal = eos_truncated or generated >= max_new_tokens
            acceptance_rounds.append(
                AcceptanceRound(
                    round_index=round_index,
                    proposal_count=proposal_count,
                    matched_proposals=matched,
                    effective_emitted_tokens=int(emitted.numel()),
                    is_partial_block=is_partial,
                    is_terminal=is_terminal,
                    is_eos_truncated=eos_truncated,
                )
            )
            if eos_truncated:
                stopped = True

        finished_at = _now(self.device)
        peak_memory = (
            torch.cuda.max_memory_allocated(self.device)
            if self.device.type == "cuda"
            else 0
        )
        return DecodeResult(
            output_ids=output_ids,
            acceptance_rounds=acceptance_rounds,
            target_forward_calls=target_calls,
            prefill_latency_s=prefill_latency_s,
            draft_latency_s=draft_latency_s,
            verify_latency_s=verify_latency_s,
            decode_latency_s=draft_latency_s + verify_latency_s,
            end_to_end_latency_s=finished_at - started_at,
            peak_memory_bytes=int(peak_memory),
            num_input_tokens=prompt_len,
        )

    def greedy_reference(
        self,
        prompt_inputs: dict[str, Any],
        *,
        max_new_tokens: int,
    ) -> tuple[torch.Tensor, float]:
        """Independent greedy reference via ``model.generate``."""

        prompt_inputs = normalize_video_inputs(prompt_inputs)
        return self._greedy_reference(
            prompt_inputs,
            max_new_tokens=max_new_tokens,
            stop_token_ids=self.stop_token_ids,
        )
