"""Unverified DFlash rollouts with explicit target-feature accounting.

In the primary prompt-tail mode, target forwards update conditioning features;
their logits never choose, replace, or reject draft proposals. An optional
pilot control may prepend one common target-greedy seed, excluded from scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn.functional as F

from src.infer.qwen25vl_dflash_compare import (
    InstrumentedDFlashDecoder,
    _as_stop_set,
    _module_device,
    _move_tensor_kwargs,
    _new_cache,
    _now,
)


def _align_target_token_ids(probabilities: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Move target token indices to the device holding their draft probabilities."""
    return token_ids.to(device=probabilities.device, dtype=torch.long)


@dataclass
class FreeRunningResult:
    """One complete raw output and provenance for each generated block."""

    output_ids: torch.Tensor
    num_input_tokens: int
    initial_anchor: str
    feature_refresh: str
    excluded_prefix_tokens: int = 0
    target_seed_token_id: int | None = None
    stop_reason: str = "max_new_tokens"
    blocks: list[dict[str, Any]] = field(default_factory=list)
    target_feature_calls: int = 0
    prefill_s: float = 0.0
    feature_refresh_s: float = 0.0
    draft_s: float = 0.0
    end_to_end_s: float = 0.0

    @property
    def num_output_tokens(self) -> int:
        return max(
            0,
            int(self.output_ids.shape[1] - self.num_input_tokens - self.excluded_prefix_tokens),
        )

    def as_dict(self) -> dict[str, Any]:
        """Describe raw generation without reporting acceptance or tau."""
        return {
            "generation_mode": "unverified_draft_rollout",
            "initial_anchor": self.initial_anchor,
            "feature_refresh": self.feature_refresh,
            "verification_calls": 0,
            "target_selected_output_tokens": int(self.excluded_prefix_tokens),
            "target_seed_token_id": self.target_seed_token_id,
            "target_feature_calls": int(self.target_feature_calls),
            "num_output_tokens": self.num_output_tokens,
            "stop_reason": self.stop_reason,
            "blocks": self.blocks,
            "timing": {
                "prefill_s": float(self.prefill_s),
                "feature_refresh_s": float(self.feature_refresh_s),
                "draft_s": float(self.draft_s),
                "end_to_end_s": float(self.end_to_end_s),
            },
        }


class FreeRunningDFlashDecoder(InstrumentedDFlashDecoder):
    """Emit every proposal and refresh features without verifying the prefix.

    Each block uses the last known token as its anchor and the target features
    for the preceding context. The primary first anchor is the final prompt
    token; a pilot control can instead use one common target-greedy seed. After
    a draft block, the target processes that anchor and the proposals before
    the next anchor to make correctly aligned features for the following block.
    """

    @torch.inference_mode()
    def decode(
        self,
        *,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        target_kwargs: dict[str, Any] | None,
        max_new_tokens: int,
        stop_token_ids: list[int] | None,
        context_keep_mask: torch.Tensor | None = None,
        prefill_target_hidden_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        feature_refresh: str = "refresh",
        initial_anchor: str = "last_prompt_token",
    ) -> FreeRunningResult:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("input_ids must have shape [1, sequence]")
        if input_ids.shape[1] < 2:
            raise ValueError("free-running DFlash needs an anchor and preceding context")
        if position_ids.ndim not in (2, 3) or position_ids.shape[-1] != input_ids.shape[1]:
            raise ValueError("position_ids must match the prompt sequence length")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if feature_refresh not in {"refresh", "frozen"}:
            raise ValueError("feature_refresh must be 'refresh' or 'frozen'")
        if initial_anchor not in {"last_prompt_token", "target_greedy_seed"}:
            raise ValueError("initial_anchor must be 'last_prompt_token' or 'target_greedy_seed'")
        block_size = int(self.draft.block_size)
        if block_size < 2:
            raise ValueError("DFlash block_size must be at least two")

        input_ids = input_ids.to(device=self.device, dtype=torch.long)
        prompt_length = int(input_ids.shape[1])
        keep = torch.ones(prompt_length, dtype=torch.bool, device=self.draft_device)
        if context_keep_mask is not None:
            if context_keep_mask.ndim != 1 or context_keep_mask.numel() != prompt_length:
                raise ValueError("context_keep_mask must match the prompt length")
            keep = context_keep_mask.to(device=self.draft_device, dtype=torch.bool)
        if initial_anchor == "last_prompt_token":
            if not bool(keep[-1]):
                raise ValueError("context_keep_mask cannot remove the initial anchor")
            context_keep = keep[:-1]
        else:
            context_keep = keep
        if not bool(context_keep.any()):
            raise ValueError("context_keep_mask must retain context rows")

        result = FreeRunningResult(
            output_ids=input_ids.clone(),
            num_input_tokens=prompt_length,
            initial_anchor=initial_anchor,
            feature_refresh=feature_refresh,
        )
        if max_new_tokens == 0:
            return result

        started = _now(self.device)
        all_positions = self._extend_positions(
            position_ids,
            prompt_length + max_new_tokens + block_size,
        )
        prefill_length = prompt_length - (initial_anchor == "last_prompt_token")
        context_positions = all_positions[..., :prefill_length][
            ..., context_keep.to(device=self.device)
        ].to(device=self.draft_device)

        prefill_kwargs = _move_tensor_kwargs(dict(target_kwargs or {}), self.device)
        attention_mask = prefill_kwargs.get("attention_mask")
        if torch.is_tensor(attention_mask):
            prefill_kwargs["attention_mask"] = attention_mask[..., :prefill_length]
        prefill_kwargs.update(
            {
                "input_ids": input_ids[:, :prefill_length],
                "position_ids": all_positions[..., :prefill_length],
                "past_key_values": _new_cache(),
                "cache_position": torch.arange(prefill_length, device=self.device),
                "use_cache": True,
                "logits_to_keep": 1,
                "return_dict": True,
            }
        )
        tick = _now(self.device)
        target_output, context = self._target_forward_with_context(prefill_kwargs)
        result.prefill_s = _now(self.device) - tick
        result.target_feature_calls = 1
        target_cache = getattr(target_output, "past_key_values", None)
        target_seed = (
            target_output.logits[:, -1:, :].argmax(dim=-1).to(self.device)
            if initial_anchor == "target_greedy_seed"
            else None
        )
        del target_output

        if prefill_target_hidden_transform is not None:
            transformed = prefill_target_hidden_transform(context)
            if (
                not torch.is_tensor(transformed)
                or transformed.shape != context.shape
            ):
                raise ValueError(
                    "prefill_target_hidden_transform must preserve the context shape"
                )
            context = transformed
        context = context[:, context_keep.to(device=context.device), :]

        embedding = self.target.get_input_embeddings()
        embedding_device = _module_device(embedding, self.device)
        lm_head = getattr(self.target, "lm_head", None)
        if lm_head is None:
            lm_head = self.target.get_output_embeddings()
        lm_head_device = _module_device(lm_head, self.device)
        stop_ids = _as_stop_set(stop_token_ids)

        if target_seed is not None:
            result.target_seed_token_id = int(target_seed.item())
            result.excluded_prefix_tokens = 1
            result.output_ids = torch.cat([result.output_ids, target_seed], dim=1)
            if result.target_seed_token_id in stop_ids:
                result.stop_reason = "eos"
                result.end_to_end_s = _now(self.device) - started
                return result

        while result.num_output_tokens < max_new_tokens:
            anchor_position = int(result.output_ids.shape[1] - 1)
            block = torch.full(
                (1, block_size),
                int(self.draft.mask_token_id),
                dtype=torch.long,
                device=self.device,
            )
            block[:, :1] = result.output_ids[:, -1:]
            block_positions = all_positions[..., anchor_position:anchor_position + block_size]
            noise_embedding = embedding(block.to(device=embedding_device)).to(
                device=self.draft_device
            )

            tick = _now(self.device)
            draft_hidden = self.draft(
                target_hidden=context,
                noise_embedding=noise_embedding,
                position_ids=torch.cat(
                    [context_positions, block_positions.to(device=self.draft_device)],
                    dim=-1,
                ),
                past_key_values=None,
                use_cache=False,
                is_causal=False,
            )
            proposals = self.draft._sample_draft_tokens(
                self.target,
                draft_hidden.to(device=lm_head_device),
                block.to(device=lm_head_device),
            ).to(device=self.device, dtype=torch.long)
            result.draft_s += _now(self.device) - tick
            expected_shape = (1, block_size - 1)
            if tuple(proposals.shape) != expected_shape:
                raise RuntimeError(
                    "DFlash must return block_size - 1 proposal tokens; "
                    f"expected {expected_shape}, got {tuple(proposals.shape)}"
                )

            remaining = max_new_tokens - result.num_output_tokens
            emitted = proposals[:, :remaining]
            eos_offset = next(
                (
                    index
                    for index, token_id in enumerate(emitted[0].tolist())
                    if int(token_id) in stop_ids
                ),
                None,
            )
            if eos_offset is not None:
                emitted = emitted[:, :eos_offset + 1]
                result.stop_reason = "eos"

            result.blocks.append(
                {
                    "block_index": len(result.blocks),
                    "answer_position_start": int(
                        anchor_position + 1 - prompt_length - result.excluded_prefix_tokens
                    ),
                    "anchor_token_id": int(block[0, 0].item()),
                    "proposed_token_ids": proposals[0].detach().cpu().tolist(),
                    "emitted_token_ids": emitted[0].detach().cpu().tolist(),
                    "is_partial_block": int(emitted.shape[1]) < block_size - 1,
                    "is_terminal": bool(
                        eos_offset is not None
                        or result.num_output_tokens + emitted.shape[1] >= max_new_tokens
                    ),
                    "target_tokens_selected": 0,
                }
            )
            result.output_ids = torch.cat([result.output_ids, emitted], dim=1)

            if eos_offset is not None or result.num_output_tokens >= max_new_tokens:
                break
            if feature_refresh == "frozen":
                continue

            # Leave the final newly proposed token as the next anchor. Refresh
            # target features for the old anchor and preceding proposals only.
            refresh_ids = torch.cat([block[:, :1], emitted[:, :-1]], dim=1)
            refresh_positions = all_positions[
                ..., anchor_position:anchor_position + refresh_ids.shape[1]
            ]
            refresh_kwargs = {
                "input_ids": refresh_ids,
                "position_ids": refresh_positions,
                "past_key_values": target_cache,
                "cache_position": torch.arange(
                    anchor_position,
                    anchor_position + refresh_ids.shape[1],
                    device=self.device,
                ),
                "use_cache": True,
                "logits_to_keep": 1,
                "return_dict": True,
            }
            tick = _now(self.device)
            refreshed_output, new_context = self._target_forward_with_context(refresh_kwargs)
            result.feature_refresh_s += _now(self.device) - tick
            result.target_feature_calls += 1
            target_cache = getattr(refreshed_output, "past_key_values", target_cache)
            del refreshed_output
            context = torch.cat([context, new_context], dim=1)
            context_positions = torch.cat(
                [context_positions, refresh_positions.to(device=self.draft_device)],
                dim=-1,
            )

        result.end_to_end_s = _now(self.device) - started
        return result

    @torch.inference_mode()
    def probe_teacher_forced(
        self,
        *,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        target_kwargs: dict[str, Any] | None,
        answer_ids: torch.Tensor,
        context_keep_mask: torch.Tensor | None = None,
        feature_refresh: str = "refresh",
    ) -> dict[str, Any]:
        """Probe per-position draft proposals on one fixed teacher-forced prefix.

        This is a sensitivity diagnostic, not a free-running output or an
        acceptance measurement. Every retention condition sees the same answer
        prefix, so changed proposals localize draft decisions sensitive to the
        retained target-hidden rows without conflating prefix drift.
        """
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("input_ids must have shape [1, sequence]")
        if answer_ids.ndim == 2:
            if answer_ids.shape[0] != 1:
                raise ValueError("answer_ids must have batch size one")
            answer_ids = answer_ids[0]
        if answer_ids.ndim != 1:
            raise ValueError("answer_ids must have shape [sequence] or [1, sequence]")
        if feature_refresh not in {"refresh", "frozen"}:
            raise ValueError("feature_refresh must be 'refresh' or 'frozen'")
        if input_ids.shape[1] < 2:
            raise ValueError("the prompt must contain preceding context and an anchor")
        if position_ids.shape[-1] != input_ids.shape[1]:
            raise ValueError("position_ids must match the prompt sequence length")
        if answer_ids.numel() == 0:
            return {"probe_positions": [], "target_feature_calls": 0}

        input_ids = input_ids.to(device=self.device, dtype=torch.long)
        answer_ids = answer_ids.to(device=self.device, dtype=torch.long)
        prompt_length = int(input_ids.shape[1])
        block_size = int(self.draft.block_size)
        keep = torch.ones(prompt_length, dtype=torch.bool, device=self.draft_device)
        if context_keep_mask is not None:
            if context_keep_mask.ndim != 1 or context_keep_mask.numel() != prompt_length:
                raise ValueError("context_keep_mask must match the prompt length")
            keep = context_keep_mask.to(device=self.draft_device, dtype=torch.bool)
        if not bool(keep[-1]) or not bool(keep[:-1].any()):
            raise ValueError("retention must preserve the anchor and preceding context")

        total_answer = int(answer_ids.numel())
        if int(self.draft.block_size) < 2:
            raise ValueError("DFlash block_size must be at least two")
        all_positions = self._extend_positions(
            position_ids, prompt_length + total_answer + block_size,
        )
        preceding_keep = keep[:-1]
        context_positions = all_positions[..., :prompt_length - 1][
            ..., preceding_keep.to(device=self.device)
        ].to(device=self.draft_device)
        prefill_kwargs = _move_tensor_kwargs(dict(target_kwargs or {}), self.device)
        attention_mask = prefill_kwargs.get("attention_mask")
        if torch.is_tensor(attention_mask):
            prefill_kwargs["attention_mask"] = attention_mask[..., :-1]
        prefill_kwargs.update({
            "input_ids": input_ids[:, :-1],
            "position_ids": all_positions[..., :prompt_length - 1],
            "past_key_values": _new_cache(),
            "cache_position": torch.arange(prompt_length - 1, device=self.device),
            "use_cache": True,
            "logits_to_keep": 1,
            "return_dict": True,
        })
        target_output, context = self._target_forward_with_context(prefill_kwargs)
        target_cache = getattr(target_output, "past_key_values", None)
        del target_output
        context = context[:, preceding_keep.to(device=context.device), :]
        embedding = self.target.get_input_embeddings()
        embedding_device = _module_device(embedding, self.device)
        lm_head = getattr(self.target, "lm_head", None)
        if lm_head is None:
            lm_head = self.target.get_output_embeddings()
        lm_head_device = _module_device(lm_head, self.device)

        probe_positions = []
        offset = 0
        anchor = input_ids[:, -1:]
        target_feature_calls = 1
        while offset < total_answer:
            anchor_position = prompt_length - 1 + offset
            block = torch.full(
                (1, block_size), int(self.draft.mask_token_id),
                dtype=torch.long, device=self.device,
            )
            block[:, :1] = anchor
            block_positions = all_positions[..., anchor_position:anchor_position + block_size]
            noise_embedding = embedding(block.to(device=embedding_device)).to(
                device=self.draft_device
            )
            draft_hidden = self.draft(
                target_hidden=context,
                noise_embedding=noise_embedding,
                position_ids=torch.cat(
                    [context_positions, block_positions.to(device=self.draft_device)],
                    dim=-1,
                ),
                past_key_values=None,
                use_cache=False,
                is_causal=False,
            )
            proposals = self.draft._sample_draft_tokens(
                self.target,
                draft_hidden.to(device=lm_head_device),
                block.to(device=lm_head_device),
            ).to(device=self.device, dtype=torch.long)
            count = min(block_size - 1, total_answer - offset)
            reference = answer_ids[offset:offset + count]
            proposed = proposals[0, :count]
            draft_logits = lm_head(
                draft_hidden[:, -block_size + 1 :, :].to(device=lm_head_device)
            )[:, :count, :].float()
            draft_log_probs = F.log_softmax(draft_logits, dim=-1)
            draft_probs = draft_log_probs.exp()
            top_k = min(32, int(draft_probs.shape[-1]))
            top_probs, top_ids = torch.topk(draft_probs[0], k=top_k, dim=-1)
            top_values = torch.topk(
                draft_logits[0], k=min(2, int(draft_logits.shape[-1])), dim=-1
            ).values
            entropy = -(draft_probs * draft_log_probs).sum(dim=-1)[0]
            target_probs = draft_probs[0].gather(
                -1, _align_target_token_ids(draft_probs, reference).unsqueeze(-1)
            ).squeeze(-1)
            tail_mass = (1.0 - top_probs.sum(dim=-1)).clamp_min(0.0)
            distribution_statistics = [
                {
                    "draft_top_token_ids": top_ids[index].detach().cpu().tolist(),
                    "draft_top_token_probabilities": top_probs[index].detach().cpu().tolist(),
                    "draft_tail_probability_mass": float(tail_mass[index].item()),
                    "draft_top1_probability": float(top_probs[index, 0].item()),
                    "draft_top1_margin": float(
                        (top_values[index, 0] - top_values[index, 1]).item()
                    ) if top_values.shape[-1] > 1 else None,
                    "draft_entropy_nats": float(entropy[index].item()),
                    "draft_target_prefix_probability": float(target_probs[index].item()),
                }
                for index in range(count)
            ]
            probe_positions.append({
                "answer_position_start": offset,
                "reference_token_ids": reference.detach().cpu().tolist(),
                "proposed_token_ids": proposed.detach().cpu().tolist(),
                "proposal_matches_reference": proposed.eq(reference).detach().cpu().tolist(),
                "proposal_distribution_statistics": distribution_statistics,
            })
            offset += count
            if offset >= total_answer:
                break
            anchor = reference[-1:].view(1, 1)
            if feature_refresh == "frozen":
                continue

            refresh_ids = torch.cat([block[:, :1], reference[:-1].view(1, -1)], dim=1)
            refresh_positions = all_positions[
                ..., anchor_position:anchor_position + refresh_ids.shape[1]
            ]
            refresh_kwargs = {
                "input_ids": refresh_ids,
                "position_ids": refresh_positions,
                "past_key_values": target_cache,
                "cache_position": torch.arange(
                    anchor_position, anchor_position + refresh_ids.shape[1],
                    device=self.device,
                ),
                "use_cache": True,
                "logits_to_keep": 1,
                "return_dict": True,
            }
            refreshed_output, new_context = self._target_forward_with_context(refresh_kwargs)
            target_cache = getattr(refreshed_output, "past_key_values", target_cache)
            del refreshed_output
            target_feature_calls += 1
            context = torch.cat([context, new_context], dim=1)
            context_positions = torch.cat(
                [context_positions, refresh_positions.to(device=self.draft_device)], dim=-1
            )

        return {
            "probe_positions": probe_positions,
            "target_feature_calls": target_feature_calls,
            "feature_refresh": feature_refresh,
        }
