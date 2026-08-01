from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable

import torch

from .data import select_context_positions
from .decode import _sample_logits
from .target import Qwen25VLTargetAdapter


def _now(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


@dataclass
class VLMDecodeResult:
    output_ids: torch.Tensor
    acceptance_lengths: list[int]
    target_forward_calls: int
    prefill_latency_s: float
    draft_latency_s: float
    verify_latency_s: float
    decode_latency_s: float
    end_to_end_latency_s: float
    peak_memory_bytes: int
    num_input_tokens: int

    @property
    def mean_acceptance_length(self) -> float:
        return sum(self.acceptance_lengths) / max(1, len(self.acceptance_lengths))

    @property
    def num_output_tokens(self) -> int:
        return int(self.output_ids.shape[1] - self.num_input_tokens)

    @property
    def decoding_tokens_per_second(self) -> float:
        return self.num_output_tokens / max(self.decode_latency_s, 1e-12)

    @property
    def end_to_end_tokens_per_second(self) -> float:
        return self.num_output_tokens / max(self.end_to_end_latency_s, 1e-12)

    def metrics(self) -> dict[str, float]:
        return {
            "num_input_tokens": float(self.num_input_tokens),
            "num_output_tokens": float(self.num_output_tokens),
            "target_forward_calls": float(self.target_forward_calls),
            "accepted_prefix": self.mean_acceptance_length,
            "prefill_latency_s": self.prefill_latency_s,
            "draft_latency_s": self.draft_latency_s,
            "verify_latency_s": self.verify_latency_s,
            "decode_latency_s": self.decode_latency_s,
            "end_to_end_latency_s": self.end_to_end_latency_s,
            "decoding_tokens_per_second": self.decoding_tokens_per_second,
            "end_to_end_tokens_per_second": self.end_to_end_tokens_per_second,
            "peak_memory_bytes": float(self.peak_memory_bytes),
        }


class Qwen25VLDFlashDecoder:
    """Qwen2.5-VL DFlash decoder with exact cache rollback and timing metrics.

    The target prefill consumes multimodal inputs once. Every subsequent target
    forward verifies an anchor plus proposed tokens, keeps the exact accepted
    prefix in the target cache, and emits no token after EOS.
    """

    def __init__(self, adapter: Qwen25VLTargetAdapter, draft_model, config):
        self.adapter = adapter
        self.draft_model = draft_model.eval()
        self.config = config
        self.image_ids, self.video_ids = adapter.visual_token_ids

    def _sequence_inputs(self, base_inputs: dict[str, Any], sequence: torch.Tensor) -> dict[str, Any]:
        inputs = {
            key: (value.clone() if torch.is_tensor(value) else value)
            for key, value in base_inputs.items()
        }
        inputs["input_ids"] = sequence
        inputs["attention_mask"] = torch.ones_like(sequence)
        inputs.pop("cache_position", None)
        inputs.pop("position_ids", None)
        inputs["position_ids"] = self.adapter._compute_position_ids(inputs)
        return inputs

    def _result(
        self,
        output: torch.Tensor,
        acceptance_lengths: list[int],
        target_calls: int,
        *,
        prefill_latency_s: float,
        draft_latency_s: float,
        verify_latency_s: float,
        started_at: float,
        prompt_length: int,
    ) -> VLMDecodeResult:
        finished_at = _now(self.adapter.device)
        peak_memory = (
            torch.cuda.max_memory_allocated(self.adapter.device)
            if self.adapter.device.type == "cuda"
            else 0
        )
        return VLMDecodeResult(
            output_ids=output,
            acceptance_lengths=acceptance_lengths,
            target_forward_calls=target_calls,
            prefill_latency_s=prefill_latency_s,
            draft_latency_s=draft_latency_s,
            verify_latency_s=verify_latency_s,
            decode_latency_s=draft_latency_s + verify_latency_s,
            end_to_end_latency_s=finished_at - started_at,
            peak_memory_bytes=int(peak_memory),
            num_input_tokens=prompt_length,
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt_inputs: dict[str, Any],
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        stop_token_ids: Iterable[int] | None = None,
        generator: torch.Generator | None = None,
    ) -> VLMDecodeResult:
        if prompt_inputs["input_ids"].shape[0] != 1:
            raise ValueError("The v1 VLM decoder supports batch size one")
        self.adapter.model.eval()
        self.draft_model.eval()
        if self.adapter.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.adapter.device)
        started_at = _now(self.adapter.device)
        stop = set(int(x) for x in (stop_token_ids or ()))
        output = prompt_inputs["input_ids"].clone()
        prompt_length = int(output.shape[1])
        prefill_inputs = self._sequence_inputs(prompt_inputs, output)
        prefill_started = _now(self.adapter.device)
        target_outputs = self.adapter.model(
            **prefill_inputs,
            output_hidden_states=True,
            output_attentions=False,
            use_cache=True,
            return_dict=True,
        )
        prefill_latency_s = _now(self.adapter.device) - prefill_started
        target_cache = target_outputs.past_key_values
        if target_cache is None:
            raise RuntimeError("Target model did not return a KV cache")
        target_calls = 1
        pending_anchor = _sample_logits(target_outputs.logits[:, -1], temperature, generator=generator)[0]
        acceptance_lengths: list[int] = []
        draft_latency_s = 0.0
        verify_latency_s = 0.0
        draft_context_cache = None
        block_size = int(self.config.block_size)
        mask_id = int(self.draft_model.mask_token_id)
        layer_ids = getattr(self.draft_model, "target_layer_ids")
        selected = self.adapter.selected_hidden_features(target_outputs, layer_ids)
        full_position_ids = prefill_inputs["position_ids"]
        context_positions = select_context_positions(
            output[0],
            context_mode=self.config.context_mode,
            image_token_ids=self.image_ids,
            video_token_ids=self.video_ids,
        )
        context_hidden = selected[:, context_positions]
        context_position_ids = full_position_ids[:, :, context_positions]
        context_original_positions = context_positions
        full_context_length = prompt_length
        last_position = full_position_ids[:, :, -1:]

        if max_new_tokens <= 0:
            return self._result(
                output, acceptance_lengths, target_calls,
                prefill_latency_s=prefill_latency_s,
                draft_latency_s=draft_latency_s,
                verify_latency_s=verify_latency_s,
                started_at=started_at,
                prompt_length=prompt_length,
            )
        output = torch.cat([output, pending_anchor.view(1, 1)], dim=1)
        if int(pending_anchor) in stop:
            return self._result(
                output, acceptance_lengths, target_calls,
                prefill_latency_s=prefill_latency_s,
                draft_latency_s=draft_latency_s,
                verify_latency_s=verify_latency_s,
                started_at=started_at,
                prompt_length=prompt_length,
            )

        while output.shape[1] - prompt_length < max_new_tokens:
            generated = int(output.shape[1] - prompt_length)
            remaining = max_new_tokens - generated
            proposal_count = min(block_size - 1, max(0, remaining - 1))
            offsets = torch.arange(1, block_size + 1, device=output.device).view(1, 1, -1)
            block_position_ids = last_position + offsets
            if proposal_count:
                draft_started = _now(self.adapter.device)
                block_ids = torch.full(
                    (1, block_size), mask_id, device=output.device, dtype=output.dtype
                )
                block_ids[:, 0] = pending_anchor
                noise_embeddings = self.adapter.input_embeddings(block_ids)
                anchors = torch.tensor([full_context_length], device=output.device, dtype=torch.long)
                draft_hidden, draft_context_cache = self.draft_model(
                    noise_embeddings=noise_embeddings,
                    target_context=context_hidden,
                    context_position_ids=context_position_ids,
                    block_position_ids=block_position_ids,
                    anchors=anchors,
                    context_original_positions=context_original_positions,
                    use_flex_attention=self.config.use_flex_attention,
                    draft_context_cache=draft_context_cache,
                    return_draft_context_cache=True,
                )
                draft_logits = self.adapter.lm_head(draft_hidden)[0, 1 : proposal_count + 1]
                proposals = draft_logits.argmax(dim=-1)
                draft_latency_s += _now(self.adapter.device) - draft_started
            else:
                proposals = torch.empty(0, device=output.device, dtype=output.dtype)

            verify_ids = torch.cat([pending_anchor.view(1, 1), proposals.view(1, -1)], dim=1)
            verify_length = verify_ids.shape[1]
            verify_inputs = {
                "input_ids": verify_ids,
                "position_ids": block_position_ids[:, :, :verify_length],
                "past_key_values": target_cache,
                "cache_position": torch.arange(
                    full_context_length,
                    full_context_length + verify_length,
                    device=output.device,
                ),
            }
            verify_started = _now(self.adapter.device)
            target_outputs = self.adapter.model(
                **verify_inputs,
                output_hidden_states=True,
                output_attentions=False,
                use_cache=True,
                return_dict=True,
            )
            verify_latency_s += _now(self.adapter.device) - verify_started
            target_calls += 1
            posterior_logits = target_outputs.logits[0, : proposal_count + 1]
            posterior = _sample_logits(posterior_logits, temperature, generator=generator)
            matches = proposals.eq(posterior[:-1])
            accepted = 0
            while accepted < proposal_count and bool(matches[accepted]):
                accepted += 1
            bonus = posterior[accepted]
            accepted_input = torch.cat([pending_anchor.view(1), proposals[:accepted]], dim=0)
            accepted_input_count = int(accepted_input.numel())
            new_selected = self.adapter.selected_hidden_features(target_outputs, layer_ids)
            context_hidden = torch.cat([context_hidden, new_selected[:, :accepted_input_count]], dim=1)
            context_position_ids = torch.cat(
                [context_position_ids, block_position_ids[:, :, :accepted_input_count]], dim=2
            )
            context_original_positions = torch.cat(
                [
                    context_original_positions,
                    torch.arange(
                        full_context_length,
                        full_context_length + accepted_input_count,
                        device=output.device,
                    ),
                ]
            )
            full_context_length += accepted_input_count
            last_position = block_position_ids[:, :, accepted_input_count - 1 : accepted_input_count]
            target_cache = target_outputs.past_key_values
            if target_cache is None or not hasattr(target_cache, "crop"):
                raise RuntimeError("Target verification cache does not support exact crop rollback")
            target_cache.crop(full_context_length)

            emitted = torch.cat([proposals[:accepted], bonus.view(1)])
            stop_index = next((idx for idx, token in enumerate(emitted.tolist()) if int(token) in stop), None)
            if stop_index is not None:
                emitted = emitted[: stop_index + 1]
            output = torch.cat([output, emitted.view(1, -1)], dim=1)
            acceptance_lengths.append(int(emitted.numel()))
            if stop_index is not None:
                break
            pending_anchor = bonus
        return self._result(
            output, acceptance_lengths, target_calls,
            prefill_latency_s=prefill_latency_s,
            draft_latency_s=draft_latency_s,
            verify_latency_s=verify_latency_s,
            started_at=started_at,
            prompt_length=prompt_length,
        )
