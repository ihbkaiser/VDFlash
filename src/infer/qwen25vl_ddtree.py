"""DDTree decoding on top of the repository's Qwen2.5-VL DFlash drafter.

The tree builder follows the best-first factorized-distribution objective from
DDTree. Target verification uses an ancestor-only additive attention mask and
compacts the target cache to the accepted root-to-leaf path.
"""

from __future__ import annotations

import heapq
import time
import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from src.infer.qwen25vl_dflash_compare import (
    InstrumentedDFlashDecoder,
    SpeculativeDecodeResult,
    _as_stop_set,
    _cache_length,
    _crop_cache,
    _module_device,
    _move_tensor_kwargs,
    _new_cache,
    _now,
)


@dataclass
class DDTreeDecodeResult(SpeculativeDecodeResult):
    """DFlash-compatible result with DDTree stage and topology telemetry."""

    tree_build_s: float = 0.0
    tree_compile_s: float = 0.0
    commit_s: float = 0.0
    tree_budget: int = 0
    tree_traces: list[dict[str, Any]] = field(default_factory=list)


def build_ddtree_tree(
    draft_logits: torch.Tensor,
    budget: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[dict[int, int]], torch.Tensor]:
    """Build the best-first DDTree from per-position draft logits.

    Returns root-excluded node token IDs/depths, parent links, child lookup
    maps (including root at index 0), and the ancestor-only visibility mask.
    """

    if draft_logits.ndim != 2:
        raise ValueError("draft_logits must have shape [draft_horizon, vocab]")
    if budget < 0:
        raise ValueError("tree budget cannot be negative")

    horizon = int(draft_logits.shape[0])
    if budget == 0 or horizon == 0:
        visibility = torch.ones((1, 1), dtype=torch.bool)
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            [-1],
            [dict()],
            visibility,
        )

    topk = min(int(budget), int(draft_logits.shape[-1]))
    logits = draft_logits.float()
    top_logits, top_token_ids = torch.topk(logits, k=topk, dim=-1)
    top_log_probs = top_logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    # The heap is intentionally on CPU: only the top `budget` candidates per
    # depth are transferred, rather than the full vocabulary distributions.
    log_probs_cpu = top_log_probs.detach().to(device="cpu", dtype=torch.float32).numpy()
    token_ids_cpu = top_token_ids.detach().to(device="cpu", dtype=torch.long).numpy()

    first_logw = float(log_probs_cpu[0, 0])
    heap: list[tuple[float, tuple[int, ...], int, int, int, float]] = [
        (-first_logw, (0,), 0, 1, 0, first_logw)
    ]
    node_token_ids: list[int] = []
    node_depths: list[int] = []
    parents = [-1]
    child_maps: list[dict[int, int]] = [dict()]

    while heap and len(node_token_ids) < budget:
        _, ranks, parent_index, depth, rank, logw = heapq.heappop(heap)
        token_id = int(token_ids_cpu[depth - 1, rank])
        node_index = len(node_token_ids) + 1
        node_token_ids.append(token_id)
        node_depths.append(depth)
        parents.append(parent_index)
        child_maps.append(dict())
        child_maps[parent_index][token_id] = node_index

        if rank + 1 < topk:
            sibling_logw = (
                logw
                - float(log_probs_cpu[depth - 1, rank])
                + float(log_probs_cpu[depth - 1, rank + 1])
            )
            heapq.heappush(
                heap,
                (
                    -sibling_logw,
                    ranks[:-1] + (rank + 1,),
                    parent_index,
                    depth,
                    rank + 1,
                    sibling_logw,
                ),
            )
        if depth < horizon:
            child_logw = logw + float(log_probs_cpu[depth, 0])
            heapq.heappush(
                heap,
                (-child_logw, ranks + (0,), node_index, depth + 1, 0, child_logw),
            )

    node_count = len(node_token_ids)
    visibility = np.zeros((node_count + 1, node_count + 1), dtype=np.bool_)
    visibility[0, 0] = True
    for index in range(1, node_count + 1):
        parent = parents[index]
        visibility[index, :index] = visibility[parent, :index]
        visibility[index, index] = True

    return (
        torch.tensor(node_token_ids, dtype=torch.long),
        torch.tensor(node_depths, dtype=torch.long),
        parents,
        child_maps,
        torch.from_numpy(visibility),
    )


def follow_verified_tree(
    child_maps: list[dict[int, int]], posterior_tokens: torch.Tensor
) -> tuple[list[int], int]:
    """Follow target argmaxes from the root until the tree has no matching edge."""

    posterior = posterior_tokens[0].detach().cpu().tolist()
    accepted_indices = [0]
    current = 0
    next_token = int(posterior[current])
    while next_token in child_maps[current]:
        current = child_maps[current][next_token]
        accepted_indices.append(current)
        next_token = int(posterior[current])
    return accepted_indices, next_token


def compact_dynamic_cache(cache: Any, past_length: int, keep_indices: list[int]) -> None:
    """Compact the newly appended KV segment to the accepted tree path."""

    if not keep_indices:
        _crop_cache(cache, past_length)
        return

    def compact_tensor(tensor: torch.Tensor) -> None:
        current_length = int(tensor.shape[-2]) - past_length
        keep = torch.as_tensor(keep_indices, dtype=torch.long, device=tensor.device)
        if current_length < int(keep.numel()):
            raise RuntimeError(
                f"cache append length {current_length} < accepted tree path {keep.numel()}"
            )
        if current_length == int(keep.numel()):
            return
        selected = tensor.narrow(-2, past_length, current_length).index_select(-2, keep)
        tensor.narrow(-2, past_length, int(keep.numel())).copy_(selected)

    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        for key, value in zip(cache.key_cache, cache.value_cache):
            if key.numel() == 0 or value.numel() == 0:
                continue
            compact_tensor(key)
            compact_tensor(value)
    elif hasattr(cache, "layers"):
        for layer in cache.layers:
            key = getattr(layer, "keys", None)
            value = getattr(layer, "values", None)
            if key is None or value is None or key.numel() == 0:
                continue
            compact_tensor(key)
            compact_tensor(value)
    else:
        raise RuntimeError(f"unsupported target cache type: {type(cache).__name__}")

    _crop_cache(cache, past_length + len(keep_indices))


class InstrumentedDDTreeDecoder(InstrumentedDFlashDecoder):
    """Apply DDTree draft-tree construction and exact target verification."""

    @torch.inference_mode()
    def decode(
        self,
        *,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        target_kwargs: dict[str, Any] | None,
        max_new_tokens: int,
        stop_token_ids: list[int] | None,
        tree_budget: int = 16,
        prefill_target_hidden_transform=None,
        prefill_target_context_keep_mask: torch.Tensor | None = None,
        debug_compare_single_token: bool = False,
    ) -> DDTreeDecodeResult:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("DDTree inference currently supports batch size one")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if tree_budget < 0:
            raise ValueError("tree_budget cannot be negative")

        input_ids = input_ids.to(device=self.device, dtype=torch.long)
        prompt_length = int(input_ids.shape[1])
        cut_mask = None
        if prefill_target_context_keep_mask is not None:
            cut_mask = prefill_target_context_keep_mask.to(
                device=self.device, dtype=torch.bool
            )
            if cut_mask.ndim != 1 or cut_mask.numel() != prompt_length:
                raise ValueError(
                    "prefill_target_context_keep_mask must have shape "
                    f"[{prompt_length}]"
                )
            if not bool(cut_mask.any()):
                raise ValueError("retention mask must keep at least one text/context token")

        cut_context = cut_mask is not None
        block_size = int(self.draft.block_size)
        horizon = block_size - 1
        max_length = prompt_length + int(max_new_tokens)
        max_nodes = min(int(tree_budget), horizon * max(1, int(tree_budget)))
        all_positions = self._extend_positions(position_ids, max_length + block_size)
        target_cache = _new_cache()
        draft_cache = None if cut_context else _new_cache()
        stop_ids = _as_stop_set(stop_token_ids)
        target_kwargs = _move_tensor_kwargs(dict(target_kwargs or {}), self.device)
        output_ids = torch.full(
            (1, max_length + max_nodes + 1),
            int(self.draft.mask_token_id),
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
                "return_dict": True,
            }
        )
        prefill_started = _now(self.device)
        target_output, target_hidden = self._target_forward_with_context(prefill_kwargs)
        prefill_s = _now(self.device) - prefill_started
        target_calls = 1
        if getattr(target_output, "past_key_values", None) is not None:
            target_cache = target_output.past_key_values

        target_embed = self.target.get_input_embeddings()
        embed_device = _module_device(target_embed, self.device)
        target_lm_head = getattr(self.target, "lm_head", None)
        if target_lm_head is None:
            target_lm_head = self.target.get_output_embeddings()
        lm_head_device = _module_device(target_lm_head, self.device)

        if prefill_target_hidden_transform is not None:
            transformed = prefill_target_hidden_transform(target_hidden)
            if (
                not torch.is_tensor(transformed)
                or transformed.ndim != 3
                or transformed.shape[0] != target_hidden.shape[0]
                or transformed.shape[-1] != target_hidden.shape[-1]
                or transformed.shape[1] <= 0
                or transformed.shape[1] > target_hidden.shape[1]
            ):
                raise ValueError("invalid transformed target hidden-state tensor")
            target_hidden = transformed

        cut_prompt_positions = None
        if cut_context:
            kept_count = int(cut_mask.sum().item())
            if target_hidden.shape[1] == prompt_length:
                target_hidden = target_hidden[:, cut_mask.to(target_hidden.device), :]
            elif target_hidden.shape[1] != kept_count:
                raise ValueError(
                    "cut target hidden length must equal the number of retained "
                    "context positions"
                )
            cut_prompt_positions = all_positions[..., :prompt_length][..., cut_mask].to(
                device=self.draft_device
            )

        pending_anchor = target_output.logits[:, -1:, :].argmax(dim=-1).to(self.device)
        output_ids[:, prompt_length : prompt_length + 1] = pending_anchor
        decode_started = _now(self.device)
        if max_new_tokens == 0 or int(pending_anchor.item()) in stop_ids:
            result = DDTreeDecodeResult(
                output_ids=output_ids[:, :max_length],
                num_input_tokens=prompt_length,
                target_forward_calls=target_calls,
                prefill_s=prefill_s,
                decode_s=_now(self.device) - decode_started,
                end_to_end_s=_now(self.device) - started,
                tree_budget=int(tree_budget),
            )
            return self._finish_result(result, stop_ids)

        start = prompt_length
        rounds: list[dict[str, Any]] = []
        draft_s = 0.0
        verify_s = 0.0
        tree_build_s = 0.0
        tree_compile_s = 0.0
        commit_s = 0.0
        tree_traces: list[dict[str, Any]] = []

        while start < max_length:
            round_start = start
            block_output_ids = output_ids[:, start : start + block_size].clone()
            noise_embedding = target_embed(
                block_output_ids.to(device=embed_device)
            ).to(device=self.draft_device)

            draft_started = _now(self.device)
            if cut_context:
                draft_position_ids = torch.cat(
                    [
                        cut_prompt_positions,
                        all_positions[..., prompt_length : start + block_size].to(
                            device=self.draft_device
                        ),
                    ],
                    dim=-1,
                )
                draft_past_key_values = None
                draft_use_cache = False
            else:
                draft_position_ids = all_positions[
                    ..., _cache_length(draft_cache) : start + block_size
                ].to(device=self.draft_device)
                draft_past_key_values = draft_cache
                draft_use_cache = True

            draft_hidden = self.draft(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=draft_position_ids,
                past_key_values=draft_past_key_values,
                use_cache=draft_use_cache,
                is_causal=False,
            )
            if not cut_context:
                _crop_cache(draft_cache, start)
            draft_logits = target_lm_head(
                draft_hidden[:, -horizon:, :].to(device=lm_head_device)
            )[0]
            draft_s += _now(self.device) - draft_started

            tree_started = _now(self.device)
            node_tokens_cpu, node_depths_cpu, parents, child_maps, visibility = (
                build_ddtree_tree(draft_logits, int(tree_budget))
            )
            tree_build_s += _now(self.device) - tree_started
            node_tokens = node_tokens_cpu.to(device=self.device)
            node_depths = node_depths_cpu.to(device=self.device)
            current_length = 1 + int(node_tokens.numel())

            compile_started = _now(self.device)
            verify_input_ids = torch.cat(
                [block_output_ids[:, :1], node_tokens.unsqueeze(0)], dim=1
            )
            depth_positions = start + node_depths
            verify_position_ids = torch.cat(
                [
                    all_positions[..., start : start + 1],
                    all_positions.index_select(-1, depth_positions),
                ],
                dim=-1,
            )
            additive_mask = torch.full(
                (1, 1, current_length, start + current_length),
                torch.finfo(getattr(self.target, "dtype", torch.bfloat16)).min,
                dtype=getattr(self.target, "dtype", torch.bfloat16),
                device=self.device,
            )
            additive_mask[..., :start] = 0
            tree_visibility = visibility.to(device=self.device)
            tree_block = additive_mask[..., start : start + current_length]
            tree_block.masked_fill_(tree_visibility.view(1, 1, current_length, current_length), 0)
            tree_attention_mask = additive_mask
            tree_compile_s += _now(self.device) - compile_started

            verify_started = _now(self.device)
            # The visual tensors were consumed by prefill and are already
            # represented in the target KV cache. Re-sending them with a
            # text-only tree block makes Qwen-VL expect video marker tokens
            # inside this verification block.
            verify_kwargs: dict[str, Any] = {}
            verify_kwargs.update(
                {
                    "input_ids": verify_input_ids,
                    "position_ids": verify_position_ids,
                    "attention_mask": tree_attention_mask,
                    "past_key_values": target_cache,
                    "cache_position": torch.arange(
                        start, start + current_length, device=self.device
                    ),
                    "use_cache": True,
                    "return_dict": True,
                }
            )
            debug_cache = copy.deepcopy(target_cache) if debug_compare_single_token else None
            target_output, new_target_hidden = self._target_forward_with_context(
                verify_kwargs
            )
            verify_s += _now(self.device) - verify_started
            target_calls += 1
            if getattr(target_output, "past_key_values", None) is not None:
                target_cache = target_output.past_key_values

            commit_started = _now(self.device)
            posterior = target_output.logits.argmax(dim=-1).to(self.device)
            accepted_indices, next_token = follow_verified_tree(child_maps, posterior)
            single_token_debug = []
            if debug_compare_single_token:
                sequential_cache = debug_cache
                for path_offset, tree_index in enumerate(accepted_indices):
                    seq_kwargs = {
                        "input_ids": verify_input_ids[:, tree_index : tree_index + 1],
                        "position_ids": verify_position_ids[..., tree_index : tree_index + 1],
                        "past_key_values": sequential_cache,
                        "cache_position": torch.tensor(
                            [start + path_offset], device=self.device
                        ),
                        "use_cache": True,
                        "return_dict": True,
                    }
                    seq_output = self._target_forward(seq_kwargs)
                    if getattr(seq_output, "past_key_values", None) is not None:
                        sequential_cache = seq_output.past_key_values
                    tree_argmax = int(posterior[0, tree_index].item())
                    sequential_argmax = int(seq_output.logits[0, -1].argmax().item())
                    single_token_debug.append(
                        {
                            "tree_index": int(tree_index),
                            "position_id_depth": int(0 if tree_index == 0 else node_depths_cpu[tree_index - 1]),
                            "tree_argmax": tree_argmax,
                            "single_token_argmax": sequential_argmax,
                            "match": tree_argmax == sequential_argmax,
                        }
                    )
            accepted_index_tensor = torch.tensor(
                accepted_indices, dtype=torch.long, device=self.device
            )
            accepted_tokens = verify_input_ids.index_select(1, accepted_index_tensor)
            emitted = len(accepted_indices)
            output_ids[:, start : start + emitted] = accepted_tokens
            output_ids[:, start + emitted] = int(next_token)

            compact_dynamic_cache(target_cache, start, accepted_indices)
            hidden_indices = accepted_index_tensor.to(device=new_target_hidden.device)
            accepted_hidden = new_target_hidden.index_select(1, hidden_indices)
            if cut_context:
                target_hidden = torch.cat([target_hidden, accepted_hidden], dim=1)
            else:
                target_hidden = accepted_hidden

            start += emitted
            stop_hit = any(
                int(token_id) in stop_ids
                for token_id in output_ids[
                    0, prompt_length : min(start + 1, max_length)
                ].tolist()
            )
            accepted_depths = [int(node_depths_cpu[index - 1]) for index in accepted_indices[1:]]
            depth_widths: dict[int, int] = {}
            for depth in node_depths_cpu.tolist():
                depth_widths[int(depth)] = depth_widths.get(int(depth), 0) + 1
            is_terminal = bool(stop_hit or start >= max_length - 1)
            is_partial = (max_length - round_start) <= block_size
            round_row = {
                "proposal_count": int(node_tokens.numel()),
                "tree_node_count": int(node_tokens.numel()),
                "matched_proposals": int(emitted - 1),
                "effective_emitted_tokens": int(emitted),
                "accepted_depths": accepted_depths,
                "accepted_node_indices": accepted_indices,
                "node_depth_counts": depth_widths,
                "max_tree_depth": max(depth_widths, default=0),
                "accepted_token_ids": accepted_tokens[0].detach().cpu().tolist(),
                "bonus_token_id": int(next_token),
                "tree_budget": int(tree_budget),
                "single_token_debug": single_token_debug,
                "is_partial_block": bool(is_partial),
                "is_terminal": is_terminal,
            }
            rounds.append(round_row)
            tree_traces.append(
                {
                    "tree_token_ids": node_tokens_cpu.tolist(),
                    "tree_depths": node_depths_cpu.tolist(),
                    "parents": parents,
                    "accepted_node_indices": accepted_indices,
                    "accepted_depths": accepted_depths,
                    "node_depth_counts": depth_widths,
                }
            )
            commit_s += _now(self.device) - commit_started

            if stop_hit:
                break

        decode_s = _now(self.device) - decode_started
        end_to_end_s = _now(self.device) - started
        result = DDTreeDecodeResult(
            output_ids=output_ids[:, :max_length],
            num_input_tokens=prompt_length,
            acceptance_rounds=rounds,
            target_forward_calls=target_calls,
            prefill_s=prefill_s,
            draft_s=draft_s,
            verify_s=verify_s,
            decode_s=decode_s,
            end_to_end_s=end_to_end_s,
            tree_build_s=tree_build_s,
            tree_compile_s=tree_compile_s,
            commit_s=commit_s,
            tree_budget=int(tree_budget),
            tree_traces=tree_traces,
        )
        return self._finish_result(result, stop_ids)


__all__ = [
    "DDTreeDecodeResult",
    "InstrumentedDDTreeDecoder",
    "build_ddtree_tree",
    "compact_dynamic_cache",
    "follow_verified_tree",
]
