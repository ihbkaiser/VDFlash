# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Version-pinned SGLang boundary for offline EAGLE3 data preparation."""

from __future__ import annotations

from array import array
from typing import Any, List, Optional

import torch
import torch.distributed as dist
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler_components.dp_attn import prepare_mlp_sync_batch_raw
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import require_mlp_sync, require_mlp_tp_gather

from specforge.distributed import get_tp_group

from .capture_hooks import configure_capture_layers
from .model_runner import SGLangRunner
from .utils import wrap_offline_eagle3_logits_processors


class OfflineSGLangCaptureBackend:
    """Frozen local target used only to materialize offline features."""

    def __init__(self, model_runner: SGLangRunner) -> None:
        self.model_runner = model_runner

    @classmethod
    def build(
        cls,
        pretrained_model_name_or_path: str,
        *,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> "OfflineSGLangCaptureBackend":
        tp_size = dist.get_world_size(get_tp_group())
        server_args = ServerArgs(
            model_path=pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            dtype=torch_dtype if torch_dtype is not None else "auto",
            enable_return_hidden_states=True,
            disable_cuda_graph=True,
            chunked_prefill_size=-1,
            tp_size=tp_size,
            pp_size=1,
            **kwargs,
        )

        tp_rank = dist.get_rank(get_tp_group())
        moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)
        model_config = ModelConfig.from_server_args(server_args)
        model_runner = SGLangRunner(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=torch.cuda.current_device(),
            tp_rank=tp_rank,
            tp_size=server_args.tp_size,
            moe_ep_rank=moe_ep_rank,
            moe_ep_size=server_args.ep_size,
            pp_rank=0,
            pp_size=1,
            server_args=server_args,
            nccl_port=None,
            is_draft_worker=False,
        )
        model_runner.alloc_memory_pool()
        model_runner.init_attention_backends()
        model_runner.init_cuda_graphs()
        wrap_offline_eagle3_logits_processors(model_runner.model)
        return cls(model_runner)

    def set_eagle3_capture_layers(self, layer_ids: Optional[List[int]] = None) -> None:
        self.model_runner.model.set_eagle3_layers_to_capture(layer_ids)

    def set_capture_layers(
        self,
        layer_ids: Optional[List[int]] = None,
        *,
        capture_method: str,
    ) -> None:
        """Set auxiliary layers through the strategy's SGLang capture API."""

        configure_capture_layers(
            self.model_runner.model,
            layer_ids,
            capture_method=capture_method,
        )

    def _maybe_prepare_mlp_sync_batch(self, batch: ScheduleBatch) -> None:
        if require_mlp_sync(self.model_runner.server_args):
            prepare_mlp_sync_batch_raw(
                batch,
                dp_size=self.model_runner.server_args.dp_size,
                attn_tp_size=1,
                attn_cp_size=getattr(self.model_runner.server_args, "attn_cp_size", 1),
                tp_group=self.model_runner.tp_group,
                get_idle_batch=None,
                disable_cuda_graph=self.model_runner.server_args.disable_cuda_graph,
                require_mlp_tp_gather=require_mlp_tp_gather(
                    self.model_runner.server_args
                ),
                disable_overlap_schedule=self.model_runner.server_args.disable_overlap_schedule,
                offload_tags=set(),
            )

    @torch.no_grad()
    def _forward_extend(self, reqs: list[Req]):
        cache_params = CacheInitParams(
            disable=False,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            page_size=self.model_runner.server_args.page_size,
        )
        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            tree_cache=RadixCache(cache_params),
            model_config=self.model_runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.prepare_for_extend()
        self._maybe_prepare_mlp_sync_batch(batch)
        if getattr(batch, "prefill_input_ids_cpu", None) is not None:
            batch.input_ids = batch.prefill_input_ids_cpu.to(
                batch.device, non_blocking=True
            )
            batch.prefill_input_ids_cpu = None
        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        forward_batch = ForwardBatch.init_new(batch, self.model_runner)
        forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        output = self.model_runner.forward(forward_batch)
        return output.logits_output if hasattr(output, "logits_output") else output

    def _clear_pools(self) -> None:
        self.model_runner.req_to_token_pool.clear()
        self.model_runner.token_to_kv_pool_allocator.clear()

    def _qwen_image_token_id(self) -> int:
        """Resolve the placeholder ID from the loaded SGLang model config."""

        candidates = (
            getattr(self.model_runner, "model_config", None),
            getattr(self.model_runner, "model", None),
            getattr(getattr(self.model_runner, "model", None), "config", None),
            getattr(getattr(self.model_runner, "model_config", None), "hf_config", None),
        )
        for candidate in candidates:
            value = getattr(candidate, "image_token_id", None)
            if value is not None:
                return int(value)
        raise RuntimeError(
            "SGLang Qwen2.5-VL capture could not resolve image_token_id from "
            "the loaded model config"
        )

    def _build_multimodal_inputs(
        self,
        *,
        input_ids: torch.Tensor,
        media: dict[str, Any],
        position_ids: Optional[torch.Tensor],
    ):
        """Convert HF processor tensors to SGLang's internal MM contract."""

        try:
            from sglang.srt.managers.schedule_batch import (
                Modality,
                MultimodalDataItem,
                MultimodalInputs,
            )
        except ImportError as exc:  # pragma: no cover - version-specific runtime
            raise RuntimeError(
                "the installed SGLang build does not expose its multimodal "
                "capture data classes"
            ) from exc

        pixel_values = media.get("pixel_values")
        image_grid_thw = media.get("image_grid_thw")
        if not isinstance(pixel_values, torch.Tensor) or not isinstance(
            image_grid_thw, torch.Tensor
        ):
            raise ValueError(
                "Qwen2.5-VL capture requires pixel_values and image_grid_thw "
                "from the processor"
            )
        image_token_id = self._qwen_image_token_id()
        token_positions = torch.nonzero(
            input_ids.view(-1) == image_token_id, as_tuple=False
        ).view(-1)
        if token_positions.numel() == 0:
            raise ValueError(
                "Qwen2.5-VL input has image features but no image placeholder tokens"
            )
        offsets: list[tuple[int, int]] = []
        start = previous = int(token_positions[0])
        for value in token_positions[1:].tolist():
            value = int(value)
            if value != previous + 1:
                offsets.append((start, previous))
                start = value
            previous = value
        offsets.append((start, previous))
        if len(offsets) != int(image_grid_thw.shape[0]):
            raise ValueError(
                "image placeholder/grid mismatch: "
                f"offsets={len(offsets)}, grids={int(image_grid_thw.shape[0])}"
            )

        # SGLang's Qwen2-VL model reads ``item.image_grid_thw`` through the
        # MultimodalDataItem model-specific field and expects flattened patch
        # features, exactly as returned by the HF Qwen processor.
        items = []
        for image_index, offset in enumerate(offsets):
            feature = pixel_values
            if pixel_values.ndim >= 2 and int(image_grid_thw.shape[0]) > 1:
                grid_t, grid_h, grid_w = image_grid_thw[image_index].tolist()
                count = int(grid_t) * int(grid_h) * int(grid_w)
                cursor = sum(
                    int(image_grid_thw[j].prod().item()) for j in range(image_index)
                )
                feature = pixel_values[cursor : cursor + count]
            item = MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=feature,
                offsets=[offset],
                model_specific_data={
                    "image_grid_thw": image_grid_thw[image_index : image_index + 1]
                },
            )
            item.set_pad_value()
            items.append(item)

        padded_input_ids = input_ids.view(-1).tolist()
        for item in items:
            for start, end in item.offsets:
                padded_input_ids[start : end + 1] = [item.pad_value] * (
                    end - start + 1
                )
        mrope_positions = None
        if position_ids is not None:
            if position_ids.ndim == 3:
                mrope_positions = position_ids[:, 0, :]
            elif position_ids.ndim == 2 and position_ids.shape[0] == 3:
                mrope_positions = position_ids
            else:
                raise ValueError(
                    "Qwen2.5-VL SGLang capture requires [3, sequence] M-RoPE positions"
                )
        return MultimodalInputs(
            mm_items=items,
            padded_input_ids=padded_input_ids,
            im_token_id=image_token_id,
            mrope_positions=mrope_positions,
        )

    @torch.no_grad()
    def capture_eagle3(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        multimodal_inputs: Optional[list[dict]] = None,
    ):
        """Capture per-request auxiliary and final hidden states without logits."""

        sampling_params = SamplingParams(temperature=0, max_new_tokens=1, top_k=1)
        reqs: list[Req] = []
        data = []
        input_rows = torch.split(input_ids, 1, dim=0)
        attention_rows = torch.split(attention_mask, 1, dim=0)
        loss_rows = torch.split(loss_mask, 1, dim=0)
        position_rows = (
            [position_ids[:, index : index + 1] for index in range(input_ids.shape[0])]
            if position_ids is not None and position_ids.ndim == 3 and position_ids.shape[0] == 3
            else [position_ids[index : index + 1] for index in range(input_ids.shape[0])]
            if position_ids is not None
            else [None] * input_ids.shape[0]
        )
        media_rows = multimodal_inputs or [None] * input_ids.shape[0]
        if len(media_rows) != input_ids.shape[0]:
            raise ValueError(
                "multimodal_inputs must contain one mapping per input row, got "
                f"{len(media_rows)} for batch size {input_ids.shape[0]}"
            )

        for idx, (input_row, attention_row, loss_row, position_row, media_row) in enumerate(
            zip(input_rows, attention_rows, loss_rows, position_rows, media_rows)
        ):
            request_media = None
            padded_input_ids = input_row.view(-1).tolist()
            if media_row:
                request_media = self._build_multimodal_inputs(
                    input_ids=input_row,
                    media=media_row,
                    position_ids=position_row,
                )
                padded_input_ids = request_media.padded_input_ids
            req_kwargs = {
                "rid": str(idx),
                "origin_input_text": "",
                "origin_input_ids": padded_input_ids,
                "origin_input_ids_unpadded": input_row.view(-1).tolist(),
                "sampling_params": sampling_params,
            }
            req = Req(**req_kwargs)
            if request_media is not None:
                req.multimodal_inputs = request_media
            req.full_untruncated_fill_ids = array("q", req.origin_input_ids)
            req.fill_len = len(req.full_untruncated_fill_ids)
            req.extend_input_len = req.fill_len - len(req.prefix_indices)
            req.logprob_start_len = len(req.origin_input_ids) - 1
            reqs.append(req)
            data.append((input_row, attention_row, loss_row))

        input_lens = [len(req.origin_input_ids) for req in reqs]
        try:
            output = self._forward_extend(reqs)
            aux_hidden_states = getattr(output, "aux_hidden_states", None)
            last_hidden_states = getattr(output, "last_hidden_states", None)
            if aux_hidden_states is None or last_hidden_states is None:
                raise RuntimeError(
                    "SGLang did not return the hidden states required for "
                    "offline feature preparation"
                )
            aux_rows = torch.split(aux_hidden_states, input_lens, dim=0)
            last_rows = torch.split(last_hidden_states, input_lens, dim=0)
        finally:
            self._clear_pools()

        return data, aux_rows, last_rows

    def capture(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        multimodal_inputs: Optional[list[dict]] = None,
    ):
        """Capture generic auxiliary and final target states."""

        return self.capture_eagle3(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            position_ids=position_ids,
            multimodal_inputs=multimodal_inputs,
        )


__all__ = ["OfflineSGLangCaptureBackend"]
