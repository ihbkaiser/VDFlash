"""Model-side utilities for native Qwen2-VL video inputs.

This module is intentionally isolated from the CPU-side audit code.  The
adapter validates the multimodal prefill before speculative decoding is
allowed.  In particular, it never silently routes a video through the
image-only helper shipped in the original MSD repository.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
import types
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from .dataset import qwen2vl_video_token_count


class RuntimeUnavailableError(RuntimeError):
    """Raised when a GPU/model-backed experiment cannot run safely."""


@dataclass
class PreparedPrefill:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor | None
    inputs_embeds: torch.Tensor
    position_ids: torch.Tensor
    rope_deltas: torch.Tensor
    video_grid_thw: torch.Tensor | None
    video_positions: torch.Tensor
    input_fingerprint: str
    video_token_id: int | None = None
    vision_start_token_id: int | None = None
    vision_end_token_id: int | None = None
    keep_mask: torch.Tensor | None = None


def _tensor(batch: Any, key: str) -> torch.Tensor | None:
    if isinstance(batch, dict):
        value = batch.get(key)
    else:
        value = getattr(batch, key, None)
    return value


def _hash_tensor(value: torch.Tensor) -> str:
    cpu = value.detach().to("cpu").contiguous()
    payload = cpu.numpy().tobytes() if cpu.numel() else b""
    return hashlib.sha256(payload + str(cpu.shape).encode() + str(cpu.dtype).encode()).hexdigest()[:16]


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeUnavailableError("CUDA is required for MSD inference; no CUDA device is available")
    return torch.device("cuda")


def model_device(model: Any) -> torch.device:
    """Return the device hosting the first model parameter.

    ``device_map=auto`` can shard a checkpoint, but every runner still needs a
    deterministic device for processor tensors and the first embedding call.
    The first parameter is the same convention used by Transformers' generate
    helpers and is sufficient for the single-GPU experiments in the paper.
    """

    try:
        return next(model.parameters()).device
    except StopIteration as exc:  # pragma: no cover - malformed external model
        raise RuntimeUnavailableError("model has no parameters") from exc


def move_batch_to_device(batch: Any, device: torch.device | str) -> Any:
    """Move a processor batch without assuming it is a plain dictionary."""

    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }
    raise TypeError(f"unsupported processor batch type: {type(batch)!r}")


def _apply_transformers_compat() -> None:
    """Compatibility shims so the vendored EAGLE (Transformers 4.49-era) code
    runs on the installed transformers version without modifying externals/.

    - transformers >= 5 removed ``SlidingWindowCache`` from ``cache_utils``;
      EAGLE only uses it in ``isinstance`` checks that never match Qwen2-VL, so
      a ``DynamicCache`` subclass stub is sufficient.
    - If the installed ``flash_attn`` wheel cannot be imported on this machine
      (e.g. broken binary), force ``is_flash_attn_2_available`` to return
      ``False`` so the vendored Qwen2-VL copy falls back to eager attention.
    """

    import transformers.cache_utils as cache_utils

    if not hasattr(cache_utils, "SlidingWindowCache"):

        class SlidingWindowCache(cache_utils.DynamicCache):
            pass

        cache_utils.SlidingWindowCache = SlidingWindowCache

    try:
        import flash_attn  # noqa: F401
    except Exception:
        import transformers.utils as hf_utils

        hf_utils.is_flash_attn_2_available = lambda: False


_apply_transformers_compat()


def select_visual_positions(
    visual_count: int,
    percentage: float,
    scores: Sequence[float] | None = None,
) -> torch.Tensor:
    """Return deterministic local visual indices for a draft context.

    If attention scores are supplied, ties are resolved by the original
    position.  Without scores, positions are spread uniformly; this is useful
    only as an explicit baseline and is never labelled ``Last Instr.`` or
    ``All Text``.
    """

    if visual_count < 0 or not 0 <= percentage <= 100:
        raise ValueError("visual_count must be non-negative and percentage must be in [0, 100]")
    if visual_count == 0 or percentage == 0:
        return torch.empty(0, dtype=torch.long)
    if percentage == 100:
        return torch.arange(visual_count, dtype=torch.long)
    keep = max(1, round(visual_count * percentage / 100))
    if scores is not None:
        if len(scores) != visual_count:
            raise ValueError("scores length must equal visual_count")
        order = sorted(range(visual_count), key=lambda index: (-float(scores[index]), index))
        return torch.tensor(sorted(order[:keep]), dtype=torch.long)
    positions = sorted({min(visual_count - 1, (index * visual_count) // keep) for index in range(keep)})
    return torch.tensor(positions, dtype=torch.long)


def make_visual_mask(total_tokens: int, visual_positions: Sequence[int]) -> torch.Tensor:
    mask = torch.zeros(total_tokens, dtype=torch.bool)
    positions = [int(value) for value in visual_positions]
    if positions:
        mask[torch.as_tensor(positions, dtype=torch.long)] = True
    return mask


def prepare_qwen2vl_prefill(model: Any, batch: Any, device: torch.device | str | None = None) -> PreparedPrefill:
    """Build video embeddings and native Qwen2-VL M-RoPE positions.

    ``model`` must expose the standard Qwen2-VL ``model``, ``visual`` and
    ``get_rope_index`` members.  The function deliberately accepts a generic
    object so it can be used with the custom KV model in MSD and with the
    standard Hugging Face model in the parity check.
    """

    input_ids = _tensor(batch, "input_ids")
    if input_ids is None:
        raise ValueError("processor batch has no input_ids")
    if device is not None:
        input_ids = input_ids.to(device)
    attention_mask = _tensor(batch, "attention_mask")
    if attention_mask is not None and device is not None:
        attention_mask = attention_mask.to(device)
    video_grid_thw = _tensor(batch, "video_grid_thw")
    if video_grid_thw is not None and device is not None:
        video_grid_thw = video_grid_thw.to(device)
    image_grid_thw = _tensor(batch, "image_grid_thw")
    if image_grid_thw is not None and device is not None:
        image_grid_thw = image_grid_thw.to(device)
    pixel_values_videos = _tensor(batch, "pixel_values_videos")
    pixel_values = _tensor(batch, "pixel_values")

    with torch.inference_mode():
        inputs_embeds = model.model.embed_tokens(input_ids)
        if pixel_values is not None:
            image_values = pixel_values.to(dtype=model.visual.get_dtype())
            image_embeds = model.visual(image_values, grid_thw=image_grid_thw)
            image_mask = (input_ids == model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            if int(image_mask[..., 0].sum()) != image_embeds.shape[0]:
                raise ValueError("image token/features mismatch during native prefill")
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds.to(inputs_embeds.dtype))
        if pixel_values_videos is None:
            raise ValueError("video experiment requires processor key pixel_values_videos")
        if int(os.environ.get("VIDEO_DFLASH_DEBUG", "0")):
            print(f"[debug] grid={video_grid_thw.tolist() if video_grid_thw is not None else None} "
                  f"video_ids={(input_ids[0] == int(getattr(model.config, 'video_token_id'))).sum().item()}", flush=True)
        video_values = pixel_values_videos.to(dtype=model.visual.get_dtype())
        video_embeds = model.visual(video_values, grid_thw=video_grid_thw)
        video_token_id = int(getattr(model.config, "video_token_id"))
        video_mask = (input_ids == video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        if int(video_mask[..., 0].sum()) != video_embeds.shape[0]:
            raise ValueError("video token/features mismatch during native prefill")
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds.to(inputs_embeds.dtype))
        position_ids, rope_deltas = model.get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask)

    video_positions = (input_ids[0] == video_token_id).nonzero(as_tuple=False).flatten()
    return PreparedPrefill(
        input_ids=input_ids,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        rope_deltas=rope_deltas,
        video_grid_thw=video_grid_thw,
        video_positions=video_positions,
        input_fingerprint=_hash_tensor(input_ids),
        video_token_id=video_token_id,
        vision_start_token_id=getattr(model.config, "vision_start_token_id", None),
        vision_end_token_id=getattr(model.config, "vision_end_token_id", None),
    )


def compact_qwen2vl_prefill(
    prepared: PreparedPrefill,
    percentage: float,
    scores: Sequence[float] | None = None,
) -> PreparedPrefill:
    """Build the draft-side context for the Figure 1(b) retention sweep.

    The target context is never modified.  Only the draft-side embedding
    sequence is compacted, while the original M-RoPE positions are retained
    for the tokens that survive.  At 0% the vision delimiters are removed as
    well, yielding a genuine text-only draft context rather than an
    image/video placeholder shortcut.
    """

    if not 0 <= percentage <= 100:
        raise ValueError("percentage must be in [0, 100]")
    visual = prepared.video_positions.detach().to("cpu")
    selected_local = select_visual_positions(int(visual.numel()), percentage, scores)
    selected_global = visual[selected_local] if selected_local.numel() else visual[:0]
    keep = torch.ones(prepared.input_ids.shape[1], dtype=torch.bool, device=prepared.input_ids.device)
    visual_mask = torch.zeros_like(keep)
    if visual.numel():
        visual_mask[visual.to(prepared.input_ids.device)] = True
    if visual.numel() and selected_global.numel() < visual.numel():
        selected_mask = torch.zeros_like(keep)
        selected_mask[selected_global.to(prepared.input_ids.device)] = True
        keep &= ~visual_mask | selected_mask
    if percentage == 0:
        marker_ids = {
            value
            for value in (
                prepared.vision_start_token_id,
                prepared.vision_end_token_id,
                prepared.video_token_id,
            )
            if value is not None
        }
        if marker_ids:
            marker_mask = torch.zeros_like(keep)
            for marker_id in marker_ids:
                marker_mask |= prepared.input_ids[0] == int(marker_id)
            keep &= ~marker_mask

    input_ids = prepared.input_ids[:, keep]
    inputs_embeds = prepared.inputs_embeds[:, keep, :]
    position_ids = prepared.position_ids[..., keep]
    attention_mask = prepared.attention_mask
    if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = attention_mask[:, keep]
    compact_visual_positions = torch.nonzero(visual_mask & keep, as_tuple=False).flatten()
    compact_visual_positions = torch.searchsorted(
        torch.nonzero(keep, as_tuple=False).flatten(), compact_visual_positions
    )
    return PreparedPrefill(
        input_ids=input_ids,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        rope_deltas=prepared.rope_deltas,
        video_grid_thw=prepared.video_grid_thw,
        video_positions=compact_visual_positions.to(input_ids.device),
        input_fingerprint=_hash_tensor(input_ids),
        video_token_id=prepared.video_token_id,
        vision_start_token_id=prepared.vision_start_token_id,
        vision_end_token_id=prepared.vision_end_token_id,
        keep_mask=keep.detach().to(device=prepared.input_ids.device),
    )


@contextmanager
def last_position_lm_head(model: Any, threshold: int = 4096):
    """Limit long-context LM-head calls to the final position.

    Accelerate wraps dispatched modules and stores the original callable in
    ``_old_forward``.  Replacing only ``lm_head.forward`` therefore has no
    effect on a sharded/quantized model: the hook still invokes the old full
    sequence projection and allocates ``[batch, sequence, vocab]`` logits.
    Patch the saved callable when present, while retaining the ordinary-module
    path for un-dispatched models.
    """

    lm_head = model.lm_head
    original_forward = lm_head.forward
    missing = object()
    original_hook_forward = getattr(lm_head, "_old_forward", missing)
    underlying_forward = (
        original_hook_forward if original_hook_forward is not missing else original_forward
    )

    def _last_position_only(hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if hidden_states.ndim == 3 and hidden_states.shape[1] > threshold:
            hidden_states = hidden_states[:, -1:]
        return underlying_forward(hidden_states, *args, **kwargs)

    if original_hook_forward is not missing:
        # Functions stored directly on an instance are not descriptor-bound;
        # this is the same calling convention used by Accelerate's hook.
        lm_head._old_forward = _last_position_only
    else:
        lm_head.forward = _last_position_only
    try:
        yield
    finally:
        if original_hook_forward is not missing:
            lm_head._old_forward = original_hook_forward
        else:
            lm_head.forward = original_forward


def validate_native_prefill_parity(model: Any, batch: Any, device: torch.device | str = "cuda", atol: float = 5e-3) -> dict[str, Any]:
    """Compare standard multimodal forward with manually fused prefill."""

    prepared = prepare_qwen2vl_prefill(model, batch, device)
    pixel_values = _tensor(batch, "pixel_values")
    pixel_values_videos = _tensor(batch, "pixel_values_videos")
    image_grid_thw = _tensor(batch, "image_grid_thw")
    if pixel_values is not None:
        pixel_values = pixel_values.to(device)
    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.to(device)
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.to(device)
    # For long sequences the full vocabulary logits [1, S, 152064] do not fit
    # the T4; compare only the last position, which sees the entire context and
    # is the strongest single alignment probe.  Short sequences keep the
    # full-length comparison.
    sequence_length = int(prepared.input_ids.shape[1])
    last_only = sequence_length > 4096
    lm_head_context = last_position_lm_head(model) if last_only else nullcontext()
    with lm_head_context:
        with torch.inference_mode():
            native = model(
                input_ids=prepared.input_ids,
                attention_mask=prepared.attention_mask,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=prepared.video_grid_thw,
                use_cache=False,
                return_dict=True,
            )
            native_logits = native.logits
            inner = model.model(
                inputs_embeds=prepared.inputs_embeds,
                attention_mask=prepared.attention_mask,
                position_ids=prepared.position_ids,
                use_cache=False,
                return_dict=True,
            )
            hidden = inner.last_hidden_state if hasattr(inner, "last_hidden_state") else inner[0]
            fused_logits = model.lm_head(hidden)
    max_error = float((native_logits - fused_logits).abs().max().item())
    return {
        "valid": max_error <= atol,
        "max_abs_logit_error": max_error,
        "atol": atol,
        "input_fingerprint": prepared.input_fingerprint,
        "video_token_count": int(prepared.video_positions.numel()),
        "parity_positions": "last_only" if last_only else "full",
    }


def build_qwen2vl_video_processor(model_id: str, min_pixels: int, max_pixels: int) -> Any:
    try:
        from transformers import AutoProcessor
    except ImportError as exc:  # pragma: no cover
        raise RuntimeUnavailableError("Transformers is required for video processing") from exc
    return AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)


def build_video_messages(
    video_path: str | Path,
    question: str,
    fps: float,
    max_pixels: int | None = None,
    max_frames: int | None = None,
) -> list[dict[str, Any]]:
    video = {"type": "video", "video": str(video_path), "fps": float(fps)}
    if max_pixels is not None:
        video["max_pixels"] = int(max_pixels)
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError(f"max_frames must be positive, got {max_frames}")
        video["max_frames"] = int(max_frames)
    return [{
        "role": "user",
        "content": [
            video,
            {"type": "text", "text": question},
        ],
    }]


def process_video(
    processor: Any,
    video_path: str | Path,
    question: str,
    fps: float,
    max_pixels: int | None = None,
    attempts: int = 2,
    max_frames: int | None = None,
) -> Any:
    """Process a video with a small number of retries.

    The torchvision reader occasionally reports zero total frames on a healthy
    file under I/O contention (concurrent downloads/calibration); a single
    retry resolves the transient reads seen on the T4 host.
    """

    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return _process_video_once(processor, video_path, question, fps, max_pixels, max_frames)
        except Exception as exc:  # noqa: BLE001 - transient video read failures
            last_exc = exc
            if attempt < attempts:
                time.sleep(3)
    assert last_exc is not None
    raise last_exc


def _process_video_once(
    processor: Any,
    video_path: str | Path,
    question: str,
    fps: float,
    max_pixels: int | None = None,
    max_frames: int | None = None,
) -> Any:
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:  # pragma: no cover
        raise RuntimeUnavailableError("qwen-vl-utils is required for Qwen video preprocessing") from exc
    messages = build_video_messages(
        video_path,
        question,
        fps,
        max_pixels=max_pixels,
        max_frames=max_frames,
    )
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    try:
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    except TypeError:
        image_inputs, video_inputs = process_vision_info(messages)
        video_kwargs = {}
    kwargs = dict(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    kwargs.update(video_kwargs)
    return processor(**kwargs)


def apply_msd_memory_patches() -> None:
    """Make the vendored MSD attention memory-efficient on small GPUs (T4).

    The vendored EAGLE code materializes full attention-score matrices:

    - ``VisionSdpaAttention`` passes a 3D query and a ``[1, S, S]`` bool mask,
      which forces SDPA's math backend and allocates ``heads * S * S`` fp32
      scores.  Reshaping to 4D ``[1, heads, S, S]`` selects the memory-
      efficient backend (measured: 684 MB vs 9.3 GB at S=12512).
    - ``Qwen2VLSdpaAttention`` passes a 4D float32 mask, which also forces the
      math backend; converting it to bool keeps the memory-efficient backend.
    - The EAGLE draft ``Qwen2VLAttention`` materializes ``S * S`` fp32 weights
      with a manual matmul; the SDPA path is used unless ``output_attentions``
      is requested (the draft-attention probe then falls back to the original
      implementation, which is small at probe sizes).

    The patches preserve the original masking semantics (masked positions are
    exactly ``finfo.min`` or ``-inf``, so ``mask > finfo.min / 2`` is a safe
    "attend" predicate).
    """

    import torch.nn.functional as F

    def _bool_attend(mask: torch.Tensor) -> torch.Tensor:
        if mask.dtype == torch.float32:
            return mask > (torch.finfo(torch.float32).min / 2)
        return mask

    try:
        from eagle.model import ea_qwen2vl_model as ea_mod
        from eagle.model import modeling_qwen2vl_kv as kv_mod
    except ImportError:  # pragma: no cover - only applied after MSD import
        return

    # 1. Vision encoder attention (modeling_qwen2vl_kv.VisionSdpaAttention).
    vision_original = kv_mod.VisionSdpaAttention.forward

    def _vision_sdpa_forward(
        self: Any,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        if position_embeddings is None:
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos, sin = emb.cos().float(), emb.sin().float()
        else:
            cos, sin = position_embeddings
        q, k = kv_mod.apply_rotary_pos_emb_vision(q, k, cos, sin)
        # The attention is block-diagonal (each frame attends only within
        # itself, fully bidirectional).  Process each frame separately so the
        # flash backend runs without materializing a [1, S, S] mask, which
        # would OOM on a T4 once the patch grid reaches tens of thousands of
        # tokens (measured: 474 MB peak vs 4.7 GB at S=50048).
        starts = cu_seqlens[:-1].tolist()
        ends = cu_seqlens[1:].tolist()
        outputs = []
        for start, end in zip(starts, ends):
            # SDPA expects [batch, heads, sequence, head_dim].  The vision
            # projection is initially [sequence, heads, head_dim].
            frame = F.scaled_dot_product_attention(
                q[start:end].transpose(0, 1).unsqueeze(0),
                k[start:end].transpose(0, 1).unsqueeze(0),
                v[start:end].transpose(0, 1).unsqueeze(0),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            ).squeeze(0).transpose(0, 1)
            outputs.append(frame)
        attn_output = torch.cat(outputs, dim=0)  # [S, heads, head_dim]
        attn_output = attn_output.reshape(seq_length, -1)
        return self.proj(attn_output)

    kv_mod.VisionSdpaAttention.forward = _vision_sdpa_forward

    # 2. Language-model SDPA attention (modeling_qwen2vl_kv.Qwen2VLSdpaAttention).
    language_original = kv_mod.Qwen2VLSdpaAttention.forward

    def _cache_empty(past_key_value: Any) -> bool:
        if past_key_value is None:
            return True
        if hasattr(past_key_value, "get_seq_length"):
            return past_key_value.get_seq_length() == 0
        try:
            return past_key_value[0][0].current_length.item() == 0
        except (AttributeError, IndexError, TypeError):
            return False

    def _language_sdpa_forward(
        self: Any,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: Any = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Any]:
        if output_attentions:
            return language_original(
                self, hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                past_key_value=past_key_value, output_attentions=True, use_cache=use_cache,
                cache_position=cache_position, position_embeddings=position_embeddings,
            )
        # During the initial full-context prefill the mask is a plain causal
        # mask.  Passing that [1, 1, S, S] tensor to torch 2.1 SDPA can select
        # the math backend and materialize a huge attention matrix.  Let SDPA
        # use its causal memory-efficient kernel instead.  Non-causal MSD
        # tree masks and non-empty KV-cache decoding retain their mask.
        q_len = hidden_states.shape[1]
        if q_len > 1 and _cache_empty(past_key_value):
            attention_mask = None
        if attention_mask is not None and attention_mask.ndim == 4:
            attention_mask = _bool_attend(attention_mask)
        return language_original(
            self, hidden_states, attention_mask=attention_mask, position_ids=position_ids,
            past_key_value=past_key_value, output_attentions=False, use_cache=use_cache,
            cache_position=cache_position, position_embeddings=position_embeddings,
        )

    kv_mod.Qwen2VLSdpaAttention.forward = _language_sdpa_forward

    # 3. EAGLE draft attention (ea_qwen2vl_model.Qwen2VLAttention).
    draft_original = ea_mod.Qwen2VLAttention.forward
    draft_apply_rope = ea_mod.apply_multimodal_rotary_pos_emb
    draft_repeat_kv = ea_mod.repeat_kv

    def _draft_sdpa_forward(
        self: Any,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: Any = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Any]:
        if output_attentions:
            # The draft-attention probe needs the explicit weights; at probe
            # sizes (<= 3k tokens) the original materialization fits VRAM.
            return draft_original(
                self, hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                past_key_value=past_key_value, output_attentions=True, use_cache=use_cache,
                cache_position=cache_position, position_embeddings=position_embeddings,
            )
        bsz, q_len, _ = hidden_states.size()
        cache_empty = _cache_empty(past_key_value)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = draft_apply_rope(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states) if use_cache else None
        key_states = draft_repeat_kv(key_states, self.num_key_value_groups)
        value_states = draft_repeat_kv(value_states, self.num_key_value_groups)
        mask = attention_mask
        if mask is not None:
            mask = mask[:, :, :, : key_states.shape[-2]]
            mask = _bool_attend(mask)
        if q_len > 1 and cache_empty:
            mask = None
        attn_weights = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=mask is None and q_len > 1,
        )
        attn_output = attn_weights.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value

    ea_mod.Qwen2VLAttention.forward = _draft_sdpa_forward


def refresh_msd_memory_patch_hooks(model: Any) -> None:
    """Refresh Accelerate's saved forwards after memory monkey-patching.

    ``load_in_4bit`` installs Accelerate hooks before the runtime patches the
    vendored attention classes.  A hook keeps the old bound method in
    ``_old_forward``, so changing the class method alone does not affect the
    actual call path.  Replace those saved methods for the loaded instances.
    """

    try:
        from eagle.model import ea_qwen2vl_model as ea_mod
        from eagle.model import modeling_qwen2vl_kv as kv_mod
    except ImportError:  # pragma: no cover - only used with the MSD runtime
        return

    for module in model.modules():
        if not hasattr(module, "_old_forward"):
            continue
        if isinstance(module, kv_mod.VisionSdpaAttention):
            module._old_forward = types.MethodType(kv_mod.VisionSdpaAttention.forward, module)
        elif isinstance(module, kv_mod.Qwen2VLSdpaAttention):
            module._old_forward = types.MethodType(kv_mod.Qwen2VLSdpaAttention.forward, module)
        elif isinstance(module, ea_mod.Qwen2VLAttention):
            module._old_forward = types.MethodType(ea_mod.Qwen2VLAttention.forward, module)


def clear_msd_runtime_state(model: Any) -> None:
    """Release per-job MSD caches and reset mutable tree state.

    The official EAGLE wrapper intentionally retains its KV buffers for reuse.
    That is useful for one long decode, but harmful for this benchmark where
    adjacent jobs can have very different visual-token counts and failed OOM
    attempts must not poison the next retry.
    """

    for cache_name in ("past_key_values", "past_key_values_data", "current_length_data"):
        if hasattr(model, cache_name):
            delattr(model, cache_name)
    try:
        model.ea_layer.reset_kv()
    except (AttributeError, RuntimeError):
        pass
    try:
        qwen = model.base_model.model
        if hasattr(qwen, "tree_mask"):
            qwen.tree_mask = None
        if hasattr(qwen, "tree_mode"):
            qwen.tree_mode = None
        model.base_model.rope_deltas = None
    except AttributeError:
        pass


def apply_chunked_video_vision(model: Any, chunk_frames: int = 8) -> None:
    """Run Qwen2-VL vision frames in bounded chunks.

    Vision attention is independent across video frames, but the stock
    implementation receives all patch tokens in one call.  Chunking keeps
    the temporary vision activations on the vision device bounded while the
    decoder still receives the exact concatenated embedding sequence.
    """

    if chunk_frames <= 0 or not hasattr(model, "visual"):
        return
    visual = model.visual
    if getattr(visual, "_msd_chunked_video", False):
        return
    original_forward = visual.forward

    def _chunked_forward(hidden_states: torch.Tensor, grid_thw: torch.Tensor, *args: Any, **kwargs: Any):
        if grid_thw is None or grid_thw.ndim != 2 or grid_thw.shape[0] != 1:
            return original_forward(hidden_states, grid_thw=grid_thw, *args, **kwargs)
        temporal, height, width = (int(value) for value in grid_thw[0].tolist())
        if temporal <= chunk_frames:
            return original_forward(hidden_states, grid_thw=grid_thw, *args, **kwargs)
        patches_per_frame = height * width
        outputs = []
        for start_frame in range(0, temporal, chunk_frames):
            frames = min(chunk_frames, temporal - start_frame)
            start = start_frame * patches_per_frame
            end = (start_frame + frames) * patches_per_frame
            chunk_grid = grid_thw.new_tensor([[frames, height, width]])
            outputs.append(
                original_forward(hidden_states[start:end], grid_thw=chunk_grid, *args, **kwargs)
            )
        return torch.cat(outputs, dim=0)

    visual.forward = _chunked_forward
    visual._msd_chunked_video = True


def _parse_max_memory(value: str | dict[int, str] | None) -> dict[int, str] | None:
    if value is None or isinstance(value, dict):
        return value
    result: dict[int, str] = {}
    for item in value.split(","):
        index, budget = item.split(":", 1)
        result[int(index)] = budget
    return result


def _model_parallel_device_map(base_model_path: str) -> dict[str, int]:
    """Place Qwen2-VL's vision/final path and EAGLE draft on GPU 0."""

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    layer_count = int(config.num_hidden_layers)
    prefix = min(
        int(os.environ.get("MSD_MODEL_PARALLEL_PREFIX_LAYERS", "3")),
        max(1, layer_count - 1),
    )
    suffix = min(
        int(os.environ.get("MSD_MODEL_PARALLEL_SUFFIX_LAYERS", "2")),
        max(1, layer_count - prefix),
    )
    mapping: dict[str, int] = {
        "visual": 0,
        "model.embed_tokens": 0,
        "model.norm": 0,
        "model.rotary_emb": 0,
        "lm_head": 0,
    }
    for index in range(layer_count):
        mapping[f"model.layers.{index}"] = (
            0 if index < prefix or index >= layer_count - suffix else 1
        )
    return mapping


def load_msd_qwen2vl(
    base_model_path: str,
    msd_model_path: str,
    device_map: str = "cuda",
    max_memory: str | dict[int, str] | None = None,
) -> Any:
    """Load the official MSD wrapper after an explicit CUDA check."""

    require_cuda()
    eagle_root = Path(__file__).parents[3] / "externals" / "MSD" / "EAGLE"
    if str(eagle_root) not in sys.path:
        sys.path.insert(0, str(eagle_root))
    try:
        from eagle.model.ea_model import EaModel
        from transformers import BitsAndBytesConfig
    except ImportError as exc:  # pragma: no cover
        raise RuntimeUnavailableError("MSD EAGLE and Transformers quantization dependencies are required") from exc
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    resolved_device_map: str | dict[str, int] = device_map
    if device_map == "model_parallel":
        resolved_device_map = _model_parallel_device_map(base_model_path)
    model_kwargs: dict[str, Any] = {
        "base_model_path": base_model_path,
        "ea_model_path": msd_model_path,
        "total_token": 30,
        "depth": 4,
        "top_k": 8,
        "torch_dtype": torch.float16,
        "device_map": resolved_device_map,
        "quantization_config": quantization_config,
    }
    parsed_max_memory = _parse_max_memory(max_memory)
    if parsed_max_memory is not None:
        model_kwargs["max_memory"] = parsed_max_memory
    result = EaModel.from_pretrained(**model_kwargs)
    # EAGLE's EaModel.from_pretrained returns (model, None) for the Qwen2-VL
    # path; unwrap it so the caller receives the EaModel itself.
    if isinstance(result, tuple):
        result = result[0]
    # The draft cnets.Model enables gradient checkpointing and nn.Module starts
    # in training mode; leaving it trainable routes inference through the
    # torch.utils.checkpoint branch whose custom_forward drops use_cache and
    # returns a single-element tuple. eval() disables that branch.
    result.eval()
    apply_msd_memory_patches()
    refresh_msd_memory_patch_hooks(result.base_model)
    vision_chunk_frames = int(os.environ.get("MSD_VISION_CHUNK_FRAMES", "8"))
    apply_chunked_video_vision(result.base_model, vision_chunk_frames)
    return result


def _video_forward(
    self: Any,
    input_ids: torch.Tensor | None = None,
    inputs_embeds: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    past_key_values: Any = None,
    output_orig: bool = False,
    position_ids: torch.Tensor | None = None,
):
    """EaModel.forward with explicit Qwen2-VL positions for prefill/tree."""

    if position_ids is None and inputs_embeds is not None:
        # Works for both the transformers Cache (get_seq_length) and the EAGLE
        # custom KVCache list (past_key_values[0][0].current_length).
        if past_key_values is None:
            cache_empty = True
        elif hasattr(past_key_values, "get_seq_length"):
            cache_empty = past_key_values.get_seq_length() == 0
        else:
            try:
                cache_empty = past_key_values[0][0].current_length.item() == 0
            except Exception:
                cache_empty = True
        if cache_empty and getattr(self, "_video_prefill_position_ids", None) is not None:
            position_ids = self._video_prefill_position_ids
    with torch.inference_mode():
        if inputs_embeds is not None:
            outputs = self.base_model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
        else:
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
        orig = self.base_model.lm_head(outputs[0]) if output_orig else None
    return (outputs, orig, outputs[0]) if output_orig else (outputs, outputs[0])


def _video_initialize_tree(input_ids, model, past_key_values, logits_processor, inputs_embeds=None):
    outputs, orig, hidden_states = model(
        input_ids,
        past_key_values=past_key_values,
        output_orig=True,
        inputs_embeds=inputs_embeds,
        position_ids=model._video_prefill_position_ids,
    )
    if logits_processor is not None:
        logits = logits_processor(None, orig[:, -1])
        token = torch.multinomial(torch.softmax(logits, dim=-1), 1)
    else:
        token = torch.argmax(orig[:, -1], dim=-1, keepdim=True)
    extended_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = model.ea_layer.topK_genrate(
        hidden_states,
        extended_ids,
        model.base_model.lm_head,
        logits_processor,
        inputs_embeds,
    )
    return draft_tokens, retrieve_indices, tree_mask, tree_position_ids, orig, hidden_states, token


def _video_tree_decoding(model, tree_candidates, past_key_values, tree_position_ids, input_ids, retrieve_indices):
    offset = int(input_ids.shape[1])
    rope_delta = getattr(model, "_video_rope_delta", 0)
    if isinstance(rope_delta, torch.Tensor):
        rope_delta = int(rope_delta.flatten()[0].item())
    positions = tree_position_ids.to(input_ids.device) + offset + rope_delta
    positions = positions.view(1, -1).expand(3, -1, -1)
    outputs, tree_logits, hidden_state = model(
        tree_candidates,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=positions,
    )
    return tree_logits[0, retrieve_indices], hidden_state, outputs


@contextmanager
def patched_msd_video_path(model: Any, prepared: PreparedPrefill):
    """Temporarily make the original MSD loop aware of Qwen2-VL positions."""

    try:
        from eagle.model import ea_model as ea_module
    except ImportError as exc:  # pragma: no cover
        raise RuntimeUnavailableError("MSD EAGLE is not importable") from exc
    old_initialize = ea_module.initialize_tree
    old_tree = ea_module.tree_decoding
    old_evaluate = ea_module.evaluate_posterior
    old_forward = model.forward
    lm_head_context = last_position_lm_head(model.base_model)
    lm_head_context.__enter__()
    capture: dict[str, Any] = {
        "trace": [],
        "prefill_seconds": None,
        "verification_start": None,
        "verification_end": None,
    }

    def evaluate_with_trace(*args, **kwargs):
        result = old_evaluate(*args, **kwargs)
        capture["trace"].append(int(result[1]))
        return result

    def initialize_with_timing(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = _video_initialize_tree(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        capture["prefill_seconds"] = time.perf_counter() - start
        return result

    model._video_prefill_position_ids = prepared.position_ids
    model._video_rope_delta = prepared.rope_deltas
    model.forward = types.MethodType(_video_forward, model)

    def tree_decoding_with_timing(*args: Any, **kwargs: Any):
        if capture["verification_start"] is None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            capture["verification_start"] = time.perf_counter()
        result = _video_tree_decoding(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        capture["verification_end"] = time.perf_counter()
        return result

    ea_module.initialize_tree = initialize_with_timing
    ea_module.tree_decoding = tree_decoding_with_timing
    ea_module.evaluate_posterior = evaluate_with_trace
    try:
        yield capture
    finally:
        model.forward = old_forward
        lm_head_context.__exit__(None, None, None)
        model._video_prefill_position_ids = None
        model._video_rope_delta = None
        ea_module.initialize_tree = old_initialize
        ea_module.tree_decoding = old_tree
        ea_module.evaluate_posterior = old_evaluate
        # msdgenerate leaves tree_mask/tree_mode set on the base model after the
        # loop (reset_tree_mode only runs at the start). A following sample in
        # the same process would otherwise reuse the stale tree mask and corrupt
        # its causal mask (e.g. emitting <|im_end|> immediately). Clear them and
        # the cached rope deltas so the next sample starts from a clean state.
        qwen = model.base_model.model
        if hasattr(qwen, "tree_mask"):
            qwen.tree_mask = None
        if hasattr(qwen, "tree_mode"):
            qwen.tree_mode = None
        try:
            model.base_model.rope_deltas = None
        except Exception:
            pass


def generate_msd_full_video(
    model: Any,
    prepared: PreparedPrefill,
    max_new_tokens: int = 512,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run the original MSD algorithm with the local native-video patch.

    ``prepared`` may be the full target context or a compacted draft context
    produced by :func:`compact_qwen2vl_prefill`; the caller owns the target
    versus draft isolation check recorded in the result row.
    """

    if prepared.input_ids.shape[0] != 1:
        raise ValueError("MSD runtime currently supports batch size one")
    start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # EaModel.msdgenerate caches its preallocated KV tensors on the wrapper.
    # Reusing that cache across calibration points is unsafe: a later point
    # can have a longer prompt than the first point, while msdgenerate only
    # zeroes (rather than reallocates) the cached tensors.  The resulting
    # index error looks like a sequence-length/parity failure.  Force the
    # official allocator to size a fresh cache for every independent job.
    clear_msd_runtime_state(model)
    with patched_msd_video_path(model, prepared) as capture:
        output = model.msdgenerate(
            prepared.input_ids,
            inputs_embeds=prepared.inputs_embeds,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
            max_length=int(prepared.input_ids.shape[1] + max_new_tokens + 64),
            log=False,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    prefill_seconds = capture["prefill_seconds"]
    decode_seconds = elapsed - prefill_seconds if prefill_seconds is not None else elapsed
    decode_seconds = max(0.0, decode_seconds)
    verification_seconds = None
    if capture.get("verification_start") is not None and capture.get("verification_end") is not None:
        verification_seconds = max(0.0, capture["verification_end"] - capture["verification_start"])
    acceptance_trace = capture["trace"]
    return output, {
        "prefill_seconds": prefill_seconds,
        "draft_tree_prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "verification_seconds": verification_seconds if verification_seconds is not None else decode_seconds,
        "speculative_seconds": elapsed,
        "end_to_end_seconds": elapsed,
        "acceptance_trace": acceptance_trace,
        "accepted_prefix_tokens": (sum(acceptance_trace) / len(acceptance_trace)) if acceptance_trace else None,
        "verification_steps": len(acceptance_trace),
        "input_fingerprint": prepared.input_fingerprint,
        "video_token_count": int(prepared.video_positions.numel()),
    }


def generate_msd_retention_video(
    model: Any,
    prepared_full: PreparedPrefill,
    prepared_draft: PreparedPrefill,
    max_new_tokens: int = 512,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """MSD decode for the Figure 1(b) retention sweep.

    The *target* is prefilled and verified on the full video context while the
    *draft* tree is built on the compacted draft context only (``keep_mask``).
    This matches the paper's setup ("the target always keeps the full
    calibrated video; only the draft-side embedding sequence is compacted").
    The original harness path fed the compacted context to the target as well,
    which broke losslessness by construction; this function restores the split.

    The first draft tree (inside ``topK_genrate``) is the only place the draft
    receives real visual features; later tree rebuilds use placeholder visual
    embeddings exactly like the official ``msdgenerate`` loop.
    """

    from eagle.model import ea_model as ea_module

    if prepared_full.input_ids.shape[0] != 1 or prepared_draft.input_ids.shape[0] != 1:
        raise ValueError("MSD runtime currently supports batch size one")
    keep = prepared_draft.keep_mask
    if keep is None:
        raise ValueError("prepared_draft must carry keep_mask (use compact_qwen2vl_prefill)")
    start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    clear_msd_runtime_state(model)
    with patched_msd_video_path(model, prepared_full) as capture:
        model.ea_layer.reset_kv()
        past_key_values, past_key_values_data, current_length_data = ea_module.initialize_past_key_values(
            model.base_model,
            max_position_embeddings=prepared_full.input_ids.shape[1] + max_new_tokens + 64,
        )
        ea_module.reset_tree_mode(model)
        input_ids = prepared_full.input_ids.clone()
        input_len = int(input_ids.shape[1])
        max_length = input_len + max_new_tokens + 64 - model.ea_layer.total_tokens - 10

        # 1. Target prefill on the FULL context (full video + text).
        _outputs, orig, hidden_states = model(
            input_ids,
            past_key_values=past_key_values,
            output_orig=True,
            inputs_embeds=prepared_full.inputs_embeds,
            position_ids=prepared_full.position_ids,
        )
        token = torch.argmax(orig[:, -1], dim=-1, keepdim=True)

        # 2. Draft tree on the COMPACTED context: the kept target hidden states
        #    and the kept input embeddings.  topK_genrate extends the embedding
        #    sequence with the first sampled token itself, so the embeddings
        #    must be aligned with the compacted hidden states (length C).
        keep_gpu = keep.to(device=hidden_states.device)
        compact_hidden = hidden_states[:, keep_gpu]
        draft_input_ids = prepared_draft.input_ids
        extended_compact_ids = torch.cat((draft_input_ids, token.to(draft_input_ids.device)), dim=1)

        # Rebuild calls from update_inference_inputs pass the *full* growing
        # input_ids, but the draft's cached KV is the compacted one.  Remap
        # them to the compacted growing sequence: the initial compacted
        # context plus the accepted tokens that follow the full context.
        full_len = int(input_ids.shape[1])
        original_topk = model.ea_layer.topK_genrate

        def compacted_topk(*args: Any, **kwargs: Any):
            inputs_embeds = kwargs.get("inputs_embeds", args[4] if len(args) > 4 else None)
            if inputs_embeds is not None:
                # First call: the compacted context (with real visual features).
                return original_topk(*args, **kwargs)
            ids = kwargs.get("input_ids", args[1] if len(args) > 1 else None)
            if ids is None:
                raise ValueError("topK_genrate requires input_ids")
            compacted_ids = torch.cat((draft_input_ids, ids[:, full_len:]), dim=1)
            if "input_ids" in kwargs:
                kwargs["input_ids"] = compacted_ids
            else:
                args = list(args)
                args[1] = compacted_ids
                args = tuple(args)
            return original_topk(*args, **kwargs)

        model.ea_layer.topK_genrate = compacted_topk
        try:
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids = model.ea_layer.topK_genrate(
                compact_hidden,
                extended_compact_ids,
                model.base_model.lm_head,
                None,
                prepared_draft.inputs_embeds,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            capture["prefill_seconds"] = time.perf_counter() - start

            # 3. Verification loop: identical to msdgenerate, the target KV is
            #    the full-context cache and grows with the accepted tokens.
            new_token = 0
            for _idx in range(max_length):
                model.base_model.model.tree_mask = tree_mask
                draft_tokens = draft_tokens.to(input_ids.device)
                logits, hidden_state_new, _outputs = ea_module.tree_decoding(
                    model,
                    draft_tokens,
                    past_key_values,
                    tree_position_ids,
                    input_ids,
                    retrieve_indices,
                )
                draft_tokens = torch.cat(
                    (draft_tokens, torch.full((1, 1), -1, dtype=torch.long, device=input_ids.device)), dim=1
                )
                candidates = draft_tokens[0, retrieve_indices]
                best_candidate, accept_length, sample_p = ea_module.evaluate_posterior(
                    logits, candidates, None
                )
                (
                    input_ids,
                    draft_tokens,
                    retrieve_indices,
                    tree_mask,
                    tree_position_ids,
                    new_token,
                    _hidden_state,
                    _sample_token,
                ) = ea_module.update_inference_inputs(
                    input_ids,
                    candidates,
                    best_candidate,
                    accept_length,
                    retrieve_indices,
                    None,
                    new_token,
                    past_key_values_data,
                    current_length_data,
                    model,
                    hidden_state_new,
                    sample_p,
                )
                if model.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                    break
                if hasattr(model.tokenizer, "eod_id") and model.tokenizer.eod_id in input_ids[0, input_len:].tolist():
                    break
                if new_token > max_new_tokens:
                    break
                if input_ids.shape[1] > max_length:
                    break
        finally:
            model.ea_layer.topK_genrate = original_topk
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    prefill_seconds = capture["prefill_seconds"]
    decode_seconds = elapsed - prefill_seconds if prefill_seconds is not None else elapsed
    decode_seconds = max(0.0, decode_seconds)
    verification_seconds = None
    if capture.get("verification_start") is not None and capture.get("verification_end") is not None:
        verification_seconds = max(0.0, capture["verification_end"] - capture["verification_start"])
    acceptance_trace = capture["trace"]
    return input_ids, {
        "prefill_seconds": prefill_seconds,
        "draft_tree_prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "verification_seconds": verification_seconds if verification_seconds is not None else decode_seconds,
        "speculative_seconds": elapsed,
        "end_to_end_seconds": elapsed,
        "acceptance_trace": acceptance_trace,
        "accepted_prefix_tokens": (sum(acceptance_trace) / len(acceptance_trace)) if acceptance_trace else None,
        "verification_steps": len(acceptance_trace),
        "input_fingerprint": prepared_draft.input_fingerprint,
        "video_token_count": int(prepared_draft.video_positions.numel()),
    }
