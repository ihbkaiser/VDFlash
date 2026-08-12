"""Model-side utilities for native Qwen2-VL video inputs.

This module is intentionally isolated from the CPU-side audit code.  The
adapter validates the multimodal prefill before speculative decoding is
allowed.  In particular, it never silently routes a video through the
image-only helper shipped in the original MSD repository.
"""

from __future__ import annotations

import hashlib
import sys
import time
import types
from contextlib import contextmanager
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
    """Compatibility shims so the vendored EAGLE (transformers 4.48-era) code
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
    )


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
) -> list[dict[str, Any]]:
    video = {"type": "video", "video": str(video_path), "fps": float(fps)}
    if max_pixels is not None:
        video["max_pixels"] = int(max_pixels)
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
) -> Any:
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:  # pragma: no cover
        raise RuntimeUnavailableError("qwen-vl-utils is required for Qwen video preprocessing") from exc
    messages = build_video_messages(video_path, question, fps, max_pixels=max_pixels)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    try:
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    except TypeError:
        image_inputs, video_inputs = process_vision_info(messages)
        video_kwargs = {}
    kwargs = dict(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    kwargs.update(video_kwargs)
    return processor(**kwargs)


def load_msd_qwen2vl(base_model_path: str, msd_model_path: str, device_map: str = "cuda") -> Any:
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
    result = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=msd_model_path,
        total_token=30,
        depth=4,
        top_k=8,
        torch_dtype=torch.float16,
        device_map=device_map,
        quantization_config=quantization_config,
    )
    # EAGLE's EaModel.from_pretrained returns (model, None) for the Qwen2-VL
    # path; unwrap it so the caller receives the EaModel itself.
    if isinstance(result, tuple):
        result = result[0]
    # The draft cnets.Model enables gradient checkpointing and nn.Module starts
    # in training mode; leaving it trainable routes inference through the
    # torch.utils.checkpoint branch whose custom_forward drops use_cache and
    # returns a single-element tuple. eval() disables that branch.
    result.eval()
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
    capture: dict[str, Any] = {"trace": [], "prefill_seconds": None}

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
    ea_module.initialize_tree = initialize_with_timing
    ea_module.tree_decoding = _video_tree_decoding
    ea_module.evaluate_posterior = evaluate_with_trace
    try:
        yield capture
    finally:
        model.forward = old_forward
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
    acceptance_trace = capture["trace"]
    return output, {
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "speculative_seconds": elapsed,
        "end_to_end_seconds": elapsed,
        "acceptance_trace": acceptance_trace,
        "accepted_prefix_tokens": (sum(acceptance_trace) / len(acceptance_trace)) if acceptance_trace else None,
        "verification_steps": len(acceptance_trace),
        "input_fingerprint": prepared.input_fingerprint,
        "video_token_count": int(prepared.video_positions.numel()),
    }
