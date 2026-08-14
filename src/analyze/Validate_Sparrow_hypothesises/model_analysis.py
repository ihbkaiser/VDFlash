"""Reusable GPU-side primitives for the Sparrow analysis experiments.

The paper's analysis figures are intentionally kept separate from the MSD
runner.  This module contains the small amount of model plumbing shared by
the attention, layer-ablation and hidden-state runners, while leaving result
schemas and dataset iteration to those scripts.
"""

from __future__ import annotations

import hashlib
import time
import types
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from .runtime import RuntimeUnavailableError, _hash_tensor, _tensor, model_device, move_batch_to_device


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _layers(model: Any) -> Any:
    """Find the decoder layer list in Qwen2-VL and Qwen2.5-VL variants."""

    candidates = [
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(model, "language_model", None),
        getattr(model, "model", None),
    ]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "layers"):
            return candidate.layers
    raise RuntimeUnavailableError("could not locate Qwen decoder layers")


def _language_model(model: Any) -> Any:
    candidates = [
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(model, "language_model", None),
        getattr(model, "model", None),
    ]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "layers"):
            return candidate
    raise RuntimeUnavailableError("could not locate Qwen language model")


def load_qwen_model(
    model_id: str,
    *,
    device_map: str = "auto",
    dtype: str = "float16",
    quantized: bool = False,
) -> Any:
    """Load a Qwen-VL checkpoint with eager attention for inspectable probes."""

    try:
        from transformers import AutoModelForVision2Seq, BitsAndBytesConfig
    except ImportError as exc:  # pragma: no cover - environment gate
        raise RuntimeUnavailableError("Transformers is required for model analysis") from exc
    if dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float32":
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float16
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": device_map,
        "torch_dtype": torch_dtype,
        "attn_implementation": "eager",
        "low_cpu_mem_usage": True,
    }
    if quantized:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype if torch_dtype != torch.float32 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    try:
        # Qwen2-VL is a vision-encoder-decoder: AutoModelForCausalLM rejects
        # Qwen2VLConfig on modern transformers, so use AutoModelForVision2Seq.
        model = AutoModelForVision2Seq.from_pretrained(model_id, **kwargs)
    except TypeError:
        # Older Transformers releases use the config field instead of the
        # from_pretrained keyword for the attention implementation.
        kwargs.pop("attn_implementation", None)
        model = AutoModelForVision2Seq.from_pretrained(model_id, **kwargs)
        if hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "eager"
    model.eval()
    return model


def _video_token_id(model: Any) -> int:
    config = getattr(model, "config", None)
    value = getattr(config, "video_token_id", None)
    if value is None:
        raise RuntimeUnavailableError("Qwen checkpoint has no video_token_id")
    return int(value)


@dataclass
class PreparedQwen25:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor | None
    inputs_embeds: torch.Tensor
    position_ids: torch.Tensor
    rope_deltas: torch.Tensor | None
    video_positions: torch.Tensor
    input_fingerprint: str


def prepare_qwen25_prefill(model: Any, batch: Any, device: torch.device | str | None = None) -> PreparedQwen25:
    """Construct the fused Qwen2.5-VL input embedding and M-RoPE positions."""

    input_ids = _tensor(batch, "input_ids")
    if input_ids is None:
        raise ValueError("processor batch has no input_ids")
    if device is not None:
        input_ids = input_ids.to(device)
    attention_mask = _tensor(batch, "attention_mask")
    if attention_mask is not None and device is not None:
        attention_mask = attention_mask.to(device)
    image_grid = _tensor(batch, "image_grid_thw")
    video_grid = _tensor(batch, "video_grid_thw")
    pixel_values = _tensor(batch, "pixel_values")
    pixel_values_videos = _tensor(batch, "pixel_values_videos")
    second_per_grid_ts = _tensor(batch, "second_per_grid_ts")
    if device is not None:
        image_grid = image_grid.to(device) if image_grid is not None else None
        video_grid = video_grid.to(device) if video_grid is not None else None
        pixel_values = pixel_values.to(device) if pixel_values is not None else None
        pixel_values_videos = pixel_values_videos.to(device) if pixel_values_videos is not None else None
        second_per_grid_ts = second_per_grid_ts.to(device) if second_per_grid_ts is not None else None

    vl_model = model.model
    with torch.inference_mode():
        inputs_embeds = vl_model.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            image_features = vl_model.get_image_features(pixel_values, image_grid)
            image_features = torch.cat(image_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = vl_model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
        if pixel_values_videos is None:
            raise ValueError("video experiment requires processor key pixel_values_videos")
        video_features = vl_model.get_video_features(pixel_values_videos, video_grid)
        video_features = torch.cat(video_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = vl_model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_features
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_features)
        position_ids, rope_deltas = vl_model.get_rope_index(
            input_ids,
            image_grid,
            video_grid,
            second_per_grid_ts=second_per_grid_ts,
            attention_mask=attention_mask,
        )
    video_positions = (input_ids[0] == _video_token_id(model)).nonzero(as_tuple=False).flatten()
    return PreparedQwen25(
        input_ids=input_ids,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        rope_deltas=rope_deltas,
        video_positions=video_positions,
        input_fingerprint=_hash_tensor(input_ids),
    )


def _special_ids(processor: Any) -> set[int]:
    tokenizer = getattr(processor, "tokenizer", processor)
    return {int(value) for value in getattr(tokenizer, "all_special_ids", [])}


def _find_subsequence(values: Sequence[int], needle: Sequence[int]) -> int | None:
    if not needle or len(needle) > len(values):
        return None
    for start in range(len(values) - len(needle) + 1):
        if list(values[start : start + len(needle)]) == list(needle):
            return start
    return None


def find_instruction_masks(
    input_ids: torch.Tensor,
    processor: Any,
    visual_positions: Iterable[int],
) -> dict[str, Any]:
    """Return disjoint instruction/visual/text masks for the paper probes.

    The final instruction token is selected from the user text immediately
    before the assistant generation marker.  This avoids accidentally probing
    a video placeholder or the assistant BOS token.
    """

    ids = [int(value) for value in input_ids[0].detach().to("cpu").tolist()]
    visual = {int(value) for value in visual_positions}
    special = _special_ids(processor)
    tokenizer = getattr(processor, "tokenizer", processor)
    assistant_ids: list[int] = []
    try:
        assistant_ids = [
            int(value)
            for value in tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        ]
    except Exception:
        assistant_ids = []
    assistant_start = _find_subsequence(ids, assistant_ids) if assistant_ids else None
    upper = assistant_start if assistant_start is not None else len(ids)
    candidates = [index for index in range(upper) if index not in visual and ids[index] not in special]
    # Remove user/video role markers while retaining ordinary question tokens.
    instruction = candidates
    if visual:
        last_visual = max(visual)
        after_video = [index for index in candidates if index > last_visual]
        if after_video:
            instruction = after_video
    if not instruction:
        instruction = [index for index in range(upper) if index not in visual]
    query_index = instruction[-1] if instruction else max(0, upper - 1)
    text = [index for index in range(len(ids)) if index not in visual and index not in instruction]
    return {
        "instruction_positions": instruction,
        "visual_positions": sorted(visual),
        "text_positions": text,
        "query_index": int(query_index),
        "attention_query": "last_instruction",
        "assistant_marker_start": assistant_start,
    }


def _attention_position_embeddings(module: Any, hidden_states: torch.Tensor, position_ids: Any, position_embeddings: Any):
    if position_embeddings is not None:
        return position_embeddings
    if position_ids is None or not hasattr(module, "rotary_emb"):
        return None
    return module.rotary_emb(hidden_states, position_ids)


def _mrope_section(module: Any) -> list[int]:
    scaling = getattr(module, "rope_scaling", None) or {}
    section = scaling.get("mrope_section") if hasattr(scaling, "get") else None
    if section is not None:
        return [int(value) for value in section]
    # Vanilla config objects used by CPU tests omit the Qwen multimodal rope
    # metadata.  The released Qwen checkpoints use [16, 24, 24] for head_dim
    # 128; derive the same partition for another divisible head dimension.
    head_dim = int(module.head_dim)
    first = head_dim // 8
    remainder = head_dim // 2 - first
    return [first, remainder // 2, remainder - remainder // 2]


def _query_attention(
    module: Any,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    position_embeddings: Any,
    query_index: int | Sequence[int],
) -> torch.Tensor:
    """Compute one attention row, avoiding an L-by-L attention tensor."""

    if hidden_states.shape[0] != 1:
        raise ValueError("query-only attention probes require batch size one")
    q_len = hidden_states.shape[1]
    query_indices = [int(query_index)] if isinstance(query_index, int) else [int(value) for value in query_index]
    if not query_indices or any(index < 0 or index >= q_len for index in query_indices):
        raise IndexError(f"query index {query_indices} outside sequence length {q_len}")
    query = module.q_proj(hidden_states)
    key = module.k_proj(hidden_states)
    query = query.view(1, q_len, -1, module.head_dim).transpose(1, 2)
    key = key.view(1, q_len, -1, module.head_dim).transpose(1, 2)
    rotary = _attention_position_embeddings(module, hidden_states, position_ids, position_embeddings)
    if rotary is not None:
        try:
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
        except ImportError:
            from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_multimodal_rotary_pos_emb
        query, key = apply_multimodal_rotary_pos_emb(
            query, key, rotary[0], rotary[1], _mrope_section(module)
        )
    groups = int(getattr(module, "num_key_value_groups", 1))
    if groups > 1:
        key = key.repeat_interleave(groups, dim=1)
    scores = torch.matmul(query[:, :, query_indices, :], key.transpose(-1, -2))
    scores = scores / (float(module.head_dim) ** 0.5)
    key_len = scores.shape[-1]
    if attention_mask is not None:
        if attention_mask.ndim == 4:
            mask = attention_mask[:, :, query_indices, :key_len]
        elif attention_mask.ndim == 3:
            mask = attention_mask[:, query_indices, :key_len].unsqueeze(1)
        elif attention_mask.ndim == 2:
            mask = attention_mask[:, None, None, :key_len]
            if mask.dtype == torch.bool or not torch.is_floating_point(mask):
                mask = torch.where(mask > 0, torch.zeros_like(mask, dtype=scores.dtype), torch.finfo(scores.dtype).min)
            else:
                mask = (1.0 - mask.to(scores.dtype)) * torch.finfo(scores.dtype).min
        else:
            raise ValueError(f"unsupported attention mask rank: {attention_mask.ndim}")
        scores = scores + mask.to(scores.dtype)
    else:
        causal = torch.arange(key_len, device=scores.device)[None, :] > torch.as_tensor(
            query_indices, device=scores.device
        )[:, None]
        scores = scores.masked_fill(causal[None, None, :, :], torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores.float(), dim=-1)
    result = weights[0].detach().to("cpu")
    return result[:, 0, :] if len(query_indices) == 1 else result.permute(1, 0, 2)


@contextmanager
def capture_query_attention(model: Any, query_index: int | Sequence[int]):
    """Capture one or more query rows from every decoder layer."""

    layers = _layers(model)
    captured: dict[int, torch.Tensor] = {}
    originals: list[tuple[Any, Any]] = []
    for layer_index, layer in enumerate(layers):
        module = layer.self_attn
        original = module.forward

        def wrapped(*args: Any, _module=module, _index=layer_index, _original=original, **kwargs: Any):
            output = _original(*args, **kwargs)
            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            if hidden_states is not None:
                attention_mask = kwargs.get("attention_mask")
                position_ids = kwargs.get("position_ids")
                position_embeddings = kwargs.get("position_embeddings")
                captured[_index] = _query_attention(
                    _module,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    position_embeddings,
                    query_index,
                )
            return output

        originals.append((module, original))
        module.forward = wrapped
    try:
        yield captured
    finally:
        for module, original in originals:
            module.forward = original


@contextmanager
def mask_visual_keys(model: Any, visual_positions: Iterable[int], first_layer: int):
    """Mask visual KV columns from ``first_layer`` onward during a forward."""

    layers = _layers(model)
    visual = tuple(sorted(int(value) for value in visual_positions))
    originals: list[tuple[Any, Any]] = []
    for layer_index, layer in enumerate(layers):
        if layer_index < first_layer:
            continue
        module = layer.self_attn
        original = module.forward

        def wrapped(*args: Any, _original=original, **kwargs: Any):
            mask = kwargs.get("attention_mask")
            if mask is not None and visual:
                patched = mask.clone()
                valid = [position for position in visual if position < patched.shape[-1]]
                if valid:
                    if patched.ndim == 4:
                        patched[..., valid] = torch.finfo(patched.dtype).min
                    elif patched.ndim == 3:
                        patched[..., valid] = torch.finfo(patched.dtype).min
                    elif patched.ndim == 2:
                        patched[:, valid] = 0
                    else:
                        raise ValueError(f"unsupported attention mask rank: {patched.ndim}")
                kwargs["attention_mask"] = patched
            return _original(*args, **kwargs)

        originals.append((module, original))
        module.forward = wrapped
    try:
        yield
    finally:
        for module, original in originals:
            module.forward = original


def _hidden_from_layer_output(output: Any) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    return output


def layerwise_input_cosine(
    model: Any,
    batch: Any,
    prepared: PreparedQwen25,
    visual_positions: Iterable[int],
    text_positions: Iterable[int],
) -> list[dict[str, float | int]]:
    """Collect per-layer visual/text cosine retention with bounded memory."""

    input_embeds = prepared.inputs_embeds
    visual = torch.as_tensor(sorted(int(value) for value in visual_positions), device=input_embeds.device)
    text = torch.as_tensor(sorted(int(value) for value in text_positions), device=input_embeds.device)
    if visual.numel() == 0 or text.numel() == 0:
        raise ValueError("visual and text positions must both be non-empty")
    result: list[dict[str, float | int]] = [{"layer": 0, "visual_cosine": 1.0, "text_cosine": 1.0}]
    originals: list[tuple[Any, Any]] = []
    layers = _layers(model)
    for layer_index, layer in enumerate(layers, start=1):
        original = layer.register_forward_hook(
            lambda _module, _inputs, output, index=layer_index: _record_cosine(
                result, index, _hidden_from_layer_output(output), input_embeds, visual, text
            )
        )
        originals.append((layer, original))
    try:
        with torch.inference_mode():
            model(**batch, use_cache=False, output_hidden_states=False, return_dict=True)
    finally:
        for _layer, handle in originals:
            handle.remove()
    return sorted(result, key=lambda row: int(row["layer"]))


def _record_cosine(
    result: list[dict[str, float | int]],
    layer: int,
    hidden: torch.Tensor,
    input_embeds: torch.Tensor,
    visual: torch.Tensor,
    text: torch.Tensor,
) -> None:
    values = torch.nn.functional.cosine_similarity(hidden[0].float(), input_embeds[0].float(), dim=-1)
    result.append({
        "layer": int(layer),
        "visual_cosine": float(values[visual].mean().item()),
        "text_cosine": float(values[text].mean().item()),
    })


def target_generation(
    model: Any,
    batch: Any,
    max_new_tokens: int,
) -> tuple[list[int], dict[str, float], torch.Tensor]:
    """Greedy native generation plus separately measured prefill/decode time."""

    input_ids = _tensor(batch, "input_ids")
    if input_ids is None:
        raise ValueError("batch has no input_ids")
    input_length = int(input_ids.shape[1])
    _sync()
    start = time.perf_counter()
    with torch.inference_mode():
        model(**batch, use_cache=True, return_dict=True)
    _sync()
    prefill = time.perf_counter() - start
    _sync()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **batch,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    _sync()
    end_to_end = time.perf_counter() - start
    # generate() includes its own prefill; it is the meaningful wall time for
    # the comparison, while decode is reported as the residual estimate.
    return (
        output[0, input_length:].detach().to("cpu").tolist(),
        {
            "prefill_seconds": float(prefill),
            "decode_seconds": float(max(0.0, end_to_end - prefill)),
            "end_to_end_seconds": float(end_to_end),
        },
        output,
    )


def prefix_length(reference: Sequence[int], candidate: Sequence[int]) -> int:
    common = 0
    for left, right in zip(reference, candidate):
        if left != right:
            break
        common += 1
    return common


def hash_tokens(tokens: Sequence[int]) -> str:
    payload = repr([int(value) for value in tokens]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
