"""Runtime for the MSD draft-hidden reverse-transfer experiment.

The target is used only to create the teacher hidden-state conditioning that
the trained MSD draft expects.  Candidate logits are always produced by
``model.ea_layer`` followed by the frozen target LM head.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence, TYPE_CHECKING

import torch

from .reverse_transfer import (
    CandidateTokens,
    ReverseTransferSample,
    answer_metrics,
    build_user_message,
    canonical_task_text,
    condition_spec,
    tokenize_candidates,
)
if TYPE_CHECKING:
    from .runtime import PreparedPrefill


class MSDReverseTransferError(RuntimeError):
    """Raised when a row cannot be certified as an MSD draft prediction."""


@dataclass
class PreparedReverseInput:
    sample: ReverseTransferSample
    condition: Any
    messages: list[dict[str, Any]]
    processor_batch: Any
    prepared: "PreparedPrefill"
    text_positions: torch.Tensor
    task_text_positions: torch.Tensor
    rendered_prompt: str

    def position_metadata(self) -> dict[str, Any]:
        visual = self.prepared.video_positions.detach().cpu().tolist()
        text = self.text_positions.detach().cpu().tolist()
        task_text = self.task_text_positions.detach().cpu().tolist()
        return {
            "input_order": self.condition.input_order,
            "visual_positions": visual,
            "text_positions": text,
            "task_text_positions": task_text,
            "visual_count": len(visual),
            "text_count": len(text),
            "task_text_count": len(task_text),
            "visual_first_position": min(visual) if visual else None,
            "visual_last_position": max(visual) if visual else None,
            "text_first_position": min(text) if text else None,
            "text_last_position": max(text) if text else None,
            "task_text_first_position": min(task_text) if task_text else None,
            "task_text_last_position": max(task_text) if task_text else None,
            "sequence_length": int(self.prepared.input_ids.shape[1]),
            "actual_video_token_id": int(self.prepared.video_token_id),
            "target_input_fingerprint": self.prepared.input_fingerprint,
        }


@dataclass
class DraftPrediction:
    logits: torch.Tensor
    draft_hidden: torch.Tensor
    target_hidden: torch.Tensor
    input_embeddings: torch.Tensor
    masked_input_embeddings: torch.Tensor
    masked_target_hidden: torch.Tensor
    feature_mask_metadata: dict[str, Any]


def _model_input_device(model: Any) -> torch.device:
    try:
        return model.base_model.model.embed_tokens.weight.device
    except Exception:
        return next(model.parameters()).device


def _last_hidden(output: Any) -> torch.Tensor:
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def _hash_tensor(value: torch.Tensor) -> str:
    data = value.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


def _get_vision_inputs(processor: Any, messages: list[dict[str, Any]]) -> tuple[Any, Any, dict[str, Any]]:
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:  # pragma: no cover - GPU environment dependency
        raise MSDReverseTransferError("qwen-vl-utils is required for ordered video inputs") from exc
    try:
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    except TypeError:
        image_inputs, video_inputs = process_vision_info(messages)
        video_kwargs = {}
    return image_inputs, video_inputs, dict(video_kwargs or {})


def _find_token_subsequence(sequence: Sequence[int], subsequence: Sequence[int]) -> torch.Tensor:
    """Return absolute positions of an exact token span, or an empty tensor."""

    values = [int(value) for value in sequence]
    needle = [int(value) for value in subsequence]
    if not needle or len(needle) > len(values):
        return torch.empty(0, dtype=torch.long)
    for start in range(len(values) - len(needle) + 1):
        if values[start : start + len(needle)] == needle:
            return torch.arange(start, start + len(needle), dtype=torch.long)
    return torch.empty(0, dtype=torch.long)


def prepare_reverse_input(
    processor: Any,
    target_model: Any,
    sample: ReverseTransferSample,
    *,
    input_order: str,
    device: torch.device,
    fps: float,
    num_frames: int | None,
    max_pixels: int | None,
    include_video: bool = True,
    include_text: bool = True,
) -> PreparedReverseInput:
    from .runtime import move_batch_to_device, prepare_qwen2vl_prefill

    messages = build_user_message(
        sample,
        input_order=input_order,
        fps=fps,
        num_frames=num_frames,
        max_pixels=max_pixels,
        include_video=include_video,
        include_text=include_text,
    )
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = _get_vision_inputs(processor, messages)
    kwargs: dict[str, Any] = {
        "text": [rendered],
        "images": image_inputs,
        "videos": video_inputs,
        "padding": True,
        "return_tensors": "pt",
    }
    kwargs.update(video_kwargs)
    batch = processor(**kwargs)
    batch = move_batch_to_device(batch, device)
    # The public runner owns the EaModel wrapper, while the multimodal
    # prefill contract belongs to its frozen target Qwen2-VL base model.
    target_base = getattr(target_model, "base_model", target_model)
    prepared = prepare_qwen2vl_prefill(target_base, batch, device)
    visual = prepared.video_positions
    all_positions = torch.arange(prepared.input_ids.shape[1], device=prepared.input_ids.device)
    text_positions = all_positions[~torch.isin(all_positions, visual)]
    if include_text:
        task_text = canonical_task_text(sample.question, sample.options)
        tokenized_text = processor.tokenizer(task_text, add_special_tokens=False)["input_ids"]
        task_text_positions = _find_token_subsequence(
            prepared.input_ids[0].detach().cpu().tolist(), tokenized_text
        ).to(prepared.input_ids.device)
        if task_text_positions.numel() == 0:
            raise MSDReverseTransferError("could not locate canonical task-text token span in the rendered prompt")
    else:
        task_text_positions = torch.empty(0, dtype=torch.long, device=prepared.input_ids.device)
    return PreparedReverseInput(
        sample,
        condition_spec("E1" if input_order == "video_text" else "E2"),
        messages,
        batch,
        prepared,
        text_positions,
        task_text_positions,
        rendered,
    )


def target_prefill(model: Any, prepared: PreparedReverseInput) -> torch.Tensor:
    """Return target hidden states; no target vocabulary logits are computed."""

    base = model.base_model
    with torch.inference_mode():
        output = base.model(
            inputs_embeds=prepared.prepared.inputs_embeds,
            attention_mask=prepared.prepared.attention_mask,
            position_ids=prepared.prepared.position_ids,
            use_cache=False,
            return_dict=True,
        )
    hidden = _last_hidden(output)
    if hidden.ndim != 3 or hidden.shape[:2] != prepared.prepared.input_ids.shape:
        raise MSDReverseTransferError(
            f"target hidden shape {tuple(hidden.shape)} does not match input {tuple(prepared.prepared.input_ids.shape)}"
        )
    return hidden


def _mask_features(
    prepared: PreparedReverseInput,
    target_hidden: torch.Tensor,
    conditioning: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if conditioning not in {"full", "visual", "text"}:
        raise ValueError(f"unsupported conditioning: {conditioning}")
    input_embeddings = prepared.prepared.inputs_embeds
    masked_embeddings = input_embeddings.clone()
    masked_hidden = target_hidden.clone()
    visual = torch.zeros(input_embeddings.shape[1], dtype=torch.bool, device=input_embeddings.device)
    visual[prepared.prepared.video_positions.to(input_embeddings.device)] = True
    text = ~visual
    if conditioning == "visual":
        masked_embeddings[:, text] = 0
        masked_hidden[:, text] = 0
    elif conditioning == "text":
        masked_embeddings[:, visual] = 0
        masked_hidden[:, visual] = 0
    metadata = {
        "conditioning": conditioning,
        "preserves_sequence_length": True,
        "visual_positions_preserved": True,
        "text_positions_zeroed": int(text.sum().item()) if conditioning == "visual" else 0,
        "visual_positions_zeroed": int(visual.sum().item()) if conditioning == "text" else 0,
        "target_prefill_was_full_multimodal": True,
        "target_visual_hidden_retained_in_visual_condition": conditioning == "visual",
        "draft_visual_embedding_source": "target_visual_tower_output",
        "draft_context_source": "target_final_hidden_states",
    }
    return masked_hidden, masked_embeddings, metadata


@contextmanager
def _freeze_draft_embedding_branch(model: Any) -> Iterator[None]:
    """Make the vendored forward consume our exact masked embeddings.

    The external code reconstructs embeddings from ``input_ids`` whenever its
    embedding parameters have ``requires_grad=True``.  That is correct during
    training but would undo zeroed feature rows during an ablation.  Inference
    temporarily disables that reconstruction and restores the flags exactly.
    """

    embedding = model.ea_layer.embed_tokens
    flags = [parameter.requires_grad for parameter in embedding.parameters()]
    try:
        for parameter in embedding.parameters():
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, flag in zip(embedding.parameters(), flags):
            parameter.requires_grad_(flag)


def _draft_forward(model: Any, prepared: PreparedReverseInput, target_hidden: torch.Tensor, conditioning: str) -> DraftPrediction:
    masked_hidden, masked_embeddings, mask_metadata = _mask_features(prepared, target_hidden, conditioning)
    draft = model.ea_layer
    draft.eval()
    draft_fc = getattr(draft, "fc", None)
    if draft_fc is not None:
        expected_width = int(draft_fc.in_features // 2)
        if masked_hidden.shape[-1] != expected_width or masked_embeddings.shape[-1] != expected_width:
            raise MSDReverseTransferError(
                "MSD draft input width mismatch: "
                f"target_hidden={tuple(masked_hidden.shape)}, "
                f"input_embeddings={tuple(masked_embeddings.shape)}, "
                f"draft_fc={tuple(draft_fc.weight.shape)}, "
                f"expected_each_width={expected_width}"
            )
    attention_mask = prepared.prepared.attention_mask
    with _freeze_draft_embedding_branch(model), torch.inference_mode():
        output = draft(
            hidden_states=masked_hidden,
            input_ids=prepared.prepared.input_ids,
            attention_mask=attention_mask,
            # The vendored EAGLE Qwen2-VL draft expects its native 2-D,
            # sequential position contract and expands it to 3 M-RoPE
            # channels internally.  Passing the target's [3, batch, seq]
            # M-RoPE tensor through cnets.Model would flatten the channel
            # axis into the batch axis.  Target M-RoPE is still used for the
            # target prefill above; the draft receives its own native
            # sequential positions here.
            position_ids=None,
            inputs_embeds=masked_embeddings,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )
    draft_hidden = _last_hidden(output)
    if draft_hidden.ndim != 3 or draft_hidden.shape[:2] != prepared.prepared.input_ids.shape:
        raise MSDReverseTransferError(f"MSD draft output has invalid shape {tuple(draft_hidden.shape)}")
    # This is the exact next-token state at the final prompt position.  The
    # candidate answer token is not appended to input_ids and cannot leak.
    predictive_hidden = draft_hidden[:, -1:, :]
    try:
        with torch.inference_mode():
            logits = model.base_model.lm_head(predictive_hidden)
    except RuntimeError:
        try:
            head_device = model.base_model.lm_head.weight.device
            with torch.inference_mode():
                logits = model.base_model.lm_head(predictive_hidden.to(head_device))
        except Exception as exc:  # pragma: no cover - sharded checkpoint path
            raise MSDReverseTransferError("MSD draft hidden could not reach the frozen LM head") from exc
    return DraftPrediction(
        logits=logits[:, 0],
        draft_hidden=predictive_hidden,
        target_hidden=target_hidden[:, -1:],
        input_embeddings=prepared.prepared.inputs_embeds[:, -1:],
        masked_input_embeddings=masked_embeddings[:, -1:],
        masked_target_hidden=masked_hidden[:, -1:],
        feature_mask_metadata={
            **mask_metadata,
            "draft_hidden_source": "model.ea_layer.output_after_draft_decoder",
            "lm_head_source": "model.base_model.lm_head(draft_hidden)",
            "target_logits_used": False,
            "answer_position": int(prepared.prepared.input_ids.shape[1] - 1),
            "draft_position_source": "model.ea_layer native sequential positions",
            "future_answer_token_provided": False,
            "draft_hidden_fingerprint": _hash_tensor(predictive_hidden),
            "target_hidden_fingerprint": _hash_tensor(target_hidden[:, -1:]),
        },
    )


def score_draft_candidates(
    model: Any,
    prepared: PreparedReverseInput,
    target_hidden: torch.Tensor,
    *,
    conditioning: str,
) -> dict[str, Any]:
    prediction = _draft_forward(model, prepared, target_hidden, conditioning)
    candidates = tokenize_candidates(model.get_tokenizer(), prepared.sample.labels)
    log_scores = {candidate.label: float(prediction.logits[0, candidate.token_id].item()) for candidate in candidates}
    metrics = answer_metrics(log_scores, prepared.sample.correct_label)
    return {
        **metrics,
        "candidate_token_ids": {candidate.label: candidate.token_id for candidate in candidates},
        "candidate_raw_token_ids": {candidate.label: list(candidate.raw_token_ids) for candidate in candidates},
        "candidate_spaced_token_ids": {candidate.label: list(candidate.spaced_token_ids) for candidate in candidates},
        "feature_mask_metadata": prediction.feature_mask_metadata,
        "draft_hidden_shape": list(prediction.draft_hidden.shape),
        "target_hidden_shape": list(prediction.target_hidden.shape),
    }


def load_msd_runtime(
    *,
    target_model: str,
    msd_checkpoint: str,
    device_map: str,
    max_memory: str | None,
    min_pixels: int,
    max_pixels: int,
) -> tuple[Any, Any, torch.device]:
    from .runtime import build_qwen2vl_video_processor, load_msd_qwen2vl, require_cuda

    require_cuda()
    model = load_msd_qwen2vl(
        target_model,
        msd_checkpoint,
        device_map=device_map,
        max_memory=max_memory,
    )
    if not hasattr(model, "ea_layer") or not hasattr(model, "base_model"):
        raise MSDReverseTransferError("checkpoint did not load as the vendored EaModel/MSD wrapper")
    config_type = str(getattr(model.base_model.config, "model_type", ""))
    if config_type != "qwen2_vl":
        raise MSDReverseTransferError(
            f"MSD reverse-transfer runner requires the vendored Qwen2-VL MSD model, got {config_type!r}"
        )
    target_width = int(getattr(model.base_model.config, "hidden_size", -1))
    draft_width = int(model.ea_layer.fc.out_features)
    if target_width <= 0 or draft_width != target_width or int(model.ea_layer.fc.in_features) != 2 * target_width:
        raise MSDReverseTransferError(
            "incompatible target/MSD hidden width: "
            f"target={target_width}, draft_fc={tuple(model.ea_layer.fc.weight.shape)}"
        )
    processor = build_qwen2vl_video_processor(target_model, min_pixels, max_pixels)
    return model, processor, _model_input_device(model)


def architecture_trace(model: Any, target_model: str, msd_checkpoint: str) -> dict[str, Any]:
    draft = model.ea_layer
    target_config = model.base_model.config
    draft_config = getattr(draft, "config", None)
    attention = draft.layers[0].self_attn
    return {
        "target_model": target_model,
        "msd_checkpoint": msd_checkpoint,
        "target_module": "model.base_model.model",
        "draft_module": "model.ea_layer (eagle.model.cnets.Model)",
        "draft_input_contract": "target_final_hidden_states + modality-aware input embeddings",
        "draft_visual_embedding_path": "prepared.inputs_embeds visual rows from target.visual",
        "draft_context_path": "target base_model.model final hidden states",
        "draft_transformer_layers": int(len(draft.layers)),
        "draft_hidden_size": int(draft.fc.out_features),
        "target_config_hidden_size": int(getattr(target_config, "hidden_size", -1)),
        "draft_config_hidden_size": int(getattr(draft_config, "hidden_size", -1)),
        "draft_config_intermediate_size": int(getattr(draft_config, "intermediate_size", -1)),
        "draft_fc_shape": list(draft.fc.weight.shape),
        "draft_attention_hidden_size": int(getattr(attention, "hidden_size", -1)),
        "draft_attention_num_heads": int(getattr(attention, "num_heads", -1)),
        "draft_attention_num_key_value_heads": int(getattr(attention, "num_key_value_heads", -1)),
        "draft_attention_num_key_value_groups": int(getattr(attention, "num_key_value_groups", -1)),
        "draft_attention_head_dim": int(getattr(attention, "head_dim", -1)),
        "draft_q_proj_shape": list(attention.q_proj.weight.shape),
        "draft_k_proj_shape": list(attention.k_proj.weight.shape),
        "draft_v_proj_shape": list(attention.v_proj.weight.shape),
        "draft_o_proj_shape": list(attention.o_proj.weight.shape),
        "projection": "model.base_model.lm_head(draft_hidden[:, -1:])",
        "target_logits_used_for_primary_scores": False,
        "eagle_first_token_note": "raw context draft state is scored at the final prompt position; standard msdgenerate still seeds its first emitted token from target logits",
    }


__all__ = [
    "DraftPrediction",
    "MSDReverseTransferError",
    "PreparedReverseInput",
    "architecture_trace",
    "load_msd_runtime",
    "prepare_reverse_input",
    "score_draft_candidates",
    "target_prefill",
]
