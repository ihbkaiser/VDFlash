from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Callable

import torch


_COMPILED_CREATE_BLOCK_MASK = None


@dataclass
class MaskedBlockBatch:
    """One packed anchor batch for the draft model.

    The tensors are batch-size one by design.  Anchor chunks bound activation
    memory on a single GPU while retaining exactly the same objective as one
    large packed FlexAttention call.
    """

    block_input_ids: torch.Tensor  # [num_blocks, block_size]
    labels: torch.Tensor  # [num_blocks, block_size], anchor label is -100
    block_position_ids: torch.Tensor  # [3, 1, num_blocks * block_size]
    anchors: torch.Tensor  # [num_blocks], original sequence positions


def _as_position_ids(position_ids: torch.Tensor, length: int) -> torch.Tensor:
    if position_ids.ndim == 2:
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    elif position_ids.ndim == 3 and position_ids.shape[0] != 3:
        # [batch, 3, length] is accepted for convenience.
        if position_ids.shape[1] == 3:
            position_ids = position_ids.transpose(0, 1)
    if position_ids.ndim != 3 or position_ids.shape[0] != 3:
        raise ValueError("position_ids must have shape [3, batch, length] or [batch, length]")
    if position_ids.shape[1] != 1 or position_ids.shape[-1] != length:
        raise ValueError("this implementation currently expects batch size one")
    return position_ids


def sample_anchor_positions(
    response_start: int,
    response_end: int,
    block_size: int,
    num_anchors: int,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Sample valid response anchors, keeping a fixed number per sequence."""

    last_anchor = response_end - block_size
    if last_anchor < response_start:
        raise ValueError(
            "response does not contain a complete anchor block: "
            f"[{response_start}, {response_end}) with block_size={block_size}"
        )
    candidates = torch.arange(response_start, last_anchor + 1, device=device)
    if candidates.numel() >= num_anchors:
        perm = torch.randperm(candidates.numel(), generator=generator, device=device)
        return candidates[perm[:num_anchors]]
    # Short responses are repeated to preserve the paper's fixed block count.
    indices = torch.randint(
        candidates.numel(), (num_anchors,), generator=generator, device=device
    )
    return candidates[indices]


def make_anchor_generator(
    seed: int,
    epoch: int,
    sample_id: str,
    *,
    device: torch.device | str | None = None,
) -> torch.Generator:
    """Create an order-independent, reproducible anchor RNG for one sample.

    Persisting a single global generator makes anchors depend on dataloader
    ordering and makes an interrupted run impossible to reproduce exactly.
    Hashing the run seed, epoch, and manifest ID gives each sample a stable
    stream without imposing an integer-only ID convention on manifests.
    """

    payload = f"dflash-anchor-v1:{seed}:{epoch}:{sample_id}".encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)
    generator = torch.Generator(device=device)
    generator.manual_seed(derived_seed)
    return generator


def build_masked_blocks(
    input_ids: torch.Tensor,
    anchors: torch.Tensor,
    *,
    block_size: int,
    mask_token_id: int,
    position_ids: torch.Tensor,
) -> MaskedBlockBatch:
    """Construct ``[anchor, MASK, ...]`` blocks and next-token labels."""

    if input_ids.ndim == 2:
        if input_ids.shape[0] != 1:
            raise ValueError("build_masked_blocks currently expects batch size one")
        input_ids = input_ids[0]
    if input_ids.ndim != 1:
        raise ValueError("input_ids must have shape [length] or [1, length]")
    position_ids = _as_position_ids(position_ids, input_ids.numel())
    offsets = torch.arange(block_size, device=input_ids.device)
    positions = anchors[:, None] + offsets[None, :]
    clean = input_ids[positions]
    block_input_ids = clean.clone()
    block_input_ids[:, 1:] = mask_token_id
    labels = clean.clone()
    labels[:, 0] = -100
    block_positions = position_ids[:, :, positions.reshape(-1)]
    return MaskedBlockBatch(block_input_ids, labels, block_positions, anchors)


def select_context_positions(
    input_ids: torch.Tensor,
    *,
    context_mode: str,
    image_token_ids: set[int] | None = None,
    video_token_ids: set[int] | None = None,
) -> torch.Tensor:
    """Return original positions retained by full/text-only target context."""

    if input_ids.ndim == 2:
        input_ids = input_ids[0]
    keep = torch.ones_like(input_ids, dtype=torch.bool)
    if context_mode == "text_only":
        visual_ids = (image_token_ids or set()) | (video_token_ids or set())
        for token_id in visual_ids:
            keep &= input_ids.ne(token_id)
    elif context_mode != "full":
        raise ValueError("context_mode must be 'full' or 'text_only'")
    return keep.nonzero(as_tuple=False).flatten()


def make_dense_attention_mask(
    anchors: torch.Tensor,
    context_original_positions: torch.Tensor,
    *,
    block_size: int,
    context_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Create a boolean ``[batch, 1, Q, KV]`` mask for CPU/tests/fallback."""

    if anchors.ndim == 1:
        anchors = anchors.unsqueeze(0)
    if context_original_positions.ndim == 1:
        context_original_positions = context_original_positions.unsqueeze(0)
    if anchors.ndim != 2 or context_original_positions.ndim != 2:
        raise ValueError("anchors and context_original_positions must be one- or two-dimensional")
    if anchors.shape[0] != context_original_positions.shape[0]:
        raise ValueError("anchors and context positions must have the same batch size")
    batch_size, num_blocks = anchors.shape
    context_len = context_original_positions.shape[1]
    if context_lengths is None:
        context_lengths = torch.full(
            (batch_size,), context_len, dtype=torch.long, device=anchors.device
        )
    else:
        context_lengths = context_lengths.to(device=anchors.device, dtype=torch.long).view(-1)
    if context_lengths.shape[0] != batch_size:
        raise ValueError("context_lengths must contain one value per batch item")
    q_len = num_blocks * block_size
    kv_len = context_len + q_len
    q_block = torch.arange(q_len, device=anchors.device) // block_size
    context_index = torch.arange(context_len, device=anchors.device)
    context_valid = context_index.unsqueeze(0) < context_lengths.unsqueeze(1)
    ctx_allowed = (
        context_valid[:, None, :]
        & (
            context_original_positions[:, None, :]
            < anchors[:, q_block, None]
        )
    )
    noise_index = torch.arange(q_len, device=anchors.device)
    noise_allowed = (noise_index[None, :] // block_size) == q_block[:, None]
    noise_allowed = noise_allowed.unsqueeze(0).expand(batch_size, -1, -1)
    mask = torch.cat([ctx_allowed, noise_allowed], dim=-1)
    if mask.shape != (batch_size, q_len, kv_len):
        raise RuntimeError("constructed dense attention mask has an invalid shape")
    return mask.unsqueeze(1)


def make_flex_block_mask(
    anchors: torch.Tensor,
    context_original_positions: torch.Tensor,
    *,
    block_size: int,
    device: torch.device,
    compile_mask: bool = True,
    context_lengths: torch.Tensor | None = None,
):
    """Build a PyTorch FlexAttention BlockMask for packed DFlash blocks."""

    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except ImportError as exc:  # pragma: no cover - depends on torch build
        raise RuntimeError("PyTorch FlexAttention is unavailable") from exc

    if anchors.ndim == 1:
        anchors = anchors.unsqueeze(0)
    if context_original_positions.ndim == 1:
        context_original_positions = context_original_positions.unsqueeze(0)
    if anchors.ndim != 2 or context_original_positions.ndim != 2:
        raise ValueError("anchors and context_original_positions must be one- or two-dimensional")
    if anchors.shape[0] != context_original_positions.shape[0]:
        raise ValueError("anchors and context positions must have the same batch size")
    batch_size, num_blocks = (int(value) for value in anchors.shape)
    context_len = int(context_original_positions.shape[1])
    q_len = num_blocks * block_size
    kv_len = context_len + q_len
    anchors = anchors.to(device)
    context_original_positions = context_original_positions.to(device)
    if context_lengths is None:
        context_lengths = torch.full(
            (batch_size,), context_len, dtype=torch.long, device=device
        )
    else:
        context_lengths = context_lengths.to(device=device, dtype=torch.long).view(-1)
    if context_lengths.shape[0] != batch_size:
        raise ValueError("context_lengths must contain one value per batch item")

    def mask_mod(batch, head, q_idx, kv_idx):
        block = q_idx // block_size
        is_context = kv_idx < context_len
        if context_len:
            context_index = torch.clamp(kv_idx, 0, context_len - 1)
            context_ok = (
                is_context
                & (context_index < context_lengths[batch])
                & (
                    context_original_positions[batch, context_index]
                    < anchors[batch, block]
                )
            )
        else:
            context_ok = torch.zeros_like(is_context)
        noise_index = kv_idx - context_len
        noise_ok = (~is_context) & ((noise_index // block_size) == block)
        return context_ok | noise_ok

    global _COMPILED_CREATE_BLOCK_MASK
    creator = create_block_mask
    if compile_mask:
        if _COMPILED_CREATE_BLOCK_MASK is None:
            _COMPILED_CREATE_BLOCK_MASK = torch.compile(create_block_mask, dynamic=False)
        creator = _COMPILED_CREATE_BLOCK_MASK
    return creator(
        mask_mod,
        B=batch_size,
        H=None,
        Q_LEN=q_len,
        KV_LEN=kv_len,
        device=device,
        _compile=False,
    )
