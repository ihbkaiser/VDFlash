from __future__ import annotations

import torch
import torch.nn.functional as F


def block_loss_weights(
    block_size: int,
    decay: float,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return weights for predictions immediately following the anchor."""

    positions = torch.arange(block_size - 1, device=device, dtype=dtype)
    return torch.exp(-positions / float(decay))


def weighted_block_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    decay: float,
    ignore_index: int = -100,
    tensor_metrics: bool = False,
) -> tuple[torch.Tensor, dict[str, float] | dict[str, torch.Tensor]]:
    """Compute DFlash's position-weighted CE.

    ``logits`` is ``[..., block_size - 1, vocab]`` and ``labels`` is
    ``[..., block_size - 1]``.  The first prediction is position ``k=1`` in
    the paper, hence its weight is one.  Invalid/padded labels are ignored.
    """

    if logits.ndim != labels.ndim + 1:
        raise ValueError("logits must have exactly one more dimension than labels")
    if logits.shape[:-1] != labels.shape:
        raise ValueError(f"shape mismatch: {tuple(logits.shape)} vs {tuple(labels.shape)}")

    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    token_loss = F.cross_entropy(
        flat_logits.float(), flat_labels, reduction="none", ignore_index=ignore_index
    ).reshape_as(labels)
    valid = labels.ne(ignore_index)
    weights = block_loss_weights(
        labels.shape[-1] + 1, decay, device=labels.device, dtype=token_loss.dtype
    )
    weights = weights.view(*([1] * (labels.ndim - 1)), -1)
    weighted = token_loss * weights * valid
    normalizer = (weights * valid).sum().clamp_min(1e-12)
    loss = weighted.sum() / normalizer
    with torch.no_grad():
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions.eq(labels) & valid).sum() / valid.sum().clamp_min(1)
        if tensor_metrics:
            # Cached training aggregates these on device and transfers only one
            # packed metric tensor per optimizer step. Calling ``.cpu()`` for
            # every anchor chunk otherwise serializes the CUDA pipeline.
            metrics = {
                "loss": loss.detach(),
                "token_accuracy": accuracy.detach(),
                "valid_tokens": valid.sum().detach(),
            }
        else:
            metrics = {
                "loss": float(loss.detach().cpu()),
                "token_accuracy": float(accuracy.detach().cpu()),
                "valid_tokens": float(valid.sum().detach().cpu()),
            }
    return loss, metrics
