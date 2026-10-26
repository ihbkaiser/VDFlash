"""Target verification attention reductions used by Video-DFlash pilots."""

from __future__ import annotations

import torch
import torch.nn.functional as F


ATTENTION_METHODS = ("M0", "M1", "M2", "M3", "M5")


def causal_mean_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    *,
    past_length: int,
    query_count: int,
) -> torch.Tensor:
    """Return mean-head causal attention for a verified query prefix.

    ``query_states`` and ``key_states`` are post-RoPE tensors in
    ``[batch, heads, sequence, head_dim]`` layout. Key/value head groups are
    repeated to the query-head count, then probabilities are averaged over
    heads. Only the accepted query prefix and its causal key domain are scored.
    """

    if query_states.ndim != 4 or key_states.ndim != 4:
        raise ValueError("query_states and key_states must be rank-four tensors")
    if query_states.shape[0] != key_states.shape[0]:
        raise ValueError("query and key batch sizes must match")
    if query_states.shape[-1] != key_states.shape[-1]:
        raise ValueError("query and key head dimensions must match")
    if past_length < 0 or query_count <= 0:
        raise ValueError("past_length must be non-negative and query_count positive")
    if query_count > query_states.shape[-2]:
        raise ValueError("query_count exceeds the captured verification queries")

    scored_key_count = int(past_length) + int(query_count)
    if scored_key_count > key_states.shape[-2]:
        raise ValueError("accepted query prefix extends beyond captured target keys")

    query_heads = query_states.shape[1]
    key_heads = key_states.shape[1]
    if query_heads % key_heads:
        raise ValueError("query heads must be divisible by key/value heads")

    queries = query_states[:, :, :query_count, :].float()
    keys = key_states[:, :, :scored_key_count, :].float()
    if query_heads != key_heads:
        keys = keys.repeat_interleave(query_heads // key_heads, dim=1)

    logits = torch.matmul(queries, keys.transpose(-2, -1))
    logits.mul_(query_states.shape[-1] ** -0.5)
    query_positions = torch.arange(query_count, device=logits.device) + int(past_length)
    key_positions = torch.arange(scored_key_count, device=logits.device)
    causal_mask = key_positions.view(1, 1, 1, -1) <= query_positions.view(1, 1, -1, 1)
    logits.masked_fill_(~causal_mask, torch.finfo(logits.dtype).min)
    probabilities = torch.softmax(logits, dim=-1)
    return probabilities.mean(dim=1)


def reduce_attention_scores(
    method: str,
    layer18_attention: torch.Tensor,
    layer35_attention: torch.Tensor,
    *,
    pool_kernel: int = 7,
) -> torch.Tensor:
    """Reduce per-query target attention into one context score per token.

    Attention inputs have shape ``[batch, queries, keys]``. M0 uses the final
    accepted query at layer 18; M1 takes the maximum across accepted-prefix
    queries at layer 18; M2/M3 pool the final layer-18 query over neighboring
    key positions; M5 uses the final query at layer 35.
    """

    method = str(method).upper()
    if method not in ATTENTION_METHODS:
        raise ValueError(f"method must be one of {ATTENTION_METHODS}, got {method!r}")
    if layer18_attention.ndim != 3 or layer35_attention.ndim != 3:
        raise ValueError("attention tensors must have shape [batch, queries, keys]")
    if layer18_attention.shape[0] != layer35_attention.shape[0]:
        raise ValueError("layer 18 and layer 35 attention batch sizes must match")
    if layer18_attention.shape[-1] != layer35_attention.shape[-1]:
        raise ValueError("layer 18 and layer 35 attention key lengths must match")

    if method == "M1":
        return layer18_attention.max(dim=1).values
    if method in {"M0", "M2", "M3"}:
        scores = layer18_attention[:, -1, :]
    else:
        return layer35_attention[:, -1, :]

    if method in {"M2", "M3"}:
        if pool_kernel <= 0 or pool_kernel % 2 == 0:
            raise ValueError("pool_kernel must be a positive odd integer")
        values = scores.unsqueeze(1)
        padding = pool_kernel // 2
        if method == "M2":
            scores = F.avg_pool1d(
                values,
                kernel_size=pool_kernel,
                stride=1,
                padding=padding,
                count_include_pad=True,
            ).squeeze(1)
        else:
            scores = F.max_pool1d(
                values,
                kernel_size=pool_kernel,
                stride=1,
                padding=padding,
            ).squeeze(1)
    return scores


def select_top_context_indices(scores: torch.Tensor, budget: int) -> torch.Tensor:
    """Select up to ``budget`` highest-scoring positions in source order."""

    if scores.ndim != 1:
        raise ValueError("scores must be a one-dimensional context vector")
    if budget <= 0:
        raise ValueError("budget must be positive")
    if scores.numel() == 0:
        raise ValueError("scores cannot be empty")
    count = min(int(budget), int(scores.numel()))
    if count == scores.numel():
        return torch.arange(scores.numel(), device=scores.device)
    ranked = torch.argsort(scores, descending=True, stable=True)[:count]
    return ranked.sort().values
