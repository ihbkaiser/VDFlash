from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch


@dataclass
class SpeculativeStats:
    acceptance_lengths: list[int]
    proposed_tokens: int
    accepted_tokens: int

    @property
    def mean_acceptance_length(self) -> float:
        return sum(self.acceptance_lengths) / max(1, len(self.acceptance_lengths))


def _sample_logits(
    logits: torch.Tensor,
    temperature: float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if temperature <= 1e-5:
        return logits.argmax(dim=-1)
    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    flat = probabilities.reshape(-1, probabilities.shape[-1])
    return torch.multinomial(flat, 1, generator=generator).reshape(logits.shape[:-1])


def speculative_decode(
    prompt_ids: torch.Tensor,
    *,
    draft_propose: Callable[[torch.Tensor, int], torch.Tensor],
    target_verify: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    max_new_tokens: int,
    block_size: int,
    temperature: float = 0.0,
    stop_token_ids: Iterable[int] | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, SpeculativeStats]:
    """Lossless callback-based DFlash verification loop.

    ``draft_propose(prefix, n)`` returns ``n`` proposed token IDs.  The target
    callback receives the current accepted prefix and those proposals and must
    return logits of shape ``[n + 1, vocab]``: one target distribution for each
    proposal and one bonus distribution.  This separation lets the VLM adapter
    own multimodal KV caches while the verification logic remains testable.
    """

    if prompt_ids.ndim != 1:
        raise ValueError("prompt_ids must be one-dimensional")
    if block_size < 2:
        raise ValueError("block_size must be at least 2")
    stop = set(int(x) for x in (stop_token_ids or ()))
    output = prompt_ids.clone()
    acceptance_lengths: list[int] = []
    proposed_total = 0
    accepted_total = 0
    while output.numel() - prompt_ids.numel() < max_new_tokens:
        remaining = max_new_tokens - (output.numel() - prompt_ids.numel())
        proposal_count = min(block_size - 1, remaining)
        proposals = draft_propose(output, proposal_count).to(output.device).flatten()
        if proposals.numel() != proposal_count:
            raise ValueError("draft_propose returned the wrong number of tokens")
        posterior_logits = target_verify(output, proposals)
        if posterior_logits.ndim != 2 or posterior_logits.shape[0] != proposal_count + 1:
            raise ValueError("target_verify must return [proposal_count + 1, vocab] logits")
        posterior = _sample_logits(posterior_logits, temperature, generator=generator)
        matches = proposals.eq(posterior[:-1])
        accepted = 0
        while accepted < proposal_count and bool(matches[accepted]):
            accepted += 1
        bonus = posterior[accepted]
        emitted = torch.cat([proposals[:accepted], bonus.view(1)])
        stop_index = next((idx for idx, token in enumerate(emitted.tolist()) if int(token) in stop), None)
        if stop_index is not None:
            emitted = emitted[: stop_index + 1]
        output = torch.cat([output, emitted])
        accepted_total += emitted.numel()
        proposed_total += proposal_count
        acceptance_lengths.append(int(emitted.numel()))
        if stop_index is not None:
            break
    return output, SpeculativeStats(acceptance_lengths, proposed_total, accepted_total)
