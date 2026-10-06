"""Raw token/text trace helpers shared by Phase-1 experiment runners."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


def enrich_position_trace(
    trace: Sequence[dict[str, Any]],
    target_token_ids: Sequence[int],
    speculative_token_ids: Sequence[int],
    decode_token: Callable[[int], str],
) -> list[dict[str, Any]]:
    """Attach target/speculative token IDs and text without aggregating rows."""

    enriched: list[dict[str, Any]] = []
    for item in trace:
        row = dict(item)
        position = int(row["position"])
        target_id = target_token_ids[position] if position < len(target_token_ids) else None
        speculative_id = (
            speculative_token_ids[position] if position < len(speculative_token_ids) else None
        )
        row.update(
            {
                "target_token_id": int(target_id) if target_id is not None else None,
                "target_token_text": decode_token(int(target_id)) if target_id is not None else None,
                "speculative_token_id": int(speculative_id) if speculative_id is not None else None,
                "speculative_token_text": (
                    decode_token(int(speculative_id)) if speculative_id is not None else None
                ),
            }
        )
        enriched.append(row)
    return enriched


def enrich_round_trace(
    trace: Sequence[dict[str, Any]],
    target_token_ids: Sequence[int],
    speculative_token_ids: Sequence[int],
    decode_token: Callable[[int], str],
) -> dict[str, list[dict[str, Any]]]:
    """Expand MSD round records into position-level token/text records.

    MSD records acceptance by speculative-decoding round rather than by a
    single ``position`` key.  Preserve each round and add a lossless,
    position-indexed view so Phase 1 can later analyze early/late output
    positions without recomputing token alignment.
    """

    enriched_rounds: list[dict[str, Any]] = []
    positions: list[dict[str, Any]] = []
    for item in trace:
        row = dict(item)
        round_positions: list[dict[str, Any]] = []
        round_index = int(row["round_index"])
        accepted_positions = [int(value) for value in row.get("accepted_proposal_positions", [])]
        fallback_position = row.get("fallback_position")
        for position in accepted_positions:
            record = _round_position_record(
                position=position,
                round_index=round_index,
                role="accepted_proposal",
                proposed=1,
                accepted=1,
                target_token_ids=target_token_ids,
                speculative_token_ids=speculative_token_ids,
                decode_token=decode_token,
            )
            round_positions.append(record)
            positions.append(record)
        if fallback_position is not None:
            record = _round_position_record(
                position=int(fallback_position),
                round_index=round_index,
                role="fallback",
                proposed=0,
                accepted=0,
                target_token_ids=target_token_ids,
                speculative_token_ids=speculative_token_ids,
                decode_token=decode_token,
            )
            round_positions.append(record)
            positions.append(record)
        row["position_records"] = round_positions
        enriched_rounds.append(row)
    return {"rounds": enriched_rounds, "positions": positions}


def _round_position_record(
    *,
    position: int,
    round_index: int,
    role: str,
    proposed: int,
    accepted: int,
    target_token_ids: Sequence[int],
    speculative_token_ids: Sequence[int],
    decode_token: Callable[[int], str],
) -> dict[str, Any]:
    target_id = target_token_ids[position] if position < len(target_token_ids) else None
    speculative_id = (
        speculative_token_ids[position] if position < len(speculative_token_ids) else None
    )
    return {
        "position": int(position),
        "round_index": int(round_index),
        "role": role,
        "proposed": int(proposed),
        "accepted": int(accepted),
        "target_token_id": int(target_id) if target_id is not None else None,
        "target_token_text": decode_token(int(target_id)) if target_id is not None else None,
        "speculative_token_id": int(speculative_id) if speculative_id is not None else None,
        "speculative_token_text": (
            decode_token(int(speculative_id)) if speculative_id is not None else None
        ),
    }


def decode_token_ids(processor: Any, token_ids: Sequence[int]) -> str:
    """Decode one token sequence using a processor/tokenizer without aggregation."""

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = processor
    if hasattr(tokenizer, "decode"):
        return str(tokenizer.decode(list(token_ids), skip_special_tokens=True))
    if hasattr(processor, "batch_decode"):
        return str(processor.batch_decode([list(token_ids)], skip_special_tokens=True)[0])
    raise TypeError("processor does not expose decode or batch_decode")
