"""Regression tests for fixed-trajectory paired DFlash distribution replay."""

from __future__ import annotations

import torch

import src.infer.qwen25vl_dflash_compare as compare


def test_distribution_replay_keeps_mask_queries_and_uses_fixed_target_tokens():
    prepare = getattr(compare, "_prepare_distribution_replay_inputs", None)
    assert callable(prepare), "fixed-trajectory replay input helper is missing"

    mask_token_id = 99
    draft_query_ids = torch.tensor([[10, mask_token_id, mask_token_id, mask_token_id]])
    full_proposal_ids = torch.tensor([[20, 21, 22]])

    verification_ids, accepted, valid_proposals = prepare(
        draft_query_ids,
        full_proposal_ids,
        [10, 20, 91, 92],
        answer_offset=0,
        mask_token_id=mask_token_id,
    )

    assert draft_query_ids.tolist() == [[10, mask_token_id, mask_token_id, mask_token_id]]
    assert verification_ids.tolist() == [[10, 20, 91, 92]]
    assert (accepted, valid_proposals) == (1, 3)


def test_distribution_replay_handles_final_short_target_suffix():
    prepare = getattr(compare, "_prepare_distribution_replay_inputs", None)
    assert callable(prepare), "fixed-trajectory replay input helper is missing"

    mask_token_id = 99
    draft_query_ids = torch.tensor([[10, mask_token_id, mask_token_id, mask_token_id]])
    full_proposal_ids = torch.tensor([[20, 21, 22]])

    verification_ids, accepted, valid_proposals = prepare(
        draft_query_ids,
        full_proposal_ids,
        [10, 20],
        answer_offset=0,
        mask_token_id=mask_token_id,
    )

    assert verification_ids.tolist() == [[10, 20, mask_token_id, mask_token_id]]
    assert (accepted, valid_proposals) == (1, 1)
