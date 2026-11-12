"""Unit tests for Visual Recall and Proposal Recall analysis.

Tests:
1. Attention Mass Pareto Coverage curve calculation.
2. Proposal Recall@K calculation from draft rankings/logits.
3. Integration on synthetic distribution replay position records.
"""

import pytest
import numpy as np
from src.analyze.Validate_Sparrow_hypothesises.analyze_visual_recall import (
    compute_attention_pareto_coverage,
    compute_proposal_recall_at_k,
    summarize_position_recalls,
)


def test_attention_pareto_coverage():
    # Synthetic attention: 10 visual tokens, highly skewed
    # Token 0 has 50% attention, Token 1 has 30%, others share 20%
    scores = np.array([0.5, 0.3, 0.05, 0.05, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01])
    
    # 20% top tokens (2 tokens) should cover 0.5 + 0.3 = 0.8 (80%) of mass
    coverage = compute_attention_pareto_coverage(scores, percentiles=[0.1, 0.2, 0.5, 1.0])
    
    assert 0.1 in coverage
    assert 0.2 in coverage
    assert np.isclose(coverage[0.1], 0.5, atol=1e-3)
    assert np.isclose(coverage[0.2], 0.8, atol=1e-3)
    assert np.isclose(coverage[1.0], 1.0, atol=1e-3)


def test_proposal_recall_at_k():
    # target ranks: 1-indexed (1 is top-1 hit)
    # 4 positions:
    # Pos 0: rank 1 (top 1) -> in top 1, 2, 3, 5
    # Pos 1: rank 2 (top 2) -> in top 2, 3, 5
    # Pos 2: rank 4 (top 4) -> in top 5
    # Pos 3: rank 11 (top 11) -> in none
    ranks = [1, 2, 4, 11]
    
    recalls = compute_proposal_recall_at_k(ranks, k_values=[1, 2, 3, 5, 10], one_indexed=True)
    
    assert np.isclose(recalls[1], 1 / 4)
    assert np.isclose(recalls[2], 2 / 4)
    assert np.isclose(recalls[3], 2 / 4)
    assert np.isclose(recalls[5], 3 / 4)
    assert np.isclose(recalls[10], 3 / 4)


def test_summarize_position_recalls():
    # Mock records mimicking distribution replay format (1-indexed ranks)
    records = [
        {
            "draft_slot_in_block": 1,
            "target_rank": 1,
            "target_hit_top_1": True,
            "target_hit_top_2": True,
            "target_hit_top_5": True,
            "budget": 25000,
            "condition": "full",
        },
        {
            "draft_slot_in_block": 1,
            "target_rank": 3,
            "target_hit_top_1": False,
            "target_hit_top_2": False,
            "target_hit_top_5": True,
            "budget": 25000,
            "condition": "full",
        },
        {
            "draft_slot_in_block": 2,
            "target_rank": 2,
            "target_hit_top_1": False,
            "target_hit_top_2": True,
            "target_hit_top_5": True,
            "budget": 25000,
            "condition": "full",
        },
    ]
    
    summary = summarize_position_recalls(records, k_values=[1, 2, 5], one_indexed=True)
    
    assert "by_slot" in summary
    assert 1 in summary["by_slot"]
    slot1 = summary["by_slot"][1]
    assert np.isclose(slot1["recall_at_1"], 0.5)
    assert np.isclose(slot1["recall_at_2"], 0.5)
    assert np.isclose(slot1["recall_at_5"], 1.0)
