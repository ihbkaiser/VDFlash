# coding=utf-8
"""Tests for DFlash 10% Visual Self-Distillation implementation.

Covers:
1. Visual context selection and retention mask (prefix vs temporal_spatial).
2. The 3 supervision signals (Target CE, Teacher KL, Target Gating).
3. All 5 training recipes (dual, target_ce, teacher_kl, gated, gated_mean_scaled).
4. Gradient isolation (student learns, teacher/head/embed stay frozen).
5. Comprehensive telemetry logging and ratio metric conservation.
6. Checkpointed chunk reduce invariance.
"""

import math
import sys
from pathlib import Path
from typing import Optional
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
SPECFORGE_PATH = REPO / "src" / "train_Dflash_SpecForge"
if str(SPECFORGE_PATH) not in sys.path:
    sys.path.insert(0, str(SPECFORGE_PATH))

from specforge.algorithms.common.dflash_self_distill import (
    OnlineDFlashSelfDistillModel,
    build_visual_context_keep_mask,
    _select_temporal_spatial,
)


# --- Helper Stubs for CPU Unit Testing ---

class _LightweightDraftStub(nn.Module):
    """Simple linear projection draft stub for unit testing without full VLM weights."""
    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.sliding_window = None

    def forward(self, position_ids, noise_embedding, target_hidden, attention_mask):
        bsz, draft_len, h_dim = noise_embedding.shape
        return self.proj(noise_embedding)


def _build_test_model(
    recipe: str = "dual",
    retention: float = 0.1,
    selector: str = "temporal_spatial",
    kd_weight: float = 0.5,
    kd_temperature: float = 1.0,
    hidden_size: int = 64,
    vocab_size: int = 100,
    block_size: int = 4,
    num_anchors: int = 8,
    objective_chunk_blocks: int = 4,
):
    student_draft = _LightweightDraftStub(hidden_size)
    teacher_draft = _LightweightDraftStub(hidden_size)
    lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    embed_tokens = nn.Embedding(vocab_size, hidden_size)
    mask_token_id = 99
    visual_token_ids = [50, 51]

    model = OnlineDFlashSelfDistillModel(
        draft_model=student_draft,
        teacher_draft_model=teacher_draft,
        target_lm_head=lm_head,
        target_embed_tokens=embed_tokens,
        mask_token_id=mask_token_id,
        block_size=block_size,
        attention_backend="eager",
        num_anchors=num_anchors,
        objective_chunk_blocks=objective_chunk_blocks,
        loss_decay_gamma=None,
        visual_token_ids=visual_token_ids,
        visual_retention=retention,
        visual_selector=selector,
        recipe=recipe,
        kd_weight=kd_weight,
        kd_temperature=kd_temperature,
    )
    return model


# --- 1. Visual Context Selection Tests ---

def test_keep_mask_exact_retention_prefix():
    # 2 batches, seq_len = 20. 10 visual tokens (id=50) and 10 text tokens (id=10)
    input_ids = torch.tensor([
        [10] * 5 + [50] * 10 + [10] * 5,
        [50] * 12 + [10] * 8,
    ])
    retention = 0.1  # ceil(10 * 0.1) = 1, ceil(12 * 0.1) = 2
    mask = build_visual_context_keep_mask(
        input_ids,
        position_ids=None,
        visual_token_ids=[50],
        retention=retention,
        selector="prefix",
    )

    # Text tokens must always be kept (True)
    assert mask[0, :5].all()
    assert mask[0, 15:].all()
    assert mask[1, 12:].all()

    # Visual tokens retained count
    assert (mask[0] & (input_ids[0] == 50)).sum().item() == 1
    assert (mask[1] & (input_ids[1] == 50)).sum().item() == 2

    # Prefix selector keeps the earliest visual positions
    assert mask[0, 5].item() is True
    assert not mask[0, 6:15].any()


def test_keep_mask_temporal_spatial_selection():
    # 1 sequence with 12 visual tokens across 3 temporal frames (4 tokens per frame: 2x2 grid)
    # Temporal IDs: frame 0, 1, 2
    # Spatial: (H, W) in { (0,0), (0,1), (1,0), (1,1) }
    T = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    H = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
    W = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    input_ids = torch.full((1, 12), 50, dtype=torch.long)
    position_ids = torch.stack([
        torch.tensor([T]),
        torch.tensor([H]),
        torch.tensor([W]),
    ], dim=0)  # Shape [3, 1, 12]

    # Retention = 0.25 -> keep ceil(12 * 0.25) = 3 tokens (1 token per frame)
    mask = build_visual_context_keep_mask(
        input_ids,
        position_ids=position_ids,
        visual_token_ids=[50],
        retention=0.25,
        selector="temporal_spatial",
    )
    kept_indices = torch.nonzero(mask[0], as_tuple=False).flatten().tolist()
    assert len(kept_indices) == 3

    # Verify that exactly 1 token is selected from each of the 3 frames
    kept_t = [T[idx] for idx in kept_indices]
    assert sorted(kept_t) == [0, 1, 2]


def test_keep_mask_pure_text_invariance():
    # No visual tokens
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    mask = build_visual_context_keep_mask(
        input_ids,
        position_ids=None,
        visual_token_ids=[50],
        retention=0.1,
        selector="prefix",
    )
    assert mask.all()


def test_keep_mask_validation_errors():
    input_ids = torch.tensor([[10, 50, 10]])
    with pytest.raises(ValueError, match="visual retention must be in"):
        build_visual_context_keep_mask(input_ids, None, visual_token_ids=[50], retention=0.0, selector="prefix")
    with pytest.raises(ValueError, match="unknown visual selector"):
        build_visual_context_keep_mask(input_ids, None, visual_token_ids=[50], retention=0.1, selector="invalid")
    with pytest.raises(ValueError, match="temporal_spatial selection requires Qwen M-RoPE"):
        build_visual_context_keep_mask(input_ids, None, visual_token_ids=[50], retention=0.1, selector="temporal_spatial")


# --- 2. Model Initialization & Freezing Tests ---

def test_model_freezing():
    model = _build_test_model()
    # Check that teacher, LM head, and embed tokens are strictly frozen
    for p in model.teacher_draft_model.parameters():
        assert not p.requires_grad
    for p in model.lm_head.parameters():
        assert not p.requires_grad
    for p in model.embed_tokens.parameters():
        assert not p.requires_grad

    # Student draft model MUST have trainable parameters
    student_trainable = [p for p in model.draft_model.parameters() if p.requires_grad]
    assert len(student_trainable) > 0


# --- 3. The 3 Supervision Signals & Recipes ---

def _prepare_dummy_batch(seq_len: int = 16, bsz: int = 1, hidden_size: int = 64):
    input_ids = torch.randint(1, 40, (bsz, seq_len))
    # Add visual tokens in the middle
    input_ids[:, 4:10] = 50
    # Loss mask: supervise tokens from pos 8 onwards
    loss_mask = torch.zeros((bsz, seq_len), dtype=torch.float32)
    loss_mask[:, 8:] = 1.0
    hidden_states = torch.randn(bsz, seq_len, hidden_size)
    position_ids = torch.stack([
        torch.arange(seq_len).unsqueeze(0).expand(bsz, -1),
        torch.zeros(bsz, seq_len, dtype=torch.long),
        torch.zeros(bsz, seq_len, dtype=torch.long),
    ], dim=0)
    return input_ids, hidden_states, loss_mask, position_ids


def test_recipe_target_ce():
    model = _build_test_model(recipe="target_ce")
    input_ids, hidden_states, loss_mask, position_ids = _prepare_dummy_batch()
    loss, acc, metrics = model(input_ids, hidden_states, loss_mask, position_ids)

    assert loss.item() > 0
    assert "ce_loss" in metrics["ratio_metrics"]
    # In target_ce, teacher is not even run for efficiency, so no kd_loss in ratio_metrics
    assert "kd_loss" not in metrics["ratio_metrics"]


def test_recipe_teacher_kl():
    kd_weight = 0.8
    model = _build_test_model(recipe="teacher_kl", kd_weight=kd_weight)
    input_ids, hidden_states, loss_mask, position_ids = _prepare_dummy_batch()
    loss, acc, metrics = model(input_ids, hidden_states, loss_mask, position_ids)

    kd_num, denom = metrics["ratio_metrics"]["kd_loss"]
    expected_loss = kd_weight * (kd_num / denom)
    assert torch.isclose(loss, expected_loss, atol=1e-5)


def test_recipe_dual():
    kd_weight = 0.5
    model = _build_test_model(recipe="dual", kd_weight=kd_weight)
    input_ids, hidden_states, loss_mask, position_ids = _prepare_dummy_batch()
    loss, acc, metrics = model(input_ids, hidden_states, loss_mask, position_ids)

    ce_num, denom = metrics["ratio_metrics"]["ce_loss"]
    kd_num, _ = metrics["ratio_metrics"]["kd_loss"]
    expected_loss = (ce_num + kd_weight * kd_num) / denom
    assert torch.isclose(loss, expected_loss, atol=1e-5)


def test_recipe_gated():
    kd_weight = 0.6
    model = _build_test_model(recipe="gated", kd_weight=kd_weight)
    input_ids, hidden_states, loss_mask, position_ids = _prepare_dummy_batch()
    loss, acc, metrics = model(input_ids, hidden_states, loss_mask, position_ids)

    ce_num, denom = metrics["ratio_metrics"]["ce_loss"]
    gated_kd_num, _ = metrics["ratio_metrics"]["gated_kd_loss"]
    expected_loss = (ce_num + kd_weight * gated_kd_num) / denom
    assert torch.isclose(loss, expected_loss, atol=1e-5)

    # Verify that gated_kd is less than or equal to un-gated kd
    kd_num, _ = metrics["ratio_metrics"]["kd_loss"]
    assert gated_kd_num.item() <= kd_num.item() + 1e-5


def test_recipe_gated_mean_scaled():
    kd_weight = 0.5
    model = _build_test_model(recipe="gated_mean_scaled", kd_weight=kd_weight)
    input_ids, hidden_states, loss_mask, position_ids = _prepare_dummy_batch()
    loss, acc, metrics = model(input_ids, hidden_states, loss_mask, position_ids)

    ce_num, denom = metrics["ratio_metrics"]["ce_loss"]
    kd_num, _ = metrics["ratio_metrics"]["kd_loss"]
    gate_num, _ = metrics["ratio_metrics"]["gate_coverage"]
    gate_cov = gate_num / denom
    expected_loss = (ce_num + kd_weight * gate_cov * kd_num) / denom
    assert torch.isclose(loss, expected_loss, atol=1e-5)


# --- 4. Gradient Isolation Tests ---

def test_gradient_isolation():
    model = _build_test_model(recipe="dual")
    input_ids, hidden_states, loss_mask, position_ids = _prepare_dummy_batch()
    loss, _, _ = model(input_ids, hidden_states, loss_mask, position_ids)
    loss.backward()

    # Student draft model MUST receive gradients
    for p in model.draft_model.parameters():
        assert p.grad is not None
        assert not torch.isnan(p.grad).any()

    # Teacher draft model, LM head, and embed tokens MUST NOT have gradients
    for p in model.teacher_draft_model.parameters():
        assert p.grad is None
    for p in model.lm_head.parameters():
        assert p.grad is None
    for p in model.embed_tokens.parameters():
        assert p.grad is None


# --- 5. Telemetry, Conservation & Ratio Metrics ---

def test_telemetry_metrics_and_conservation():
    model = _build_test_model(recipe="dual")
    input_ids, hidden_states, loss_mask, position_ids = _prepare_dummy_batch()
    _, _, metrics = model(input_ids, hidden_states, loss_mask, position_ids)

    ratio_metrics = metrics["ratio_metrics"]
    expected_keys = {
        "acc", "student_acc", "teacher_acc",
        "both_correct", "teacher_only_correct", "student_only_correct", "both_wrong",
        "gate_coverage", "ce_loss", "kd_loss", "gated_kd_loss",
    }
    for key in expected_keys:
        assert key in ratio_metrics, f"Missing telemetry key: {key}"

    # Check partition of 4 mutually exclusive proposal states:
    both_correct, denom = ratio_metrics["both_correct"]
    teacher_only, _ = ratio_metrics["teacher_only_correct"]
    student_only, _ = ratio_metrics["student_only_correct"]
    both_wrong, _ = ratio_metrics["both_wrong"]

    # Sum of the 4 mutually exclusive states must equal the total supervised denominator
    state_sum = both_correct + teacher_only + student_only + both_wrong
    assert torch.isclose(state_sum, denom, atol=1e-5)

    # Teacher accuracy conservation
    teacher_correct, _ = ratio_metrics["teacher_acc"]
    assert torch.isclose(both_correct + teacher_only, teacher_correct, atol=1e-5)

    # Student accuracy conservation
    student_correct, _ = ratio_metrics["student_acc"]
    assert torch.isclose(both_correct + student_only, student_correct, atol=1e-5)

    # Gate coverage equals teacher accuracy numerator
    gate_num, _ = ratio_metrics["gate_coverage"]
    assert torch.isclose(gate_num, teacher_correct, atol=1e-5)


# --- 6. Chunking Invariance ---

def test_objective_chunk_reduce_invariance():
    # Comparing objective_chunk_blocks = 1 vs 4
    torch.manual_seed(42)
    model_chunk1 = _build_test_model(recipe="dual", objective_chunk_blocks=1)
    torch.manual_seed(42)
    model_chunk4 = _build_test_model(recipe="dual", objective_chunk_blocks=4)

    # Copy weights
    model_chunk4.load_state_dict(model_chunk1.state_dict())

    torch.manual_seed(123)
    input_ids, hidden_states, loss_mask, position_ids = _prepare_dummy_batch()

    loss1, _, metrics1 = model_chunk1(input_ids, hidden_states, loss_mask, position_ids)
    loss4, _, metrics4 = model_chunk4(input_ids, hidden_states, loss_mask, position_ids)

    assert torch.isclose(loss1, loss4, atol=1e-5)
    for k in metrics1["ratio_metrics"]:
        val1 = metrics1["ratio_metrics"][k][0]
        val4 = metrics4["ratio_metrics"][k][0]
        assert torch.isclose(val1, val4, atol=1e-5)
