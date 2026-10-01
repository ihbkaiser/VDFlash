from __future__ import annotations

import pytest
import torch

from specforge.modeling.draft.msd import MSDConfig, MSDDraftModel
from specforge.modeling.draft.registry import resolve_draft
from specforge.modeling.auto import AutoDraftModelConfig


def tiny_config(depth: int) -> MSDConfig:
    return MSDConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=depth,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        mrope_section=[1, 1, 2],
    )


@pytest.mark.parametrize("depth", [1, 3, 5])
def test_msd_supports_exact_depth_sweep(depth: int) -> None:
    model = MSDDraftModel(tiny_config(depth))

    assert len(model.layers) == depth
    assert model.config.architectures == ["MSDDraftModel"]
    assert not model.embed_tokens.weight.requires_grad


@pytest.mark.parametrize("depth", [0, 2, 4, 6])
def test_msd_rejects_depths_outside_replication_sweep(depth: int) -> None:
    with pytest.raises(ValueError, match="1, 3, or 5"):
        tiny_config(depth)


def test_msd_draft_is_registered_as_first_class_architecture() -> None:
    assert resolve_draft("MSDDraftModel") is MSDDraftModel
    assert MSDDraftModel.config_class is MSDConfig


def test_msd_prepare_inputs_decouples_visual_tokens() -> None:
    model = MSDDraftModel(tiny_config(1))
    model.fusion_projection.weight.data.zero_()
    model.fusion_projection.bias.data.fill_(2.0)
    hidden = torch.randn(1, 3, 32)
    next_embeddings = torch.randn(1, 3, 32)
    visual_embeddings = torch.randn(1, 3, 32)
    visual_mask = torch.tensor([[False, True, False]])

    actual = model.prepare_inputs(
        hidden,
        next_embeddings,
        visual_embeddings,
        visual_mask,
    )

    torch.testing.assert_close(actual[:, 0], torch.full((1, 32), 2.0))
    torch.testing.assert_close(actual[:, 1], visual_embeddings[:, 1])
    torch.testing.assert_close(actual[:, 2], torch.full((1, 32), 2.0))


def test_msd_forward_accepts_qwen25vl_three_axis_positions_and_backpropagates() -> None:
    model = MSDDraftModel(tiny_config(3))
    batch, length, hidden_size = 2, 5, 32
    conditioning = torch.randn(batch, length, hidden_size)
    next_embeddings = torch.randn(batch, length, hidden_size)
    visual_embeddings = torch.randn(batch, length, hidden_size)
    visual_mask = torch.tensor(
        [[False, True, True, False, False], [False, False, True, True, False]]
    )
    position_ids = torch.arange(length).repeat(3, batch, 1)
    position_ids[1, :, 1:3] += 3
    position_ids[2, :, 1:3] += 7

    output = model(
        conditioning_hidden_state=conditioning,
        next_token_embeddings=next_embeddings,
        visual_embeddings=visual_embeddings,
        visual_token_mask=visual_mask,
        attention_mask=torch.ones(batch, length, dtype=torch.bool),
        position_ids=position_ids,
    )

    assert output.shape == (batch, length, hidden_size)
    output.square().mean().backward()
    assert model.fusion_projection.weight.grad is not None
    assert model.layers[0].self_attn.q_proj.weight.grad is not None
    assert model.embed_tokens.weight.grad is None


def test_msd_config_matches_qwen25vl_3b_replication_shape() -> None:
    config = MSDConfig.qwen25vl_3b(num_hidden_layers=5)

    assert config.hidden_size == 2048
    assert config.intermediate_size == 11008
    assert config.num_attention_heads == 16
    assert config.num_key_value_heads == 2
    assert config.head_dim == 128
    assert config.mrope_section == [16, 24, 24]


def test_checked_in_msd_config_loads_through_specforge_registry() -> None:
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "configs" / "qwen2.5-vl-3b-msd.json"
    config = AutoDraftModelConfig.from_file(str(path))

    assert isinstance(config, MSDConfig)
    assert config.architectures == ["MSDDraftModel"]
    assert config.num_hidden_layers == 1
