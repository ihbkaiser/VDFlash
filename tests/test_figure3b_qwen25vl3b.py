import math
from types import SimpleNamespace

import pytest

from src.analyze.figure3b_qwen25vl3b import (
    aggregate_visual_attention_rows,
    build_attention_row,
    normalize_attention_matrix,
    resolve_model_source,
    validate_attention_rows,
)


def _row(sample_id, layer, heads, layer_count=3, **extra):
    row = {
        "sample_id": sample_id,
        "task": "action_prediction",
        "layer": layer,
        "layer_count": layer_count,
        "per_head_visual_mass": heads,
        "visual_mass": sum(heads),
        "condition": "visual_attention",
    }
    row.update(extra)
    return row


def test_validate_attention_rows_requires_zero_based_complete_layers():
    rows = [
        _row("s1", 0, [0.1, 0.2]),
        _row("s1", 1, [0.3, 0.4]),
        _row("s1", 2, [0.5, 0.6]),
    ]

    report = validate_attention_rows(rows, expected_samples=1, expected_layer_count=3)

    assert report["num_samples"] == 1
    assert report["layer_indices"] == [0, 1, 2]
    assert report["num_success_rows"] == 3

    with pytest.raises(ValueError, match="zero-based"):
        validate_attention_rows([_row("s1", 1, [0.1, 0.2])], expected_layer_count=3)


def test_normalize_attention_matrix_uses_one_global_scale_and_preserves_shape():
    matrix = [[2.0, 4.0], [1.0, 0.0]]

    normalized = normalize_attention_matrix(matrix)

    assert normalized == [[0.5, 1.0], [0.25, 0.0]]


def test_aggregate_visual_attention_rows_orders_heads_and_reports_layer_stats():
    rows = [
        _row("s1", 0, [1.0, 3.0], layer_count=2),
        _row("s1", 1, [2.0, 4.0], layer_count=2),
        _row("s2", 0, [3.0, 1.0], layer_count=2),
        _row("s2", 1, [4.0, 2.0], layer_count=2),
    ]

    report = aggregate_visual_attention_rows(rows)

    assert report["layer_indices"] == [0, 1]
    assert report["head_order"] == [0, 1]
    assert report["matrix_raw"] == [[2.0, 2.0], [3.0, 3.0]]
    assert report["matrix_normalized"] == [[2 / 3, 2 / 3], [1.0, 1.0]]
    assert report["layer_stats"][0]["mean_visual_mass"] == pytest.approx(4.0)
    assert report["layer_stats"][1]["mean_visual_mass"] == pytest.approx(6.0)
    assert report["layer_stats"][0]["std_visual_mass"] == pytest.approx(0.0)


def test_build_attention_row_is_explicit_about_native_zero_based_layer():
    row = build_attention_row(
        sample_id="s1",
        task="action_prediction",
        layer=0,
        layer_count=36,
        query_index=123,
        visual_positions=[10, 11],
        per_head_visual_mass=[0.25, 0.75],
        model="Qwen/Qwen2.5-VL-3B-Instruct",
    )

    assert row["layer"] == 0
    assert row["layer_index_convention"] == "zero_based_native_decoder_layer"
    assert row["visual_token_count"] == 2
    assert row["visual_mass"] == pytest.approx(1.0)
    assert row["per_head_visual_mass"] == [0.25, 0.75]


def test_capture_uses_positional_qwen_attention_arguments_and_returns_all_layers():
    torch = pytest.importorskip("torch")
    from src.analyze.figure3b_qwen25vl3b import _capture_qwen25_query_attention

    class Attention(torch.nn.Module):
        head_dim = 2
        num_key_value_groups = 1

        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(4, 4, bias=False)
            self.k_proj = torch.nn.Linear(4, 4, bias=False)

        def forward(self, hidden_states, position_embeddings=None, attention_mask=None, position_ids=None):
            return hidden_states

    layers = [SimpleNamespace(self_attn=Attention()) for _ in range(2)]
    model = SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(layers=layers)))
    hidden = torch.randn(1, 3, 4)

    with _capture_qwen25_query_attention(model, query_index=2) as captured:
        layers[0].self_attn(hidden, None, None, None)
        layers[1].self_attn(hidden, None, None, None)

    assert sorted(captured) == [0, 1]
    assert tuple(captured[0].shape) == (2, 3)
    assert torch.isfinite(captured[0]).all()
    assert torch.allclose(captured[0].sum(dim=-1), torch.ones(2), atol=1e-5)


def test_resolve_model_source_falls_back_to_default_huggingface_cache(tmp_path, monkeypatch):
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    monkeypatch.setenv("HF_HOME", str(tmp_path / "wrong-cache"))
    snapshot = (
        tmp_path
        / "home"
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--Qwen--Qwen2.5-VL-3B-Instruct"
        / "snapshots"
        / "abc123"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    assert resolve_model_source(model_id) == str(snapshot)
