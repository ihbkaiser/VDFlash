"""CPU tests for the MSD draft-attention probe (Figure 2, draft source)."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from src.analyze.Validate_Sparrow_hypothesises.run_draft_attention import _rows_for_policy


def _fake_captured() -> dict[int, torch.Tensor]:
    # 3 draft layers x [heads=4, query=1, key=10]
    key_len = 10
    result = {}
    for layer in range(3):
        weights = torch.zeros(4, 1, key_len)
        # Most mass on the instruction position 6; some on visual 2,3.
        weights[:, 0, 6] = 0.6
        weights[:, 0, 2] = 0.15
        weights[:, 0, 3] = 0.15
        weights[:, 0, 0] = 0.1
        result[layer] = weights
    return result


def _fake_args() -> SimpleNamespace:
    return SimpleNamespace(base_model="Qwen/Qwen2-VL-7B-Instruct", msd_model="lucylyn/MSD-Qwen2VL-7B-Instruct")


def _fake_prepared() -> SimpleNamespace:
    return SimpleNamespace(input_ids=torch.arange(10).unsqueeze(0))


def _fake_masks() -> dict:
    return {
        "visual_positions": [2, 3],
        "instruction_positions": [6],
        "text_positions": [0, 1, 4, 5, 7, 8, 9],
        "query_index": 6,
    }


def test_draft_rows_have_paper_schema_and_masses():
    rows = _rows_for_policy(
        sample=SimpleNamespace(sample_id="v_test"),
        args=_fake_args(),
        point={"target_visual_tokens": 3000, "status": "ok"},
        prepared=_fake_prepared(),
        masks=_fake_masks(),
        policy="last_instruction",
        query_positions=[6],
        captured=_fake_captured(),
        fps=8.0,
        max_pixels=None,
    )
    summaries = [row for row in rows if row["modality"] == "summary"]
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["paper_figure"] == "Figure 2"
    assert summary["attention_source"] == "msd_draft"
    assert summary["draft_model"] == "lucylyn/MSD-Qwen2VL-7B-Instruct"
    assert summary["attention_query"] == "last_instruction"
    assert summary["query_position"] == 6
    assert summary["layers"] == 3
    assert summary["heads"] == 4
    # Masses are near the synthetic distribution (0.6 instr + 0.3 visual + 0.1 text).
    assert abs(summary["instruction_mass"] - 0.6) < 1e-5
    assert abs(summary["visual_mass"] - 0.3) < 1e-5
    assert abs(summary["text_mass"] - 0.1) < 1e-5
    assert 0.0 <= summary["visual_entropy"] <= 1.0
    assert len(summary["layer_visual_masses"]) == 3
    assert len(summary["per_head_visual_mass"]) == 4
    # Rows: 2 visual + 1 instruction + 7 text + 1 summary.
    assert len(rows) == 11
    assert sum(1 for row in rows if row["modality"] == "visual") == 2
    # Every non-summary row has an attention weight and a row_id.
    for row in rows:
        if row["modality"] != "summary":
            assert isinstance(row["attention_weight"], float)
        assert row["row_id"].startswith("v_test:")
        assert row["target_visual_tokens"] == 3000
        assert row["actual_visual_tokens"] == 2


def test_draft_rows_all_text_policy_averages_query_rows():
    rows = _rows_for_policy(
        sample=SimpleNamespace(sample_id="v_test"),
        args=_fake_args(),
        point=None,
        prepared=_fake_prepared(),
        masks=_fake_masks(),
        policy="all_text",
        query_positions=[0, 1, 4, 5, 6, 7, 8, 9],
        captured=_fake_captured(),
        fps=8.0,
        max_pixels=None,
    )
    summaries = [row for row in rows if row["modality"] == "summary"]
    assert len(summaries) == 1
    assert summaries[0]["attention_policy"] == "all_text"
    assert summaries[0]["query_position"] is None
    assert len(summaries[0]["query_positions"]) == 8


def test_draft_rows_reject_empty_capture():
    try:
        _rows_for_policy(
            sample=SimpleNamespace(sample_id="v_test"),
            args=_fake_args(),
            point=None,
            prepared=_fake_prepared(),
            masks=_fake_masks(),
            policy="last_instruction",
            query_positions=[6],
            captured={},
            fps=8.0,
            max_pixels=None,
        )
    except RuntimeError as exc:
        assert "no layers" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected RuntimeError for empty capture")
