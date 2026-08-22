from __future__ import annotations

from src.analyze.Validate_Sparrow_hypothesises.dflash_contract import (
    DFLASH_RETENTION_PERCENTAGES,
    DFlashExperiment,
    DFlashSemanticStatus,
    validate_dflash_grid,
    validate_dflash_row,
)


def _row(**overrides):
    row = {
        "backend": "dflash",
        "experiment": DFlashExperiment.LENGTH_SWEEP.value,
        "semantic_status": DFlashSemanticStatus.DIRECT.value,
        "target_model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "draft_checkpoint": "dataset/qwen25vl-3b-dflash-llava68k-latest/training_state.pt",
        "draft_config": "src/train_Dflash_SpecForge/configs/qwen2.5-vl-3b-dflash.json",
        "sample_id": "sample-0",
        "input_fingerprint": "abc",
        "full_target_input_fingerprint": "abc",
        "target_input_fingerprint": "abc",
        "target_output_ids": [1, 2],
        "speculative_output_ids": [1, 2],
        "metrics": {},
    }
    row.update(overrides)
    return row


def test_valid_direct_length_row_has_no_contract_errors():
    assert validate_dflash_row(_row()) == []


def test_adapted_retention_row_requires_full_target_fingerprint():
    row = _row(
        experiment=DFlashExperiment.TARGET_HIDDEN_VISUAL_RETENTION.value,
        semantic_status=DFlashSemanticStatus.ADAPTED.value,
        condition="retention",
        retention_percentage=25,
    )
    assert validate_dflash_row(row) == []

    del row["full_target_input_fingerprint"]
    errors = validate_dflash_row(row)
    assert "full_target_input_fingerprint" in " ".join(errors)


def test_target_side_diagnostic_is_valid_without_decode_ids():
    row = _row(
        experiment=DFlashExperiment.TARGET_HIDDEN_COSINE.value,
        semantic_status=DFlashSemanticStatus.TARGET_SIDE_DIAGNOSTIC.value,
        target_output_ids=None,
        speculative_output_ids=None,
        layer_index=4,
    )
    assert validate_dflash_row(row) == []


def test_msd_attention_label_is_rejected():
    row = _row(
        experiment=DFlashExperiment.CONTEXT_ATTENTION.value,
        semantic_status=DFlashSemanticStatus.ADAPTED.value,
        attention_source="msd_draft",
        query_policy="draft_block",
    )
    errors = validate_dflash_row(row)
    assert any("msd_draft" in error for error in errors)


def test_length_sweep_requires_direct_semantics():
    errors = validate_dflash_row(
        _row(semantic_status=DFlashSemanticStatus.ADAPTED.value)
    )

    assert any("must be direct" in error for error in errors)


def test_grid_validator_reports_missing_retention_milestones():
    errors = validate_dflash_grid({"retention_percentages": [100, 0]})
    assert "25" in " ".join(errors)
    assert set(DFLASH_RETENTION_PERCENTAGES) == {100, 25, 10, 5, 1, 0}


def test_error_row_is_retained_without_decode_outputs():
    row = _row(status="unsupported", error="position budget unavailable")
    del row["target_output_ids"]
    del row["speculative_output_ids"]

    assert validate_dflash_row(row) == []


def test_retention_rejects_changed_target_verifier_input():
    row = _row(
        experiment=DFlashExperiment.TARGET_HIDDEN_VISUAL_RETENTION.value,
        semantic_status=DFlashSemanticStatus.ADAPTED.value,
        condition="retention",
        retention_percentage=25,
        full_target_input_fingerprint="full",
        target_input_fingerprint="changed",
    )

    errors = validate_dflash_row(row)

    assert any("target_input_fingerprint" in error for error in errors)
