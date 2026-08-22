from __future__ import annotations

from src.analyze.Validate_Sparrow_hypothesises.dflash_contract import (
    DFLASH_LENGTH_TARGETS,
    DFLASH_RETENTION_PERCENTAGES,
    DFlashExperiment,
    DFlashSemanticStatus,
    make_dflash_metadata,
    validate_dflash_row,
)
from src.analyze.Validate_Sparrow_hypothesises.run_dflash_length import (
    run_hidden_visual_retention,
    run_length_sweep,
)
from src.analyze.Validate_Sparrow_hypothesises.run_dflash_attention import (
    run_dflash_context_attention,
)
from src.analyze.Validate_Sparrow_hypothesises.run_dflash_layers import (
    eager_target_attention,
    run_qwen25vl_layer_diagnostics,
)


METADATA = make_dflash_metadata(
    target_model="Qwen/Qwen2.5-VL-3B-Instruct",
    draft_checkpoint="checkpoint/training_state.pt",
    draft_config="draft.json",
    experiment=DFlashExperiment.LENGTH_SWEEP,
    semantic_status=DFlashSemanticStatus.DIRECT,
)


def test_length_sweep_emits_one_valid_direct_row_per_milestone():
    samples = [{"id": "sample-0", "input_fingerprint": "input-0"}]

    def decode(sample, condition):
        return {
            "status": "ok",
            "input_fingerprint": sample["input_fingerprint"],
            "target_output_ids": [1, 2],
            "speculative_output_ids": [1, 2],
            "metrics": {"length_target": condition["length_target"]},
        }

    rows = run_length_sweep(
        samples,
        decode,
        metadata=METADATA,
        length_targets=(400, 3000),
    )

    assert [row["length_target"] for row in rows] == [400, 3000]
    assert [row["target_visual_tokens"] for row in rows] == [400, 3000]
    assert all(validate_dflash_row(row) == [] for row in rows)
    assert all(row["semantic_status"] == "direct" for row in rows)


def test_length_sweep_persists_rows_and_cleans_up_after_each_condition():
    samples = [{"id": "sample-0", "input_fingerprint": "input-0"}]
    persisted = []
    cleanups = []

    def decode(sample, condition):
        return {
            "status": "ok",
            "input_fingerprint": sample["input_fingerprint"],
            "target_output_ids": [condition["length_target"]],
            "speculative_output_ids": [condition["length_target"]],
            "metrics": {},
        }

    rows = run_length_sweep(
        samples,
        decode,
        metadata=METADATA,
        length_targets=(400, 3000),
        row_sink=persisted.append,
        cleanup=lambda: cleanups.append("released"),
    )

    assert persisted == rows
    assert cleanups == ["released", "released"]


def test_retention_keeps_full_target_fingerprint_and_records_masked_condition():
    samples = [
        {
            "id": "sample-0",
            "input_fingerprint": "conditioned-0",
            "full_target_input_fingerprint": "full-target-0",
            "context_length": 6,
            "visual_positions": [1, 2, 3, 4],
        }
    ]

    def decode(sample, condition):
        assert condition["hidden_context_mask"] == [False, False, True, True, True, False]
        return {
            "status": "ok",
            "input_fingerprint": sample["input_fingerprint"],
            "target_input_fingerprint": sample["full_target_input_fingerprint"],
            "target_output_ids": [1],
            "speculative_output_ids": [1],
            "metrics": {},
        }

    rows = run_hidden_visual_retention(
        samples,
        decode,
        metadata=METADATA,
        retention_percentages=(25,),
    )

    assert rows[0]["full_target_input_fingerprint"] == "full-target-0"
    assert rows[0]["retention_percentage"] == 25
    assert rows[0]["target_visual_tokens"] == 4
    assert rows[0]["hidden_context_mask"] == [False, False, True, True, True, False]
    assert validate_dflash_row(rows[0]) == []


def test_length_and_retention_defaults_match_full_grid():
    assert DFLASH_LENGTH_TARGETS == (400, 3000, 13000, 25000)
    assert DFLASH_RETENTION_PERCENTAGES == (100, 25, 10, 5, 1, 0)


def test_dflash_attention_rows_have_dflash_specific_query_semantics():
    samples = [{"id": "sample-0", "input_fingerprint": "input-0"}]

    def probe(sample):
        del sample
        return [{
            "layer_index": 0,
            "context_length": 4,
            "query_length": 2,
            "attention_source": "dflash_context",
            "metrics": {"context_attention_mass": 0.75},
        }]

    rows = run_dflash_context_attention(samples, probe, metadata=METADATA)

    assert len(rows) == 1
    assert rows[0]["semantic_status"] == "adapted"
    assert rows[0]["query_policy"] == "draft_block"
    assert rows[0]["attention_source"] == "dflash_context"
    assert rows[0]["attention_source"] != "msd_draft"
    assert validate_dflash_row(rows[0]) == []


def test_target_layer_diagnostics_are_not_decode_rows():
    samples = [{"id": "sample-0", "input_fingerprint": "input-0"}]

    def probe(sample):
        del sample
        return [
            {
                "experiment": "qwen25vl_target_hidden_cosine",
                "layer_index": 4,
                "metrics": {"cosine": 0.5},
            }
        ]

    rows = run_qwen25vl_layer_diagnostics(samples, probe, metadata=METADATA)

    assert rows[0]["semantic_status"] == "target_side_diagnostic"
    assert rows[0]["target_output_ids"] is None
    assert rows[0]["speculative_output_ids"] is None
    assert validate_dflash_row(rows[0]) == []


def test_target_layer_context_forces_eager_attention_and_restores_backend():
    class Target:
        class Config:
            _attn_implementation = "sdpa"

        config = Config()

    target = Target()
    with eager_target_attention(target):
        assert target.config._attn_implementation == "eager"
    assert target.config._attn_implementation == "sdpa"
