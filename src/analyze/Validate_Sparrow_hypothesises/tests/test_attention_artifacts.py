from __future__ import annotations

import json

import torch
from torch import nn

from src.analyze.Validate_Sparrow_hypothesises.attention_artifacts import (
    AttentionShardWriter,
    load_attention_shard,
)
from src.analyze.Validate_Sparrow_hypothesises.analyze_dflash_visualcut_attention import _valid_row
from src.analyze.Validate_Sparrow_hypothesises.validate_dflash_visualcut_pilot import validate
from src.infer.qwen25vl_dflash_compare import capture_dflash_attention


class _Attention(nn.Module):
    def forward(self, value):
        weights = torch.arange(30, dtype=torch.float32).reshape(1, 2, 3, 5)
        return value, weights


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()


class _Draft(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(), _Layer()])


def _record(value: float = 1.0):
    return {
        "layer_index": 0,
        "forward_index": 0,
        "key_length": 5,
        "weights": torch.full((1, 2, 3, 5), value, dtype=torch.bfloat16),
        "key_map": {
            "source_ids": torch.zeros(5, dtype=torch.uint8),
            "positions": torch.arange(5, dtype=torch.int32),
            "token_ids": torch.arange(5, dtype=torch.int32),
            "is_cached": torch.zeros(5, dtype=torch.uint8),
        },
    }


def test_attention_shard_writer_round_trips_bfloat16(tmp_path):
    writer = AttentionShardWriter(tmp_path, max_buffer_bytes=1)
    writer.begin_case("sample:400:full")
    writer.append(_record(0.5))
    case = writer.finish_case()

    assert case["record_count"] == 1
    assert len(case["shards"]) == 1
    loaded = load_attention_shard(case["shards"][0]["path"])
    assert loaded[0]["case_id"] == "sample:400:full"
    assert loaded[0]["weights"].dtype == torch.bfloat16
    assert torch.equal(loaded[0]["weights"], torch.full((1, 2, 3, 5), 0.5, dtype=torch.bfloat16))
    assert loaded[0]["key_map"]["token_ids"].tolist() == [0, 1, 2, 3, 4]


def test_attention_shard_writer_resume_does_not_overwrite_existing_shards(tmp_path):
    first_writer = AttentionShardWriter(tmp_path, max_buffer_bytes=1)
    first_writer.begin_case("sample-1:full")
    first_writer.append(_record(0.25))
    first_shards = first_writer.finish_case()["shards"]
    first_path = first_shards[0]["path"]
    first_hash = first_shards[0]["sha256"]

    resumed_writer = AttentionShardWriter(tmp_path, max_buffer_bytes=1)
    resumed_writer.begin_case("sample-2:full")
    resumed_writer.append(_record(0.75))
    resumed_shards = resumed_writer.finish_case()["shards"]

    assert resumed_shards[0]["path"] != first_path
    assert resumed_shards[0]["record_count"] == 1
    assert first_shards[0]["sha256"] == first_hash
    assert load_attention_shard(first_path)[0]["case_id"] == "sample-1:full"
    assert load_attention_shard(resumed_shards[0]["path"])[0]["case_id"] == "sample-2:full"


def test_attention_shard_writer_can_abort_a_case_and_continue(tmp_path):
    writer = AttentionShardWriter(tmp_path, max_buffer_bytes=1024 * 1024)
    writer.begin_case("sample-1:full")
    writer.append(_record())

    aborted = writer.abort_case()

    assert aborted["record_count"] == 0
    assert aborted["discarded_record_count"] == 1
    assert writer.active_case_id is None
    writer.begin_case("sample-2:full")
    writer.abort_case()


def test_capture_can_stream_every_forward_without_retaining_weights():
    draft = _Draft()
    records = []
    metadata = {
        "round_index": 3,
        "key_map": {"token_ids": torch.arange(5, dtype=torch.int32)},
    }
    for layer in draft.layers:
        layer.self_attn._dflash_capture_metadata = metadata

    with capture_dflash_attention(
        draft,
        record_sink=records.append,
        retain_records=False,
    ) as retained:
        for layer in draft.layers:
            layer.self_attn(torch.zeros(1))
            layer.self_attn(torch.zeros(1))

    assert retained == []
    assert [record["forward_index"] for record in records] == [0, 1, 0, 1]
    assert [record["round_index"] for record in records] == [3, 3, 3, 3]
    assert all(record["key_length"] == 5 for record in records)
    assert all(not hasattr(layer.self_attn, "_dflash_capture_metadata") for layer in draft.layers)


def test_attention_key_map_mismatch_fails_closed():
    draft = _Draft()
    draft.layers[0].self_attn._dflash_capture_metadata = {
        "key_map": {"token_ids": torch.arange(4, dtype=torch.int32)}
    }
    try:
        with capture_dflash_attention(draft):
            draft.layers[0].self_attn(torch.zeros(1))
    except RuntimeError as exc:
        assert "key map length does not match" in str(exc)
    else:  # pragma: no cover - assertion path
        raise AssertionError("expected a key map length error")


def test_pilot_validator_checks_all_conditions_and_raw_tensor_alignment(tmp_path):
    task_id = "dflash_3layer_26_28_30_32_34_v400"
    task_dir = tmp_path / "tasks" / task_id
    writer = AttentionShardWriter(task_dir / "attention", max_buffer_bytes=1024 * 1024)
    rows = []
    for condition in ("full", "zero", "cut", "retained_75", "retained_50", "retained_25"):
        row_id = f"checkpoint:sample-1:400:{condition}"
        writer.begin_case(row_id)
        writer.append(
            {
                "layer_index": 0,
                "forward_index": 0,
                "round_index": 0,
                "answer_position_start": 0,
                "query_length": 1,
                "key_length": 3,
                "weights": torch.tensor([[[[0.2, 0.3, 0.5]], [[0.1, 0.2, 0.7]]]]),
                "key_map": {
                    "source_ids": torch.tensor([0, 0, 1], dtype=torch.uint8),
                    "positions": torch.tensor([0, 1, 0], dtype=torch.int32),
                    "token_ids": torch.tensor([11, 12, 13], dtype=torch.int32),
                    "is_cached": torch.tensor([0, 0, 0], dtype=torch.uint8),
                },
            }
        )
        capture = writer.finish_case()
        rows.append(
            {
                "row_id": row_id,
                "sample_id": "sample-1",
                "visual_condition": condition,
                "status": "ok",
                "lossless": True,
                "capture_and_timing_outputs_match": True,
                "target_output_hash": "same-target-output",
                "attention_capture": capture,
            }
        )
    journal = task_dir / "cases.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    report = validate(tmp_path, task_id)

    assert report["passed"] is True
    assert report["artifact_integrity_passed"] is True
    assert report["lossless_passed"] is True
    assert report["condition_count"] == 6
    assert report["attention_record_count"] == 6


def test_pilot_artifact_validation_retains_nonlossless_rows(tmp_path):
    task_id = "dflash_3layer_26_28_30_32_34_v400"
    task_dir = tmp_path / "tasks" / task_id
    writer = AttentionShardWriter(task_dir / "attention", max_buffer_bytes=1024 * 1024)
    rows = []
    for condition in ("full", "zero", "cut", "retained_75", "retained_50", "retained_25"):
        row_id = f"checkpoint:sample-1:400:{condition}"
        writer.begin_case(row_id)
        record = _record()
        record.update({"round_index": 0, "answer_position_start": 0, "query_length": 3})
        writer.append(record)
        capture = writer.finish_case()
        rows.append(
            {
                "row_id": row_id,
                "sample_id": "sample-1",
                "visual_condition": condition,
                "status": "mismatch",
                "lossless": False,
                "capture_and_timing_outputs_match": True,
                "target_output_hash": "same-target-output",
                "attention_capture": capture,
            }
        )
    journal = task_dir / "cases.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    report = validate(tmp_path, task_id)

    assert report["passed"] is True
    assert report["artifact_integrity_passed"] is True
    assert report["lossless_passed"] is False
    assert len(report["lossless_failures"]) == 6
    assert _valid_row(rows[0]) is True
