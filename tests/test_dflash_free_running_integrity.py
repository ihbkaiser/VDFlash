import csv
import hashlib
import json
import math

import pytest
import torch

from src.analyze.Validate_Sparrow_hypothesises.analyze_dflash_free_running import (
    _coarse_distribution_js,
    analyze_free_running,
    summarize_acceptance,
    summarize_visual_sensitivity,
)
from src.analyze.Validate_Sparrow_hypothesises.free_running_experiment import (
    append_jsonl,
    build_matched_prefix_pairs,
    protocol_fingerprint,
    summarize_judge_pairs,
)
from src.analyze.Validate_Sparrow_hypothesises.free_running_experiment import (
    build_blind_pairs,
    build_visual_retention_mask,
    classify_token_surface,
    make_judge_prompt,
    read_jsonl,
    write_blinded_review_files,
    write_grounding_review_files,
)


def _rollout(retention, *, fingerprint="same-input", prediction=None):
    return {
        "row_id": f"v1:retention-{retention}",
        "sample_id": "v1",
        "retention_percentage": retention,
        "status": "ok",
        "prediction": prediction or f"answer at retention {retention}",
        "input_fingerprint": fingerprint,
        "initial_anchor": "last_prompt_token",
        "feature_refresh": "refresh",
        "max_new_tokens": 256,
        "target_model": "local-target",
        "checkpoint": {"sha256": "checkpoint"},
    }


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_free_running_pairs_require_identical_inputs_and_protocol():
    full = _rollout(100)
    cut = _rollout(0, fingerprint="changed-video")
    with pytest.raises(ValueError, match="input_fingerprint"):
        build_blind_pairs([full, cut], seed=7)

    cut = _rollout(0)
    cut["initial_anchor"] = "target_greedy_seed"
    with pytest.raises(ValueError, match="initial_anchor"):
        build_blind_pairs([full, cut], seed=7)


def test_acceptance_pairs_require_same_input_and_target_greedy_reference():
    rows = []
    for retention in (100, 0):
        rows.append({
            "sample_id": "v1",
            "retention_percentage": retention,
            "status": "ok",
            "outputs_match_target_greedy": True,
            "input_fingerprint": "same-input",
            "target_reference_token_sha256": f"reference-{retention}",
            "tau_effective": 3.0,
            "tau_proposal": 2.0,
        })
    with pytest.raises(ValueError, match="target_reference_token_sha256"):
        summarize_acceptance(rows, seed=7, bootstrap_replicates=100)

    rows[1]["target_reference_token_sha256"] = rows[0]["target_reference_token_sha256"]
    rows[1]["input_fingerprint"] = "different-input"
    with pytest.raises(ValueError, match="input_fingerprint"):
        summarize_acceptance(rows, seed=7, bootstrap_replicates=100)


def test_pair_identifier_changes_when_output_text_changes():
    pairs_a, key_a = build_blind_pairs(
        [_rollout(100), _rollout(0, prediction="old output")], seed=7
    )
    pairs_b, key_b = build_blind_pairs(
        [_rollout(100), _rollout(0, prediction="new output")], seed=7
    )
    assert pairs_a[0]["pair_id"] != pairs_b[0]["pair_id"]
    assert key_a != key_b


def test_judge_summary_rejects_score_for_different_pair_content():
    pairs, key = build_blind_pairs(
        [_rollout(100), _rollout(0, prediction="current text")], seed=7
    )
    pair = pairs[0]
    stale = {
        "pair_id": pair["pair_id"],
        "pair_fingerprint": "old-content",
        "retention_percentage": pair["retention_percentage"],
        "status": "judge_parse_error",
    }
    with pytest.raises(ValueError, match="stale or different text"):
        summarize_judge_pairs([stale], key, expected_pair_ids=[pair["pair_id"]])


@pytest.mark.parametrize("review_kind", ["coherence", "grounding"])
def test_resumed_review_csv_adds_new_pairs_and_keeps_existing_ratings(tmp_path, review_kind):
    def export(retentions):
        pairs, key = build_blind_pairs(
            [_rollout(value) for value in retentions], seed=7
        )
        if review_kind == "coherence":
            result = write_blinded_review_files(
                pairs, key, tmp_path, seed=7, sample_limit=10
            )
        else:
            result = write_grounding_review_files(
                pairs, key, {"v1": "question"}, tmp_path, seed=7, sample_limit=10
            )
        return pairs, result

    first_pairs, result = export([100, 0])
    first_review = _read_csv(result["blinded_review_csv"])
    first_review[0]["human_winner" if review_kind == "coherence" else "grounding_winner"] = "a"
    with open(result["blinded_review_csv"], "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=first_review[0].keys())
        writer.writeheader()
        writer.writerows(first_review)

    all_pairs, result = export([100, 75, 50, 25, 0])
    review_rows = _read_csv(result["blinded_review_csv"])
    winner_field = "human_winner" if review_kind == "coherence" else "grounding_winner"
    assert len(review_rows) == result["review_pairs"] == 4
    assert {row["review_id"] for row in review_rows} == {
        pair["pair_id"] for pair in all_pairs
    }
    assert any(row[winner_field] == "a" for row in review_rows)
    assert len(first_pairs) == 1


def test_jsonl_reader_discards_only_an_incomplete_final_record(tmp_path):
    path = tmp_path / "journal.jsonl"
    valid = _rollout(100)
    path.write_text(json.dumps(valid) + "\n{\"row_id\":\"truncated", encoding="utf-8")
    assert read_jsonl(path) == [valid]
    assert path.read_bytes().endswith(b"\n")

    path.write_text(json.dumps(valid) + "\nnot-json\n", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(path)


def test_jsonl_append_separates_a_complete_record_without_newline(tmp_path):
    path = tmp_path / "journal.jsonl"
    first = {"row_id": "first"}
    second = {"row_id": "second"}
    path.write_text(json.dumps(first), encoding="utf-8")
    append_jsonl(path, second)
    assert read_jsonl(path) == [first, second]


def test_top_k_js_pools_only_shared_token_ids_and_is_a_lower_bound():
    # Two 33-token distributions differ only by swapping close probabilities
    # at the top-32 boundary. The unshared token remains in each residual tail.
    full = {
        "draft_top_token_ids": list(range(31)) + [31],
        "draft_top_token_probabilities": [0.031] * 31 + [0.0196],
        "draft_tail_probability_mass": 0.0194,
    }
    cut = {
        "draft_top_token_ids": list(range(31)) + [32],
        "draft_top_token_probabilities": [0.031] * 31 + [0.0196],
        "draft_tail_probability_mass": 0.0194,
    }
    probabilities_full = [0.031] * 31 + [0.0196, 0.0194]
    probabilities_cut = [0.031] * 31 + [0.0194, 0.0196]
    exact_js = sum(
        0.5 * left * math.log(left / (0.5 * (left + right)))
        + 0.5 * right * math.log(right / (0.5 * (left + right)))
        for left, right in zip(probabilities_full, probabilities_cut)
    )

    estimated_js = _coarse_distribution_js(full, cut)
    assert estimated_js is not None
    assert estimated_js <= exact_js + 1e-12


def test_analyzer_uses_protocol_seed_and_reports_missing_cohort_rows(tmp_path):
    sample_ids = ["v1", "v2"]
    protocol = {
        "random_seed": 17,
        "cohort_sample_ids": sample_ids,
        "cohort_sha256": hashlib.sha256("v1\nv2".encode()).hexdigest(),
        "review_sample_count": 1,
        "review_sample_ids": sorted(
            sample_ids,
            key=lambda sample_id: hashlib.sha256(f"17:{sample_id}".encode()).digest(),
        )[:1],
        "sensitivity_sample_ids": [],
        "retention_percentages": [100, 0],
    }
    protocol["matched_prefix_tokens"] = 2
    (tmp_path / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    row = _rollout(100)
    row["protocol_sha256"] = protocol_fingerprint(protocol)
    append_jsonl(tmp_path / "raw_rollouts.jsonl", row)

    result = analyze_free_running(
        output_dir=tmp_path, bootstrap_replicates=100
    )
    assert result["rollout_study_ready"] is False
    assert result["rollout_coverage"]["cohort_complete"] is False
    conditions = {row["retention_percentage"]: row
                  for row in result["rollout_coverage"]["conditions"]}
    assert conditions[100]["missing_sample_ids"] == ["v2"]
    assert conditions[0]["missing_sample_ids"] == ["v1", "v2"]
    assert result["human_review"]["preselected_review_samples"] == 1
    assert result["human_review"]["review_pairs"] == 0
    with pytest.raises(ValueError, match="differs from protocol seed"):
        analyze_free_running(output_dir=tmp_path, seed=18, bootstrap_replicates=100)


def _protocol_for_one_sample():
    protocol = {
        "random_seed": 17,
        "cohort_sample_ids": ["v1"],
        "cohort_sha256": hashlib.sha256(b"v1").hexdigest(),
        "cohort_size": 1,
        "review_sample_count": 1,
        "review_sample_ids": ["v1"],
        "sensitivity_sample_ids": [],
        "retention_percentages": [100, 0],
        "matched_prefix_tokens": 2,
    }
    return protocol


def _protocol_rollout(retention, protocol_hash, *, first_token=10):
    row = _rollout(retention, prediction=f"token {first_token} at {retention}")
    row.update({
        "protocol_sha256": protocol_hash,
        "prediction_token_ids": [first_token, 20, 30, 40],
        "stop_reason": "eos",
        "prompt_tokens": 20,
        "visual_token_count": 8,
        "draft_config_sha256": "draft-config",
    })
    return row


def _judge_row(pair, protocol_hash):
    return {
        "pair_id": pair["pair_id"],
        "pair_fingerprint": pair["pair_fingerprint"],
        "sample_id": pair["sample_id"],
        "retention_percentage": pair["retention_percentage"],
        "protocol_sha256": protocol_hash,
        "status": "ok",
        "judge": {
            "scores": {
                side: {dimension: 4 for dimension in (
                    "grammar", "reference", "discourse", "repetition", "overall"
                )}
                for side in ("A", "B")
            },
            "winner": "tie",
        },
    }


def test_analyzer_rejects_rollouts_and_scores_from_another_protocol(tmp_path):
    protocol = _protocol_for_one_sample()
    protocol_hash = protocol_fingerprint(protocol)
    (tmp_path / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    stale_rows = [_protocol_rollout(value, "stale-protocol") for value in (100, 0)]
    pairs, _ = build_blind_pairs(stale_rows, seed=17)
    for row in stale_rows:
        append_jsonl(tmp_path / "raw_rollouts.jsonl", row)
    append_jsonl(tmp_path / "judge_pairs.jsonl", _judge_row(pairs[0], "stale-protocol"))

    result = analyze_free_running(output_dir=tmp_path, bootstrap_replicates=100)
    assert result["rollout_study_ready"] is False
    assert result["rollout_coverage"]["protocol_mismatch_row_ids"]
    assert result["coherence"]["scored_pairs"] == 0
    assert result["coherence"]["protocol_mismatch_pair_ids"] == [pairs[0]["pair_id"]]


def test_analyzer_rebuilds_matched_prefix_key_after_rollout_text_changes(tmp_path):
    protocol = _protocol_for_one_sample()
    protocol_hash = protocol_fingerprint(protocol)
    (tmp_path / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    old_rows = [_protocol_rollout(value, protocol_hash, first_token=10) for value in (100, 0)]
    for row in old_rows:
        append_jsonl(tmp_path / "raw_rollouts.jsonl", row)
    old_pairs, old_key, old_coverage = build_matched_prefix_pairs(
        old_rows, seed=17, prefix_tokens=2,
        decode_token_ids=lambda token_ids: " ".join(map(str, token_ids)),
    )
    (tmp_path / "matched_prefix_key.json").write_text(json.dumps(old_key), encoding="utf-8")
    (tmp_path / "matched_prefix_coverage.json").write_text(json.dumps(old_coverage), encoding="utf-8")
    append_jsonl(
        tmp_path / "judge_matched_prefix_pairs.jsonl",
        _judge_row(old_pairs[0], protocol_hash),
    )

    new_rows = [_protocol_rollout(value, protocol_hash, first_token=99) for value in (100, 0)]
    for row in new_rows:
        append_jsonl(tmp_path / "raw_rollouts.jsonl", row)
    result = analyze_free_running(output_dir=tmp_path, bootstrap_replicates=100)

    matched = result["coherence_matched_prefix"]
    assert matched["scored_pairs"] == 0
    assert matched["missing_pair_ids"]
    assert json.loads((tmp_path / "matched_prefix_key.json").read_text()) != old_key


def test_judge_prompt_identifies_capped_side_without_scoring_its_fragment():
    prompt = make_judge_prompt(
        "A text", "B text", capped_a=True, capped_b=False
    )
    assert "TEXT A (generation status: generation cap reached; ignore an unfinished final clause; JSON string)" in prompt
    assert "TEXT B (generation status: generation ended before the cap; JSON string)" in prompt


def test_judge_prompt_uses_structured_text_content_for_qwen_processor():
    from src.analyze.Validate_Sparrow_hypothesises.run_dflash_free_running import (
        _tokenize_judge_prompt,
    )

    class Processor:
        seen_messages = None

        def apply_chat_template(self, messages, **kwargs):
            self.seen_messages = messages
            assert messages == [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "judge this pair"}],
                }
            ]
            assert kwargs["tokenize"] is True
            return {"input_ids": torch.tensor([[1, 2, 3]])}

    processor = Processor()
    encoded = _tokenize_judge_prompt(processor, "judge this pair")

    assert torch.equal(encoded["input_ids"], torch.tensor([[1, 2, 3]]))


def test_free_running_runner_defaults_to_fp16_after_bf16_parity_issue():
    from src.analyze.Validate_Sparrow_hypothesises.run_dflash_free_running import (
        build_parser,
    )

    args = build_parser().parse_args(["--phase", "preflight"])

    assert args.dtype == "fp16"


def test_matched_prefix_judge_prompt_marks_deliberately_truncated_sides():
    rows = []
    for retention in (100, 0):
        row = _protocol_rollout(retention, "protocol")
        row["row_id"] = f"v1:{retention}"
        rows.append(row)
    pairs, _, _ = build_matched_prefix_pairs(
        rows, seed=17, prefix_tokens=2,
        decode_token_ids=lambda token_ids: " ".join(map(str, token_ids)),
    )
    pair = pairs[0]
    assert pair["truncated_a"] is True
    assert pair["truncated_b"] is True
    prompt = make_judge_prompt(
        pair["text_a"], pair["text_b"],
        truncated_a=pair["truncated_a"], truncated_b=pair["truncated_b"],
    )
    assert "truncated at the shared token length" in prompt
    assert "generation ended before the cap" not in prompt


def test_preflight_returns_its_report(tmp_path, monkeypatch):
    from src.analyze.Validate_Sparrow_hypothesises import run_dflash_free_running as runner

    checkpoint = tmp_path / "training_state.pt"
    checkpoint.write_bytes(b"checkpoint")
    draft_config = tmp_path / "draft.json"
    manifest = tmp_path / "manifest.json"
    calibration = tmp_path / "calibration.json"
    for path in (draft_config, manifest, calibration):
        path.write_text("{}", encoding="utf-8")
    args = type("Args", (), {
        "checkpoint": checkpoint,
        "draft_config": draft_config,
        "manifest": manifest,
        "calibration": calibration,
        "output_dir": tmp_path / "output",
        "sensitivity_limit": 1,
        "initial_anchor_control": "prompt_tail",
    })()
    monkeypatch.setattr(runner, "_selected_records", lambda _args: [{"sample_id": "v1"}])

    result = runner.run_preflight(args)
    assert result["status"] == "preflight_ok"
    assert result["selected_samples"] == 1


def test_surface_token_classes_are_not_presented_as_pos_labels():
    assert classify_token_surface(" dog") == "alphabetic"
    assert classify_token_surface(".") == "punctuation"
    assert classify_token_surface(" 42") == "numeric"
    assert classify_token_surface("<|endoftext|>", is_special=True) == "special"


def test_score_phase_returns_coverage_when_no_complete_pairs_exist(tmp_path, monkeypatch):
    from src.analyze.Validate_Sparrow_hypothesises import run_dflash_free_running as runner

    append_jsonl(tmp_path / "raw_rollouts.jsonl", _rollout(100))
    monkeypatch.setattr(runner, "_write_protocol_and_check", lambda _args: {})
    args = type("Args", (), {
        "output_dir": tmp_path,
        "seed": 7,
        "matched_prefix_tokens": 128,
        "resume": True,
    })()
    result = runner.run_judging(args)
    assert result["candidate_pairs"] == 0
    assert result["judge_rows_written"] == 0
    assert json.loads((tmp_path / "blind_key.json").read_text()) == {}


def test_tiny_qwen25vl_dflash_probe_records_probability_distribution():
    import sys
    import torch

    from src.infer.dflash_free_running import FreeRunningDFlashDecoder
    from src.infer.qwen25vl_dflash_compare import _compute_position_ids

    specforge_root = "/path/to/local-resource"
    if specforge_root not in sys.path:
        sys.path.insert(0, specforge_root)
    from specforge.modeling.draft.dflash import DFlashDraftModel
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen3Config

    target_config = Qwen2_5_VLConfig(
        text_config={
            "vocab_size": 100, "hidden_size": 16, "intermediate_size": 32,
            "num_hidden_layers": 4, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 4, "rope_theta": 1_000_000,
            "rope_scaling": {"type": "mrope", "mrope_section": [1, 1, 0]},
            "pad_token_id": 0,
        },
        vision_config={
            "depth": 1, "hidden_size": 16, "intermediate_size": 32,
            "num_heads": 2, "out_hidden_size": 16, "spatial_merge_size": 1,
            "patch_size": 2, "temporal_patch_size": 1,
        },
        image_token_id=98, video_token_id=99,
        vision_start_token_id=96, vision_end_token_id=97,
    )
    target_config._attn_implementation = "sdpa"
    target = Qwen2_5_VLForConditionalGeneration(target_config).eval()
    draft_config = Qwen3Config(
        vocab_size=100, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        head_dim=4, max_position_embeddings=64, block_size=4,
        num_target_layers=4, layer_types=["full_attention"],
        dflash_config={
            "mask_token_id": 0, "target_layer_ids": [0],
            "position_encoding": "qwen2_5_vl_mrope", "mrope_section": [1, 1, 0],
        },
    )
    draft_config._attn_implementation = "sdpa"
    draft = DFlashDraftModel(draft_config).eval()
    input_ids = torch.tensor([[1, 96, 99, 99, 99, 99, 97, 5, 6]])
    target_kwargs = {
        "attention_mask": torch.ones_like(input_ids),
        "pixel_values_videos": torch.randn(4, 12),
        "video_grid_thw": torch.tensor([[1, 2, 2]]),
    }
    position_ids = _compute_position_ids(target, {"input_ids": input_ids, **target_kwargs})
    keep, _ = build_visual_retention_mask(input_ids.shape[1], [2, 3, 4, 5], 50)
    decoder = FreeRunningDFlashDecoder(target, draft, device=torch.device("cpu"))

    probe = decoder.probe_teacher_forced(
        input_ids=input_ids, position_ids=position_ids,
        target_kwargs=target_kwargs, answer_ids=torch.tensor([20, 21, 22, 23, 24]),
        context_keep_mask=keep,
    )
    stats = [
        row
        for block in probe["probe_positions"]
        for row in block["proposal_distribution_statistics"]
    ]
    assert len(stats) == 5
    for row in stats:
        assert len(row["draft_top_token_ids"]) == 32
        assert sum(row["draft_top_token_probabilities"]) + row["draft_tail_probability_mass"] == pytest.approx(1.0, abs=1e-5)
        assert 0 <= row["draft_top1_probability"] <= 1
        assert row["draft_entropy_nats"] >= 0
        assert 0 <= row["draft_target_prefix_probability"] <= 1


def test_probability_gather_moves_target_indices_to_lm_head_device():
    import torch

    from src.infer.dflash_free_running import _align_target_token_ids

    probabilities = torch.empty((2, 100), device="meta")
    reference_ids = torch.tensor([5, 6], dtype=torch.long, device="cpu")
    aligned = _align_target_token_ids(probabilities, reference_ids)
    assert aligned.device == probabilities.device


def test_sensitivity_measures_probability_shift_when_greedy_proposal_is_unchanged():
    full_stats = {
        "top_token_ids": [10, 11],
        "top_token_probabilities": [0.8, 0.1],
        "tail_probability_mass": 0.1,
    }
    cut_stats = {
        "top_token_ids": [10, 12],
        "top_token_probabilities": [0.6, 0.3],
        "tail_probability_mass": 0.1,
    }
    rows = []
    for retention in (100, 0):
        stats = full_stats if retention == 100 else cut_stats
        rows.append({
            "sample_id": "v1",
            "retention_percentage": retention,
            "status": "ok",
            "input_fingerprint": "same-input",
            "teacher_forced_answer_sha256": "same-answer",
            "probe_tokens": [{
                "answer_position": 0,
                "target_token_id": 20,
                "draft_proposal_token_id": 10,
                "draft_top_token_ids": stats["top_token_ids"],
                "draft_top_token_probabilities": stats["top_token_probabilities"],
                "draft_tail_probability_mass": stats["tail_probability_mass"],
            }],
        })
    summary, _, token_rows = summarize_visual_sensitivity(
        rows, seed=7, bootstrap_replicates=100
    )
    cut = summary["primary"]
    assert cut["mean_changed_proposal_fraction_vs_full"] == 0
    assert cut["mean_coarse_distribution_js_vs_full"] > 0
    assert token_rows[-1]["coarse_distribution_js_vs_full"] > 0
