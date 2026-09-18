"""CPU tests for the DFlash H2.1/H3.1 follow-up experiment helpers."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest

from src.analyze.Validate_Sparrow_hypothesises.run_dflash_h21_h31 import (
    acceptance_by_position,
    build_prompt_regions,
    build_prompt_record,
    build_visual_keep_mask,
    find_answer_hint_positions,
    summarize_dflash_attention_regions,
)


def test_dflash_target_loader_uses_torch_dtype_keyword(monkeypatch):
    import transformers
    from transformers.models.auto import modeling_auto, processing_auto
    from src.infer.qwen25vl_dflash_compare import _load_target

    observed = {}

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return object()

    class FakeTarget:
        def eval(self):
            return self

        def to(self, **_kwargs):
            return self

        def parameters(self):
            return iter(())

    class FakeModel:
        @classmethod
        def from_pretrained(cls, *_args, **kwargs):
            observed.update(kwargs)
            return FakeTarget()

    # Transformers exposes these through a lazy module, so patch the concrete
    # auto modules rather than the package-level proxy attributes.
    monkeypatch.setattr(processing_auto, "AutoProcessor", FakeProcessor)
    monkeypatch.setattr(modeling_auto, "AutoModelForImageTextToText", FakeModel)

    _load_target(
        "fake-model",
        device=torch.device("cpu"),
        dtype=torch.float16,
        attention="sdpa",
        device_map="cuda",
    )

    assert observed["torch_dtype"] is torch.float16
    assert "dtype" not in observed


def test_build_prompt_record_only_changes_question_variant():
    sample = SimpleNamespace(
        video_name="v0",
        question="What happens?",
        answer="A person waves.",
        local_video_path="v0.mp4",
    )

    record = build_prompt_record(sample, "answer_hint")

    assert record["video_name"] == "v0"
    assert record["local_video_path"] == "v0.mp4"
    assert record["answer"] == "A person waves."
    assert record["question"].endswith("Reference information (oracle hint): A person waves.")


def test_visual_keep_mask_retains_text_and_deterministic_visual_prefix():
    mask = build_visual_keep_mask(10, [2, 3, 4, 7], 25)

    assert mask.tolist() == [True, True, True, False, False, True, True, False, True, True]


def test_find_answer_hint_positions_finds_contiguous_token_span():
    full = [10, 11, 20, 21, 22, 30]
    hint = [20, 21, 22]

    assert find_answer_hint_positions(full, hint) == [2, 3, 4]


def test_find_answer_hint_positions_supports_answer_only_boundary_fallback():
    full = [10, 11, 20, 21, 22, 30]
    answer_only = [20, 21, 22]

    assert find_answer_hint_positions(full, answer_only) == [2, 3, 4]


def test_prompt_regions_falls_back_to_answer_suffix_when_hint_boundary_does_not_match():
    class FakeTokenizer:
        all_special_ids = []

        def encode(self, text, add_special_tokens=False):
            if text == "<|im_start|>assistant\n":
                return [15]
            if text == "The answer":
                return [7, 8]
            return [99]

    processor = SimpleNamespace(tokenizer=FakeTokenizer())
    sample = SimpleNamespace(answer="The answer")
    input_ids = torch.tensor([[1, 2, 10, 11, 12, 5, 6, 7, 8, 15]])

    regions = build_prompt_regions(input_ids, processor, [2, 3, 4], sample, "answer_hint")

    assert regions["answer_hint"] == [7, 8]
    assert regions["question_text"] == [5, 6]


def test_attention_region_summary_reports_mass_and_density():
    # [batch=1, heads=1, query=1, key=6], with context length 4.
    weights = torch.tensor([[[[0.10, 0.20, 0.30, 0.10, 0.20, 0.10]]]])
    records = [{"layer_index": 0, "forward_index": 0, "context_length": 4, "weights": weights}]

    summary = summarize_dflash_attention_regions(
        records,
        {"visual": [0], "question_text": [1, 2], "answer_hint": [3]},
    )

    assert summary["visual_mass"] == pytest.approx(0.1)
    assert summary["question_text_mass"] == pytest.approx(0.5)
    assert summary["answer_hint_mass"] == pytest.approx(0.1)
    assert summary["question_text_density"] == pytest.approx(0.25)
    assert summary["answer_hint_density_vs_question_text"] == pytest.approx(0.4)
    assert summary["answer_hint_mass_share_of_question_plus_hint"] == pytest.approx(1 / 6)
    assert summary["captured_layers"] == 1


def test_acceptance_by_position_marks_proposals_not_bonus_tokens():
    rounds = [
        {
            "answer_position_start": 0,
            "proposal_count": 3,
            "matched_proposals": 2,
            "accepted_proposal_positions": [0, 1],
            "proposal_positions": [0, 1, 2],
        },
        {
            "answer_position_start": 3,
            "proposal_count": 3,
            "matched_proposals": 0,
            "accepted_proposal_positions": [],
            "proposal_positions": [3, 4, 5],
        },
    ]

    trace = acceptance_by_position(rounds, 6)

    assert trace[0] == {"position": 0, "proposed": 1, "accepted": 1, "rate": 1.0}
    assert trace[1]["rate"] == 1.0
    assert trace[2]["rate"] == 0.0
    assert trace[3]["rate"] == 0.0


def test_acceptance_by_position_reconstructs_missing_round_offsets():
    rounds = [
        {
            "proposal_count": 2,
            "matched_proposals": 1,
            "effective_emitted_tokens": 2,
            "accepted_proposal_positions": [0],
        },
        {
            "proposal_count": 2,
            "matched_proposals": 0,
            "effective_emitted_tokens": 1,
            "accepted_proposal_positions": [],
        },
    ]

    trace = acceptance_by_position(rounds, 4)

    assert [item["proposed"] for item in trace] == [1, 1, 1, 1]
    assert [item["accepted"] for item in trace] == [1, 0, 0, 0]
