import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.infer.qwen25vl_dflash_compare import (
    InstrumentedDFlashDecoder,
    PreparedVideoPrompt,
    SpeculativeDecodeResult,
    _checkpoint_result,
    _compute_position_ids,
    _eos_token_ids,
    _print_report,
    _target_greedy,
    build_parser,
    run_comparison,
    load_manifest_sample,
    resolve_video_path,
    score_caption,
    validate_report_success,
)


def _text_metrics():
    return {
        "exact_match": 1.0,
        "bleu1": 1.0,
        "bleu2": 1.0,
        "bleu3": 1.0,
        "bleu4": 1.0,
        "bleu": 1.0,
        "rouge_l": 1.0,
        "coverage": 1.0,
        "unigram_precision": 1.0,
        "unigram_recall": 1.0,
        "unigram_f1": 1.0,
    }


def _timing():
    return {
        "prefill_s": 0.1,
        "draft_s": 0.1,
        "verify_s": 0.1,
        "decode_s": 0.3,
        "end_to_end_s": 1.2,
        "tokens_per_second": 4.0,
        "checkpoint_load_s": 0.2,
        "target_prefill_s": 0.2,
        "target_decode_s": 0.8,
        "target_greedy_s": 1.0,
        "speedup_vs_target": 0.83,
    }


def _acceptance():
    return {
        "tau": 2.0,
        "tau_proposal": 1.0,
        "tau_effective": 2.0,
    }


def _speedup():
    return {"esr": 1.2, "dsr": 1.3}


def test_load_manifest_sample_and_resolve_vdc_video():
    manifest = Path("dataset/VideoDetailCaption/test.jsonl")

    sample = load_manifest_sample(manifest, 0)
    video = resolve_video_path(sample, Path("dataset/VideoDetailCaption"))

    assert sample["video_name"] == "v_AwgGYaV1lT0"
    assert video.name == "v_AwgGYaV1lT0.mp4"
    assert video.is_file()


def test_load_manifest_sample_rejects_out_of_range_index(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"id": "only"}) + "\n", encoding="utf-8")

    with pytest.raises(IndexError, match="sample-index 1"):
        load_manifest_sample(manifest, 1)


def test_score_caption_reports_perfect_and_partial_overlap():
    perfect = score_caption("A man chops wood.", "A man chops wood.")
    partial = score_caption("A woman runs outside.", "A man chops wood.")

    assert perfect["exact_match"] == 1.0
    assert perfect["rouge_l"] == 1.0
    assert perfect["unigram_f1"] == 1.0
    assert 0.0 <= partial["bleu"] <= 1.0
    assert 0.0 <= partial["rouge_l"] <= 1.0
    assert 0.0 <= partial["coverage"] <= 1.0


def test_validate_report_success_requires_both_lossless_results():
    report = {
        "target_baseline": {"prediction": "target answer"},
        "checkpoints": [
            {
                "status": "ok",
                "prediction": "target answer",
                "outputs_match": True,
                "text_metrics": _text_metrics(),
                "timing": _timing(),
                "acceptance": _acceptance(),
                "speedup": _speedup(),
            },
            {
                "status": "ok",
                "prediction": "target answer",
                "outputs_match": True,
                "text_metrics": _text_metrics(),
                "timing": {**_timing(), "end_to_end_s": 1.1, "tokens_per_second": 4.2},
                "acceptance": _acceptance(),
                "speedup": _speedup(),
            },
        ],
    }

    assert validate_report_success(report) is True

    report["checkpoints"][1]["outputs_match"] = False
    assert validate_report_success(report) is False


def test_validate_report_rejects_missing_prediction():
    report = {
        "target_baseline": {"prediction": "target answer"},
        "checkpoints": [
            {
                "status": "ok",
                "prediction": "",
                "outputs_match": True,
                "text_metrics": _text_metrics(),
                "timing": _timing(),
                "acceptance": _acceptance(),
                "speedup": _speedup(),
            },
            {
                "status": "ok",
                "prediction": "target answer",
                "outputs_match": True,
                "text_metrics": _text_metrics(),
                "timing": {**_timing(), "end_to_end_s": 1.1, "tokens_per_second": 4.2},
                "acceptance": _acceptance(),
                "speedup": _speedup(),
            },
        ],
    }

    assert validate_report_success(report) is False


def test_validate_report_rejects_invalid_metric_or_timing():
    report = {
        "target_baseline": {"prediction": "target answer"},
        "checkpoints": [
            {
                "status": "ok",
                "prediction": "target answer",
                "outputs_match": True,
                "text_metrics": {**_text_metrics(), "bleu": 1.5},
                "timing": _timing(),
                "acceptance": _acceptance(),
                "speedup": _speedup(),
            },
            {
                "status": "ok",
                "prediction": "target answer",
                "outputs_match": True,
                "text_metrics": _text_metrics(),
                "timing": {**_timing(), "verify_s": -0.1},
                "acceptance": _acceptance(),
                "speedup": _speedup(),
            },
        ],
    }

    assert validate_report_success(report) is False


def test_speculative_result_reports_tau_on_full_non_terminal_rounds():
    result = SpeculativeDecodeResult(
        output_ids=torch.zeros(1, 20, dtype=torch.long),
        num_input_tokens=10,
        acceptance_rounds=[
            {
                "matched_proposals": 5,
                "effective_emitted_tokens": 6,
                "is_partial_block": False,
                "is_terminal": False,
            },
            {
                "matched_proposals": 7,
                "effective_emitted_tokens": 8,
                "is_partial_block": False,
                "is_terminal": False,
            },
            {
                "matched_proposals": 3,
                "effective_emitted_tokens": 4,
                "is_partial_block": False,
                "is_terminal": True,
            },
            {
                "matched_proposals": 2,
                "effective_emitted_tokens": 3,
                "is_partial_block": True,
                "is_terminal": True,
            },
        ],
    )

    metrics = result.as_dict()

    assert metrics["tau_proposal"] == pytest.approx(6.0)
    assert metrics["tau_effective"] == pytest.approx(7.0)
    assert metrics["tau"] == pytest.approx(7.0)


def test_checkpoint_speedups_separate_esr_and_dsr_without_load_or_data_time():
    target_ids = torch.tensor([[1, 2, 3, 4, 5]])
    speculative = SpeculativeDecodeResult(
        output_ids=target_ids.clone(),
        num_input_tokens=3,
        acceptance_rounds=[],
        prefill_s=1.0,
        draft_s=1.0,
        verify_s=3.0,
        decode_s=4.0,
        end_to_end_s=5.0,
    )

    result = _checkpoint_result(
        label="checkpoint",
        checkpoint="checkpoint",
        reference="4 5",
        target_ids=target_ids,
        speculative=speculative,
        processor=_Processor(),
        target_timing={
            "prefill_s": 2.0,
            "decode_s": 8.0,
            "end_to_end_s": 10.0,
        },
        load_s=99.0,
    )

    assert result["speedup"]["esr"] == pytest.approx(2.0)
    assert result["speedup"]["dsr"] == pytest.approx(2.0)
    assert result["timing"]["checkpoint_load_s"] == pytest.approx(99.0)


def test_compute_position_ids_uses_qwen_three_axis_rope_api():
    class RopeModel:
        def get_rope_index(self, **kwargs):
            length = kwargs["input_ids"].shape[1]
            positions = torch.arange(length).view(1, 1, -1).expand(3, 1, -1)
            return positions, torch.zeros(1, 1, dtype=torch.long)

    target = SimpleNamespace(model=RopeModel())
    inputs = {
        "input_ids": torch.tensor([[10, 11, 12]]),
        "video_grid_thw": torch.tensor([[1, 2, 2]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }

    positions = _compute_position_ids(target, inputs)

    assert positions.shape == (3, 1, 3)
    assert positions[0, 0].tolist() == [0, 1, 2]


def test_eos_token_ids_accepts_scalar_and_list_values():
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(eos_token_id=[63, 64], im_end_id=65)
    )
    target = SimpleNamespace(
        generation_config=SimpleNamespace(eos_token_id=[64, 66])
    )

    assert _eos_token_ids(processor, target) == [63, 64, 65, 66]


def test_target_greedy_uses_cached_one_token_forwards_and_reports_split_timing():
    class CachedTarget(_Target):
        def __init__(self):
            super().__init__()
            self.forward_lengths = []
            self.cache_positions = []
            self.position_shapes = []

        def __call__(self, input_ids, cache_position=None, position_ids=None, **kwargs):
            self.forward_lengths.append(int(input_ids.shape[1]))
            self.cache_positions.append(
                None if cache_position is None else cache_position.detach().cpu().tolist()
            )
            self.position_shapes.append(
                None if position_ids is None else tuple(position_ids.shape)
            )
            return super().__call__(
                input_ids,
                cache_position=cache_position,
                position_ids=position_ids,
                **kwargs,
            )

        def generate(self, **kwargs):
            del kwargs
            raise AssertionError("target.generate must not be used for the baseline")

    target = CachedTarget()
    prompt = PreparedVideoPrompt(
        record={"video_name": "sample"},
        video_path=Path("sample.mp4"),
        inputs={
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        },
        position_ids=torch.arange(3).view(1, 1, 3).expand(3, 1, 3),
        target_kwargs={"pixel_values_videos": torch.ones(1)},
        frame_counts=(8,),
        video_grid_thw=((1, 2, 2),),
    )

    output, timing = _target_greedy(
        target,
        prompt,
        max_new_tokens=3,
        stop_token_ids=[],
        device=torch.device("cpu"),
    )

    assert output[0].tolist() == [1, 2, 3, 4, 5, 6]
    assert target.forward_lengths == [3, 1, 1]
    assert target.cache_positions == [[0, 1, 2], [3], [4]]
    assert target.position_shapes == [(3, 1, 3), (3, 1, 1), (3, 1, 1)]
    assert timing["num_output_tokens"] == 3
    assert timing["prefill_s"] >= 0.0
    assert timing["decode_s"] >= 0.0
    assert timing["end_to_end_s"] >= timing["prefill_s"]


class _Cache:
    def __init__(self, length=0):
        self.length = length

    def get_seq_length(self):
        return self.length

    def crop(self, length):
        self.length = int(length)


class _Embedding(nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(64, hidden_size))
        self.hidden_size = hidden_size

    def forward(self, input_ids):
        values = input_ids.to(dtype=torch.float32).unsqueeze(-1)
        return values.expand(*input_ids.shape, self.hidden_size)


class _Head(nn.Module):
    def __init__(self, vocab_size=64):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(vocab_size, 8))
        self.vocab_size = vocab_size

    def forward(self, hidden_states):
        next_ids = (hidden_states[..., 0].long() + 1) % self.vocab_size
        logits = torch.full(
            (*hidden_states.shape[:-1], self.vocab_size), -100.0,
            device=hidden_states.device,
        )
        return logits.scatter(-1, next_ids.unsqueeze(-1), 100.0)


class _Target:
    def __init__(self):
        self.anchor = nn.Parameter(torch.zeros(()))
        self.embed_tokens = _Embedding()
        self.lm_head = _Head()
        self.calls = []

    def parameters(self):
        return iter((self.anchor,))

    def get_input_embeddings(self):
        return self.embed_tokens

    def generate(self, input_ids, max_new_tokens, **kwargs):
        del kwargs
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            generated = torch.cat(
                [generated, (generated[:, -1:] + 1) % self.lm_head.vocab_size], dim=1
            )
        return generated

    def __call__(self, input_ids, past_key_values=None, output_hidden_states=False, **kwargs):
        self.calls.append({"has_video": "pixel_values_videos" in kwargs})
        cache = past_key_values or _Cache()
        hidden = self.embed_tokens(input_ids)
        output_cache = _Cache(cache.get_seq_length() + input_ids.shape[1])
        return SimpleNamespace(
            logits=self.lm_head(hidden),
            hidden_states=(hidden, hidden) if output_hidden_states else None,
            past_key_values=output_cache,
        )


class _Draft:
    block_size = 4
    mask_token_id = 0
    target_layer_ids = [0]

    def __call__(self, target_hidden, noise_embedding, **kwargs):
        del kwargs
        base = target_hidden[:, -1:, :1] + 1
        offsets = torch.arange(
            noise_embedding.shape[1], device=noise_embedding.device,
            dtype=noise_embedding.dtype,
        ).view(1, -1, 1)
        hidden = noise_embedding.clone()
        hidden[..., :1] = base + offsets - 1
        return hidden

    def _sample_draft_tokens(self, target, draft_hidden, block_output_ids):
        del block_output_ids
        return target.lm_head(draft_hidden[:, -self.block_size + 1 :, :]).argmax(-1)


class _Tokenizer:
    eos_token_id = 63
    im_end_id = None

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids)


class _Processor:
    tokenizer = _Tokenizer()


def test_instrumented_decoder_accepts_stub_block_and_forwards_video_once():
    target = _Target()
    draft = _Draft()
    decoder = InstrumentedDFlashDecoder(
        target,
        draft,
        device=torch.device("cpu"),
        token_decoder=lambda token_ids: "|".join(str(token_id) for token_id in token_ids),
    )
    input_ids = torch.tensor([[1, 2, 3]])
    positions = torch.arange(input_ids.shape[1]).view(1, -1)

    result = decoder.decode(
        input_ids=input_ids,
        position_ids=positions,
        target_kwargs={"pixel_values_videos": torch.ones(1), "video_grid_thw": torch.ones(1, 3)},
        max_new_tokens=5,
        stop_token_ids=[],
    )

    assert result.output_ids[0, input_ids.shape[1] :].tolist() == [4, 5, 6, 7, 8]
    assert result.acceptance_rounds
    assert result.acceptance_rounds[0]["matched_proposals"] == 3
    assert result.acceptance_rounds[0]["block_text"]
    assert result.acceptance_rounds[0]["draft_proposal_text"]
    assert result.acceptance_rounds[0]["block_token_ids"]
    assert result.acceptance_rounds[0]["draft_proposal_token_ids"]
    assert result.target_forward_calls >= 2
    assert target.calls[0]["has_video"] is True
    assert all(call["has_video"] is False for call in target.calls[1:])


def test_instrumented_decoder_trims_at_stop_token_after_full_budget():
    target = _Target()
    draft = _Draft()
    decoder = InstrumentedDFlashDecoder(target, draft, device=torch.device("cpu"))
    input_ids = torch.tensor([[1, 2, 3]])

    result = decoder.decode(
        input_ids=input_ids,
        position_ids=torch.arange(3).view(1, 3),
        target_kwargs=None,
        max_new_tokens=5,
        stop_token_ids=[8],
    )

    assert result.output_ids[0, input_ids.shape[1] :].tolist() == [4, 5, 6, 7, 8]
    assert result.num_output_tokens == 5


def test_run_comparison_records_mismatch_and_continues_to_second_checkpoint(
    monkeypatch, tmp_path
):
    import src.infer.qwen25vl_dflash_compare as runner

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "video_name": "sample",
                "question": "Describe the video.",
                "answer": "4 5 6 7 8",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    target = _Target()
    prompt = PreparedVideoPrompt(
        record={"video_name": "sample"},
        video_path=tmp_path / "sample.mp4",
        inputs={
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        },
        position_ids=torch.arange(3).view(1, 3),
        target_kwargs={"pixel_values_videos": torch.ones(1)},
        frame_counts=(8,),
        video_grid_thw=((1, 2, 2),),
    )
    monkeypatch.setattr(
        runner, "_load_target", lambda *args, **kwargs: (_Processor(), target, 0.1)
    )
    monkeypatch.setattr(runner, "prepare_video_prompt", lambda *args, **kwargs: prompt)

    drafts = iter([SimpleNamespace(match=False), SimpleNamespace(match=True)])

    def fake_load_draft(*args, **kwargs):
        return next(drafts), 0.2

    class FakeDecoder:
        def __init__(self, target_model, draft_model, *, device, token_decoder=None):
            del target_model, device, token_decoder
            self.draft = draft_model

        def decode(self, *, input_ids, **kwargs):
            del kwargs
            suffix = [4, 5, 6, 7, 8] if self.draft.match else [9, 9, 9, 9, 9]
            output_ids = torch.tensor([input_ids[0].tolist() + suffix])
            return SpeculativeDecodeResult(
                output_ids=output_ids,
                num_input_tokens=input_ids.shape[1],
                target_forward_calls=2,
                prefill_s=0.1,
                draft_s=0.1,
                verify_s=0.1,
                decode_s=0.2,
                end_to_end_s=0.2,
            )

    monkeypatch.setattr(runner, "_load_draft", fake_load_draft)
    monkeypatch.setattr(runner, "InstrumentedDFlashDecoder", FakeDecoder)
    args = build_parser().parse_args(
        [
            "--manifest",
            str(manifest),
            "--video-root",
            str(tmp_path),
            "--checkpoint",
            "first",
            "--checkpoint",
            "second",
            "--device",
            "cpu",
            "--dtype",
            "no",
            "--max-new-tokens",
            "5",
        ]
    )

    report = run_comparison(args)

    assert report["success"] is False
    assert [item["status"] for item in report["checkpoints"]] == ["mismatch", "ok"]
    assert report["checkpoints"][1]["outputs_match"] is True


def test_run_comparison_builds_two_checkpoint_report(monkeypatch, tmp_path, capsys):
    import src.infer.qwen25vl_dflash_compare as runner

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "video_name": "sample",
                "question": "Describe the video.",
                "answer": "4 5 6 7 8",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    target = _Target()
    prompt = PreparedVideoPrompt(
        record={"video_name": "sample"},
        video_path=tmp_path / "sample.mp4",
        inputs={
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        },
        position_ids=torch.arange(3).view(1, 3),
        target_kwargs={"pixel_values_videos": torch.ones(1)},
        frame_counts=(8,),
        video_grid_thw=((1, 2, 2),),
    )
    monkeypatch.setattr(runner, "_load_target", lambda *args, **kwargs: (_Processor(), target, 0.1))
    monkeypatch.setattr(runner, "prepare_video_prompt", lambda *args, **kwargs: prompt)
    monkeypatch.setattr(runner, "_load_draft", lambda *args, **kwargs: (_Draft(), 0.2))

    args = build_parser().parse_args(
        [
            "--manifest", str(manifest),
            "--video-root", str(tmp_path),
            "--checkpoint", "llava",
            "--checkpoint", "sharegpt",
            "--device", "cpu",
            "--dtype", "no",
            "--max-new-tokens", "5",
        ]
    )
    report = run_comparison(args)

    assert report["success"] is True
    assert len(report["checkpoints"]) == 2
    assert all(item["outputs_match"] for item in report["checkpoints"])
    assert all(item["prediction"] == "4 5 6 7 8" for item in report["checkpoints"])

    _print_report(report)
    printed = capsys.readouterr().out
    assert "target metrics:" in printed
    assert "rouge_l=1.000" in printed
    assert "bleu=1.000" in printed


def test_run_all_comparisons_persists_every_sample_and_counts_mismatches(
    monkeypatch, tmp_path
):
    import src.infer.qwen25vl_dflash_compare as runner

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps({"video_name": f"sample-{index}"}) for index in range(3)
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "batch"
    calls = []

    def fake_run_comparison(args):
        calls.append(args.sample_index)
        return {
            "sample_index": args.sample_index,
            "sample_id": f"sample-{args.sample_index}",
            "success": args.sample_index == 0,
        }

    monkeypatch.setattr(runner, "run_comparison", fake_run_comparison)
    args = SimpleNamespace(
        manifest=manifest,
        output_dir=output_dir,
        resume=False,
    )

    summary = runner.run_all_comparisons(args)

    assert calls == [0, 1, 2]
    assert summary["total_samples"] == 3
    assert summary["completed_samples"] == 3
    assert summary["lossless_samples"] == 1
    assert summary["mismatch_samples"] == 2
    assert summary["runtime_errors"] == 0
    assert len(list(output_dir.glob("sample_*.json"))) == 3
    assert (output_dir / "summary.json").is_file()


def test_build_batch_statistics_reports_numeric_aggregates_and_lossless_rate():
    import src.infer.qwen25vl_dflash_compare as runner

    reports = [
        {
            "run_status": "completed",
            "target_baseline": {
                "text_metrics": {"rouge_l": 0.2},
                "timing": {"end_to_end_s": 2.0},
            },
            "checkpoints": [
                {
                    "label": "draft-a",
                    "status": "ok",
                    "outputs_match": True,
                    "text_metrics": {"rouge_l": 0.4},
                    "timing": {"end_to_end_s": 1.0},
                    "acceptance": {"tau": 2.0},
                    "speedup": {"esr": 2.0},
                }
            ],
        },
        {
            "run_status": "completed",
            "target_baseline": {
                "text_metrics": {"rouge_l": 0.6},
                "timing": {"end_to_end_s": 4.0},
            },
            "checkpoints": [
                {
                    "label": "draft-a",
                    "status": "mismatch",
                    "outputs_match": False,
                    "text_metrics": {"rouge_l": 0.8},
                    "timing": {"end_to_end_s": 2.0},
                    "acceptance": {"tau": 1.0},
                    "speedup": {"esr": 1.0},
                }
            ],
        },
    ]

    statistics = runner.build_batch_statistics(reports)
    target = statistics["groups"]["target_baseline"]
    draft = statistics["groups"]["draft-a"]

    assert target["sample_count"] == 2
    assert target["metrics"]["text_metrics.rouge_l"]["n"] == 2
    assert target["metrics"]["text_metrics.rouge_l"]["mean"] == pytest.approx(0.4)
    assert draft["sample_count"] == 2
    assert draft["lossless_count"] == 1
    assert draft["lossless_rate"] == pytest.approx(0.5)
    assert draft["metrics"]["acceptance.tau"]["mean"] == pytest.approx(1.5)


def test_write_vdc50_report_includes_run_configuration_and_metric_table(tmp_path):
    import src.infer.qwen25vl_dflash_compare as runner

    summary = {
        "manifest": "/data/test.jsonl",
        "total_samples": 2,
        "completed_samples": 2,
        "lossless_samples": 1,
        "mismatch_samples": 1,
        "runtime_errors": 0,
        "run_completed": True,
        "statistics": {
            "completed_reports": 2,
            "groups": {
                "draft-a": {
                    "sample_count": 2,
                    "lossless_count": 1,
                    "lossless_rate": 0.5,
                    "metrics": {
                        "acceptance.tau": {
                            "n": 2,
                            "mean": 1.5,
                            "std": 0.5,
                            "min": 1.0,
                            "max": 2.0,
                            "spread": 1.0,
                            "bootstrap_ci95": [1.0, 2.0],
                        }
                    },
                }
            },
        },
    }
    reports = [
        {
            "target_model": "Qwen/Qwen2.5-VL-3B-Instruct",
            "draft_config": "draft.json",
            "device": "cuda:1",
            "dtype": "torch.bfloat16",
            "preprocessing": {
                "num_frames": 8,
                "video_min_pixels": 50176,
                "video_max_pixels": 50176,
                "max_new_tokens": 256,
            },
            "checkpoints": [{"label": "draft-a"}],
        }
    ]

    runner.write_vdc50_report(tmp_path, summary, reports)
    report = (tmp_path / "VDC50_REPORT.md").read_text(encoding="utf-8")

    assert "Qwen2.5-VL-3B-Instruct" in report
    assert "num_frames: `8`" in report
    assert "draft-a" in report
    assert "acceptance.tau" in report
