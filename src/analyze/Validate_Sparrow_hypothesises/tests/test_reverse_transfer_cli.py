from __future__ import annotations

from src.analyze.Validate_Sparrow_hypothesises.run_reverse_transfer import build_parser


def test_cli_defaults_require_real_msd_and_expose_matrix():
    args = build_parser().parse_args([
        "--dataset", "mvbench",
        "--dataset-path", "manifest.jsonl",
        "--video-root", "videos",
        "--output-dir", "out",
    ])
    assert args.msd_checkpoint == "lucylyn/MSD-Qwen2VL-7B-Instruct"
    assert args.conditions == ["E1", "E2", "E3", "E4", "E5", "E6"]
    assert args.permutations is False


def test_cli_filters_condition_matrix():
    args = build_parser().parse_args([
        "--dataset", "vidhalluc",
        "--dataset-path", "manifest.jsonl",
        "--video-root", "videos",
        "--output-dir", "out",
        "--input-order", "text_video",
        "--conditioning", "visual",
        "--permutations",
    ])
    assert args.input_order == "text_video"
    assert args.conditioning == "visual"
    assert args.permutations is True
