import pytest

from src.infer.qwen25vl_dflash_parallel import (
    build_qwen25vl_model_parallel_map,
    parse_max_memory,
)


def test_qwen25vl_model_parallel_map_keeps_boundary_modules_on_gpu_zero():
    mapping = build_qwen25vl_model_parallel_map(36)

    assert mapping["model.visual"] == 0
    assert mapping["model.language_model.embed_tokens"] == 0
    assert mapping["model.language_model.rotary_emb"] == 0
    assert mapping["model.language_model.norm"] == 0
    assert mapping["lm_head"] == 0
    assert [mapping[f"model.language_model.layers.{index}"] for index in range(36)] == (
        [0, 0, 0] + [1] * 31 + [0, 0]
    )


def test_qwen25vl_model_parallel_map_rejects_invalid_boundary_counts():
    with pytest.raises(ValueError, match="cannot exceed"):
        build_qwen25vl_model_parallel_map(4, prefix_layers=3, suffix_layers=2)


def test_parse_max_memory_accepts_visible_gpu_budgets():
    assert parse_max_memory("0:22GiB,1:14GiB") == {
        0: "22GiB",
        1: "14GiB",
    }
    assert parse_max_memory(None) is None
    assert parse_max_memory({0: "10GiB"}) == {0: "10GiB"}


def test_two_gpu_launcher_supports_selected_sample_and_split_draft_device():
    from pathlib import Path

    launcher = Path(
        "src/analyze/Validate_Sparrow_hypothesises/"
        "run_dflash_model_parallel_2gpu.sh"
    )
    contents = launcher.read_text(encoding="utf-8")

    assert "--sample-id" in contents
    assert 'SAMPLE_ID="${SAMPLE_ID:-}"' in contents
    assert 'DRAFT_DEVICE="${DRAFT_DEVICE:-cuda:1}"' in contents
    assert "--draft-device" in contents


def test_two_gpu_launcher_defaults_to_memory_safe_target_attention():
    from pathlib import Path

    launcher = Path(
        "src/analyze/Validate_Sparrow_hypothesises/"
        "run_dflash_model_parallel_2gpu.sh"
    )
    contents = launcher.read_text(encoding="utf-8")

    assert 'TARGET_ATTENTION="${TARGET_ATTENTION:-flash_attention_2}"' in contents
    assert "--target-attention" in contents
