from src.analyze.Validate_Sparrow_hypothesises.mcq2_metrics import (
    parse_vdc_mcq2_response,
)


def test_mcq2_response_metrics_separate_choice_correctness_from_full_answer():
    result = parse_vdc_mcq2_response(
        "(B) A person walks across the room carrying a box.", "B"
    )

    assert result == {
        "selected_option": "B",
        "answer_text": "A person walks across the room carrying a box.",
        "answer_word_count": 9,
        "answer_sentence_count": 1,
        "format_compliant": True,
        "choice_correct": True,
    }


def test_option_only_is_parsed_but_not_counted_as_mcq2_format_compliance():
    result = parse_vdc_mcq2_response("(B)", "B")

    assert result["selected_option"] == "B"
    assert result["answer_text"] == ""
    assert result["answer_word_count"] == 0
    assert result["answer_sentence_count"] == 0
    assert result["format_compliant"] is False
    assert result["choice_correct"] is True


def test_h21_runner_accepts_mcq2_prompt_variant():
    from src.analyze.Validate_Sparrow_hypothesises.run_dflash_h21_h31 import (
        build_parser,
    )

    args = build_parser().parse_args(["h2_1", "--prompt-variants", "mcq2"])

    assert args.prompt_variants == ["mcq2"]
