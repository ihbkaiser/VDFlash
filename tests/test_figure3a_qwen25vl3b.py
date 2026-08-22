from src.analyze.figure3a_qwen25vl3b import (
    aggregate_figure3a_rows,
    build_mvbench_prompt,
    build_parser,
    default_layer_cut_points,
    score_mvbench_prediction,
)


def test_default_cut_points_use_paper_spacing_and_exclude_native_baseline():
    assert default_layer_cut_points(36) == [0, 4, 8, 12, 16, 20, 24, 28, 32]
    assert default_layer_cut_points(3, stride=4) == [0]


def test_default_video_budget_matches_sparrow_mvbench_prompt():
    args = build_parser().parse_args([])

    assert args.max_frames == 8
    assert args.max_pixels == 360 * 420


def test_mvbench_prompt_matches_canonical_task_template():
    row = {
        "question": "What color is it?",
        "candidates": ["red", "blue"],
    }

    assert build_mvbench_prompt(row) == (
        "Question:What color is it?\n"
        "Option:\n"
        "(A) red\n"
        "(B) blue\n"
        "Only give the best option.\n"
    )


def test_score_prediction_maps_answer_text_and_option_letters():
    row = {"answer": "blue", "candidates": ["red", "blue"]}

    assert score_mvbench_prediction(row, "(B)") == {
        "correct": True,
        "target_option": "B",
        "predicted_option": "B",
    }
    assert score_mvbench_prediction(row, "Answer:B") == {
        "correct": True,
        "target_option": "B",
        "predicted_option": "B",
    }
    assert score_mvbench_prediction(row, "blue") == {
        "correct": False,
        "target_option": "B",
        "predicted_option": None,
    }


def test_aggregate_figure3a_rows_reports_task_and_cutoff_accuracy():
    rows = [
        {"task": "action_prediction", "layer_cut": 0, "correct": True, "prefix_agreement": 1.0},
        {"task": "action_prediction", "layer_cut": 0, "correct": False, "prefix_agreement": 0.5},
        {"task": "action_prediction", "layer_cut": 4, "correct": True, "prefix_agreement": 1.0},
        {"task": "moving_direction", "layer_cut": 0, "correct": True, "prefix_agreement": 1.0},
    ]

    summary = aggregate_figure3a_rows(rows)

    assert summary == [
        {
            "task": "action_prediction",
            "layer_cut": 0,
            "num_samples": 2,
            "num_correct": 1,
            "accuracy": 0.5,
            "mean_prefix_agreement": 0.75,
            "lossless_rate": 0.5,
        },
        {
            "task": "action_prediction",
            "layer_cut": 4,
            "num_samples": 1,
            "num_correct": 1,
            "accuracy": 1.0,
            "mean_prefix_agreement": 1.0,
            "lossless_rate": 1.0,
        },
        {
            "task": "moving_direction",
            "layer_cut": 0,
            "num_samples": 1,
            "num_correct": 1,
            "accuracy": 1.0,
            "mean_prefix_agreement": 1.0,
            "lossless_rate": 1.0,
        },
    ]
