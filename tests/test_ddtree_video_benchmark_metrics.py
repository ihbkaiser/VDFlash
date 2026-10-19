from scripts.run_ddtree_qwen25vl3b_eval import score_vidhalluc


def test_sequence_ranking_accepts_natural_language_a_then_b():
    record = {"task_type": "Sequence_Ranking", "answer": "AB"}

    result = score_vidhalluc(
        "Action A. playing football happens first, followed by Action B. running.",
        record,
    )

    assert result["predicted_label"] == "AB"
    assert result["task_correct"] is True


def test_sequence_ranking_accepts_natural_language_b_then_a():
    record = {"task_type": "Sequence_Ranking", "answer": "BA"}

    result = score_vidhalluc(
        "Action B. boiling water happens first, then Action A. stirring.",
        record,
    )

    assert result["predicted_label"] == "BA"
    assert result["task_correct"] is True


def test_sequence_ranking_does_not_score_incomplete_action_order():
    record = {"task_type": "Sequence_Ranking", "answer": "AB"}

    result = score_vidhalluc("Action B. Adjusting ski bindings.", record)

    assert result["predicted_label"] is None
    assert result["task_correct"] is False
