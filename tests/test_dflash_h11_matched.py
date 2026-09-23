"""Matched-budget sampling and paired outcome checks without CUDA."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.analyze.Validate_Sparrow_hypothesises.analyze_dflash_h11_matched import (
    analyze_matched, choose_positions,
)


class MatchedBudgetTests(unittest.TestCase):
    def test_equal_budget_and_eligibility(self):
        ids = [10, 99, 12, 99, 13, 20, 14, 99, 21, 22]
        first, n_visual, n_text = choose_positions(
            ids, 10, visual_ids={99}, special_ids={10, 20},
            budget=2, seed=4, sample_id="sample",
        )
        second, _, _ = choose_positions(
            ids, 10, visual_ids={99}, special_ids={10, 20},
            budget=2, seed=4, sample_id="sample",
        )
        self.assertEqual(first, second)
        self.assertEqual((n_visual, n_text), (3, 5))
        self.assertEqual(len(first["visual_k"]), len(first["text_k"]))
        self.assertTrue(all(ids[position] == 99 for position in first["visual_k"]))
        self.assertTrue(all(ids[position] not in {10, 20, 99} for position in first["text_k"]))
        self.assertIsNone(choose_positions(
            ids, 10, visual_ids={99}, special_ids={10},
            budget=4, seed=4, sample_id="sample",
        )[0])

    def test_matched_pairs_compare_same_checkpoint_and_context_length(self):
        rng = np.random.default_rng(11)
        n_reference, n_test = 80, 20
        train = rng.normal(size=(n_reference, 2, 2, 12)).astype("float32")
        base = rng.normal(size=(n_test, 2, 2, 12)).astype("float32")
        visual = base + 5.0
        text = base.copy()
        ids = np.asarray([f"sample{i}" for i in range(n_test)])
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            matched = Path(temporary) / "matched"
            source.mkdir()
            matched.mkdir()
            np.savez_compressed(source / "train_full.npz", vectors=train, checkpoint="phase2")
            np.savez_compressed(source / "test_full_cut.npz", full=base, cut=visual,
                                sample_ids=ids, checkpoint="phase2")
            np.savez_compressed(matched / "matched_vectors.npz", visual=visual,
                                text=text, sample_ids=ids, checkpoint="phase2", budget=2)
            with (source / "test_full_cut.jsonl").open("w", encoding="utf-8") as original, \
                    (matched / "matched_reports.jsonl").open("w", encoding="utf-8") as current:
                for sid in ids:
                    for condition, accepted in (("full", 10), ("cut", 2)):
                        original.write(json.dumps({
                            "sample_id": sid, "visual_condition": condition,
                            "target_input_fingerprint": sid,
                            "target_output_hash": sid,
                            "acceptance_rounds": [{"matched_proposals": accepted}],
                        }) + "\n")
                    for condition, accepted, dropped in (
                        ("visual_k", 5, [1, 3]), ("text_k", 9, [2, 4])
                    ):
                        current.write(json.dumps({
                            "sample_id": sid, "matched_condition": condition,
                            "budget": 2, "dropped_positions": dropped,
                            "context_length": 20,
                            "target_input_fingerprint": sid,
                            "target_output_hash": sid,
                            "matched_proposals": accepted,
                        }) + "\n")
            result = analyze_matched(source, matched, bootstrap=100, components=3)
            self.assertEqual(result["n_eligible"], n_test)
            self.assertEqual(result["mean_visual_minus_text_matches"], -4.0)
            self.assertGreater(result["layers"][0]["distance_ci95"][0], 0)
            # A broken control with unequal remaining key counts must fail.
            report = matched / "matched_reports.jsonl"
            contents = report.read_text(encoding="utf-8")
            report.write_text(contents.replace('"context_length": 20', '"context_length": 21', 1),
                              encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unequal deletion budget or context length"):
                analyze_matched(source, matched, bootstrap=100, components=3)


if __name__ == "__main__":
    unittest.main()
