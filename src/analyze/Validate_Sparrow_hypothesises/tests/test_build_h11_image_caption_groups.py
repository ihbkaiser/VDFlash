import json
import tempfile
import unittest
from pathlib import Path

from src.analyze.Validate_Sparrow_hypothesises.build_h11_image_caption_groups import (
    build_groups,
)


def _write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class BuildH11GroupsTest(unittest.TestCase):
    def test_build_groups_are_deterministic_and_disjoint_by_id_and_image(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            train = [
                {"id": "train-1", "image": "images/train1.jpg", "prompt": "Describe.",
                 "response": "caption 1", "messages": []},
                {"id": "train-2", "image": "images/train2.jpg", "prompt": "Describe.",
                 "response": "caption 2", "messages": []},
                {"id": "train-3", "image": "images/train3.jpg", "prompt": "Describe.",
                 "response": "caption 3", "messages": []},
            ]
            source = [
                {"id": "train-1", "image": "images/train1.jpg", "prompt": "Describe.", "response": "seen id"},
                {"id": "other-id-same-image", "image": "images/train2.jpg", "prompt": "Describe.",
                 "response": "seen image"},
                {"id": "eval-1", "image": "images/eval1.jpg", "prompt": "Describe <image>.",
                 "response": "new caption 1"},
                {"id": "eval-2", "image": "images/eval2.jpg", "prompt": "Describe.",
                 "response": "new caption 2"},
                {"id": "eval-3", "image": "images/eval3.jpg", "prompt": "Describe.",
                 "response": "new caption 3"},
            ]
            train_path, source_path = tmp_path / "train.jsonl", tmp_path / "source.jsonl"
            _write_jsonl(train_path, train)
            _write_jsonl(source_path, source)

            first = build_groups(
                source_annotations=source_path,
                train_manifest=train_path,
                output_dir=tmp_path / "out1",
                reference_count=2,
                eval_count=2,
                seed=77,
            )
            second = build_groups(
                source_annotations=source_path,
                train_manifest=train_path,
                output_dir=tmp_path / "out2",
                reference_count=2,
                eval_count=2,
                seed=77,
            )

            ref1 = (tmp_path / "out1/reference_manifest.jsonl").read_text(encoding="utf-8")
            ref2 = (tmp_path / "out2/reference_manifest.jsonl").read_text(encoding="utf-8")
            eval1 = (tmp_path / "out1/heldout_eval_manifest.jsonl").read_text(encoding="utf-8")
            eval2 = (tmp_path / "out2/heldout_eval_manifest.jsonl").read_text(encoding="utf-8")
            self.assertEqual(ref1, ref2)
            self.assertEqual(eval1, eval2)

            refs = [json.loads(line) for line in ref1.splitlines()]
            evals = [json.loads(line) for line in eval1.splitlines()]
            ref_images = {row["image"] for row in refs}
            eval_images = {row["image"] for row in evals}
            eval_ids = {row["source_id"] for row in evals}
            self.assertEqual(len(refs), 2)
            self.assertEqual(len(evals), 2)
            self.assertFalse(ref_images & eval_images)
            self.assertFalse({row["image"] for row in train} & eval_images)
            self.assertFalse({row["id"] for row in train} & eval_ids)
            self.assertTrue({row["id"] for row in evals}.issubset(
                {"heldout:eval-1:2", "heldout:eval-2:3", "heldout:eval-3:4"}
            ))
            self.assertTrue(all(row["messages"][0]["content"][0] == {"type": "image"}
                                for row in evals))
            self.assertEqual(first["source_scan"]["excluded_train_id"], 1)
            self.assertEqual(first["source_scan"]["excluded_train_image"], 1)
            self.assertEqual(second["source_annotations_sha256"], first["source_annotations_sha256"])

    def test_fails_if_heldout_pool_is_too_small(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            train_path = tmp_path / "train.jsonl"
            source_path = tmp_path / "source.jsonl"
            _write_jsonl(train_path, [{"id": "train", "image": "images/1.jpg"}])
            _write_jsonl(source_path, [{"id": "train", "image": "images/1.jpg", "prompt": "Q", "response": "A"}])

            with self.assertRaisesRegex(ValueError, "found 0 after exclusions"):
                build_groups(
                    source_annotations=source_path,
                    train_manifest=train_path,
                    output_dir=tmp_path / "out",
                    reference_count=1,
                    eval_count=1,
                )
