import json
import unittest
from pathlib import Path
from unittest.mock import patch

from src.train_VLM.prepare_llava_heldout import build_heldout_manifest
from src.train_VLM.prepare_responses_vllm import _chat_for_record, _image_path


class HeldoutManifestTests(unittest.TestCase):
    def test_vllm_chat_conversion_preserves_image_and_prompt_order(self):
        image = object()
        record = {
            "id": "heldout-1",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "image": "file:///tmp/sample.jpg"},
                    {"type": "text", "text": "Describe the picture."},
                ],
            }],
        }
        converted = _chat_for_record(record, image)
        self.assertIs(converted[0]["content"][0]["image_pil"], image)
        self.assertEqual(converted[0]["content"][1], {"type": "text", "text": "Describe the picture."})
        self.assertEqual(_image_path("file:///tmp/a%20b.jpg"), Path("/tmp/a b.jpg"))

    def test_excludes_seen_ids_and_images_and_uses_train_prompt(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp:
            tmp_path = Path(temp)
            source = tmp_path / "annotations.json"
            source.write_text("[]", encoding="utf-8")
            train = tmp_path / "train.jsonl"
            train.write_text(
                json.dumps({
                    "id": "train-id",
                    "image": "seen/also-seen.jpg",
                    "prompt": "Caption with exact training prompt",
                    "_teacher_model": "qwen2.5-VL-3b",
                }) + "\n",
                encoding="utf-8",
            )
            image_root = tmp_path / "images"
            for relative in ("seen/also-seen.jpg", "new/id-seen-image.jpg", "new/heldout.jpg"):
                path = image_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"image")

            source_rows = [
                {"id": "train-id", "image": "different.jpg"},
                {"id": "new-id", "image": "seen/also-seen.jpg"},
                {"id": "image-seen-id", "image": "new/id-seen-image.jpg"},
                {"id": "heldout-id", "image": "new/heldout.jpg"},
            ]
            output = tmp_path / "heldout.manifest.jsonl"
            with patch("src.train_VLM.prepare_llava_heldout._iter_json_array", lambda _p: iter(source_rows)):
                build_heldout_manifest(source, train, image_root, output, num_samples=1, seed=7)

            record = json.loads(output.read_text(encoding="utf-8").strip())
            self.assertEqual(record["id"], "heldout-id")
            self.assertEqual(record["source"]["image"], "new/heldout.jpg")
            self.assertEqual(record["messages"][0]["content"][1]["text"], "Caption with exact training prompt")

    def test_path_validation_rejects_parent_traversal(self):
        from src.train_VLM.prepare_llava_heldout import _clean_image_path

        with self.assertRaisesRegex(ValueError, "unsafe image path"):
            _clean_image_path("../outside.jpg")


if __name__ == "__main__":
    unittest.main()
