from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_llava_caption_manifest import (
    prepare_manifest,
    safe_relative_image_path,
)


class LLaVACaptionManifestTest(unittest.TestCase):
    def test_normalizes_flat_records_and_preserves_response(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "00001").mkdir()
            (root / "00001" / "one.jpg").write_bytes(b"image")
            source = root / "source.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "id": "one",
                        "image": "00001/one.jpg",
                        "prompt": "Describe <image> briefly.",
                        "response": "the exact response",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "manifest.jsonl"

            metadata = prepare_manifest(
                source,
                output,
                expected_records=1,
                image_root=root,
            )

            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(metadata["records"], 1)
            self.assertEqual(record["response"], "the exact response")
            self.assertEqual(record["messages"][0]["content"][0], {"type": "image"})
            self.assertNotIn("<image>", record["messages"][0]["content"][1]["text"])

    def test_partial_jsonl_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            source.write_text('{"id":"broken"\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                prepare_manifest(source, root / "manifest.jsonl", expected_records=0)

    def test_path_traversal_is_rejected(self):
        for value in ("../escape.jpg", "/absolute.jpg", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                safe_relative_image_path(value)


if __name__ == "__main__":
    unittest.main()
