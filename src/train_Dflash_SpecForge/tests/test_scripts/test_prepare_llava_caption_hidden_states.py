"""Tests for batched LLaVA hidden-state preparation helpers."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from scripts import prepare_llava_caption_hidden_states as llava_capture

_collate_prepared = llava_capture._collate_prepared
parse_args = llava_capture.parse_args


def _prepared(length: int, offset: int) -> dict:
    input_ids = torch.arange(offset, offset + length).reshape(1, length)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "loss_mask": torch.ones_like(input_ids, dtype=torch.float32),
        "position_ids": torch.arange(3 * length).reshape(3, 1, length),
        "multimodal_inputs": {"pixel_values": torch.tensor([offset])},
    }


class CollatePreparedTest(unittest.TestCase):
    def test_right_pads_variable_length_multimodal_samples(self):
        first = _prepared(3, 10)
        second = _prepared(5, 20)

        batch, media, lengths = _collate_prepared(
            [first, second],
            pad_token_id=99,
        )

        self.assertEqual(lengths, [3, 5])
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 5))
        self.assertEqual(tuple(batch["position_ids"].shape), (3, 2, 5))
        self.assertTrue(torch.equal(batch["input_ids"][0, 3:], torch.tensor([99, 99])))
        self.assertTrue(torch.equal(batch["attention_mask"][0, 3:], torch.zeros(2)))
        self.assertTrue(torch.equal(batch["loss_mask"][0, 3:], torch.zeros(2)))
        self.assertIs(media[0], first["multimodal_inputs"])
        self.assertIs(media[1], second["multimodal_inputs"])


class ParseArgsTest(unittest.TestCase):
    def test_accepts_sglang_attention_backend(self):
        argv = [
            "prepare_llava_caption_hidden_states.py",
            "--target-model-path",
            "target",
            "--draft-model-config",
            "draft.json",
            "--manifest",
            "manifest.jsonl",
            "--image-root",
            "images",
            "--output-path",
            "features",
            "--sglang-attention-backend",
            "triton",
        ]
        with patch("sys.argv", argv):
            args = parse_args()

        self.assertEqual(args.sglang_attention_backend, "triton")

    def test_defaults_avoid_all_flashinfer_backends(self):
        argv = [
            "prepare_llava_caption_hidden_states.py",
            "--target-model-path",
            "target",
            "--draft-model-config",
            "draft.json",
            "--manifest",
            "manifest.jsonl",
            "--image-root",
            "images",
            "--output-path",
            "features",
        ]
        with patch("sys.argv", argv):
            args = parse_args()

        self.assertEqual(args.sglang_attention_backend, "triton")
        self.assertEqual(args.sglang_sampling_backend, "pytorch")
        self.assertEqual(args.sglang_mm_attention_backend, "sdpa")
        self.assertEqual(
            llava_capture._offline_capture_kwargs(args),
            {
                "attention_backend": "triton",
                "sampling_backend": "pytorch",
                "mm_attention_backend": "sdpa",
                "mem_fraction_static": 0.4,
            },
        )


if __name__ == "__main__":
    unittest.main()
