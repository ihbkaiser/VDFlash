"""Tests for batched LLaVA hidden-state preparation helpers."""

from __future__ import annotations

import unittest

import torch

from scripts.prepare_llava_caption_hidden_states import _collate_prepared


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


if __name__ == "__main__":
    unittest.main()
