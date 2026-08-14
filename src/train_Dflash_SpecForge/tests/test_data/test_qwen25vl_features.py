from __future__ import annotations

import types
import unittest

import torch

from specforge.algorithms.common.dflash_family_data import (
    build_qwen25vl_collator,
    normalize_qwen25vl_offline_sample,
)
from specforge.qwen25vl import compute_qwen25vl_position_ids


class Qwen25VLFeatureTest(unittest.TestCase):
    def test_normalizer_accepts_topology_and_promotes_int32_token_ids(self):
        raw = {
            "input_ids": torch.arange(5, dtype=torch.int32),
            "loss_mask": torch.tensor([0, 0, 1, 1, 1], dtype=torch.float32),
            "hidden_states": torch.zeros(5, 8),
            "position_ids": torch.arange(15, dtype=torch.int32).reshape(3, 5),
        }
        normalized = normalize_qwen25vl_offline_sample(
            raw,
            max_len=5,
            ttt_length=7,
            use_usp_preprocess=False,
        )
        self.assertEqual(normalized["input_ids"].dtype, torch.long)

    def test_normalizer_requires_and_preserves_three_axis_positions(self):
        raw = {
            "input_ids": torch.arange(5),
            "loss_mask": torch.tensor([0, 0, 1, 1, 1], dtype=torch.float32),
            "hidden_states": torch.zeros(5, 8),
            "position_ids": torch.arange(15).reshape(3, 5),
        }
        normalized = normalize_qwen25vl_offline_sample(raw, max_len=5)
        self.assertEqual(tuple(normalized["position_ids"].shape), (3, 1, 5))
        with self.assertRaises(KeyError):
            normalize_qwen25vl_offline_sample({k: v for k, v in raw.items() if k != "position_ids"}, 5)

    def test_collator_batches_position_axes_on_batch_dimension(self):
        def feature(length):
            return {
                "input_ids": torch.ones(1, length, dtype=torch.long),
                "loss_mask": torch.ones(1, length),
                "hidden_states": torch.ones(1, length, 4),
                "position_ids": torch.ones(3, 1, length, dtype=torch.long),
            }

        batch = build_qwen25vl_collator()([feature(4), feature(2)])
        self.assertEqual(tuple(batch["position_ids"].shape), (3, 2, 4))
        self.assertEqual(tuple(batch["hidden_states"].shape), (2, 4, 4))

    def test_position_builder_assigns_visual_and_text_axes(self):
        config = types.SimpleNamespace(
            image_token_id=99,
            vision_start_token_id=98,
            vision_config=types.SimpleNamespace(
                spatial_merge_size=2,
                tokens_per_second=25,
            ),
        )
        ids = torch.tensor([[10, 98, 99, 99, 99, 99, 11, 12]])
        positions = compute_qwen25vl_position_ids(
            ids,
            image_grid_thw=torch.tensor([[1, 4, 4]]),
            config=config,
        )
        self.assertEqual(tuple(positions.shape), (3, 1, 8))
        self.assertEqual(positions[:, 0, 2].tolist(), [2, 2, 2])
        self.assertEqual(positions[:, 0, 5].tolist(), [2, 3, 3])
        self.assertEqual(positions[:, 0, 6].tolist(), [4, 4, 4])


if __name__ == "__main__":
    unittest.main()
