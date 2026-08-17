"""Unit tests for the lazy public offline-capture wrapper."""

from __future__ import annotations

import unittest

import torch

from specforge.offline_capture.sglang import OfflineSGLangCapture


class _VariableLengthBackend:
    def capture(self, *, input_ids, attention_mask, loss_mask, **kwargs):
        del kwargs
        data = [
            (
                input_ids[index : index + 1],
                attention_mask[index : index + 1],
                loss_mask[index : index + 1],
            )
            for index in range(input_ids.shape[0])
        ]
        aux = (torch.ones(4, 6), torch.full((2, 6), 2.0))
        last = (torch.ones(4, 3), torch.full((2, 3), 2.0))
        return data, aux, last


class OfflineSGLangWrapperTest(unittest.TestCase):
    def test_variable_length_states_are_padded_to_one_batch(self):
        capture = OfflineSGLangCapture(_VariableLengthBackend())
        input_ids = torch.arange(8).reshape(2, 4)
        attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
        output = capture.capture(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=attention_mask.float(),
        )

        self.assertEqual(tuple(output.hidden_states.shape), (2, 4, 6))
        self.assertEqual(tuple(output.last_hidden_states.shape), (2, 4, 3))
        self.assertTrue(torch.equal(output.hidden_states[1, 2:], torch.zeros(2, 6)))
        self.assertTrue(
            torch.equal(output.last_hidden_states[1, 2:], torch.zeros(2, 3))
        )


if __name__ == "__main__":
    unittest.main()
