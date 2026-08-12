from types import SimpleNamespace
import unittest

from specforge.offline_capture.sglang_backend.capture_hooks import (
    configure_capture_layers,
)


class _Qwen25VLTarget:
    def __init__(self):
        self.config = SimpleNamespace(model_type="qwen2_5_vl")
        self.capture_aux_hidden_states = False
        self.model = SimpleNamespace(
            config=SimpleNamespace(num_hidden_layers=28),
            layers_to_capture=[],
        )


class CaptureHookTests(unittest.TestCase):
    def test_qwen25vl_dflash_uses_text_decoder_fallback(self):
        model = _Qwen25VLTarget()

        mode = configure_capture_layers(
            model,
            [1, 7, 13, 19, 25],
            capture_method="dflash",
        )

        self.assertEqual(mode, "qwen2_5_vl_text")
        self.assertTrue(model.capture_aux_hidden_states)
        self.assertEqual(model.model.layers_to_capture, [2, 8, 14, 20, 26])

    def test_native_hook_remains_authoritative(self):
        calls = []
        model = SimpleNamespace(
            set_dflash_layers_to_capture=lambda values: calls.append(values)
        )

        mode = configure_capture_layers(
            model,
            [1, 3],
            capture_method="dflash",
        )

        self.assertEqual(mode, "native")
        self.assertEqual(calls, [[1, 3]])

    def test_fallback_rejects_last_layer(self):
        model = _Qwen25VLTarget()

        with self.assertRaisesRegex(ValueError, "following decoder boundary"):
            configure_capture_layers(
                model,
                [27],
                capture_method="dflash",
            )


if __name__ == "__main__":
    unittest.main()
