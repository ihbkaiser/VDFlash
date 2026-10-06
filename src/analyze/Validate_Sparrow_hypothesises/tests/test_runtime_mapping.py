from __future__ import annotations

from types import SimpleNamespace

import torch

from src.analyze.Validate_Sparrow_hypothesises import runtime


def test_model_parallel_map_can_place_vision_on_second_gpu(monkeypatch):
    monkeypatch.setenv("MSD_MODEL_PARALLEL_VISUAL_DEVICE", "1")
    monkeypatch.setenv("MSD_MODEL_PARALLEL_SPLIT_LAYER", "10")
    monkeypatch.setattr(
        "transformers.AutoConfig.from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(num_hidden_layers=28),
    )

    mapping = runtime._model_parallel_device_map("unused")

    assert mapping["visual"] == 1
    assert mapping["model.layers.9"] == 0
    assert mapping["model.layers.10"] == 1


def test_align_msd_draft_inputs_moves_all_draft_tensors_to_eagle_device():
    hidden, input_ids, token = runtime.align_msd_draft_inputs(
        torch.ones(1, 2, 3),
        torch.ones(1, 2, dtype=torch.long),
        torch.ones(1, 1, dtype=torch.long),
        torch.device("cpu"),
    )

    assert hidden.device.type == "cpu"
    assert input_ids.device.type == "cpu"
    assert token.device.type == "cpu"


def test_msd_eagle_device_follows_base_embedding_device():
    base = SimpleNamespace(model=SimpleNamespace(embed_tokens=torch.nn.Embedding(4, 2)))
    model = SimpleNamespace(base_model=base)

    assert runtime.msd_eagle_device(model).type == "cpu"


def test_place_msd_eagle_rebuilds_tree_state_after_move():
    class FakeEagle:
        def __init__(self):
            self.to_device = None
            self.init_count = 0

        def to(self, device):
            self.to_device = device
            return self

        def init_tree(self):
            self.init_count += 1

    base = SimpleNamespace(model=SimpleNamespace(embed_tokens=torch.nn.Embedding(4, 2)))
    model = SimpleNamespace(base_model=base, ea_layer=FakeEagle())

    device = runtime.place_msd_eagle(model)

    assert device.type == "cpu"
    assert model.ea_layer.to_device == device
    assert model.ea_layer.init_count == 1


def test_model_parallel_hook_moves_decoder_auxiliary_tensors():
    class FakeLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.empty(1, device="meta"))

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            cache_position=None,
            position_embeddings=None,
            **kwargs,
        ):
            devices = {
                "hidden": hidden_states.device,
                "mask": attention_mask.device,
                "position_ids": position_ids.device,
                "cache_position": cache_position.device,
                "cos": position_embeddings[0].device,
                "sin": position_embeddings[1].device,
            }
            return devices

    layer = FakeLayer()
    base = SimpleNamespace(model=SimpleNamespace(layers=[layer]))
    runtime.install_msd_model_parallel_transfer_hooks(base)

    output = layer(
        torch.ones(1, 2, 3),
        attention_mask=torch.ones(1, 1, 2, 2),
        position_ids=torch.ones(3, 1, 2, dtype=torch.long),
        cache_position=torch.ones(2, dtype=torch.long),
        position_embeddings=(torch.ones(1), torch.ones(1)),
    )

    assert all(device.type == "meta" for device in output.values())
