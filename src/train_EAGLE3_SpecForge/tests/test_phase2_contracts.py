import sys
from pathlib import Path

import torch


PACKAGE = Path(__file__).resolve().parents[1]
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))


def test_eagle3_qwen25vl_offline_contract_requires_positions():
    from specforge.algorithms.contracts import FeatureMode
    from specforge.algorithms.eagle3.providers import algorithm_spec

    spec = algorithm_spec()
    contract = spec.feature_contract(FeatureMode.OFFLINE, "qwen2_5_vl")

    assert "position_ids" in contract.required_tensors
    assert "position_ids" in contract.storage.required_tensors


def test_eagle3_text_contract_remains_text_only():
    from specforge.algorithms.contracts import FeatureMode
    from specforge.algorithms.eagle3.providers import algorithm_spec

    spec = algorithm_spec()
    contract = spec.feature_contract(FeatureMode.OFFLINE, "text")

    assert "position_ids" in contract.optional_tensors
    assert "position_ids" not in contract.storage.required_tensors


def test_qwen25vl_normalizer_preserves_three_axis_positions():
    from specforge.algorithms.eagle3.data import normalize_offline_sample

    raw = {
        "input_ids": torch.tensor([10, 11, 12]),
        "loss_mask": torch.tensor([0.0, 1.0, 0.0]),
        "aux_hidden_state": torch.zeros(1, 3, 12),
        "hidden_state": torch.zeros(1, 3, 4),
        "position_ids": torch.arange(9, dtype=torch.int32).view(3, 3),
    }

    normalized = normalize_offline_sample(raw, max_len=3)

    assert tuple(normalized["position_ids"].shape) == (3, 1, 3)
    torch.testing.assert_close(normalized["position_ids"][:, 0], raw["position_ids"])


def test_qwen25vl_reader_loads_position_ids_from_feature_files(tmp_path):
    from specforge.algorithms.eagle3.providers import algorithm_providers

    providers = algorithm_providers()
    text_reader = providers.offline_for("text").build_reader(
        str(tmp_path), run_id="text", ttt_length=1, max_len=8
    )
    qwen_reader = providers.offline_for("qwen2_5_vl").build_reader(
        str(tmp_path), run_id="qwen", ttt_length=1, max_len=8
    )

    assert "position_ids" not in text_reader.feature_keys
    assert "position_ids" in qwen_reader.feature_keys


def test_eagle3_capture_layers_are_exactly_three_distinct_target_layers():
    from types import SimpleNamespace

    import pytest

    from specforge.algorithms.model_providers import resolve_eagle_capture_layers

    config = SimpleNamespace(model=SimpleNamespace(aux_hidden_state_layer_ids=None))
    draft = {"eagle_config": {"eagle_aux_hidden_state_layer_ids": [1, 8, 20]}}
    target = SimpleNamespace(num_hidden_layers=28)
    assert resolve_eagle_capture_layers(config, draft, target) == [1, 8, 20]

    draft["eagle_config"]["eagle_aux_hidden_state_layer_ids"] = [1, 1, 20]
    with pytest.raises(ValueError, match="three different"):
        resolve_eagle_capture_layers(config, draft, target)

    draft["eagle_config"]["eagle_aux_hidden_state_layer_ids"] = [1, 8, 28]
    with pytest.raises(ValueError, match="outside"):
        resolve_eagle_capture_layers(config, draft, target)
