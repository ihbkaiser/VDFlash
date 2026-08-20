import json
from pathlib import Path


def test_qwen25vl_3b_eagle3_config_has_phase1_contract():
    path = Path(__file__).parents[1] / "configs" / "qwen2.5-vl-3b-eagle3.json"
    payload = json.loads(path.read_text())
    assert payload["architectures"] == ["LlamaForCausalLMEagle3"]
    assert payload["target_model_type"] == "qwen2_5_vl"
    assert payload["num_hidden_layers"] == 1
    assert payload["vocab_size"] == 151936
    assert payload["draft_vocab_size"] == 32000
    assert payload["hidden_size"] == 2048
    assert payload["rope_scaling"]["mrope_section"] == [16, 24, 24]


def test_qwen25vl_7b_eagle3_config_has_phase1_contract():
    path = Path(__file__).parents[1] / "configs" / "qwen2.5-vl-7b-eagle3.json"
    payload = json.loads(path.read_text())
    assert payload["architectures"] == ["LlamaForCausalLMEagle3"]
    assert payload["target_model_type"] == "qwen2_5_vl"
    assert payload["num_hidden_layers"] == 1
    assert payload["vocab_size"] == 152064
    assert payload["draft_vocab_size"] == 32000
    assert payload["hidden_size"] == 3584
    assert payload["rope_scaling"]["mrope_section"] == [16, 24, 24]
