from types import SimpleNamespace


def _llama_config():
    return SimpleNamespace(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        pretraining_tp=1,
        rope_theta=10000,
        rope_scaling=None,
    )


def test_eagle3_decoder_selects_eager_attention_backend():
    from specforge.modeling.draft.llama3_eagle import (
        LlamaDecoderLayer,
        LlamaEagerAttention,
    )

    layer = LlamaDecoderLayer(_llama_config(), attention_backend="eager")

    assert isinstance(layer.self_attn, LlamaEagerAttention)


def test_eagle3_capabilities_advertise_eager_backend():
    from specforge.algorithms.eagle3.providers import algorithm_spec

    assert "eager" in algorithm_spec().capabilities.attention_backends
