"""Native MSD draft transformer for Qwen2.5-VL teacher features."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from specforge.algorithms.msd.model import decouple_msd_inputs

from .registry import register_draft


class MSDConfig(PretrainedConfig):
    """Configuration for the MSD 1/3/5-layer replication sweep."""

    model_type = "msd"

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 2048,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 1,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 2,
        head_dim: Optional[int] = 128,
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1_000_000.0,
        mrope_section: Optional[list[int]] = None,
        attention_dropout: float = 0.0,
        attention_bias: bool = True,
        initializer_range: float = 0.02,
        **kwargs,
    ) -> None:
        architectures = kwargs.pop("architectures", None)
        kwargs.pop("tie_word_embeddings", None)
        super().__init__(
            architectures=architectures or ["MSDDraftModel"],
            tie_word_embeddings=False,
            **kwargs,
        )
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim or hidden_size // num_attention_heads)
        self.max_position_embeddings = int(max_position_embeddings)
        self.rms_norm_eps = float(rms_norm_eps)
        self.rope_theta = float(rope_theta)
        self.mrope_section = list(mrope_section or [16, 24, 24])
        self.attention_dropout = float(attention_dropout)
        self.attention_bias = bool(attention_bias)
        self.initializer_range = float(initializer_range)
        self._validate_replication_shape()

    def _validate_replication_shape(self) -> None:
        if self.num_hidden_layers not in {1, 3, 5}:
            raise ValueError("MSD replication depth must be 1, 3, or 5")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_key_value_heads must divide num_attention_heads")
        if self.num_attention_heads * self.head_dim != self.hidden_size:
            raise ValueError("num_attention_heads * head_dim must equal hidden_size")
        if len(self.mrope_section) != 3:
            raise ValueError("mrope_section must contain temporal, height, width sections")
        if sum(self.mrope_section) * 2 != self.head_dim:
            raise ValueError("twice the mrope_section sum must equal head_dim")

    @classmethod
    def qwen25vl_3b(cls, num_hidden_layers: int) -> "MSDConfig":
        """Return the draft shape paired with Qwen2.5-VL-3B-Instruct."""

        return cls(
            vocab_size=151936,
            hidden_size=2048,
            intermediate_size=11008,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=16,
            num_key_value_heads=2,
            head_dim=128,
            max_position_embeddings=32768,
            rms_norm_eps=1e-6,
            rope_theta=1_000_000.0,
            mrope_section=[16, 24, 24],
            attention_bias=True,
        )


class MSDRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        dtype = hidden_states.dtype
        variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
        normalized = hidden_states.float() * torch.rsqrt(variance + self.eps)
        return (self.weight.float() * normalized).to(dtype)


def _rotate_half(hidden_states: Tensor) -> Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class MSDMultiModalRotaryEmbedding(nn.Module):
    """Qwen2.5-VL temporal/height/width rotary position embedding."""

    def __init__(self, config: MSDConfig) -> None:
        super().__init__()
        self.sections = tuple(config.mrope_section)
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, config.head_dim, 2).float() / config.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, hidden_states: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        if position_ids.ndim != 3 or position_ids.shape[0] != 3:
            raise ValueError("position_ids must have shape [3, batch, sequence]")
        positions = position_ids.to(self.inv_freq.device, torch.float32)
        frequencies = torch.einsum("d,tbs->tbsd", self.inv_freq, positions)
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        chunk_sizes = list(self.sections) * 2
        cos_chunks = embeddings.cos().split(chunk_sizes, dim=-1)
        sin_chunks = embeddings.sin().split(chunk_sizes, dim=-1)
        cos = torch.cat(
            [chunk[index % 3] for index, chunk in enumerate(cos_chunks)], dim=-1
        )
        sin = torch.cat(
            [chunk[index % 3] for index, chunk in enumerate(sin_chunks)], dim=-1
        )
        return cos.to(hidden_states.dtype), sin.to(hidden_states.dtype)


class MSDAttention(nn.Module):
    def __init__(self, config: MSDConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.dropout = config.attention_dropout
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_key_value_heads * self.head_dim
        self.q_proj = nn.Linear(config.hidden_size, q_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, kv_size, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, kv_size, bias=config.attention_bias)
        self.o_proj = nn.Linear(q_size, config.hidden_size, bias=False)
        self.rotary = MSDMultiModalRotaryEmbedding(config)

    def _shape(self, tensor: Tensor, heads: int) -> Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor],
        position_ids: Tensor,
    ) -> Tensor:
        query = self._shape(self.q_proj(hidden_states), self.num_heads)
        key = self._shape(self.k_proj(hidden_states), self.num_key_value_heads)
        value = self._shape(self.v_proj(hidden_states), self.num_key_value_heads)
        cos, sin = self.rotary(hidden_states, position_ids)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        query = query * cos + _rotate_half(query) * sin
        key = key * cos + _rotate_half(key) * sin
        key = key.repeat_interleave(self.num_key_value_groups, dim=1)
        value = value.repeat_interleave(self.num_key_value_groups, dim=1)

        batch, _, length, _ = query.shape
        causal = torch.ones(length, length, dtype=torch.bool, device=query.device).tril()
        allowed = causal.view(1, 1, length, length).expand(batch, 1, -1, -1)
        if attention_mask is not None:
            if attention_mask.shape != (batch, length):
                raise ValueError("attention_mask must have shape [batch, sequence]")
            token_mask = attention_mask.to(torch.bool)
            allowed = allowed & token_mask[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed,
            dropout_p=self.dropout if self.training else 0.0,
            scale=1.0 / math.sqrt(self.head_dim),
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, -1)
        output = self.o_proj(attended)
        if attention_mask is not None:
            output = output * attention_mask.unsqueeze(-1).to(output.dtype)
        return output


class MSDMLP(nn.Module):
    def __init__(self, config: MSDConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class MSDDecoderLayer(nn.Module):
    def __init__(self, config: MSDConfig) -> None:
        super().__init__()
        self.input_layernorm = MSDRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = MSDAttention(config)
        self.post_attention_layernorm = MSDRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.mlp = MSDMLP(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor],
        position_ids: Tensor,
    ) -> Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), attention_mask, position_ids
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


@register_draft
class MSDDraftModel(PreTrainedModel):
    """Trainable MSD draft backbone; target embeddings and LM head stay frozen."""

    config_class = MSDConfig
    _supports_sdpa = True
    base_model_prefix = "msd"
    main_input_name = "conditioning_hidden_state"

    def __init__(self, config: MSDConfig) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.fusion_projection = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.layers = nn.ModuleList(
            [MSDDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.post_init()
        self.embed_tokens.weight.requires_grad_(False)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value
        self.embed_tokens.weight.requires_grad_(False)

    def embed_input_ids(self, input_ids: Tensor) -> Tensor:
        return self.embed_tokens(input_ids)

    def prepare_inputs(
        self,
        conditioning_hidden_state: Tensor,
        next_token_embeddings: Tensor,
        visual_embeddings: Tensor,
        visual_token_mask: Tensor,
    ) -> Tensor:
        return decouple_msd_inputs(
            conditioning_hidden_state,
            next_token_embeddings,
            visual_embeddings,
            visual_token_mask,
            self.fusion_projection,
        )

    def forward(
        self,
        conditioning_hidden_state: Tensor,
        next_token_embeddings: Tensor,
        visual_embeddings: Tensor,
        visual_token_mask: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        **_: object,
    ) -> Tensor:
        hidden_states = self.prepare_inputs(
            conditioning_hidden_state,
            next_token_embeddings,
            visual_embeddings,
            visual_token_mask,
        )
        batch, length, _ = hidden_states.shape
        if position_ids is None:
            base = torch.arange(length, device=hidden_states.device).expand(batch, -1)
            position_ids = base.unsqueeze(0).expand(3, -1, -1)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids)
        return hidden_states


__all__ = ["MSDConfig", "MSDDraftModel", "MSDMultiModalRotaryEmbedding"]
