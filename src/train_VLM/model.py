from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .data import make_dense_attention_mask, make_flex_block_mask


def build_target_layer_ids(num_target_layers: int, num_features: int = 5) -> list[int]:
    """Select layers from the second through the third-to-last layer."""

    if num_features < 1:
        raise ValueError("num_features must be positive")
    if num_target_layers < 4:
        return [max(0, num_target_layers // 2)] * num_features
    start, end = 1, num_target_layers - 3
    if num_features == 1:
        return [start + (end - start) // 2]
    return [int(round(start + i * (end - start) / (num_features - 1))) for i in range(num_features)]


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps).to(dtype=x.dtype)
        return self.weight * x


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class MultiModalRotaryEmbedding(nn.Module):
    """Small self-contained Qwen2-VL style 3-axis rotary embedding."""

    def __init__(self, config: Any):
        super().__init__()
        self.head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
        rope_parameters = getattr(config, "rope_parameters", None)
        if rope_parameters is None:
            rope_parameters = getattr(config, "rope_scaling", None) or {}
        if isinstance(rope_parameters, dict) and "default" in rope_parameters:
            rope_parameters = rope_parameters["default"]
        if not isinstance(rope_parameters, dict):
            rope_parameters = {}
        theta = float(rope_parameters.get("rope_theta", getattr(config, "rope_theta", 1_000_000.0)))
        sections = rope_parameters.get("mrope_section", getattr(config, "mrope_section", None))
        if sections is None:
            sections = [self.head_dim // 2, self.head_dim // 4, self.head_dim - (self.head_dim // 2 + self.head_dim // 4)]
        sections = list(sections)
        if sum(sections) * 2 != self.head_dim:
            sections = [self.head_dim // 2]
        self.mrope_section = tuple(int(x) for x in sections)
        inv_freq = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        if position_ids.ndim != 3 or position_ids.shape[0] != 3:
            raise ValueError("position_ids must have shape [3, batch, length]")
        pos = position_ids.to(device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.einsum("d,abl->abdl", self.inv_freq, pos).transpose(-1, -2)
        emb = torch.cat((freqs, freqs), dim=-1)  # [3, batch, length, head_dim]
        cos, sin = emb.cos(), emb.sin()
        return cos.to(dtype=dtype), sin.to(dtype=dtype)

    def mix_axes(self, values: torch.Tensor) -> torch.Tensor:
        # values: [3, batch, length, head_dim]. Qwen2-VL interleaves the three
        # axes by channel section; text positions have equal values on all axes.
        sections = [2 * s for s in self.mrope_section]
        chunks = values.split(sections, dim=-1)
        return torch.cat([chunk[i % 3] for i, chunk in enumerate(chunks)], dim=-1)

    def apply(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_position_ids: torch.Tensor,
        key_position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qcos, qsin = self(query_position_ids, query.dtype)
        kcos, ksin = self(key_position_ids, key.dtype)
        qcos, qsin = self.mix_axes(qcos), self.mix_axes(qsin)
        kcos, ksin = self.mix_axes(kcos), self.mix_axes(ksin)
        qcos, qsin = qcos.unsqueeze(1), qsin.unsqueeze(1)
        kcos, ksin = kcos.unsqueeze(1), ksin.unsqueeze(1)
        query = (query * qcos) + (_rotate_half(query) * qsin)
        key = (key * kcos) + (_rotate_half(key) * ksin)
        return query, key


def _repeat_kv(hidden_states: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return hidden_states
    return hidden_states.repeat_interleave(groups, dim=1)


class DFlashAttention(nn.Module):
    def __init__(
        self,
        config: Any,
        rotary: MultiModalRotaryEmbedding,
        *,
        compile_flex_attention: bool = True,
    ):
        super().__init__()
        self.config = config
        self.rotary = rotary
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(getattr(config, "num_key_value_heads", self.num_heads))
        self.head_dim = int(getattr(config, "head_dim", self.hidden_size // self.num_heads))
        bias = bool(getattr(config, "attention_bias", True))
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.scale = self.head_dim ** -0.5
        self.compile_flex_attention = compile_flex_attention
        self._compiled_flex_attention = None

    def _flex_attention(self):
        from torch.nn.attention.flex_attention import flex_attention

        if not self.compile_flex_attention:
            return flex_attention
        if self._compiled_flex_attention is None:
            self._compiled_flex_attention = torch.compile(flex_attention, dynamic=False)
        return self._compiled_flex_attention

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_context: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_position_ids: torch.Tensor,
        attention_mask: Any,
        context_kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_context_kv: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, query_len, _ = hidden_states.shape
        context_len = target_context.shape[1]
        q = self.q_proj(hidden_states).view(batch, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_noise = self.k_proj(hidden_states).view(batch, query_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_noise = self.v_proj(hidden_states).view(batch, query_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k_noise = self.rotary.apply(q, k_noise, block_position_ids, block_position_ids)
        cached_context_len = 0
        if context_kv_cache is not None:
            cached_key, cached_value = context_kv_cache
            if cached_key.shape[:2] != (batch, self.num_kv_heads) or cached_value.shape != cached_key.shape:
                raise ValueError("invalid cached DFlash context K/V shape")
            cached_context_len = int(cached_key.shape[2])
            if cached_context_len > context_len:
                raise ValueError("cached DFlash context is longer than the supplied target context")
        if cached_context_len < context_len:
            new_context = target_context[:, cached_context_len:]
            new_positions = context_position_ids[:, :, cached_context_len:]
            new_len = int(new_context.shape[1])
            k_new = self.k_proj(new_context).view(
                batch, new_len, self.num_kv_heads, self.head_dim
            ).transpose(1, 2)
            v_new = self.v_proj(new_context).view(
                batch, new_len, self.num_kv_heads, self.head_dim
            ).transpose(1, 2)
            _, k_new = self.rotary.apply(
                q[:, :, :0], k_new, block_position_ids[:, :, :0], new_positions
            )
            if context_kv_cache is None:
                k_context, v_context = k_new, v_new
            else:
                k_context = torch.cat([cached_key, k_new], dim=2)
                v_context = torch.cat([cached_value, v_new], dim=2)
        else:
            assert context_kv_cache is not None
            k_context, v_context = context_kv_cache
        key = torch.cat([k_context, k_noise], dim=2)
        value = torch.cat([v_context, v_noise], dim=2)

        if hasattr(attention_mask, "kv_num_blocks"):
            output = self._flex_attention()(
                q,
                key,
                value,
                block_mask=attention_mask,
                scale=self.scale,
                enable_gqa=self.num_heads != self.num_kv_heads,
            )
        else:
            if attention_mask.ndim == 4:
                attention_mask = attention_mask.to(dtype=torch.bool)
            key = _repeat_kv(key, self.num_heads // self.num_kv_heads)
            value = _repeat_kv(value, self.num_heads // self.num_kv_heads)
            output = F.scaled_dot_product_attention(
                q, key, value, attn_mask=attention_mask, dropout_p=0.0, scale=self.scale
            )
        output = output.transpose(1, 2).reshape(batch, query_len, self.num_heads * self.head_dim)
        output = self.o_proj(output)
        if return_context_kv:
            return output, (k_context, v_context)
        return output


class DFlashDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Any,
        rotary: MultiModalRotaryEmbedding,
        *,
        compile_flex_attention: bool = True,
    ):
        super().__init__()
        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.input_layernorm = RMSNorm(config.hidden_size, eps)
        self.self_attn = DFlashAttention(
            config, rotary, compile_flex_attention=compile_flex_attention
        )
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps)
        intermediate = int(config.intermediate_size)
        self.gate_proj = nn.Linear(config.hidden_size, intermediate, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, config.hidden_size, bias=False)
        self.hidden_act = getattr(config, "hidden_act", "silu")

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        return_context_kv: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(
            hidden_states=hidden_states,
            return_context_kv=return_context_kv,
            **kwargs,
        )
        if return_context_kv:
            hidden_states, context_kv = attn_output
        else:
            hidden_states = attn_output
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        if self.hidden_act in {"silu", "swish"}:
            hidden_states = F.silu(gate) * up
        elif self.hidden_act == "gelu":
            hidden_states = F.gelu(gate) * up
        else:
            raise ValueError(f"Unsupported hidden_act={self.hidden_act!r}")
        hidden_states = residual + self.down_proj(hidden_states)
        if return_context_kv:
            return hidden_states, context_kv
        return hidden_states


class DFlashVLMModel(nn.Module):
    """Qwen2.5-VL language-side DFlash draft adapter.

    The target embedding and LM head are intentionally external and frozen.
    ``target_context`` is the concatenated selected target hidden states before
    projection, while ``noise_embeddings`` contains the anchor/mask blocks.
    """

    def __init__(
        self,
        text_config: Any,
        *,
        num_draft_layers: int = 5,
        num_target_features: int = 5,
        block_size: int = 16,
        compile_flex_attention: bool = True,
    ):
        super().__init__()
        self.text_config = text_config
        self.hidden_size = int(text_config.hidden_size)
        self.num_target_features = int(num_target_features)
        self.block_size = int(block_size)
        self.compile_flex_attention = bool(compile_flex_attention)
        self.rotary = MultiModalRotaryEmbedding(text_config)
        self.fc = nn.Linear(self.hidden_size * self.num_target_features, self.hidden_size, bias=False)
        self.hidden_norm = RMSNorm(self.hidden_size, float(getattr(text_config, "rms_norm_eps", 1e-6)))
        self.layers = nn.ModuleList(
            [
                DFlashDecoderLayer(
                    text_config,
                    self.rotary,
                    compile_flex_attention=self.compile_flex_attention,
                )
                for _ in range(num_draft_layers)
            ]
        )
        self.norm = RMSNorm(self.hidden_size, float(getattr(text_config, "rms_norm_eps", 1e-6)))
        self.gradient_checkpointing = False
        self._reset_parameters(float(getattr(text_config, "initializer_range", 0.02)))

    def _reset_parameters(self, std: float) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

    def forward(
        self,
        *,
        noise_embeddings: torch.Tensor,
        target_context: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_position_ids: torch.Tensor,
        anchors: torch.Tensor,
        context_original_positions: torch.Tensor,
        use_flex_attention: bool = True,
        draft_context_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        return_draft_context_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        if noise_embeddings.ndim != 3 or target_context.ndim != 3:
            raise ValueError("noise_embeddings and target_context must be [batch, length, hidden]")
        if noise_embeddings.shape[0] != 1 or target_context.shape[0] != 1:
            raise ValueError("the v1 trainer uses batch size one")
        context = self.hidden_norm(self.fc(target_context))
        q_len = noise_embeddings.shape[1]
        if block_position_ids.shape[-1] != q_len:
            raise ValueError("block_position_ids length does not match noise_embeddings")
        if use_flex_attention and noise_embeddings.is_cuda:
            attention_mask = make_flex_block_mask(
                anchors,
                context_original_positions,
                block_size=self.block_size,
                device=noise_embeddings.device,
                compile_mask=self.compile_flex_attention,
            )
        else:
            attention_mask = make_dense_attention_mask(
                anchors, context_original_positions, block_size=self.block_size
            ).to(device=noise_embeddings.device)
        hidden_states = noise_embeddings
        next_context_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        if draft_context_cache is not None and len(draft_context_cache) != len(self.layers):
            raise ValueError("draft_context_cache must contain one K/V entry per draft layer")
        for layer_index, layer in enumerate(self.layers):
            layer_cache = draft_context_cache[layer_index] if draft_context_cache is not None else None
            if self.training and self.gradient_checkpointing and not return_draft_context_cache:
                def layer_forward(
                    hidden: torch.Tensor,
                    context_arg: torch.Tensor,
                    context_pos_arg: torch.Tensor,
                    block_pos_arg: torch.Tensor,
                    layer_arg: DFlashDecoderLayer = layer,
                ) -> torch.Tensor:
                    return layer_arg(
                        hidden_states=hidden,
                        target_context=context_arg,
                        context_position_ids=context_pos_arg,
                        block_position_ids=block_pos_arg,
                        attention_mask=attention_mask,
                    )

                hidden_states = checkpoint(
                    layer_forward,
                    hidden_states,
                    context,
                    context_position_ids,
                    block_position_ids,
                    use_reentrant=False,
                )
            else:
                layer_output = layer(
                    hidden_states=hidden_states,
                    target_context=context,
                    context_position_ids=context_position_ids,
                    block_position_ids=block_position_ids,
                    attention_mask=attention_mask,
                    context_kv_cache=layer_cache,
                    return_context_kv=return_draft_context_cache,
                )
                if return_draft_context_cache:
                    hidden_states, context_kv = layer_output
                    next_context_cache.append(context_kv)
                else:
                    hidden_states = layer_output
        hidden_states = self.norm(hidden_states)
        if return_draft_context_cache:
            return hidden_states, next_context_cache
        return hidden_states

    def draft_logits(self, hidden_states: torch.Tensor, lm_head: nn.Module) -> torch.Tensor:
        return lm_head(hidden_states)
