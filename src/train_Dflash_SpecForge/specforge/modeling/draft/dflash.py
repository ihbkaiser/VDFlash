import time
from typing import Any, Callable, Optional

import torch
from torch import nn
from transformers import DynamicCache
from transformers.cache_utils import Cache
try:
    from transformers.integrations.flex_attention import compile_friendly_flex_attention
except ImportError:
    try:
        from transformers.integrations.flex_attention import flex_attention_forward as compile_friendly_flex_attention
    except ImportError:
        compile_friendly_flex_attention = None
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3Config,
    Qwen3PreTrainedModel,
    Qwen3RotaryEmbedding,
    eager_attention_forward,
    rotate_half,
)
from typing_extensions import Tuple, Unpack

from .dflash_kernels import DEFAULT_DFLASH_KERNELS, DFlashKernels
from .flex_attention_backend import flex_attention_backend
from .registry import register_draft

FULL_ATTENTION = "full_attention"
SLIDING_ATTENTION = "sliding_attention"
_VALID_DFLASH_LAYER_TYPES = {FULL_ATTENTION, SLIDING_ATTENTION}


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size)
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


def _cache_tensor_bytes(cache: Any) -> int:
    """Best-effort resident K/V tensor size for common Transformers cache APIs."""

    seen: set[int] = set()
    total = 0

    def visit(value: Any) -> None:
        nonlocal total
        if torch.is_tensor(value):
            if id(value) not in seen:
                seen.add(id(value))
                total += int(value.numel() * value.element_size())
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    for name in ("key_cache", "value_cache"):
        visit(getattr(cache, name, None))
    for layer in getattr(cache, "layers", ()) or ():
        for name in ("keys", "values", "key_cache", "value_cache"):
            visit(getattr(layer, name, None))
    return total


def resolve_dflash_attention_layout(
    config: Qwen3Config,
) -> tuple[tuple[str, ...], Optional[int]]:
    """Validate and return the configured per-layer DFlash attention layout."""

    num_hidden_layers = config.num_hidden_layers
    layer_types = tuple(config.layer_types)

    if len(layer_types) != num_hidden_layers:
        raise ValueError(
            "DFlash config.layer_types must contain exactly "
            f"num_hidden_layers={num_hidden_layers} entries, got "
            f"{len(layer_types)}"
        )
    invalid = set(layer_types) - _VALID_DFLASH_LAYER_TYPES
    if invalid:
        raise ValueError(
            "DFlash config.layer_types supports only full_attention and "
            f"sliding_attention, got {sorted(invalid)}"
        )

    if SLIDING_ATTENTION not in layer_types:
        return layer_types, None

    sliding_window = config.sliding_window
    if sliding_window is None or sliding_window <= 0:
        raise ValueError(
            "DFlash sliding_attention layers require use_sliding_window=true "
            "and a positive config.sliding_window"
        )
    return layer_types, sliding_window


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen25VLMultiModalRotaryEmbedding(nn.Module):
    """Qwen2.5-VL three-axis rotary embedding for the DFlash backbone.

    The DFlash parameter layout remains Qwen3-compatible.  Only the position
    encoding is selected by ``dflash_config.position_encoding`` so Phase 1
    checkpoints can be warm-started without adding or reshaping parameters.
    """

    def __init__(self, config: Qwen3Config, dflash_config: dict):
        super().__init__()
        self.head_dim = int(
            getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        )
        rope_parameters = getattr(config, "rope_parameters", None)
        if rope_parameters is None:
            rope_parameters = getattr(config, "rope_scaling", None) or {}
        if isinstance(rope_parameters, dict) and "default" in rope_parameters:
            rope_parameters = rope_parameters["default"]
        if not isinstance(rope_parameters, dict):
            rope_parameters = {}
        theta = float(
            dflash_config.get(
                "rope_theta",
                rope_parameters.get("rope_theta", getattr(config, "rope_theta", 1_000_000.0)),
            )
        )
        sections = dflash_config.get(
            "mrope_section",
            rope_parameters.get("mrope_section", getattr(config, "mrope_section", None)),
        )
        if sections is None:
            sections = [
                self.head_dim // 2,
                self.head_dim // 4,
                self.head_dim - (self.head_dim // 2 + self.head_dim // 4),
            ]
        sections = [int(value) for value in sections]
        if len(sections) != 3 or sum(sections) * 2 != self.head_dim:
            raise ValueError(
                "Qwen2.5-VL mrope_section must contain three sections whose "
                f"doubled sum equals head_dim={self.head_dim}, got {sections}"
            )
        self.mrope_section = tuple(sections)
        inv_freq = 1.0 / (
            theta
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor):
        output_dtype = hidden_states.dtype
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        if position_ids.ndim != 3 or position_ids.shape[0] != 3:
            raise ValueError(
                "Qwen2.5-VL position_ids must have shape [3, batch, length], "
                f"got {tuple(position_ids.shape)}"
            )
        positions = position_ids.to(device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.einsum("d,abl->abdl", self.inv_freq, positions).transpose(-1, -2)
        embeddings = torch.cat((freqs, freqs), dim=-1)
        cos, sin = embeddings.cos(), embeddings.sin()
        sections = list(self.mrope_section) * 2
        chunks_cos = cos.split(sections, dim=-1)
        chunks_sin = sin.split(sections, dim=-1)
        mixed_cos = torch.cat(
            [chunk[index % 3] for index, chunk in enumerate(chunks_cos)], dim=-1
        )
        mixed_sin = torch.cat(
            [chunk[index % 3] for index, chunk in enumerate(chunks_sin)], dim=-1
        )
        # Keep RoPE numerics in FP32, then restore the activation dtype.  If
        # FP32 cos/sin escape this module, applying them promotes q/k while v
        # remains BF16 and both SDPA and FlexAttention receive mixed dtypes.
        return mixed_cos.to(dtype=output_dtype), mixed_sin.to(dtype=output_dtype)


def _prepare_dflash_eager_mask(
    attention_mask: Optional[torch.Tensor],
    dtype: torch.dtype,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Convert a boolean allow-mask to eager's additive representation."""

    if attention_mask is None or attention_mask.dtype != torch.bool:
        return attention_mask, None

    valid_queries = attention_mask.any(dim=-1, keepdim=True)
    # A finite minimum keeps eager softmax stable. Fully masked query rows are
    # explicitly zeroed after attention so they cannot average forbidden values.
    additive_mask = torch.zeros_like(attention_mask, dtype=dtype)
    additive_mask.masked_fill_(~attention_mask, torch.finfo(dtype).min)
    return additive_mask, valid_queries


class Qwen3DFlashAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        kernels: DFlashKernels,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        if config._attn_implementation == "flex_attention":
            assert (
                config.attention_dropout == 0.0
            ), "DFlash FlexAttention requires attention_dropout=0.0"
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = kernels.make_rms_norm(self.head_dim, config.rms_norm_eps)
        self.k_norm = kernels.make_rms_norm(self.head_dim, config.rms_norm_eps)
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == SLIDING_ATTENTION
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        v = torch.cat([v_ctx, v_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        # Keep the cache dense and gather only the keys/values used by this
        # attention call. This lets the verifier change its selected context
        # between rounds without rebuilding or losing the full DFlash KV bank.
        context_keep_indices = getattr(self, "context_keep_indices", None)
        if context_keep_indices is not None:
            context_length = int(k.shape[-2]) - int(q_len)
            indices = context_keep_indices.to(device=k.device, dtype=torch.long)
            if indices.ndim != 1 or indices.numel() == 0:
                raise ValueError("context_keep_indices must be a non-empty vector")
            context_keys = k[..., :context_length, :].index_select(-2, indices)
            context_values = v[..., :context_length, :].index_select(-2, indices)
            noise_keys = k[..., context_length:, :]
            noise_values = v[..., context_length:, :]
            k = torch.cat((context_keys, noise_keys), dim=-2)
            v = torch.cat((context_values, noise_values), dim=-2)
            if attention_mask is not None:
                if not torch.is_tensor(attention_mask) or attention_mask.ndim < 3:
                    raise ValueError(
                        "selected context requires a tensor attention mask with a key axis"
                    )
                if attention_mask.shape[-1] != context_length + q_len:
                    raise ValueError(
                        "attention mask key length does not match the cached context and draft block"
                    )
                mask_context = attention_mask[..., :context_length].index_select(
                    -1, indices.to(device=attention_mask.device)
                )
                attention_mask = torch.cat(
                    (mask_context, attention_mask[..., context_length:]), dim=-1
                )
        valid_queries = None
        if self.config._attn_implementation == "flex_attention":
            kernel_options = dict(kwargs.pop("kernel_options", None) or {})
            backend = flex_attention_backend()
            if backend is not None:
                kernel_options["BACKEND"] = backend

            attn_output = compile_friendly_flex_attention(
                q,
                k,
                v,
                block_mask=attention_mask,
                enable_gqa=True,
                scale=self.scaling,
                kernel_options=kernel_options or None,
            ).transpose(1, 2)
            attn_weights = None
        else:
            attn_fn: Callable = eager_attention_forward
            if self.config._attn_implementation == "eager":
                attention_mask, valid_queries = _prepare_dflash_eager_mask(
                    attention_mask,
                    q.dtype,
                )
            else:
                attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
            attn_output, attn_weights = attn_fn(
                self,
                q,
                k,
                v,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )
            if valid_queries is not None and attn_weights is not None:
                attn_weights = attn_weights.masked_fill(~valid_queries, 0)
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        if valid_queries is not None:
            attn_output = attn_output.masked_fill(
                ~valid_queries.any(dim=1),
                0,
            )
        return attn_output, attn_weights


class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        kernels: DFlashKernels,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DFlashAttention(
            config=config,
            layer_idx=layer_idx,
            kernels=kernels,
        )
        self.mlp = kernels.make_mlp(config)
        self.input_layernorm = kernels.make_rms_norm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_attention_layernorm = kernels.make_rms_norm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    target_layer_ids = [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]
    return target_layer_ids


def extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: Optional[list[int]],
) -> torch.Tensor:
    offset = 1
    selected_states = []
    for layer_id in layer_ids:
        selected_states.append(hidden_states[layer_id + offset])
    target_hidden = torch.cat(selected_states, dim=-1)
    return target_hidden


def normalize_draft_head_checkpoint_keys(*args, **kwargs):
    """Map checkpoint-only nested head names onto the direct module layout.

    Early Domino/DSpark checkpoints saved their auxiliary heads beneath a
    ``logit_head`` container. The live architecture no longer owns that wrapper,
    but those tensors remain valid and must not be dropped during warm start or
    full resume.
    """
    if len(args) == 8:
        state_dict, prefix = args[1], args[2]
    elif len(args) >= 2:
        state_dict, prefix = args[0], args[1]
    else:
        state_dict = kwargs.get("state_dict", {})
        prefix = kwargs.get("prefix", "")
    checkpoint_prefixes = (
        ("logit_head.prefix_gru.", "prefix_gru."),
        ("logit_head.embed_proj.", "embed_proj."),
        ("logit_head.markov_head.", "markov_head."),
        ("logit_head.confidence_head.", "confidence_head."),
    )
    for key in list(state_dict):
        if not key.startswith(prefix):
            continue
        local_key = key[len(prefix) :]
        for checkpoint_prefix, model_prefix in checkpoint_prefixes:
            if not local_key.startswith(checkpoint_prefix):
                continue
            normalized_key = prefix + model_prefix + local_key[len(checkpoint_prefix) :]
            if normalized_key not in state_dict:
                state_dict[normalized_key] = state_dict[key]
            state_dict.pop(key)
            break


@register_draft
class DFlashDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]

    def __init__(
        self,
        config,
        dflash_kernels: Optional[DFlashKernels] = None,
    ) -> None:
        super().__init__(config)
        self.config = config
        self.layer_types, self.sliding_window = resolve_dflash_attention_layout(config)
        kernels = dflash_kernels or DEFAULT_DFLASH_KERNELS
        self.layers = nn.ModuleList(
            [
                Qwen3DFlashDecoderLayer(config, layer_idx, kernels)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        dflash_config = getattr(config, "dflash_config", {}) or {}
        self.target_layer_ids = dflash_config.get(
            "target_layer_ids",
            build_target_layer_ids(config.num_target_layers, config.num_hidden_layers),
        )
        self.norm = kernels.make_rms_norm(config.hidden_size, config.rms_norm_eps)
        if dflash_config.get("position_encoding") == "qwen2_5_vl_mrope":
            self.rotary_emb = Qwen25VLMultiModalRotaryEmbedding(config, dflash_config)
        else:
            self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = kernels.make_rms_norm(
            config.hidden_size, config.rms_norm_eps
        )
        self.block_size = config.block_size
        self.mask_token_id = dflash_config.get("mask_token_id", None)
        self.projector_type = dflash_config.get("projector_type", None)
        self.pure_draft_prefix_len = dflash_config.get("pure_draft_prefix_len", 0)
        self.shift_label = dflash_config.get("shift_label", False)
        # Draft families can add checkpointed logits/confidence heads without
        # duplicating the shared DFlash backbone constructor. The base DFlash
        # implementation is a no-op; DSpark overrides this hook.
        self._init_draft_head(config, dflash_config)
        hook_fn = getattr(self, "register_load_state_dict_pre_hook", getattr(self, "_register_load_state_dict_pre_hook", None))
        if hook_fn is not None:
            hook_fn(normalize_draft_head_checkpoint_keys)
        self.post_init()

    def _init_draft_head(self, config, dflash_config: dict) -> None:
        del config, dflash_config

    def apply_logits_head(
        self,
        base_logits: torch.Tensor,
        *,
        prev_token_ids: Optional[torch.Tensor] = None,
        prev_token_embeddings: Optional[torch.Tensor] = None,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        del prev_token_ids, prev_token_embeddings, hidden_states
        return base_logits

    def apply_markov_logits(
        self,
        base_logits: torch.Tensor,
        *,
        prev_token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.apply_logits_head(
            base_logits,
            prev_token_ids=prev_token_ids,
            hidden_states=hidden_states,
        )

    def predict_confidence(
        self,
        hidden_states: torch.Tensor,
        *,
        prev_token_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        del hidden_states, prev_token_ids
        return None

    def _sample_draft_tokens(
        self,
        target: nn.Module,
        draft_hidden: torch.Tensor,
        block_output_ids: torch.LongTensor,
    ) -> torch.LongTensor:
        """Sample one speculative block from the draft-model hidden states.

        DFlash predicts the whole suffix in one LM-head call. Draft families
        with an auxiliary logits head can override this boundary without
        duplicating the target-cache and acceptance logic in ``spec_generate``.
        """
        draft_logits = self._compute_draft_logits(
            target,
            draft_hidden,
            block_output_ids,
        )
        return sample(draft_logits)

    def _compute_draft_logits(
        self,
        target: nn.Module,
        draft_hidden: torch.Tensor,
        block_output_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """Return the raw vocabulary logits for one draft block."""
        del block_output_ids
        lm_head = getattr(target, "lm_head", None)
        if lm_head is None:
            lm_head = target.get_output_embeddings()
        return lm_head(draft_hidden[:, -self.block_size + 1 :, :])

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[object] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer_type, layer in zip(self.layer_types, self.layers):
            layer_attention_mask = (
                attention_mask[layer_type]
                if isinstance(attention_mask, dict)
                else attention_mask
            )
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=layer_attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)

    @torch.inference_mode()
    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
        *,
        position_ids: Optional[torch.LongTensor] = None,
        target_kwargs: Optional[dict] = None,
        visual_retention: Optional[float] = None,
        visual_selector: str = "temporal_spatial",
        profile_timing: bool = False,
    ):
        self.eval()
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens
        target_device = next(target.parameters()).device
        total_started = time.perf_counter()

        def synchronize_target_device() -> None:
            if profile_timing and target_device.type == "cuda":
                torch.cuda.synchronize(target_device)

        def start_device_timer():
            if profile_timing and target_device.type == "cuda":
                event = torch.cuda.Event(enable_timing=True)
                event.record(torch.cuda.current_stream(target_device))
                return event
            return time.perf_counter()

        def stop_device_timer(started):
            if profile_timing and target_device.type == "cuda":
                ended = torch.cuda.Event(enable_timing=True)
                ended.record(torch.cuda.current_stream(target_device))
                return started, ended
            return time.perf_counter() - started

        def timer_seconds(value):
            if isinstance(value, tuple):
                return value[0].elapsed_time(value[1]) / 1000.0
            return float(value)

        target_prefill_time = None
        visual_selection_time = 0.0
        draft_timers = []
        verifier_timers = []

        block_size = self.block_size
        if input_ids.shape[0] != 1:
            raise ValueError("DFlash HF inference currently supports batch size one")
        output_ids = torch.full(
            (1, max_length + block_size),
            self.mask_token_id,
            dtype=torch.long,
            device=target_device,
        )
        if position_ids is None:
            position_ids = torch.arange(
                output_ids.shape[1], device=target_device
            ).unsqueeze(0)
        else:
            position_ids = position_ids.to(device=target_device, dtype=torch.long)
            if position_ids.ndim == 2:
                if position_ids.shape != input_ids.shape:
                    raise ValueError("2-D position_ids must match input_ids")
                continuation = torch.arange(
                    position_ids[:, -1].item() + 1,
                    position_ids[:, -1].item() + 1 + output_ids.shape[1] - num_input_tokens,
                    device=target_device,
                ).view(1, -1)
                position_ids = torch.cat([position_ids, continuation], dim=1)
            elif position_ids.ndim == 3 and position_ids.shape[0] == 3:
                if position_ids.shape[1:] != input_ids.shape:
                    raise ValueError(
                        "3-D position_ids must have shape [3, 1, input_length]"
                    )
                starts = position_ids[:, :, -1:] + 1
                offsets = torch.arange(
                    output_ids.shape[1] - num_input_tokens,
                    device=target_device,
                ).view(1, 1, -1)
                continuation = starts + offsets
                position_ids = torch.cat([position_ids, continuation], dim=-1)
            else:
                raise ValueError(
                    "position_ids must have shape [1, input_length] or "
                    f"[3, 1, input_length], got {tuple(position_ids.shape)}"
                )

        past_key_values_target = DynamicCache()
        past_key_values_draft = DynamicCache()
        target_kv_bytes_peak = 0
        draft_kv_bytes_peak = 0

        target_input_embeddings = getattr(target, "get_input_embeddings", None)
        if callable(target_input_embeddings):
            target_embed_tokens = target_input_embeddings()
        else:
            target_embed_tokens = target.model.embed_tokens

        def forward_target(**kwargs):
            try:
                return target(**kwargs)
            except TypeError as exc:
                if "logits_to_keep" not in str(exc):
                    raise
                kwargs.pop("logits_to_keep", None)
                return target(**kwargs)

        # Prefill stage
        prefill_kwargs = dict(target_kwargs or {})
        prefill_kwargs.update(
            {
                "input_ids": input_ids,
                "position_ids": position_ids[..., :num_input_tokens],
                "past_key_values": past_key_values_target,
                "use_cache": True,
                "logits_to_keep": 1,
                "output_hidden_states": True,
            }
        )
        prefill_started = start_device_timer()
        output = forward_target(**prefill_kwargs)
        target_kv_bytes_peak = max(
            target_kv_bytes_peak, _cache_tensor_bytes(past_key_values_target)
        )

        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(
            output.logits[:, -1:, :], temperature
        )
        target_hidden = extract_context_feature(
            output.hidden_states, self.target_layer_ids
        )
        target_prefill_timer = stop_device_timer(prefill_started)
        draft_prompt_positions = None
        retained_context_count = num_input_tokens
        visual_context_counts = None
        if visual_retention is not None:
            selection_started = time.perf_counter()
            from specforge.algorithms.common.dflash_self_distill import (
                build_visual_context_keep_mask,
            )

            visual_token_ids = set()
            for config in (
                getattr(target, "config", None),
                getattr(getattr(target, "config", None), "text_config", None),
            ):
                if config is None:
                    continue
                for name in (
                    "image_token_id",
                    "image_token_index",
                    "video_token_id",
                    "video_token_index",
                ):
                    value = getattr(config, name, None)
                    if value is not None:
                        visual_token_ids.add(int(value))
            prompt_positions = position_ids[..., :num_input_tokens]
            context_keep_mask = build_visual_context_keep_mask(
                input_ids,
                prompt_positions,
                visual_token_ids=sorted(visual_token_ids),
                retention=visual_retention,
                selector=visual_selector,
            )
            draft_prompt_positions = torch.nonzero(
                context_keep_mask[0], as_tuple=False
            ).flatten()
            retained_context_count = int(draft_prompt_positions.numel())
            target_visual_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for token_id in visual_token_ids:
                target_visual_mask |= input_ids == token_id
            visual_total = int(target_visual_mask.sum().item())
            visual_retained = int(
                (target_visual_mask & context_keep_mask).sum().item()
            )
            visual_context_counts = {
                "visual_tokens_total": visual_total,
                "visual_tokens_retained": visual_retained,
                "visual_retention_actual": (
                    visual_retained / visual_total if visual_total else 1.0
                ),
            }
            target_hidden = target_hidden.index_select(
                1, draft_prompt_positions.to(target_hidden.device)
            )
            draft_prompt_positions = draft_prompt_positions.to(position_ids.device)
            visual_selection_time = time.perf_counter() - selection_started

        # Decode stage
        acceptance_lengths = []
        accepted_draft_lengths = []
        full_accepted_draft_lengths = []
        proposal_capacities = []
        target_hidden_position_start = None
        start = input_ids.shape[1]
        while start < max_length:
            if max_length - start <= 1:
                break
            round_start = start
            proposal_capacity = min(block_size - 1, max_length - round_start - 1)
            block_output_ids = output_ids[:, start : start + block_size].clone()
            block_position_ids = position_ids[..., start : start + block_size]
            noise_embedding = target_embed_tokens(block_output_ids)
            if draft_prompt_positions is not None and target_hidden_position_start is None:
                draft_context_positions = position_ids[..., :num_input_tokens].index_select(
                    -1, draft_prompt_positions
                )
                draft_context_hidden = target_hidden
            elif draft_prompt_positions is not None:
                context_start = int(target_hidden_position_start)
                context_end = context_start + target_hidden.shape[1]
                if context_end != start:
                    raise RuntimeError(
                        "verified target features do not end at the next draft anchor: "
                        f"{context_end} != {start}"
                    )
                draft_context_positions = position_ids[..., context_start:context_end]
                draft_context_hidden = target_hidden
            else:
                draft_context_positions = None
                draft_context_hidden = target_hidden
            if draft_context_positions is not None:
                draft_position_ids = torch.cat(
                    (
                        draft_context_positions,
                        position_ids[..., start : start + block_size],
                    ),
                    dim=-1,
                )
            else:
                draft_position_ids = position_ids[
                    ...,
                    past_key_values_draft.get_seq_length() : start + block_size,
                ]
            draft_started = start_device_timer()
            draft_hidden = self(
                target_hidden=draft_context_hidden,
                noise_embedding=noise_embedding,
                position_ids=draft_position_ids,
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )
            draft_kv_bytes_peak = max(
                draft_kv_bytes_peak, _cache_tensor_bytes(past_key_values_draft)
            )
            draft_cache_length = (
                retained_context_count + start - num_input_tokens
                if draft_prompt_positions is not None
                else start
            )
            past_key_values_draft.crop(draft_cache_length)
            block_output_ids[:, 1:] = self._sample_draft_tokens(
                target,
                draft_hidden,
                block_output_ids,
            )
            draft_timers.append(stop_device_timer(draft_started))

            verifier_started = start_device_timer()
            output = forward_target(
                input_ids=block_output_ids,
                position_ids=block_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True,
            )
            target_kv_bytes_peak = max(
                target_kv_bytes_peak, _cache_tensor_bytes(past_key_values_target)
            )

            posterior = sample(output.logits, temperature)
            acceptance_length = (
                (block_output_ids[:, 1:] == posterior[:, :-1])
                .cumprod(dim=1)
                .sum(dim=1)[0]
                .item()
            )
            verifier_timers.append(stop_device_timer(verifier_started))
            output_ids[:, start : start + acceptance_length + 1] = block_output_ids[
                :, : acceptance_length + 1
            ]
            output_ids[:, start + acceptance_length + 1] = posterior[
                :, acceptance_length
            ]
            start += acceptance_length + 1
            past_key_values_target.crop(start)
            target_hidden = extract_context_feature(
                output.hidden_states, self.target_layer_ids
            )[:, : acceptance_length + 1, :]
            accepted_in_output = min(acceptance_length, proposal_capacity)
            bonus_fits = round_start + accepted_in_output + 1 < max_length
            acceptance_lengths.append(accepted_in_output + int(bonus_fits))
            accepted_draft_lengths.append(accepted_in_output)
            full_accepted_draft_lengths.append(acceptance_length)
            proposal_capacities.append(proposal_capacity)
            target_hidden_position_start = start - (acceptance_length + 1)
            if stop_token_ids is not None and any(
                stop_token_id in output_ids[:, num_input_tokens:]
                for stop_token_id in stop_token_ids
            ):
                break
        output_ids = output_ids[:, :max_length]
        output_ids = output_ids[:, output_ids[0] != self.mask_token_id]
        if stop_token_ids is not None:
            stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
            stop_token_indices = torch.isin(
                output_ids[0][num_input_tokens:], stop_token_ids
            ).nonzero(as_tuple=True)[0]
            if stop_token_indices.numel() > 0:
                output_ids = output_ids[
                    :, : num_input_tokens + stop_token_indices[0] + 1
                ]

        mean_accepted = (
            sum(accepted_draft_lengths) / len(accepted_draft_lengths)
            if accepted_draft_lengths
            else 0.0
        )
        mean_effective = (
            sum(acceptance_lengths) / len(acceptance_lengths)
            if acceptance_lengths
            else 0.0
        )
        if profile_timing:
            synchronize_target_device()
            target_prefill_time = timer_seconds(target_prefill_timer)
        draft_forward_time = (
            sum(timer_seconds(timer) for timer in draft_timers)
            if profile_timing
            else None
        )
        target_verifier_time = (
            sum(timer_seconds(timer) for timer in verifier_timers)
            if profile_timing
            else None
        )
        self.last_speculative_metrics = {
            "rounds": len(accepted_draft_lengths),
            "accepted_draft_tokens_per_round": accepted_draft_lengths,
            "full_accepted_draft_tokens_per_round": full_accepted_draft_lengths,
            "proposal_capacity_per_round": proposal_capacities,
            "effective_tokens_per_round": acceptance_lengths,
            "tau_proposal": mean_accepted,
            "tau_effective": mean_effective,
            "target_prefill_time_s": target_prefill_time,
            "visual_selection_time_s": visual_selection_time,
            "draft_forward_time_s": draft_forward_time,
            "target_verifier_time_s": target_verifier_time,
            "speculative_total_time_s": (
                time.perf_counter() - total_started if profile_timing else None
            ),
            "target_kv_bytes_peak": target_kv_bytes_peak,
            "draft_kv_bytes_peak": draft_kv_bytes_peak,
            **(visual_context_counts or {}),
        }

        return output_ids
