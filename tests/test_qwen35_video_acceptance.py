from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.analyze.Whether_they_are_appliable_for_dDrafter.qwen35_dflash_video_decode import (
    AcceptanceRound,
    DecodeResult,
    Qwen35DFlashDecoder,
    normalize_video_inputs,
    sha256_tokens,
    split_video_grid_for_groups,
    text_anchor_positions,
)
from src.analyze.qwen35_video_acceptance import (
    config_hash,
    sample_frame_indices,
)


def test_text_anchor_removes_whole_vision_span():
    ids = torch.tensor([1, 2, 248053, 248057, 27, 248057, 248054, 3, 4])
    keep = text_anchor_positions(ids, 248053, 248054)
    assert keep.tolist() == [0, 1, 7, 8]


def test_text_anchor_without_vision_keeps_everything():
    ids = torch.tensor([1, 2, 3])
    keep = text_anchor_positions(ids, 248053, 248054)
    assert keep.tolist() == [0, 1, 2]


def test_split_video_grid_matches_groups():
    mm = torch.tensor([[0, 0, 2, 2, 0, 2, 2, 0]])
    grid = torch.tensor([[2, 14, 14]])
    fixed = split_video_grid_for_groups(mm, grid)
    assert fixed.tolist() == [[1, 14, 14], [1, 14, 14]]


def test_split_video_grid_passthrough_when_rows_match_groups():
    mm = torch.tensor([[0, 2, 2, 2, 2, 0]])
    grid = torch.tensor([[1, 14, 14]])
    assert torch.equal(split_video_grid_for_groups(mm, grid), grid)


def test_normalize_video_inputs():
    mm = torch.tensor([[0, 2, 2, 0, 2, 2, 0]])
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7]]),
        "mm_token_type_ids": mm,
        "video_grid_thw": torch.tensor([[2, 14, 14]]),
    }
    normalized = normalize_video_inputs(inputs)
    assert normalized["video_grid_thw"].tolist() == [[1, 14, 14], [1, 14, 14]]
    # original untouched
    assert inputs["video_grid_thw"].tolist() == [[2, 14, 14]]


def test_sample_frame_indices_uniform_and_monotonic():
    indices = sample_frame_indices(
        1000, 100.0, experiment_type="natural", requested_count=None
    )
    assert len(indices) == 100
    assert indices == sorted(indices)
    assert indices[0] == 0
    assert indices[-1] == 999


def test_sample_frame_indices_natural_cap():
    indices = sample_frame_indices(
        5000, 3000.0, experiment_type="natural", requested_count=None
    )
    assert len(indices) == 1020
    assert indices == sorted(indices)


def test_sample_frame_indices_controlled_counts():
    for count in (16, 122, 532, 1020):
        indices = sample_frame_indices(
            90000, 3600.0, experiment_type="controlled", requested_count=count
        )
        assert len(indices) == count
        assert indices == sorted(indices)


def test_config_hash_deterministic_and_mode_independent():
    kwargs = dict(
        seed=42,
        target_model="Qwen/Qwen3.5-4B",
        draft_model="z-lab/Qwen3.5-4B-DFlash",
        context_mode="full",
        verify_mode="exact",
        max_new_tokens=256,
        temperature=0.0,
        block_size=16,
        budgets=[392, 2989, 13034, 24990],
        dataset_id="lmms-eval/Video-MME",
    )
    first = config_hash(**kwargs)
    assert first == config_hash(**{**kwargs, "context_mode": "text_anchor"})
    assert first != config_hash(**{**kwargs, "verify_mode": "block"})


def test_sha256_tokens_stable():
    assert sha256_tokens([1, 2, 3]) == sha256_tokens([1, 2, 3])
    assert sha256_tokens([1, 2, 3]) != sha256_tokens([1, 2, 4])


def test_tau_metrics_exclude_terminal_and_partial_rounds():
    result = DecodeResult(
        output_ids=torch.zeros(1, 20, dtype=torch.long),
        num_input_tokens=10,
        acceptance_rounds=[
            AcceptanceRound(0, 15, 5, 6, False, False, False),
            AcceptanceRound(1, 15, 7, 8, False, False, False),
            AcceptanceRound(2, 15, 3, 4, False, True, True),
            AcceptanceRound(3, 4, 2, 3, True, True, False),
        ],
    )
    assert result.tau_proposal() == pytest.approx(6.0)
    assert result.tau_effective() == pytest.approx(7.0)
    assert result.full_block_rate() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Decoder loop with stub target/draft
# ---------------------------------------------------------------------------

class StubCache:
    def __init__(self, length: int):
        self.length = length

    def get_seq_length(self):
        return self.length


class StubEmbeddings(torch.nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.hidden = hidden

    def forward(self, input_ids):
        return input_ids.float().unsqueeze(-1).expand(-1, -1, self.hidden)


class StubHead(torch.nn.Module):
    def __init__(self, vocab: int = 64):
        super().__init__()
        self.vocab = vocab

    def forward(self, hidden):
        token_ids = hidden[..., 0].long().clamp(max=self.vocab - 2)
        logits = torch.full(
            (*hidden.shape[:-1], self.vocab), -10.0, dtype=hidden.dtype
        )
        logits.scatter_(-1, (token_ids + 1).unsqueeze(-1), 10.0)
        return logits


class StubTarget:
    def __init__(self, vocab: int = 64, hidden: int = 32):
        self.vocab = vocab
        self.hidden = hidden
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=hidden),
            video_token_id=57,
            vision_start_token_id=53,
            vision_end_token_id=54,
        )
        self.embeddings = StubEmbeddings(hidden)
        self.lm_head = StubHead(vocab)
        self.rope_deltas = torch.zeros(1, 1, dtype=torch.long)
        self.model = SimpleNamespace(
            rope_deltas=self.rope_deltas,
            get_rope_index=lambda ids, **kwargs: (
                torch.arange(ids.shape[1]).view(1, 1, -1).expand(3, 1, -1),
                torch.zeros(1, 1, dtype=torch.long),
            ),
        )

    def eval(self):
        return self

    def train(self, mode=True):
        return self

    def get_input_embeddings(self):
        return self.embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        cache_position=None,
        use_cache=True,
        logits_to_keep=0,
        output_hidden_states=False,
        output_attentions=False,
        return_dict=True,
        **kwargs,
    ):
        del attention_mask, position_ids, past_key_values, cache_position
        del use_cache, output_attentions, kwargs
        hidden = self.embeddings(input_ids)
        logits = self.lm_head(hidden)
        if isinstance(logits_to_keep, int) and logits_to_keep >= 0:
            if logits_to_keep == 0:
                logits = logits
            else:
                logits = logits[:, -logits_to_keep:, :]
        return SimpleNamespace(
            logits=logits,
            past_key_values=StubCache(input_ids.shape[1]),
            hidden_states=(hidden, hidden),
        )

    def generate(self, **kwargs):
        ids = kwargs["input_ids"]
        prompt_len = ids.shape[1]
        max_new_tokens = kwargs["max_new_tokens"]
        generated = [ids[0, -1].item() + 1]
        token = generated[0]
        while len(generated) < max_new_tokens:
            token = token + 1
            generated.append(token)
        full = torch.cat(
            [ids[0], torch.tensor(generated, dtype=torch.long)], dim=0
        ).view(1, -1)
        return SimpleNamespace(sequences=full)


class StubDraft:
    block_size = 16
    mask_token_id = 60
    target_layer_ids = [0]

    def eval(self):
        return self

    def train(self, mode=True):
        return self

    def __call__(
        self,
        target_hidden,
        noise_embedding,
        position_ids,
        past_key_values=None,
        use_cache=False,
        is_causal=False,
    ):
        del target_hidden, position_ids, past_key_values, use_cache, is_causal
        return noise_embedding


def _run_stub_decode(*, verify_mode: str, max_new_tokens: int = 6):
    torch.manual_seed(0)
    target = StubTarget()
    draft = StubDraft()
    decoder = Qwen35DFlashDecoder(
        target,
        draft,
        device=torch.device("cpu"),
        context_mode="full",
        verify_mode=verify_mode,
        block_size=16,
        stop_token_ids=[99],
    )
    prompt = torch.tensor([[10, 11, 12]])
    inputs = {
        "input_ids": prompt,
        "attention_mask": torch.ones_like(prompt),
        "mm_token_type_ids": torch.zeros_like(prompt),
    }
    result = decoder.decode(inputs, max_new_tokens=max_new_tokens)
    return decoder, result


@pytest.mark.parametrize("verify_mode", ["exact", "block"])
def test_stub_decoder_output_matches_stub_greedy(verify_mode):
    decoder, result = _run_stub_decode(verify_mode=verify_mode)
    inputs = {
        "input_ids": torch.tensor([[10, 11, 12]]),
        "attention_mask": torch.ones(1, 3),
        "mm_token_type_ids": torch.zeros(1, 3),
    }
    greedy, _ = decoder.greedy_reference(inputs, max_new_tokens=6)
    spec_ids = result.output_ids[0, result.num_input_tokens :].tolist()
    assert spec_ids == greedy[0].tolist()
    assert result.acceptance_rounds
    # 6 tokens -> every round is partial, so the main metric stays undefined
    assert result.tau_effective() is None


def test_stub_decoder_round_accounting():
    _, result = _run_stub_decode(verify_mode="exact", max_new_tokens=20)
    assert result.acceptance_rounds[0].proposal_count == 15
    assert result.acceptance_rounds[0].matched_proposals == 0
    assert result.acceptance_rounds[0].effective_emitted_tokens == 1
    assert result.acceptance_rounds[-1].is_partial_block
    assert result.acceptance_rounds[-1].is_terminal
    assert result.tau_effective() == pytest.approx(1.0)


def test_text_anchor_draft_context_shorter():
    torch.manual_seed(0)
    target = StubTarget()
    draft = StubDraft()
    decoder = Qwen35DFlashDecoder(
        target,
        draft,
        device=torch.device("cpu"),
        context_mode="text_anchor",
        verify_mode="exact",
        stop_token_ids=[99],
    )
    prompt = torch.tensor([[10, 11, 53, 57, 57, 54, 12]])
    inputs = {
        "input_ids": prompt,
        "attention_mask": torch.ones_like(prompt),
        "mm_token_type_ids": (prompt == 57).long(),
    }
    result = decoder.decode(inputs, max_new_tokens=4)
    # vision span 53..54 removed -> 4 text positions kept
    assert result.output_ids.shape[1] > prompt.shape[1]
