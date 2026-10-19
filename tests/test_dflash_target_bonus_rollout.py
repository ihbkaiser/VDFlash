from types import SimpleNamespace

import torch

from src.infer.dflash_free_running import FreeRunningDFlashDecoder


class _FixedDraft(torch.nn.Module):
    block_size = 4
    mask_token_id = 31

    def forward(self, *, target_hidden, noise_embedding, **kwargs):
        del kwargs
        length = target_hidden.shape[1] + noise_embedding.shape[1]
        return torch.zeros((1, length, 4), device=noise_embedding.device)

    def _sample_draft_tokens(self, target, draft_hidden, block):
        del target, block
        return torch.tensor([[11, 12, 13]], device=draft_hidden.device)


class _RecordingTarget:
    def __init__(self):
        self.embedding = torch.nn.Embedding(32, 4)
        self.lm_head = torch.nn.Linear(4, 32, bias=False)
        self.forward_inputs = []

    def get_input_embeddings(self):
        return self.embedding


class _FixedTargetBonusDecoder(FreeRunningDFlashDecoder):
    def _target_forward_with_context(self, kwargs):
        input_ids = kwargs["input_ids"]
        self.target.forward_inputs.append(input_ids.detach().clone())
        logits = torch.full((1, 1, 32), -10.0, device=input_ids.device)
        logits[..., 7] = 10.0
        output = SimpleNamespace(past_key_values=object(), logits=logits)
        context = torch.zeros((1, input_ids.shape[1], 4), device=input_ids.device)
        return output, context


def test_target_bonus_follows_full_draft_block_and_becomes_next_anchor():
    target = _RecordingTarget()
    decoder = _FixedTargetBonusDecoder(
        target, _FixedDraft(), device=torch.device("cpu")
    )

    result = decoder.decode(
        input_ids=torch.tensor([[1, 2, 3]]),
        position_ids=torch.tensor([[0, 1, 2]]),
        target_kwargs=None,
        max_new_tokens=7,
        stop_token_ids=[31],
        target_bonus_per_block=True,
    )

    assert result.output_ids[0, 3:].tolist() == [11, 12, 13, 7, 11, 12, 13]
    assert result.target_bonus_tokens == 1
    assert result.blocks[0]["target_bonus_token_id"] == 7
    assert result.blocks[0]["emitted_token_ids"] == [11, 12, 13, 7]
    assert result.blocks[1]["anchor_token_id"] == 7
    assert result.blocks[1]["target_bonus_token_id"] is None
    assert target.forward_inputs[1][0].tolist() == [3, 11, 12, 13]
