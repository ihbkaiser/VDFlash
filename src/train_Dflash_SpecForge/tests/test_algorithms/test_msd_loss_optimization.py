from __future__ import annotations

import torch

from specforge.algorithms.msd.model import msd_loss


class RecordingHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]],
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.input_shapes: list[tuple[int, ...]] = []

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.input_shapes.append(tuple(hidden_states.shape))
        return torch.nn.functional.linear(hidden_states, self.weight)


def test_msd_loss_projects_only_nonzero_loss_mask_tokens() -> None:
    predicted = torch.tensor(
        [
            [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
            [[4.0, 5.0], [5.0, 6.0], [6.0, 7.0]],
        ],
        requires_grad=True,
    )
    target = predicted.detach() + 0.25
    mask = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0]])
    head = RecordingHead()

    result = msd_loss(predicted, target, mask, head)
    result.loss.backward()

    assert head.input_shapes == [(2, 2), (2, 2)]
    assert predicted.grad is not None
    assert torch.count_nonzero(predicted.grad[mask == 0]).item() == 0
