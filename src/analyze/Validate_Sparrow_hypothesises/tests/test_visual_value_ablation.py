from types import SimpleNamespace

import torch

from src.analyze.Validate_Sparrow_hypothesises.runtime import zero_msd_draft_visual_values


def test_zero_msd_draft_visual_values_zeros_visual_prefill_rows_and_restores_forward():
    projection = torch.nn.Linear(2, 2, bias=False)
    projection.weight.data.copy_(torch.eye(2))
    model = SimpleNamespace(
        ea_layer=SimpleNamespace(
            layers=[SimpleNamespace(self_attn=SimpleNamespace(v_proj=projection))]
        )
    )
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
    original_forward = projection.forward

    with zero_msd_draft_visual_values(model, [1, 3]):
        result = projection(hidden)
        assert torch.equal(result[:, [0, 2]], hidden[:, [0, 2]])
        assert torch.count_nonzero(result[:, [1, 3]]) == 0

    assert projection.forward == original_forward
    assert torch.equal(projection(hidden), hidden)
