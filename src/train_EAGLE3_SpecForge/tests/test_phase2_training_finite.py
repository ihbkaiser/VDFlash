import sys
from pathlib import Path

import pytest
import torch


PACKAGE = Path(__file__).resolve().parents[1]
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))


def test_eagle3_optimizer_rejects_nonfinite_gradient_norm():
    from specforge.optimizer import BF16Optimizer

    model = torch.nn.Linear(2, 2, bias=False)
    optimizer = BF16Optimizer(
        model,
        lr=1e-3,
        total_steps=1,
        warmup_ratio=0.0,
    )
    model.weight.grad = torch.full_like(model.weight, float("nan"))

    with pytest.raises(FloatingPointError, match="non-finite gradient norm"):
        optimizer.step()
