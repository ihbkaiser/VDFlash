from __future__ import annotations

from dataclasses import dataclass
import os

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    """Small torchrun context shared by offline caching and draft training."""

    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    initialized_here: bool = False

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    def close(self) -> None:
        if self.initialized_here and dist.is_initialized():
            dist.destroy_process_group()


def initialize_distributed(device: str, backend: str = "nccl") -> DistributedContext:
    """Initialize from torchrun environment variables, or return a local context."""

    requested_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    initialized_here = False
    if requested_world_size > 1 and not dist.is_initialized():
        selected_backend = backend
        if selected_backend == "nccl" and not torch.cuda.is_available():
            raise RuntimeError("NCCL distributed execution requires CUDA")
        dist.init_process_group(backend=selected_backend, init_method="env://")
        initialized_here = True

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    requested = torch.device(device)
    if requested.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        selected_device = torch.device("cuda", local_rank if world_size > 1 else requested.index or 0)
        torch.cuda.set_device(selected_device)
    else:
        if world_size > 1 and backend == "nccl":
            raise RuntimeError("NCCL cannot train on a non-CUDA device; use --distributed-backend gloo")
        selected_device = requested
    return DistributedContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=selected_device,
        initialized_here=initialized_here,
    )
