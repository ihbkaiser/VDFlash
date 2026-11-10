"""Exact distribution metrics for paired full/cut DFlash logits."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


TOP_K_VALUES = (1, 2, 5, 10, 50, 100)
TARGET_HIT_K_VALUES = (1, 2, 5, 10)
TOP_N = 100
_LN2 = math.log(2.0)


def paired_block_metrics(
    full_logits: torch.Tensor,
    cut_logits: torch.Tensor,
    target_ids: list[int] | tuple[int, ...],
) -> list[dict[str, Any]]:
    """Summarize a block of full/cut logits without retaining raw distributions.

    Inputs are raw logits at temperature one. Calculations use float32 even when
    the model itself emits bf16/fp16 logits. Target rank uses competition rank.
    """
    full = torch.as_tensor(full_logits).detach().float()
    cut = torch.as_tensor(cut_logits).detach().float()
    if full.ndim == 1:
        full = full.unsqueeze(0)
    if cut.ndim == 1:
        cut = cut.unsqueeze(0)
    if full.ndim != 2 or full.shape != cut.shape or full.shape[-1] < TOP_N:
        raise ValueError(
            "full/cut logits must have shape [positions, vocabulary] and equal "
            "vocabulary dimensions of at least 100"
        )
    target_ids_tensor = torch.as_tensor(target_ids, dtype=torch.long, device=full.device).flatten()
    positions, vocab_size = full.shape
    if target_ids_tensor.numel() != positions:
        raise ValueError("target_ids must have one ID for each logit position")
    if bool(((target_ids_tensor < 0) | (target_ids_tensor >= vocab_size)).any()):
        raise ValueError("a target token ID is outside the logits vocabulary")

    full_log_probs = F.log_softmax(full, dim=-1)
    cut_log_probs = F.log_softmax(cut, dim=-1)
    full_probs = full_log_probs.exp()
    cut_probs = cut_log_probs.exp()
    full_top_logits, full_top_ids = torch.topk(full, k=TOP_N, dim=-1, sorted=True)
    cut_top_logits, cut_top_ids = torch.topk(cut, k=TOP_N, dim=-1, sorted=True)
    full_top_probs = full_probs.gather(1, full_top_ids)
    cut_top_probs = cut_probs.gather(1, cut_top_ids)
    target_logits_full = full.gather(1, target_ids_tensor[:, None]).squeeze(1)
    target_logits_cut = cut.gather(1, target_ids_tensor[:, None]).squeeze(1)
    target_logp_full = full_log_probs.gather(1, target_ids_tensor[:, None]).squeeze(1)
    target_logp_cut = cut_log_probs.gather(1, target_ids_tensor[:, None]).squeeze(1)
    target_prob_full = full_probs.gather(1, target_ids_tensor[:, None]).squeeze(1)
    target_prob_cut = cut_probs.gather(1, target_ids_tensor[:, None]).squeeze(1)
    rank_full = (full > target_logits_full[:, None]).sum(dim=-1) + 1
    rank_cut = (cut > target_logits_cut[:, None]).sum(dim=-1) + 1
    entropy_full = -torch.where(
        full_probs > 0, full_probs * full_log_probs, torch.zeros_like(full_probs)
    ).sum(dim=-1) / _LN2
    entropy_cut = -torch.where(
        cut_probs > 0, cut_probs * cut_log_probs, torch.zeros_like(cut_probs)
    ).sum(dim=-1) / _LN2
    log_mixture = torch.logaddexp(full_log_probs, cut_log_probs) - _LN2
    full_jsd_term = torch.where(
        full_probs > 0,
        full_probs * (full_log_probs - log_mixture),
        torch.zeros_like(full_probs),
    ).sum(dim=-1)
    cut_jsd_term = torch.where(
        cut_probs > 0,
        cut_probs * (cut_log_probs - log_mixture),
        torch.zeros_like(cut_probs),
    ).sum(dim=-1)
    full_jsd = (0.5 * (full_jsd_term + cut_jsd_term) / _LN2).clamp(0.0, 1.0)

    union_mask = torch.zeros_like(full, dtype=torch.bool)
    union_mask.scatter_(1, full_top_ids, True)
    union_mask.scatter_(1, cut_top_ids, True)
    full_union = full_probs.masked_fill(~union_mask, 0.0)
    cut_union = cut_probs.masked_fill(~union_mask, 0.0)
    full_union_mass = full_union.sum(dim=-1)
    cut_union_mass = cut_union.sum(dim=-1)
    full_head = full_union / full_union_mass[:, None].clamp_min(torch.finfo(full.dtype).tiny)
    cut_head = cut_union / cut_union_mass[:, None].clamp_min(torch.finfo(cut.dtype).tiny)
    head_mix = 0.5 * (full_head + cut_head)
    full_head_term = torch.where(
        full_head > 0,
        full_head * (full_head.clamp_min(torch.finfo(full.dtype).tiny).log() - head_mix.clamp_min(torch.finfo(full.dtype).tiny).log()),
        torch.zeros_like(full_head),
    ).sum(dim=-1)
    cut_head_term = torch.where(
        cut_head > 0,
        cut_head * (cut_head.clamp_min(torch.finfo(cut.dtype).tiny).log() - head_mix.clamp_min(torch.finfo(cut.dtype).tiny).log()),
        torch.zeros_like(cut_head),
    ).sum(dim=-1)
    head_jsd = (0.5 * (full_head_term + cut_head_term) / _LN2).clamp(0.0, 1.0)

    full_cpu_ids = full_top_ids.detach().cpu().tolist()
    cut_cpu_ids = cut_top_ids.detach().cpu().tolist()
    full_cpu_logits = full_top_logits.detach().cpu().tolist()
    cut_cpu_logits = cut_top_logits.detach().cpu().tolist()
    full_cpu_probs = full_top_probs.detach().cpu().tolist()
    cut_cpu_probs = cut_top_probs.detach().cpu().tolist()
    target_cpu_ids = target_ids_tensor.detach().cpu().tolist()
    target_rank_full = rank_full.detach().cpu().tolist()
    target_rank_cut = rank_cut.detach().cpu().tolist()
    target_logp_full_cpu = target_logp_full.detach().cpu().tolist()
    target_logp_cut_cpu = target_logp_cut.detach().cpu().tolist()
    target_prob_full_cpu = target_prob_full.detach().cpu().tolist()
    target_prob_cut_cpu = target_prob_cut.detach().cpu().tolist()
    entropy_full_cpu = entropy_full.detach().cpu().tolist()
    entropy_cut_cpu = entropy_cut.detach().cpu().tolist()
    full_union_mass_cpu = full_union_mass.detach().cpu().tolist()
    cut_union_mass_cpu = cut_union_mass.detach().cpu().tolist()
    full_top1 = full_top_ids[:, 0].detach().cpu().tolist()
    cut_top1 = cut_top_ids[:, 0].detach().cpu().tolist()
    full_margins = (
        (full_top_logits[:, 0] - full_top_logits[:, 1]).detach().cpu().tolist()
    )
    cut_margins = (
        (cut_top_logits[:, 0] - cut_top_logits[:, 1]).detach().cpu().tolist()
    )
    pair_jsd_cpu = full_jsd.detach().cpu().tolist()
    head_jsd_cpu = head_jsd.detach().cpu().tolist()
    union_size_cpu = union_mask.sum(dim=-1).detach().cpu().tolist()

    output: list[dict[str, Any]] = []
    for index, target_id in enumerate(target_cpu_ids):
        full_ids = [int(value) for value in full_cpu_ids[index]]
        cut_ids = [int(value) for value in cut_cpu_ids[index]]
        overlaps: dict[str, dict[str, float | int]] = {}
        for k in TOP_K_VALUES:
            left = set(full_ids[:k])
            right = set(cut_ids[:k])
            intersection = len(left & right)
            overlaps[str(k)] = {
                "intersection_count": intersection,
                "fraction_of_k": intersection / k,
                "jaccard": intersection / max(len(left | right), 1),
            }

        def make_condition(
            *,
            ids: list[int],
            top_logits: list[float],
            top_probs: list[float],
            rank: int,
            log_probability: float,
            target_probability: float,
            entropy: float,
            union_mass: float,
            margin: float,
        ) -> dict[str, Any]:
            return {
                "top1_id": ids[0],
                "top2_id": ids[1],
                "top1_probability": float(top_probs[0]),
                "top2_probability": float(top_probs[1]),
                "top1_logit_margin": float(margin),
                "target_rank": int(rank),
                "target_log_probability_nats": float(log_probability),
                "target_probability": float(target_probability),
                "entropy_bits": float(entropy),
                "union_top100_mass": float(union_mass),
                "tail_mass_outside_union_top100": max(0.0, 1.0 - float(union_mass)),
                **{
                    f"target_hit_top_{k}": bool(target_id in ids[:k])
                    for k in TARGET_HIT_K_VALUES
                },
                "top100_token_ids": ids,
                "top100_logits": [float(value) for value in top_logits],
                "top100_probabilities": [float(value) for value in top_probs],
            }

        output.append(
            {
                "full_metrics": make_condition(
                    ids=full_ids,
                    top_logits=full_cpu_logits[index],
                    top_probs=full_cpu_probs[index],
                    rank=target_rank_full[index],
                    log_probability=target_logp_full_cpu[index],
                    target_probability=target_prob_full_cpu[index],
                    entropy=entropy_full_cpu[index],
                    union_mass=full_union_mass_cpu[index],
                    margin=full_margins[index],
                ),
                "cut_metrics": make_condition(
                    ids=cut_ids,
                    top_logits=cut_cpu_logits[index],
                    top_probs=cut_cpu_probs[index],
                    rank=target_rank_cut[index],
                    log_probability=target_logp_cut_cpu[index],
                    target_probability=target_prob_cut_cpu[index],
                    entropy=entropy_cut_cpu[index],
                    union_mass=cut_union_mass_cpu[index],
                    margin=cut_margins[index],
                ),
                "pair_metrics": {
                    "jsd_bits_full_vocab": float(pair_jsd_cpu[index]),
                    "jsd_bits_head_union_top100": float(head_jsd_cpu[index]),
                    "top1_changed": bool(full_top1[index] != cut_top1[index]),
                    "topk_overlap": overlaps,
                    "union_top100_size": int(union_size_cpu[index]),
                    "full_top100_id_count": len(set(full_ids)),
                    "cut_top100_id_count": len(set(cut_ids)),
                },
            }
        )
    return output


def paired_position_metrics(
    full_logits: torch.Tensor,
    cut_logits: torch.Tensor,
    target_id: int,
) -> dict[str, Any]:
    """Single-position convenience wrapper for :func:`paired_block_metrics`."""
    return paired_block_metrics(full_logits, cut_logits, [int(target_id)])[0]
