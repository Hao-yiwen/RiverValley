# -*- coding: utf-8 -*-
"""
Mixture of Experts (MoE) 模块

包含:
- Router: Top-K 路由器
- MoELayer: 混合专家层（带共享专家）
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from config import RiverValleyConfig
from model.feedforward import FeedForward


class Router(nn.Module):
    """Top-K 路由器，带负载均衡损失"""

    def __init__(self, n_embd: int, num_experts: int, moe_top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.moe_top_k = moe_top_k
        self.gate = nn.Linear(n_embd, num_experts, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        x_flat = x.view(-1, C)

        logits = self.gate(x_flat)
        probs = F.softmax(logits, dim=-1)

        weights, indices = torch.topk(probs, self.moe_top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        mask = F.one_hot(indices, self.num_experts).sum(dim=1)
        f = mask.float().mean(dim=0)
        P = probs.mean(dim=0)
        aux_loss = self.num_experts * (f * P).sum()

        return weights, indices, aux_loss


class MoELayer(nn.Module):
    """
    Mixture of Experts 层 (带共享专家)
    """

    def __init__(self, config: RiverValleyConfig):
        super().__init__()
        n_embd = config.n_embed
        num_experts = config.num_experts
        num_shared_experts = config.num_shared_experts
        moe_top_k = config.moe_top_k

        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.moe_top_k = moe_top_k

        self.experts = nn.ModuleList([FeedForward(config) for _ in range(num_experts)])
        self.router = Router(n_embd, num_experts, moe_top_k)

        if num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config) for _ in range(num_shared_experts)
            ])
        else:
            self.shared_experts = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        x_flat = x.view(-1, C)

        shared_output = torch.zeros_like(x_flat)
        if self.shared_experts is not None:
            for shared_expert in self.shared_experts:
                shared_output += shared_expert(x_flat)

        weights, indices, aux_loss = self.router(x)

        routed_output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (indices == i).any(dim=-1)
            if mask.any():
                expert_input = x_flat[mask]
                expert_output = expert(expert_input)

                expert_weights = torch.where(
                    indices == i,
                    weights,
                    torch.zeros_like(weights)
                )
                expert_weights = expert_weights.sum(dim=-1, keepdim=True)[mask]

                routed_output[mask] += expert_weights * expert_output

        output = shared_output + routed_output

        return output.view(B, T, C), aux_loss
