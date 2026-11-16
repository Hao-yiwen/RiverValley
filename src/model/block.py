# -*- coding: utf-8 -*-
"""
Transformer Block 模块

包含:
- Block: Transformer 基础块（自注意力 + 前馈网络）

结构图 (Pre-LayerNorm + Residual):

┌───────────────────────────────┐
│      Transformer Block        │
└───────────────────────────────┘

        Input (x)
            │
            ├───────────────────────┐
            │                       │
            ▼                       │
      ┌──────────┐                  │
      │  RMSNorm │ (ln1)            │
      └────┬─────┘                  │
           │                        │
           ▼                        │
    ┌─────────────────┐             │
    │ CausalSelfAttn  │             │ (Residual)
    │   (sa)          │             │
    └────────┬────────┘             │
             │                      │
             ▼                      │
           (+) ◄────────────────────┘
            │
            ├───────────────────────┐
            │                       │
            ▼                       │
      ┌──────────┐                  │
      │  RMSNorm │ (ln2)            │
      └────┬─────┘                  │
           │                        │
           ▼                        │
    ┌─────────────────┐             │
    │   FeedForward   │             │ (Residual)
    │  or MoELayer    │             │
    └────────┬────────┘             │
             │                      │
             ▼                      │
           (+) ◄────────────────────┘
            │
            ▼
        Output (x)

"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from config import RiverValleyConfig
from model.attention import CausalSelfAttention
from model.feedforward import FeedForward
from model.moe import MoELayer
from model.embeddings import RMSNorm


class Block(nn.Module):
    """
    Transformer Block: 自注意力 + 前馈网络

    使用 Pre-LayerNorm 结构（先 LayerNorm 再计算）和残差连接
    支持 MoE（Mixture of Experts）
    """

    def __init__(self, config: RiverValleyConfig, layer_idx: int):
        super().__init__()
        self.use_moe = config.use_moe and (layer_idx % config.moe_freq == 1)
        self.layer_idx = layer_idx

        self.sa = CausalSelfAttention(config)
        if self.use_moe:
            self.ff = MoELayer(config)
        else:
            self.ff = FeedForward(config)
        self.ln1 = RMSNorm(config.n_embed, eps=config.rms_norm_eps)
        self.ln2 = RMSNorm(config.n_embed, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Union[
        Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
    ]:
        h, new_kv_cache = self.sa(self.ln1(x), attention_mask, start_pos, kv_cache)
        x = x + h

        if self.use_moe:
            ff_out, aux_loss = self.ff(self.ln2(x))
            x = x + ff_out
            return x, new_kv_cache, aux_loss
        else:
            x = x + self.ff(self.ln2(x))
            return x, new_kv_cache
