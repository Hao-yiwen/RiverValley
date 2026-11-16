# -*- coding: utf-8 -*-
"""
前馈网络模块

包含:
- FeedForward: SwiGLU 激活的前馈网络
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from config import RiverValleyConfig


class FeedForward(nn.Module):
    """前馈网络：SwiGLU 激活"""

    def __init__(self, config: RiverValleyConfig):
        super().__init__()
        n_embd = config.n_embed
        # 使用配置的 intermediate_size，如果未设置则用 8/3 * n_embd
        if config.intermediate_size is not None:
            hidden_dim = config.intermediate_size
        else:
            hidden_dim = int(8 / 3 * n_embd)

        self.w1 = nn.Linear(n_embd, hidden_dim, bias=False)  # 门控分支
        self.w2 = nn.Linear(n_embd, hidden_dim, bias=False)  # 值分支
        self.w3 = nn.Linear(hidden_dim, n_embd, bias=False)  # 输出投影
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 第一步：门控分支，过 SiLU 激活
        gate = self.w1(x)
        gate = F.silu(gate)

        # 第二步：值分支，不过激活函数
        up = self.w2(x)

        # 第三步：门控相乘
        hidden = gate * up

        # 第四步：输出投影
        out = self.w3(hidden)
        out = self.dropout(out)

        return out
