# -*- coding: utf-8 -*-
"""
嵌入和归一化模块

包含:
- RMSNorm: RMS 归一化层
- YaRNRotaryEmbedding: YaRN 旋转位置编码 (兼容标准 RoPE)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from config import RiverValleyConfig


class RMSNorm(nn.Module):
    """RMSNorm: 比 LayerNorm 更快，效果相当"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_float = x.float()
        # 归一化在 FP32 中完成，降低 BF16/FP16 训练早期的数值抖动
        rms = torch.sqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        x_norm = x_float / rms
        return (x_norm * self.weight.float()).to(input_dtype)


class YaRNRotaryEmbedding(nn.Module):
    """
    YaRN (Yet another RoPE extensioN) 旋转位置编码

    支持两种模式:
    - scaling_type='none': 标准 RoPE
    - scaling_type='yarn': YaRN 动态位置编码扩展

    YaRN 核心思想:
    - 高频维度（短波长）: 保持原始频率，捕捉局部细节
    - 低频维度（长波长）: 应用插值缩放，支持更长序列
    - 中间维度: 平滑混合
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int,
        base: float = 10000.0,
        scaling_type: str = "none",
        scaling_factor: float = 1.0,
        original_max_seq_len: Optional[int] = None,
        beta_fast: int = 32,
        beta_slow: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_type = scaling_type
        self.scaling_factor = scaling_factor
        self.original_max_seq_len = original_max_seq_len or max_seq_len
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        # 当前缓存的最大长度
        self.max_seq_len_cached = max_seq_len

        # 计算频率并缓存
        inv_freq = self._compute_inv_freq()
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 预计算 cos/sin 缓存
        self._set_cos_sin_cache(max_seq_len)

    def _compute_inv_freq(self) -> torch.Tensor:
        """计算逆频率，根据 scaling_type 选择不同策略"""
        # 原始 RoPE 频率: 1 / base^(2i/d)
        inv_freq_original = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )

        if self.scaling_type == "none" or self.scaling_factor <= 1.0:
            return inv_freq_original

        # YaRN: 基于波长的混合策略
        # 插值频率 (全拉伸)
        inv_freq_interpolated = inv_freq_original / self.scaling_factor

        # 计算波长: λ = 2π / freq
        wavelen = 2 * math.pi / inv_freq_original

        # 计算 ramp 混合比例
        ramp = (wavelen / self.original_max_seq_len - self.beta_slow) / (
            self.beta_fast - self.beta_slow
        )
        ramp = torch.clamp(ramp, 0.0, 1.0)

        # 混合: (1-ramp)*原始 + ramp*插值
        inv_freq = (1 - ramp) * inv_freq_original + ramp * inv_freq_interpolated

        return inv_freq

    def _set_cos_sin_cache(self, seq_len: int) -> None:
        """预计算 cos/sin 缓存"""
        self.max_seq_len_cached = seq_len

        # 位置索引: [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, device=self.inv_freq.device).float()

        # 频率矩阵: [seq_len, dim//2]
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        # 拼接为 [seq_len, dim]
        emb = torch.cat([freqs, freqs], dim=-1)

        # 缓存 cos/sin
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取位置编码的 cos/sin

        Args:
            seq_len: 序列长度

        Returns:
            (cos, sin): 形状均为 [seq_len, dim]
        """
        # 动态扩展缓存（如果需要）
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)

        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


# 保持向后兼容的别名
RotaryPositionalEmbedding = YaRNRotaryEmbedding
