# -*- coding: utf-8 -*-
"""
注意力机制模块

包含:
- CausalSelfAttention: 分组查询注意力 GQA（带 RoPE + Flash Attention + KV Cache）
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from config import RiverValleyConfig
from model.embeddings import YaRNRotaryEmbedding


class CausalSelfAttention(nn.Module):
    """分组查询注意力 GQA（带 RoPE + Flash Attention）"""

    def __init__(self, config: RiverValleyConfig):
        super().__init__()
        n_embd = config.n_embed
        n_head = config.n_heads
        n_kv_heads = config.n_kv_heads

        self.n_head = n_head
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_head

        # 支持独立的 head_dim 配置（Qwen3 兼容）
        if config.head_dim is not None:
            self.head_dim = config.head_dim
        else:
            assert n_embd % n_head == 0, "n_embd 必须能被 n_head 整除"
            self.head_dim = n_embd // n_head

        self.n_rep = n_head // self.n_kv_heads  # 每个 KV 头对应多少个 Q 头

        assert n_head % self.n_kv_heads == 0, "n_head 必须能被 n_kv_heads 整除"

        # 计算 Q/K/V 的实际维度
        self.q_dim = self.n_head * self.head_dim
        self.kv_dim = self.n_kv_heads * self.head_dim

        # GQA: Q 投影从 hidden_size 到 q_dim，K/V 投影从 hidden_size 到 kv_dim
        self.q_proj = nn.Linear(n_embd, self.q_dim, bias=False)
        self.k_proj = nn.Linear(n_embd, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(n_embd, self.kv_dim, bias=False)
        self.c_proj = nn.Linear(self.q_dim, n_embd, bias=False)

        self.resid_dropout = nn.Dropout(config.dropout)

        # YaRN RoPE 实例（支持动态位置编码扩展）
        self.rope = YaRNRotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_theta,
            scaling_type=config.rope_scaling_type,
            scaling_factor=config.rope_scaling_factor,
            original_max_seq_len=config.yarn_original_max_seq_len,
            beta_fast=config.yarn_beta_fast,
            beta_slow=config.yarn_beta_slow,
        )

        # YaRN 温度缩放因子
        self.scaling_factor = config.rope_scaling_factor

        # Flash Attention 的 dropout 比率
        self.dropout = config.dropout
        self.use_native_gqa = config.use_native_gqa

    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """将 KV 头扩展以匹配 Q 头数量"""
        if n_rep == 1:
            return x
        B, n_kv_heads, T, head_dim = x.shape
        x = x[:, :, None, :, :].expand(B, n_kv_heads, n_rep, T, head_dim)
        return x.reshape(B, n_kv_heads * n_rep, T, head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: 输入 (B, T, C)
            attention_mask: 注意力掩码 (B, S)，1 表示有效，0 表示忽略（padding）
            start_pos: 当前位置索引（用于 RoPE 和 KV Cache）
            kv_cache: 缓存的 (k, v)，形状 (B, n_kv_heads, cache_len, head_dim)

        Returns:
            out: 输出
            new_kv_cache: 更新后的 (k, v) 缓存
        """
        B, T, C = x.shape

        # 分别计算 Q, K, V
        q = self.q_proj(x)  # (B, T, n_embd)
        k = self.k_proj(x)  # (B, T, n_kv_heads * head_dim)
        v = self.v_proj(x)  # (B, T, n_kv_heads * head_dim)

        # 重塑为多头
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 应用 RoPE（使用正确的位置索引）
        seq_len = start_pos + T
        cos, sin = self.rope(seq_len)
        cos = cos[start_pos:seq_len]
        sin = sin[start_pos:seq_len]
        q, k = self._apply_rope(q, k, cos, sin)

        # KV Cache 处理
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            if k_cache is not None and v_cache is not None:
                k = torch.cat([k_cache, k], dim=2)
                v = torch.cat([v_cache, v], dim=2)

        # 保存新的缓存
        new_kv_cache = (k, v)

        # CUDA 原生 GQA 避免显式复制 KV；CPU 和旧配置继续走兼容路径。
        native_gqa = self.use_native_gqa and q.is_cuda and self.n_rep > 1
        k_expanded = k if native_gqa else self._repeat_kv(k, self.n_rep)
        v_expanded = v if native_gqa else self._repeat_kv(v, self.n_rep)

        # 构建注意力掩码
        S = k_expanded.shape[2]
        attn_mask = None

        use_causal = S == T and T > 1

        if attention_mask is not None:
            if attention_mask.shape[-1] < S:
                raise ValueError(f"注意力掩码长度 {attention_mask.shape[-1]} 小于 KV 长度 {S}")
            attn_mask = attention_mask[:, -S:][:, None, None, :].bool()

        # 带缓存的多 token 输入也必须屏蔽后续 token；单 token 可以访问全部已有键。
        if T > 1 and (attn_mask is not None or S != T):
            query_pos = torch.arange(T, device=x.device) + S - T
            key_pos = torch.arange(S, device=x.device)
            causal_mask = key_pos[None, :] <= query_pos[:, None]
            attn_mask = causal_mask if attn_mask is None else attn_mask & causal_mask
            use_causal = False

        # 使用 Yarn 缩放
        attn_scale = 1.0 / math.sqrt(self.head_dim * self.scaling_factor)  # Yarn 缩放

        gqa_kwargs = {"enable_gqa": True} if native_gqa else {}
        out = F.scaled_dot_product_attention(
            q, k_expanded, v_expanded,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=use_causal,
            scale=attn_scale,        # 这里的 attn_scale 是 yarn 的缩放因子，不是 flash attention 本身
            **gqa_kwargs,
        )

        # 合并多头（使用 q_dim 而非 hidden_size）
        out = out.transpose(1, 2).contiguous().view(B, T, self.q_dim)
        out = self.resid_dropout(self.c_proj(out))

        return out, new_kv_cache

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """对多头 Q、K 应用 RoPE"""
        cos = cos.unsqueeze(0).unsqueeze(0).to(q.dtype)
        sin = sin.unsqueeze(0).unsqueeze(0).to(q.dtype)

        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat([-x2, x1], dim=-1)

        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        return q, k
