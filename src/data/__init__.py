# -*- coding: utf-8 -*-
"""
数据加载与预处理模块

包含:
- PretrainDataset: 预训练数据集
- SFTDataset: SFT 微调数据集
- DPODataset: DPO 训练数据集
- RLAIFDataset: RLAIF 数据集（用于 GRPO 训练）
- 数据加载和预处理函数
"""

from data.dataset import PretrainDataset, SFTDataset, DPODataset, RLAIFDataset
from data.preprocessing import load_jsonl, format_chat_to_text

__all__ = [
    "PretrainDataset",
    "SFTDataset",
    "DPODataset",
    "RLAIFDataset",
    "load_jsonl",
    "format_chat_to_text",
]
