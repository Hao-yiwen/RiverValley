# -*- coding: utf-8 -*-
"""
分词器模块

包含:
- BPE 分词器相关功能
- ChatML 模板支持
"""

from tokenizer.bpe import (
    train_bpe_tokenizer,
    load_tokenizer,
    encode,
    decode,
    get_vocab_size,
    get_pad_id,
    get_eos_id,
    get_bos_id,
    get_im_start_id,
    get_im_end_id,
    apply_chat_template,
    format_chat_text,
    save_for_hub,
    PAD_TOKEN,
    EOS_TOKEN,
    IM_START_TOKEN,
    IM_END_TOKEN,
    SPECIAL_TOKENS_LIST,
    CHAT_TEMPLATE,
)

__all__ = [
    # 函数
    "train_bpe_tokenizer",
    "load_tokenizer",
    "encode",
    "decode",
    "get_vocab_size",
    "get_pad_id",
    "get_eos_id",
    "get_bos_id",
    "get_im_start_id",
    "get_im_end_id",
    "apply_chat_template",
    "format_chat_text",
    "save_for_hub",
    # 常量
    "PAD_TOKEN",
    "EOS_TOKEN",
    "IM_START_TOKEN",
    "IM_END_TOKEN",
    "SPECIAL_TOKENS_LIST",
    "CHAT_TEMPLATE",
]
