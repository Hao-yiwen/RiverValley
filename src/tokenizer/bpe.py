# -*- coding: utf-8 -*-
"""
BPE 分词器模块

使用 HuggingFace tokenizers 库进行 BPE 分词
支持 ChatML 格式的 chat template
"""

import os
from typing import List, Dict, Optional, Union, TYPE_CHECKING

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

if TYPE_CHECKING:
    import torch

from tokenizer_pretokenizer import build_bpe_pre_tokenizer, describe_bpe_pre_tokenizer
from config import (
    DEFAULT_BPE_VOCAB_SIZE,
    DEFAULT_BPE_MODEL_PATH,
    SPECIAL_TOKENS,
    DEFAULT_MAX_SEQ_LEN,
)


# 特殊 token 定义
PAD_TOKEN = SPECIAL_TOKENS["pad_token"]
EOS_TOKEN = SPECIAL_TOKENS["eos_token"]
IM_START_TOKEN = SPECIAL_TOKENS["im_start_token"]
IM_END_TOKEN = SPECIAL_TOKENS["im_end_token"]

# 特殊 token 列表（按 ID 顺序）
SPECIAL_TOKENS_LIST = [PAD_TOKEN, EOS_TOKEN, IM_START_TOKEN, IM_END_TOKEN]

# 全局 tokenizer 缓存，按路径区分，避免同一进程误复用不同词表
_tokenizer_cache: Dict[str, Tokenizer] = {}


def train_bpe_tokenizer(
    text: str,
    vocab_size: int = DEFAULT_BPE_VOCAB_SIZE,
    model_path: str = DEFAULT_BPE_MODEL_PATH,
    cjk_split_chars: Optional[int] = 1,
    disable_cjk_split: bool = False,
) -> Tokenizer:
    """
    训练 HuggingFace BPE tokenizer

    Args:
        text: 训练文本
        vocab_size: 词汇表大小
        model_path: 模型保存路径
        cjk_split_chars: 连续 CJK 文本每多少字切分为一个 pre-token
        disable_cjk_split: 是否关闭 CJK 切分，恢复纯 ByteLevel 行为

    Returns:
        Tokenizer 实例
    """
    if os.path.exists(model_path):
        print(f"加载已有 BPE 模型: {model_path}")
        tokenizer = Tokenizer.from_file(model_path)
        return tokenizer

    print(f"训练 BPE tokenizer (vocab_size={vocab_size})...")
    print(f"Pre-tokenizer: {describe_bpe_pre_tokenizer(cjk_split_chars, disable_cjk_split)}")

    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = build_bpe_pre_tokenizer(
        cjk_split_chars=cjk_split_chars,
        disable_cjk_split=disable_cjk_split,
    )
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS_LIST,
        show_progress=True,
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    temp_corpus = "temp_corpus.txt"
    with open(temp_corpus, "w", encoding="utf-8") as f:
        f.write(text)

    tokenizer.train([temp_corpus], trainer)
    tokenizer.save(model_path)
    print(f"BPE 模型训练完成，保存至: {model_path}")

    if os.path.exists(temp_corpus):
        os.remove(temp_corpus)

    save_dir = os.path.splitext(model_path)[0]
    save_for_hub(tokenizer, save_dir)

    return tokenizer


def load_tokenizer(model_path: str = DEFAULT_BPE_MODEL_PATH) -> Tokenizer:
    """加载已训练的 tokenizer"""
    cache_key = os.path.abspath(model_path)
    if cache_key not in _tokenizer_cache:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"BPE 模型不存在: {model_path}")
        _tokenizer_cache[cache_key] = Tokenizer.from_file(model_path)
        print(f"加载 BPE 模型: {model_path}")
    return _tokenizer_cache[cache_key]


def encode(text: str, tokenizer: Optional[Tokenizer] = None) -> List[int]:
    """编码文本为 token ids"""
    if tokenizer is None:
        tokenizer = load_tokenizer()
    encoding = tokenizer.encode(text)
    return encoding.ids


def decode(
    ids: Union[List[int], "torch.Tensor"],
    tokenizer: Optional[Tokenizer] = None,
) -> str:
    """解码 token ids 为文本"""
    if tokenizer is None:
        tokenizer = load_tokenizer()
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    elif hasattr(ids, "__iter__"):
        ids = list(ids)
    else:
        ids = [ids]
    return tokenizer.decode(ids)


def get_vocab_size(tokenizer: Optional[Tokenizer] = None) -> int:
    """获取词汇表大小"""
    if tokenizer is None:
        tokenizer = load_tokenizer()
    return tokenizer.get_vocab_size()


def get_pad_id(tokenizer: Optional[Tokenizer] = None) -> int:
    """获取 PAD token id"""
    if tokenizer is None:
        tokenizer = load_tokenizer()
    return tokenizer.token_to_id(PAD_TOKEN)


def get_eos_id(tokenizer: Optional[Tokenizer] = None) -> int:
    """获取 EOS token id"""
    if tokenizer is None:
        tokenizer = load_tokenizer()
    return tokenizer.token_to_id(EOS_TOKEN)


def get_bos_id(tokenizer: Optional[Tokenizer] = None) -> int:
    """获取 BOS token id"""
    if tokenizer is None:
        tokenizer = load_tokenizer()
    return tokenizer.token_to_id(IM_START_TOKEN)


def get_im_start_id(tokenizer: Optional[Tokenizer] = None) -> int:
    """获取 <|im_start|> token id"""
    if tokenizer is None:
        tokenizer = load_tokenizer()
    return tokenizer.token_to_id(IM_START_TOKEN)


def get_im_end_id(tokenizer: Optional[Tokenizer] = None) -> int:
    """获取 <|im_end|> token id"""
    if tokenizer is None:
        tokenizer = load_tokenizer()
    return tokenizer.token_to_id(IM_END_TOKEN)


def apply_chat_template(
    messages: List[Dict[str, str]],
    tokenizer: Optional[Tokenizer] = None,
    add_generation_prompt: bool = False,
) -> List[int]:
    """应用 ChatML 模板格式化对话"""
    if tokenizer is None:
        tokenizer = load_tokenizer()

    text_parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        text_parts.append(f"{IM_START_TOKEN}{role}\n{content}{IM_END_TOKEN}\n")

    if add_generation_prompt:
        text_parts.append(f"{IM_START_TOKEN}assistant\n")

    full_text = "".join(text_parts)
    return encode(full_text, tokenizer)


def format_chat_text(
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = False,
) -> str:
    """格式化对话为 ChatML 文本（不编码）"""
    text_parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        text_parts.append(f"{IM_START_TOKEN}{role}\n{content}{IM_END_TOKEN}\n")

    if add_generation_prompt:
        text_parts.append(f"{IM_START_TOKEN}assistant\n")

    return "".join(text_parts)


# ChatML Jinja2 模板
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def save_for_hub(
    tokenizer: Tokenizer,
    save_dir: str = "tokenizer_hub",
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
) -> None:
    """保存 tokenizer 为 HuggingFace Hub 兼容格式"""
    from transformers import PreTrainedTokenizerFast

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=IM_START_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        unk_token=None,
        additional_special_tokens=[IM_START_TOKEN, IM_END_TOKEN],
        model_max_length=max_seq_len,
    )

    hf_tokenizer.chat_template = CHAT_TEMPLATE
    hf_tokenizer.save_pretrained(save_dir)
    print(f"Tokenizer Hub 文件保存完成: {save_dir}/")
