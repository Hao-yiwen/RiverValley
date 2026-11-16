# -*- coding: utf-8 -*-
"""
BPE pre-tokenizer 构造工具。

集中管理训练和旧 helper 使用的 pre-tokenizer，避免不同入口产生不一致的词表行为。
"""

from typing import Optional

from tokenizers import Regex, pre_tokenizers


CJK_CHAR_CLASS = r"\p{Han}\p{Hiragana}\p{Katakana}\p{Hangul}"


def build_bpe_pre_tokenizer(
    cjk_split_chars: Optional[int] = 1,
    disable_cjk_split: bool = False,
):
    """构建 BPE tokenizer 使用的 pre-tokenizer。"""
    byte_level = pre_tokenizers.ByteLevel(add_prefix_space=False)

    if disable_cjk_split or cjk_split_chars is None:
        return byte_level

    if cjk_split_chars <= 0:
        raise ValueError("cjk_split_chars 必须是正整数，或使用 disable_cjk_split 关闭 CJK 切分")

    if cjk_split_chars == 1:
        pattern = rf"([{CJK_CHAR_CLASS}])"
    else:
        pattern = rf"([{CJK_CHAR_CLASS}]{{1,{cjk_split_chars}}})"

    return pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(pattern), behavior="isolated"),
            byte_level,
        ]
    )


def describe_bpe_pre_tokenizer(
    cjk_split_chars: Optional[int] = 1,
    disable_cjk_split: bool = False,
) -> str:
    """返回便于日志记录的 pre-tokenizer 配置描述。"""
    if disable_cjk_split or cjk_split_chars is None:
        return "ByteLevel(add_prefix_space=False)"
    return (
        f"Split(CJK连续文本每 {cjk_split_chars} 字切分, isolated) "
        "-> ByteLevel(add_prefix_space=False)"
    )
