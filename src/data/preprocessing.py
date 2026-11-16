# -*- coding: utf-8 -*-
"""
数据预处理模块

包含:
- load_jsonl: 加载 JSONL 格式数据
- format_chat_to_text: 将对话格式转换为文本
"""

from typing import List, Dict, Any

# 优先使用 orjson（快 5-10 倍），fallback 到 json
try:
    import orjson
    def _loads(s):
        return orjson.loads(s)
except ImportError:
    import json
    def _loads(s):
        return json.loads(s)

from config import SPECIAL_TOKENS


# 特殊 token
IM_START = SPECIAL_TOKENS["im_start_token"]
IM_END = SPECIAL_TOKENS["im_end_token"]


def load_jsonl(file_path: str, show_progress: bool = True) -> List[Dict[str, Any]]:
    """
    加载 JSONL 格式数据文件

    Args:
        file_path: 文件路径
        show_progress: 是否显示进度条

    Returns:
        数据列表
    """
    from tqdm import tqdm

    # 先统计行数用于进度条
    if show_progress:
        with open(file_path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)
    else:
        total_lines = None

    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        iterator = tqdm(f, total=total_lines, desc="加载 JSONL") if show_progress else f
        for line in iterator:
            line = line.strip()
            if line:
                data.append(_loads(line))
    return data


def format_chat_to_text(conversations: List[Dict[str, str]]) -> str:
    """
    将对话格式转换为 ChatML 格式文本

    Args:
        conversations: 对话列表 [{"role": "user/assistant", "content": "..."}]

    Returns:
        ChatML 格式文本
    """
    text_parts = []
    for msg in conversations:
        role = msg["role"]
        content = msg["content"]
        text_parts.append(f"{IM_START}{role}\n{content}{IM_END}\n")

    return "".join(text_parts)
