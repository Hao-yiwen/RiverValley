# -*- coding: utf-8 -*-
"""
数据集模块

包含:
- PretrainDataset: 预训练数据集，加载 {"text": "..."} 格式
- SFTDataset: SFT 数据集，加载 {"conversations": [...]} 格式
"""

from typing import Tuple, List, Union

import torch
from torch.utils.data import Dataset

from data.preprocessing import load_jsonl


def _truncate_tokens_for_supervised_loss(
    tokens: List[int],
    loss_mask: List[int],
    max_len: int,
) -> Tuple[List[int], List[int]]:
    """
    截断长对话时优先保留含监督 token 的窗口，避免长 user prompt 把 assistant 回复挤掉。
    """
    if len(tokens) <= max_len:
        return tokens, loss_mask

    supervised_positions = [i for i, mask in enumerate(loss_mask) if mask]
    if not supervised_positions:
        return tokens[:max_len], loss_mask[:max_len]

    # 找到最后一段回复的开头，给上下文保留至少半个窗口，避免只训练长答案的尾部。
    answer_start = supervised_positions[-1]
    while answer_start > 0 and loss_mask[answer_start - 1]:
        answer_start -= 1
    end = min(len(tokens), answer_start + max(1, max_len // 2))
    start = max(0, end - max_len)
    end = start + max_len
    return tokens[start:end], loss_mask[start:end]


class PretrainDataset(Dataset):
    """
    预训练数据集（懒加载模式）

    加载 {"text": "..."} 格式的 JSONL 文件
    编码在 __getitem__ 时进行，由 DataLoader 多 worker 并行处理
    """

    def __init__(
        self,
        data: Union[str, List[dict]],
        tokenizer,
        block_size: int,
        pad_id: int,
    ):
        """
        Args:
            data: JSONL 数据文件路径，或已加载的数据列表
            tokenizer: tokenizer 实例
            block_size: 最大序列长度
            pad_id: padding token id
        """
        self.block_size = block_size
        self.pad_id = pad_id
        self.tokenizer = tokenizer

        # 只加载原始数据，不编码
        if isinstance(data, str):
            raw_data = load_jsonl(data)
        else:
            raw_data = data

        # 过滤无效数据
        self.samples = [s for s in raw_data if s.get("text", "").strip()]
        print(f"预训练数据集: 加载 {len(self.samples)} 条样本")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取单个样本（懒加载编码）

        Returns:
            x: 输入 token ids (block_size,)
            y: 目标 token ids (block_size,)
            loss_mask: 损失掩码 (block_size,)
            attention_mask: 注意力掩码 (block_size,)
        """
        text = self.samples[idx]["text"]

        # 懒加载编码
        encoding = self.tokenizer.encode(text)
        tokens = encoding.ids if hasattr(encoding, "ids") else encoding

        # 截断或填充
        if len(tokens) > self.block_size + 1:
            tokens = tokens[: self.block_size + 1]
        else:
            tokens = tokens + [self.pad_id] * (self.block_size + 1 - len(tokens))

        tokens = torch.tensor(tokens, dtype=torch.long)
        x = tokens[:-1]
        y = tokens[1:]

        loss_mask = (y != self.pad_id).long()
        attention_mask = (x != self.pad_id).long()

        return x, y, loss_mask, attention_mask


class SFTDataset(Dataset):
    """
    SFT 微调数据集（懒加载模式）

    加载 {"conversations": [{"role": "user/assistant", "content": "..."}]} 格式的 JSONL 文件
    支持只对 assistant 回复计算 loss（mask_user=True）
    编码在 __getitem__ 时进行，由 DataLoader 多 worker 并行处理
    """

    def __init__(
        self,
        data: Union[str, List[dict]],
        tokenizer,
        block_size: int,
        pad_id: int,
        mask_user: bool = True,
    ):
        """
        Args:
            data: JSONL 数据文件路径，或已加载的数据列表
            tokenizer: tokenizer 实例
            block_size: 最大序列长度
            pad_id: padding token id
            mask_user: 是否在 loss 中屏蔽 user 部分（仅计算 assistant 的 loss）
        """
        self.block_size = block_size
        self.pad_id = pad_id
        self.tokenizer = tokenizer
        self.mask_user = mask_user

        # 只加载原始数据，不编码
        if isinstance(data, str):
            raw_data = load_jsonl(data)
        else:
            raw_data = data

        # 过滤无效数据
        def valid(sample):
            turns = sample.get("conversations")
            return (isinstance(turns, list) and bool(turns)
                    and all(isinstance(turn, dict) and turn.get("role") in {"system", "user", "assistant"}
                            and isinstance(turn.get("content"), str) for turn in turns)
                    and any(turn["role"] == "assistant" and turn["content"].strip() for turn in turns))

        self.samples = [sample for sample in raw_data if valid(sample)]
        print(f"SFT 数据集: 加载 {len(self.samples)} 条，过滤 {len(raw_data) - len(self.samples)} 条无效样本 (mask_user={mask_user})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取单个样本（懒加载编码）

        Returns:
            x: 输入 token ids (block_size,)
            y: 目标 token ids (block_size,)
            loss_mask: 损失掩码 (block_size,) - 只在 assistant 回复位置为 1
            attention_mask: 注意力掩码 (block_size,)
        """
        conversations = self.samples[idx]["conversations"]

        # 支持偏移量的 BPE 一次编码完整 ChatML，避免分段编码改变换行处的分词结果。
        parts, spans, offset = [], [], 0
        for turn in conversations:
            prefix = f"<|im_start|>{turn['role']}\n"
            text = prefix + turn["content"] + "<|im_end|>\n"
            if turn["role"] == "assistant" or not self.mask_user:
                spans.append((offset + len(prefix) if self.mask_user else offset, offset + len(text)))
            parts.append(text)
            offset += len(text)
        encoded = self.tokenizer.encode("".join(parts))
        if hasattr(encoded, "offsets"):
            all_tokens = encoded.ids
            all_loss_mask = [int(any(end > left and start < right for left, right in spans))
                             for start, end in encoded.offsets]

        else:
            # 懒加载编码
            all_tokens = []
            all_loss_mask = []
            for turn in conversations:
                role = turn.get("role", "")
                content = turn.get("content", "")
                if role == "assistant":
                    prefix = f"<|im_start|>{role}\n"
                    answer = f"{content}<|im_end|>\n"
                    prefix_encoding = self.tokenizer.encode(prefix)
                    answer_encoding = self.tokenizer.encode(answer)
                    prefix_tokens = prefix_encoding.ids if hasattr(prefix_encoding, "ids") else prefix_encoding
                    answer_tokens = answer_encoding.ids if hasattr(answer_encoding, "ids") else answer_encoding

                    all_tokens.extend(prefix_tokens)
                    all_loss_mask.extend([0] * len(prefix_tokens))
                    all_tokens.extend(answer_tokens)
                    all_loss_mask.extend([1] * len(answer_tokens))
                else:
                    text = f"<|im_start|>{role}\n{content}<|im_end|>\n"
                    encoding = self.tokenizer.encode(text)
                    turn_tokens = encoding.ids if hasattr(encoding, "ids") else encoding

                    if self.mask_user:
                        turn_loss_mask = [0] * len(turn_tokens)
                    else:
                        turn_loss_mask = [1] * len(turn_tokens)
                    all_tokens.extend(turn_tokens)
                    all_loss_mask.extend(turn_loss_mask)


        # 截断或填充
        if len(all_tokens) > self.block_size + 1:
            all_tokens, all_loss_mask = _truncate_tokens_for_supervised_loss(
                all_tokens,
                all_loss_mask,
                self.block_size + 1,
            )
        else:
            pad_len = self.block_size + 1 - len(all_tokens)
            all_tokens = all_tokens + [self.pad_id] * pad_len
            all_loss_mask = all_loss_mask + [0] * pad_len

        tokens = torch.tensor(all_tokens, dtype=torch.long)
        loss_mask_raw = torch.tensor(all_loss_mask, dtype=torch.long)

        x = tokens[:-1]
        y = tokens[1:]

        # loss_mask 需要与 y 对齐（shifted by 1）
        loss_mask = loss_mask_raw[1:] * (y != self.pad_id).long()

        # 注意力掩码
        attention_mask = (x != self.pad_id).long()

        return x, y, loss_mask, attention_mask


class DPODataset(Dataset):
    """
    DPO 训练数据集（懒加载模式）

    加载 {"chosen": [...], "rejected": [...]} 格式的 JSONL 文件
    每个 chosen/rejected 是对话列表: [{"role": "user/assistant", "content": "..."}]
    编码在 __getitem__ 时进行，由 DataLoader 多 worker 并行处理
    """

    def __init__(
        self,
        data: Union[str, List[dict]],
        tokenizer,
        block_size: int,
        pad_id: int,
    ):
        """
        Args:
            data: JSONL 数据文件路径，或已加载的数据列表
            tokenizer: tokenizer 实例
            block_size: 最大序列长度
            pad_id: padding token id
        """
        self.block_size = block_size
        self.pad_id = pad_id
        self.tokenizer = tokenizer

        # 只加载原始数据，不编码
        if isinstance(data, str):
            raw_data = load_jsonl(data)
        else:
            raw_data = data

        # 过滤无效数据：DPO 必须在完全相同的 prompt 下比较最后一个 assistant 回复
        self.samples = []
        missing_final_assistant = 0
        mismatched_prompt = 0
        missing_pair = 0
        for sample in raw_data:
            if not sample.get("chosen") or not sample.get("rejected"):
                missing_pair += 1
                continue
            chosen_split = self._split_prompt_and_answer(sample["chosen"])
            rejected_split = self._split_prompt_and_answer(sample["rejected"])
            if chosen_split is None or rejected_split is None:
                missing_final_assistant += 1
                continue
            chosen_prompt, _ = chosen_split
            rejected_prompt, _ = rejected_split
            if chosen_prompt != rejected_prompt:
                mismatched_prompt += 1
                continue
            self.samples.append(sample)

        print(
            f"DPO 数据集: 加载 {len(self.samples)} 条样本 "
            f"(跳过缺少 chosen/rejected={missing_pair}, "
            f"缺少最终 assistant={missing_final_assistant}, prompt 不一致={mismatched_prompt})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _split_prompt_and_answer(
        self, conversations: List[dict]
    ) -> Union[Tuple[List[Tuple[str, str]], str], None]:
        """拆分 DPO 对话，要求最后一轮必须是 assistant。"""
        if not conversations:
            return None
        if conversations[-1].get("role", "") != "assistant":
            return None

        prompt_turns = [
            (turn.get("role", ""), turn.get("content", ""))
            for turn in conversations[:-1]
        ]
        answer = conversations[-1].get("content", "")
        return prompt_turns, answer

    def _encode_dpo_pair(
        self, sample: dict
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        """
        编码 DPO 样本，保证 chosen/rejected 使用完全相同的 prompt token 窗口。
        """
        chosen_prompt, chosen_answer = self._split_prompt_and_answer(sample["chosen"])
        rejected_prompt, rejected_answer = self._split_prompt_and_answer(sample["rejected"])
        if chosen_prompt != rejected_prompt:
            raise ValueError("DPO 样本的 chosen/rejected prompt 不一致，初始化过滤逻辑可能失效")

        prompt_text = "".join(
            f"<|im_start|>{role}\n{content}<|im_end|>\n"
            for role, content in chosen_prompt
        )
        prompt_text += "<|im_start|>assistant\n"
        chosen_answer_text = f"{chosen_answer}<|im_end|>\n"
        rejected_answer_text = f"{rejected_answer}<|im_end|>\n"

        prompt_encoding = self.tokenizer.encode(prompt_text)
        chosen_answer_encoding = self.tokenizer.encode(chosen_answer_text)
        rejected_answer_encoding = self.tokenizer.encode(rejected_answer_text)
        prompt_tokens = prompt_encoding.ids if hasattr(prompt_encoding, "ids") else prompt_encoding
        chosen_answer_tokens = (
            chosen_answer_encoding.ids if hasattr(chosen_answer_encoding, "ids") else chosen_answer_encoding
        )
        rejected_answer_tokens = (
            rejected_answer_encoding.ids if hasattr(rejected_answer_encoding, "ids") else rejected_answer_encoding
        )

        return self._truncate_shared_prompt_pair(
            prompt_tokens,
            chosen_answer_tokens,
            rejected_answer_tokens,
        )

    def _pad_and_create_tensors(
        self, tokens: List[int], loss_mask: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        填充/截断并创建张量

        Returns:
            x: 输入 token ids (block_size,)
            y: 目标 token ids (block_size,)
            loss_mask: 损失掩码 (block_size,)
            attention_mask: 注意力掩码 (block_size,)
        """
        # 截断或填充
        if len(tokens) > self.block_size + 1:
            tokens, loss_mask = _truncate_tokens_for_supervised_loss(
                tokens,
                loss_mask,
                self.block_size + 1,
            )
        else:
            pad_len = self.block_size + 1 - len(tokens)
            tokens = tokens + [self.pad_id] * pad_len
            loss_mask = loss_mask + [0] * pad_len

        tokens = torch.tensor(tokens, dtype=torch.long)
        loss_mask_raw = torch.tensor(loss_mask, dtype=torch.long)

        x = tokens[:-1]
        y = tokens[1:]

        # loss_mask 需要与 y 对齐（shifted by 1）
        final_loss_mask = loss_mask_raw[1:] * (y != self.pad_id).long()

        # 注意力掩码
        attention_mask = (x != self.pad_id).long()

        return x, y, final_loss_mask, attention_mask

    def _truncate_shared_prompt_pair(
        self,
        prompt_tokens: List[int],
        chosen_answer_tokens: List[int],
        rejected_answer_tokens: List[int],
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        """
        用同一个 prompt 后缀构造 chosen/rejected，并只监督最终回答。
        """
        max_len = self.block_size + 1
        max_answer_len = max(len(chosen_answer_tokens), len(rejected_answer_tokens))
        prompt_budget = max(0, max_len - max_answer_len)
        prompt_window = prompt_tokens[-prompt_budget:] if prompt_budget > 0 else []
        answer_budget = max_len - len(prompt_window)

        chosen_answer_tokens = chosen_answer_tokens[:answer_budget]
        rejected_answer_tokens = rejected_answer_tokens[:answer_budget]

        chosen_tokens = prompt_window + chosen_answer_tokens
        rejected_tokens = prompt_window + rejected_answer_tokens
        chosen_loss_mask = [0] * len(prompt_window) + [1] * len(chosen_answer_tokens)
        rejected_loss_mask = [0] * len(prompt_window) + [1] * len(rejected_answer_tokens)
        return chosen_tokens, chosen_loss_mask, rejected_tokens, rejected_loss_mask

    def __getitem__(self, idx: int) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        获取单个样本（懒加载编码）

        Returns:
            chosen_x: chosen 输入 token ids (block_size,)
            chosen_y: chosen 目标 token ids (block_size,)
            chosen_loss_mask: chosen 损失掩码 (block_size,)
            chosen_attention_mask: chosen 注意力掩码 (block_size,)
            rejected_x: rejected 输入 token ids (block_size,)
            rejected_y: rejected 目标 token ids (block_size,)
            rejected_loss_mask: rejected 损失掩码 (block_size,)
            rejected_attention_mask: rejected 注意力掩码 (block_size,)
        """
        sample = self.samples[idx]

        # 编码 chosen/rejected 对话，并保持同一个 prompt 上下文窗口
        chosen_tokens, chosen_loss_mask, rejected_tokens, rejected_loss_mask = self._encode_dpo_pair(sample)

        chosen_x, chosen_y, chosen_lm, chosen_am = self._pad_and_create_tensors(
            chosen_tokens, chosen_loss_mask
        )
        rejected_x, rejected_y, rejected_lm, rejected_am = self._pad_and_create_tensors(
            rejected_tokens, rejected_loss_mask
        )

        return (
            chosen_x, chosen_y, chosen_lm, chosen_am,
            rejected_x, rejected_y, rejected_lm, rejected_am
        )


class RLAIFDataset(Dataset):
    """
    RLAIF 数据集（用于 GRPO 训练）

    加载 {"conversations": [...]} 格式的 JSONL 文件
    返回 prompt 和 answer 用于强化学习训练
    """

    def __init__(
        self,
        data: Union[str, List[dict]],
        tokenizer,
        max_length: int = 1024,
    ):
        """
        Args:
            data: JSONL 数据文件路径，或已加载的数据列表
            tokenizer: tokenizer 实例
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 只加载原始数据
        if isinstance(data, str):
            raw_data = load_jsonl(data)
        else:
            raw_data = data

        # 过滤无效数据
        self.samples = [s for s in raw_data if s.get("conversations")]
        print(f"RLAIF 数据集: 加载 {len(self.samples)} 条样本")

    def __len__(self) -> int:
        return len(self.samples)

    def _create_chat_prompt(self, conversations: List[dict]) -> Tuple[str, str]:
        """
        构建 ChatML 格式的 prompt 和提取 answer

        Args:
            conversations: 对话列表

        Returns:
            prompt: 不含最后一个 assistant 回复的 prompt
            answer: 最后一个 assistant 的回复内容
        """
        messages = []
        answer = ""

        for turn in conversations:
            role = turn.get("role", "")
            content = turn.get("content", "")
            messages.append({"role": role, "content": content})
            if role == "assistant":
                answer = content

        # 构建 prompt（不包含最后一个 assistant 回复）
        prompt_parts = []
        for msg in messages[:-1]:
            role = msg["role"]
            content = msg["content"]
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

        # 添加 generation prompt
        prompt_parts.append("<|im_start|>assistant\n")
        prompt = "".join(prompt_parts)

        return prompt, answer

    def __getitem__(self, idx: int) -> dict:
        """
        获取单个样本

        Returns:
            dict: {"prompt": str, "answer": str}
        """
        sample = self.samples[idx]
        prompt, answer = self._create_chat_prompt(sample["conversations"])

        return {
            "prompt": prompt,
            "answer": answer,
        }


class MemmapDataset(Dataset):
    """
    Memmap 预训练数据集

    从预处理好的 memmap 文件加载数据，避免重复 tokenize
    memmap 文件格式: (num_samples, seq_len), dtype=uint16
    """

    def __init__(
        self,
        memmap_path: str,
        meta_path: str,
        pad_id: int = 0,
    ):
        """
        Args:
            memmap_path: memmap 文件路径 (train.bin)
            meta_path: 元数据文件路径 (meta.json)
            pad_id: padding token id
        """
        import json
        import numpy as np

        # 加载元数据
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.num_samples = self.meta["num_samples"]
        self.seq_len = self.meta["seq_len"]
        self.pad_id = pad_id if pad_id is not None else self.meta.get("pad_id", 0)

        import hashlib
        from pathlib import Path
        path = Path(memmap_path)
        expected_bytes = self.num_samples * self.seq_len * np.dtype(np.uint16).itemsize
        if self.num_samples <= 0 or self.seq_len < 2 or path.stat().st_size != expected_bytes:
            raise ValueError(f"memmap 文件与元数据的长度不一致: {path}，期望字节={expected_bytes}")
        if self.meta.get("data_sha256"):
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != self.meta["data_sha256"]:
                raise ValueError(f"memmap 内容校验失败，停止使用损坏或已修改的数据: {path}")
            print(f"Memmap 内容 SHA256 校验通过: {path}")

        # 加载 memmap（只读模式）
        self.data = np.memmap(
            memmap_path,
            dtype=np.uint16,
            mode="r",
            shape=(self.num_samples, self.seq_len)
        )

        print(f"Memmap 数据集: 加载 {self.num_samples:,} 条样本, 序列长度 {self.seq_len}")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取单个样本

        Returns:
            x: 输入 token ids (seq_len-1,)
            y: 目标 token ids (seq_len-1,)
            loss_mask: 损失掩码 (seq_len-1,)
            attention_mask: 注意力掩码 (seq_len-1,)
        """
        # 从 memmap 读取
        tokens = torch.from_numpy(self.data[idx].astype('int64'))

        # 构造输入输出
        x = tokens[:-1]
        y = tokens[1:]

        # 构造掩码
        loss_mask = (y != self.pad_id).long()
        attention_mask = (x != self.pad_id).long()

        return x, y, loss_mask, attention_mask
