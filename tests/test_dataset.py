# -*- coding: utf-8 -*-
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

pytest.importorskip("torch")

from data.dataset import DPODataset, SFTDataset  # noqa: E402


class _Encoding:
    def __init__(self, ids):
        self.ids = ids


class _CharTokenizer:
    def encode(self, text):
        return _Encoding([ord(ch) for ch in text])


def _masked_text(y, loss_mask):
    return "".join(
        chr(int(token_id))
        for token_id, mask in zip(y.tolist(), loss_mask.tolist())
        if mask
    )


def test_sft_truncation_keeps_assistant_loss_for_long_prompt():
    sample = {
        "conversations": [
            {"role": "user", "content": "u" * 200},
            {"role": "assistant", "content": "ok"},
        ]
    }

    dataset = SFTDataset([sample], _CharTokenizer(), block_size=16, pad_id=0)
    _, _, loss_mask, _ = dataset[0]

    assert loss_mask.sum().item() > 0


def test_sft_masks_assistant_header_and_trains_answer_only():
    sample = {
        "conversations": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "OK"},
        ]
    }

    dataset = SFTDataset([sample], _CharTokenizer(), block_size=128, pad_id=0)
    _, y, loss_mask, _ = dataset[0]

    supervised = _masked_text(y, loss_mask)
    assert supervised == "OK<|im_end|>\n"
    assert "assistant" not in supervised


def test_dpo_uses_shared_prompt_and_final_answer_mask_only():
    sample = {
        "chosen": [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "pick"},
            {"role": "assistant", "content": "good"},
        ],
        "rejected": [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "pick"},
            {"role": "assistant", "content": "bad"},
        ],
    }

    dataset = DPODataset([sample], _CharTokenizer(), block_size=128, pad_id=0)
    chosen_x, chosen_y, chosen_lm, _, rejected_x, rejected_y, rejected_lm, _ = dataset[0]

    chosen_supervised = _masked_text(chosen_y, chosen_lm)
    rejected_supervised = _masked_text(rejected_y, rejected_lm)
    assert chosen_supervised == "good<|im_end|>\n"
    assert rejected_supervised == "bad<|im_end|>\n"

    prompt_len = len("<|im_start|>system\nbe helpful<|im_end|>\n<|im_start|>user\npick<|im_end|>\n<|im_start|>assistant\n")
    assert chosen_x[:prompt_len].tolist() == rejected_x[:prompt_len].tolist()


def test_dpo_filters_mismatched_prompt_and_missing_final_assistant():
    valid = {
        "chosen": [
            {"role": "user", "content": "same"},
            {"role": "assistant", "content": "yes"},
        ],
        "rejected": [
            {"role": "user", "content": "same"},
            {"role": "assistant", "content": "no"},
        ],
    }
    mismatched = {
        "chosen": [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "yes"},
        ],
        "rejected": [
            {"role": "user", "content": "b"},
            {"role": "assistant", "content": "no"},
        ],
    }
    missing_final = {
        "chosen": [{"role": "user", "content": "same"}],
        "rejected": [
            {"role": "user", "content": "same"},
            {"role": "assistant", "content": "no"},
        ],
    }

    dataset = DPODataset([valid, mismatched, missing_final], _CharTokenizer(), block_size=64, pad_id=0)

    assert len(dataset.samples) == 1
    assert dataset.samples[0] == valid
