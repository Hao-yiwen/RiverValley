# -*- coding: utf-8 -*-
import os
import sys
import types

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "scripts"))

pytest.importorskip("tokenizers")

config_stub = types.ModuleType("config")
config_stub.SPECIAL_TOKENS = {
    "pad_token": "<|pad|>",
    "eos_token": "<|endoftext|>",
    "im_start_token": "<|im_start|>",
    "im_end_token": "<|im_end|>",
}
config_stub.DEFAULT_BPE_VOCAB_SIZE = 32000
config_stub.DEFAULT_BPE_MODEL_PATH = "tokenizer.json"
config_stub.DEFAULT_MAX_SEQ_LEN = 1024
sys.modules.setdefault("config", config_stub)

from tokenizers import Tokenizer, models  # noqa: E402
import train_tokenizer as tt  # noqa: E402
from tokenizer_pretokenizer import build_bpe_pre_tokenizer  # noqa: E402
import tokenizer.bpe as bpe  # noqa: E402


def test_cjk_pre_tokenizer_splits_cjk_chars_and_keeps_english_words():
    tokenizer = build_bpe_pre_tokenizer(cjk_split_chars=1)

    cjk_splits = tokenizer.pre_tokenize_str("你好世界人工智能")
    english_splits = tokenizer.pre_tokenize_str("hello world this is a test")

    assert [span for _, span in cjk_splits] == [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 8),
    ]
    assert [piece for piece, _ in english_splits] == [
        "hello",
        "Ġworld",
        "Ġthis",
        "Ġis",
        "Ġa",
        "Ġtest",
    ]


def test_split_train_units_keeps_text_and_bounds():
    text = (
        "第一段中文内容很长，需要按照标点切分。"
        "第二段继续测试，不应该丢字。"
        "long english segment with spaces and punctuation, should split softly. "
        "def hello(): print('hello world')"
    )

    units = list(tt.split_train_units(text, 24))

    assert units
    assert "".join(units) == text
    assert all(unit.strip() for unit in units)
    assert all(len(unit) <= 24 for unit in units)


def test_sample_files_low_memory_training_preserves_special_ids(tmp_path, monkeypatch):
    zh = tmp_path / "zh_samples.txt"
    en = tmp_path / "en_samples.txt"
    out = tmp_path / "tokenizer.json"

    zh.write_text(
        "\n".join(
            [
                "人工智能正在改变我们的生活方式，从智能手机到自动驾驶汽车。" * 4,
                "中文样本包含标点、数字123、英文AI，以及代码片段 def test(): pass" * 3,
            ]
            * 80
        ),
        encoding="utf-8",
    )
    en.write_text(
        "\n".join(
            [
                "Artificial intelligence is transforming how we live and work. " * 5,
                "Tokenizer training should handle code, email test@example.com, and symbols." * 3,
            ]
            * 80
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(tt, "save_for_hub", lambda tokenizer, save_dir: os.makedirs(save_dir, exist_ok=True))

    tt.train_tokenizer(
        cci3_dir=None,
        fineweb_dir=None,
        vocab_size=320,
        output_path=str(out),
        min_frequency=1,
        max_doc_chars=None,
        train_unit_chars=64,
        sample_files=[str(zh), str(en)],
        keep_samples=True,
        max_token_length=32,
        num_threads=1,
        disable_parallelism=True,
    )

    tokenizer = Tokenizer.from_file(str(out))

    assert tokenizer.token_to_id("<|pad|>") == 0
    assert tokenizer.token_to_id("<|endoftext|>") == 1
    assert tokenizer.token_to_id("<|im_start|>") == 2
    assert tokenizer.token_to_id("<|im_end|>") == 3
    assert tokenizer.get_vocab_size() == 320
    assert tokenizer.decode(tokenizer.encode("你好AI world").ids) == "你好AI world"
    assert zh.exists()
    assert en.exists()
    assert (tmp_path / "tokenizer").is_dir()


def test_load_tokenizer_cache_is_path_specific(tmp_path):
    tok_a = Tokenizer(models.WordLevel({"a": 0}, unk_token=None))
    tok_b = Tokenizer(models.WordLevel({"b": 0, "c": 1}, unk_token=None))
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    tok_a.save(str(path_a))
    tok_b.save(str(path_b))

    bpe._tokenizer_cache.clear()

    loaded_a = bpe.load_tokenizer(str(path_a))
    loaded_a_again = bpe.load_tokenizer(str(path_a))
    loaded_b = bpe.load_tokenizer(str(path_b))

    assert loaded_a is loaded_a_again
    assert loaded_a is not loaded_b
    assert loaded_a.get_vocab_size() == 1
    assert loaded_b.get_vocab_size() == 2
