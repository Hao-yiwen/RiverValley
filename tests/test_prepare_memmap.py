# -*- coding: utf-8 -*-
"""
prepare_memmap.py 测试

验证：
- --max_samples 按 packed block 数限量
- 数据不足时 metadata 反映实际 block 数
- 每个文档追加 EOS，输出为 packed blocks
- 不同随机种子会改变流式 shuffle 输出
- 多进程 (num_workers>0) 与单进程 (num_workers=0) 输出字节级一致
- 输出格式与 MemmapDataset 兼容
- meta.json 字段齐全
"""

import os
import sys
import json

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "datasets"))

# 导入被测脚本
import prepare_memmap as pm  # noqa: E402

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders  # noqa: E402


SPECIAL_TOKENS = ["<|pad|>", "<|endoftext|>", "<|im_start|>", "<|im_end|>"]


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def tiny_tokenizer(tmp_path_factory):
    """训练一个迷你 BPE（vocab=256），存到 tmp_path"""
    tmp_dir = tmp_path_factory.mktemp("tokenizer")
    tokenizer_path = tmp_dir / "tokenizer.json"

    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=256,
        special_tokens=SPECIAL_TOKENS,
        show_progress=False,
        min_frequency=1,
    )

    # 训练语料：拼些中英文 ASCII 文本，让 BPE 收敛
    corpus_path = tmp_dir / "corpus.txt"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for i in range(200):
            f.write(f"hello world this is sample {i} for tokenizer training.\n")
            f.write(f"中文样本 {i} 测试 tokenizer 训练 数据 内容 一些重复字符 abcdef\n")

    tokenizer.train([str(corpus_path)], trainer)
    tokenizer.save(str(tokenizer_path))
    return str(tokenizer_path)


def _write_jsonl(path, texts):
    with open(path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")


@pytest.fixture
def tiny_jsonl_dir(tmp_path):
    """50 条假中文 jsonl"""
    d = tmp_path / "cci3"
    d.mkdir()
    texts = [f"中文文档样本 {i}：一些用于测试的内容，长度足够触发 tokenize。" for i in range(50)]
    _write_jsonl(d / "part_001.jsonl", texts[:25])
    _write_jsonl(d / "part_002.jsonl", texts[25:])
    return str(d)


@pytest.fixture
def tiny_parquet_dir(tmp_path):
    """50 条假英文 parquet（如果 pyarrow 不可用就跳过）"""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    d = tmp_path / "fineweb"
    d.mkdir()
    texts = [f"english document sample {i} for memmap testing purposes." for i in range(50)]
    table = pa.table({"text": texts[:25]})
    pq.write_table(table, str(d / "001.parquet"))
    table2 = pa.table({"text": texts[25:]})
    pq.write_table(table2, str(d / "002.parquet"))
    return str(d)


# ============================================================================
# 基础场景：max_samples 截断 / 数据不足
# ============================================================================

def test_max_samples_truncates(tmp_path, tiny_tokenizer, tiny_jsonl_dir):
    """max_samples=30 时，应该正好写出 30 个 packed blocks"""
    out = tmp_path / "out_truncate"
    seq_len = 16

    meta = pm.create_memmap(
        output_dir=str(out),
        cci3_dir=tiny_jsonl_dir,
        fineweb_dir=None,
        tokenizer_path=tiny_tokenizer,
        seq_len=seq_len,
        batch_size=10,
        max_samples=30,
        num_workers=0,
        shuffle_files=False,
        shuffle_buffer_size=0,
    )

    assert meta["num_samples"] == 30
    assert meta["train_seq_len"] == seq_len
    assert meta["seq_len"] == seq_len + 1
    assert meta["packed"] is True
    train_bin = out / "train.bin"
    assert train_bin.stat().st_size == 30 * (seq_len + 1) * 2

    # 直接 memmap 读应该形状对
    arr = np.memmap(str(train_bin), dtype=np.uint16, mode="r", shape=(30, seq_len + 1))
    assert arr.shape == (30, seq_len + 1)
    assert arr.dtype == np.uint16


def test_max_samples_exceeds_data(tmp_path, tiny_tokenizer, tiny_jsonl_dir):
    """数据不足以写满 max_samples 时，metadata 应反映实际 block 数"""
    out = tmp_path / "out_exceed"
    seq_len = 32

    meta = pm.create_memmap(
        output_dir=str(out),
        cci3_dir=tiny_jsonl_dir,
        fineweb_dir=None,
        tokenizer_path=tiny_tokenizer,
        seq_len=seq_len,
        batch_size=10,
        max_samples=1000,
        num_workers=0,
        shuffle_files=False,
        shuffle_buffer_size=0,
    )

    assert 0 < meta["num_samples"] < 1000
    assert meta["document_count"] == 50
    train_bin = out / "train.bin"
    assert train_bin.stat().st_size == meta["num_samples"] * (seq_len + 1) * 2


def test_no_max_samples_uses_count(tmp_path, tiny_tokenizer, tiny_jsonl_dir):
    """不指定 max_samples 时应流式写完全部数据"""
    out = tmp_path / "out_count"
    seq_len = 32

    meta = pm.create_memmap(
        output_dir=str(out),
        cci3_dir=tiny_jsonl_dir,
        fineweb_dir=None,
        tokenizer_path=tiny_tokenizer,
        seq_len=seq_len,
        batch_size=10,
        max_samples=None,
        num_workers=0,
        shuffle_files=False,
        shuffle_buffer_size=0,
    )

    assert meta["num_samples"] > 0
    assert meta["document_count"] == 50


def test_packed_blocks_include_eos(tmp_path, tiny_tokenizer, tiny_jsonl_dir):
    """packed token 流中应包含文档边界 EOS"""
    out = tmp_path / "out_eos"
    seq_len = 16

    meta = pm.create_memmap(
        output_dir=str(out),
        cci3_dir=tiny_jsonl_dir,
        fineweb_dir=None,
        tokenizer_path=tiny_tokenizer,
        seq_len=seq_len,
        batch_size=10,
        max_samples=10,
        num_workers=0,
        shuffle_files=False,
        shuffle_buffer_size=0,
    )

    arr = np.memmap(
        str(out / "train.bin"),
        dtype=np.uint16,
        mode="r",
        shape=(meta["num_samples"], meta["seq_len"]),
    )
    assert meta["eos_id"] in arr


# ============================================================================
# 多进程 vs 单进程一致性
# ============================================================================

def test_workers_match_sequential(tmp_path, tiny_tokenizer, tiny_jsonl_dir):
    """num_workers=2 和 num_workers=0 应输出字节级一致的 train.bin（imap 保序）"""
    seq_len = 16
    out_seq = tmp_path / "seq"
    out_par = tmp_path / "par"

    meta_seq = pm.create_memmap(
        output_dir=str(out_seq),
        cci3_dir=tiny_jsonl_dir, fineweb_dir=None,
        tokenizer_path=tiny_tokenizer,
        seq_len=seq_len, batch_size=10,
        max_samples=40, num_workers=0,
        shuffle_files=False, shuffle_buffer_size=0,
    )
    meta_par = pm.create_memmap(
        output_dir=str(out_par),
        cci3_dir=tiny_jsonl_dir, fineweb_dir=None,
        tokenizer_path=tiny_tokenizer,
        seq_len=seq_len, batch_size=10,
        max_samples=40, num_workers=2,
        shuffle_files=False, shuffle_buffer_size=0,
    )

    assert meta_seq["num_samples"] == meta_par["num_samples"] == 40
    assert meta_seq["total_tokens"] == meta_par["total_tokens"]
    assert meta_seq["truncated_count"] == meta_par["truncated_count"]

    bytes_seq = (out_seq / "train.bin").read_bytes()
    bytes_par = (out_par / "train.bin").read_bytes()
    assert bytes_seq == bytes_par, "多进程与单进程输出 train.bin 应字节级一致"


def test_random_seed_changes_output(tmp_path, tiny_tokenizer, tiny_jsonl_dir):
    """启用流式 shuffle 时，不同 random_seed 应改变 packed 输出顺序"""
    seq_len = 16
    out_a = tmp_path / "seed_a"
    out_b = tmp_path / "seed_b"

    pm.create_memmap(
        output_dir=str(out_a),
        cci3_dir=tiny_jsonl_dir, fineweb_dir=None,
        tokenizer_path=tiny_tokenizer,
        seq_len=seq_len, batch_size=10,
        max_samples=20, num_workers=0,
        random_seed=1, shuffle_files=True, shuffle_buffer_size=50,
    )
    pm.create_memmap(
        output_dir=str(out_b),
        cci3_dir=tiny_jsonl_dir, fineweb_dir=None,
        tokenizer_path=tiny_tokenizer,
        seq_len=seq_len, batch_size=10,
        max_samples=20, num_workers=0,
        random_seed=2, shuffle_files=True, shuffle_buffer_size=50,
    )

    assert (out_a / "train.bin").read_bytes() != (out_b / "train.bin").read_bytes()


# ============================================================================
# 双语 round-robin
# ============================================================================

def test_bilingual_balance(tmp_path, tiny_tokenizer, tiny_jsonl_dir, tiny_parquet_dir):
    """两个 dir 都给 + balance=True，应交替消费"""
    out = tmp_path / "out_bi"
    seq_len = 16

    meta = pm.create_memmap(
        output_dir=str(out),
        cci3_dir=tiny_jsonl_dir,
        fineweb_dir=tiny_parquet_dir,
        tokenizer_path=tiny_tokenizer,
        seq_len=seq_len,
        batch_size=10,
        max_samples=40,
        num_workers=0,
        bilingual_balance=True,
        shuffle_files=False,
        shuffle_buffer_size=0,
    )

    assert meta["num_samples"] == 40
    assert "cci3_hq" in meta["sources"]
    assert "fineweb_edu" in meta["sources"]


# ============================================================================
# MemmapDataset 兼容性 + meta.json schema
# ============================================================================

def test_memmap_dataset_roundtrip(tmp_path, tiny_tokenizer, tiny_jsonl_dir):
    """产出的文件能被 MemmapDataset 加载"""
    pytest.importorskip("torch")
    from data.dataset import MemmapDataset  # noqa: E402

    out = tmp_path / "out_dataset"
    seq_len = 32

    pm.create_memmap(
        output_dir=str(out),
        cci3_dir=tiny_jsonl_dir, fineweb_dir=None,
        tokenizer_path=tiny_tokenizer,
        seq_len=seq_len, batch_size=10,
        max_samples=20, num_workers=0,
        shuffle_files=False, shuffle_buffer_size=0,
    )

    ds = MemmapDataset(
        memmap_path=str(out / "train.bin"),
        meta_path=str(out / "meta.json"),
        pad_id=0,
    )
    assert len(ds) == 20

    x, y, loss_mask, attention_mask = ds[0]
    assert x.shape == (seq_len,)
    assert y.shape == (seq_len,)
    assert loss_mask.shape == (seq_len,)
    assert attention_mask.shape == (seq_len,)


def test_meta_fields(tmp_path, tiny_tokenizer, tiny_jsonl_dir):
    """meta.json 字段齐全"""
    out = tmp_path / "out_meta"
    pm.create_memmap(
        output_dir=str(out),
        cci3_dir=tiny_jsonl_dir, fineweb_dir=None,
        tokenizer_path=tiny_tokenizer,
        seq_len=32, batch_size=10,
        max_samples=20, num_workers=0,
        shuffle_files=False, shuffle_buffer_size=0,
    )
    with open(out / "meta.json") as f:
        meta = json.load(f)

    expected = {
        "num_samples", "seq_len", "train_seq_len", "block_size", "dtype",
        "vocab_size", "pad_id", "eos_id", "packed", "document_count",
        "total_tokens", "avg_length", "truncated_count", "truncated_ratio",
        "partial_blocks", "max_samples", "random_seed", "shuffle_files",
        "shuffle_buffer_size", "sources",
    }
    assert expected <= set(meta.keys())
    assert meta["dtype"] == "uint16"
    assert meta["seq_len"] == 33
    assert meta["train_seq_len"] == 32
    assert meta["packed"] is True
    assert meta["sources"] == ["cci3_hq"]
