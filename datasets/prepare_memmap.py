# -*- coding: utf-8 -*-
"""
预训练数据 Memmap 生成脚本（多进程打包版）

将 CCI3-HQ (JSONL) 和 FineWeb-Edu (Parquet) 数据集进行 tokenize，
用 <|endoftext|> 连接文档，并按固定训练长度切成 memmap 格式。

特性：
- 多进程并行 tokenize（--num_workers）
- 多文档 packed blocks，减少 padding 浪费
- 每个文档追加 EOS，保留文档边界
- 流式 shuffle，避免只取排序后的前 N 条
- 限量输出（--max_samples）直接控制训练样本 / block 数
- 中英双语 round-robin 均衡（两个 dir 都给时交替取样）
- num_workers=0 时单进程 fallback，方便调试和测试

使用方法:
    python datasets/prepare_memmap.py \
        --cci3_dir /root/autodl-fs/cci3_hq_30/data \
        --fineweb_dir /root/autodl-fs/fineweb_edu_30b/sample/100BT \
        --output_dir /root/autodl-tmp/pretrain_memmap \
        --tokenizer_path /root/autodl-tmp/tokenizer.json \
        --seq_len 1024 \
        --max_samples 5000000 \
        --num_workers 16
"""

import os
import sys
import glob
import json
import argparse
import multiprocessing as mp
import random
from typing import Iterator, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# 添加 src 到 path（便于复用 tokenizer 模块的常量，但 worker 不依赖项目代码）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# 优先使用 orjson（快 5-10 倍）
try:
    import orjson
    def _loads(s):
        return orjson.loads(s)
except ImportError:
    def _loads(s):
        return json.loads(s)


PAD_TOKEN = "<|pad|>"
EOS_TOKEN = "<|endoftext|>"
UINT16_VOCAB_LIMIT = 65536


# ============================================================================
# 数据源迭代器
# ============================================================================

def load_jsonl_files(
    data_dir: str,
    random_seed: Optional[int] = None,
    shuffle_files: bool = False,
) -> Iterator[str]:
    """流式加载所有 JSONL 文件的 text 字段"""
    jsonl_files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
    if not jsonl_files:
        print(f"警告: 在 {data_dir} 中未找到 JSONL 文件")
        return
    if shuffle_files:
        rng = random.Random(random_seed)
        rng.shuffle(jsonl_files)
    print(f"找到 {len(jsonl_files)} 个 JSONL 文件 ({data_dir})")
    for file_path in jsonl_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = _loads(line)
                    text = data.get("text", "")
                    if text and text.strip():
                        yield text
                except Exception:
                    continue


def load_parquet_files(
    data_dir: str,
    batch_size: int = 10000,
    random_seed: Optional[int] = None,
    shuffle_files: bool = False,
) -> Iterator[str]:
    """流式加载所有 Parquet 文件的 text 字段"""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("错误: 需要安装 pyarrow: pip install pyarrow")
        return
    parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not parquet_files:
        print(f"警告: 在 {data_dir} 中未找到 Parquet 文件")
        return
    if shuffle_files:
        rng = random.Random(random_seed)
        rng.shuffle(parquet_files)
    print(f"找到 {len(parquet_files)} 个 Parquet 文件 ({data_dir})")
    for file_path in parquet_files:
        parquet_file = pq.ParquetFile(file_path)
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["text"]):
            for text in batch.column("text"):
                text_str = text.as_py()
                if text_str and text_str.strip():
                    yield text_str


def shuffle_buffered(
    texts: Iterator[str],
    buffer_size: int,
    random_seed: int,
) -> Iterator[str]:
    """流式 shuffle：固定大小缓冲区内随机弹出，避免只取排序后的前 N 条。"""
    if buffer_size <= 1:
        yield from texts
        return

    rng = random.Random(random_seed)
    buffer: List[str] = []
    for text in texts:
        if len(buffer) < buffer_size:
            buffer.append(text)
            continue
        idx = rng.randrange(len(buffer))
        yield buffer[idx]
        buffer[idx] = text

    while buffer:
        idx = rng.randrange(len(buffer))
        yield buffer.pop(idx)


def merged_texts(
    cci3_dir: Optional[str],
    fineweb_dir: Optional[str],
    bilingual_balance: bool = True,
    random_seed: int = 1337,
    shuffle_files: bool = True,
    shuffle_buffer_size: int = 10000,
) -> Iterator[str]:
    """
    合并两个数据源的文本流。

    bilingual_balance=True 且两个 dir 都给：交替 yield 中/英各一条；
    某一边耗尽后另一边继续。
    bilingual_balance=False 或只有一个 dir：顺序拼接。
    """
    has_zh = bool(cci3_dir and os.path.exists(cci3_dir))
    has_en = bool(fineweb_dir and os.path.exists(fineweb_dir))

    if has_zh and has_en and bilingual_balance:
        zh_iter = shuffle_buffered(
            load_jsonl_files(
                cci3_dir,
                random_seed=random_seed,
                shuffle_files=shuffle_files,
            ),
            shuffle_buffer_size,
            random_seed,
        )
        en_iter = shuffle_buffered(
            load_parquet_files(
                fineweb_dir,
                random_seed=random_seed + 1,
                shuffle_files=shuffle_files,
            ),
            shuffle_buffer_size,
            random_seed + 1,
        )
        zh_done = en_done = False
        while not (zh_done and en_done):
            if not zh_done:
                try:
                    yield next(zh_iter)
                except StopIteration:
                    zh_done = True
            if not en_done:
                try:
                    yield next(en_iter)
                except StopIteration:
                    en_done = True
    else:
        streams = []
        if has_zh:
            streams.append(
                load_jsonl_files(
                    cci3_dir,
                    random_seed=random_seed,
                    shuffle_files=shuffle_files,
                )
            )
        if has_en:
            streams.append(
                load_parquet_files(
                    fineweb_dir,
                    random_seed=random_seed + 1,
                    shuffle_files=shuffle_files,
                )
            )

        def chained():
            for stream in streams:
                yield from stream

        yield from shuffle_buffered(chained(), shuffle_buffer_size, random_seed)


def count_samples(cci3_dir: Optional[str], fineweb_dir: Optional[str]) -> int:
    """统计总样本数（旧模式：未指定 max_samples 时使用）"""
    count = 0
    if cci3_dir and os.path.exists(cci3_dir):
        print("统计 CCI3-HQ 样本数...")
        for _ in tqdm(load_jsonl_files(cci3_dir), desc="CCI3-HQ count"):
            count += 1
    if fineweb_dir and os.path.exists(fineweb_dir):
        print("统计 FineWeb-Edu 样本数...")
        for _ in tqdm(load_parquet_files(fineweb_dir), desc="FineWeb-Edu count"):
            count += 1
    return count


# ============================================================================
# Worker 实现（模块级函数，spawn / fork 都安全）
# ============================================================================

# Worker 进程局部状态（通过 initializer 注入）
_W_TOK = None        # tokenizers.Tokenizer
_W_EOS_ID: int = 1


def _init_worker(tokenizer_path: str) -> None:
    """worker 进程初始化：加载 tokenizer 一次，存到全局变量"""
    global _W_TOK, _W_EOS_ID
    # 已用 multiprocessing 拆分，关闭 Rust Rayon 内并行避免线程过载
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from tokenizers import Tokenizer
    _W_TOK = Tokenizer.from_file(tokenizer_path)
    eos_id = _W_TOK.token_to_id(EOS_TOKEN)
    _W_EOS_ID = eos_id if eos_id is not None else 1


def _encode_batch_worker(texts: List[str]) -> Tuple[np.ndarray, int]:
    """
    worker 任务：tokenize 一批文本，并在每个文档后追加 EOS。

    Returns:
        flat_ids: 已追加 EOS 的一维 token ids
        doc_count: 文档数
    """
    encodings = _W_TOK.encode_batch(texts)
    return _flatten_encodings(encodings, _W_EOS_ID)


def _flatten_encodings(encodings, eos_id: int) -> Tuple[np.ndarray, int]:
    """把 encode_batch 输出展平成 token 流，并在每条文档后追加 EOS。"""
    flat_ids: List[int] = []
    for enc in encodings:
        flat_ids.extend(enc.ids)
        flat_ids.append(eos_id)
    return np.asarray(flat_ids, dtype=np.uint16), len(encodings)


def _encode_batch_inline(texts: List[str], tokenizer, eos_id: int):
    """单进程 fallback（num_workers=0）"""
    encodings = tokenizer.encode_batch(texts)
    return _flatten_encodings(encodings, eos_id)


# ============================================================================
# Batch 切分
# ============================================================================

def _batch_iter(
    texts: Iterator[str], batch_size: int,
) -> Iterator[List[str]]:
    """把文本流切成定长 batch，最后一个 batch 可能不足 batch_size。"""
    batch: List[str] = []
    for text in texts:
        batch.append(text)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# ============================================================================
# 主流程
# ============================================================================

def create_memmap(
    output_dir: str,
    cci3_dir: Optional[str],
    fineweb_dir: Optional[str],
    tokenizer_path: str,
    seq_len: int = 2048,
    batch_size: int = 1000,
    max_samples: Optional[int] = None,
    num_workers: Optional[int] = None,
    bilingual_balance: bool = True,
    random_seed: int = 1337,
    shuffle_files: bool = True,
    shuffle_buffer_size: int = 10000,
) -> dict:
    """
    创建 memmap 文件（多进程 packed 版）。

    seq_len 表示训练输入长度；磁盘实际每行保存 seq_len + 1 个 token，
    MemmapDataset 读取后会 shift 成长度为 seq_len 的 x/y。
    max_samples 表示最多输出多少个训练样本 / block。
    """
    os.makedirs(output_dir, exist_ok=True)
    if seq_len <= 0:
        raise ValueError("seq_len 必须是正整数")
    if batch_size <= 0:
        raise ValueError("batch_size 必须是正整数")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples 必须是正整数")
    if shuffle_buffer_size < 0:
        raise ValueError("shuffle_buffer_size 不能为负数")

    # 主进程也要一份 tokenizer 用于元数据（vocab_size, pad_id）
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    pad_id_lookup = tokenizer.token_to_id(PAD_TOKEN)
    pad_id = pad_id_lookup if pad_id_lookup is not None else 0
    eos_id_lookup = tokenizer.token_to_id(EOS_TOKEN)
    if eos_id_lookup is None:
        raise ValueError(f"tokenizer 缺少 EOS token: {EOS_TOKEN}")
    eos_id = eos_id_lookup
    vocab_size = tokenizer.get_vocab_size()
    import hashlib
    with open(tokenizer_path, "rb") as source:
        tokenizer_sha256 = hashlib.sha256(source.read()).hexdigest()
    if vocab_size > UINT16_VOCAB_LIMIT:
        raise ValueError(
            f"vocab_size={vocab_size} 超过 uint16 可存储范围，"
            "请改用 uint32 或降低词表大小。"
        )
    print(f"Tokenizer vocab_size: {vocab_size}, pad_id: {pad_id}, eos_id: {eos_id}")

    block_size = seq_len + 1
    if max_samples is not None:
        print(f"\n模式: max_samples={max_samples:,} blocks")
    else:
        print("\n模式: 不限制 blocks（流式写入直到数据耗尽）")

    output_path = os.path.join(output_dir, "train.bin")
    print(f"\n写入 memmap binary: {output_path}")
    print(f"训练 seq_len: {seq_len}, 存储 block_size: {block_size}")
    if max_samples is not None:
        print(f"预计大小: {max_samples * block_size * 2 / 1024**3:.2f} GB")

    print("\n" + "=" * 60)
    workers_label = num_workers if num_workers is not None else "auto"
    print(f"Tokenize（{workers_label} workers）-> EOS 拼接 -> packed blocks")
    print("=" * 60)

    texts = merged_texts(
        cci3_dir,
        fineweb_dir,
        bilingual_balance=bilingual_balance,
        random_seed=random_seed,
        shuffle_files=shuffle_files,
        shuffle_buffer_size=shuffle_buffer_size,
    )
    batches = _batch_iter(texts, batch_size)

    total_tokens = 0
    actual_samples = 0
    document_count = 0
    partial_blocks = 0
    carry = np.empty((0,), dtype=np.uint16)

    pbar = tqdm(total=max_samples, desc="Packing blocks")

    def write_blocks(blocks: np.ndarray, out_f) -> None:
        blocks.astype(np.uint16, copy=False).tofile(out_f)

    def consume_tokens(token_ids: np.ndarray, doc_count: int, out_f) -> bool:
        """消费 token 流，写出完整 block。返回 True 表示已达到 max_samples。"""
        nonlocal carry, actual_samples, total_tokens, document_count
        document_count += doc_count
        if token_ids.size == 0:
            return False
        combined = np.concatenate((carry, token_ids)) if carry.size else token_ids
        available_blocks = combined.size // block_size
        if max_samples is not None:
            available_blocks = min(available_blocks, max_samples - actual_samples)
        if available_blocks > 0:
            token_count = available_blocks * block_size
            blocks = combined[:token_count].reshape(available_blocks, block_size)
            write_blocks(blocks, out_f)
            actual_samples += available_blocks
            total_tokens += token_count
            pbar.update(available_blocks)
        else:
            token_count = 0
        if max_samples is not None and actual_samples >= max_samples:
            carry = np.empty((0,), dtype=np.uint16)
            return True
        carry = combined[token_count:]
        return False

    stop = False
    with open(output_path, "wb") as out_f:
        if num_workers is not None and num_workers <= 0:
            # 单进程 fallback
            for batch in batches:
                token_ids, doc_count = _encode_batch_inline(batch, tokenizer, eos_id)
                stop = consume_tokens(token_ids, doc_count, out_f)
                if stop:
                    break
        else:
            if num_workers is None:
                num_workers = min(max(1, mp.cpu_count() // 4), 16)
            ctx = mp.get_context("spawn")
            with ctx.Pool(
                processes=num_workers,
                initializer=_init_worker,
                initargs=(tokenizer_path,),
            ) as pool:
                results = pool.imap(_encode_batch_worker, batches, chunksize=1)
                for token_ids, doc_count in results:
                    stop = consume_tokens(token_ids, doc_count, out_f)
                    if stop:
                        # 主动停止，避免后续 batch 还在排队
                        pool.terminate()
                        break

        if not stop and carry.size:
            block = np.full((block_size,), pad_id, dtype=np.uint16)
            block[: carry.size] = carry
            write_blocks(block.reshape(1, block_size), out_f)
            actual_samples += 1
            total_tokens += int(carry.size)
            partial_blocks = 1
            pbar.update(1)

    pbar.close()
    if actual_samples == 0:
        print("错误: 没有写出任何训练样本")
        return {}

    avg_len = total_tokens / actual_samples

    print("\n" + "=" * 60)
    print("统计信息")
    print("=" * 60)
    print(f"实际训练样本: {actual_samples:,} blocks")
    print(f"消费文档数: {document_count:,}")
    print(f"有效 token 数: {total_tokens:,}")
    print(f"平均有效长度: {avg_len:.1f} / {block_size}")
    print(f"末尾 partial block: {partial_blocks}")

    sources = []
    if cci3_dir and os.path.exists(cci3_dir):
        sources.append("cci3_hq")
    if fineweb_dir and os.path.exists(fineweb_dir):
        sources.append("fineweb_edu")

    metadata = {
        "num_samples": actual_samples,
        "seq_len": block_size,
        "train_seq_len": seq_len,
        "block_size": block_size,
        "dtype": "uint16",
        "vocab_size": vocab_size,
        "tokenizer_sha256": tokenizer_sha256,
        "pad_id": pad_id,
        "eos_id": eos_id,
        "packed": True,
        "document_count": int(document_count),
        "total_tokens": int(total_tokens),
        "avg_length": float(avg_len),
        "partial_blocks": int(partial_blocks),
        # 保留旧字段，避免下游统计代码因 schema 变化崩溃。
        "truncated_count": 0,
        "truncated_ratio": 0.0,
        "max_samples": max_samples,
        "random_seed": random_seed,
        "shuffle_files": bool(shuffle_files),
        "shuffle_buffer_size": int(shuffle_buffer_size),
        "sources": sources,
    }

    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"\n元数据已保存: {meta_path}")

    return metadata


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="预训练数据 Memmap 生成（多进程 packed blocks）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 中英平衡 5M 个训练样本，16 worker，训练输入 seq_len=1024
    python prepare_memmap.py \\
        --cci3_dir /root/autodl-fs/cci3_hq_30/data \\
        --fineweb_dir /root/autodl-fs/fineweb_edu_30b/sample/100BT \\
        --output_dir /root/autodl-tmp/pretrain_memmap \\
        --tokenizer_path /root/autodl-tmp/tokenizer.json \\
        --seq_len 1024 --max_samples 5000000 --num_workers 16

    # 调试单进程
    python prepare_memmap.py --cci3_dir ... --output_dir ... --num_workers 0
        """,
    )
    parser.add_argument("--cci3_dir", type=str, default=None,
                        help="CCI3-HQ JSONL 数据目录")
    parser.add_argument("--fineweb_dir", type=str, default=None,
                        help="FineWeb-Edu Parquet 数据目录")
    parser.add_argument("--output_dir", type=str,
                        default="/root/autodl-fs/pretrain_memmap",
                        help="输出目录")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer.json",
                        help="Tokenizer 模型路径")
    parser.add_argument("--seq_len", type=int, default=2048,
                        help="训练输入长度；磁盘每行会保存 seq_len+1 个 token")
    parser.add_argument("--batch_size", type=int, default=1000,
                        help="单个 worker 任务的文本数")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大训练样本 / block 数")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="并行 worker 进程数；0 = 单进程 fallback；不指定 = min(cpu//4, 16)")
    parser.add_argument("--no_bilingual_balance", action="store_true",
                        help="关闭中英 round-robin（默认开启）")
    parser.add_argument("--random_seed", type=int, default=1337,
                        help="文件顺序和流式 shuffle 随机种子")
    parser.add_argument("--no_shuffle_files", action="store_true",
                        help="关闭文件级随机顺序")
    parser.add_argument("--shuffle_buffer_size", type=int, default=10000,
                        help="流式 shuffle 缓冲区大小；0/1 表示关闭")

    args = parser.parse_args()

    if not args.cci3_dir and not args.fineweb_dir:
        parser.error("至少需要指定 --cci3_dir 或 --fineweb_dir 中的一个")

    print("=" * 60)
    print("预训练数据 Memmap 生成")
    print("=" * 60)
    print(f"CCI3-HQ 目录: {args.cci3_dir}")
    print(f"FineWeb-Edu 目录: {args.fineweb_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"Tokenizer: {args.tokenizer_path}")
    print(f"序列长度: {args.seq_len}")
    print(f"批处理大小: {args.batch_size}")
    print(f"最大样本数: {args.max_samples if args.max_samples else '不限制'}")
    print(f"Worker 数: {args.num_workers if args.num_workers is not None else 'auto'}")
    print(f"中英均衡: {not args.no_bilingual_balance}")
    print(f"随机种子: {args.random_seed}")
    print(f"文件随机: {not args.no_shuffle_files}")
    print(f"shuffle buffer: {args.shuffle_buffer_size}")

    metadata = create_memmap(
        output_dir=args.output_dir,
        cci3_dir=args.cci3_dir,
        fineweb_dir=args.fineweb_dir,
        tokenizer_path=args.tokenizer_path,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        bilingual_balance=not args.no_bilingual_balance,
        random_seed=args.random_seed,
        shuffle_files=not args.no_shuffle_files,
        shuffle_buffer_size=args.shuffle_buffer_size,
    )

    if metadata:
        print("\n" + "=" * 60)
        print("完成!")
        print("=" * 60)
        print(f"Memmap 文件: {args.output_dir}/train.bin")
        print(f"元数据文件: {args.output_dir}/meta.json")


if __name__ == "__main__":
    main()
