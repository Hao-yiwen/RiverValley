# -*- coding: utf-8 -*-
"""
BPE Tokenizer 训练脚本

从 CCI3-HQ (JSONL) 和 FineWeb-Edu (Parquet) 数据训练 BPE tokenizer

使用方法:
    python src/scripts/train_tokenizer.py \
        --cci3_dir /root/autodl-fs/cci3_hq_30/data \
        --fineweb_dir /root/autodl-fs/fineweb_edu_30b/sample/100BT \
        --vocab_size 65000 \
        --output_path tokenizer.json

    # 快速训练（限制样本数）
    python src/scripts/train_tokenizer.py \
        --cci3_dir /root/autodl-fs/cci3_hq_30/data \
        --vocab_size 65000 \
        --max_samples 10000000
"""

import os
import sys
import glob
import json
import argparse
import random
import tempfile
import shutil
from typing import Iterator, Optional, List

from tqdm import tqdm

# 获取项目根目录和 src 目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# 添加 src 到路径
sys.path.insert(0, SRC_DIR)

# 切换到项目根目录
os.chdir(PROJECT_ROOT)

from config import SPECIAL_TOKENS, DEFAULT_BPE_VOCAB_SIZE, DEFAULT_BPE_MODEL_PATH
from tokenizer_pretokenizer import build_bpe_pre_tokenizer, describe_bpe_pre_tokenizer

# 优先使用 orjson
try:
    import orjson
    def _loads(s):
        return orjson.loads(s)
except ImportError:
    def _loads(s):
        return json.loads(s)

# 特殊 token 列表（按 ID 顺序）
SPECIAL_TOKENS_LIST = [
    SPECIAL_TOKENS["pad_token"],      # id=0
    SPECIAL_TOKENS["eos_token"],      # id=1
    SPECIAL_TOKENS["im_start_token"], # id=2
    SPECIAL_TOKENS["im_end_token"],   # id=3
]


def _normalize_text(text: Optional[str], max_doc_chars: Optional[int]) -> Optional[str]:
    if not text:
        return None
    text = text.strip()
    if not text:
        return None
    if max_doc_chars is not None and len(text) > max_doc_chars:
        text = text[:max_doc_chars]
    return text


_TRAIN_UNIT_BOUNDARIES = set(
    " \t\n\r"
    ".,!?;:"
    "，。！？；：、"
    ")]}）】》」』"
)


def split_train_units(text: str, max_unit_chars: Optional[int]) -> Iterator[str]:
    """把长文本切成较短训练单元，降低 BPE 训练阶段的统计内存峰值。"""
    if max_unit_chars is None or max_unit_chars <= 0 or len(text) <= max_unit_chars:
        if text:
            yield text
        return

    start = 0
    text_len = len(text)
    min_soft_pos = max(1, max_unit_chars // 3)

    while start < text_len:
        hard_end = min(start + max_unit_chars, text_len)
        if hard_end >= text_len:
            unit = text[start:text_len]
            if unit.strip():
                yield unit
            break

        window = text[start:hard_end]
        split_at = -1
        for idx in range(len(window) - 1, min_soft_pos - 1, -1):
            if window[idx] in _TRAIN_UNIT_BOUNDARIES:
                split_at = start + idx + 1
                break

        if split_at <= start:
            split_at = hard_end

        unit = text[start:split_at]
        if unit.strip():
            yield unit
        start = split_at


def load_jsonl_texts(
    data_dir: str,
    max_doc_chars: Optional[int] = None,
) -> Iterator[str]:
    """流式加载 JSONL 文件的 text 字段"""
    jsonl_files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
    if not jsonl_files:
        print(f"警告: 在 {data_dir} 中未找到 JSONL 文件")
        return

    for file_path in jsonl_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = _loads(line)
                        text = _normalize_text(data.get("text", ""), max_doc_chars)
                        if text:
                            yield text
                    except Exception:
                        continue


def load_parquet_texts(
    data_dir: str,
    batch_size: int = 10000,
    max_doc_chars: Optional[int] = None,
) -> Iterator[str]:
    """流式加载 Parquet 文件的 text 字段"""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("错误: 需要安装 pyarrow: pip install pyarrow")
        return

    parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not parquet_files:
        print(f"警告: 在 {data_dir} 中未找到 Parquet 文件")
        return

    for file_path in parquet_files:
        parquet_file = pq.ParquetFile(file_path)
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["text"]):
            for text in batch.column("text"):
                text_str = _normalize_text(text.as_py(), max_doc_chars)
                if text_str:
                    yield text_str


def count_files(cci3_dir: Optional[str], fineweb_dir: Optional[str]) -> tuple:
    """统计数据文件数量"""
    jsonl_count = 0
    parquet_count = 0

    if cci3_dir and os.path.exists(cci3_dir):
        jsonl_count = len(glob.glob(os.path.join(cci3_dir, "*.jsonl")))

    if fineweb_dir and os.path.exists(fineweb_dir):
        parquet_count = len(glob.glob(os.path.join(fineweb_dir, "*.parquet")))

    return jsonl_count, parquet_count


def reservoir_sample_to_disk(
    iterator: Iterator[str],
    k: int,
    output_path: str,
    seed: int = 42,
) -> int:
    """
    Reservoir sampling，采样结果写入磁盘（低内存版本）

    Args:
        iterator: 文本迭代器
        k: 采样数量
        output_path: 输出文件路径（纯文本格式，每行一个文档）
        seed: 随机种子

    Returns:
        总文档数
    """
    random.seed(seed)
    reservoir: List[str] = []
    count = 0

    for item in iterator:
        if count < k:
            reservoir.append(item)
        else:
            j = random.randint(0, count)
            if j < k:
                reservoir[j] = item
        count += 1

    # 打乱后写入磁盘（纯文本格式，换行符替换为空格）
    print(f"  写入采样文件: {output_path}")
    random.shuffle(reservoir)
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in reservoir:
            # 将文档内的换行符替换为空格，确保每行一个文档
            text_single_line = text.replace('\n', ' ').replace('\r', ' ')
            f.write(text_single_line + '\n')

    # 释放内存
    del reservoir

    return count


def count_texts(iterator: Iterator[str], desc: str) -> int:
    """统计迭代器文本条数"""
    count = 0
    for _ in tqdm(iterator, desc=desc, unit=" docs"):
        count += 1
    return count


def _close_iterator(iterator: Iterator[str]) -> None:
    close = getattr(iterator, "close", None)
    if callable(close):
        close()


def stream_sample_to_disk(
    iterator: Iterator[str],
    total_count: int,
    k: int,
    output_path: str,
    seed: int = 42,
) -> int:
    """
    两遍流式采样（需要已知总数），精确采样 k 条，低内存
    """
    if total_count <= 0 or k <= 0:
        open(output_path, "w", encoding="utf-8").close()
        return 0

    target = min(k, total_count)
    remaining = target
    remaining_total = total_count
    rng = random.Random(seed)
    written = 0

    with open(output_path, "w", encoding="utf-8") as f:
        try:
            for text in iterator:
                if remaining <= 0:
                    break
                prob = remaining / remaining_total
                if rng.random() < prob:
                    text_single_line = text.replace("\n", " ").replace("\r", " ")
                    f.write(text_single_line + "\n")
                    written += 1
                    remaining -= 1
                remaining_total -= 1
        finally:
            _close_iterator(iterator)

    return written


def count_lines(file_path: str) -> int:
    """统计文件行数"""
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for _ in f:
            count += 1
    return count


def iter_texts_from_files(
    file_paths: List[str],
    max_doc_chars: Optional[int] = None,
    train_unit_chars: Optional[int] = None,
) -> Iterator[str]:
    """按行流式读取样本文本，避免一次性预处理占用大量内存"""
    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                text = _normalize_text(line, max_doc_chars)
                if text:
                    yield from split_train_units(text, train_unit_chars)


def parse_sample_files(sample_files: Optional[str]) -> Optional[List[str]]:
    """解析逗号分隔的采样文件列表。"""
    if not sample_files:
        return None
    files = [path.strip() for path in sample_files.split(",") if path.strip()]
    if not files:
        raise ValueError("--sample_files 至少要包含一个有效文件路径")
    missing = [path for path in files if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"--sample_files 中存在不存在的文件: {missing}")
    return files


def _format_bytes(num_bytes: int) -> str:
    """把字节数格式化为便于阅读的容量字符串"""
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def _read_cgroup_memory_limit() -> Optional[int]:
    """读取容器 cgroup 内存上限，用于定位 OOM 风险"""
    candidates = [
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ]
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                value = f.read().strip()
        except OSError:
            continue
        if not value or value == "max":
            return None
        try:
            return int(value)
        except ValueError:
            continue
    return None


def _log_runtime_context(
    output_path: str,
    sample_files: List[str],
    cjk_split_chars: Optional[int],
    disable_cjk_split: bool,
) -> None:
    """打印训练关键上下文，方便云端问题回溯"""
    print(f"进程 PID: {os.getpid()}")
    memory_limit = _read_cgroup_memory_limit()
    if memory_limit is None:
        print("cgroup 内存上限: 未限制或无法读取")
    else:
        print(f"cgroup 内存上限: {_format_bytes(memory_limit)} ({memory_limit} bytes)")
    print(f"Pre-tokenizer: {describe_bpe_pre_tokenizer(cjk_split_chars, disable_cjk_split)}")
    print(f"输出文件绝对路径: {os.path.abspath(output_path)}")

    if sample_files:
        print("采样文件信息:")
        for path in sample_files:
            try:
                size = os.path.getsize(path)
                print(f"  - {path}: {_format_bytes(size)} ({size} bytes)")
            except OSError as exc:
                print(f"  - {path}: 无法读取大小 ({exc})")


def sample_to_disk(
    cci3_dir: Optional[str],
    fineweb_dir: Optional[str],
    max_samples: int,
    temp_dir: str,
    max_doc_chars: Optional[int] = None,
    seed: int = 42,
) -> tuple:
    """
    采样数据到临时文件（低内存版本）

    Args:
        cci3_dir: CCI3-HQ 数据目录（中文）
        fineweb_dir: FineWeb-Edu 数据目录（英文）
        max_samples: 最大样本数
        temp_dir: 临时文件目录
        max_doc_chars: 单条文本最大字符数
        seed: 随机种子

    Returns:
        (zh_file, en_file, zh_count, en_count) 或 (single_file, None, count, 0)
    """
    os.makedirs(temp_dir, exist_ok=True)

    has_cci3 = cci3_dir and os.path.exists(cci3_dir)
    has_fineweb = fineweb_dir and os.path.exists(fineweb_dir)

    if has_cci3 and has_fineweb:
        # 两个数据集都有，各采样一半
        samples_per_source = max_samples // 2
        print(f"中英文均衡采样: 各 {samples_per_source:,} 样本")

        zh_file = os.path.join(temp_dir, 'zh_samples.txt')
        en_file = os.path.join(temp_dir, 'en_samples.txt')

        # 采样中文 (CCI3-HQ)
        if os.path.exists(zh_file):
            zh_count = count_lines(zh_file)
            print(f"发现已有采样文件: {zh_file}，包含 {zh_count:,} 样本，跳过采样")
        else:
            print("统计 CCI3-HQ (中文) 文档数...")
            zh_total = count_texts(
                load_jsonl_texts(cci3_dir, max_doc_chars=max_doc_chars),
                "CCI3-HQ count",
            )
            zh_target = min(samples_per_source, zh_total)
            print(f"采样 CCI3-HQ (中文)... 目标 {zh_target:,} / 总 {zh_total:,}")
            zh_count = stream_sample_to_disk(
                load_jsonl_texts(cci3_dir, max_doc_chars=max_doc_chars),
                zh_total,
                zh_target,
                zh_file,
                seed=seed,
            )
            print(f"  中文: 总 {zh_total:,}, 采样 {zh_count:,}")

        # 采样英文 (FineWeb-Edu)
        if os.path.exists(en_file):
            en_count = count_lines(en_file)
            print(f"发现已有采样文件: {en_file}，包含 {en_count:,} 样本，跳过采样")
        else:
            print("统计 FineWeb-Edu (英文) 文档数...")
            en_total = count_texts(
                load_parquet_texts(fineweb_dir, max_doc_chars=max_doc_chars),
                "FineWeb-Edu count",
            )
            en_target = min(samples_per_source, en_total)
            print(f"采样 FineWeb-Edu (英文)... 目标 {en_target:,} / 总 {en_total:,}")
            en_count = stream_sample_to_disk(
                load_parquet_texts(fineweb_dir, max_doc_chars=max_doc_chars),
                en_total,
                en_target,
                en_file,
                seed=seed + 1,
            )
            print(f"  英文: 总 {en_total:,}, 采样 {en_count:,}")

        return zh_file, en_file, zh_count, en_count

    elif has_cci3:
        # 只有中文
        zh_file = os.path.join(temp_dir, 'zh_samples.txt')
        if os.path.exists(zh_file):
            zh_count = count_lines(zh_file)
            print(f"发现已有采样文件: {zh_file}，包含 {zh_count:,} 样本，跳过采样")
        else:
            print("只有 CCI3-HQ 数据，采样中文...")
            print("统计 CCI3-HQ (中文) 文档数...")
            zh_total = count_texts(
                load_jsonl_texts(cci3_dir, max_doc_chars=max_doc_chars),
                "CCI3-HQ count",
            )
            zh_target = min(max_samples, zh_total)
            print(f"采样 CCI3-HQ (中文)... 目标 {zh_target:,} / 总 {zh_total:,}")
            zh_count = stream_sample_to_disk(
                load_jsonl_texts(cci3_dir, max_doc_chars=max_doc_chars),
                zh_total,
                zh_target,
                zh_file,
                seed=seed,
            )
            print(f"中文: 总 {zh_total:,}, 采样 {zh_count:,}")
        return zh_file, None, zh_count, 0

    elif has_fineweb:
        # 只有英文
        en_file = os.path.join(temp_dir, 'en_samples.txt')
        if os.path.exists(en_file):
            en_count = count_lines(en_file)
            print(f"发现已有采样文件: {en_file}，包含 {en_count:,} 样本，跳过采样")
        else:
            print("只有 FineWeb-Edu 数据，采样英文...")
            print("统计 FineWeb-Edu (英文) 文档数...")
            en_total = count_texts(
                load_parquet_texts(fineweb_dir, max_doc_chars=max_doc_chars),
                "FineWeb-Edu count",
            )
            en_target = min(max_samples, en_total)
            print(f"采样 FineWeb-Edu (英文)... 目标 {en_target:,} / 总 {en_total:,}")
            en_count = stream_sample_to_disk(
                load_parquet_texts(fineweb_dir, max_doc_chars=max_doc_chars),
                en_total,
                en_target,
                en_file,
                seed=seed,
            )
            print(f"英文: 总 {en_total:,}, 采样 {en_count:,}")
        return en_file, None, en_count, 0

    else:
        print("错误: 没有找到数据")
        return None, None, 0, 0


def text_iterator_with_limit(
    cci3_dir: Optional[str],
    fineweb_dir: Optional[str],
    max_doc_chars: Optional[int] = None,
) -> Iterator[str]:
    """
    生成文本迭代器（无限制模式，直接流式输出）

    Args:
        cci3_dir: CCI3-HQ 数据目录（中文）
        fineweb_dir: FineWeb-Edu 数据目录（英文）
        max_doc_chars: 单条文本最大字符数
    """
    if cci3_dir and os.path.exists(cci3_dir):
        yield from load_jsonl_texts(cci3_dir, max_doc_chars=max_doc_chars)
    if fineweb_dir and os.path.exists(fineweb_dir):
        yield from load_parquet_texts(fineweb_dir, max_doc_chars=max_doc_chars)


def configure_tokenizers_env(
    num_threads: Optional[int],
    disable_parallelism: bool,
) -> None:
    if num_threads is not None:
        os.environ["RAYON_RS_NUM_THREADS"] = str(num_threads)
    if disable_parallelism:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train_tokenizer(
    cci3_dir: Optional[str],
    fineweb_dir: Optional[str],
    vocab_size: int,
    output_path: str,
    max_samples: Optional[int] = None,
    min_frequency: int = 2,
    temp_dir: Optional[str] = None,
    max_doc_chars: Optional[int] = None,
    train_unit_chars: Optional[int] = None,
    sample_files: Optional[List[str]] = None,
    keep_samples: bool = False,
    max_token_length: Optional[int] = None,
    num_threads: Optional[int] = None,
    disable_parallelism: bool = False,
    cjk_split_chars: Optional[int] = 1,
    disable_cjk_split: bool = False,
):
    """
    训练 BPE tokenizer

    Args:
        cci3_dir: CCI3-HQ 数据目录
        fineweb_dir: FineWeb-Edu 数据目录
        vocab_size: 词表大小
        output_path: 输出路径
        max_samples: 最大训练样本数（None 不限制）
        min_frequency: 最小词频
        temp_dir: 临时文件目录（用于存储采样结果）
        max_doc_chars: 单条文本最大字符数
        train_unit_chars: BPE 训练时的最大文本单元长度
        sample_files: 已采样文本文件列表；指定后跳过原始数据采样
        keep_samples: 训练成功后保留采样文件
        max_token_length: BPE trainer 的最大 token 长度
        num_threads: tokenizers 训练线程数
        disable_parallelism: 是否禁用 tokenizers 并行
        cjk_split_chars: 连续 CJK 文本每多少字切分为一个 pre-token
        disable_cjk_split: 是否关闭 CJK 切分，恢复纯 ByteLevel 行为
    """
    configure_tokenizers_env(num_threads, disable_parallelism)

    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers

    print("=" * 60)
    print("训练 BPE Tokenizer")
    print("=" * 60)
    print(f"CCI3-HQ 目录: {cci3_dir}")
    print(f"FineWeb-Edu 目录: {fineweb_dir}")
    print(f"词表大小: {vocab_size}")
    print(f"最大样本数: {max_samples if max_samples else '不限制'}")
    print(f"最小词频: {min_frequency}")
    print(f"输出路径: {output_path}")
    if max_doc_chars is not None:
        print(f"单条文本最大字符数: {max_doc_chars}")
    if train_unit_chars is not None:
        print(f"BPE 训练单元最大字符数: {train_unit_chars}")
    if sample_files:
        print(f"复用采样文件: {sample_files}")
    if keep_samples:
        print("采样文件清理: 保留")
    if max_token_length is not None:
        print(f"BPE 最大 token 长度: {max_token_length}")
    if num_threads is not None:
        print(f"tokenizers 线程数: {num_threads}")
    if disable_parallelism:
        print("tokenizers 并行: 禁用")
    if disable_cjk_split or cjk_split_chars is None:
        print("CJK pre-tokenizer 切分: 禁用")
    else:
        print(f"CJK pre-tokenizer 切分: 每 {cjk_split_chars} 字")
    if temp_dir:
        print(f"临时目录: {temp_dir}")
    _log_runtime_context(output_path, list(sample_files or []), cjk_split_chars, disable_cjk_split)

    # 如果指定了 max_samples，先采样到磁盘
    sample_dir = None
    zh_file = None
    en_file = None
    total_samples = max_samples
    files_to_train = list(sample_files or [])

    if sample_files:
        print("\n跳过原始数据采样，直接使用已有采样文件")
        total_samples = None
    else:
        # 统计文件数量
        jsonl_count, parquet_count = count_files(cci3_dir, fineweb_dir)
        print(f"\n数据文件: {jsonl_count} JSONL + {parquet_count} Parquet")

        if jsonl_count == 0 and parquet_count == 0:
            print("错误: 没有找到任何数据文件")
            return

    if max_samples is not None and not sample_files:
        # 使用指定的临时目录或创建临时目录
        if temp_dir:
            sample_dir = os.path.join(temp_dir, 'tokenizer_samples')
        else:
            sample_dir = tempfile.mkdtemp(prefix='tokenizer_samples_')

        print(f"\n采样数据到临时目录: {sample_dir}")

        zh_file, en_file, zh_count, en_count = sample_to_disk(
            cci3_dir,
            fineweb_dir,
            max_samples,
            sample_dir,
            max_doc_chars=max_doc_chars,
        )

        if zh_file is None and en_file is None:
            print("错误: 采样失败")
            return

        total_samples = zh_count + en_count
        print(f"\n采样完成，总样本数: {total_samples:,}")
        if zh_file:
            files_to_train.append(zh_file)
        if en_file:
            files_to_train.append(en_file)

    # 创建 tokenizer
    print(f"\n配置 tokenizer...")

    tokenizer = Tokenizer(models.BPE(unk_token=None))

    # Normalizer: Unicode NFC 归一化
    tokenizer.normalizer = normalizers.NFC()

    # Pre-tokenizer: 先限制连续 CJK pre-token 长度，再走 ByteLevel 保证任意 Unicode 无 OOV
    tokenizer.pre_tokenizer = build_bpe_pre_tokenizer(
        cjk_split_chars=cjk_split_chars,
        disable_cjk_split=disable_cjk_split,
    )

    # Decoder
    tokenizer.decoder = decoders.ByteLevel()

    # Trainer
    trainer_kwargs = {
        "vocab_size": vocab_size,
        "special_tokens": SPECIAL_TOKENS_LIST,
        "show_progress": True,
        "min_frequency": min_frequency,
        "initial_alphabet": pre_tokenizers.ByteLevel.alphabet(),  # 确保所有字节都在词表中
    }
    if max_token_length is not None:
        trainer_kwargs["max_token_length"] = max_token_length
    trainer = trainers.BpeTrainer(**trainer_kwargs)

    # 训练
    print("\n开始训练 BPE tokenizer...")

    if files_to_train:
        # 从采样文件训练（流式读取，避免预处理占用过高内存）
        print(f"训练文件: {files_to_train}")
        train_iterator = iter_texts_from_files(
            files_to_train,
            max_doc_chars=max_doc_chars,
            train_unit_chars=train_unit_chars,
        )
        train_kwargs = {"trainer": trainer}
        if train_unit_chars is None and total_samples is not None:
            train_kwargs["length"] = total_samples
        tokenizer.train_from_iterator(train_iterator, **train_kwargs)
    else:
        # 直接流式读取原始数据
        tokenizer.train_from_iterator(
            text_iterator_with_limit(
                cci3_dir,
                fineweb_dir,
                max_doc_chars=max_doc_chars,
            ),
            trainer=trainer,
        )

    # 清理临时文件
    if sample_dir and os.path.exists(sample_dir) and not keep_samples:
        print(f"\n清理临时目录: {sample_dir}")
        shutil.rmtree(sample_dir)

    # 保存
    tokenizer.save(output_path)
    print(f"\nTokenizer 已保存: {output_path}")
    print(f"实际词汇表大小: {tokenizer.get_vocab_size()}")

    # 保存 HuggingFace 格式
    save_dir = os.path.splitext(output_path)[0]
    save_for_hub(tokenizer, save_dir)

    # 验证 tokenizer
    validate_tokenizer(tokenizer)

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)


def validate_tokenizer(tokenizer):
    """验证 tokenizer 质量"""
    print("\n" + "-" * 40)
    print("Tokenizer 验证")
    print("-" * 40)

    # 测试文本
    test_cases = [
        ("中文短句", "你好世界"),
        ("中文长句", "人工智能正在改变我们的生活方式，从智能手机到自动驾驶汽车。"),
        ("英文短句", "Hello World"),
        ("英文长句", "Artificial intelligence is transforming how we live and work."),
        ("中英混合", "AI人工智能是2024年最热门的技术领域"),
        ("代码片段", "def hello():\n    print('Hello, World!')"),
        ("特殊字符", "Email: test@example.com, Price: $99.99"),
        ("数学公式", "E = mc², ∑(i=1 to n) = n(n+1)/2"),
    ]

    print("\n分词测试:")
    total_chars = 0
    total_tokens = 0

    for name, text in test_cases:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)
        tokens = len(encoded.ids)
        chars = len(text)
        ratio = chars / tokens if tokens > 0 else 0

        total_chars += chars
        total_tokens += tokens

        print(f"  [{name}]")
        print(f"    原文: {text[:50]}{'...' if len(text) > 50 else ''}")
        print(f"    tokens: {tokens}, 压缩率: {ratio:.2f} chars/token")

        # 检查解码是否一致
        if decoded != text:
            print(f"    ⚠️  解码不一致: {decoded[:50]}")

    # 总体统计
    overall_ratio = total_chars / total_tokens if total_tokens > 0 else 0
    print(f"\n总体压缩率: {overall_ratio:.2f} chars/token")

    # 检查特殊 token
    print("\n特殊 token ID:")
    for name, token in SPECIAL_TOKENS.items():
        token_id = tokenizer.token_to_id(token)
        print(f"  {token}: {token_id}")


def save_for_hub(tokenizer, save_dir: str):
    """保存为 HuggingFace Hub 格式"""
    from transformers import PreTrainedTokenizerFast
    from config import DEFAULT_MAX_SEQ_LEN

    os.makedirs(save_dir, exist_ok=True)

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=SPECIAL_TOKENS["im_start_token"],
        eos_token=SPECIAL_TOKENS["eos_token"],
        pad_token=SPECIAL_TOKENS["pad_token"],
        unk_token=None,
        additional_special_tokens=[
            SPECIAL_TOKENS["im_start_token"],
            SPECIAL_TOKENS["im_end_token"],
        ],
        model_max_length=DEFAULT_MAX_SEQ_LEN,
    )

    # ChatML 模板
    hf_tokenizer.chat_template = (
        "{% for message in messages %}"
        "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )

    hf_tokenizer.save_pretrained(save_dir)
    print(f"HuggingFace 格式已保存: {save_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="BPE Tokenizer 训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 完整训练
    python train_tokenizer.py \\
        --cci3_dir /root/autodl-fs/cci3_hq_30/data \\
        --fineweb_dir /root/autodl-fs/fineweb_edu_30b/sample/100BT \\
        --vocab_size 65000

    # 限制样本数 + 指定临时目录
    python train_tokenizer.py \\
        --cci3_dir /root/autodl-fs/cci3_hq_30/data \\
        --fineweb_dir /root/autodl-fs/fineweb_edu_30b/sample/100BT \\
        --vocab_size 65000 \\
        --max_samples 10000000 \\
        --temp_dir /root/autodl-tmp
        """,
    )

    parser.add_argument(
        "--cci3_dir",
        type=str,
        default=None,
        help="CCI3-HQ JSONL 数据目录",
    )
    parser.add_argument(
        "--fineweb_dir",
        type=str,
        default=None,
        help="FineWeb-Edu Parquet 数据目录",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=DEFAULT_BPE_VOCAB_SIZE,
        help=f"词表大小 (默认: {DEFAULT_BPE_VOCAB_SIZE})",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=DEFAULT_BPE_MODEL_PATH,
        help=f"输出路径 (默认: {DEFAULT_BPE_MODEL_PATH})",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大训练样本数 (默认: 不限制，建议 500 万-1000 万)",
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=2,
        help="最小词频 (默认: 2)",
    )
    parser.add_argument(
        "--temp_dir",
        type=str,
        default=None,
        help="临时文件目录（用于存储采样结果，默认系统临时目录）",
    )
    parser.add_argument(
        "--sample_files",
        type=str,
        default=None,
        help="逗号分隔的已采样文本文件列表；指定后跳过原始数据扫描/采样",
    )
    parser.add_argument(
        "--keep_samples",
        action="store_true",
        help="训练成功后保留采样文件，不清理 temp_dir/tokenizer_samples",
    )
    parser.add_argument(
        "--max_doc_chars",
        type=int,
        default=None,
        help="限制每条文本最大字符数（用于控制内存峰值）",
    )
    parser.add_argument(
        "--train_unit_chars",
        type=int,
        default=None,
        help="BPE 训练时将长文本切成不超过该字符数的短单元，降低内存峰值",
    )
    parser.add_argument(
        "--max_token_length",
        type=int,
        default=None,
        help="限制 BPE 生成 token 的最大长度，降低异常长 token 带来的内存峰值",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=None,
        help="限制 tokenizers 训练线程数（设置 RAYON_RS_NUM_THREADS）",
    )
    parser.add_argument(
        "--disable_parallelism",
        action="store_true",
        help="禁用 tokenizers 并行（设置 TOKENIZERS_PARALLELISM=false）",
    )
    parser.add_argument(
        "--cjk_split_chars",
        type=int,
        default=1,
        help="连续 CJK 文本每多少字切分为一个 pre-token，默认 1，用于降低 BPE 训练内存峰值",
    )
    parser.add_argument(
        "--disable_cjk_split",
        action="store_true",
        help="关闭 CJK pre-tokenizer 切分，恢复旧的纯 ByteLevel 行为",
    )

    args = parser.parse_args()

    try:
        sample_files = parse_sample_files(args.sample_files)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    if not sample_files and not args.cci3_dir and not args.fineweb_dir:
        parser.error("至少需要指定 --cci3_dir / --fineweb_dir，或使用 --sample_files")
    if args.max_doc_chars is not None and args.max_doc_chars <= 0:
        parser.error("--max_doc_chars 必须是正整数")
    if args.train_unit_chars is not None and args.train_unit_chars <= 0:
        parser.error("--train_unit_chars 必须是正整数")
    if args.max_token_length is not None and args.max_token_length <= 0:
        parser.error("--max_token_length 必须是正整数")
    if args.num_threads is not None and args.num_threads <= 0:
        parser.error("--num_threads 必须是正整数")
    if args.cjk_split_chars is not None and args.cjk_split_chars <= 0:
        parser.error("--cjk_split_chars 必须是正整数")

    train_tokenizer(
        cci3_dir=args.cci3_dir,
        fineweb_dir=args.fineweb_dir,
        vocab_size=args.vocab_size,
        output_path=args.output_path,
        max_samples=args.max_samples,
        min_frequency=args.min_frequency,
        temp_dir=args.temp_dir,
        max_doc_chars=args.max_doc_chars,
        train_unit_chars=args.train_unit_chars,
        sample_files=sample_files,
        keep_samples=args.keep_samples,
        max_token_length=args.max_token_length,
        num_threads=args.num_threads,
        disable_parallelism=args.disable_parallelism,
        cjk_split_chars=args.cjk_split_chars,
        disable_cjk_split=args.disable_cjk_split,
    )


if __name__ == "__main__":
    main()
