"""在租用 GPU 前验证训练材料的摘要、长度、tokenizer 和中英文分片。"""
import argparse
from collections import Counter
import hashlib
import json
import logging
from pathlib import Path
import shutil
import sys

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "datasets")]
from config import PretrainConfig
from training_utils import validate_training_options
from download_bilingual_sources import sha256_file
from prepare_bilingual import normalized, text_digest, SPECIAL

logger = logging.getLogger(__name__)


def inspect_memmap(folder, tokenizer, tokenizer_hash):
    metadata = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    path = folder / "train.bin"
    if not (metadata['packed'] and metadata['partial_blocks'] == 0):
        raise ValueError(f'非完整 packing: {folder}')
    if not (metadata['vocab_size'] == tokenizer.get_vocab_size()):
        raise ValueError(f'词表大小不一致: {folder}')
    if not (metadata['tokenizer_sha256'] == tokenizer_hash):
        raise ValueError(f'tokenizer 不一致: {folder}')
    if not (path.stat().st_size == metadata['num_samples'] * metadata['seq_len'] * 2):
        raise ValueError(f'数据长度不一致: {folder}')
    if not (metadata['seq_len'] == metadata['train_seq_len'] + 1):
        raise ValueError(f'训练标签位移长度不一致: {folder}')
    actual_hash = sha256_file(path)
    if not (actual_hash == metadata['data_sha256']):
        raise ValueError(f'数据摘要不一致: {folder}')
    data = np.memmap(path, mode="r", dtype=np.uint16)
    if not (int(data.max()) < tokenizer.get_vocab_size()):
        raise ValueError(f'token ID 越界: {folder}')
    if not (not np.any(data == tokenizer.token_to_id('<|pad|>'))):
        raise ValueError(f'完整 packing 含 PAD: {folder}')
    logger.info("memmap 通过: %s，输入 token=%d", folder, metadata["input_tokens"])
    return metadata


def inspect_sft(root, tokenizer, seq_len, probes):
    counts, split_keys = Counter(), {}
    for split in ("train", "val"):
        keys = set()
        for line in (root / "data" / f"sft_{split}.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            turns = row["conversations"]
            if not (turns[-2]['role'] == 'user' and turns[-1]['role'] == 'assistant'):
                raise ValueError("材料一致性检查失败: turns[-2]['role'] == 'user' and turns[-1]['role'] == 'assistant'")
            key = text_digest(turns[-2]["content"])
            if not (key not in keys):
                raise ValueError(f'SFT 分片内重复问题: {split}')
            keys.add(key)
            rendered = "".join(f"<|im_start|>{turn['role']}\n{turn['content']}<|im_end|>\n" for turn in turns)
            ids = tokenizer.encode(rendered).ids
            if not (len(ids) <= seq_len + 1):
                raise ValueError('SFT 超长样本会截断答案')
            if not (tokenizer.token_to_id('<|im_end|>') in ids):
                raise ValueError('缺失回答终止符')
            joined = normalized(turns[-2]["content"] + turns[-1]["content"])
            if not (not any((probe in joined for probe in probes))):
                raise ValueError('SFT 与冻结验收题重叠')
            counts[f"{row['language']}_{split}"] += 1
        split_keys[split] = keys
    if not (split_keys['train'].isdisjoint(split_keys['val'])):
        raise ValueError('SFT 训练与验证问题重复')
    if not (all((counts[f'{language}_{split}'] for language in ('zh', 'en') for split in ('train', 'val')))):
        raise ValueError("材料一致性检查失败: all((counts[f'{language}_{split}'] for language in ('zh', 'en') for split in ('train', 'val')))")
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description="只读取并验证正式训练材料")
    parser.add_argument("--data-root", default="checkpoints/bilingual_20261011")
    parser.add_argument("--config", default="configs/colab_a100_1b_pretrain.json")
    parser.add_argument("--eval-file", default="evals/bilingual_acceptance_v1.jsonl")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.world_size <= 0:
        raise ValueError("world_size 必须为正数")
    root = Path(args.data_root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    config = PretrainConfig(**json.loads(Path(args.config).read_text(encoding="utf-8")))
    validate_training_options(config)
    questions = Path(args.eval_file)
    if not (sha256_file(questions) == manifest['recipe']['eval_sha256']):
        raise ValueError('验收题在数据清洗后被修改')
    rows = [json.loads(line) for line in questions.read_text(encoding="utf-8").splitlines()]
    probes = [normalized(row["prompt"]) for row in rows if len(normalized(row["prompt"])) >= 20]
    for split in ("train", "val"):
        if sha256_file(root / "data" / f"sft_{split}.jsonl") != manifest["sft"][f"{split}_sha256"]:
            raise ValueError(f"SFT {split} 内容摘要与材料清单不一致")
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    tokenizer_hash = sha256_file(root / "tokenizer.json")
    if not (tokenizer_hash == manifest['tokenizer_sha256']):
        raise ValueError("材料一致性检查失败: tokenizer_hash == manifest['tokenizer_sha256']")
    if not ([tokenizer.token_to_id(token) for token in SPECIAL] == [0, 1, 2, 3]):
        raise ValueError('材料一致性检查失败: [tokenizer.token_to_id(token) for token in SPECIAL] == [0, 1, 2, 3]')
    roundtrip_texts = ["中文与 English 都需要保留。", "I'm learning English. 12345", "你好，世界！🌍"]
    if not (all((tokenizer.decode(tokenizer.encode(text).ids) == text for text in roundtrip_texts))):
        raise ValueError('材料一致性检查失败: all((tokenizer.decode(tokenizer.encode(text).ids) == text for text in roundtrip_texts))')
    memmaps = {split: inspect_memmap(root / "data/pretrain" / split, tokenizer, tokenizer_hash)
               for split in ("train", "val", "val_zh", "val_en")}
    if not (all((meta['train_seq_len'] == config.max_seq_len for meta in memmaps.values()))):
        raise ValueError("材料一致性检查失败: all((meta['train_seq_len'] == config.max_seq_len for meta in memmaps.values()))")
    effective_tokens = config.batch_size * config.accumulation_steps * args.world_size * config.max_seq_len
    steps = memmaps["train"]["num_samples"] // (config.batch_size * config.accumulation_steps * args.world_size) * config.epochs
    if not (steps > 0):
        raise ValueError('数据不足一个完整有效批量')
    stop = min(steps, config.max_steps) if config.max_steps else steps
    report = {"status": "data_ready_gpu_validation_pending", "tokenizer_sha256": tokenizer_hash,
              "vocab_size": tokenizer.get_vocab_size(), "memmaps": memmaps,
              "sft": inspect_sft(root, tokenizer, config.max_seq_len, probes),
              "effective_input_tokens_per_update": effective_tokens,
              "available_updates": steps, "planned_updates": stop,
              "planned_input_tokens": stop * effective_tokens,
              "lr_schedule_steps": config.schedule_steps or steps,
              "free_disk_gib": shutil.disk_usage(root).free / 1024 ** 3,
              "gpu_allocated_by_this_check": False,
              "remaining_gates": ["正式 GPU 上的吞吐与数值对照", "实际 GPU 价格与本轮预算", "checkpoint 持久存储和中断恢复演练"]}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("训练材料检查通过: 更新步=%d，实际训练输入 token=%d；本次没有分配 GPU", stop, stop * effective_tokens)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
