# -*- coding: utf-8 -*-
"""下载有界的公开中文语料样本，按文档去重和隔离验证集，再训练新 tokenizer 并打包。"""
import argparse
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

DATASET = "opencsg/chinese-fineweb-edu"


def prepare(output, pages=20, seed=1337, seq_len=512):
    """每页最多 100 条；该样本用于流水线验证，不代表完整语料分布。"""
    if not 1 <= pages <= 100:
        raise ValueError("小实验限定 1～100 页，每页最多 100 条")
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    response = requests.get(f"https://huggingface.co/api/datasets/{DATASET}", timeout=30)
    response.raise_for_status(); revision = response.json()["sha"]
    # 数据查看器仅转换了部分全量数据；从这一可见范围的不同位置确定性抽样。
    offsets = random.Random(seed).sample(range(0, 1_000_000, 100), pages)

    def fetch(offset):
        response = requests.get("https://datasets-server.huggingface.co/rows", timeout=60,
            params={"dataset": DATASET, "config": "default", "split": "train", "offset": offset, "length": 100})
        response.raise_for_status()
        rows = response.json()["rows"]
        print(f"语料读取: offset={offset}, rows={len(rows)}", flush=True)
        return [row["row"]["text"] for row in rows]

    with ThreadPoolExecutor(max_workers=4) as pool:
        raw = [text for batch in pool.map(fetch, offsets) for text in batch]
    unique = {hashlib.sha256(text.strip().encode()).hexdigest(): text.strip()
              for text in raw if isinstance(text, str) and len(text.strip()) >= 100}
    splits = {"train": [], "val": []}
    for digest, text in unique.items():
        splits["val" if int(digest[:8], 16) % 20 == 0 else "train"].append(text)
    if not all(splits.values()):
        raise ValueError("样本不足以产生训练和验证两个分片，请增加页数")
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(splits["train"], trainers.BpeTrainer(vocab_size=32000, min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<|pad|>", "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<unk>"]))
    tokenizer.save(str(output / "tokenizer.json"))
    tokenizer_hash = hashlib.sha256((output / "tokenizer.json").read_bytes()).hexdigest()
    manifest = {"source": DATASET, "source_url": f"https://huggingface.co/datasets/{DATASET}",
        "source_revision_at_download": revision, "seed": seed, "offsets": offsets,
        "raw_documents": len(raw), "unique_documents": len(unique), "tokenizer_sha256": tokenizer_hash,
        "limitation": "查看器可见范围的有界样本；tokenizer 仅用于先导实验，正式训练需在更大、平衡的语料上重新训练", "splits": {}}
    for split, texts in splits.items():
        folder = output / split; folder.mkdir(exist_ok=True)
        tokens = [token for text in texts for token in tokenizer.encode(text).ids + [1]]
        count = len(tokens) // (seq_len + 1)
        if not count:
            raise ValueError(f"{split} token 数不足一个 block")
        np.asarray(tokens[:count * (seq_len + 1)], dtype=np.uint16).tofile(folder / "train.bin")
        metadata = {"num_samples": count, "seq_len": seq_len + 1, "train_seq_len": seq_len,
                    "vocab_size": tokenizer.get_vocab_size(), "pad_id": 0, "eos_id": 1,
                    "packed": True, "partial_blocks": 0, "tokenizer_sha256": tokenizer_hash,
                    "documents": len(texts), "total_tokens": count * (seq_len + 1)}
        (folder / "meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["splits"][split] = metadata
    manifest["sample_sha256"] = hashlib.sha256("\n".join(sorted(unique)).encode()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)



def prepare_sft(output, pages=5, seed=1447, max_tokens=513):
    """用同一 tokenizer 筛选公开中文短问答，保存抽样位置以便复现。"""
    if not 1 <= pages <= 100:
        raise ValueError("SFT 小实验限定 1～100 页")
    output = Path(output)
    tokenizer = Tokenizer.from_file(str(output / "tokenizer.json"))
    dataset = "opencsg/smoltalk-chinese"
    offsets = random.Random(seed).sample(range(0, 10000, 100), pages)
    def fetch(offset):
        response = requests.get("https://datasets-server.huggingface.co/rows", timeout=60,
            params={"dataset": dataset, "config": "default", "split": "train", "offset": offset, "length": 100})
        response.raise_for_status()
        rows = response.json()["rows"]
        print(f"SFT 读取: offset={offset}, rows={len(rows)}", flush=True)
        return rows
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = [row for batch in pool.map(fetch, offsets) for row in batch]
    selected = []
    for row in rows:
        turns = row["row"].get("conversations")
        if not isinstance(turns, list) or len(turns) < 2:
            continue
        turns = turns[:2]
        if [turn.get("role") for turn in turns] != ["user", "assistant"]:
            continue
        if not all(isinstance(turn.get("content"), str) and turn["content"].strip() for turn in turns):
            continue
        text = "".join(f"<|im_start|>{turn['role']}\n{turn['content']}<|im_end|>\n" for turn in turns)
        if len(tokenizer.encode(text).ids) <= max_tokens:
            selected.append({"conversations": turns})
    if not selected:
        raise ValueError("过滤后没有完整短问答，请增加采样量或长度上限")
    (output / "sft_pilot.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in selected), encoding="utf-8")
    manifest = {"source": dataset, "offsets": offsets, "raw": len(rows), "selected": len(selected),
                "filter": f"仅首轮 user/assistant，完整 ChatML 不超过 {max_tokens} token，不截断答案"}
    (output / "sft_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SFT 筛选: {len(selected)}/{len(rows)} 条", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="准备有界中文训练实验语料")
    parser.add_argument("--output", required=True)
    parser.add_argument("--pages", type=int, default=20)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--sft", action="store_true", help="同时准备 500 条候选中文问答并过滤长样本")
    args = parser.parse_args()
    prepare(args.output, pages=args.pages, seq_len=args.seq_len)

    if args.sft:
        prepare_sft(args.output, max_tokens=args.seq_len + 1)
