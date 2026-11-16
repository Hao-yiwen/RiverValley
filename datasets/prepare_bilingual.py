"""准备可审计的中英文语料、冻结 tokenizer、独立验证集与按 token 配比的 memmap。"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from download_bilingual_sources import sha256_file

logger = logging.getLogger(__name__)
SPECIAL = ["<|pad|>", "<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def normalized(text):
    """统一兼容字符和空白，用于去重而不改变保留文档的标点与换行。"""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).casefold()


def text_digest(text):
    return hashlib.sha256(normalized(text).encode("utf-8")).hexdigest()


def language_ok(text, language):
    """轻量脚本比例检查，不把它表述为通用语言识别器。"""
    han = len(re.findall(r"[\u3400-\u9fff]", text))
    latin = len(re.findall(r"[a-zA-Z]", text))
    letters = max(1, han + latin)
    return han / letters >= 0.5 if language == "zh" else latin / letters >= 0.95


def reject_reason(text, language, probes=()):
    """拒绝明显空白、乱码、模板重复或验收题污染，记录每类过滤数量。"""
    if not isinstance(text, str) or not 200 <= len(text.strip()) <= 80000:
        return "length"
    if "\ufffd" in text or any(token in text for token in SPECIAL):
        return "encoding_or_control_token"
    if not language_ok(text, language):
        return "language"
    if re.search(r"(.)\1{19,}", text) or max(Counter(text).values()) / len(text) > 0.3:
        return "character_repetition"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 8 and len(set(lines)) / len(lines) < 0.6:
        return "line_repetition"
    if len(re.findall(r"https?://", text)) > max(4, len(text) // 500):
        return "url_density"
    compact = normalized(text)
    if any(probe in compact for probe in probes):
        return "evaluation_overlap"
    return None


class NearDeduplicator:
    """对采样字符 5-gram 的 SimHash 做汉明距离检查；它是近重复启发式而非语义去重。"""
    def __init__(self):
        self.exact = set()
        self.bands = defaultdict(list)

    def add(self, text):
        compact = normalized(text)
        digest = hashlib.sha256(compact.encode("utf-8")).hexdigest()
        if digest in self.exact:
            return None, "exact_duplicate"
        self.exact.add(digest)
        # 确定性采样限制每篇文档的开销；四个 16-bit 桶用于候选查找。
        stride = max(1, (len(compact) - 4) // 128)
        grams = sorted(set(compact[i:i + 5] for i in range(0, len(compact) - 4, stride)))
        raw = b"".join(hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest() for gram in grams)
        bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8).reshape(-1, 8), axis=1)
        fingerprint = int.from_bytes(np.packbits(bits.sum(0) * 2 >= len(bits)).tobytes(), "big")
        candidates = set()
        keys = [(band, (fingerprint >> (band * 16)) & 65535) for band in range(4)]
        for key in keys:
            candidates.update(self.bands[key])
        if any((fingerprint ^ other).bit_count() <= 3 for other in candidates):
            return None, "near_duplicate"
        for key in keys:
            self.bands[key].append(fingerprint)
        return digest, None


def read_rows(paths):
    """逐批读取 Parquet，避免把整个源文件展开到内存。"""
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
            yield from batch.to_pylist()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def clean_language(raw, output, language, probes):
    # macOS 子进程采用 spawn，需显式初始化日志才能看到耗时清洗阶段的进度。
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    folder = output / "clean"
    folder.mkdir(exist_ok=True)
    report_path = folder / f"{language}_report.json"
    paths = sorted(raw.glob(f"pretrain_{language}_*.parquet"))
    raw_files = {path.name: path.stat().st_size for path in paths}
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous.get("raw_files") == raw_files:
            logger.info("复用已完成的清洗结果: %s", report_path)
            return previous
        if (output / "tokenizer.json").exists():
            raise ValueError("原始数据已改变而 tokenizer 已冻结，请在新输出目录准备材料")
        logger.info("原始文件集合改变，重新清洗: %s", language)
    if not paths:
        raise FileNotFoundError(f"缺少 {language} 原始 Parquet")
    stats, dedup = Counter(), NearDeduplicator()
    files = {split: (folder / f"{language}_{split}.jsonl").open("w", encoding="utf-8") for split in ("train", "val")}
    try:
        for index, row in enumerate(read_rows(paths)):
            stats["raw"] += 1
            text = row.get("text")
            reason = reject_reason(text, language, probes)
            if reason:
                stats[reason] += 1
                continue
            digest, reason = dedup.add(text)
            if reason:
                stats[reason] += 1
                continue
            # 先按文档去重和拆分，再训练 tokenizer 和 packing；两个语言单独保留验证集。
            split = "val" if int(digest[:8], 16) % 100 == 0 else "train"
            record = {"id": digest, "language": language, "text": text.strip(), "source_row": index}
            files[split].write(json.dumps(record, ensure_ascii=False) + "\n")
            stats[split] += 1
            stats[f"{split}_characters"] += len(text)
            if stats["raw"] % 10000 == 0:
                logger.info("清洗进度: 语言=%s，原始=%d，保留=%d", language, stats["raw"], stats["train"] + stats["val"])
    finally:
        for sink in files.values():
            sink.close()
    if not stats["train"] or not stats["val"]:
        raise ValueError(f"{language} 清洗后训练或验证为空")
    report = {**dict(stats), "raw_files": raw_files}
    write_json(report_path, report)
    return report


def train_tokenizer(output):
    target = output / "tokenizer.json"
    if target.exists():
        return Tokenizer.from_file(str(target))
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    def texts():
        # 中文通常比英文每个字符携带更多 token；两种语言均进入词表训练。
        for language, budget in [("zh", 12_000_000), ("en", 24_000_000)]:
            used = 0
            with (output / "clean" / f"{language}_train.jsonl").open(encoding="utf-8") as source:
                for line in source:
                    row = json.loads(line)
                    if int(row["id"][:8], 16) % 7:
                        continue
                    text = row["text"][:min(8000, budget - used)]
                    if text:
                        yield text
                        used += len(text)
                    if used >= budget:
                        break
            logger.info("词表训练取样: 语言=%s，字符=%d", language, used)
    tokenizer.train_from_iterator(texts(), trainers.BpeTrainer(
        vocab_size=32000, min_frequency=3, max_token_length=32,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=SPECIAL))
    tokenizer.save(str(target))
    logger.info("词表冻结: vocab=%d，SHA256=%s", tokenizer.get_vocab_size(), sha256_file(target))
    return tokenizer


def pack_language(output, tokenizer, language, split, max_blocks, seq_len):
    folder = output / "tokenized"
    folder.mkdir(exist_ok=True)
    target = folder / f"{language}_{split}.bin"
    width, written, buffer, examples = seq_len + 1, 0, [], []
    next_log = 100000
    path = output / "clean" / f"{language}_{split}.jsonl"
    with path.open(encoding="utf-8") as source, target.open("wb") as sink:
        def emit(texts):
            nonlocal buffer, written
            for encoded in tokenizer.encode_batch(texts):
                buffer.extend(encoded.ids)
                buffer.append(tokenizer.token_to_id("<|endoftext|>"))
            blocks = min(len(buffer) // width, max_blocks - written)
            np.asarray(buffer[:blocks * width], dtype=np.uint16).tofile(sink)
            buffer = buffer[blocks * width:]
            written += blocks
        for line in source:
            examples.append(json.loads(line)["text"])
            if len(examples) == 128:
                emit(examples)
                examples.clear()
                if written >= next_log:
                    logger.info("编码进度: 语言=%s，分片=%s，块数=%d/%d", language, split, written, max_blocks)
                    next_log += 100000
                if written >= max_blocks:
                    break
        if examples and written < max_blocks:
            emit(examples)
    if not written:
        raise ValueError(f"{language}/{split} 没有完整 token 块")
    logger.info("编码完成: 语言=%s，分片=%s，块数=%d，输入 token=%d", language, split, written, written * seq_len)
    return target, written


def write_memmap(folder, blocks, tokenizer_hash, vocab_size, seq_len, language_counts):
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "train.bin"
    count = 0
    with target.open("wb") as sink:
        for batch in blocks:
            np.asarray(batch, dtype=np.uint16).tofile(sink)
            count += len(batch)
    metadata = {"num_samples": count, "seq_len": seq_len + 1, "train_seq_len": seq_len,
                "vocab_size": vocab_size, "pad_id": 0, "eos_id": 1, "packed": True,
                "partial_blocks": 0, "tokenizer_sha256": tokenizer_hash,
                "data_sha256": sha256_file(target), "input_tokens": count * seq_len,
                "total_tokens": count * (seq_len + 1), "language_blocks": language_counts}
    write_json(folder / "meta.json", metadata)
    return metadata


def build_memmaps(output, tokenizer, token_budget, zh_fraction, seq_len, seed):
    total_blocks = token_budget // seq_len
    counts = {"zh": int(total_blocks * zh_fraction)}
    counts["en"] = total_blocks - counts["zh"]
    tokenizer_hash = sha256_file(output / "tokenizer.json")
    sources, available, result = {}, {}, {}
    for language in ("zh", "en"):
        path, available[language] = pack_language(output, tokenizer, language, "train", counts[language], seq_len)
        sources[language] = np.memmap(path, mode="r", dtype=np.uint16).reshape(-1, seq_len + 1)
        val_path, val_count = pack_language(output, tokenizer, language, "val", 2048, seq_len)
        val = np.memmap(val_path, mode="r", dtype=np.uint16).reshape(-1, seq_len + 1)
        result[f"val_{language}"] = write_memmap(output / "data/pretrain" / f"val_{language}",
            [val], tokenizer_hash, tokenizer.get_vocab_size(), seq_len, {language: val_count})
    # 原始数据不足时停止，不通过重复采样伪装成足量独立训练材料。
    if any(available[language] < counts[language] for language in counts):
        raise ValueError(f"语料不足目标预算，需要补充锁定的源文件: 需要={counts}，已有={available}")
    order = np.random.default_rng(seed).permutation(total_blocks)
    def mixed_blocks():
        for start in range(0, total_blocks, 4096):
            ids = order[start:start + 4096]
            batch = np.empty((len(ids), seq_len + 1), dtype=np.uint16)
            zh = ids < counts["zh"]
            batch[zh] = sources["zh"][ids[zh]]
            batch[~zh] = sources["en"][ids[~zh] - counts["zh"]]
            yield batch
    result["train"] = write_memmap(output / "data/pretrain/train", mixed_blocks(), tokenizer_hash,
                                    tokenizer.get_vocab_size(), seq_len, counts)
    validation = []
    for language in ("zh", "en"):
        path = output / "data/pretrain" / f"val_{language}" / "train.bin"
        validation.append(np.memmap(path, mode="r", dtype=np.uint16).reshape(-1, seq_len + 1))
    val = np.concatenate(validation)
    np.random.default_rng(seed).shuffle(val)
    result["val"] = write_memmap(output / "data/pretrain/val", [val], tokenizer_hash,
        tokenizer.get_vocab_size(), seq_len, {lang: result[f"val_{lang}"]["num_samples"] for lang in ("zh", "en")})
    return result


def sft_record(row, filename, language, max_assistant_turns=4):
    """统一对话格式，保留完整的前四轮及来源；仅去掉末尾未回答的问题。"""
    group = filename.removesuffix('_0000.parquet').removeprefix('sft_zh_cqia_')
    if filename.startswith('sft_zh_cqia_'):
        instruction, context, answer = row.get('instruction'), row.get('input', ''), row.get('output')
        if not isinstance(instruction, str) or not isinstance(context, str) or not isinstance(answer, str):
            return None
        question = instruction.strip() + ('\n\n' + context.strip() if context.strip() else '')
        turns = [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}]
        source = 'm-a-p/COIG-CQIA/' + group
    elif filename.startswith(('sft_zh_alpaca_gpt4_', 'sft_zh_magpie_')):
        turns = [{'role': {'human': 'user', 'gpt': 'assistant'}.get(t.get('from')),
                  'content': t.get('value')} for t in row.get('conversations', [])]
        group, source = 'alpaca_gpt4_zh', 'FreedomIntelligence/alpaca-gpt4-chinese'
        if filename.startswith('sft_zh_magpie_'):
            group, source = 'magpie_zh', 'Magpie-Align/Magpie-Qwen2-Pro-200K-Chinese'
    else:
        turns = row.get('conversations', row.get('messages'))
        group = 'smoltalk_zh' if language == 'zh' else str(row.get('source', 'smol-smoltalk'))
        source = 'opencsg/smoltalk-chinese' if language == 'zh' else 'HuggingFaceTB/smol-smoltalk/' + group
    if not isinstance(turns, list):
        return None
    selected, expected, assistants = [], 'user', 0
    for turn in turns:
        if not isinstance(turn, dict):
            return None
        role, content = turn.get('role'), turn.get('content')
        if role not in ('system', 'user', 'assistant') or not isinstance(content, str) or not content.strip():
            return None
        if role == 'system':
            if selected:
                return None
        elif role != expected:
            return None
        else:
            expected = 'assistant' if role == 'user' else 'user'
        selected.append({'role': role, 'content': content.strip()})
        if role == 'assistant':
            assistants += 1
            if assistants == max_assistant_turns:
                break
    if selected and selected[-1]['role'] == 'user':
        selected.pop()
    if not assistants:
        return None
    return {'conversations': selected, 'language': language, 'source': source, 'group': group,
            'source_file': filename, 'source_copyright': row.get('copyright'),
            'source_human_verified': row.get('human_verified'), 'task_type': row.get('task_type', row.get('classify'))}


def sft_reject_reason(record):
    """按首阶段短问答目标筛选；源标签不准确时同时检查正文。"""
    question, answer = [t['content'] for t in record['conversations'][-2:]]
    joined = question + '\n' + answer
    if record['group'] in ('wiki', 'zhihu', 'chinese_traditional', 'wikihow', 'self-oss-instruct'):
        return 'deferred_source'
    if '```' in joined or re.search(r'\b(?:python|javascript|sql|function|class|algorithm)\b|编写.*程序|写.*代码|每一个字符分类|分词', question, re.I):
        return 'code_or_tagging_task'
    if re.search(r'治疗|诊断|服药|用药|处方|淋病|抑郁症|肩周炎|现任|getState|GeomycellaAI|推-make|\b(?:diagnosis|medication|prescription|latest|current president)\b', joined, re.I):
        return 'scope_or_noise'
    if re.search(r'扮演|模拟.*角色|play the role|assume the role|pretend|you are (?:a|an|the)\b', question, re.I):
        return 'roleplay'
    if re.search(r'ChatGPT|OpenAI|GPT-4', answer, re.I):
        return 'provider_identity'
    if len(normalized(answer)) >= 40 and normalized(answer) in normalized(question):
        return 'answer_inside_prompt'
    if len(answer) > (600 if record['language'] == 'zh' else 1600) or len(question) > 2000:
        return 'long_text'
    return None


def finalize_sft_rows(rows, seed):
    """复核发现的问题按任务类别统一过滤，再确定性重排并恢复中英条数比例。"""
    kept, stats = [], Counter()
    for row in rows:
        question, answer = [t['content'] for t in row['conversations'][-2:]]
        reason = None
        if row['group'] == 'alpaca_gpt4_zh' and re.search(r'字母|拼写|单词|重新排列|沟通类型|分类|判断.*类型', question):
            reason = 'translated_language_task'
        elif row['language'] == 'zh' and (question.rstrip().endswith((':', '：')) or
              (re.search(r'给定|以下|下面|所给|提供的', question) and '\n' not in question and len(question) < 80)):
            reason = 'missing_context'
        elif re.search(r'\brole\s*:|pretend|role.?play', question, re.I):
            reason = 'roleplay'
        elif row['language'] == 'en' and re.search(r'data ?set|statistics|statistical|correlation|standard deviation|sales trend', question, re.I):
            reason = 'advanced_statistics'
        elif row['language'] == 'zh' and re.search(r'法律建议|金融建议|GPT-\d', question):
            reason = 'out_of_scope'
        if reason:
            stats[f'review_{row["language"]}_{reason}'] += 1
        else:
            kept.append(row)
    # 对初次处理和已有材料复核使用同一顺序，不依赖 train/val 文件拼接的先后。
    kept.sort(key=lambda row: hashlib.sha256(f'{seed}:{row["id"]}'.encode('utf-8')).hexdigest())
    zh = [r for r in kept if r['language'] == 'zh']
    en = [r for r in kept if r['language'] == 'en']
    target_en = round(len(zh) * 2 / 3)
    if len(en) < target_en:
        zh = zh[:round(len(en) * 1.5)]
    else:
        en = en[:target_en]
    chosen = {r['id'] for r in zh + en}
    return [r for r in kept if r['id'] in chosen], dict(stats)


def save_sft_rows(rows, output, report):
    """写入固定问题分片，并以实际内容更新数量和摘要。"""
    folder = output / 'data'
    folder.mkdir(exist_ok=True)
    stats = Counter()
    with (folder / 'sft_train.jsonl').open('w', encoding='utf-8') as train, (folder / 'sft_val.jsonl').open('w', encoding='utf-8') as val:
        for row in rows:
            split = 'val' if int(row['id'][:8], 16) % 20 == 0 else 'train'
            (val if split == 'val' else train).write(json.dumps(row, ensure_ascii=False) + '\n')
            stats[f"{row['language']}_{split}"] += 1
            stats[f'{split}_tokens'] += row['tokens']
    if not all(stats[f'{lang}_{split}'] for lang in ('zh', 'en') for split in ('train', 'val')):
        raise ValueError('SFT 缺少某一语言的训练或验证样本')
    report.update(dict(stats))
    report.update(selected_by_source=dict(Counter(row['source'] for row in rows)), pipeline_version=4,
                  train_sha256=sha256_file(folder / 'sft_train.jsonl'), val_sha256=sha256_file(folder / 'sft_val.jsonl'))
    write_json(output / 'sft_manifest.json', report)
    logger.info('SFT 准备完成: 训练=%d，验证=%d，来源=%s', stats['zh_train'] + stats['en_train'],
                stats['zh_val'] + stats['en_val'], report['selected_by_source'])
    return report


def prepare_sft(raw, output, tokenizer, seq_len, probes, seed):
    from langid.langid import LanguageIdentifier, model as language_model
    classifier = LanguageIdentifier.from_modelstring(language_model, norm_probs=True)
    classifier.set_languages(['en', 'zh', 'ja', 'ko', 'fr', 'de', 'it', 'pt', 'es', 'ru', 'ar'])
    stats, buckets, seen = Counter(), {lang: [] for lang in ('zh', 'en')}, set()
    source_raw, source_kept = Counter(), Counter()

    def flush(pending, language):
        if not pending:
            return
        texts = []
        for record in pending:
            turns = record['conversations']
            texts.extend([''.join(f"<|im_start|>{t['role']}\n{t['content']}<|im_end|>\n" for t in turns),
                          turns[-2]['content'], turns[-1]['content']])
        encoded = tokenizer.encode_batch(texts)
        for i, record in enumerate(pending):
            full, question, answer = encoded[i * 3:i * 3 + 3]
            if len(full.ids) > seq_len + 1 or len(question.ids) > 256 or len(answer.ids) > 256:
                stats[f'{language}_length'] += 1
                continue
            # 拉丁字母比例不能区分英文与意大利文；只对足够长的回答使用统计语言识别。
            text = record['conversations'][-1]['content']
            if language == 'en' and len(text) >= 40 and classifier.classify(text)[0] != 'en':
                stats['en_language_detector'] += 1
                continue
            key = text_digest(record['conversations'][-2]['content'])
            if key in seen:
                stats[f'{language}_duplicate_prompt'] += 1
                continue
            seen.add(key)
            record.update(id=key, tokens=len(full.ids), answer_tokens=len(answer.ids))
            buckets[language].append(record)
            source_kept[record['group']] += 1
        pending.clear()

    for language in buckets:
        for path in sorted(raw.glob(f'sft_{language}_*.parquet')):
            pending = []
            logger.info('开始筛选 SFT: 文件=%s，语言=%s', path.name, language)
            for index, row in enumerate(read_rows([path])):
                stats[f'{language}_raw'] += 1
                source_raw[path.name] += 1
                if index and index % 10000 == 0:
                    logger.info('SFT 筛选进度: 文件=%s，已读取=%d，累计有效=%d', path.name, index, len(buckets[language]))
                record = sft_record(row, path.name, language)
                if record is None:
                    stats[f'{language}_structure'] += 1
                    continue
                reason = sft_reject_reason(record)
                if reason:
                    stats[f'{language}_{reason}'] += 1
                    continue
                turns = record['conversations']
                question, answer = turns[-2]['content'], turns[-1]['content']
                if not language_ok(answer, language) or any(token in ''.join(t['content'] for t in turns) for token in SPECIAL):
                    stats[f'{language}_language_or_control'] += 1
                    continue
                if record['group'] == 'smoltalk_zh' and (row.get('score', 0) < 4 or row.get('difficulty') == '困难'):
                    stats[f'{language}_difficulty_or_score'] += 1
                    continue
                domains = ' '.join(row.get('domain') or [])
                if any(term in domains for term in ('健康', '医学', '医疗', '法律', '金融')):
                    stats[f'{language}_out_of_scope_domain'] += 1
                    continue
                if any(probe in normalized(question + answer) for probe in probes):
                    stats[f'{language}_evaluation_overlap'] += 1
                    continue
                record['source_row'] = index
                pending.append(record)
                if len(pending) >= 256:
                    flush(pending, language)
            flush(pending, language)
            logger.info('SFT 过滤完成: 文件=%s，累计中文=%d，累计英文=%d', path.name, len(buckets['zh']), len(buckets['en']))
    rng = np.random.default_rng(seed)
    for rows in buckets.values():
        rng.shuffle(rows)
    # 通用短指令为主体；抽取/摘要与建议来源分别限额，避免某一种任务主导配方。
    caps = {'alpaca_gpt4_zh': 6000, 'coig_pc': 1000, 'smoltalk_zh': 500}
    zh, counts = [], Counter()
    for row in buckets['zh']:
        group = row['group']
        if group in caps and counts[group] < caps[group]:
            zh.append(row)
            counts[group] += 1
    if counts['alpaca_gpt4_zh'] < 1000:
        raise ValueError('通用中文短问答材料不足，请先下载 --sft-supplement')
    en_count = min(len(buckets['en']), round(len(zh) * 2 / 3))
    selected = zh + buckets['en'][:en_count]
    rng.shuffle(selected)
    selected, review_stats = finalize_sft_rows(selected, seed)
    report = {**dict(stats), **review_stats, 'raw_by_file': dict(source_raw),
              'eligible_by_group': dict(source_kept), 'answer_token_limit': 256,
              'question_token_limit': 256, 'language_detector': 'langid 1.1.6',
              'supplement_lock_sha256': sha256_file(Path('configs/bilingual_sft_sources.lock.json'))}
    return save_sft_rows(selected, output, report)


def main():
    parser = argparse.ArgumentParser(description="准备第一阶段中英文训练材料，不启动 GPU")
    parser.add_argument("--sft-only", action="store_true", help="复用冻结 tokenizer，仅重新准备 SFT 材料")
    parser.add_argument("--output", default="checkpoints/bilingual_20261011")
    parser.add_argument("--tokens", type=int, default=1_000_000_000)
    parser.add_argument("--zh-fraction", type=float, default=0.7)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--eval-file", default="evals/bilingual_acceptance_v1.jsonl")
    args = parser.parse_args()
    if not 0 < args.zh_fraction < 1 or args.tokens < 10000 or args.seq_len < 8:
        raise ValueError("中英文比例须在 (0,1)，token 预算至少 10000，序列长度至少 8")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    eval_path = Path(args.eval_file)
    questions = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    probes = [normalized(row["prompt"]) for row in questions if len(normalized(row["prompt"])) >= 20]
    recipe = {"tokens": args.tokens, "zh_fraction": args.zh_fraction, "seq_len": args.seq_len,
              "seed": args.seed, "eval_sha256": sha256_file(eval_path),
              "source_lock_sha256": sha256_file(Path("configs/bilingual_sources.lock.json")), "pipeline_version": 1}
    recipe_path = output / "recipe.json"
    if recipe_path.exists() and json.loads(recipe_path.read_text(encoding="utf-8")) != recipe:
        raise ValueError("准备参数与缓存不一致，请使用新的输出目录，避免混用旧数据或 tokenizer")
    write_json(recipe_path, recipe)
    if args.sft_only:
        report = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        tokenizer = Tokenizer.from_file(str(output / "tokenizer.json"))
        report["sft"] = prepare_sft(output / "raw", output, tokenizer, args.seq_len, probes, args.seed)
        write_json(output / "manifest.json", report)
        return
    report = {"recipe": recipe, "cleaning": {}}
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs = {language: pool.submit(clean_language, output / "raw", output, language, probes)
                for language in ("zh", "en")}
        for language, job in jobs.items():
            report["cleaning"][language] = job.result()
    tokenizer = train_tokenizer(output)
    report["memmaps"] = build_memmaps(output, tokenizer, args.tokens, args.zh_fraction, args.seq_len, args.seed)
    report["sft"] = prepare_sft(output / "raw", output, tokenizer, args.seq_len, probes, args.seed)
    report["tokenizer_sha256"] = sha256_file(output / "tokenizer.json")
    report["status"] = "prepared_not_trained"
    write_json(output / "manifest.json", report)
    logger.info("材料准备完成: %s，实际输入 token=%d", output, report["memmaps"]["train"]["input_tokens"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
