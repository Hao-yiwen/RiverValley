"""保留已运行的小实验材料，为正式 SFT 单独准备严格隔离并逐条审计的候选。"""
import argparse
from collections import Counter
import json
import logging
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from data.dataset import SFTDataset
from tokenizer.bpe import load_tokenizer, get_pad_id
from download_bilingual_sources import sha256_file
from prepare_sft_v2 import summarize, write_rows
from sft_quality import normalized

log = logging.getLogger(__name__)


def first_question(row):
    return normalized(next(t['content'] for t in row['conversations'] if t['role'] == 'user'))


def isolated_validation(training, validation):
    """首问复用也视作重叠，即使后续更正及最终提问不同。"""
    questions = {first_question(row) for row in training}
    kept, removed = [], []
    for row in validation:
        (removed if first_question(row) in questions else kept).append(row)
    if {r['split_group'] for r in training} & {r['split_group'] for r in kept}:
        raise ValueError('候选仍有分组交叉，必须回到材料构建阶段处理')
    return kept, removed


def audit_supervision(rows, tokenizer):
    dataset = SFTDataset(rows, tokenizer, block_size=1024, pad_id=get_pad_id(tokenizer))
    if len(dataset) != len(rows):
        raise ValueError('实际加载时丢弃样本，候选清单不能作为真实训练量')
    end_id = tokenizer.token_to_id('<|im_end|>')
    counts = Counter()
    for index, row in enumerate(rows):
        _, labels, mask, _ = dataset[index]
        supervised = int(mask.sum())
        end_markers = int(((labels == end_id) & mask.bool()).sum())
        expected_turns = sum(t['role'] == 'assistant' for t in row['conversations'])
        if supervised != row['answer_tokens'] or end_markers != expected_turns:
            raise ValueError(f'监督或结束标记不符：id={row["id"]} token={supervised} end={end_markers}')
        counts.update(rows=1, supervised_tokens=supervised, assistant_end_markers=end_markers)
        if index % 5000 == 0:
            log.info('实际监督检查：已完成=%d/%d', index, len(rows))
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='checkpoints/bilingual_v2_20261019')
    parser.add_argument('--output', default='checkpoints/bilingual_v2_20261019/formal_candidate_01')
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    if output.exists():
        raise FileExistsError('不覆盖已冻结候选，请选新的版本目录：' + str(output))
    manifest = json.loads((source / 'sft_manifest.json').read_text(encoding='utf-8'))
    rows = {}
    for key, name in [('full', 'sft_train.jsonl'), ('val', 'sft_val.jsonl')]:
        path = source / 'data' / name
        if sha256_file(path) != manifest['hashes'][key]:
            raise ValueError('输入已偏离冻结清单：' + name)
        rows[key] = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    val, removed = isolated_validation(rows['full'], rows['val'])
    tokenizer = load_tokenizer(str(source / 'tokenizer.json'))
    audits = {name: audit_supervision(group, tokenizer) for name, group in [('train', rows['full']), ('val', val)]}
    output.mkdir(parents=True)
    write_rows(output / 'sft_train.jsonl', rows['full'])
    write_rows(output / 'sft_val.jsonl', val)
    write_rows(output / 'excluded_validation.jsonl', removed)
    report = {'status': 'candidate_structural_checks_passed', 'time': time.time(),
              'approved_for_training': False, 'source_manifest_sha256': sha256_file(source / 'sft_manifest.json'),
              'source_hashes': manifest['hashes'], 'train': summarize(rows['full']), 'val': summarize(val),
              'removed_shared_initial_contexts': len(removed), 'supervision': audits,
              'hashes': {name: sha256_file(output / name) for name in ['sft_train.jsonl', 'sft_val.jsonl']},
              'tokenizer_sha256': sha256_file(source / 'tokenizer.json'), 'code_sha256': sha256_file(Path(__file__)),
              'pending': '小实验问答结果、任务配比和公开问答内容复核后再批准；结构正确不等于答案正确。'}
    (output / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    log.info('正式候选结构检查完成：训练=%d 验证=%d 排除首问重叠=%d；尚未批准训练', len(rows['full']), len(val), len(removed))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
