"""以第一版全部已清洗文档为排除库，准备冻结 tokenizer 的增量预训练材料。"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import logging
from pathlib import Path
import shutil

from tokenizers import Tokenizer
from download_bilingual_sources import sha256_file
from prepare_bilingual import NearDeduplicator, reject_reason, read_rows, build_memmaps
from sft_quality import normalized

log = logging.getLogger(__name__)


def clean(old, output, language, probes, signature):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    old, output = Path(old), Path(output)
    folder = output / 'clean'
    folder.mkdir(exist_ok=True)
    report_file = folder / f'{language}_report.json'
    if report_file.exists():
        previous = json.loads(report_file.read_text())
        if previous.get('signature') == signature:
            if all(sha256_file(folder / f'{language}_{split}.jsonl') == previous['sha256'][split] for split in ('train', 'val')):
                log.info('复用摘要通过的增量清洗：%s', language)
                return previous
        raise ValueError('增量清洗缓存签名不同，禁止覆盖混用')
    dedup, stats = NearDeduplicator(), Counter()
    for split in ('train', 'val'):
        with (old / 'clean' / f'{language}_{split}.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                dedup.add(row['text'])
                stats['v1_exclusion_docs'] += 1
                if stats['v1_exclusion_docs'] % 100000 == 0:
                    log.info('建立跨版本去重索引：%s %d 篇', language, stats['v1_exclusion_docs'])
    targets = {split: (folder / f'{language}_{split}.jsonl').open('w', encoding='utf-8') for split in ('train', 'val')}
    try:
        paths = sorted((output / 'pretrain_raw').glob(f'pretrain_{language}_*.parquet'))
        for path in paths:
            for index, row in enumerate(read_rows([path])):
                stats['raw'] += 1
                text = row.get('text')
                reason = reject_reason(text, language, probes)
                if reason:
                    stats[reason] += 1
                    continue
                key, reason = dedup.add(text)
                if reason:
                    stats[reason] += 1
                    continue
                split = 'val' if int(key[:8], 16) % 100 == 0 else 'train'
                record = {'id': key, 'text': text.strip(), 'language': language,
                          'source_file': path.name, 'source_row': index,
                          'source': row.get('source'), 'source_score': row.get('score')}
                targets[split].write(json.dumps(record, ensure_ascii=False) + '\n')
                stats[split] += 1
                stats[split + '_characters'] += len(text)
            if stats['raw'] % 100000 < 3000:
                log.info('增量清洗：语言=%s 原始=%d 保留=%d', language, stats['raw'], stats['train'])
    finally:
        for stream in targets.values():
            stream.close()
    report = {'signature': signature, 'counts': dict(stats),
              'sha256': {split: sha256_file(folder / f'{language}_{split}.jsonl') for split in ('train', 'val')},
              'novelty_scope': '与旧版全部清洗文档及本轮文档做规范化精确去重和字符SimHash近重复启发式检查；不保证语义绝无重叠'}
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    log.info('增量清洗完成：%s %s', language, stats)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-root', default='checkpoints/bilingual_20261011')
    parser.add_argument('--output', default='checkpoints/bilingual_v2_20261019')
    parser.add_argument('--tokens', type=int, default=2_000_000_000)
    args = parser.parse_args()
    old, output = Path(args.old_root), Path(args.output)
    manifest = output / 'pretrain_download_manifest.json'
    if not manifest.exists():
        raise FileNotFoundError('下载尚未完成，不允许把部分原始文件当成全量材料')
    for file in json.loads(manifest.read_text()):
        if sha256_file(output / 'pretrain_raw' / file['filename']) != file['sha256']:
            raise ValueError('源文件摘要不匹配：' + file['filename'])
    probes = []
    for path in (Path('evals/bilingual_acceptance_v1.jsonl'), Path('evals/bilingual_acceptance_v2_holdout.jsonl')):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if len(normalized(row['prompt'])) >= 20:
                probes.append(normalized(row['prompt']))
            if row.get('category') == 'translation' and len(normalized(row['reference'])) >= 12:
                probes.append(normalized(row['reference']))
    signature = {'raw_sha256': sha256_file(manifest), 'code_sha256': sha256_file(Path(__file__)),
                 'v1_clean': {p.name: sha256_file(p) for p in (old / 'clean').glob('*.jsonl')},
                 'tokenizer_sha256': sha256_file(old / 'tokenizer.json'), 'tokens': args.tokens,
                 'probes_sha256': __import__('hashlib').sha256(json.dumps(probes).encode()).hexdigest()}
    shutil.copy2(old / 'tokenizer.json', output / 'tokenizer.json')
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs = {lang: pool.submit(clean, old, output, lang, probes, signature) for lang in ('zh', 'en')}
        reports = {lang: future.result() for lang, future in jobs.items()}
    tokenizer = Tokenizer.from_file(str(output / 'tokenizer.json'))
    memmaps = build_memmaps(output, tokenizer, args.tokens, .7, 1024, 1337)
    report = {'version': 2, 'signature': signature, 'cleaning': reports, 'memmaps': memmaps,
              'target_added_tokens': args.tokens, 'training_updates': memmaps['train']['num_samples'] // 128,
              'actual_training_tokens': (memmaps['train']['num_samples'] // 128) * 131072}
    (output / 'pretrain_manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    log.info('增量预训练准备完成：更新数=%d，新增训练输入token=%d', report['training_updates'], report['actual_training_tokens'])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
