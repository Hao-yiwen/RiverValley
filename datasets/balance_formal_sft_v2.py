"""按真实助手监督量限制改写/摘要占比，避免少量长答案主导简单问答训练。"""
import argparse
from collections import Counter
import json
import logging
from pathlib import Path
import time

from download_bilingual_sources import sha256_file
from prepare_sft_v2 import summarize, write_rows
from prepare_formal_sft_v2 import audit_supervision, isolated_validation
from sft_quality import digest
from tokenizer.bpe import load_tokenizer

log = logging.getLogger(__name__)


def label_task(row):
    row = dict(row)
    if row['source'].endswith('/smollm-rewrite-30k'):
        row['task'] = 'rewrite'
    elif row['source'].endswith('/smol-summarize-20k'):
        row['task'] = 'summary'
    return row


def cap_long_form_share(rows, maximum=.15):
    if not 0 < maximum < 1:
        raise ValueError('监督量上限必须在0和1之间')
    core = [r for r in rows if r['task'] not in ('rewrite', 'summary')]
    optional = [r for r in rows if r['task'] in ('rewrite', 'summary')]
    core_tokens = sum(r['answer_tokens'] for r in core)
    allowed = int(core_tokens * maximum / (1-maximum))
    kept, excluded, spent = [], [], 0
    for row in sorted(optional, key=lambda r: digest('formal-token-mix/'+r['id'])):
        if spent + row['answer_tokens'] <= allowed:
            kept.append(row);spent += row['answer_tokens']
        else:
            excluded.append(row)
    selected = {r['id'] for r in core+kept}
    return [r for r in rows if r['id'] in selected], excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='checkpoints/bilingual_v2_20261019/formal_candidate_02')
    parser.add_argument('--output', default='checkpoints/bilingual_v2_20261019/formal_candidate_03')
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    if output.exists():raise FileExistsError('不覆盖既有材料')
    original=json.loads((source/'manifest.json').read_text());splits={}
    for split in ('train','val'):
        path=source/f'sft_{split}.jsonl'
        if sha256_file(path)!=original['hashes'][path.name]:raise ValueError('源候选摘要不符')
        splits[split]=[label_task(json.loads(line)) for line in path.read_text().splitlines()]
    splits['train'],excluded=cap_long_form_share(splits['train'])
    splits['val'],overlap=isolated_validation(splits['train'],splits['val'])
    if overlap:raise ValueError('首问隔离出现意外变化')
    tokenizer=load_tokenizer(str(source.parent/'tokenizer.json'))
    audits={name:audit_supervision(rows,tokenizer) for name,rows in splits.items()}
    output.mkdir(parents=True)
    for name,rows in splits.items():write_rows(output/f'sft_{name}.jsonl',rows)
    write_rows(output/'excluded_by_token_mix.jsonl',excluded)
    tokens=Counter()
    for row in splits['train']:tokens[row['task']]+=row['answer_tokens']
    report={'status':'candidate_token_balanced','approved_for_training':False,'time':time.time(),
        'source_manifest_sha256':sha256_file(source/'manifest.json'),'max_rewrite_summary_supervision_share':.15,
        'actual_rewrite_summary_supervision_share':(tokens['rewrite']+tokens['summary'])/sum(tokens.values()),
        'task_supervised_tokens':dict(tokens),'excluded_optional_rows':len(excluded),
        'train':summarize(splits['train']),'val':summarize(splits['val']),'supervision':audits,
        'hashes':{f'sft_{name}.jsonl':sha256_file(output/f'sft_{name}.jsonl') for name in splits},
        'code_sha256':sha256_file(Path(__file__)),'tokenizer_sha256':sha256_file(source.parent/'tokenizer.json'),
        'limitations':'上限为本轮显式配方选择，不是已证明最优值；抽检边界继承候选02。正式开训仍需对照结果决策。'}
    (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    log.info('按监督token重配完成：训练=%d 可选改写/摘要=%d/%d token 比例=%.3f%%，排除=%d条',
             len(splits['train']),tokens['rewrite']+tokens['summary'],sum(tokens.values()),report['actual_rewrite_summary_supervision_share']*100,len(excluded))


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    main()
