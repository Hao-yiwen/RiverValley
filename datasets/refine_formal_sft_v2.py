"""按实际内容抽检清理正式候选；保留试验数据和上一候选，禁止覆盖冻结文件。"""
import argparse
from collections import Counter
import json
import logging
from pathlib import Path
import re
import time

from download_bilingual_sources import sha256_file
from prepare_formal_sft_v2 import audit_supervision, isolated_validation
from prepare_sft_v2 import summarize, write_rows
from tokenizer.bpe import load_tokenizer

log = logging.getLogger(__name__)
EXCLUDED_SOURCES = {
    'HuggingFaceTB/smol-smoltalk/smol-summarize-5k': '抽样存在字数违例、答案内部矛盾和未经核验的操作建议',
    'HuggingFaceTB/smol-smoltalk/smol-magpie-ultra-short': '抽样包含角色虚构及错误统计数值，暂不纳入简单问答训练',
    'opencsg/smoltalk-chinese': '抽样出现明显乱码、角色文本串接和不必要的语言切换',
}
EXCLUDED_IDS = {
    '92c2b612eb1cbf297e95fd99c12451948e83651b8bedfdde15c69a1f39cca908': '未给出时间依据的流行款式断言',
    '1fe1dd55af2472c40cd13a31ab5ace1118baa0397e1a00e55f8cc270a6bea6a2': '缺少航司和航线信息，餐食回答作了未经证实的概括',
    'ff2b7798044dd07f0b83209b2df6a6f195ccfc518525ae64dcee59e7b45a4af1': '背景声称真菌是唯一能分解木材的生物，排除未经核验的排他性知识断言',
    'c6c8d16b13650daa7ec0354aec65fa0dce4ae95ee6373d4e9dcdd5ae149ae6a2': '译文省略明确的未来时间限定，双向句对一起排除',
    'b73cb5c383797a6f12b929f76171475a8bb10fde878753e8f8639988f2eaf08f': '要求Yes或No却添加整段解释，且包含本轮范围外的西班牙语翻译',
}


def rejection(row):
    if row['source'] in EXCLUDED_SOURCES:
        return 'source_quarantined'
    if row['id'] in EXCLUDED_IDS:
        return 'reviewed_content_issue'
    questions = '\n'.join(t['content'] for t in row['conversations'] if t['role'] == 'user')
    instructions = '\n'.join(t['content'] for t in row['conversations'] if t['role'] != 'assistant')
    if re.search(r'给首句写全诗|古诗|文言文|写.{0,3}古诗', questions):
        return 'outside_simple_modern_qa'
    if re.search(r'首部|数据报|路由器|ICMP', questions, re.I):
        return 'unverified_network_specialist'
    if re.search(r'Steam of consciousness|think step.by.step|justify your answer', instructions, re.I):
        return 'verbose_reasoning_style'
    if re.search(r'role.?play|角色扮演|You are (?:a \d+|the owner|an AI superhero)|you can use energy beams', instructions, re.I):
        return 'roleplay_outside_scope'
    other_languages = r'Arabic|Italian|Spanish|French|German|Russian|Japanese|Korean|Portuguese|Hindi|Turkish'
    if re.search(r'(?:translat(?:e|ion)[^\n]*(?:' + other_languages + r')|mean in (?:' + other_languages + r'))', questions, re.I):
        return 'translation_outside_zh_en'
    # 无日期和外部观测来源的实时天气断言不适合作为事实标签；澄清类回答仍可保留。
    for turn, following in zip(row['conversations'], row['conversations'][1:]):
        if turn['role'] != 'user' or following['role'] != 'assistant':
            continue
        # 翻译、改写和带材料的长阅读不是实时查询，不能把正文中的天气当成事实幻觉。
        if row.get('task') in ('translation', 'rewrite', 'summary') or len(turn['content']) > 160:
            continue
        weather = re.search(r'\b(?:weather|rain(?:ed|ing)?|snow(?:ed|ing)?)\b|天气|降雨|下雪', turn['content'], re.I)
        temporal = re.search(r'\b(?:current|today|yesterday|right now)\b|今天|昨天|现在', turn['content'], re.I)
        limitation = re.search(r"(?:can't|cannot|don't have|do not have).*(?:access|real.time|current)|which (?:city|location)|what.*location|无法.*实时|请.*城市", following['content'], re.I)
        if weather and temporal and not limitation:
            return 'ungrounded_current_weather'
    # 多轮也逐条检查长度指令，不能只在单轮或来源标签上判断。
    system = '\n'.join(t['content'] for t in row['conversations'] if t['role'] == 'system')
    match = re.search(r'shorter than (\d+) characters', system, re.I)
    if match and any(len(t['content']) >= int(match[1]) for t in row['conversations'] if t['role'] == 'assistant'):
        return 'explicit_length_constraint_violation'
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='checkpoints/bilingual_v2_20261019/formal_candidate_01')
    parser.add_argument('--output', default='checkpoints/bilingual_v2_20261019/formal_candidate_02')
    parser.add_argument('--review-sample', default='evals/results/v2_20261019/formal_review_sample_01.jsonl')
    parser.add_argument('--cap-long-form-share', type=float)
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    if output.exists():
        raise FileExistsError('正式候选不可覆盖，请使用新目录')
    manifest = json.loads((source / 'manifest.json').read_text())
    splits = {}
    for split in ('train', 'val'):
        path = source / f'sft_{split}.jsonl'
        if sha256_file(path) != manifest['hashes'][path.name]:
            raise ValueError('输入摘要不符：' + str(path))
        splits[split] = [json.loads(line) for line in path.read_text().splitlines()]
    rejected_groups = {r['split_group'] for rows in splits.values() for r in rows if r['id'] in EXCLUDED_IDS}
    exclusions, counts = [], Counter()
    for split, rows in splits.items():
        kept = []
        for row in rows:
            reason = rejection(row) or ('reviewed_pair_group' if row['split_group'] in rejected_groups else None)
            if reason:
                counts[split + '/' + reason] += 1
                exclusions.append(dict(row, exclusion_reason=reason, original_split=split))
            else:
                kept.append(row)
        splits[split] = kept
    if args.cap_long_form_share is not None:
        from balance_formal_sft_v2 import cap_long_form_share
        splits['train'], optional = cap_long_form_share(splits['train'], args.cap_long_form_share)
        counts['train/token_mix_rebalance'] += len(optional)
        exclusions.extend(dict(r, exclusion_reason='token_mix_rebalance', original_split='train') for r in optional)
    splits['val'], overlap = isolated_validation(splits['train'], splits['val'])
    if overlap:
        raise ValueError('上一候选的首问隔离被破坏')
    tokenizer_path = source.parent / 'tokenizer.json'
    tokenizer = load_tokenizer(str(tokenizer_path))
    audits = {name: audit_supervision(rows, tokenizer) for name, rows in splits.items()}
    output.mkdir(parents=True)
    for split, rows in splits.items():
        write_rows(output / f'sft_{split}.jsonl', rows)
    write_rows(output / 'excluded_content.jsonl', exclusions)
    report = {'status': 'candidate_content_refined', 'approved_for_training': False, 'time': time.time(),
        'source_manifest_sha256': sha256_file(source / 'manifest.json'),
        'review_sample': args.review_sample,
        'review_sample_sha256': sha256_file(Path(args.review_sample)),
        'max_rewrite_summary_supervision_share': args.cap_long_form_share,
        'source_exclusion_reasons': EXCLUDED_SOURCES, 'row_exclusion_reasons': EXCLUDED_IDS,
        'exclusions': dict(counts), 'train': summarize(splits['train']), 'val': summarize(splits['val']),
        'supervision': audits, 'hashes': {f'sft_{name}.jsonl': sha256_file(output / f'sft_{name}.jsonl') for name in splits},
        'code_sha256': sha256_file(Path(__file__)), 'tokenizer_sha256': sha256_file(tokenizer_path),
        'limitations': '公开答案仅做来源分层抽查和结构/规则检查，不能声称全量事实正确；尚待小实验效果和配比决策。'}
    (output / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    log.info('新候选完成：训练=%d 验证=%d 排除=%d，未覆盖原材料，未批准开训', len(splits['train']), len(splits['val']), len(exclusions))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
