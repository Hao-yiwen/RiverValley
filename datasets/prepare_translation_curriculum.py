"""构建短句翻译措辞实验，按句子连通分组隔离训练、开发和新留出。"""
import argparse
from collections import Counter
import hashlib
import json
import logging
from pathlib import Path
import random
import re
import unicodedata

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = '4bedf597ed83f1d9a8dfd5380d57dbdd87ec2671b22f92bd5921f5f039067885'
ZH = ['请将下面这句话翻译成英文，只输出译文：\n{}', '翻译成英语：{}',
      '这句话用英文怎么说？\n{}', '中译英，仅给译文。原文：{}',
      '请用自然的英语表达下面这句话，不加解释：\n{}', '英文译文是什么？\n{}']
EN = ['Translate the following sentence into Chinese. Return only the translation:\n{}',
      'Translate into Chinese: {}', 'How would you say this in Chinese?\n{}',
      'English to Chinese. Translation only:\n{}', 'Give a natural Chinese translation, without explanation:\n{}',
      'What is the Chinese translation?\n{}']


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def norm(value):
    return re.sub(r'\W+', '', unicodedata.normalize('NFKC', value).casefold())


def read(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def write(path, rows):
    Path(path).write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('拒绝覆盖已冻结材料')
    source = ROOT/'checkpoints/bilingual_v2_20261019/raw/tatoeba_hsk_train.jsonl'
    if sha(source) != SOURCE_SHA:
        raise ValueError('冻结 Tatoeba 原件摘要不匹配')
    raw = [r for r in read(source) if r.get('cmn-txt') and r.get('eng-txt')]
    parents = {}

    def find(value):
        parents.setdefault(value, value)
        if parents[value] != value:
            parents[value] = find(parents[value])
        return parents[value]

    # 同一句任一语言的多种译文不能被随机拆到不同数据分区。
    for row in raw:
        a, b = find('zh/'+norm(row['cmn-txt'])), find('en/'+norm(row['eng-txt']))
        if a != b:
            parents[max(a, b)] = min(a, b)
    old = ROOT/'checkpoints/bilingual_v2_20261019/formal_candidate_05'
    old_train, old_val = read(old/'sft_train.jsonl'), read(old/'sft_val.jsonl')
    used, excluded = set(), set()
    for rows, target in [(old_train, used), (old_val, excluded)]:
        for row in rows:
            if row.get('task') != 'translation':
                continue
            for turn in row['conversations']:
                text = turn['content'].split('\n')[-1]
                for lang in ('zh', 'en'):
                    node = lang+'/'+norm(text)
                    if node in parents:
                        target.add(find(node))
    # 历史开发、留出不用于新训练；只取题目与参考以排除重复，不读取旧模型回答。
    retired = [ROOT/'evals/bilingual_acceptance_v1.jsonl',
               ROOT/'checkpoints/behavior_repair_20261026/data/dev.jsonl',
               ROOT/'checkpoints/behavior_repair_20261026/data/holdout.jsonl']
    retired += list((ROOT/'evals').glob('*holdout*.jsonl'))
    protected = []
    for path in retired:
        for q in read(path):
            protected += [norm(q.get('prompt', '')), norm(q.get('reference', ''))]
    candidates = {}
    for row in raw:
        z, e = row['cmn-txt'], row['eng-txt']
        if row['hsk-level'] not in (1, 2, 3) or not 4 <= len(z) <= 32 or not 8 <= len(e) <= 100:
            continue
        if re.search(r'[^\x00-\x7f]', e) or re.search(r'肏|屄|fuck|shit|bitch|sex|suicid|murder', z+' '+e, re.I):
            continue
        if len(re.findall(r'[\u4e00-\u9fff]', z)) < 3:
            continue
        normalized_zh, normalized_en = norm(z), norm(e)
        group = find('zh/'+normalized_zh)
        if any(normalized_zh in q or normalized_en in q for q in protected if q):
            excluded.add(group)
        # 同一连通分组仅取一对句子，避免多译文重复加权。
        candidates.setdefault(group, row)
    available = [(g, r) for g, r in sorted(candidates.items()) if g not in excluded]
    random.Random(20261110).shuffle(available)
    fresh = [(g, r) for g, r in available if g not in used]
    if len(fresh) < 160:
        raise ValueError('从未参与旧训练/验证的独立句子不足')
    dev, holdout, val = fresh[:12], fresh[12:36], fresh[36:100]
    reserved = {g for g, r in dev+holdout+val}
    train_pairs = [(g, r) for g, r in available if g not in reserved][:3584]
    if len(train_pairs) != 3584:
        raise ValueError('合格翻译训练分组不足')

    def sample(group, row, direction, form, split):
        zh = direction == 'zh'
        question = (ZH if zh else EN)[form].format(row['cmn-txt' if zh else 'eng-txt'])
        answer = row['eng-txt' if zh else 'cmn-txt']
        return {'id': hashlib.sha256((group+'/'+direction+'/'+str(form)).encode()).hexdigest(),
                'conversations': [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}],
                'source': 'Tatoeba/bdx33/hsk-cmn-eng', 'split_group': group, 'split': split,
                'task': 'translation', 'language': direction, 'target_language': 'en' if zh else 'zh',
                'source_ids': {'cmn': row['cmn-id'], 'eng': row['eng-id']}, 'form': form,
                'source_copyright': f"CC-BY 2.0; https://tatoeba.org/en/sentences/show/{row['cmn-id']}; https://tatoeba.org/en/sentences/show/{row['eng-id']}",
                'verification': '双语句对、语言/长度过滤；抽检而非逐条人工核验'}

    train = [sample(g, r, lang, i % 6, 'train') for i, (g, r) in enumerate(train_pairs) for lang in ('zh', 'en')]
    validation = [sample(g, r, lang, i % 6, 'val') for i, (g, r) in enumerate(val) for lang in ('zh', 'en')]
    repair = read(ROOT/'checkpoints/behavior_repair_20261026/data/sft_train.jsonl')
    # 2048 条旧训练回放，不把重复样本算作新增语料。
    replay = []
    for lang in ('zh', 'en'):
        natural = [r for r in repair if r['language'] == lang and r['task'] == 'general' and len(r['conversations'][-1]['content']) < 400]
        structured = [r for r in repair if r['language'] == lang and r['task'] in ('extraction', 'json', 'sort', 'multi_turn', 'copy')]
        random.Random(20261111).shuffle(natural)
        random.Random(20261111).shuffle(structured)
        if len(natural) < 256 or len(structured) < 768:
            raise ValueError('短自然回放或结构回放不足')
        replay += [dict(r, replay=True) for r in natural[:256]+structured[:768]]
    train += replay
    arithmetic = [r for r in repair if r['task'] == 'arithmetic']
    random.Random(20261110).shuffle(arithmetic)
    train += [dict(r, replay=True) for r in arithmetic[:1024]]
    random.Random(20261110).shuffle(train)
    from tokenizers import Tokenizer
    tokenizer_path = ROOT/'checkpoints/delivery_behavior_20261026/sft_best/tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    for row in train+validation:
        text = ''.join(f"<|im_start|>{t['role']}\n{t['content']}<|im_end|>\n" for t in row['conversations'])
        row['tokens'] = len(tokenizer.encode(text).ids)
        if row['tokens'] > 512:
            raise ValueError('样本超过本轮实际512token上限，拒绝静默截断')
    if len(train) != 10240:
        raise ValueError('160步有效batch64需要恰好10240行')
    output.mkdir(parents=True)
    write(output/'sft_train.jsonl', train)
    write(output/'sft_val.jsonl', validation)
    for name, pairs, forms in [('development', dev, (0, 1)), ('holdout', holdout, (5,))]:
        questions = []
        for g, r in pairs:
            for lang in ('zh', 'en'):
                for form in forms:
                    row = sample(g, r, lang, form, name)
                    questions.append({'id': name+'/'+row['id'], 'language': lang, 'category': 'translation',
                        'prompt': row['conversations'][0]['content'], 'reference': row['conversations'][1]['content'],
                        'scoring': 'manual', 'split_group': g, 'form': form, 'source_ids': row['source_ids']})
        write(output/(name+'.jsonl'), questions)
    manifest = {'version': 'translation_curriculum_20261110', 'generator_sha256': sha(__file__),
        'source_sha256': SOURCE_SHA, 'source_revision': 'e861cb8efd67d1d2a8fac9412228f9854c6f3345',
        'tokenizer_sha256': sha(tokenizer_path), 'train_rows': len(train), 'translation_pairs': 3584,
        'train_tasks': dict(Counter(r['task'] for r in train)), 'max_tokens': max(r['tokens'] for r in train),
        'fresh_dev_groups': 12, 'fresh_holdout_groups': 24,
        'split_policy': '按中英文规范化句子连通分组；新开发/留出排除旧正式训练及验证；留出不用于训练或选优。',
        'limits': '翻译占70%，旧回放30%；自然回放并非重新逐条核验。Tatoeba可能有轻微不对齐；不是新增预训练token。',
        'hashes': {p.name: sha(p) for p in sorted(output.glob('*.jsonl'))}}
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    log.info('材料完成：%d行，翻译分组=%d，新开发48题、新留出48题，最长=%dtoken', len(train), len(train_pairs), manifest['max_tokens'])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
