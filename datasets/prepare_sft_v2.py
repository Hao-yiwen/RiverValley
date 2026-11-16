"""构建第二版分任务 SFT；保留旧材料，用内容摘要锁定新产物。"""
import argparse
from collections import Counter, defaultdict
import json
import logging
from pathlib import Path
import random
import re

from tokenizers import Tokenizer
from prepare_bilingual import read_rows, sft_record
from download_bilingual_sources import sha256_file
from sft_quality import annotate, digest, normalized, split_for

log = logging.getLogger(__name__)


def pair(question, answer):
    return [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}]


def synthetic_records(count=16000, seed=20261019):
    """程序可核验的小任务；模板家族显式分离，不冒充自然语料或通用能力。"""
    rng = random.Random(seed)
    names = {'zh': ['小林', '小叶', '小周', '小陈', '小夏', '小顾', '小陆', '小许'],
             'en': ['Nora', 'Leo', 'Iris', 'Owen', 'Lina', 'Eric', 'Mila', 'Evan']}
    items = {'zh': ['彩笔', '卡片', '积木', '铅笔', '贴纸', '纽扣', '夹子', '硬币'],
             'en': ['cards', 'blocks', 'pencils', 'stickers', 'buttons', 'clips', 'coins', 'beads']}
    colors = {'zh': ['蓝色', '绿色', '红色', '黄色', '紫色', '白色', '黑色', '橙色'],
              'en': ['blue', 'green', 'red', 'yellow', 'purple', 'white', 'black', 'orange']}
    for i in range(count):
        language = 'zh' if (i // 5) % 5 < 3 else 'en'
        zh = language == 'zh'
        a, b, c = rng.sample(range(1, 100), 3)
        name = rng.choice(names[language])
        item = rng.choice(items[language])
        color, other = rng.sample(colors[language], 2)
        task = ['arithmetic', 'extraction', 'format', 'multi_turn', 'clarification'][i % 5]
        # 每 20 个家族桶的第 0 个仅用于验证，措辞与训练桶不同。
        variant = (i // 25) % 20
        validation = variant == 0
        group = f'synthetic/{task}/{language}/family-{variant}'
        proof = {'a': a, 'b': b, 'c': c, 'name': name, 'item': item, 'color': color, 'other': other}
        if task == 'arithmetic':
            operation = (i // 100) % 4
            if operation == 0:
                expression, answer = f'{a} + {b}', str(a + b)
            elif operation == 1:
                expression, answer = f'{max(a,b)} - {min(a,b)}', str(abs(a - b))
            elif operation == 2:
                a, b = a % 12 + 1, b % 12 + 1
                expression, answer = f'{a} × {b}', str(a * b)
            else:
                expression, answer = f'{a*b} ÷ {b}', str(a)
            if validation:
                question = f'求这个算式的结果：{expression}。回复一个数字即可。' if zh else f'Find the value of {expression}. Respond with one number.'
            else:
                templates = ['计算 {x}，只写结果。', '{x} 等于多少？仅输出数字。', '请算出 {x}，不要解释。', '直接给出数字答案：{x}。'] if zh else ['Calculate {x}. Return only the number.', 'What is {x}? Answer with digits only.', 'Give the result of {x}, without explanation.', 'Reply with only the numerical value: {x}.']
                question = templates[variant % 4].format(x=expression)
            turns = pair(question, answer)
            proof.update(expression=expression, result=answer)
        elif task == 'extraction':
            if validation:
                question = f'记录显示：{name}有{a}枚硬币和{b}张卡片。卡片数量是多少？仅写数字。' if zh else f'The record says {name} has {a} coins and {b} cards. How many cards? Digits only.'
                answer = str(b)
            else:
                question = f'根据文字回答，不要补充解释：{name}带来了{a}个{item}，还有{b}本书。\n{name}带来了多少个{item}？只写数字。' if zh else f'Use only this text: {name} brought {a} {item} and {b} books.\nHow many {item} did {name} bring? Return just the number.'
                answer = str(a)
            turns = pair(question, answer)
            proof['result'] = answer
        elif task == 'format':
            values = [a, b, c]
            answer = ', '.join(map(str, sorted(values)))
            question = (f'将这三个数按从小到大排列，用英文逗号和一个空格连接：{a}, {b}, {c}。只输出排序结果。' if zh else f'Sort these three numbers in ascending order: {a}, {b}, {c}. Separate them with a comma and a space. Output only the sorted list.')
            if validation:
                question = (f'写出 {a}、{b}、{c} 的升序列表；以“, ”分隔，不加文字。' if zh else f'Arrange {a}, {b}, {c} from lowest to highest. Use ", " and no other text.')
                if (i // 500) % 2:
                    question = (f'返回 JSON 对象：label 填“{name}”，units 填数字 {a}。只保留这两个键。' if zh else f'Return a JSON object: label is "{name}", units is the number {a}. Include only these two keys.')
                    answer = json.dumps({'label': name, 'units': a}, ensure_ascii=False)
            elif variant % 3 == 1:
                question = (f'只输出合法 JSON：name 是字符串“{name}”，count 是数字 {a}。不要额外说明。' if zh else f'Output valid JSON only: name is the string "{name}" and count is the number {a}. No extra explanation.')
                answer = json.dumps({'name': name, 'count': a}, ensure_ascii=False)
            elif variant % 3 == 2:
                question = (f'将 {color}、{other} 两种颜色各写一行，保留原顺序，不要编号。' if zh else f'Write {color} and {other} on separate lines, in that order, without numbering.')
                answer = color + '\n' + other
            turns = pair(question, answer)
            proof.update(values=values, result=answer)
        elif task == 'multi_turn':
            turns = pair(f'我叫{name}，我喜欢{color}。' if zh else f'My name is {name} and I like {color}.',
                         f'记住了，你叫{name}，喜欢{color}。' if zh else f'Got it. Your name is {name} and you like {color}.')
            turns += pair(f'我现在改为喜欢{other}。' if zh else f'I now prefer {other}.',
                          f'好的，你现在喜欢{other}。' if zh else f'Okay, you now prefer {other}.')
            question = ('那我现在喜欢什么颜色？只写颜色。' if zh else 'Which color do I prefer now? Only the color, please.')
            if validation:
                question = '按我最后的修改，给出我喜欢的颜色，不写其他内容。' if zh else 'Recall my latest correction. State my preferred color and nothing else.'
            turns += pair(question, other)
            proof['result'] = other
        else:
            # 变化的业务标识不能推出真实状态；示范具体补充信息，避免无依据猜测。
            contexts = [
                (f'订单编号是 A{a}{b}{c}，它什么时候送到？', '仅凭订单编号无法确定送达时间。请提供物流状态或预计送达日期。'),
                (f'我的座位是{a}号，活动几点开始？', '座位号不能说明开始时间。请提供活动名称或票面上的时间。'),
                (f'商品编号是 P{a}{b}，现在卖多少钱？', '仅凭商品编号无法确定价格。请提供商品名称和商店信息。'),
                (f'我明天要去{a}号楼，应该几点出发？', '还需要知道你的出发地点、交通方式和要到达的时间。'),
            ] if zh else [
                (f'My order number is A{a}{b}{c}. When will it arrive?', 'I cannot determine delivery time from the order number alone. Please provide the tracking status or estimated delivery date.'),
                (f'My seat is number {a}. When does the event start?', 'The seat number does not indicate the start time. Please provide the event name or the time on your ticket.'),
                (f'The product code is P{a}{b}. What is its price now?', 'A product code alone does not tell me the price. Please provide the product name and store.'),
                (f'I need to reach building {a} tomorrow. When should I leave?', 'Please tell me your starting location, travel method, and required arrival time.'),
            ]
            question, answer = contexts[variant % 4]
            if validation:
                question, answer = ((f'快递编号为 Z{a}{b}，包裹重量是多少？', '仅凭快递编号无法知道包裹重量，请提供称重信息。') if zh else (f'Tracking code Z{a}{b} belongs to my package. How much does it weigh?', 'The tracking code alone does not provide its weight. Please provide the package weight measurement.'))
            turns = pair(question, answer)
        yield {'conversations': turns, 'language': language, 'task': task,
               'source': 'RiverValley/program_checked_v2', 'source_row': i,
               'split_group': group, 'forced_split': 'val' if validation else 'train',
               'verification': '程序构造与答案计算；非独立人工标注', 'proof': proof}


def translation_records(path):
    """双向翻译共用句对组，保留原句链接和整理来源。"""
    seen_en, seen_zh = set(), set()
    for index, line in enumerate(path.read_text(encoding='utf-8').splitlines()):
        pieces = line.split('\t')
        if len(pieces) != 3:
            continue
        en, zh, attribution = pieces
        if not 3 <= len(en.split()) <= 22 or not 4 <= len(zh) <= 70:
            continue
        if re.search(r'[<>\[\]{}]|https?://|fuck|shit|bitch', en, re.I):
            continue
        a, b = normalized(en), normalized(zh)
        if a in seen_en or b in seen_zh:
            continue
        seen_en.add(a); seen_zh.add(b)
        group = 'tatoeba/' + digest(en)
        for language, target, question, answer in [
            ('zh', 'en', '请将下面这句话翻译成英文，只输出译文：\n' + zh, en),
            ('en', 'zh', 'Translate the following sentence into Chinese. Return only the translation:\n' + en, zh)]:
            yield {'conversations': pair(question, answer), 'language': language, 'target_language': target,
                   'task': 'translation', 'source': 'Tatoeba/bdx33/hsk-cmn-eng', 'source_row': index,
                   'split_group': group, 'source_copyright': attribution,
                   'verification': 'Tatoeba 句对；仍需分层抽检，未声称逐条人工核验'}


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')


def summarize(rows):
    return {'rows': len(rows), 'tasks': dict(Counter(r['task'] for r in rows)),
            'languages': dict(Counter(r['language'] for r in rows)),
            'sources': dict(Counter(r['source'] for r in rows)),
            'assistant_turns': dict(Counter(r['assistant_turns'] for r in rows)),
            'tokens': sum(r['tokens'] for r in rows), 'answer_tokens': sum(r['answer_tokens'] for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-root', default='checkpoints/bilingual_20261011')
    parser.add_argument('--output', default='checkpoints/bilingual_v2_20261019')
    args = parser.parse_args()
    old, output = Path(args.old_root), Path(args.output)
    tokenizer = Tokenizer.from_file(str(old / 'tokenizer.json'))
    from langid.langid import LanguageIdentifier, model
    classifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    classifier.set_languages(['zh', 'en', 'fr', 'de', 'it', 'es', 'ja', 'ko', 'ru', 'ar'])
    probes = []
    for path in [Path('evals/bilingual_acceptance_v1.jsonl'), Path('evals/bilingual_acceptance_v2_holdout.jsonl')]:
        if path.exists():
            questions = [json.loads(line) for line in path.read_text().splitlines()]
            probes.extend(normalized(row['prompt']) for row in questions if len(normalized(row['prompt'])) >= 12)
            probes.extend(normalized(row['reference']) for row in questions if row.get('category') == 'translation' and len(normalized(row['reference'])) >= 8)
    old_groups = set()
    for path in (old / 'data/sft_train.jsonl', old / 'data/sft_val.jsonl'):
        for line in path.read_text(encoding='utf-8').splitlines():
            turns = json.loads(line)['conversations']
            old_groups.add(digest(next(t['content'] for t in turns if t['role'] == 'user')))
    buckets, stats, seen = defaultdict(list), Counter(), set()

    def add(row):
        kept, reason = annotate(row, tokenizer, probes, classifier)
        if reason:
            stats[reason] += 1
            return
        if kept['id'] in seen:
            stats['duplicate'] += 1
            return
        seen.add(kept['id'])
        split = kept.get('forced_split') or split_for(kept['split_group'])
        if split == 'val' and digest(next(t['content'] for t in kept['conversations'] if t['role'] == 'user')) in old_groups:
            stats['validation_seen_in_v1'] += 1
            return
        buckets[(split, kept['task'], kept['language'])].append(kept)

    sources = sorted((old / 'raw').glob('sft_*.parquet')) + sorted((output / 'raw').glob('sft_zh_magpie_*.parquet'))
    for path in sources:
        language = 'zh' if path.name.startswith('sft_zh_') else 'en'
        for index, raw in enumerate(read_rows([path])):
            row = sft_record(raw, path.name, language)
            stats['public_raw'] += 1
            if row is None:
                stats['structure'] += 1
                continue
            if row['group'] == 'alpaca_gpt4_zh':
                stats['rejected_alpaca_source_after_review'] += 1
                continue
            if row['group'] == 'magpie_zh' and (raw.get('difficulty') not in ('very easy', 'easy') or raw.get('input_quality') != 'good' or raw.get('instruct_reward', -10) < 0):
                stats['magpie_quality_or_difficulty'] += 1
                continue
            # 排除高风险或专业代码任务；不再误删普通分类、翻译及数字题。
            text = '\n'.join(t['content'] for t in row['conversations'])
            if re.search(r'诊断|处方|服药|用药|药物|病证|胶囊|药膏|现任|法律建议|投资建议|古诗上句|古诗下句|续写古诗|\b(?:prescription|diagnosis|current president)\b', text, re.I):
                stats['out_of_scope'] += 1
                continue
            if row['group'] == 'smoltalk_zh' and raw.get('score', 0) < 4:
                stats['source_score'] += 1
                continue
            original_turns = sum(t['role'] == 'assistant' for t in row['conversations'])
            while sum(t['role'] == 'assistant' for t in row['conversations']) > 2:
                turns = row['conversations']
                serialized = ''.join(f"<|im_start|>{t['role']}\n{t['content']}<|im_end|>\n" for t in turns)
                if len(tokenizer.encode(serialized).ids) <= 1025 and all(len(tokenizer.encode(t['content']).ids) <= (450 if t['role'] == 'assistant' else 600) for t in turns):
                    break
                row['conversations'] = turns[:-2]
            row['source_assistant_turns'] = original_turns
            if any(len(tokenizer.encode(t['content']).ids) > (450 if t['role'] == 'assistant' else 600) for t in row['conversations']):
                stats['long_turn'] += 1
                continue
            row['source_row'] = index
            add(row)
        log.info('公开来源完成：%s，候选累计=%d', path.name, sum(map(len, buckets.values())))
    for row in synthetic_records():
        add(row)
    for row in translation_records(output / 'raw/cmn.txt'):
        add(row)
    for rows in buckets.values():
        rows.sort(key=lambda row: digest('v2-order/' + row['id']))
    # 配方明确包含程序任务的比例；来源不足时明确失败，不用重复行补足。
    quotas = {'general': 2500, 'format': 1500, 'extraction': 1500, 'translation': 1500,
              'multi_turn': 1500, 'arithmetic': 1000, 'clarification': 500}
    pilot, full, validation = [], [], []
    for task, quota in quotas.items():
        for language, ratio in [('zh', .6), ('en', .4)]:
            rows = buckets[('train', task, language)]
            take = round(quota * ratio)
            if len(rows) < take:
                raise ValueError(f'材料不足：{task}/{language}，需要={take}，实际={len(rows)}')
            if task == 'general':
                counts = Counter(); selected = []
                for row in rows:
                    if counts[row['source']] >= max(1, take // 2):
                        continue
                    selected.append(row); counts[row['source']] += 1
                    if len(selected) == take:
                        break
                if len(selected) < take:
                    raise ValueError(f'通用问答来源不足以满足单来源半数上限：{language} {len(selected)}/{take}')
                pilot.extend(selected)
            else:
                pilot.extend(rows[:take])
            # 全量中通用问答约 2.4 万，翻译约 1 万，其余使用现有去重候选。
            cap = round((24000 if task == 'general' else 10000 if task == 'translation' else quota * 3) * ratio)
            chosen = rows[:cap]
            if task == 'general':
                counts = Counter(); chosen = []
                for row in rows:
                    limit = 6000
                    if counts[row['source']] >= limit:
                        continue
                    chosen.append(row); counts[row['source']] += 1
                    if len(chosen) == cap:
                        break
            full.extend(chosen)
            validation.extend(buckets[('val', task, language)][:max(30, take // 10)])
    for rows in (pilot, full, validation):
        random.Random(20261019).shuffle(rows)
    groups = {r['split_group'] for r in validation}
    assert not groups.intersection(r['split_group'] for r in full + pilot)
    paths = {'pilot': output / 'data/sft_pilot.jsonl', 'full': output / 'data/sft_train.jsonl', 'val': output / 'data/sft_val.jsonl'}
    for name, rows in [('pilot', pilot), ('full', full), ('val', validation)]:
        write_rows(paths[name], rows)
    # 分层样本用于人工抽检；不能把抽样文件的存在等同于已审核通过。
    review = [row for key, rows in sorted(buckets.items()) for row in rows[:8]]
    write_rows(output / 'sft_review_sample.jsonl', review)
    report = {'version': 2, 'seed': 20261019, 'tokenizer_sha256': sha256_file(old / 'tokenizer.json'),
              'filters': dict(stats), 'pilot': summarize(pilot), 'full': summarize(full), 'val': summarize(validation),
              'hashes': {name: sha256_file(path) for name, path in paths.items()},
              'review_status': 'pending', 'split_policy': '公共首问/双向句对分组；程序任务模板家族隔离',
              'rejected_source': 'OPUS-100 存在严重错译；Alpaca-GPT4 中文抽检多处任务和答案不匹配；均不进入修复组'}
    (output / 'sft_manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    log.info('第二版 SFT 生成完成：%s', json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    main()
