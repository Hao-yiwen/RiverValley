"""根据有记录的抽检收窄任务范围，保留修订前材料，补充可核对的简短多轮任务。"""
from collections import Counter, defaultdict
import json
import logging
from pathlib import Path
import random
import re
import shutil

from tokenizers import Tokenizer
from download_bilingual_sources import sha256_file
from prepare_sft_v2 import summarize, write_rows
from sft_quality import annotate, digest, normalized

log = logging.getLogger(__name__)


def dialogue_records():
    rng = random.Random(202610192)
    for i in range(6000):
        language = 'zh' if i % 2 == 0 else 'en'
        family, kind = (i // 2) % 20, (i // 40) % 5
        a, b, c = rng.sample(range(1, 100), 3)
        name, other = rng.sample(['小叶', '小夏', '小林', '小雨', '小海', '小月'] if language == 'zh'
                                 else ['Emma', 'Noah', 'Mia', 'Leo', 'Iris', 'Owen'], 2)
        zh = language == 'zh'
        if kind == 0:
            turns = [(f'仓库里有{a}箱铅笔。' if zh else f'The warehouse has {a} boxes of pencils.',
                      f'已记录：{a}箱。' if zh else f'Recorded: {a} boxes.'),
                     (f'更正一下，现在是{b}箱。' if zh else f'Correction: there are now {b} boxes.',
                      f'已更正为{b}箱。' if zh else f'Updated to {b} boxes.'),
                     ('现在有多少箱？只写数字。' if zh else 'How many boxes are there now? Return only the number.', str(b))]
        elif kind == 1:
            turns = [(f'{name}负责送{a}份材料，{other}负责送{b}份材料。' if zh
                      else f'{name} will deliver {a} documents, and {other} will deliver {b} documents.',
                      '两个人的任务已记下。' if zh else 'Both assignments are recorded.'),
                     (f'{other}的任务改为{c}份，其他不变。' if zh
                      else f'Change {other}\'s assignment to {c} documents; the other assignment is unchanged.',
                      '已更新。' if zh else 'Updated.'),
                     (f'{name}要送几份？只写数字。' if zh else f'How many documents will {name} deliver? Return only the number.', str(a))]
        elif kind == 2:
            items = ['苹果', '香蕉', '梨', '橙子', '葡萄', '桃子'] if zh else ['apples', 'bananas', 'pears', 'oranges', 'grapes', 'peaches']
            first, second, third = rng.sample(items, 3)
            turns = [(f'购物清单是：{first}、{second}、{third}。' if zh else f'The shopping list is {first}, {second}, and {third}.',
                      '已记下购物清单。' if zh else 'Shopping list recorded.'),
                     (f'从清单中删掉{second}和{third}。' if zh else f'Remove {second} and {third} from the list.',
                      '已删除这两项。' if zh else 'Both items removed.'),
                     ('还剩什么？只写名称。' if zh else 'What remains? Return only its name.', first)]
        elif kind == 3:
            turns = [(f'快递件数先记为{a}件，收件人是{name}。' if zh else f'Record {a} parcels for recipient {name}.',
                      '已记下件数和收件人。' if zh else 'Parcel count and recipient recorded.'),
                     (f'收件人改为{other}，件数不变。' if zh else f'Change the recipient to {other}, keeping the same parcel count.',
                      '收件人已更改。' if zh else 'Recipient updated.'),
                     ('现在的收件人是谁？只写名字。' if zh else 'Who is the recipient now? Return only the name.', other)]
        else:
            turns = [(f'比赛中，{name}得了{a}分，{other}得了{b}分。' if zh else f'In the game, {name} scored {a} and {other} scored {b}.',
                      '已记录两人的分数。' if zh else 'Both scores recorded.'),
                     (f'把{name}的分数改为{c}。' if zh else f'Change {name}\'s score to {c}.',
                      '已更正。' if zh else 'Corrected.'),
                     (f'{name}现在得了多少分？只写数字。' if zh else f'What is {name}\'s current score? Return only the number.', str(c))]
        if family == 0:
            # 验证使用完整的另一套提问措辞，不能只换一层前缀便声称模板隔离。
            questions = {
                0: lambda: ([f'铅笔包装箱的初始记录数量为{a}。', f'此前数量有误，应当记为{b}箱。', '依据修订记录给出箱数，不写其他内容。'] if zh else
                    [f'The initial recorded pencil-box count is {a}.', f'That count was mistaken; it should be {b}.', 'Give the corrected box count and nothing else.']),
                1: lambda: ([f'送材料的分工如下：{name}送{a}份，{other}送{b}份。', f'只调整{other}的任务，将其份数设为{c}。', f'用一个数字回答{name}的送件份数。'] if zh else
                    [f'Document delivery assignments: {name}: {a}; {other}: {b}.', f'Adjust only the assignment for {other} to {c}.', f'State the number assigned to {name}, with no other text.']),
                2: lambda: ([f'需要购买的三样东西分别为{first}、{second}、{third}。', f'不再购买{second}和{third}了。', '请仅给出仍需购买的那一样东西。'] if zh else
                    [f'The three purchases needed are {first}, {second}, and {third}.', f'We no longer need to buy {second} or {third}.', 'Name only the item we still need to purchase.']),
                3: lambda: ([f'当前寄送记录：{a}件包裹；接收者{name}。', f'保持数量，但把接收者修改成{other}。', '只给出修改后接收者的姓名。'] if zh else
                    [f'Current shipping record: {a} parcels; receiving person: {name}.', f'Keep the quantity, but set the receiving person to {other}.', 'Give only the updated receiving person\'s name.']),
                4: lambda: ([f'初始计分表为：{name}是{a}分，{other}是{b}分。', f'有一处需要修正：{name}实际得到{c}分。', f'只用数字表示修正后{name}的得分。'] if zh else
                    [f'Initial scoreboard: {name} has {a} points; {other} has {b}.', f'A correction is needed: {name} actually has {c} points.', f'Write only the corrected points total for {name}.'])
            }[kind]()
            turns = [(question, pair[1]) for question, pair in zip(questions, turns)]
        conversations = [{'role': role, 'content': content} for pair in turns
                         for role, content in zip(('user', 'assistant'), pair)]
        yield {'language': language, 'conversations': conversations, 'source': 'RiverValley/program_checked_v2',
               'task': 'multi_turn', 'forced_split': 'val' if family == 0 else 'train',
               'split_group': f'v2-dialogue:{kind}:{family}',
               'proof': {'kind': kind, 'a': a, 'b': b, 'c': c, 'last_answer': turns[-1][1]}}


def rejection(row):
    text = '\n'.join(turn['content'] for turn in row['conversations'])
    question = ' '.join(turn['content'] for turn in row['conversations'] if turn['role'] == 'user')
    if row['source'].endswith(('/zhihu', '/chinese_traditional', '/explore-instruct-rewrite')):
        return 'forum_classical_or_rewrite_source_removed'
    system = ' '.join(t['content'] for t in row['conversations'] if t['role'] == 'system')
    answer = ' '.join(t['content'] for t in row['conversations'] if t['role'] == 'assistant')
    if (re.search(r'without.*pronouns', system, re.I)
            and re.search(r'\b(?:he|she|they|them|their|you|your|his|her)\b', answer, re.I)):
        return 'explicit_pronoun_constraint_violation'
    if re.search(r'远期利率|金融衍生|金融资产|仲裁|劳动法|劳动局|诉讼|股票|基金|症状|治疗|心脏病|肿瘤|\b(?:cancer|diagnosis|treatment|medication|derivatives)\b', text, re.I):
        return 'specialist_topic'
    if row['task'] == 'general' and (row['answer_tokens'] > 220 or row['tokens'] - row['answer_tokens'] > 350):
        return 'long_general'
    if row['task'] == 'general':
        if re.search(r'日语|日文|韩文|韩语|法文|法语|西班牙文|西班牙语', question):
            return 'outside_bilingual_goal'
        if '标题' in question and len([line for line in answer.splitlines() if line.strip()]) > 1:
            return 'title_request_with_extra_paragraphs'
        if (re.search(r'\d', question) and re.search(r'calculate|how (?:many|much)|多少|求|计算|mass|distance|速度|重力|引力|连续.{0,4}数', question, re.I)):
            return 'unverified_public_numeric_solution'
    if row['task'] == 'multi_turn' and row['answer_tokens'] > 450:
        return 'long_multi_turn'
    if row['task'] == 'translation':
        if row['source'] != 'Tatoeba/bdx33/hsk-cmn-eng':
            return 'unreviewed_translation_source'
        zh = row['conversations'][0]['content'].split('\n', 1)[-1] if row['language'] == 'zh' else row['conversations'][1]['content']
        if len(re.findall(r'[\u3400-\u9fff]', zh)) <= 5 or '见意' in zh or re.search(r'([\u3400-\u9fff]{2,4})\1', zh):
            return 'very_short_ambiguous_pair_or_typo'
        if '汽车有四个轮胎' in zh:
            return 'reviewed_wheel_tire_mismatch'
    return None


def main():
    root = Path('checkpoints/bilingual_v2_20261019')
    draft = root / 'draft_before_review_1'
    if not draft.exists():
        draft.mkdir()
        for name in ('sft_train.jsonl', 'sft_pilot.jsonl', 'sft_val.jsonl'):
            shutil.copy2(root / 'data' / name, draft / name)
        shutil.copy2(root / 'sft_manifest.json', draft / 'sft_manifest.json')
    original = json.loads((draft / 'sft_manifest.json').read_text())
    for key, name in [('full', 'sft_train'), ('pilot', 'sft_pilot'), ('val', 'sft_val')]:
        if sha256_file(draft / (name + '.jsonl')) != original['hashes'][key]:
            raise ValueError('修订前冻结数据摘要不匹配')
    stats, buckets, seen = Counter(), defaultdict(list), set()
    for name in ('sft_train', 'sft_pilot', 'sft_val'):
        split = 'val' if name == 'sft_val' else 'train'
        for line in (draft / (name + '.jsonl')).read_text().splitlines():
            row = json.loads(line)
            if row['id'] in seen:
                continue
            seen.add(row['id'])
            reason = rejection(row)
            if reason:
                stats[reason] += 1
                continue
            buckets[(split, row['task'], row['language'])].append(row)
    tokenizer = Tokenizer.from_file('checkpoints/bilingual_20261011/tokenizer.json')
    probes = []
    for path in ['evals/bilingual_acceptance_v1.jsonl', 'evals/bilingual_acceptance_v2_holdout.jsonl']:
        probes.extend(normalized(json.loads(line)['prompt']) for line in Path(path).read_text().splitlines())
    for row in dialogue_records():
        kept, reason = annotate(row, tokenizer, probes)
        if reason:
            stats['new_dialogue_' + reason] += 1
            continue
        if kept['id'] in seen:
            continue
        seen.add(kept['id'])
        buckets[(row['forced_split'], 'multi_turn', row['language'])].append(kept)
    quotas = {'general': 2500, 'format': 1500, 'extraction': 1500, 'translation': 1500,
              'multi_turn': 1500, 'arithmetic': 1000, 'clarification': 500}
    pilot, full, val = [], [], []
    for task, quota in quotas.items():
        for lang, ratio in [('zh', .6), ('en', .4)]:
            rows = sorted(buckets[('train', task, lang)], key=lambda row: digest('refined/' + row['id']))
            take = round(quota * ratio)
            source_counts, selected = Counter(), []
            for row in rows:
                if task == 'general' and source_counts[row['source']] >= round(take * (.6 if lang == 'zh' else .5)):
                    continue
                selected.append(row); source_counts[row['source']] += 1
                if len(selected) == take:
                    break
            if len(selected) != take:
                raise ValueError(f'精筛后材料不足：{task}/{lang} {len(selected)}/{take}')
            pilot.extend(selected); full.extend(rows)
            validation = sorted(buckets[('val', task, lang)], key=lambda row: digest('refined/' + row['id']))
            val.extend(validation[:max(30, take // 10)])
    val_groups = {x['split_group'] for x in val}
    if val_groups.intersection(x['split_group'] for x in full):
        raise ValueError('修订数据的分组存在训练验证重叠')
    paths = {}
    for key, filename, rows in [('pilot', 'sft_pilot', pilot), ('full', 'sft_train', full), ('val', 'sft_val', val)]:
        random.Random(202610192).shuffle(rows)
        paths[key] = root / 'data' / (filename + '.jsonl')
        write_rows(paths[key], rows)
    result = dict(original, revision='review_refined_1', review_status='pending_final_sample',
        prior_hashes=original['hashes'], refinement_filters=dict(stats),
        pilot=summarize(pilot), full=summarize(full), val=summarize(val),
        hashes={key: sha256_file(path) for key, path in paths.items()},
        refinement_code_sha256=sha256_file(Path(__file__)),
        quality_limits='来源分数与抽样不能保证全部事实正确；极短配对及已发现错配被排除，多轮补充含较多程序模板。')
    (root / 'sft_manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    log.info('精筛完成：pilot=%d full=%d val=%d，过滤=%s', len(pilot), len(full), len(val), dict(stats))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    main()
