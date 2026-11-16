"""构造位置、键名和修改对象均衡的修复实验；保留旧材料及独立留出。"""
import argparse
from collections import Counter
import hashlib
import json
import logging
from pathlib import Path
import random

log = logging.getLogger(__name__)
TASKS = ('extraction', 'json', 'sort', 'multi_turn', 'arithmetic', 'copy')


def sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def pair(question, answer):
    return [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}]


def make_row(task, language, index, split, rng):
    """答案由结构化事实生成；查询位置与措辞、语言、任务独立变化。"""
    zh = language == 'zh'
    family = index % 4 if split == 'train' else (4 if split == 'dev' else 5)
    names = rng.sample(['林宁', '何舟', '陈悦', '周言', '夏安', '方雨', '白露', '江辰'] if zh else
                       ['Alice', 'Ben', 'Clara', 'David', 'Elena', 'Finn', 'Grace', 'Hugo'], 3)
    values = rng.sample(range(1, 150), 3)
    query = rng.randrange(3)
    answer = str(values[query])
    facts = dict(zip(names, values))
    proof = {'facts': facts, 'query': names[query], 'position': query}
    if task == 'extraction':
        text = ('；' if zh else '; ').join(f'{name}: {value}' for name, value in facts.items())
        forms = [f'读取以下记录：{text}。{names[query]}对应的数字是多少？只写数字。',
                 f'资料为【{text}】。只返回{names[query]}的数值。',
                 f'{text}\n问题：{names[query]}是多少？答案不要解释。',
                 f'查表得到{names[query]}的数值，只输出数值。表：{text}',
                 f'下面列出三人的数量：{text}。请仅报告{names[query]}的数量。',
                 f'参考数据 {text}；我想知道{names[query]}那一项的值，请用数字回答。'] if zh else [
                 f'Read this record: {text}. What number belongs to {names[query]}? Digits only.',
                 f'Data: [{text}]. Return only the value for {names[query]}.',
                 f'{text}\nQuestion: how many for {names[query]}? No explanation.',
                 f'Look up {names[query]} and output its value only. Table: {text}',
                 f'The quantities for three people are {text}. Report only the quantity for {names[query]}.',
                 f'Using {text}, tell me the value assigned to {names[query]}. Give digits, nothing else.']
        turns = pair(forms[family], answer)
    elif task == 'json':
        keys = rng.sample(['name', 'label', 'person', 'owner', 'title', 'city', 'count', 'units', 'score', 'value', 'total', 'size'], 2)
        assignments = {keys[0]: names[0], keys[1]: values[0]}
        if rng.randrange(2):
            assignments = dict(reversed(list(assignments.items())))
        text = '；'.join(f'{k} = {json.dumps(v, ensure_ascii=False)}' for k, v in assignments.items())
        forms = [f'只输出 JSON 对象，包含这些字段：{text}。', f'把以下字段写为合法 JSON：{text}。不要增加其他键或说明。',
                 f'字段及值：{text}\n请返回 JSON，保留字符串与数字类型。', f'按要求生成 JSON：{text}。回答只能有对象。',
                 f'请把 {text} 编码为一个 JSON 对象，不加额外文字。', f'返回只有以下键值的 JSON 对象：{text}。'] if zh else [
                 f'Output only a JSON object containing these fields: {text}.', f'Convert these fields to valid JSON: {text}. No additional keys or explanation.',
                 f'Fields and values: {text}\nReturn JSON, preserving string and number types.', f'Create JSON with {text}. Reply with the object alone.',
                 f'Encode {text} as one JSON object, with no extra text.', f'Return a JSON object with exactly these key-value pairs: {text}.']
        answer = json.dumps(assignments, ensure_ascii=False)
        proof = {'object': assignments}
        turns = pair(forms[family], answer)
    elif task == 'sort':
        descending = rng.choice([False, True])
        order = ('从大到小' if descending else '从小到大') if zh else ('descending' if descending else 'ascending')
        text = ', '.join(map(str, values))
        forms = [f'把 {text} 按{order}排列，用逗号和空格分隔。只返回列表。', f'数字：{text}。请按{order}排序，以“, ”连接。不要说明。',
                 f'只写排序结果：{text}，顺序为{order}，分隔符为“, ”。', f'请输出 {text} 的{order}列表，数字之间用英文逗号和一个空格。',
                 f'对数列 {text} 进行{order}排序。仅用“, ”分隔排序后的数。', f'数值为 {text}。以{order}顺序重排，仅输出以逗号空格连接的数列。'] if zh else [
                 f'Sort {text} in {order} order. Use comma and space. Return only the list.', f'Numbers: {text}. Sort in {order} order and join with ", ". No explanation.',
                 f'Output the {order} ordering of {text}, separated by ", ".', f'Give a {order} list of {text}. Separate numbers with a comma and one space.',
                 f'Arrange the sequence {text} in {order} order. Use only ", " between numbers.', f'The values are {text}. Reorder them in {order} order; output comma-space-separated numbers only.']
        answer = ', '.join(map(str, sorted(values, reverse=descending)))
        proof = {'values': values, 'descending': descending}
        turns = pair(forms[family], answer)
    elif task == 'multi_turn':
        changed = rng.randrange(3)
        new = rng.choice([n for n in range(1, 150) if n not in values])
        text = ('；' if zh else '; ').join(f'{k}: {v}' for k, v in facts.items())
        starts = [f'请记录：{text}。', f'初始数值是 {text}。', f'有以下三项：{text}。', f'接下来使用这份记录：{text}。', f'先保存这份清单：{text}。', f'我们有三个人的数据，{text}。'] if zh else [f'Please record: {text}.', f'The initial values are {text}.', f'Here are three entries: {text}.', f'Use this record from now on: {text}.', f'First, save this list: {text}.', f'We have data for three people: {text}.']
        updates = [f'把{names[changed]}改为{new}，其他不变。', f'更正：{names[changed]}应为{new}。', f'只更新{names[changed]}为{new}。', f'{names[changed]}的最新值是{new}。', f'需要修改其中一项：{names[changed]}现在是{new}。', f'最新记录中，{names[changed]}是{new}，其余数据不变。'] if zh else [f'Change {names[changed]} to {new}; keep the rest.', f'Correction: {names[changed]} should be {new}.', f'Update only {names[changed]} to {new}.', f'The latest value for {names[changed]} is {new}.', f'One entry needs a change: {names[changed]} is now {new}.', f'In the latest record, {names[changed]} has {new}. Other entries are unchanged.']
        questions = [f'{names[query]}现在是多少？只写数字。', f'现在请只返回{names[query]}的值。', f'只写{names[query]}最新的数字。', f'{names[query]}的当前值？不要解释。', f'根据修改后的记录，仅报告{names[query]}的数值。', f'综合以上信息，{names[query]}最终对应哪个数字？只返回数字。'] if zh else [f'What is {names[query]} now? Digits only.', f'Now return only the value for {names[query]}.', f'Write only the latest number for {names[query]}.', f'Current value for {names[query]}? No explanation.', f'Based on the corrected record, report only the value for {names[query]}.', f'Taking the changes into account, what number is finally assigned to {names[query]}? Digits only.']
        facts[names[changed]] = new
        answer = str(facts[names[query]])
        proof.update(changed=names[changed], new=new, queried_changed=changed == query)
        turns = pair(starts[family], '已记录。' if zh else 'Recorded.') + pair(updates[family], '已更新。' if zh else 'Updated.') + pair(questions[family], answer)
    elif task == 'arithmetic':
        op = rng.choice(['+', '-', '×', '÷'])
        a, b = rng.randrange(1, 21), rng.randrange(1, 21)
        if op == '×':
            a, b = rng.randrange(1, 10), rng.randrange(1, 10)
        if op == '÷':
            a *= b
        if op == '-':
            a, b = max(a, b), min(a, b)
        answer = str({'+': lambda: a+b, '-': lambda: a-b, '×': lambda: a*b, '÷': lambda: a//b}[op]())
        x = f'{a} {op} {b}'
        forms = [f'{x} = ? 只写结果。', f'计算：{x}。不要解释，只给数字。', f'{x} 等于几？只输出数字答案。', f'请直接写出 {x} 的结果。', f'这个运算的答案是什么：{x}？仅写数字。', f'完成算式 {x}，回复结果数字即可。'] if zh else [f'{x} = ? Only the result.', f'Calculate: {x}. No explanation, just the number.', f'What is {x}? Output only the numerical answer.', f'Give just the result of {x}.', f'What is the answer to this operation: {x}? Digits only.', f'Complete the calculation {x}; reply with the resulting number alone.']
        proof = {'a': a, 'b': b, 'operator': op}
        turns = pair(forms[family], answer)
    else:
        word = rng.choice(['清晨', '湖水', '星光', '安静', '木桥', '纸船', '山谷', '白云'] if zh else ['morning', 'lake', 'starlight', 'quiet', 'bridge', 'boat', 'valley', 'cloud'])
        answer = word + str(rng.randrange(100, 1000))
        forms = [f'只回复“{answer}”，不要其他文字。', f'原样输出以下内容：{answer}', f'请逐字复制：{answer}。不要加引号和解释。', f'你的回答只能是：{answer}', f'把标记【{answer}】中的文字原样返回，不包括括号。', f'我需要你返回这段原文且不附加说明：{answer}'] if zh else [f'Reply only with "{answer}" and nothing else.', f'Output this text exactly: {answer}', f'Copy verbatim: {answer}. Do not add quotes or explanation.', f'Your answer must be exactly: {answer}', f'Return the text inside [{answer}] exactly, excluding the brackets.', f'I need this literal text back without any additions: {answer}']
        proof = {'literal': answer}
        turns = pair(forms[family], answer)
    raw = json.dumps(turns, ensure_ascii=False, sort_keys=True)
    return {'id': hashlib.sha256(raw.encode()).hexdigest(), 'conversations': turns,
            'task': task, 'language': language, 'family': family, 'split': split,
            'source': 'RiverValley/program_checked_repair_20261026', 'proof': proof}


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--old-data', default='checkpoints/bilingual_v2_20261019/formal_candidate_05')
    parser.add_argument('--tokenizer', default='checkpoints/delivery_v2_20261019/sft_best/tokenizer.json')
    args = parser.parse_args()
    output, old = Path(args.output), Path(args.old_data)
    if output.exists():
        raise FileExistsError('拒绝覆盖修复材料')
    old_manifest = json.loads((old/'manifest.json').read_text())
    if sha(old/'sft_train.jsonl') != old_manifest['hashes']['sft_train.jsonl']:
        raise ValueError('回放来源摘要不匹配')
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)
    seen, splits = set(), {}
    for split, count, seed in [('train', 2000, 20261026), ('dev', 16, 20261027), ('holdout', 32, 20261028)]:
        rng, rows = random.Random(seed), []
        for task in TASKS:
            for language in ('zh', 'en'):
                accepted = 0
                while accepted < count:
                    row = make_row(task, language, accepted, split, rng)
                    if row['id'] in seen:
                        continue
                    seen.add(row['id']); rows.append(row); accepted += 1
        splits[split] = rows
    old_rows = [json.loads(line) for line in (old/'sft_train.jsonl').read_text().splitlines()]
    replay = []
    for language in ('zh', 'en'):
        candidates = [r for r in old_rows if r['language'] == language and r['task'] in ('general', 'translation', 'rewrite', 'summary')]
        candidates.sort(key=lambda r: hashlib.sha256(('repair-replay/'+r['id']).encode()).hexdigest())
        if len(candidates) < 4000:
            raise ValueError('自然语言回放不足，不以重复样本补足')
        replay += [dict(r, split='train', replay=True) for r in candidates[:4000]]
    train = splits['train'] + replay
    random.Random(20261026).shuffle(train)
    # 原验证集只用于监测回放任务；新留出不参与训练或选优。
    val = splits['dev'] + [dict(json.loads(line), replay=True) for line in (old/'sft_val.jsonl').read_text().splitlines() if json.loads(line)['task'] in ('general', 'translation', 'rewrite', 'summary')]
    manifest = {'purpose': 'small_behavior_repair_probe', 'generator_sha256': sha(__file__),
                'old_train_sha256': sha(old/'sft_train.jsonl'), 'tokenizer_sha256': sha(args.tokenizer),
                'source_note': '程序生成任务并非新增预训练语料；8000条自然任务为旧训练回放。未读取旧240题。', 'splits': {}}
    output.mkdir(parents=True)
    for split, rows in [('train', train), ('val', val)]:
        maximum = 0
        answer_tokens = 0
        for row in rows:
            serialized = ''.join(f"<|im_start|>{t['role']}\n{t['content']}<|im_end|>\n" for t in row['conversations'])
            length = len(tokenizer.encode(serialized).ids)
            maximum = max(maximum, length)
            if length > 1025:
                raise ValueError('修复材料超长，拒绝静默截断：'+row['id'])
            answer_tokens += sum(len(tokenizer.encode(t['content']).ids) for t in row['conversations'] if t['role'] == 'assistant')
        write_rows(output/f'sft_{split}.jsonl', rows)
        manifest['splits'][split] = {'rows': len(rows), 'tasks': dict(Counter(r['task'] for r in rows)), 'max_tokens': maximum, 'answer_content_tokens': answer_tokens}
    for split in ('dev', 'holdout'):
        questions = []
        for row in splits[split]:
            turns = row['conversations']
            questions.append({'id': 'repair-'+row['id'], 'language': row['language'], 'category': row['task'],
                'prompt': turns[-2]['content'], 'messages': turns[:-1], 'reference': turns[-1]['content'],
                'scoring': 'json' if row['task'] == 'json' else 'exact', 'family': row['family']})
        write_rows(output/(split+'.jsonl'), questions)
    old_dev = [json.loads(x) for x in Path('evals/bilingual_acceptance_v1.jsonl').read_text().splitlines()]
    new_dev = [json.loads(x) for x in (output/'dev.jsonl').read_text().splitlines()]
    write_rows(output/'development.jsonl', new_dev+old_dev)
    manifest['hashes'] = {p.name: sha(p) for p in sorted(output.glob('*.jsonl'))}
    manifest['evaluation_note'] = '192题新开发集与384题新留出各用未训练的措辞家族；仍是内部合成题，不能证明通用能力。旧100题只监测回归。'
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    log.info('修复数据已冻结：训练=%d 回放=%d 验证=%d 新开发=%d 新留出=%d；未申请GPU', len(train), len(replay), len(val), len(new_dev), len(splits['holdout']))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
