"""第二版任务感知清洗：短答案、翻译、多轮和分组隔离规则。"""
import hashlib
import json
import re
import unicodedata

SPECIAL = ('<|pad|>', '<|endoftext|>', '<|im_start|>', '<|im_end|>')


def normalized(text):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text)).casefold()


def digest(text):
    return hashlib.sha256(normalized(text).encode('utf-8')).hexdigest()


def translation_target(question):
    """只根据明确的翻译指令推断目标语言，不凭答案语言反向放宽规则。"""
    rules = (
        ('en', r'(?:翻译|译|转换).{0,12}(?:英文|英语)|(?:into|to)\s+english\b'),
        ('zh', r'(?:翻译|译|转换).{0,12}(?:中文|汉语)|(?:into|to)\s+(?:chinese|mandarin)\b'),
    )
    if not re.search(r'翻译|译成|译为|translate|translation', question, re.I):
        return None
    found = [language for language, pattern in rules if re.search(pattern, question, re.I)]
    return found[0] if len(found) == 1 else None


def answer_language_ok(answer, expected, classifier=None):
    """无字母的数字/结构答案不作语言分类；较长自然语言再使用识别器。"""
    if not answer.strip():
        return False
    han = len(re.findall(r'[\u3400-\u9fff]', answer))
    latin = len(re.findall(r'[a-zA-Z]', answer))
    if not han and not latin:
        return bool(re.search(r'\d', answer))
    if re.fullmatch(r'(?:true|false|yes|no|null|none|是|否|对|错)', answer.strip(), re.I):
        return True
    if expected == 'zh':
        return han / max(1, han + latin) >= 0.35
    if expected == 'en':
        if han / max(1, han + latin) > 0.05:
            return False
        return not (classifier and len(answer) >= 80 and classifier.classify(answer)[0] != 'en')
    return False


def annotate(record, tokenizer, probes=(), classifier=None, max_tokens=1024):
    """返回新记录或明确拒绝原因，不裁掉超长对话中的依赖上下文。"""
    turns = record.get('conversations', [])
    start = 1 if turns and turns[0].get('role') == 'system' else 0
    body = turns[start:]
    if not body or len(body) % 2 or len(body) > 8:
        return None, 'incomplete_turns'
    if any(t.get('role') != ('user' if i % 2 == 0 else 'assistant') for i, t in enumerate(body)):
        return None, 'role_order'
    if any(not isinstance(t.get('content'), str) or not t['content'].strip() for t in turns):
        return None, 'empty_content'
    joined = '\n'.join(t['content'] for t in turns)
    if '\ufffd' in joined or any(token in joined for token in SPECIAL):
        return None, 'encoding_or_control'
    compact = normalized(joined)
    if any(probe in compact for probe in probes):
        return None, 'evaluation_overlap'
    answer_languages = []
    for i in range(0, len(body), 2):
        question, answer = body[i]['content'], body[i + 1]['content']
        target = record.get('target_language') or translation_target(question) or record['language']
        structured = False
        if record.get('task') == 'format' and re.search(r'\bJSON\b', question, re.I):
            try:
                structured = isinstance(json.loads(answer), (dict, list))
            except ValueError:
                return None, 'invalid_requested_json'
            if not structured:
                return None, 'invalid_requested_json'
        if not structured and not answer_language_ok(answer, target, classifier):
            return None, 'answer_language'
        answer_languages.append(target)
        if re.search(r'(.)\1{14,}', answer):
            return None, 'character_repeat'
        lines = [normalized(line) for line in answer.splitlines() if line.strip()]
        if len(lines) >= 4 and len(set(lines)) / len(lines) < 0.7:
            return None, 'line_repeat'
        if len(normalized(answer)) > 15 and normalized(answer) == normalized(question):
            return None, 'echo'
        if (record.get('group') == 'smoltalk_zh' and len(normalized(answer)) >= 60
                and normalized(answer) in normalized(question)):
            return None, 'answer_leaked_in_question'
        if len(answer) >= 80 and re.search(r'[,;:]$|\b(?:and|or|the|of|to)$', answer.strip(), re.I):
            return None, 'unfinished_answer'
        if re.search(r'ChatGPT|OpenAI|GPT-4|作为.{0,8}(?:模型|人工智能)|我是.{0,6}AI语言模型|as an ai language model', answer, re.I):
            return None, 'provider_or_boilerplate'
        if '```' in joined or re.search(r'编写.{0,5}(?:程序|代码)|\b(?:python|javascript|sql)\b', question, re.I):
            return None, 'code_out_of_scope'
        if record.get('group') == 'smol-contraints':
            return None, 'unverified_complex_constraints'
    text = ''.join(f"<|im_start|>{t['role']}\n{t['content']}<|im_end|>\n" for t in turns)
    encoded = tokenizer.encode(text)
    if len(encoded.ids) > max_tokens + 1:
        return None, 'context_overflow'
    # 统计与训练一致的偏移量监督，包含每一轮的结束标记，不包含用户和助手角色头。
    spans, offset = [], 0
    for turn in turns:
        prefix = f"<|im_start|>{turn['role']}\n"
        part = prefix + turn['content'] + '<|im_end|>\n'
        if turn['role'] == 'assistant':
            spans.append((offset + len(prefix), offset + len(part)))
        offset += len(part)
    supervised = sum(any(end > left and begin < right for left, right in spans) for begin, end in encoded.offsets[1:])
    if not supervised:
        return None, 'no_supervision'
    key = digest(json.dumps(turns, ensure_ascii=False, sort_keys=True))
    # 同一个首问的续写不能横跨训练与验证；翻译双向使用调用方提供的同一分组。
    group = record.get('split_group') or digest(body[0]['content'])
    task = record.get('task') or ('multi_turn' if len(body) > 2 else
            'translation' if translation_target(body[0]['content']) else 'general')
    return dict(record, id=key, split_group=group, task=task, tokens=len(encoded.ids),
                answer_tokens=supervised, assistant_turns=len(body) // 2,
                answer_languages=answer_languages), None


def split_for(group):
    """整个题意/模板家族只落入一个分片，不能只按随机行拆分。"""
    return 'val' if int(digest(group)[:8], 16) % 20 == 0 else 'train'
