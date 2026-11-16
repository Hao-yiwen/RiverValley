"""通过已启动的本地聊天服务运行冻结题集，复用模型，逐题保存真实流式输出。"""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import uuid

ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger('rivervalley.local_eval')


def ask(messages, *, port=8765, max_tokens=160, repetition_penalty=1.0):
    """等待已有页面请求完成；串行推理，保留解码设置和完成状态。"""
    body = {'request_id': str(uuid.uuid4()), 'messages': messages, 'temperature': 0,
            'max_tokens': max_tokens, 'repetition_penalty': repetition_penalty}
    base = f'http://127.0.0.1:{port}'
    deadline = time.monotonic() + 180
    while True:
        request = Request(base + '/api/chat', data=json.dumps(body).encode(),
                          headers={'Content-Type': 'application/json'}, method='POST')
        try:
            response = urlopen(request, timeout=60)
            break
        except HTTPError as error:
            if error.code != 409 or time.monotonic() >= deadline:
                raise
            logger.info('模型正在加载或处理另一条请求，稍后重试')
            time.sleep(2)
    answer, event, meta, done = '', '', None, None
    with response:
        for raw in response:
            line = raw.decode('utf-8').rstrip('\n')
            if line.startswith('event: '):
                event = line[7:]
            elif line.startswith('data: '):
                value = json.loads(line[6:])
                if event == 'delta':
                    answer += value['text']
                elif event == 'meta':
                    meta = value
                elif event == 'done':
                    done = value
                elif event == 'error':
                    raise RuntimeError(value['message'])
    if not meta or not done or done['finish_reason'] == 'cancelled':
        raise RuntimeError('生成未完整结束，不将部分输出计作正常评估')
    return {'answer': answer, 'generation': meta, 'completion': done}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--questions', type=Path, default=ROOT / 'evals/bilingual_acceptance_v1.jsonl')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--repetition-penalty', type=float, default=1.0)
    args = parser.parse_args()
    questions = [json.loads(line) for line in args.questions.read_text(encoding='utf-8').splitlines()]
    digest = hashlib.sha256(args.questions.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # 使用独占创建，防止新一轮评估覆盖已有真实输出和评分。
    with args.output.open('x', encoding='utf-8') as sink:
        for index, question in enumerate(questions, 1):
            messages = question.get('messages') or [{'role': 'user', 'content': question['prompt']}]
            result = ask(messages, port=args.port, repetition_penalty=args.repetition_penalty)
            score = None
            if question.get('scoring') == 'exact':
                score = 2 if result['answer'].strip() == question['reference'] else 0
            elif not result['answer'].strip():
                score = 0
            row = {**question, **result, 'score': score, 'question_sha256': digest,
                   'model': str(ROOT / 'checkpoints/delivery_20261018/sft_best')}
            sink.write(json.dumps(row, ensure_ascii=False) + '\n')
            sink.flush()
            logger.info('完成本地评估 %d/%d：id=%s，字符=%d，自动评分=%s',
                        index, len(questions), row['id'], len(row['answer']), score)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
