"""验证本地聊天边界，不加载完整模型或申请云端计算资源。"""
import json
from pathlib import Path
import queue
import sys
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts'))
from local_chat import ChatEngine, ChatHandler, ThreadingHTTPServer, fit_context, validate_request


def request_body(**changes):
    return {'request_id': 'test-request-001', 'messages': [{'role': 'user', 'content': '你好'}], **changes}


@pytest.mark.parametrize('changes', [
    {'max_tokens': True}, {'max_tokens': 1000}, {'temperature': float('nan')},
    {'repetition_penalty': float('inf')}, {'messages': [{'role': 'system', 'content': 'x'}]},
    {'messages': [{'role': 'user', 'content': ' '}]},
    {'messages': [{'role': 'user', 'content': 'x'}, {'role': 'assistant', 'content': 'y'}]},
])
def test_reject_invalid_generation(changes):
    with pytest.raises(ValueError):
        validate_request(request_body(**changes))


def test_context_preserves_latest_question_and_whole_turns():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return [m['content'] for m in messages for _ in range(10)]
    messages = [{'role': role, 'content': str(i)} for i, role in enumerate(['user', 'assistant', 'user'])]
    ids, removed = fit_context(Tokenizer(), messages, 20, 35)
    assert ids == ['2'] * 10 and removed == 1
    assert len(messages) == 3
    with pytest.raises(ValueError, match='当前问题太长'):
        fit_context(Tokenizer(), messages[-1:], 30, 35)


def test_cancel_does_not_stop_another_request():
    engine = ChatEngine('.', '.', 'cpu')
    job = {'id': 'active-job', 'stop': threading.Event(), 'events': queue.Queue()}
    engine.active = job
    assert not engine.cancel('other-job')
    assert not job['stop'].is_set()
    assert engine.cancel('active-job')
    assert job['stop'].is_set()
    engine.state = 'ready'
    with pytest.raises(RuntimeError, match='另一个回答'):
        engine.start(request_body())


def test_http_origin_validation_and_stream_protocol():
    class Engine:
        def status(self):
            return {'state': 'ready'}

        def start(self, body):
            validate_request(body)
            events = queue.Queue()
            for name, data in [('meta', {'input_tokens': 2}), ('delta', {'text': '你好'}),
                               ('done', {'finish_reason': 'stop'})]:
                events.put((name, data))
            return {'id': body['request_id'], 'events': events}

    server = ThreadingHTTPServer(('127.0.0.1', 0), ChatHandler)
    server.engine = Engine()
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f'http://127.0.0.1:{server.server_port}'
    try:
        assert json.load(urlopen(url + '/api/status'))['state'] == 'ready'
        with pytest.raises(HTTPError) as failure:
            urlopen(Request(url + '/api/status', headers={'Origin': 'https://example.com'}))
        assert failure.value.code == 403
        with pytest.raises(HTTPError) as failure:
            urlopen(Request(url + '/api/chat', data=b'not-json', headers={'Content-Type': 'application/json'}))
        assert failure.value.code == 400
        response = urlopen(Request(url + '/api/chat', data=json.dumps(request_body()).encode(),
                                  headers={'Content-Type': 'application/json'})).read().decode()
        assert 'event: delta\ndata: {"text": "你好"}' in response
        assert 'event: done' in response
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
