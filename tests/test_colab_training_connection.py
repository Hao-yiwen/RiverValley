"""验证输出断线后只重新订阅，不重复执行训练。"""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location('connection', Path(__file__).parents[1] / 'src/scripts/colab_training_connection.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_disconnect_refreshes_credentials_without_execution(tmp_path):
    sessions = iter([SimpleNamespace(url='https://example.test', token='old', endpoint='original'),
                     SimpleNamespace(url='https://example.test', token='fresh', endpoint='original'), None])
    urls = []
    received = iter([ConnectionError(), json.dumps({'msg_type': 'stream', 'content': {'text': '训练步数 2020\n'}}), ConnectionError()])
    class Socket:
        def recv(self):
            value = next(received)
            if isinstance(value, Exception):
                raise value
            return value
        def close(self):
            pass
        def send(self, data):
            raise AssertionError('只读订阅不得发送执行请求')
    def connect(url, **kwargs):
        urls.append(url)
        return Socket()
    module.consume('run', 'kernel', 'original', tmp_path, 10,
                   lambda *a, **kw: next(sessions), connect, clock=lambda: 1, sleep=lambda _: None)
    assert len(urls) == 2 and 'old' in urls[0] and 'fresh' in urls[1]
    assert (tmp_path / 'training_stream.log').read_text() == '训练步数 2020\n'
    assert json.loads((tmp_path / 'connection_heartbeat.json').read_text())['connections'] == 2


def test_replacement_instance_is_not_connected(tmp_path):
    module.consume('run', 'kernel', 'original', tmp_path, 10,
                   lambda *a, **kw: SimpleNamespace(endpoint='replacement'),
                   lambda *a, **kw: (_ for _ in ()).throw(AssertionError('不得连接替代实例')),
                   clock=lambda: 1)
