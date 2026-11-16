"""验证提交前断线的恢复边界，防止向忙碌内核重复发送训练。"""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts'))
import launch_version as launch

TRACE = 'Ensure we are in /content\nRuntimeError: Connection was lost.'
ABSENT = {'pipeline.pid': 404, 'pipeline.log': 404, 'status.json': 404}


@pytest.mark.parametrize('kernels,files,text,allowed', [
    ([{'execution_state': 'idle'}], ABSENT, TRACE, True),
    ([{'execution_state': 'busy'}], ABSENT, TRACE, False),
    ([], ABSENT, TRACE, False),
    ([{'execution_state': 'idle'}], {**ABSENT, 'pipeline.pid': 200}, TRACE, False),
    ([{'execution_state': 'idle'}], {**ABSENT, 'status.json': 503}, TRACE, False),
    ([{'execution_state': 'idle'}], {'pipeline.pid': 404}, TRACE, False),
    ([{'execution_state': 'idle'}], ABSENT, 'Connection was lost.', False),
])
def test_retry_requires_known_preflight_failure_and_absent_pipeline(kernels, files, text, allowed):
    assert launch.startup_retry_allowed(text, kernels, files) is allowed


def response(code, body=None):
    return SimpleNamespace(status_code=code, json=lambda: body, raise_for_status=lambda: None)


def session():
    return SimpleNamespace(url='https://runtime.invalid', token='test', endpoint='test-endpoint')


def test_retry_stops_hung_client_then_observes_pipeline(tmp_path, monkeypatch):
    (tmp_path / 'training_cell.log').write_text(TRACE)
    old = Mock(pid=11)
    old.poll.return_value = None
    new = Mock(pid=12)
    popen = Mock(return_value=new)
    monkeypatch.setattr(launch.subprocess, 'Popen', popen)
    monkeypatch.setattr(launch.time, 'sleep', lambda _: None)
    replies = iter([response(404), response(404), response(404),
                    response(200, [{'execution_state': 'idle'}]),
                    response(200), response(200), response(200)])
    monkeypatch.setattr(launch.requests, 'get', lambda *a, **k: next(replies))
    result = launch.confirm_pipeline_start(session(), tmp_path, old, ['colab', 'exec'], 0)
    old.terminate.assert_called_once()
    old.wait.assert_called_once_with(timeout=5)
    popen.assert_called_once()
    assert result['status'] == 'pipeline_observed' and result['retries'] == 1
    assert (tmp_path / 'training_cell.pid').read_text() == '12'


def test_existing_pipeline_never_terminates_or_retries(tmp_path, monkeypatch):
    process = Mock(pid=11)
    monkeypatch.setattr(launch.requests, 'get', lambda *a, **k: response(200))
    popen = Mock()
    monkeypatch.setattr(launch.subprocess, 'Popen', popen)
    result = launch.confirm_pipeline_start(session(), tmp_path, process, ['unused'], 0)
    assert result['status'] == 'pipeline_observed'
    process.terminate.assert_not_called()
    popen.assert_not_called()


def test_exhausted_known_failure_stops_client_without_resubmission(tmp_path, monkeypatch):
    (tmp_path / 'training_cell.log').write_text(TRACE)
    process = Mock(pid=11)
    process.poll.return_value = None
    replies = iter([response(404), response(404), response(404), response(200, [{'execution_state': 'idle'}])])
    monkeypatch.setattr(launch.requests, 'get', lambda *a, **k: next(replies))
    popen = Mock()
    monkeypatch.setattr(launch.subprocess, 'Popen', popen)
    with pytest.raises(RuntimeError, match='重试耗尽'):
        launch.confirm_pipeline_start(session(), tmp_path, process, ['unused'], 0, max_retries=0)
    process.terminate.assert_called_once()
    popen.assert_not_called()
    assert json.loads((tmp_path / 'pipeline_startup.json').read_text())['status'] == 'preflight_failed_idle'
