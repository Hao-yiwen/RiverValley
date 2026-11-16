"""验证服务器校验门禁、失败保留以及 SFT 阶段恢复。"""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts'))
from drive_checkpoint_store import DriveStore, check_remote, fingerprints
from colab_train_pipeline import publish_snapshot, restore_stage


@pytest.mark.parametrize('reason,retry', [('rateLimitExceeded', True), ('userRateLimitExceeded', True),
                                       ('insufficientFilePermissions', False), ('storageQuotaExceeded', False)])
def test_drive_403_distinguishes_transient_quota_from_real_permissions(monkeypatch, reason, retry):
    from types import SimpleNamespace
    from drive_checkpoint_store import DriveRequestError
    error = SimpleNamespace(status_code=403, json=lambda: {'error': {'errors': [{'reason': reason}]}})
    success = SimpleNamespace(status_code=200)
    responses = [error, success]
    store = object.__new__(DriveStore)
    store.http = SimpleNamespace(request=lambda *a, **kw: responses.pop(0))
    monkeypatch.setattr('drive_checkpoint_store.time.sleep', lambda seconds: None)
    if retry:
        assert store.request('GET', 'https://www.googleapis.com/drive/v3/files') is success
    else:
        with pytest.raises(DriveRequestError) as caught:
            store.request('GET', 'https://www.googleapis.com/drive/v3/files')
        assert caught.value.status == 403 and caught.value.reasons == [reason]
        assert len(responses) == 1


def test_server_checksum_missing_or_corrupt_blocks_release(tmp_path):
    path = tmp_path / 'state'
    path.write_bytes(b'optimizer and rng')
    expected = fingerprints(path)
    check_remote({'size': str(expected['bytes']), 'sha256Checksum': expected['sha256']}, expected)
    check_remote({'size': str(expected['bytes']), 'md5Checksum': expected['md5']}, expected)
    for remote in ({'size': 0}, {'size': expected['bytes']},
                   {'size': expected['bytes'], 'sha256Checksum': 'wrong'}):
        with pytest.raises(ValueError):
            check_remote(remote, expected)


def test_remote_failure_cannot_advance_local_pointer(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'model.safetensors').write_bytes(b'weights')
    parent = tmp_path / 'pretrain'
    first = publish_snapshot(source, parent, 6500)
    class FailedStore:
        def publish(self, *args):
            raise ValueError('服务器训练状态缺失')
    monkeypatch.setattr('colab_train_pipeline.configured_store', lambda: FailedStore())
    with pytest.raises(ValueError, match='服务器训练状态缺失'):
        publish_snapshot(source, parent, 7000)
    assert first.exists()
    assert json.loads((parent / 'LATEST.json').read_text())['step'] == 6500


def test_remote_manifest_and_pointer_follow_all_verified_objects(tmp_path):
    (tmp_path / 'model.safetensors').write_bytes(b'weights')
    (tmp_path / 'training_state.pt').write_bytes(b'state')
    store = object.__new__(DriveStore)
    store.root_id = 'root'
    events = []
    store.folder = lambda parent, name: parent + '/' + name
    store.upload = lambda path, parent: dict(fingerprints(path), id=path.name)
    store.json_file = lambda value, parent, name, workspace: events.append(name)
    store.verify = lambda manifest: events.append('server_verified')
    manifest = store.publish(tmp_path, 'sft', 100)
    assert set(manifest['files']) == {'model.safetensors', 'training_state.pt'}
    assert events == ['BACKUP_COMPLETE.json', 'server_verified', 'LATEST.json']


def test_sft_restores_its_own_best_and_does_not_touch_pretrain(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'model.safetensors').write_bytes(b'sft weights')
    (source / 'training_state.pt').write_bytes(b'sft state')
    drive, run = tmp_path / 'drive', tmp_path / 'run'
    full = publish_snapshot(source, drive / 'sft', 200)
    publish_snapshot(source, drive / 'sft_best', 150)
    monkeypatch.setattr('colab_train_pipeline.inspect_training_state', lambda path: 200)
    assert restore_stage(full, run, drive, 'sft') == 200
    assert (run / 'checkpoints/sft/best/model.safetensors').read_bytes() == b'sft weights'
    assert not (run / 'checkpoints/pretrain').exists()


def test_local_delivery_rejects_corrupt_download(tmp_path):
    from types import SimpleNamespace
    from deliver_training_weights import download
    source = tmp_path / 'source'
    source.write_bytes(b'correct weights')
    expected = fingerprints(source)
    class Response:
        status_code = 200
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def iter_content(self, size):
            yield b'corrupt weights'
    store = SimpleNamespace(http=SimpleNamespace(get=lambda *args, **kwargs: Response()))
    with pytest.raises(ValueError, match='摘要不符'):
        download(store, 'model', tmp_path / 'model.safetensors', expected)
    assert not (tmp_path / 'model.safetensors').exists()


def upload_fixture(tmp_path, monkeypatch, replies, object_states):
    """构造断开后的真实协议响应，验证重建不会跳过摘要或制造新对象。"""
    from types import SimpleNamespace
    from drive_checkpoint_store import DriveRequestError
    path = tmp_path / 'model.safetensors'
    path.write_bytes(b'weights')
    expected = fingerprints(path)
    correct = {'size': expected['bytes'], 'sha256Checksum': expected['sha256']}
    requests_seen = []
    store = object.__new__(DriveStore)
    store.child = lambda *args: None
    def metadata(file_id):
        assert file_id == 'reserved'
        value = object_states.pop(0) if len(object_states) > 1 else object_states[0]
        if value == 'missing':
            raise DriveRequestError(404, ['notFound'])
        return correct if value == 'valid' else {'size': 1, 'sha256Checksum': 'bad'}
    store.metadata = metadata
    def request(method, url, **kwargs):
        requests_seen.append((method, url, kwargs))
        if url.endswith('generateIds'):
            return SimpleNamespace(json=lambda: {'ids': ['reserved']})
        if method == 'PUT':
            raise DriveRequestError(410, ['internalError'])
        return SimpleNamespace(headers={'Location': 'https://upload/session'})
    store.request = request
    def put(*args, **kwargs):
        code = replies.pop(0)
        return SimpleNamespace(status_code=code, headers={}, json=lambda: {'id': 'reserved', 'error': {'errors': [{'reason': 'internalError'}]}})
    store.http = SimpleNamespace(put=put)
    monkeypatch.setattr('drive_checkpoint_store.time.sleep', lambda seconds: None)
    return store, path, requests_seen


def test_expired_final_response_uses_completed_object_without_reupload(tmp_path, monkeypatch):
    store, path, seen = upload_fixture(tmp_path, monkeypatch, [503], ['valid'])
    assert store.upload(path, 'parent')['id'] == 'reserved'
    assert len([r for r in seen if r[0] == 'POST']) == 1


def test_expired_missing_object_restarts_with_same_reserved_id(tmp_path, monkeypatch):
    store, path, seen = upload_fixture(tmp_path, monkeypatch, [410, 201], ['missing', 'missing', 'valid'])
    assert store.upload(path, 'parent')['id'] == 'reserved'
    creations = [r for r in seen if r[0] == 'POST']
    assert len(creations) == 2
    assert all(r[2]['json']['id'] == 'reserved' for r in creations)


def test_expired_corrupt_object_is_reuploaded_not_accepted(tmp_path, monkeypatch):
    store, path, seen = upload_fixture(tmp_path, monkeypatch, [410, 201], ['bad', 'bad', 'valid'])
    store.upload(path, 'parent')
    assert len([r for r in seen if r[0] == 'PATCH' and r[1].endswith('/reserved')]) == 1


def test_expired_upload_has_bounded_restarts(tmp_path, monkeypatch):
    store, path, seen = upload_fixture(tmp_path, monkeypatch, [410, 410, 410], ['missing'])
    with pytest.raises(RuntimeError, match='三次仍失败'):
        store.upload(path, 'parent')
    assert len([r for r in seen if r[0] == 'POST']) == 3


def test_upload_permission_denial_does_not_create_more_sessions(tmp_path, monkeypatch):
    from drive_checkpoint_store import DriveRequestError
    store, path, seen = upload_fixture(tmp_path, monkeypatch, [403], ['missing'])
    with pytest.raises(DriveRequestError) as error:
        store.upload(path, 'parent')
    assert error.value.status == 403
    assert len([r for r in seen if r[0] == 'POST']) == 1
