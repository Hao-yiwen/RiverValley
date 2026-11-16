"""验证持久化备份发布的完整性和失败时的保留行为。"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts'))
from colab_train_pipeline import publish_snapshot, verify_snapshot, remaining_targets, restore_pretrain, write_json, sync_run_logs


def test_snapshot_hashes_retention_and_collision(tmp_path):
    source, parent = tmp_path / 'source', tmp_path / 'backup'
    source.mkdir()
    for step in (3, 250, 1000):
        (source / 'model.safetensors').write_bytes(bytes([step % 255]) * 80)
        (source / 'training_state.pt').write_bytes(b'optimizer state')
        saved = publish_snapshot(source, parent, step, keep=2)
        assert verify_snapshot(saved)['step'] == step
    assert not (parent / 'step_000003').exists()
    assert (parent / 'step_000250').exists()
    assert json.loads((parent / 'LATEST.json').read_text())['step'] == 1000
    (source / 'model.safetensors').write_bytes(b'different')
    with pytest.raises(ValueError, match='内容不同'):
        publish_snapshot(source, parent, 1000)
    (saved / 'training_state.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='摘要不匹配'):
        verify_snapshot(saved)


def test_failed_backup_does_not_publish_pointer_or_remove_previous(tmp_path, monkeypatch):
    source, parent = tmp_path / 'source', tmp_path / 'backup'
    source.mkdir()
    (source / 'weights').write_bytes(b'good')
    previous = publish_snapshot(source, parent, 3)
    monkeypatch.setattr('colab_train_pipeline.verify_snapshot', lambda path: (_ for _ in ()).throw(ValueError('模拟摘要失败')))
    with pytest.raises(ValueError, match='模拟摘要失败'):
        publish_snapshot(source, parent, 250)
    assert previous.exists()
    assert not (parent / 'step_000250').exists()
    assert json.loads((parent / 'LATEST.json').read_text())['step'] == 3


def test_resume_targets_preserve_global_plan_and_skip_completed_steps():
    targets = remaining_targets(1000, 500)
    assert targets[:2] == [1500, 2000]
    assert targets[-2:] == [7500, 7629]
    assert remaining_targets(7629, 500) == []
    assert remaining_targets(0)[:3] == [3, 250, 1000]
    with pytest.raises(ValueError):
        remaining_targets(1000, 0)


def test_resume_requires_matching_cursor_and_restores_best(tmp_path, monkeypatch):
    source, drive, run = tmp_path/'source', tmp_path/'drive', tmp_path/'run'
    source.mkdir()
    (source/'model.safetensors').write_bytes(b'weights')
    (source/'training_state.pt').write_bytes(b'complete optimizer state')
    complete = publish_snapshot(source, drive/'pretrain', 1000)
    publish_snapshot(source, drive/'pretrain_best', 1000)
    monkeypatch.setattr('colab_train_pipeline.inspect_training_state', lambda path: 999)
    with pytest.raises(ValueError, match='清单步数'):
        restore_pretrain(complete, run, drive)
    monkeypatch.setattr('colab_train_pipeline.inspect_training_state', lambda path: 1000)
    assert restore_pretrain(complete, run, drive) == 1000
    assert (run/'checkpoints/pretrain/best/model.safetensors').read_bytes() == b'weights'
    assert not (run/'checkpoints/pretrain/best/BACKUP_COMPLETE.json').exists()


def test_resume_rejects_best_from_future_or_unsafe_pointer(tmp_path, monkeypatch):
    source, drive = tmp_path/'source', tmp_path/'drive'
    source.mkdir()
    (source/'model.safetensors').write_bytes(b'weights')
    complete = publish_snapshot(source, drive/'pretrain', 1000)
    publish_snapshot(source, drive/'pretrain_best', 1200)
    monkeypatch.setattr('colab_train_pipeline.inspect_training_state', lambda path: 1000)
    with pytest.raises(ValueError, match='恢复游标之后'):
        restore_pretrain(complete, tmp_path/'run', drive)
    write_json(drive/'pretrain_best/LATEST.json', {'path':'../unexpected','step':1000})
    with pytest.raises(ValueError, match='指针非法'):
        restore_pretrain(complete, tmp_path/'run', drive)


def test_log_sync_ignores_jupyter_temporary_files(tmp_path):
    run, drive = tmp_path/'run', tmp_path/'drive'
    run.mkdir()
    drive.mkdir()
    (run/'.~budget_usage.json').write_text('{"partial":', encoding='utf-8')
    (run/'budget_usage.json').write_text('{"used_cu": 61.86}', encoding='utf-8')
    metrics = run/'checkpoints/pretrain/metrics.jsonl'
    metrics.parent.mkdir(parents=True)
    metrics.write_text('{"step": 5000}\n', encoding='utf-8')
    sync_run_logs(run, drive)
    assert not (drive/'.~budget_usage.json').exists()
    assert (drive/'budget_usage.json').read_text() == '{"used_cu": 61.86}'
    assert (drive/'pretrain_metrics.jsonl').read_text() == metrics.read_text()


@pytest.mark.parametrize('failure', ['source_removed', 'destination_write'])
def test_log_sync_failure_preserves_previous_and_continues(tmp_path, monkeypatch, caplog, failure):
    import shutil
    run, drive = tmp_path/'run', tmp_path/'drive'
    run.mkdir()
    drive.mkdir()
    (run/'budget_usage.json').write_text('new', encoding='utf-8')
    (drive/'budget_usage.json').write_text('previous', encoding='utf-8')
    (run/'pipeline.log').write_text('still training', encoding='utf-8')
    original = shutil.copy2

    def failing_copy(source, destination):
        if source.name == 'budget_usage.json':
            if failure == 'source_removed':
                source.unlink()
                raise FileNotFoundError(str(source))
            destination.write_text('partial', encoding='utf-8')
            raise OSError('模拟 Drive 写入失败')
        return original(source, destination)

    monkeypatch.setattr('colab_train_pipeline.shutil.copy2', failing_copy)
    sync_run_logs(run, drive)
    assert (drive/'budget_usage.json').read_text() == 'previous'
    assert (drive/'pipeline.log').read_text() == 'still training'
    assert not list(drive.glob('.*.sync-*.tmp'))
    assert '继续训练' in caplog.text
