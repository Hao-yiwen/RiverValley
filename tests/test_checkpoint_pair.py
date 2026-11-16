"""模拟两份备份之间断线，恢复不能混用新完整状态与旧最佳权重。"""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts'))
import train_version
from train_version import committed_resume_pair, commit_resume_pair


def pair():
    return {'stage': 'sft', 'status': 'pair_verified',
            'full': {'step': 80, 'drive_folder_id': 'full80', 'files': {'training_state.pt': {}}},
            'best': {'step': 40, 'drive_folder_id': 'best40', 'files': {'model.safetensors': {}}}}


class Store:
    root_id = 'run'

    def __init__(self, marker=True):
        self.marker = marker
        self.verified = []
        self.published = []

    def child(self, parent, name):
        return 'paired' if self.marker and name == 'sft_RESUME.json' else None

    def verify(self, value):
        self.verified.append(value)

    def json_file(self, value, parent, name, workspace):
        self.published.append(value)


def test_partial_new_backup_does_not_advance_committed_cursor(monkeypatch):
    store, committed = Store(), pair()
    monkeypatch.setattr(train_version, 'read_remote_json', lambda *_: committed)
    monkeypatch.setattr(train_version, 'latest_remote', lambda *_: {'step': 160, 'drive_folder_id': 'uncommitted160'})
    restored = committed_resume_pair(store, 'sft', True)
    assert restored['full']['step'] == 80 and restored['best']['step'] == 40
    assert store.verified == [committed['full'], committed['best']]


def test_first_half_backup_cannot_be_mixed_with_missing_best(monkeypatch):
    monkeypatch.setattr(train_version, 'latest_remote', lambda *_: {'step': 80})
    with pytest.raises(ValueError, match='未成对提交'):
        committed_resume_pair(Store(False), 'sft', True)


def test_failed_best_verification_preserves_previous_commit():
    store, proposed = Store(), pair()
    def verify(value):
        if value is proposed['best']:
            raise ValueError('服务器摘要不符')
    store.verify = verify
    with pytest.raises(ValueError):
        commit_resume_pair(store, 'sft', proposed['full'], proposed['best'], '.')
    assert not store.published


@pytest.mark.parametrize('case', ['wrong_stage', 'future_best', 'missing_optimizer'])
def test_bad_pair_fails_before_restore(monkeypatch, case):
    invalid = copy.deepcopy(pair())
    if case == 'wrong_stage': invalid['stage'] = 'pretrain'
    if case == 'future_best': invalid['best']['step'] = 100
    if case == 'missing_optimizer': invalid['full']['files'] = {}
    monkeypatch.setattr(train_version, 'read_remote_json', lambda *_: invalid)
    with pytest.raises(ValueError): committed_resume_pair(Store(), 'sft', True)


def test_successful_pair_is_published_after_both_checks(tmp_path):
    store, proposed = Store(), pair()
    result = commit_resume_pair(store, 'sft', proposed['full'], proposed['best'], tmp_path)
    assert store.verified == [proposed['full'], proposed['best']]
    assert store.published == [result]
