"""交付门禁必须绑定最终训练来源及实际问答权重，不能使用小实验冒充新版。"""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts'))
import deliver_version
from deliver_version import validate_delivery, selected_checkpoint


def completed():
    plan = {'purpose': 'final_sft_candidate', 'stages': [{'id': 'sft', 'kind': 'sft', 'target_steps': 980,
                                                       'init_folder_id': 'new-pretrain'}]}
    pretrain = {'status': 'plan_completed', 'checkpoints': {'pretrain': {'drive_folder_id': 'new-pretrain', 'step': 15258}}}
    receipt = {'status': 'plan_completed', 'stages': {'sft': 980}, 'checkpoints': {
        'sft': {'step': 980, 'files': {'training_state.pt': {}}},
        'sft_best': {'step': 490, 'files': {'model.safetensors': {'sha256': 'reviewed-model'}}}}}
    review = {'complete': True, 'model_sha256': 'reviewed-model', 'question_set_sha256': 'fixed-questions',
              'quality_status': '效果仍有限，真实问答已保存'}
    return plan, receipt, pretrain, review


def test_delivery_can_report_limitations_without_claiming_quality_passed():
    args = completed()
    stage, best, origin = validate_delivery(*args, 'sft')
    assert best['step'] == 490 and stage['target_steps'] == 980 and origin['step'] == 15258


@pytest.mark.parametrize('case', ['pilot', 'unfinished', 'missing_state', 'wrong_origin', 'wrong_model', 'unreviewed'])
def test_delivery_rejects_incomplete_or_mismatched_evidence(case):
    plan, receipt, pretrain, review = copy.deepcopy(completed())
    if case == 'pilot': plan['purpose'] = 'pilot'
    if case == 'unfinished': receipt['stages']['sft'] = 490
    if case == 'missing_state': receipt['checkpoints']['sft']['files'] = {}
    if case == 'wrong_origin': plan['stages'][0]['init_folder_id'] = 'old-pretrain'
    if case == 'wrong_model': review['model_sha256'] = 'other-model'
    if case == 'unreviewed': review['complete'] = False
    with pytest.raises(ValueError): validate_delivery(plan, receipt, pretrain, review, 'sft')


def milestone_store(monkeypatch):
    """模拟本阶段的完整里程碑与服务器摘要门禁。"""
    plan, receipt, pretrain, review = completed()
    plan['stages'][0]['milestone_steps'] = [490, 980]
    manifest = {'step': 980, 'drive_folder_id': 'epoch-two',
                'files': {'training_state.pt': {}, 'model.safetensors': {'sha256': 'epoch-two-model'}}}
    decision = {'basis': 'development', 'reason': '开发集基础题改善且验证没有退化',
                'development_review_sha256': 'review-digest', 'step': 980,
                'drive_folder_id': 'epoch-two', 'model_sha256': 'epoch-two-model'}

    class Store:
        root_id = 'this-run'
        checked = None

        def child(self, parent, name):
            return {('this-run', 'sft'): 'this-sft',
                    ('this-sft', 'step_000980'): 'epoch-two',
                    ('epoch-two', 'BACKUP_COMPLETE.json'): 'marker'}.get((parent, name))

        def verify(self, value):
            self.checked = value

    monkeypatch.setattr(deliver_version, 'read_remote_json', lambda store, marker: manifest)
    return Store(), plan, receipt, pretrain, review, decision, manifest


def test_delivery_can_select_development_epoch_different_from_validation_best(monkeypatch):
    store, plan, receipt, pretrain, review, decision, manifest = milestone_store(monkeypatch)
    candidate = selected_checkpoint(store, plan, receipt, 'sft', decision)
    review['model_sha256'] = 'epoch-two-model'
    _, chosen, _ = validate_delivery(plan, receipt, pretrain, review, 'sft', candidate)
    assert chosen['step'] == 980 and store.checked is manifest
    assert receipt['checkpoints']['sft_best']['step'] == 490


@pytest.mark.parametrize('case', ['other_folder', 'other_weight', 'non_milestone', 'holdout_selection', 'no_state', 'bad_manifest_step'])
def test_delivery_rejects_unbound_milestone_selection(monkeypatch, case):
    store, plan, receipt, _, _, decision, manifest = milestone_store(monkeypatch)
    if case == 'other_folder': decision['drive_folder_id'] = 'another-run'
    if case == 'other_weight': decision['model_sha256'] = 'unreviewed'
    if case == 'non_milestone': decision['step'] = 700
    if case == 'holdout_selection': decision['basis'] = 'holdout'
    if case == 'no_state': manifest['files'].pop('training_state.pt')
    if case == 'bad_manifest_step': manifest['step'] = 979
    with pytest.raises(ValueError): selected_checkpoint(store, plan, receipt, 'sft', decision)
