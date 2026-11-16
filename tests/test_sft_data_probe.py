"""验证数据配方对照保持训练数学，并拒绝错误来源、未知数据及tokenizer变化。"""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts'))
from prepare_sft_data_probe import build_plan, REMOTE


def inputs():
    baseline = {'purpose': 'milestone_diagnostic', 'origin': {
        'stage': 'continued_pretrain', 'step': 7629, 'drive_folder_id': 'cpt'},
        'input_sha256': {REMOTE + 'tokenizer.json': 'tokenizer'},
        'stages': [{'id': 'probe_7629', 'kind': 'sft', 'target_steps': 160,
                    'init_folder_id': 'cpt', 'config': {'schedule_steps': 160,
                        'epochs': 2, 'batch_size': 4, 'accumulation_steps': 16,
                        'eval_iters': 100, 'learning_rate': 2e-5, 'random_seed': 1337}}]}
    manifest = {'hashes': {'sft_train.jsonl': 'data'}, 'tokenizer_sha256': 'tokenizer',
                'train': {'rows': 27667, 'answer_tokens': 672598}, 'val': {'rows': 813}}
    approval = {'status': 'approved_for_formal_sft', 'data_hashes': manifest['hashes'].copy()}
    return baseline, manifest, approval


def test_matched_data_probe_preserves_config_and_original():
    baseline, manifest, approval = inputs()
    old = copy.deepcopy(baseline)
    plan = build_plan(baseline, manifest, approval)
    assert baseline == old
    assert plan['stages'][0]['config'] == old['stages'][0]['config']
    assert plan['origin'] == old['origin']
    assert plan['stages'][0]['milestone_steps'] == [160]
    assert plan['stages'][0]['evaluate_milestones']
    assert plan['data_accounting']['sample_presentations'] == 10240
    assert plan['purpose'] == 'sft_data_diagnostic'


@pytest.mark.parametrize('change', ['stage', 'steps', 'schedule', 'origin', 'folder',
                                  'approval', 'hash', 'tokenizer', 'rows', 'batch'])
def test_unmatched_data_probe_is_rejected(change):
    baseline, manifest, approval = inputs()
    stage = baseline['stages'][0]
    if change == 'stage': stage['kind'] = 'pretrain'
    if change == 'steps': stage['target_steps'] = 640
    if change == 'schedule': stage['config']['schedule_steps'] = 640
    if change == 'origin': baseline['origin']['step'] = 3814
    if change == 'folder': stage['init_folder_id'] = 'sft-best'
    if change == 'approval': approval['status'] = 'unreviewed'
    if change == 'hash': approval['data_hashes']['sft_train.jsonl'] = 'changed'
    if change == 'tokenizer': manifest['tokenizer_sha256'] = 'changed'
    if change == 'rows': manifest['train']['rows'] = 10000
    if change == 'batch': stage['config']['batch_size'] = 8
    with pytest.raises(ValueError):
        build_plan(baseline, manifest, approval)
