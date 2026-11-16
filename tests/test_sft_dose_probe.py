"""检查监督量诊断的对照不变项、完整来源约束及训练步数可达性。"""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts'))
from prepare_sft_dose_probe import build_plan


def baseline():
    return {'purpose': 'milestone_diagnostic', 'input_sha256': {'data': 'fixed'},
            'origin': {'stage': 'continued_pretrain', 'step': 7629, 'drive_folder_id': 'cpt'},
            'stages': [{'id': 'probe_7629', 'kind': 'sft', 'target_steps': 160,
                        'init_folder_id': 'cpt', 'config': {
                            'schedule_steps': 160, 'epochs': 2, 'batch_size': 4,
                            'accumulation_steps': 16, 'learning_rate': 2e-5,
                            'min_lr': 2e-6, 'random_seed': 1337, 'eval_interval': 80}}]}


def test_dose_changes_only_duration_and_keeps_independent_origin():
    source = baseline()
    frozen = copy.deepcopy(source)
    plan = build_plan(source, 10000)
    stage = plan['stages'][0]
    changed = {key for key, value in stage['config'].items()
               if value != source['stages'][0]['config'][key]}
    assert changed == {'schedule_steps', 'epochs'}
    assert stage['config']['epochs'] * (10000 // 64) >= stage['target_steps'] == 640
    assert stage['milestone_steps'] == [320, 640]
    assert stage['init_folder_id'] == 'cpt'
    assert plan['input_sha256'] == source['input_sha256']
    assert source == frozen


@pytest.mark.parametrize('change', ['kind', 'target', 'schedule', 'origin', 'folder', 'rows'])
def test_changed_or_wrong_baseline_is_rejected(change):
    source = baseline()
    stage = source['stages'][0]
    if change == 'kind': stage['kind'] = 'pretrain'
    if change == 'target': stage['target_steps'] = 320
    if change == 'schedule': stage['config']['schedule_steps'] = 320
    if change == 'origin': source['origin']['step'] = 3814
    if change == 'folder': stage['init_folder_id'] = 'diagnostic-sft'
    with pytest.raises(ValueError):
        build_plan(source, 9999 if change == 'rows' else 10000)
