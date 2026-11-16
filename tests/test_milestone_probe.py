"""验证短SFT对照冻结、里程碑停止窗口及完整来源门禁。"""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/scripts'))
from prepare_milestone_probe import build_plan
from milestone_handoff import should_request_stop, validate_origin, clean_index, training_updates


def plans():
    pilot = {'stages': [{'id': 'clean_from_pretrain', 'kind': 'sft', 'target_steps': 160,
                        'config': {'schedule_steps': 160, 'random_seed': 1337}, 'init_folder_id': 'old'}],
             'input_sha256': {'/content/river-valley-run/tokenizer.json': 'tokenizer'}}
    cpt = {'stages': [{'id': 'continued_pretrain', 'kind': 'pretrain', 'milestone_steps': [3814]}]}
    return pilot, cpt


def test_plan_preserves_baseline_and_keeps_separate_stage():
    pilot, cpt = plans(); before = copy.deepcopy(pilot)
    probe = build_plan(pilot, cpt, 3814, 'milestone-folder')
    assert probe['stages'][0]['config'] == pilot['stages'][0]['config']
    assert probe['stages'][0]['init_folder_id'] == 'milestone-folder'
    assert probe['purpose'] == 'milestone_diagnostic'
    assert probe['atomic_resume_pairs'] is True
    probe['stages'][0]['config']['random_seed'] = 2
    assert pilot == before


def test_unplanned_milestone_rejected():
    with pytest.raises(ValueError):
        build_plan(*plans(), 4000, 'folder')


@pytest.mark.parametrize('status,expected', [
    ({'stage': 'continued_pretrain', 'status': 'training', 'completed_steps': 3814}, False),
    ({'stage': 'continued_pretrain', 'status': 'backing_up', 'completed_steps': 3500}, False),
    ({'stage': 'continued_pretrain', 'status': 'backing_up', 'completed_steps': 3814}, True),
    ({'stage': 'continued_pretrain', 'status': 'backing_up', 'completed_steps': 4000}, True),
    ({'stage': 'probe_3814', 'status': 'backing_up', 'completed_steps': 4000}, False),
])
def test_stop_only_after_training_in_backup_window(status, expected):
    assert should_request_stop(status, 3814) is expected


@pytest.mark.parametrize('change', ['step', 'folder', 'tiny_state', 'missing_model', 'tokenizer'])
def test_origin_rejects_wrong_or_incomplete_checkpoint(change):
    plan = build_plan(*plans(), 3814, 'folder')
    manifest = {'step': 3814, 'drive_folder_id': 'folder', 'files': {
        'training_state.pt': {'bytes': 8_000_000_000}, 'model.safetensors': {},
        'tokenizer.json': {'sha256': 'tokenizer'}}}
    validate_origin(manifest, plan)
    if change == 'step': manifest['step'] = 3500
    if change == 'folder': manifest['drive_folder_id'] = 'other'
    if change == 'tiny_state': manifest['files']['training_state.pt']['bytes'] = 9501
    if change == 'missing_model': del manifest['files']['model.safetensors']
    if change == 'tokenizer': manifest['files']['tokenizer.json']['sha256'] = 'other'
    with pytest.raises(ValueError): validate_origin(manifest, plan)


def test_new_phase_does_not_inherit_verified_training_progress():
    index = {'session': 'cpt', 'evidence_dir': 'cpt-path', 'verified_step': 3814,
             'last_hourly_check': 'old', 'initial_balance_cu': 258.45, 'server_verified_durable_step': 3814}
    result = clean_index(index, ROOT / 'evals/results/probe', 'probe_3814')
    assert 'verified_step' not in result and 'last_hourly_check' not in result
    assert 'server_verified_durable_step' not in result
    assert result['initial_balance_cu'] == 258.45
    assert result['cpt_evidence_dir'] == 'cpt-path'


def test_real_sft_log_without_event_confirms_updates():
    rows = [{'event': 'eval', 'step': 0, 'test': 3.93},
            {'step': 10, 'lr': 1.995e-5, 'loss': 1.9798, 'tokens': 2751, 'grad_norm': 2.71},
            {'event': 'eval', 'step': 80, 'train': 1.39, 'test': 1.67},
            {'event': 'train', 'step': 100, 'loss': 2.8},
            {'step': 160, 'completed': True},
            {'step': 0, 'loss': 1.0, 'lr': 2e-5, 'tokens': 2000}]
    assert [row['step'] for row in training_updates(rows)] == [10, 100]
