"""覆盖低余额小实验的真实保存余量及旧计划兼容性。"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts'))
from launch_version import balance_policy, check_launch_balance, recalibrate_limits


def test_explicit_small_plan_and_legacy_policy():
    plan = {'budget_estimate': {'setup_training_evaluation_cu': 5.5,
            'save_release_reserve_cu': 4, 'basis': '缩短到160步，问答仅本机运行'},
            'balance_policy': {'soft_balance_cu': 4, 'hard_balance_cu': 2}}
    check_launch_balance(10.63, plan)
    assert balance_policy(plan) == {'soft_balance_cu': 4, 'hard_balance_cu': 2}
    assert balance_policy({}) == {'soft_balance_cu': 10, 'hard_balance_cu': 2}
    with pytest.raises(RuntimeError):
        check_launch_balance(9.5, plan)


@pytest.mark.parametrize('soft,hard', [(3, 2), (4, 1), (4, 4), (float('nan'), 2), (True, 2)])
def test_refuse_unsafe_reserves(soft, hard):
    with pytest.raises(ValueError):
        balance_policy({'balance_policy': {'soft_balance_cu': soft, 'hard_balance_cu': hard}})


def test_reserve_cannot_undercut_estimate():
    with pytest.raises(ValueError):
        balance_policy({'balance_policy': {'soft_balance_cu': 4, 'hard_balance_cu': 2},
                        'budget_estimate': {'save_release_reserve_cu': 5}})


def test_real_rate_shortens_release_without_resetting_runtime():
    limits={'soft_deadline':3700,'hard_deadline':5500,'soft_balance_cu':4,'hard_balance_cu':2}
    revised=recalibrate_limits(limits,10.47,8.88,now=1000)
    assert revised['soft_deadline']<=limits['soft_deadline']
    assert revised['hard_deadline']==pytest.approx(1000+(10.47-2)/8.88*3600)
    # 后续余额增加或低费率不延长此前已经部署的独立截止。
    assert recalibrate_limits(revised,100,1,now=1100)['hard_deadline']==revised['hard_deadline']
