"""核验修复任务无固定查询位置捷径，以及低余额启动仍保留保存余量。"""
from collections import Counter
import json
from pathlib import Path
import random
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'datasets'), str(ROOT/'src/scripts')]
from prepare_behavior_repair import make_row, TASKS
from launch_version import check_launch_balance


def test_answers_and_query_positions_are_balanced():
    rng = random.Random(17)
    positions, updated = Counter(), Counter()
    for language in ('zh', 'en'):
        for task in TASKS:
            for i in range(300):
                row = make_row(task, language, i, 'train', rng)
                proof = row['proof']; answer = row['conversations'][-1]['content']
                if task in ('extraction', 'multi_turn'):
                    assert answer == str(proof['facts'][proof['query']])
                    positions[proof['position']] += 1
                    if task == 'multi_turn':
                        updated[proof['queried_changed']] += 1
                elif task == 'json':
                    assert json.loads(answer) == proof['object']
                    assert len(proof['object']) == 2
                elif task == 'sort':
                    numbers = list(map(int, answer.split(', ')))
                    assert set(numbers) == set(proof['values'])
                    assert all((a > b if proof['descending'] else a < b) for a, b in zip(numbers, numbers[1:]))
                elif task == 'arithmetic':
                    a, b, result = proof['a'], proof['b'], int(answer)
                    op = proof['operator']
                    assert (result-a == b if op == '+' else result+b == a if op == '-' else result == a*b if op == '×' else result*b == a)
                else:
                    assert answer == proof['literal']
                assert '<|im_start|>' not in answer
    assert set(positions) == {0, 1, 2} and min(positions.values()) > 300
    assert 150 < updated[True] < 250 and updated[False] > 350


def test_split_families_are_distinct():
    for task in TASKS:
        rows = [make_row(task, 'en', 0, split, random.Random(12)) for split in ('train', 'dev', 'holdout')]
        assert [r['family'] for r in rows] == [0, 4, 5]
        assert len({r['id'] for r in rows}) == 3
        assert len({r['conversations'][-1]['content'] for r in rows}) == 1


def test_small_probe_balance_requires_complete_estimate_and_reserve():
    plan = {'budget_estimate': {'setup_training_evaluation_cu': 10, 'save_release_reserve_cu': 4, 'basis': '既往同结构实验时间'}}
    check_launch_balance(18.57, plan)
    for balance in (14, 10, float('nan')):
        with pytest.raises(RuntimeError):
            check_launch_balance(balance, plan)
    with pytest.raises(RuntimeError):
        check_launch_balance(18.57, {})
    for change in ({'save_release_reserve_cu': 2}, {'setup_training_evaluation_cu': float('nan')}, {'basis': ''}):
        with pytest.raises(ValueError):
            check_launch_balance(18.57, {'budget_estimate': dict(plan['budget_estimate'], **change)})
