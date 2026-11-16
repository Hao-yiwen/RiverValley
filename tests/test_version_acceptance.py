"""独立留出必须先锁候选，不能换权重、丢题或静默覆盖不完整回答。"""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts'))
from prepare_version_acceptance import freeze_identity, validate_answers, validate_selection
from colab_train_pipeline import sha256_file


def test_lock_reuses_same_identity_and_rejects_any_change(tmp_path):
    path = tmp_path / 'lock.json'
    identity = {'model_sha256': 'a', 'selection_sha256': 's', 'device': 'mps'}
    first = freeze_identity(path, identity)
    before = path.read_bytes()
    assert freeze_identity(path, dict(identity)) == first
    assert path.read_bytes() == before
    for key in identity:
        with pytest.raises(ValueError, match='禁止更换'):
            freeze_identity(path, dict(identity, **{key: 'changed'}))
        assert path.read_bytes() == before


@pytest.mark.parametrize('change', ['basis', 'review_hash', 'reason', 'unreviewed'])
def test_selection_requires_actual_development_report(tmp_path, change):
    report = tmp_path / 'review.json'
    report.write_text('{"complete":true}')
    selection = {'basis': 'development', 'reason': '一轮开发回答更好',
                 'development_review_sha256': sha256_file(report)}
    validate_selection(selection, report)
    if change == 'basis': selection['basis'] = 'holdout'
    if change == 'review_hash': report.write_text('{"changed":true}')
    if change == 'reason': selection['reason'] = ''
    if change == 'unreviewed':
        report.write_text('{"complete":false}')
        selection['development_review_sha256'] = sha256_file(report)
    with pytest.raises(ValueError, match='开发复核'):
        validate_selection(selection, report)


@pytest.mark.parametrize('change', ['none', 'missing', 'duplicate', 'model', 'question', 'hash'])
def test_answers_cannot_reuse_other_model_or_partial_questions(tmp_path, change):
    questions = tmp_path / 'questions.jsonl'
    items = [{'id': str(i), 'prompt': f'问题{i}', 'reference': '1', 'scoring': 'exact', 'language': 'zh'} for i in range(2)]
    questions.write_text(''.join(json.dumps(x) + '\n' for x in items))
    digest = sha256_file(questions)
    identity = {'model_directory': '/locked/model', 'question_set_sha256': digest}
    rows = [dict(x, answer='1', model='/locked/model', question_sha256=digest) for x in items]
    if change == 'missing': rows.pop()
    if change == 'duplicate': rows[1] = rows[0]
    if change == 'model': rows[0]['model'] = '/other/model'
    if change == 'question': rows[0]['prompt'] = '新问题'
    if change == 'hash': rows[0]['question_sha256'] = 'changed'
    output = tmp_path / 'answers.jsonl'
    output.write_text(''.join(json.dumps(x) + '\n' for x in rows))
    before = output.read_bytes()
    if change == 'none':
        assert validate_answers(output, questions, identity) == rows
    else:
        with pytest.raises(ValueError): validate_answers(output, questions, identity)
    assert output.read_bytes() == before
