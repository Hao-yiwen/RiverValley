"""开发分数不足时拒绝把翻译实验当作改善版本交付。"""
import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/scripts'))
from deliver_translation_probe import eligible, verify_review


def test_translation_improvement_cannot_hide_old_task_regression():
    plan={'selection_rule':{'minimum_translation_point_gain':12,'maximum_old_dev_point_drop':5}}
    baseline={'groups':{'translation':{'rows':48,'points':10},'old':{'rows':100,'points':60}}}
    candidate={'groups':{'translation':{'rows':48,'points':22},'old':{'rows':100,'points':55}}}
    assert eligible(plan,baseline,candidate)
    for group,points in [('translation',21),('old',54)]:
        bad=copy.deepcopy(candidate);bad['groups'][group]['points']=points
        assert not eligible(plan,baseline,bad)
    candidate['groups']['translation']['rows']=47
    with pytest.raises(ValueError):eligible(plan,baseline,candidate)


def test_review_rejects_answers_bound_to_another_weight(tmp_path):
    questions=tmp_path/'questions.jsonl';answers=tmp_path/'candidate_answers.jsonl'
    binding=tmp_path/'candidate_binding.json';review=tmp_path/'candidate_review.json'
    question={'id':'example','prompt':'翻译：你好','reference':'Hello','scoring':'manual','language':'zh'}
    questions.write_text(json.dumps(question)+'\n')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    answers.write_text(json.dumps(dict(question,answer='Hello',question_sha256=sha(questions)))+'\n')
    run={'status':'completed','model_sha256':'a'*64,'questions_sha256':sha(questions),
         'answers_sha256':sha(answers),'device':'mps','dtype':'float32','do_sample':False,
         'repetition_penalty':1.0,'max_tokens':160}
    binding.write_text(json.dumps(run))
    report={'complete':True,'model_sha256':'a'*64,'answers_sha256':sha(answers),
            'questions_sha256':sha(questions),
            'binding_sha256':sha(binding),'scores':{'example':{'score':2,'reason':'译意完整'}},
            'groups':{'translation':{'rows':1,'points':2,'correct':1}}}
    review.write_text(json.dumps(report))
    assert verify_review(review,answers,questions,'a'*64)['complete']
    # 即使人工报告声称模型未变，实际运行绑定变化也必须拒绝。
    run['model_sha256']='b'*64;binding.write_text(json.dumps(run))
    report['binding_sha256']=sha(binding);review.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='运行绑定'):
        verify_review(review,answers,questions,'a'*64)
