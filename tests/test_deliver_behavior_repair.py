"""交付必须绑定开发门槛、一次留出和同一权重，不能只凭训练完成。"""
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/scripts'))
from deliver_behavior_repair import validate_repair_delivery


def inputs():
    plan={'purpose':'behavior_repair_probe','stages':[{'target_steps':500}],
          'selection_rule':{'minimum_new_dev_accuracy_gain':.15,'maximum_old_dev_point_drop':5}}
    selected={'eligible':True,'basis':'development','new_dev_accuracy_gain':.7,
              'old_dev_points':60,'baseline_old_dev_points':47,'model_sha256':'model'}
    review={'complete':True,'questions':384,'model_sha256':'model','question_set_sha256':'questions',
            'answers_sha256':'answers','quality_status':'内部任务改善，通用问答仍弱'}
    return plan,selected,review


def test_delivery_gates_and_binding():
    plan,selected,review=inputs()
    validate_repair_delivery(plan,selected,review,'questions','answers')
    for change in ({'new_dev_accuracy_gain':.1},{'old_dev_points':41},{'eligible':False}):
        with pytest.raises(ValueError):
            validate_repair_delivery(plan,dict(selected,**change),review,'questions','answers')
    for change in ({'complete':False},{'questions':383},{'model_sha256':'wrong'},
                   {'question_set_sha256':'wrong'},{'answers_sha256':'wrong'},{'quality_status':''}):
        with pytest.raises(ValueError):
            validate_repair_delivery(plan,selected,dict(review,**change),'questions','answers')
