"""监控数值必须来自正确阶段和有效日志，不能伪造质量或进度。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/scripts'))
from training_monitor import map_training_record


def test_train_and_validation_metrics_stay_separate():
    training=map_training_record('pretrain',{'step':20,'event':'train','loss':9.2,'tokens':131072,'grad_norm':2.1,'update_seconds':4})
    assert training['进度/已训练token数']==20*131072
    assert training['性能/计算token每秒']==32768
    assert '预训练/验证loss' not in training
    validation=map_training_record('pretrain',{'step':200,'event':'eval','train':8.4,'test':8.7,'val_zh':8.6,'val_en':8.8})
    assert validation['预训练/中文验证loss']==8.6
    assert '预训练/训练loss' not in validation
    sft=map_training_record('sft',{'step':324,'loss':4.1})
    assert sft['进度/SFT完成百分比']==100
    assert '进度/已训练token数' not in sft


def test_monitor_rejects_invalid_step_and_nonfinite_values():
    assert map_training_record('pretrain',{'step':-1,'loss':5})=={}
    assert '预训练/训练loss' not in map_training_record('pretrain',{'step':1,'loss':float('nan')})
    assert '性能/单步秒数' not in map_training_record('pretrain',{'step':1,'update_seconds':0})
