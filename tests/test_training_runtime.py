"""预算停止请求必须在期限或停止文件生效时才触发。"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from training_runtime import TrainingStopper


def test_stop_file_and_deadline(monkeypatch, tmp_path):
    monkeypatch.delenv("RV_TRAIN_STOP_FILE", raising=False)
    monkeypatch.delenv("RV_TRAIN_DEADLINE", raising=False)
    assert not TrainingStopper(torch.device("cpu")).requested()
    path = tmp_path / "STOP"
    monkeypatch.setenv("RV_TRAIN_STOP_FILE", str(path))
    stopper = TrainingStopper(torch.device("cpu"))
    assert not stopper.requested()
    path.touch()
    assert stopper.requested()
    path.unlink()
    monkeypatch.setenv("RV_TRAIN_DEADLINE", "100")
    monkeypatch.setattr("training_runtime.time.time", lambda: 99)
    stopper = TrainingStopper(torch.device("cpu"))
    assert not stopper.requested()
    monkeypatch.setattr("training_runtime.time.time", lambda: 100)
    assert stopper.requested()


@pytest.mark.parametrize("value", ["nan", "inf", "-1", "zero"])
def test_invalid_deadline_fails(monkeypatch, value):
    monkeypatch.setenv("RV_TRAIN_DEADLINE", value)
    with pytest.raises(ValueError):
        TrainingStopper(torch.device("cpu"))
