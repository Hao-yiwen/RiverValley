"""长跑的外部停止请求：仅在完整优化器更新之间停止并交还保存流程。"""
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist


class TrainingStopper:
    """预算监控写入停止文件，或截止时间到达后，请求保存可恢复状态。"""

    def __init__(self, device):
        self.device = device
        self.path = os.environ.get("RV_TRAIN_STOP_FILE")
        value = os.environ.get("RV_TRAIN_DEADLINE")
        self.deadline = float(value) if value else None
        if self.deadline is not None and (not math.isfinite(self.deadline) or self.deadline <= 0):
            raise ValueError("RV_TRAIN_DEADLINE 必须是有限的正数 Unix 时间戳")

    def requested(self):
        """所有 rank 在同一更新边界停止，避免某个进程单独退出造成死锁。"""
        if self.path is None and self.deadline is None:
            return False
        stop = bool(self.path and Path(self.path).exists())
        stop = stop or (self.deadline is not None and time.time() >= self.deadline)
        if dist.is_initialized():
            value = torch.tensor(int(stop), device=self.device)
            dist.all_reduce(value, op=dist.ReduceOp.MAX)
            stop = bool(value.item())
        return stop
