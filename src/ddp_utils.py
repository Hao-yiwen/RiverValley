# -*- coding: utf-8 -*-
"""
RiverValley 分布式训练工具模块

提供 DDP (Distributed Data Parallel) 训练支持
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def setup_ddp():
    """
    初始化分布式训练环境

    支持 torchrun 启动方式:
        torchrun --nproc_per_node=4 script.py

    启动前确保 torchrun 注入环境变量:
        LOCAL_RANK, RANK, WORLD_SIZE
    每个进程会先 set_device(local_rank)，再将 device_id 传入 init_process_group，
    避免 NCCL 的 device-id 警告并确保进程绑定到对应 GPU。

    Returns:
        local_rank: 当前进程的本地 rank
        world_size: 总进程数
        is_main_process: 是否为主进程
    """
    # 检查是否为分布式环境
    if not dist.is_initialized():
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            # torchrun 启动
            rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])

            # 先切换到当前 rank 对应的 GPU，再告知 init_process_group
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
                dist.init_process_group(backend="nccl", device_id=local_rank)
            else:
                dist.init_process_group(backend="gloo")

            return local_rank, world_size, rank == 0
        else:
            # 单卡模式
            return 0, 1, True
    else:
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = dist.get_world_size()
        return local_rank, world_size, rank == 0


def cleanup_ddp():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_device(local_rank: int) -> torch.device:
    """
    获取当前进程的设备

    Args:
        local_rank: 本地 rank

    Returns:
        device: torch 设备
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def wrap_model_ddp(model, local_rank: int, use_ddp: bool = True):
    """
    将模型包装为 DDP

    Args:
        model: PyTorch 模型
        local_rank: 本地 rank
        use_ddp: 是否使用 DDP（单卡时为 False）

    Returns:
        model: 包装后的模型（DDP 或原模型）
    """
    if use_ddp and dist.is_initialized():
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)
    return model


def get_raw_model(model):
    """
    获取原始模型（去除 DDP/compile 包装）

    Args:
        model: 可能被包装的模型

    Returns:
        raw_model: 原始模型
    """
    while True:
        # 处理 DDP 包装
        if hasattr(model, "module"):
            model = model.module
            continue
        # 处理 torch.compile 包装
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
            continue
        return model


def create_distributed_sampler(dataset, shuffle: bool = True, seed: int = 0):
    """
    创建分布式采样器

    Args:
        dataset: PyTorch 数据集
        shuffle: 是否打乱
        seed: 随机种子

    Returns:
        sampler: DistributedSampler 或 None（单卡模式）
    """
    if dist.is_initialized():
        return DistributedSampler(
            dataset,
            shuffle=shuffle,
            seed=seed,
        )
    return None


def set_epoch_sampler(sampler, epoch: int):
    """
    设置采样器的 epoch（用于打乱数据）

    Args:
        sampler: DistributedSampler
        epoch: 当前 epoch
    """
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    跨进程求平均（用于汇总 loss 等指标）

    Args:
        tensor: 输入张量

    Returns:
        reduced_tensor: 平均后的张量
    """
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor = tensor / dist.get_world_size()
    return tensor


def print_rank0(*args, **kwargs):
    """只在主进程打印"""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


def is_main_process() -> bool:
    """检查是否为主进程"""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def barrier():
    """同步所有进程"""
    if dist.is_initialized():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def get_world_size() -> int:
    """获取总进程数"""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """获取当前进程的 rank"""
    if dist.is_initialized():
        return dist.get_rank()
    return 0
