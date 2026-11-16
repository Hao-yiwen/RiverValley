"""输出头分块交叉熵：控制词表 logits 的峰值内存，保留精确的监督目标。"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def chunked_linear_cross_entropy(hidden, weight, targets, chunk_size):
    """返回已对齐标签的 CE 总和；训练时重算输出头，避免保留所有分块 logits。"""
    if chunk_size <= 0:
        raise ValueError("输出头分块大小必须为正数")
    if hidden.shape[:-1] != targets.shape:
        raise ValueError("输出头输入与监督标签形状不一致")
    hidden = hidden.reshape(-1, hidden.shape[-1])
    targets = targets.reshape(-1)

    def loss_for_chunk(x, classifier, labels):
        # 显式 FP32 归约也覆盖非 autocast 调用；不筛除小梯度、不近似 softmax。
        logits = F.linear(x, classifier)
        return F.cross_entropy(logits.float(), labels, ignore_index=-100, reduction="sum")

    result = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, len(targets), chunk_size):
        inputs = (hidden[start:start + chunk_size], weight, targets[start:start + chunk_size])
        if torch.is_grad_enabled():
            # 仅顺序相加各块 loss 不会节省反向保存的 logits；必须配合重算。
            value = checkpoint(loss_for_chunk, *inputs, use_reentrant=False)
        else:
            value = loss_for_chunk(*inputs)
        result = result + value
    return result
