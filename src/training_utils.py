"""预训练和 SFT 共用的损失统计、数据游标与续训状态工具。"""

import logging
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Sampler, default_collate

logger = logging.getLogger(__name__)


def append_metrics(folder, record):
    """主进程追加结构化日志；即使没有外部实验平台，也保留可比较的指标。"""
    import json
    from pathlib import Path
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "metrics.jsonl").open("a", encoding="utf-8") as sink:
        sink.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


class ResumableBatchSampler(Sampler):
    """按完整梯度累积组切分 epoch，直接从已完成的更新步恢复数据。"""

    def __init__(self, size, batch_size, accumulation_steps, seed, rank=0, world_size=1):
        if min(size, batch_size, accumulation_steps, world_size) <= 0:
            raise ValueError("样本数、批量、累积步数和进程数必须大于 0")
        self.size, self.batch_size, self.accumulation_steps = size, batch_size, accumulation_steps
        self.seed, self.rank, self.world_size = seed, rank, world_size
        self.steps_per_epoch = size // (batch_size * accumulation_steps * world_size)
        if not self.steps_per_epoch:
            raise ValueError("数据不足一个有效批量，请减小 batch_size 或 accumulation_steps")
        self.epoch, self.start_batch = 0, 0

    def set_step(self, next_step):
        self.epoch, offset = divmod(next_step, self.steps_per_epoch)
        self.start_batch = offset * self.accumulation_steps

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.size, generator=generator)
        count = self.steps_per_epoch * self.accumulation_steps * self.batch_size
        order = order[:count * self.world_size][self.rank::self.world_size]
        for start in range(self.start_batch * self.batch_size, count, self.batch_size):
            yield order[start:start + self.batch_size].tolist()

    def __len__(self):
        return self.steps_per_epoch * self.accumulation_steps - self.start_batch


def trim_padding_collate(samples):
    """在 CPU 上裁掉整批公共的右侧 padding，保持标签和掩码对齐。"""
    batch = default_collate(samples)
    valid_columns = batch[3].bool().any(dim=0).nonzero().flatten()
    length = int(valid_columns[-1]) + 1 if len(valid_columns) else 1
    return tuple(tensor[:, :length].contiguous() for tensor in batch)


def token_loss_sum(logits, targets, loss_mask):
    """返回有效目标 token 的 CE 总和，避免短回复与长回复获得相同权重。"""
    labels = targets.masked_fill(~loss_mask.bool(), -100)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                           ignore_index=-100, reduction="sum")


def sum_across_ranks(value):
    """汇总各进程统计量；单进程原样返回。"""
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def forward_loss_sum(model, x, y, mask, attention):
    """为可选分块输出头提供已对齐标签，其余训练路径仍返回完整 logits。"""
    from ddp_utils import get_raw_model
    if getattr(get_raw_model(model).config, "loss_chunk_size", 0):
        outputs = model(input_ids=x, attention_mask=attention, use_cache=False,
                        loss_targets=y.masked_fill(~mask.bool(), -100))
        return outputs.loss, outputs
    outputs = model(input_ids=x, attention_mask=attention, use_cache=False)
    return token_loss_sum(outputs.logits, y, mask), outputs


def train_update(model, optimizer, batches, device, grad_clip, skip_attention_mask=False):
    """按整个有效批量的目标 token 数归一化，并在更新前检查有限梯度。"""
    from ddp_utils import get_raw_model, get_world_size

    world_size = get_world_size()
    total_tokens = torch.tensor(sum(int(batch[2].sum()) for batch in batches), device=device)
    sum_across_ranks(total_tokens)
    if total_tokens.item() == 0:
        raise ValueError("有效批量没有任何监督 token，已停止更新，请检查数据和截断设置")
    optimizer.zero_grad(set_to_none=True)
    stats = torch.zeros(2, device=device, dtype=torch.float64)
    for index, batch in enumerate(batches):
        x, y, mask, attention = (item.to(device, non_blocking=True) for item in batch)
        sync = model.no_sync() if world_size > 1 and index < len(batches) - 1 else nullcontext()
        with sync:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                loss_sum, outputs = forward_loss_sum(model, x, y, mask,
                                                     None if skip_attention_mask else attention)
                loss = loss_sum * (world_size / total_tokens)
                aux = getattr(outputs, "aux_loss", None)
                config = get_raw_model(model).config
                if aux is not None and config.use_moe:
                    loss = loss + config.aux_loss_coef * aux / len(batches)
            loss.backward()
        stats[0] += loss_sum.detach()
        stats[1] += mask.sum()
    sum_across_ranks(stats)
    if not torch.isfinite(stats).all():
        raise FloatingPointError(f"训练损失非有限，已停止更新: {stats.tolist()}")
    # 非有限梯度会直接报错，避免把已损坏的状态写入 latest。
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
    optimizer.step()
    return {"loss": (stats[0] / stats[1]).item(), "tokens": int(stats[1].item()),
            "grad_norm": float(grad_norm)}


@torch.no_grad()
def evaluate_loss(model, loaders, device, eval_iters, skip_attention_mask=False):
    """按监督 token 加权评估；调用方提供固定、无重复的验证分片。"""
    from ddp_utils import get_raw_model

    raw_model = get_raw_model(model)
    was_training = raw_model.training
    raw_model.eval()
    results = {}
    try:
        for name, loader in loaders.items():
            stats = torch.zeros(2, device=device, dtype=torch.float64)
            for index, batch in enumerate(loader):
                if index >= eval_iters:
                    break
                x, y, mask, attention = (item.to(device, non_blocking=True) for item in batch)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                    loss_sum, _ = forward_loss_sum(raw_model, x, y, mask,
                                                   None if skip_attention_mask else attention)
                    stats[0] += loss_sum
                stats[1] += mask.sum()
            sum_across_ranks(stats)
            if stats[1].item() == 0 or not torch.isfinite(stats).all():
                raise ValueError(f"{name} 评估没有有效监督 token 或损失非有限: {stats.tolist()}")
            results[name] = (stats[0] / stats[1]).item()
    finally:
        raw_model.train(was_training)
    return results


def collect_rng_states():
    """所有 rank 一起收集随机数状态，支持 dropout 断点续训。"""
    state = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    if not dist.is_initialized():
        return [state]
    states = [None] * dist.get_world_size()
    dist.all_gather_object(states, state)
    return states


def restore_rng_states(state):
    """恢复当前 rank 随机数；旧格式明确记录不能完全重现的限制。"""
    states = state.get("rng_states")
    if not states:
        logger.warning("旧 checkpoint 缺少 RNG 状态，无法保证与原训练逐位一致")
        return
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if len(states) != world_size:
        raise ValueError("续训进程数已改变，请使用仅加载权重模式开始新训练")
    torch.set_rng_state(states[rank]["cpu"].cpu())
    if "cuda" in states[rank] and torch.cuda.is_available():
        torch.cuda.set_rng_state(states[rank]["cuda"].cpu())


def resume_signature(config, dataset_size, world_size):
    """记录影响数据顺序和学习率调度的配置，拒绝静默改变续训语义。"""
    names = ("random_seed", "batch_size", "accumulation_steps", "max_seq_len", "epochs",
             "learning_rate", "min_lr", "warmup_ratio", "weight_decay", "grad_clip", "data_fraction", "data_path", "memmap_dir", "val_memmap_dir",
             "val_data_path", "extra_val_memmap_dirs", "schedule_steps", "adam_beta1", "adam_beta2", "adam_eps", "loss_chunk_size")
    return {**{key: getattr(config, key) for key in names if hasattr(config, key)},
            "dataset_size": dataset_size, "world_size": world_size, "sampler_version": 1}


def validate_training_options(config):
    """提前拒绝无效学习率和优化器参数，避免启动 GPU 后才发现配置错误。"""
    import math
    for key in ("learning_rate", "min_lr", "warmup_ratio", "adam_beta1", "adam_beta2", "adam_eps", "grad_clip"):
        if not math.isfinite(getattr(config, key)):
            raise ValueError(f"{key} 必须为有限数")
    if not 0 <= config.min_lr <= config.learning_rate or config.learning_rate <= 0:
        raise ValueError("学习率必须满足 0 <= min_lr <= learning_rate 且 learning_rate > 0")
    if not 0 <= config.warmup_ratio < 1:
        raise ValueError("warmup_ratio 必须在 [0, 1) 内")
    if not 0 <= config.adam_beta1 < 1 or not 0 <= config.adam_beta2 < 1 or config.adam_eps <= 0:
        raise ValueError("Adam 参数必须满足 0 <= beta < 1 且 eps > 0")
    if config.grad_clip <= 0 or config.loss_chunk_size < 0:
        raise ValueError("grad_clip 必须为正数，loss_chunk_size 不能为负数")
    if config.schedule_steps is not None and config.schedule_steps <= 0:
        raise ValueError("schedule_steps 必须为正数")


def prune_interval_checkpoints(folder, keep):
    """只在新 checkpoint 已完整保存后清理历史周期权重。"""
    import re
    import shutil
    from pathlib import Path
    if keep < 0:
        raise ValueError("keep_interval_checkpoints 不能为负数")
    paths = [p for p in Path(folder).iterdir()
             if p.is_dir() and not p.is_symlink() and re.fullmatch(r"ckpt_\d+", p.name)]
    paths.sort(key=lambda p: int(p.name.split("_")[-1]), reverse=True)
    for path in paths[keep:]:
        logger.info("清理历史周期 checkpoint: %s，保留数=%d", path, keep)
        shutil.rmtree(path)


def validate_resume(state, signature):
    """完整续训必须有优化器状态；旧数据排列不能被误认为精确恢复。"""
    if not state or not state.get("optimizer"):
        raise ValueError("checkpoint 缺少优化器状态；若只想加载权重，请设置 resume_weights_only=True")
    previous = state.get("resume_signature")
    if previous is None:
        logger.warning("旧 checkpoint 缺少数据游标配置，将按新版固定采样恢复，无法重现原数据顺序")
    elif previous != signature:
        raise ValueError(f"续训配置与 checkpoint 不一致: 原配置={previous}, 当前配置={signature}")


def validate_new_pretrain_stage(state, signature, expected_step):
    """显式阶段迁移保留 Adam/RNG，重置新数据游标；与普通精确恢复分开校验。"""
    if not isinstance(expected_step, int) or expected_step <= 0:
        raise ValueError('新阶段必须指定前一阶段的最终步数')
    if not state or not state.get('has_optimizer_state') or not state.get('optimizer') or not state.get('rng_states'):
        raise ValueError('新阶段初始化缺少完整优化器或随机状态')
    previous = state.get('resume_signature') or {}
    if not previous.get('memmap_dir'):
        raise ValueError('新预训练阶段只能从预训练状态初始化，不能使用 SFT 状态')
    if state.get('next_iter') != expected_step or previous.get('schedule_steps') != expected_step:
        raise ValueError('来源不是已完成原计划的最终断点')
    immutable = ('random_seed', 'batch_size', 'accumulation_steps', 'max_seq_len', 'world_size',
                 'adam_beta1', 'adam_beta2', 'adam_eps', 'weight_decay', 'grad_clip', 'tokenizer_sha256')
    for key in immutable:
        if key not in previous or previous[key] != signature.get(key):
            raise ValueError(f'阶段迁移不允许改变 {key}')
    if previous.get('metadata_sha256') == signature.get('metadata_sha256'):
        raise ValueError('新阶段仍使用原数据签名，不能把重复训练记为新增语料')
    logger.info('已验证阶段迁移：前阶段=%d 步，新阶段=%s 步；保留优化器/RNG，重置游标和最佳验证值',
                expected_step, signature.get('schedule_steps'))
    return {'source_completed_steps': expected_step, 'source_resume_signature': previous,
            'transition': '保留优化器和RNG；新数据从0开始；新warmup/余弦计划；重置最佳验证值'}


def validate_checkpoint_tokenizer(checkpoint, tokenizer):
    """词表大小相同仍可能拥有不同 token ID；比较编码结构后才能继续训练。"""
    import json
    from pathlib import Path
    path = Path(checkpoint) / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(f"checkpoint 缺少原始 tokenizer: {path}")
    saved = json.loads(path.read_text(encoding="utf-8"))
    current = json.loads(tokenizer.to_str())
    for key in ("model", "normalizer", "pre_tokenizer"):
        if saved.get(key) != current.get(key):
            raise ValueError(f"checkpoint 与当前 tokenizer 的 {key} 不一致，请保持同一 tokenizer")


def resolve_checkpoint_path(path):
    """提交新目录时若意外中断，使用上一份已完整写入的 checkpoint。"""
    from pathlib import Path
    path = Path(path)
    previous = path.with_name(path.name + ".previous")
    if not path.exists() and previous.is_dir():
        logger.warning("checkpoint 提交中断，恢复上一份完整目录: %s", previous)
        return str(previous)
    return str(path)


def write_checkpoint_directory(path, writer):
    """先完整写入临时目录，再切换目录；写入失败时原 checkpoint 保持可用。"""
    import os
    import shutil
    import tempfile
    from pathlib import Path
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    previous = target.with_name(target.name + ".previous")
    try:
        writer(str(stage))
        if target.exists():
            if previous.exists():
                shutil.rmtree(previous)
            os.replace(target, previous)
        try:
            os.replace(stage, target)
        except OSError:
            if not target.exists() and previous.exists():
                os.replace(previous, target)
            raise
    except Exception:
        logger.exception("checkpoint 写入失败，保留上一份完整状态: %s", target)
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)
