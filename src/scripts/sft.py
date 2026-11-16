# -*- coding: utf-8 -*-
"""监督微调入口：支持 JSON 配置、小实验、精确数据游标和单卡／DDP。"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)
os.chdir(PROJECT_ROOT)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import DataLoader, Subset

from config import SFTConfig
from data import SFTDataset
from ddp_utils import (setup_ddp, cleanup_ddp, get_device, wrap_model_ddp,
                       get_raw_model, get_rank, print_rank0, barrier)
from model import RiverValleyModel
from tokenizer import load_tokenizer, get_vocab_size, get_pad_id
from pretrain import save_model_hf
from training_runtime import TrainingStopper
from training_utils import (ResumableBatchSampler, trim_padding_collate, train_update,
                            evaluate_loss, resume_signature, validate_resume, restore_rng_states,
                            validate_checkpoint_tokenizer, resolve_checkpoint_path, validate_training_options, append_metrics)


def get_lr(iter_num, train_iters, warmup_iters, learning_rate, min_lr):
    """按完整训练步数调度，小实验提前停止时不压缩学习率曲线。"""
    if iter_num < warmup_iters:
        return learning_rate * (iter_num + 1) / warmup_iters
    ratio = min(1.0, (iter_num - warmup_iters) / max(1, train_iters - warmup_iters))
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (learning_rate - min_lr)


def main(config=None):
    """先检查数据和 checkpoint，再开始有界的训练任务。"""
    config = config or SFTConfig()
    validate_training_options(config)
    local_rank, world_size, is_main = setup_ddp()
    rank = get_rank()
    device = get_device(local_rank) if world_size > 1 else torch.device(config.device)
    if min(config.epochs, config.eval_iters, config.eval_interval, config.save_interval) <= 0:
        raise ValueError("epochs、eval_iters、eval_interval、save_interval 必须为正数")
    if config.max_steps is not None and config.max_steps <= 0:
        raise ValueError("max_steps 必须为正数")
    torch.manual_seed(config.random_seed + rank)
    print_rank0(f"SFT 启动: device={device}, world_size={world_size}, 数据={config.data_path}")
    tokenizer = load_tokenizer(config.bpe_model_path)
    pad_id = get_pad_id(tokenizer)
    dataset = SFTDataset(config.data_path, tokenizer, config.max_seq_len, pad_id)

    # 完全重复的对话只保留一条；相同提问的不同答案归入同一分片，避免验证泄漏。
    unique, groups = {}, {}
    for sample in dataset.samples:
        key = json.dumps(sample["conversations"], ensure_ascii=False, sort_keys=True)
        unique.setdefault(key, sample)
    for sample in unique.values():
        prompt = [turn for turn in sample["conversations"] if turn["role"] != "assistant"]
        key = json.dumps(prompt, ensure_ascii=False, sort_keys=True)
        groups.setdefault(key, []).append(sample)
    if len(groups) < 2:
        raise ValueError("SFT 至少需要两个不同的提问组，才能划分独立验证集")
    order = torch.randperm(len(groups), generator=torch.Generator().manual_seed(config.random_seed)).tolist()
    grouped = list(groups.values())
    split = min(len(order) - 1, max(1, int(0.9 * len(order))))
    train_samples = [sample for index in order[:split] for sample in grouped[index]]
    test_samples = [sample for index in order[split:] for sample in grouped[index]]
    if config.val_data_path:
        validation = SFTDataset(config.val_data_path, tokenizer, config.max_seq_len, pad_id)
        val_unique = {json.dumps(sample["conversations"], ensure_ascii=False, sort_keys=True): sample
                      for sample in validation.samples}
        val_prompts = {json.dumps([turn for turn in sample["conversations"] if turn["role"] != "assistant"],
                                 ensure_ascii=False, sort_keys=True) for sample in val_unique.values()}
        if not val_unique:
            raise ValueError("独立 SFT 验证集没有有效样本")
        if val_prompts.intersection(groups):
            raise ValueError("独立 SFT 验证集与训练集存在相同提问，必须先去重再划分")
        train_samples, test_samples = list(unique.values()), list(val_unique.values())
        print_rank0(f"使用冻结的独立 SFT 验证集: {config.val_data_path}")
    print_rank0(f"数据检查: 有效={len(dataset)}, 去重后={len(unique)}, 训练={len(train_samples)}, 验证={len(test_samples)}")
    train_dataset = SFTDataset(train_samples, tokenizer, config.max_seq_len, pad_id)
    test_dataset = SFTDataset(test_samples, tokenizer, config.max_seq_len, pad_id)
    sampler = ResumableBatchSampler(len(train_dataset), config.batch_size,
                                   config.accumulation_steps, config.random_seed, rank, world_size)
    train_iters = sampler.steps_per_epoch * config.epochs
    stop_iter = min(train_iters, config.max_steps) if config.max_steps is not None else train_iters
    schedule_iters = config.schedule_steps or train_iters
    config._resume_signature = resume_signature(config, len(train_dataset), world_size)
    # 数据内容和 tokenizer 改变时必须开启新任务，即使行数相同也不能静默续训。
    import hashlib
    config._resume_signature["data_sha256"] = hashlib.sha256(Path(config.data_path).read_bytes()).hexdigest()
    config._resume_signature["tokenizer_sha256"] = hashlib.sha256(Path(config.bpe_model_path).read_bytes()).hexdigest()
    if config.val_data_path:
        config._resume_signature["val_data_sha256"] = hashlib.sha256(Path(config.val_data_path).read_bytes()).hexdigest()

    loader_options = dict(num_workers=config.num_workers, pin_memory=config.pin_memory,
                          generator=torch.Generator().manual_seed(config.random_seed + rank))
    if config.dynamic_padding:
        loader_options["collate_fn"] = trim_padding_collate
    if config.num_workers:
        loader_options.update(prefetch_factor=config.prefetch_factor, persistent_workers=True)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, **loader_options)
    # 验证不补齐重复样本，且不复用训练游标；空 rank 分片也能正确汇总。
    eval_options = {key: value for key, value in loader_options.items()
                    if key not in ("num_workers", "persistent_workers", "prefetch_factor")}
    eval_options["num_workers"] = 0
    eval_loaders = {name: DataLoader(Subset(ds, range(rank, len(ds), world_size)),
                                   batch_size=config.batch_size, **eval_options)
                    for name, ds in [("train", train_dataset), ("test", test_dataset)]}
    for language in ("zh", "en"):
        positions = [i for i, sample in enumerate(test_dataset.samples) if sample.get("language") == language]
        if positions:
            eval_loaders[f"val_{language}"] = DataLoader(Subset(test_dataset, positions[rank::world_size]),
                batch_size=config.batch_size, **eval_options)

    checkpoint = resolve_checkpoint_path(config.resume_path if config.resume else config.pretrain_checkpoint)
    if not os.path.isdir(checkpoint):
        raise FileNotFoundError(f"模型目录不存在: {checkpoint}")
    validate_checkpoint_tokenizer(checkpoint, tokenizer)
    model = RiverValleyModel.from_pretrained(checkpoint).float().to(device)
    if model.config.vocab_size != get_vocab_size(tokenizer) or model.config.pad_token_id != pad_id:
        raise ValueError("模型词表大小或 pad_id 与 tokenizer 不一致，请使用预训练时的 tokenizer")
    model.config.use_gradient_checkpointing = config.use_gradient_checkpointing
    model.config.max_seq_len = config.max_seq_len
    model.gradient_checkpointing = config.use_gradient_checkpointing
    model.config.use_native_gqa = config.use_native_gqa
    model.config.loss_chunk_size = config.loss_chunk_size
    for block in model.blocks:
        block.sa.use_native_gqa = config.use_native_gqa
    state, next_iter, best_loss = None, 0, float("inf")
    if config.resume and not config.resume_weights_only:
        state_path = Path(checkpoint) / "training_state.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"续训状态不存在: {state_path}")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        validate_resume(state, config._resume_signature)
        if state.get("next_iter") is None:
            raise ValueError("旧 SFT checkpoint 未记录 next_iter，无法判定保存时机，请仅加载权重开始新任务")
        next_iter = state["next_iter"]
        best_loss = state.get("best_val_loss", float("inf"))
    if next_iter > stop_iter:
        raise ValueError(f"checkpoint 已完成 {next_iter} 步，超过本次停止位置 {stop_iter}")
    if config.use_compile:
        print_rank0("启用编译：首次编译和长度变化会产生额外开销")
        model = torch.compile(model)
    if world_size > 1:
        model = wrap_model_ddp(model, local_rank)
    raw = get_raw_model(model)
    groups = [{"params": [p for p in raw.parameters() if p.requires_grad and p.ndim >= 2], "weight_decay": config.weight_decay},
              {"params": [p for p in raw.parameters() if p.requires_grad and p.ndim < 2], "weight_decay": 0.0}]
    optimizer = torch.optim.AdamW(groups, lr=config.learning_rate,
                                  betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_eps,
                                  fused=config.adamw_fused and device.type == "cuda")
    if state:
        optimizer.load_state_dict(state["optimizer"])
        restore_rng_states(state)
        del state
    print_rank0(f"模型参数={sum(p.numel() for p in raw.parameters()):,}, 有效批量={config.batch_size * world_size * config.accumulation_steps}, "
                f"数据可用={train_iters} 步, 学习率计划={schedule_iters} 步, 本轮={next_iter}→{stop_iter}, GQA={config.use_native_gqa}, 重算={config.use_gradient_checkpointing}")
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    tracker = None
    if config.use_swanlab and is_main:
        import swanlab
        tracker = swanlab
        tracker.init(project=config.swanlab_project, experiment_name=config.swanlab_run_name,
                     config=config.to_dict())

    def save(kind, completed, with_optimizer):
        barrier()
        save_model_hf(model, tokenizer, str(Path(config.checkpoint_dir) / kind),
                      optimizer.state_dict() if with_optimizer and config.save_optimizer_state else None,
                      completed - 1, best_loss, config, next_iter=completed)
        barrier()

    def evaluate(completed):
        nonlocal best_loss
        losses = evaluate_loss(model, eval_loaders, device, config.eval_iters)
        print_rank0(f"评估 step={completed}: train={losses['train']:.4f}, test={losses['test']:.4f}")
        if is_main:
            append_metrics(config.checkpoint_dir, {"event": "eval", "step": completed, **losses})
            print_rank0(f"完整验证指标: {json.dumps(losses, ensure_ascii=False)}")
        if tracker:
            tracker.log({"step": completed, "train_loss": losses["train"], "test_loss": losses["test"]})
        if config.save_best and losses["test"] < best_loss:
            best_loss = losses["test"]
            save("best", completed, False)
        return losses

    model.train()
    evaluate(next_iter)
    iterator = None
    started = time.perf_counter()
    stopper = TrainingStopper(device)
    for step in range(next_iter, stop_iter):
        if stopper.requested():
            stop_iter = step
            print_rank0(f"收到预算或外部停止请求：SFT 已完成 {stop_iter} 步，开始保存完整断点")
            if is_main:
                append_metrics(config.checkpoint_dir, {"event": "stop_requested", "step": stop_iter})
            break
        if iterator is None or step % sampler.steps_per_epoch == 0:
            sampler.set_step(step)
            iterator = iter(train_loader)
            print_rank0(f"数据游标: epoch={sampler.epoch}, 起始 micro-batch={sampler.start_batch}")
        lr = get_lr(step, schedule_iters, int(schedule_iters * config.warmup_ratio), config.learning_rate, config.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        batches = [next(iterator) for _ in range(config.accumulation_steps)]
        stats = train_update(model, optimizer, batches, device, config.grad_clip)
        completed = step + 1
        if config.log_interval and (completed % config.log_interval == 0 or completed == stop_iter):
            record = {"step": completed, "lr": lr, **stats, "elapsed_s": time.perf_counter() - started}
            print_rank0(f"训练记录: {json.dumps(record, ensure_ascii=False)}")
            if is_main:
                with open(Path(config.checkpoint_dir) / "metrics.jsonl", "a", encoding="utf-8") as output:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
        if completed % config.eval_interval == 0 and completed != stop_iter:
            evaluate(completed)
        if completed % config.save_interval == 0 and completed != stop_iter:
            save("latest", completed, True)
    losses = evaluate(stop_iter)
    save("latest", stop_iter, True)
    if tracker:
        tracker.finish()
    cleanup_ddp()
    return {"next_iter": stop_iter, "best_val_loss": best_loss, **losses}


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="RiverValley SFT 训练")
    parser.add_argument("--config", help="覆盖 SFTConfig 默认值的 JSON 文件")
    args = parser.parse_args()
    main(SFTConfig(**json.loads(Path(args.config).read_text(encoding="utf-8"))) if args.config else None)
