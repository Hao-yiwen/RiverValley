# -*- coding: utf-8 -*-
"""
RiverValley 思维链蒸馏训练脚本

参考 minimind 项目的 train_distill_reason.py 实现
对 <think>, </think>, <answer>, </answer> 标签位置增加权重

DDP 使用方法:
    单卡: python src/scripts/train_reason.py
    多卡: torchrun --nproc_per_node=4 src/scripts/train_reason.py
"""

import os
import sys
import math

# 获取项目根目录和 src 目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# 添加 src 到路径
sys.path.insert(0, SRC_DIR)

# 切换到项目根目录，确保相对路径正确
os.chdir(PROJECT_ROOT)

import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import PreTrainedTokenizerFast, GenerationConfig

from config import ReasonDistillConfig, DEFAULT_MAX_SEQ_LEN
from model import RiverValleyModel
from tokenizer import (
    load_tokenizer,
    get_vocab_size,
    get_pad_id,
    get_eos_id,
    get_im_start_id,
    get_im_end_id,
    CHAT_TEMPLATE,
    PAD_TOKEN,
    EOS_TOKEN,
    IM_START_TOKEN,
    IM_END_TOKEN,
)
from data import SFTDataset
from ddp_utils import (
    setup_ddp,
    cleanup_ddp,
    get_device,
    wrap_model_ddp,
    get_raw_model,
    create_distributed_sampler,
    set_epoch_sampler,
    reduce_mean,
    print_rank0,
    is_main_process,
    barrier,
    get_world_size,
)

swanlab = None


def get_generate_model(model):
    return get_raw_model(model)


def add_moe_aux_loss(loss, outputs, model_config):
    """在自定义 CE loss 外追加 MoE 路由负载均衡损失。"""
    aux_loss = getattr(outputs, "aux_loss", None)
    if aux_loss is not None and getattr(model_config, "use_moe", False):
        return loss + model_config.aux_loss_coef * aux_loss
    return loss


def save_model_hf(model, tokenizer, save_path, optimizer_state=None, iter_num=None, best_val_loss=None, config=None):
    """保存模型（只在主进程保存）"""
    if not is_main_process():
        return

    os.makedirs(save_path, exist_ok=True)
    max_seq_len = config.max_seq_len if config else DEFAULT_MAX_SEQ_LEN

    model_to_save = get_raw_model(model)
    model_to_save.save_pretrained(save_path)

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=IM_START_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        unk_token=None,
        additional_special_tokens=[IM_START_TOKEN, IM_END_TOKEN],
        model_max_length=max_seq_len,
    )
    hf_tokenizer.chat_template = CHAT_TEMPLATE
    hf_tokenizer.save_pretrained(save_path)

    gen_config = GenerationConfig(
        max_new_tokens=200,
        do_sample=True,
        temperature=1.0,
        top_p=0.9,
        pad_token_id=get_pad_id(tokenizer),
        eos_token_id=[get_eos_id(tokenizer), get_im_end_id(tokenizer)],
        bos_token_id=get_im_start_id(tokenizer),
    )
    gen_config.save_pretrained(save_path)

    if optimizer_state is not None or iter_num is not None:
        training_state = {
            "optimizer": optimizer_state,
            "iter_num": iter_num,
            "best_val_loss": best_val_loss,
        }
        torch.save(training_state, os.path.join(save_path, "training_state.pt"))

    print(f"  -> 模型已保存: {save_path}")


def get_lr(iter_num: int, train_iters: int, warmup_iters: int, learning_rate: float, min_lr: float) -> float:
    """余弦退火学习率调度"""
    if iter_num < warmup_iters:
        return learning_rate * (iter_num + 1) / warmup_iters

    decay_iters = train_iters - warmup_iters
    iter_after_warmup = iter_num - warmup_iters
    decay_ratio = iter_after_warmup / decay_iters
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coeff * (learning_rate - min_lr)


def main(config: ReasonDistillConfig = None):
    """思维链蒸馏训练主函数"""
    if config is None:
        config = ReasonDistillConfig()

    # 初始化 DDP
    local_rank, world_size, is_main = setup_ddp()
    use_ddp = world_size > 1

    # 设置设备
    if use_ddp:
        device = get_device(local_rank)
        config.device = str(device)
    else:
        device = torch.device(config.device)

    print_rank0(f"Using device: {config.device}")
    print_rank0(f"World size: {world_size}")
    print_rank0(f"数据文件: {config.data_path}")
    print_rank0(f"DPO 模型: {config.dpo_checkpoint}")
    print_rank0(f"思维链标签权重: {config.reason_tag_weight}")
    print_rank0(f"梯度累积步数: {config.accumulation_steps}")

    if is_main:
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    # 加载 tokenizer
    tokenizer = load_tokenizer(config.bpe_model_path)
    vocab_size = get_vocab_size(tokenizer)
    pad_id = get_pad_id(tokenizer)
    print_rank0(f"\nBPE 词汇表大小: {vocab_size}")

    # 获取思维链标签的 token ids
    start_of_think_ids = tokenizer.encode("<think>").ids
    end_of_think_ids = tokenizer.encode("</think>").ids
    start_of_answer_ids = tokenizer.encode("<answer>").ids
    end_of_answer_ids = tokenizer.encode("</answer>").ids

    # 保留完整标签序列，避免只按单个 token id 匹配导致误加权
    reason_tag_sequences = [
        torch.tensor(ids, dtype=torch.long, device=device)
        for ids in [
            start_of_think_ids,
            end_of_think_ids,
            start_of_answer_ids,
            end_of_answer_ids,
        ]
        if ids
    ]

    print_rank0(f"思维链标签 token ids:")
    print_rank0(f"  <think>: {start_of_think_ids}")
    print_rank0(f"  </think>: {end_of_think_ids}")
    print_rank0(f"  <answer>: {start_of_answer_ids}")
    print_rank0(f"  </answer>: {end_of_answer_ids}")

    # 创建数据集
    dataset = SFTDataset(config.data_path, tokenizer, config.max_seq_len, pad_id)

    # 先打乱再分割训练集和测试集，避免源数据排序污染验证集
    generator = torch.Generator()
    generator.manual_seed(config.random_seed)
    indices = torch.randperm(len(dataset.samples), generator=generator).tolist()
    shuffled_samples = [dataset.samples[i] for i in indices]
    n = int(0.9 * len(shuffled_samples))
    train_samples = shuffled_samples[:n]
    test_samples = shuffled_samples[n:]
    print_rank0(f"训练集: {len(train_samples)} 条")
    print_rank0(f"测试集: {len(test_samples)} 条")

    # 计算训练步数（考虑 DDP 和梯度累积）
    effective_batch_size = config.batch_size * world_size * config.accumulation_steps
    steps_per_epoch = len(train_samples) // (config.batch_size * world_size * config.accumulation_steps)
    train_iters = steps_per_epoch * config.epochs
    warmup_iters = int(train_iters * config.warmup_ratio)
    print_rank0(f"每 epoch 步数: {steps_per_epoch}")
    print_rank0(f"总训练步数: {train_iters} ({config.epochs} epochs)")
    print_rank0(f"Warmup 步数: {warmup_iters} ({config.warmup_ratio * 100:.0f}%)")
    print_rank0(f"有效 batch size: {effective_batch_size} (per GPU: {config.batch_size}, GPUs: {world_size}, accum: {config.accumulation_steps})")

    # 创建 DataLoader
    train_dataset = SFTDataset.__new__(SFTDataset)
    train_dataset.samples = train_samples
    train_dataset.block_size = config.max_seq_len
    train_dataset.pad_id = pad_id
    train_dataset.tokenizer = tokenizer
    train_dataset.mask_user = True

    test_dataset = SFTDataset.__new__(SFTDataset)
    test_dataset.samples = test_samples
    test_dataset.block_size = config.max_seq_len
    test_dataset.pad_id = pad_id
    test_dataset.tokenizer = tokenizer
    test_dataset.mask_user = True

    # 设置随机种子
    torch.manual_seed(config.random_seed + local_rank)

    # 创建分布式采样器
    train_sampler = create_distributed_sampler(train_dataset, shuffle=True, seed=config.random_seed)
    test_sampler = create_distributed_sampler(test_dataset, shuffle=False, seed=config.random_seed)

    def build_dataloader(ds, sampler, shuffle: bool, drop_last: bool):
        loader_kwargs = {
            "batch_size": config.batch_size,
            "shuffle": shuffle and sampler is None,
            "sampler": sampler,
            "num_workers": config.num_workers,
            "pin_memory": config.pin_memory,
            "drop_last": drop_last,
        }
        if config.num_workers > 0:
            loader_kwargs["prefetch_factor"] = config.prefetch_factor
            loader_kwargs["persistent_workers"] = True
        return DataLoader(ds, **loader_kwargs)

    train_loader = build_dataloader(train_dataset, train_sampler, shuffle=True, drop_last=True)
    test_loader = build_dataloader(test_dataset, test_sampler, shuffle=False, drop_last=False)

    def build_reason_loss_mask(labels: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        """只在 assistant loss 区间内提高思维链标签权重，避免把 user/pad token 重新纳入 loss。"""
        weighted_mask = loss_mask.float().clone()
        tag_positions = torch.zeros_like(loss_mask, dtype=torch.bool)
        for tag_ids in reason_tag_sequences:
            tag_len = tag_ids.numel()
            if tag_len == 0 or labels.size(1) < tag_len:
                continue
            windows = labels.unfold(dimension=1, size=tag_len, step=1)
            matches = (windows == tag_ids.view(1, 1, -1)).all(dim=-1)
            if not matches.any():
                continue
            for offset in range(tag_len):
                tag_positions[:, offset:offset + matches.size(1)] |= matches
        tag_positions &= loss_mask.bool()
        weighted_mask[tag_positions] = config.reason_tag_weight
        return weighted_mask

    # 初始化 swanlab（只在主进程）
    global swanlab
    if config.use_swanlab and is_main:
        import swanlab as _swanlab
        swanlab = _swanlab

        run_name = config.swanlab_run_name or f"Reason-{config.epochs}epochs"
        swanlab.init(
            project=config.swanlab_project,
            experiment_name=run_name,
            config=config.to_dict(),
        )

    # 加载模型
    start_iter = 0
    best_val_loss = float("inf")
    training_state = None

    if config.resume:
        print_rank0(f"\n恢复训练: {config.resume_path}")
        model = RiverValleyModel.from_pretrained(
            config.resume_path,
        ).to(device)  # 保持 fp32，使用 autocast 进行混合精度

        # 加载训练状态
        state_path = os.path.join(config.resume_path, "training_state.pt")
        if os.path.exists(state_path):
            training_state = torch.load(state_path, weights_only=False)
            start_iter = training_state.get("iter_num", 0) + 1
            best_val_loss = training_state.get("best_val_loss", float("inf"))
            print_rank0(f"从 step {start_iter} 恢复，best_val_loss: {best_val_loss:.4f}")
        else:
            print_rank0(f"警告: 未找到训练状态文件 {state_path}，从头开始")
    else:
        print_rank0(f"\n加载 DPO 模型: {config.dpo_checkpoint}")
        model = RiverValleyModel.from_pretrained(
            config.dpo_checkpoint,
        ).to(device)  # 保持 fp32，使用 autocast 进行混合精度
    print_rank0("训练精度: BF16 (autocast, optimizer states in FP32)")

    if config.use_compile and hasattr(torch, "compile"):
        print_rank0("启用 torch.compile 加速...")
        model = torch.compile(model)

    # 先 compile 再包装 DDP，保持和其他训练脚本一致
    if use_ddp:
        model = wrap_model_ddp(model, local_rank, use_ddp=True)
        print_rank0("已启用 DDP 分布式训练")

    num_params = sum(p.numel() for p in get_raw_model(model).parameters())
    print_rank0(f"模型参数量: {num_params:,}")

    # 创建优化器（使用原始模型参数）
    raw_model = get_raw_model(model)
    decay_params = []
    no_decay_params = []
    for name, param in raw_model.named_parameters():
        if param.requires_grad:
            if "bias" in name or "ln" in name or "embedding" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=config.learning_rate)

    # 恢复 optimizer 状态
    if training_state and training_state.get("optimizer"):
        optimizer.load_state_dict(training_state["optimizer"])
        print_rank0("已恢复 optimizer 状态")

    # 评估函数（带思维链标签加权）
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split, loader in [("train", train_loader), ("test", test_loader)]:
            losses = []
            for i, (X, Y, loss_mask, attn_mask) in enumerate(loader):
                if i >= config.eval_iters:
                    break
                X = X.to(device, non_blocking=True)
                Y = Y.to(device, non_blocking=True)
                loss_mask = loss_mask.to(device, non_blocking=True)
                attn_mask = attn_mask.to(device, non_blocking=True)

                with autocast(device_type=device.type, dtype=torch.bfloat16):
                    outputs = model(input_ids=X, attention_mask=attn_mask, use_cache=False)
                    logits = outputs.logits
                    B, T, C = logits.shape
                    loss_per_token = F.cross_entropy(
                        logits.view(B * T, C), Y.view(B * T), reduction="none"
                    ).view(B, T)

                    weighted_loss_mask = build_reason_loss_mask(Y, loss_mask)
                    loss = (
                        (loss_per_token * weighted_loss_mask).sum()
                        / weighted_loss_mask.sum().clamp_min(1.0)
                    )
                    loss = add_moe_aux_loss(loss, outputs, raw_model.config)

                # 跨进程求平均
                if use_ddp:
                    loss = reduce_mean(loss.clone())

                losses.append(loss.item())
            out[split] = sum(losses) / len(losses) if losses else 0.0
        model.train()
        return out

    # 训练循环
    print_rank0("\n" + "=" * 50)
    if start_iter > 0:
        print_rank0(f"恢复思维链蒸馏训练 (从 step {start_iter} 到 {train_iters}, {config.epochs} epochs)")
    else:
        print_rank0(f"开始思维链蒸馏训练 (共 {train_iters} 步, {config.epochs} epochs)")
    print_rank0("=" * 50)

    train_iter = iter(train_loader)
    current_epoch = 0

    for iter_num in range(start_iter, train_iters):
        # 设置 epoch（用于 DistributedSampler 打乱数据）
        new_epoch = iter_num // steps_per_epoch
        if new_epoch != current_epoch:
            current_epoch = new_epoch
            set_epoch_sampler(train_sampler, current_epoch)

        lr = get_lr(iter_num, train_iters, warmup_iters, config.learning_rate, config.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_num % config.eval_interval == 0:
            losses = estimate_loss()
            print_rank0(f"Step {iter_num:5d} | train loss: {losses['train']:.4f} | test loss: {losses['test']:.4f} | lr: {lr:.2e}")

            if config.use_swanlab and swanlab and is_main:
                swanlab.log({
                    "train_loss": losses["train"],
                    "test_loss": losses["test"],
                    "learning_rate": lr,
                    "iter": iter_num,
                })

            if config.save_best and losses["test"] < best_val_loss:
                best_val_loss = losses["test"]
                barrier()
                save_model_hf(model, tokenizer, f"{config.checkpoint_dir}/best",
                              optimizer.state_dict(), iter_num, best_val_loss, config)
                barrier()

            if iter_num > 0 and iter_num % config.save_interval == 0:
                barrier()
                save_model_hf(model, tokenizer, f"{config.checkpoint_dir}/ckpt_{iter_num}",
                              optimizer.state_dict(), iter_num, best_val_loss, config)
                barrier()

        # 梯度累积
        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(config.accumulation_steps):
            try:
                xb, yb, loss_mask, attn_mask = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                xb, yb, loss_mask, attn_mask = next(train_iter)

            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            loss_mask = loss_mask.to(device, non_blocking=True)
            attn_mask = attn_mask.to(device, non_blocking=True)

            B, T = loss_mask.shape
            weighted_loss_mask = build_reason_loss_mask(yb, loss_mask)

            # 在梯度累积的最后一步才同步梯度
            if use_ddp and micro_step < config.accumulation_steps - 1:
                with model.no_sync():
                    with autocast(device_type=device.type, dtype=torch.bfloat16):
                        outputs = model(input_ids=xb, attention_mask=attn_mask, use_cache=False)
                        logits = outputs.logits
                        _, _, C = logits.shape
                        loss_per_token = F.cross_entropy(
                            logits.view(B * T, C), yb.view(B * T), reduction="none"
                        ).view(B, T)
                        loss = (
                            (loss_per_token * weighted_loss_mask).sum()
                            / weighted_loss_mask.sum().clamp_min(1.0)
                        )
                        loss = add_moe_aux_loss(loss, outputs, raw_model.config)
                        loss = loss / config.accumulation_steps
                    loss.backward()
            else:
                with autocast(device_type=device.type, dtype=torch.bfloat16):
                    outputs = model(input_ids=xb, attention_mask=attn_mask, use_cache=False)
                    logits = outputs.logits
                    _, _, C = logits.shape
                    loss_per_token = F.cross_entropy(
                        logits.view(B * T, C), yb.view(B * T), reduction="none"
                    ).view(B, T)
                    loss = (
                        (loss_per_token * weighted_loss_mask).sum()
                        / weighted_loss_mask.sum().clamp_min(1.0)
                    )
                    loss = add_moe_aux_loss(loss, outputs, raw_model.config)
                    loss = loss / config.accumulation_steps
                loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

    # 保存最终模型
    barrier()
    save_model_hf(model, tokenizer, f"{config.checkpoint_dir}/latest",
                  optimizer.state_dict(), train_iters - 1, best_val_loss, config)
    barrier()

    losses = estimate_loss()
    print_rank0("\n思维链蒸馏训练完成!")
    print_rank0(f"最终 train loss: {losses['train']:.4f} | test loss: {losses['test']:.4f}")
    print_rank0(f"最佳 test loss: {best_val_loss:.4f}")

    if config.use_swanlab and swanlab and is_main:
        swanlab.finish()

    print_rank0(f"\n模型保存位置: {config.checkpoint_dir}/")

    # 清理 DDP
    cleanup_ddp()


if __name__ == "__main__":
    main()
