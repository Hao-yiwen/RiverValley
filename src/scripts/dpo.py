# -*- coding: utf-8 -*-
"""
RiverValley DPO 训练脚本

Direct Preference Optimization (DPO) 训练
从 SFT 模型开始进行偏好对齐

DDP 使用方法:
    单卡: python src/scripts/dpo.py
    多卡: torchrun --nproc_per_node=4 src/scripts/dpo.py
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

from config import RiverValleyConfig, DPOConfig, DEFAULT_MAX_SEQ_LEN
from model import RiverValleyModel
from tokenizer import (
    load_tokenizer,
    encode,
    decode,
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
from data import DPODataset
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


def add_moe_aux_loss(loss, model_config, *outputs):
    """在 DPO 主损失外追加 policy 模型的 MoE 路由负载均衡损失。"""
    if not getattr(model_config, "use_moe", False):
        return loss
    aux_losses = [
        getattr(output, "aux_loss", None)
        for output in outputs
        if getattr(output, "aux_loss", None) is not None
    ]
    if not aux_losses:
        return loss
    aux_loss = torch.stack([aux.float() for aux in aux_losses]).mean()
    return loss + model_config.aux_loss_coef * aux_loss.to(loss.dtype)


def save_model_hf(model, tokenizer, save_path, config=None):
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


def compute_log_probs(logits: torch.Tensor, labels: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """
    计算序列的 log probabilities（总和，符合 DPO 论文）

    Args:
        logits: 模型输出 logits (B, T, V)
        labels: 目标 token ids (B, T)
        loss_mask: 损失掩码 (B, T)

    Returns:
        log_probs: 每个样本的 log probability 总和 (B,)
    """
    # 计算每个 token 的 log probability
    log_probs_all = F.log_softmax(logits.float(), dim=-1)

    # 获取目标 token 的 log probability
    # labels: (B, T) -> (B, T, 1)
    labels_expanded = labels.unsqueeze(-1)
    log_probs_selected = log_probs_all.gather(dim=-1, index=labels_expanded).squeeze(-1)  # (B, T)

    # 应用 loss mask 并计算总和（不做长度归一化，符合 DPO 论文）
    masked_log_probs = log_probs_selected * loss_mask
    sum_log_probs = masked_log_probs.sum(dim=-1)  # (B,)

    return sum_log_probs


def compute_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
) -> tuple:
    """
    计算 DPO loss

    Args:
        policy_chosen_logps: policy model 对 chosen 的 log probs (B,)
        policy_rejected_logps: policy model 对 rejected 的 log probs (B,)
        ref_chosen_logps: reference model 对 chosen 的 log probs (B,)
        ref_rejected_logps: reference model 对 rejected 的 log probs (B,)
        beta: DPO 温度参数

    Returns:
        loss: DPO loss
        chosen_rewards: chosen 样本的隐式 reward
        rejected_rewards: rejected 样本的隐式 reward
    """
    # 计算 log probability ratios
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps

    # 计算隐式 rewards
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    # DPO loss: -log(sigmoid(beta * (log(pi_c/pi_r) - log(ref_c/ref_r))))
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits).mean()

    return loss, chosen_rewards.mean(), rejected_rewards.mean()


def main(config: DPOConfig = None):
    """DPO 训练主函数"""
    if config is None:
        config = DPOConfig()

    # 初始化 DDP
    local_rank, world_size, is_main = setup_ddp()
    use_ddp = world_size > 1

    # 设置设备
    if use_ddp:
        device = get_device(local_rank)
        config.device = str(device)
    else:
        device = torch.device(config.device)

    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp_dtype = torch.bfloat16

    print_rank0(f"Using device: {config.device}")
    print_rank0(f"World size: {world_size}")
    print_rank0(f"数据文件: {config.data_path}")
    print_rank0(f"SFT 模型: {config.sft_checkpoint}")
    print_rank0(f"DPO beta: {config.beta}")
    print_rank0(f"梯度累积步数: {config.accumulation_steps}")

    if is_main:
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    # 加载 tokenizer
    tokenizer = load_tokenizer(config.bpe_model_path)
    vocab_size = get_vocab_size(tokenizer)
    pad_id = get_pad_id(tokenizer)
    print_rank0(f"\nBPE 词汇表大小: {vocab_size}")

    # 创建数据集并打乱后再划分
    dataset = DPODataset(config.data_path, tokenizer, config.max_seq_len, pad_id)
    if len(dataset.samples) == 0:
        raise ValueError(f"数据集为空: {config.data_path}")

    generator = torch.Generator()
    generator.manual_seed(config.random_seed)
    indices = torch.randperm(len(dataset.samples), generator=generator).tolist()
    shuffled_samples = [dataset.samples[i] for i in indices]

    # 分割训练集和测试集
    n = int(0.9 * len(shuffled_samples))
    train_samples = shuffled_samples[:n]
    test_samples = shuffled_samples[n:]
    print_rank0(f"训练集: {len(train_samples)} 条")
    print_rank0(f"测试集: {len(test_samples)} 条")

    # 计算训练步数（考虑 DDP 和梯度累积）
    global_batch = config.batch_size * world_size * config.accumulation_steps
    effective_batch_size = global_batch
    if len(train_samples) < global_batch:
        raise ValueError(
            f"训练样本数不足以组成一个批次: {len(train_samples)} 条, "
            f"需要至少 {global_batch} (batch_size={config.batch_size}, world_size={world_size}, "
            f"accumulation_steps={config.accumulation_steps})"
        )
    steps_per_epoch = len(train_samples) // global_batch
    if steps_per_epoch == 0:
        raise ValueError("每个 epoch 的训练步数为 0，请减小 batch_size/accumulation_steps 或增大数据量。")
    train_iters = steps_per_epoch * config.epochs
    warmup_iters = int(train_iters * config.warmup_ratio)
    print_rank0(f"每 epoch 步数: {steps_per_epoch}")
    print_rank0(f"总训练步数: {train_iters} ({config.epochs} epochs)")
    print_rank0(f"Warmup 步数: {warmup_iters} ({config.warmup_ratio * 100:.0f}%)")
    print_rank0(f"有效 batch size: {effective_batch_size} (per GPU: {config.batch_size}, GPUs: {world_size}, accum: {config.accumulation_steps})")

    # 创建 DataLoader
    train_dataset = DPODataset.__new__(DPODataset)
    train_dataset.samples = train_samples
    train_dataset.block_size = config.max_seq_len
    train_dataset.pad_id = pad_id
    train_dataset.tokenizer = tokenizer

    test_dataset = DPODataset.__new__(DPODataset)
    test_dataset.samples = test_samples
    test_dataset.block_size = config.max_seq_len
    test_dataset.pad_id = pad_id
    test_dataset.tokenizer = tokenizer

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

    # 初始化 swanlab（只在主进程）
    global swanlab
    if config.use_swanlab and is_main:
        import swanlab as _swanlab
        swanlab = _swanlab

        run_name = config.swanlab_run_name or f"DPO-beta{config.beta}"
        swanlab.init(
            project=config.swanlab_project,
            experiment_name=run_name,
            config=config.to_dict(),
        )

    # 加载 policy model（要训练的）
    print_rank0(f"\n加载 Policy Model: {config.sft_checkpoint}")
    policy_model = RiverValleyModel.from_pretrained(
        config.sft_checkpoint,
    ).to(device)  # 保持 fp32，使用 autocast 进行混合精度

    # 加载 reference model（冻结的）
    print_rank0(f"加载 Reference Model: {config.sft_checkpoint}")
    ref_model = RiverValleyModel.from_pretrained(
        config.sft_checkpoint,
        dtype=amp_dtype if device.type == "cuda" else torch.float32,
    ).to(device)  # 使用较低精度减少显存占用
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    amp_dtype_name = "BF16" if amp_dtype == torch.bfloat16 else "FP16"
    print_rank0(f"训练精度: {amp_dtype_name} (autocast, optimizer states in FP32)")

    # 先 compile 再包装 DDP（PyTorch 不支持 compile DDP wrapper）
    if config.use_compile and hasattr(torch, "compile"):
        print_rank0("启用 torch.compile 加速...")
        policy_model = torch.compile(policy_model)

    # 包装 policy model 为 DDP（ref_model 不需要包装，因为它是冻结的）
    if use_ddp:
        policy_model = wrap_model_ddp(policy_model, local_rank, use_ddp=True)
        print_rank0("已启用 DDP 分布式训练")

    num_params = sum(p.numel() for p in get_raw_model(policy_model).parameters())
    print_rank0(f"模型参数量: {num_params:,}")

    # 创建优化器（使用原始模型参数）
    raw_policy_model = get_raw_model(policy_model)
    decay_params = []
    no_decay_params = []
    for name, param in raw_policy_model.named_parameters():
        if param.requires_grad:
            if "bias" in name or "ln" in name or "embedding" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=config.learning_rate)

    # 评估函数
    @torch.no_grad()
    def estimate_loss():
        out = {}
        policy_model.eval()
        for split, loader in [("train", train_loader), ("test", test_loader)]:
            losses = []
            accuracies = []
            for i, batch in enumerate(loader):
                if i >= config.eval_iters:
                    break

                # 解包数据
                (chosen_x, chosen_y, chosen_lm, chosen_am,
                 rejected_x, rejected_y, rejected_lm, rejected_am) = batch

                chosen_x = chosen_x.to(device, non_blocking=True)
                chosen_y = chosen_y.to(device, non_blocking=True)
                chosen_lm = chosen_lm.to(device, non_blocking=True)
                chosen_am = chosen_am.to(device, non_blocking=True)
                rejected_x = rejected_x.to(device, non_blocking=True)
                rejected_y = rejected_y.to(device, non_blocking=True)
                rejected_lm = rejected_lm.to(device, non_blocking=True)
                rejected_am = rejected_am.to(device, non_blocking=True)

                with autocast(device_type=device.type, dtype=amp_dtype):
                    # Policy model forward
                    policy_chosen_out = policy_model(input_ids=chosen_x, attention_mask=chosen_am, use_cache=False)
                    policy_rejected_out = policy_model(input_ids=rejected_x, attention_mask=rejected_am, use_cache=False)
                    policy_chosen_logps = compute_log_probs(policy_chosen_out.logits, chosen_y, chosen_lm)
                    policy_rejected_logps = compute_log_probs(policy_rejected_out.logits, rejected_y, rejected_lm)

                    # Reference model forward
                    ref_chosen_out = ref_model(input_ids=chosen_x, attention_mask=chosen_am, use_cache=False)
                    ref_rejected_out = ref_model(input_ids=rejected_x, attention_mask=rejected_am, use_cache=False)
                    ref_chosen_logps = compute_log_probs(ref_chosen_out.logits, chosen_y, chosen_lm)
                    ref_rejected_logps = compute_log_probs(ref_rejected_out.logits, rejected_y, rejected_lm)

                    # 计算 DPO loss
                    loss, _, _ = compute_dpo_loss(
                        policy_chosen_logps, policy_rejected_logps,
                        ref_chosen_logps, ref_rejected_logps,
                        config.beta
                    )
                    loss = add_moe_aux_loss(
                        loss,
                        raw_policy_model.config,
                        policy_chosen_out,
                        policy_rejected_out,
                    )

                # 跨进程求平均
                if use_ddp:
                    loss = reduce_mean(loss.clone())

                losses.append(loss.item())

                # 计算准确率（chosen 的 reward 是否大于 rejected）
                chosen_rewards = config.beta * (policy_chosen_logps - ref_chosen_logps)
                rejected_rewards = config.beta * (policy_rejected_logps - ref_rejected_logps)
                accuracy = (chosen_rewards > rejected_rewards).float().mean()

                if use_ddp:
                    accuracy = reduce_mean(accuracy.clone())

                accuracies.append(accuracy.item())

            out[split] = {
                "loss": sum(losses) / len(losses) if losses else 0.0,
                "accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
            }
        policy_model.train()
        return out

    best_val_loss = float("inf")

    # 训练循环
    print_rank0("\n" + "=" * 50)
    print_rank0(f"开始 DPO 训练 (共 {train_iters} 步, {config.epochs} epochs)")
    print_rank0("=" * 50)

    train_iter = iter(train_loader)
    current_epoch = 0

    for iter_num in range(train_iters):
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
            train_loss = losses["train"]["loss"]
            train_acc = losses["train"]["accuracy"]
            test_loss = losses["test"]["loss"]
            test_acc = losses["test"]["accuracy"]

            print_rank0(f"Step {iter_num:5d} | train loss: {train_loss:.4f} acc: {train_acc:.2%} | "
                  f"test loss: {test_loss:.4f} acc: {test_acc:.2%} | lr: {lr:.2e}")

            if config.use_swanlab and swanlab and is_main:
                swanlab.log({
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "test_loss": test_loss,
                    "test_accuracy": test_acc,
                    "learning_rate": lr,
                    "iter": iter_num,
                })

            if config.save_best and test_loss < best_val_loss:
                best_val_loss = test_loss
                barrier()
                save_model_hf(policy_model, tokenizer, f"{config.checkpoint_dir}/best", config)
                barrier()

            if iter_num > 0 and iter_num % config.save_interval == 0:
                barrier()
                save_model_hf(policy_model, tokenizer, f"{config.checkpoint_dir}/ckpt_{iter_num}", config)
                barrier()

        # 梯度累积
        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(config.accumulation_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            # 解包数据
            (chosen_x, chosen_y, chosen_lm, chosen_am,
             rejected_x, rejected_y, rejected_lm, rejected_am) = batch

            chosen_x = chosen_x.to(device, non_blocking=True)
            chosen_y = chosen_y.to(device, non_blocking=True)
            chosen_lm = chosen_lm.to(device, non_blocking=True)
            chosen_am = chosen_am.to(device, non_blocking=True)
            rejected_x = rejected_x.to(device, non_blocking=True)
            rejected_y = rejected_y.to(device, non_blocking=True)
            rejected_lm = rejected_lm.to(device, non_blocking=True)
            rejected_am = rejected_am.to(device, non_blocking=True)

            # Reference model forward (no grad)
            with torch.no_grad():
                with autocast(device_type=device.type, dtype=amp_dtype):
                    ref_chosen_out = ref_model(input_ids=chosen_x, attention_mask=chosen_am, use_cache=False)
                    ref_rejected_out = ref_model(input_ids=rejected_x, attention_mask=rejected_am, use_cache=False)
                    ref_chosen_logps = compute_log_probs(ref_chosen_out.logits, chosen_y, chosen_lm)
                    ref_rejected_logps = compute_log_probs(ref_rejected_out.logits, rejected_y, rejected_lm)

            # 在梯度累积的最后一步才同步梯度
            if use_ddp and micro_step < config.accumulation_steps - 1:
                with policy_model.no_sync():
                    with autocast(device_type=device.type, dtype=amp_dtype):
                        # Policy model forward
                        policy_chosen_out = policy_model(input_ids=chosen_x, attention_mask=chosen_am, use_cache=False)
                        policy_rejected_out = policy_model(input_ids=rejected_x, attention_mask=rejected_am, use_cache=False)
                        policy_chosen_logps = compute_log_probs(policy_chosen_out.logits, chosen_y, chosen_lm)
                        policy_rejected_logps = compute_log_probs(policy_rejected_out.logits, rejected_y, rejected_lm)

                        # 计算 DPO loss
                        loss, _, _ = compute_dpo_loss(
                            policy_chosen_logps, policy_rejected_logps,
                            ref_chosen_logps, ref_rejected_logps,
                            config.beta
                        )
                        loss = add_moe_aux_loss(
                            loss,
                            raw_policy_model.config,
                            policy_chosen_out,
                            policy_rejected_out,
                        )
                        loss = loss / config.accumulation_steps
                    loss.backward()
            else:
                with autocast(device_type=device.type, dtype=amp_dtype):
                    # Policy model forward
                    policy_chosen_out = policy_model(input_ids=chosen_x, attention_mask=chosen_am, use_cache=False)
                    policy_rejected_out = policy_model(input_ids=rejected_x, attention_mask=rejected_am, use_cache=False)
                    policy_chosen_logps = compute_log_probs(policy_chosen_out.logits, chosen_y, chosen_lm)
                    policy_rejected_logps = compute_log_probs(policy_rejected_out.logits, rejected_y, rejected_lm)

                    # 计算 DPO loss
                    loss, _, _ = compute_dpo_loss(
                        policy_chosen_logps, policy_rejected_logps,
                        ref_chosen_logps, ref_rejected_logps,
                        config.beta
                    )
                    loss = add_moe_aux_loss(
                        loss,
                        raw_policy_model.config,
                        policy_chosen_out,
                        policy_rejected_out,
                    )
                    loss = loss / config.accumulation_steps
                loss.backward()

        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), config.grad_clip)
        optimizer.step()

    # 保存最终模型
    barrier()
    save_model_hf(policy_model, tokenizer, f"{config.checkpoint_dir}/latest", config)
    barrier()

    losses = estimate_loss()
    print_rank0("\nDPO 训练完成!")
    print_rank0(f"最终 train loss: {losses['train']['loss']:.4f} acc: {losses['train']['accuracy']:.2%}")
    print_rank0(f"最终 test loss: {losses['test']['loss']:.4f} acc: {losses['test']['accuracy']:.2%}")
    print_rank0(f"最佳 test loss: {best_val_loss:.4f}")

    if config.use_swanlab and swanlab and is_main:
        swanlab.finish()

    print_rank0(f"\n模型保存位置: {config.checkpoint_dir}/")

    # 清理 DDP
    cleanup_ddp()


if __name__ == "__main__":
    main()
