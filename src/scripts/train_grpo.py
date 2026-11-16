# -*- coding: utf-8 -*-
"""
RiverValley GRPO (Group Relative Policy Optimization) 训练脚本

参考 minimind 项目的 train_grpo.py 实现
使用强化学习方法优化模型的推理能力

DDP 使用方法:
    单卡: python src/scripts/train_grpo.py
    多卡: torchrun --nproc_per_node=4 src/scripts/train_grpo.py
"""

import os
import sys
import re
import math
from contextlib import nullcontext

# 获取项目根目录和 src 目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# 添加 src 到路径
sys.path.insert(0, SRC_DIR)

# 切换到项目根目录，确保相对路径正确
os.chdir(PROJECT_ROOT)

import torch
from torch import optim
from torch.amp import autocast
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from config import GRPOConfig
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
from data import RLAIFDataset
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


def get_generation_eos_ids(tokenizer) -> list:
    """获取生成停止 token，ChatML 下同时识别 <|endoftext|> 和 <|im_end|>。"""
    eos_ids = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)
    im_end_id = tokenizer.convert_tokens_to_ids(IM_END_TOKEN)
    if im_end_id is not None and im_end_id not in eos_ids:
        eos_ids.append(im_end_id)
    return eos_ids


def calculate_rewards(prompts, responses, reward_model, reward_tokenizer, config, device):
    """
    计算奖励分数

    Args:
        prompts: prompt 列表
        responses: 生成的响应列表
        reward_model: reward 模型
        reward_tokenizer: reward 模型的 tokenizer
        config: GRPO 配置
        device: 设备

    Returns:
        rewards: 奖励张量
    """
    def reasoning_model_reward(rewards):
        """计算推理模型的格式奖励"""
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"
        matches_pattern = [re.match(pattern, response, re.S) for response in responses]
        matches_pattern2 = [re.match(pattern2, response, re.S) for response in responses]

        format_rewards = []
        for match_pattern, match_pattern2 in zip(matches_pattern, matches_pattern2):
            if match_pattern or match_pattern2:
                format_rewards.append(0.5)
            else:
                format_rewards.append(0.0)
        rewards += torch.tensor(format_rewards, device=device)

        def mark_num(text):
            reward = 0
            if text.count("<think>") == 1: reward += 0.25
            if text.count("</think>") == 1: reward += 0.25
            if text.count("<answer>") == 1: reward += 0.25
            if text.count("</answer>") == 1: reward += 0.25
            return reward

        mark_rewards = [mark_num(response) for response in responses]
        rewards += torch.tensor(mark_rewards, device=device)
        return rewards

    rewards = torch.zeros(len(responses), device=device)

    # 推理模型格式奖励
    if config.reasoning:
        rewards = reasoning_model_reward(rewards)

    # Reward 模型评分
    with torch.no_grad():
        reward_model_scores = []
        batch_size = len(prompts)
        scale = 3.0

        for i in range(batch_size):
            for j in range(config.num_generations):
                response_idx = i * config.num_generations + j
                response = responses[response_idx]
                prompt = prompts[i]

                # 解析 ChatML 格式的 prompt
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]

                tmp_chat = messages + [{"role": "assistant", "content": response}]
                score = reward_model.get_score(reward_tokenizer, tmp_chat)
                score = max(min(score, scale), -scale)

                # 对推理模型，额外提取 answer 内容评分
                if config.reasoning:
                    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                    if answer_match:
                        answer_content = answer_match.group(1).strip()
                        tmp_chat = messages + [{"role": "assistant", "content": answer_content}]
                        answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
                        answer_score = max(min(answer_score, scale), -scale)
                        score = score * 0.4 + answer_score * 0.6

                reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=device)
        rewards += reward_model_scores

    return rewards


def get_per_token_logps(model, input_ids, completion_ids, attention_mask=None, return_aux_loss=False):
    """
    计算 completion 每个 token 的 log probability

    Args:
        model: 语言模型
        input_ids: prompt + completion 的完整 token ids
        completion_ids: completion token ids
        attention_mask: prompt + completion 的完整注意力掩码

    Returns:
        per_token_logps: 每个 token 的 log probability
    """
    device_type = input_ids.device.type
    with autocast(device_type=device_type, dtype=torch.bfloat16):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs.logits[:, :-1, :].float()
    n_keep = completion_ids.size(1)
    completion_logits = logits[:, -n_keep:, :]
    per_token_logps = torch.gather(
        completion_logits.log_softmax(dim=-1),
        dim=-1,
        index=completion_ids.unsqueeze(-1),
    ).squeeze(-1)
    if return_aux_loss:
        return per_token_logps, getattr(outputs, "aux_loss", None)
    return per_token_logps


def grpo_train_epoch(
    epoch, loader, iters, model, ref_model, reward_model, reward_tokenizer,
    optimizer, scheduler, tokenizer, config, use_ddp=False, is_main=True, wandb=None
):
    """
    GRPO 训练一个 epoch

    Args:
        epoch: 当前 epoch
        loader: 数据加载器
        iters: 总迭代次数
        model: Policy 模型（训练）
        ref_model: Reference 模型（frozen）
        reward_model: Reward 模型（frozen）
        reward_tokenizer: Reward 模型的 tokenizer
        optimizer: 优化器
        scheduler: 学习率调度器
        tokenizer: tokenizer
        config: GRPO 配置
        use_ddp: 是否使用 DDP
        is_main: 是否是主进程
        wandb: wandb 日志
    """
    optimizer.zero_grad(set_to_none=True)
    generation_eos_ids = get_generation_eos_ids(tokenizer)
    stop_token_ids = torch.tensor(generation_eos_ids, dtype=torch.long, device=config.device)

    for step, batch in enumerate(loader, start=1):
        prompts = batch['prompt']  # list[str], length B

        # Tokenize prompts
        prompt_inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
            add_special_tokens=False
        ).to(config.device)

        # 截断 prompt
        if config.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -config.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -config.max_seq_len:]

        # 生成多个响应（使用原始模型进行生成）
        raw_model = get_raw_model(model)
        device_type = torch.device(config.device).type
        with torch.no_grad():
            with autocast(device_type=device_type, dtype=torch.bfloat16):
                outputs = raw_model.generate(
                    **prompt_inputs,
                    max_new_tokens=config.max_gen_len,
                    do_sample=True,
                    temperature=0.8,
                    num_return_sequences=config.num_generations,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=generation_eos_ids,
                )  # [B*num_gen, P+R]

        completion_ids = outputs[:, prompt_inputs["input_ids"].size(1):]  # [B*num_gen, R]
        prompt_attention_mask = prompt_inputs["attention_mask"].repeat_interleave(
            config.num_generations,
            dim=0,
        )
        completion_attention_mask = torch.ones_like(completion_ids)
        full_attention_mask = torch.cat(
            [prompt_attention_mask, completion_attention_mask],
            dim=1,
        )

        # 计算 policy 和 reference 的 log probabilities
        per_token_logps, policy_aux_loss = get_per_token_logps(
            model,
            outputs,
            completion_ids,
            full_attention_mask,
            return_aux_loss=True,
        )  # [B*num_gen, R]
        with torch.no_grad():
            ref_per_token_logps = get_per_token_logps(
                ref_model,
                outputs,
                completion_ids,
                full_attention_mask,
            )  # [B*num_gen, R]

        # 解码生成的响应
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        # 计算奖励
        rewards = calculate_rewards(
            prompts, completions, reward_model, reward_tokenizer, config, config.device
        )  # [B*num_gen]

        # 计算组内相对优势
        grouped_rewards = rewards.view(-1, config.num_generations)  # [B, num_gen]
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(config.num_generations)  # [B*num_gen]
        std_r = grouped_rewards.std(dim=1).repeat_interleave(config.num_generations)  # [B*num_gen]
        advantages = torch.clamp((rewards - mean_r) / (std_r + 1e-4), -10, 10)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # [B*num_gen]

        # 计算 completion mask（到 EOS 为止）
        is_eos = (completion_ids.unsqueeze(-1) == stop_token_ids.view(1, 1, -1)).any(dim=-1)  # [B*num_gen, R]
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=config.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        completion_mask = (
            torch.arange(is_eos.size(1), device=config.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)
        ).int()  # [B*num_gen, R]

        # 计算 KL 散度惩罚
        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1  # [B*num_gen, R]

        # 计算 GRPO Loss
        per_token_loss = -(
            torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
            - config.beta * per_token_kl
        )  # [B*num_gen, R]
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        raw_policy_model = get_raw_model(model)
        if policy_aux_loss is not None and getattr(raw_policy_model.config, "use_moe", False):
            loss = loss + raw_policy_model.config.aux_loss_coef * policy_aux_loss
        loss = loss / config.accumulation_steps

        should_step = (step % config.accumulation_steps == 0) or (step == iters)
        backward_context = model.no_sync() if use_ddp and not should_step else nullcontext()
        with backward_context:
            loss.backward()

        # 梯度累积
        if should_step:
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        # 日志
        if step % config.log_interval == 0 or step == iters:
            policy_loss_val = loss.item() * config.accumulation_steps
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            current_lr = optimizer.param_groups[0]['lr']

            # 跨进程求平均
            if use_ddp:
                avg_reward_val = reduce_mean(rewards.mean().clone()).item()

            print_rank0(f'Epoch: {epoch+1}, Step: {step}/{iters}, '
                  f'Loss: {policy_loss_val:.6f}, Reward: {avg_reward_val:.6f}, '
                  f'Avg Len: {avg_len_val:.2f}, LR: {current_lr:.2e}')

            if wandb and is_main:
                wandb.log({
                    "policy_loss": policy_loss_val,
                    "reward": avg_reward_val,
                    "avg_response_len": avg_len_val,
                    "advantages_mean": advantages.mean().item(),
                    "learning_rate": current_lr,
                })

        # 保存模型
        if step % config.save_interval == 0 or step == iters:
            barrier()
            if is_main:
                model.eval()
                save_path = f"{config.checkpoint_dir}/ckpt_{step}"
                os.makedirs(save_path, exist_ok=True)
                model_to_save = get_raw_model(model)
                model_to_save.save_pretrained(save_path)
                print_rank0(f"  -> 模型已保存: {save_path}")
                model.train()
            barrier()

        # 清理显存
        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps, policy_aux_loss
        del prompt_attention_mask, completion_attention_mask, full_attention_mask
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask


def main(config: GRPOConfig = None):
    """GRPO 训练主函数"""
    if config is None:
        config = GRPOConfig()

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
    print_rank0(f"Reason 模型: {config.reason_checkpoint}")
    print_rank0(f"Reward 模型: {config.reward_model_path}")
    print_rank0(f"梯度累积步数: {config.accumulation_steps}")

    if is_main:
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    # 加载 tokenizer
    tokenizer_base = load_tokenizer(config.bpe_model_path)
    vocab_size = get_vocab_size(tokenizer_base)
    pad_id = get_pad_id(tokenizer_base)
    eos_id = get_eos_id(tokenizer_base)
    im_end_id = get_im_end_id(tokenizer_base)
    print_rank0(f"\nBPE 词汇表大小: {vocab_size}")
    print_rank0(f"生成停止 token ids: eos={eos_id}, im_end={im_end_id}")

    # 创建 HuggingFace tokenizer（用于 generate）
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_base,
        bos_token=IM_START_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        unk_token=None,
        additional_special_tokens=[IM_START_TOKEN, IM_END_TOKEN],
        padding_side="left",
    )
    tokenizer.chat_template = CHAT_TEMPLATE

    # 加载 Policy 模型（训练）
    print_rank0(f"\n加载 Policy 模型: {config.reason_checkpoint}")
    model = RiverValleyModel.from_pretrained(
        config.reason_checkpoint,
        local_files_only=True,
    ).to(device)  # 保持 fp32，使用 autocast 进行混合精度
    model.train()

    # 加载 Reference 模型（frozen）
    print_rank0(f"加载 Reference 模型: {config.reason_checkpoint}")
    ref_model = RiverValleyModel.from_pretrained(
        config.reason_checkpoint,
        local_files_only=True,
    ).to(device)  # 保持 fp32，使用 autocast 进行混合精度
    ref_model.eval()
    ref_model.requires_grad_(False)
    print_rank0("训练精度: BF16 (autocast, optimizer states in FP32)")

    # 加载 Reward 模型（frozen）
    print_rank0(f"加载 Reward 模型: {config.reward_model_path}")
    reward_model = AutoModel.from_pretrained(
        config.reward_model_path,
        dtype=torch.float16,
        trust_remote_code=True,
    ).to(device)
    reward_model.eval()
    reward_model.requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(
        config.reward_model_path,
        trust_remote_code=True,
    )

    # 包装 Policy 模型为 DDP（只有 policy model 需要训练）
    if use_ddp:
        model = wrap_model_ddp(model, local_rank, use_ddp=True)
        print_rank0("已启用 DDP 分布式训练")

    num_params = sum(p.numel() for p in get_raw_model(model).parameters())
    print_rank0(f"Policy 模型参数量: {num_params:,}")

    # 创建数据集
    train_ds = RLAIFDataset(config.data_path, tokenizer_base, max_length=config.max_seq_len + config.max_gen_len)

    # 设置随机种子
    torch.manual_seed(config.random_seed + local_rank)

    # 创建分布式采样器
    train_sampler = create_distributed_sampler(train_ds, shuffle=True, seed=config.random_seed)

    # 创建 DataLoader
    loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )
    iters = len(loader)

    # 计算有效 batch size
    effective_batch_size = config.batch_size * world_size * config.accumulation_steps
    print_rank0(f"有效 batch size: {effective_batch_size} (per GPU: {config.batch_size}, GPUs: {world_size}, accum: {config.accumulation_steps})")

    # 创建优化器和调度器（使用原始模型参数）
    raw_model = get_raw_model(model)
    optimizer = optim.AdamW(raw_model.parameters(), lr=config.learning_rate)
    total_optimizer_steps = math.ceil(iters / config.accumulation_steps) * config.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=config.learning_rate / 10)

    # 初始化 swanlab（只在主进程）
    global swanlab
    if config.use_swanlab and is_main:
        import swanlab as _swanlab
        swanlab = _swanlab

        run_name = config.swanlab_run_name or f"GRPO-{config.epochs}epochs"
        swanlab.init(
            project=config.swanlab_project,
            experiment_name=run_name,
            config=config.to_dict(),
        )

    # 训练
    print_rank0("\n" + "=" * 50)
    print_rank0(f"开始 GRPO 训练 (共 {iters} 步, {config.epochs} epochs)")
    print_rank0("=" * 50)

    for epoch in range(config.epochs):
        # 设置 epoch（用于 DistributedSampler 打乱数据）
        set_epoch_sampler(train_sampler, epoch)

        grpo_train_epoch(
            epoch, loader, iters, model, ref_model, reward_model, reward_tokenizer,
            optimizer, scheduler, tokenizer, config, use_ddp, is_main, swanlab
        )

    # 保存最终模型
    barrier()
    if is_main:
        save_path = f"{config.checkpoint_dir}/latest"
        os.makedirs(save_path, exist_ok=True)
        model_to_save = get_raw_model(model)
        model_to_save.save_pretrained(save_path)
    barrier()

    print_rank0("\nGRPO 训练完成!")
    print_rank0(f"模型保存位置: {config.checkpoint_dir}/")

    if config.use_swanlab and swanlab and is_main:
        swanlab.finish()

    # 清理 DDP
    cleanup_ddp()


if __name__ == "__main__":
    main()
