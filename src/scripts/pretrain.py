# -*- coding: utf-8 -*-
"""
RiverValley 预训练脚本

预训练语言模型

DDP 使用方法:
    单卡: python src/scripts/pretrain.py
    多卡: torchrun --nproc_per_node=4 src/scripts/pretrain.py
"""

import os
import sys
import time
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
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import PreTrainedTokenizerFast, GenerationConfig

from config import RiverValleyConfig, PretrainConfig, DEFAULT_MAX_SEQ_LEN
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
from data.dataset import MemmapDataset
from training_utils import (ResumableBatchSampler, collect_rng_states, restore_rng_states,
                            resume_signature, validate_resume, evaluate_loss, train_update, validate_checkpoint_tokenizer,
                            write_checkpoint_directory, resolve_checkpoint_path, validate_training_options,
                            prune_interval_checkpoints, append_metrics)
from ddp_utils import get_rank
from training_runtime import TrainingStopper
from ddp_utils import (
    setup_ddp,
    cleanup_ddp,
    get_device,
    wrap_model_ddp,
    get_raw_model,
    print_rank0,
    is_main_process,
    barrier,
)

# swanlab 延迟导入
swanlab = None


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _save_model_hf_contents(
    model,
    tokenizer,
    save_path,
    optimizer_state=None,
    iter_num=None,
    best_val_loss=None,
    config=None,
    next_iter=None,
    rng_states=None,
):
    """保存模型为 HuggingFace 格式（只在主进程保存）"""
    rng_states = collect_rng_states() if rng_states is None else rng_states
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
            "next_iter": next_iter,
            "best_val_loss": best_val_loss,
            "has_optimizer_state": optimizer_state is not None,
            "rng_states": rng_states,
            "resume_signature": getattr(config, "_resume_signature", None),
            "stage_origin": getattr(config, "_stage_origin", None),
        }
        torch.save(training_state, os.path.join(save_path, "training_state.pt"))

    state_note = "含 optimizer state" if optimizer_state is not None else "权重/元数据"
    print(f"[{now_str()}]   -> 模型已保存: {save_path} ({state_note})")


def save_model_hf(model, tokenizer, save_path, optimizer_state=None, iter_num=None,
                  best_val_loss=None, config=None, next_iter=None):
    """所有 rank 收集续训状态，主进程用完整目录替换避免混合新旧权重和优化器。"""
    rng_states = collect_rng_states()
    if not is_main_process():
        return
    write_checkpoint_directory(save_path, lambda stage: _save_model_hf_contents(
        model, tokenizer, stage, optimizer_state, iter_num, best_val_loss, config,
        next_iter=next_iter, rng_states=rng_states))
    print_rank0(f"checkpoint 已提交: {save_path}, next_iter={next_iter}")


def load_model_hf(load_path, device, dtype=None):
    """从 HuggingFace 格式加载模型（默认 fp32 用于训练）"""
    print(f"加载模型: {load_path}")

    model = RiverValleyModel.from_pretrained(load_path, dtype=dtype or torch.float32).to(device)

    training_state_path = os.path.join(load_path, "training_state.pt")
    training_state = None
    if os.path.exists(training_state_path):
        try:
            training_state = torch.load(training_state_path, map_location="cpu", weights_only=False)
        except TypeError:
            training_state = torch.load(training_state_path, map_location="cpu")
        if training_state.get("next_iter") is not None:
            print(f"  -> 恢复训练状态: next_iter={training_state.get('next_iter')}")
        else:
            print(f"  -> 恢复训练状态: iter={training_state.get('iter_num', 0)}")

    return model, training_state


def should_save_optimizer_state(config: PretrainConfig, checkpoint_kind: str) -> bool:
    """按 checkpoint 用途决定是否保存体积很大的 optimizer state。"""
    if not getattr(config, "save_optimizer_state", True):
        return False
    if checkpoint_kind == "latest":
        return True
    if checkpoint_kind == "best":
        return bool(getattr(config, "save_optimizer_state_for_best", False))
    if checkpoint_kind == "interval":
        return bool(getattr(config, "save_optimizer_state_for_interval", False))
    raise ValueError(f"未知 checkpoint 类型: {checkpoint_kind}")


def get_optimizer_state_for_save(optimizer, use_ddp: bool, is_main: bool):
    """获取用于保存的优化器状态（ZeroRedundancyOptimizer 需先 consolidate）"""
    if use_ddp and hasattr(optimizer, "consolidate_state_dict"):
        try:
            optimizer.consolidate_state_dict(to=0)
        except TypeError:
            optimizer.consolidate_state_dict()
        if not is_main:
            return None
    if not is_main:
        return None
    return optimizer.state_dict()


def get_lr(iter_num: int, train_iters: int, warmup_iters: int, learning_rate: float, min_lr: float) -> float:
    """余弦退火学习率调度"""
    if iter_num < warmup_iters:
        return learning_rate * (iter_num + 1) / warmup_iters

    decay_iters = max(1, train_iters - warmup_iters)
    iter_after_warmup = iter_num - warmup_iters
    decay_ratio = min(1.0, max(0.0, iter_after_warmup / decay_iters))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coeff * (learning_rate - min_lr)


def select_data_fraction_indices(total_samples: int, data_fraction: float, random_seed: int) -> list:
    """按固定随机种子抽取 data_fraction 子集，避免只取 memmap 前 N 条造成来源偏差。"""
    if total_samples <= 0:
        raise ValueError("数据集为空，无法抽样")
    if data_fraction <= 0 or data_fraction > 1:
        raise ValueError(f"data_fraction 必须在 (0, 1] 范围内，当前为 {data_fraction}")

    used_samples = int(data_fraction * total_samples)
    used_samples = max(1, min(total_samples, used_samples))
    generator = torch.Generator()
    generator.manual_seed(random_seed)
    return torch.randperm(total_samples, generator=generator).tolist()[:used_samples]


def can_skip_attention_mask(meta: dict) -> bool:
    """判断 packed memmap 是否可以跳过 attention_mask，走更快的 causal attention 路径。"""
    return meta.get("packed") is True and meta.get("partial_blocks") == 0


def compile_model_if_needed(model, config: PretrainConfig):
    """按配置包装 torch.compile，统一正式训练和 benchmark 的 compile 模式。"""
    if config.use_compile and hasattr(torch, "compile"):
        print_rank0(f"启用 torch.compile 加速 (mode={config.compile_mode})...")
        return torch.compile(model, mode=config.compile_mode)
    return model


def run_compile_benchmark(model_config, train_loader, device, config, skip_attention_mask, use_ddp, local_rank, world_size):
    """对比未 compile 与 compile 的训练吞吐，跑完后直接返回，不保存 checkpoint。"""
    if config.compile_benchmark_steps <= 0:
        raise ValueError("compile_benchmark_steps 必须大于 0")
    if config.compile_benchmark_warmup_steps < 0:
        raise ValueError("compile_benchmark_warmup_steps 不能为负数")

    candidates = [False]
    if hasattr(torch, "compile"):
        candidates.append(True)
    else:
        print_rank0("当前 PyTorch 不支持 torch.compile，仅 benchmark eager 模式")

    results = []
    total_steps = config.compile_benchmark_warmup_steps + config.compile_benchmark_steps

    for compile_enabled in candidates:
        label = f"compile={compile_enabled}"
        print_rank0(f"\n[{now_str()}] 开始预训练吞吐 benchmark: {label}")
        model = RiverValleyModel(model_config).to(device)
        if compile_enabled:
            print_rank0(f"benchmark 使用 torch.compile(mode={config.compile_mode})")
            model = torch.compile(model, mode=config.compile_mode)
        if use_ddp:
            model = wrap_model_ddp(model, local_rank, use_ddp=True)

        raw_model = get_raw_model(model)
        groups = [{"params": [p for p in raw_model.parameters() if p.requires_grad and p.ndim >= 2], "weight_decay": config.weight_decay},
                  {"params": [p for p in raw_model.parameters() if p.requires_grad and p.ndim < 2], "weight_decay": 0.0}]
        optimizer_options = {"lr": config.learning_rate, "betas": (config.adam_beta1, config.adam_beta2), "eps": config.adam_eps}
        if config.adamw_fused and device.type == "cuda" and not use_ddp:
            optimizer_options["fused"] = True
        else:
            optimizer_options["foreach"] = config.adamw_foreach
        if use_ddp:
            from torch.distributed.optim import ZeroRedundancyOptimizer
            optimizer = ZeroRedundancyOptimizer(groups, optimizer_class=torch.optim.AdamW, **optimizer_options)
        else:
            optimizer = torch.optim.AdamW(groups, **optimizer_options)
        train_iter = iter(train_loader)
        model.train()
        measured_steps = 0
        start_time = None
        failed = None

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        try:
            for step in range(total_steps):
                batches = []
                for _ in range(config.accumulation_steps):
                    try:
                        batches.append(next(train_iter))
                    except StopIteration:
                        train_iter = iter(train_loader)
                        batches.append(next(train_iter))
                train_update(model, optimizer, batches, device, config.grad_clip, skip_attention_mask)

                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                if step + 1 == config.compile_benchmark_warmup_steps:
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    start_time = time.time()
                    measured_steps = 0
                elif step + 1 > config.compile_benchmark_warmup_steps:
                    if start_time is None:
                        start_time = time.time()
                    measured_steps += 1
        except RuntimeError as exc:
            failed = str(exc)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        else:
            peak_memory_gb = 0.0

        elapsed = max(time.time() - start_time, 1e-9) if start_time is not None and measured_steps else 0.0
        tokens_per_sec = (
            measured_steps * config.batch_size * world_size * config.accumulation_steps * config.max_seq_len / elapsed
            if elapsed > 0 else 0.0
        )
        if use_ddp:
            barrier()

        if failed:
            print_rank0(f"[{now_str()}] Benchmark {label} 失败: {failed}")
        else:
            print_rank0(
                f"[{now_str()}] Benchmark {label}: "
                f"{tokens_per_sec:.0f} tokens/s, peak_mem={peak_memory_gb:.2f} GB, "
                f"measured_steps={measured_steps}"
            )
        results.append((label, tokens_per_sec, peak_memory_gb, failed))

        del model, raw_model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if use_ddp:
            barrier()

    print_rank0("\n吞吐 benchmark 汇总:")
    for label, tokens_per_sec, peak_memory_gb, failed in results:
        status = "FAILED" if failed else "OK"
        print_rank0(f"  {label}: {status}, {tokens_per_sec:.0f} tokens/s, peak_mem={peak_memory_gb:.2f} GB")


def main(config: PretrainConfig = None):
    """预训练主函数"""
    if config is None:
        config = PretrainConfig()
    validate_training_options(config)
    if config.new_stage_from_complete and (not config.resume or config.resume_weights_only):
        raise ValueError('新阶段迁移必须加载完整状态，不能同时使用仅加载权重模式')
    if config.keep_interval_checkpoints < 0:
        raise ValueError("keep_interval_checkpoints 不能为负数")

    if min(config.batch_size, config.accumulation_steps, config.epochs,
           config.eval_iters, config.eval_interval, config.save_interval) <= 0:
        raise ValueError("批量、累积步数、epoch 和评估／保存间隔必须为正数")

    # 初始化 DDP
    local_rank, world_size, is_main = setup_ddp()
    use_ddp = world_size > 1

    # 设置设备
    if use_ddp:
        device = get_device(local_rank)
        config.device = str(device)
    else:
        device = torch.device(config.device)

    # 启用 TF32 (Ampere/Hopper/Blackwell 必须开启)
    torch.set_float32_matmul_precision('high')

    print_rank0(f"Using device: {config.device}")
    print_rank0(f"World size: {world_size}")
    print_rank0(f"梯度累积步数: {config.accumulation_steps}")

    if is_main:
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    # 加载 memmap 数据
    memmap_path = os.path.join(config.memmap_dir, "train.bin")
    meta_path = os.path.join(config.memmap_dir, "meta.json")

    if not os.path.exists(memmap_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(f"memmap 数据不存在: {config.memmap_dir}\n请先运行 datasets/prepare_memmap.py 生成数据")

    print_rank0(f"memmap 数据目录: {config.memmap_dir}")
    print_rank0("正在加载 memmap 预训练数据...")

    # 加载 tokenizer。预训练数据已离线 tokenize，训练时必须使用同一个 tokenizer。
    if not os.path.exists(config.bpe_model_path):
        raise FileNotFoundError(
            f"BPE tokenizer 不存在: {config.bpe_model_path}\n"
            "请先运行 src/scripts/train_tokenizer.py 生成 tokenizer。"
        )
    tokenizer = load_tokenizer(config.bpe_model_path)
    vocab_size = get_vocab_size(tokenizer)
    pad_id = get_pad_id(tokenizer)
    print_rank0(f"BPE 词汇表大小: {vocab_size}")

    # 创建 memmap 数据集
    full_dataset = MemmapDataset(memmap_path, meta_path, pad_id)
    import hashlib
    from pathlib import Path
    tokenizer_sha256 = hashlib.sha256(Path(config.bpe_model_path).read_bytes()).hexdigest()
    recorded_sha256 = full_dataset.meta.get("tokenizer_sha256")
    if recorded_sha256 and recorded_sha256 != tokenizer_sha256:
        raise ValueError("memmap 与当前 tokenizer 的 SHA256 不一致，即使词表大小相同也不能混用")
    meta_vocab_size = full_dataset.meta.get("vocab_size")
    if meta_vocab_size is not None and meta_vocab_size != vocab_size:
        raise ValueError(
            f"memmap 词表大小 ({meta_vocab_size}) 与 tokenizer ({vocab_size}) 不一致，"
            "请用当前 tokenizer 重新生成 memmap。"
        )
    meta_pad_id = full_dataset.meta.get("pad_id")
    if meta_pad_id is not None and meta_pad_id != pad_id:
        raise ValueError(
            f"memmap pad_id ({meta_pad_id}) 与 tokenizer ({pad_id}) 不一致，"
            "请用当前 tokenizer 重新生成 memmap。"
        )
    train_seq_len = full_dataset.meta.get("train_seq_len", full_dataset.seq_len - 1)
    if train_seq_len != config.max_seq_len:
        raise ValueError(
            f"memmap 训练序列长度 ({train_seq_len}) 与 PretrainConfig.max_seq_len "
            f"({config.max_seq_len}) 不一致。请统一 prepare_memmap.py --seq_len 和训练配置。"
        )
    skip_attention_mask = can_skip_attention_mask(full_dataset.meta)
    if skip_attention_mask:
        print_rank0("packed memmap 无 partial block，训练时跳过 attention_mask 以使用更快的 causal attention 路径")
    else:
        packed = full_dataset.meta.get("packed")
        partial_blocks = full_dataset.meta.get("partial_blocks")
        message = (
            f"memmap 未满足 packed 快路径: packed={packed}, partial_blocks={partial_blocks}。"
            "正式预训练建议重新生成 packed 且无 partial block 的 memmap。"
        )
        if config.require_packed_fast_path:
            raise ValueError(message)
        print_rank0(f"警告: {message}")

    # 先按固定随机种子抽取指定比例的数据，避免只取 memmap 前 N 条造成来源偏差
    total_samples = len(full_dataset)
    indices = select_data_fraction_indices(total_samples, config.data_fraction, config.random_seed)
    used_samples = len(indices)
    if used_samples < 2:
        raise ValueError(f"抽样后样本数过少: {used_samples}，请增大 data_fraction 或检查 memmap。")

    # 分割训练集和测试集 (9:1)
    n = int(0.9 * used_samples)

    from torch.utils.data import Subset
    train_dataset = Subset(full_dataset, indices[:n])
    test_dataset = Subset(full_dataset, indices[n:used_samples])
    if config.val_memmap_dir:
        # 文档先划分再打包，避免同一篇长文的不同块同时出现在训练和验证中。
        train_dataset = Subset(full_dataset, indices)
        test_dataset = MemmapDataset(os.path.join(config.val_memmap_dir, "train.bin"),
                                     os.path.join(config.val_memmap_dir, "meta.json"), pad_id)
        for key in ("vocab_size", "pad_id", "train_seq_len", "tokenizer_sha256"):
            if test_dataset.meta.get(key) != full_dataset.meta.get(key):
                raise ValueError(f"独立验证集与训练集的 {key} 不一致")
        if test_dataset.seq_len != full_dataset.seq_len or not can_skip_attention_mask(test_dataset.meta):
            raise ValueError("独立验证集必须与训练集同长度，且使用无 padding 的 packed 格式")
        print_rank0(f"使用按文档隔离的验证集: {config.val_memmap_dir}")
    else:
        print_rank0("提示：当前按 token 块划分验证集，正式重训请设置 val_memmap_dir，避免同文档泄漏")

    print_rank0(f"数据使用比例: {config.data_fraction * 100:.0f}% ({used_samples:,} / {total_samples:,})")

    print_rank0(f"训练集: {len(train_dataset)} 条")
    print_rank0(f"测试集: {len(test_dataset)} 条")

    # 计算训练步数
    effective_batch_size = config.batch_size * world_size * config.accumulation_steps
    steps_per_epoch = len(train_dataset) // (config.batch_size * world_size * config.accumulation_steps)
    train_iters = steps_per_epoch * config.epochs
    stop_iter = min(train_iters, config.max_steps) if config.max_steps is not None else train_iters
    if stop_iter <= 0:
        raise ValueError("实际训练步数必须大于 0")
    config._resume_signature = resume_signature(config, len(train_dataset), world_size)
    import hashlib
    from pathlib import Path
    config._resume_signature["tokenizer_sha256"] = hashlib.sha256(Path(config.bpe_model_path).read_bytes()).hexdigest()
    config._resume_signature["metadata_sha256"] = hashlib.sha256(Path(meta_path).read_bytes()).hexdigest()
    config._resume_signature["memmap_bytes"] = Path(memmap_path).stat().st_size
    schedule_iters = config.schedule_steps or train_iters
    warmup_iters = int(schedule_iters * config.warmup_ratio)
    print_rank0(f"学习率计划={schedule_iters} 步，停止位置={stop_iter}，Adam betas=({config.adam_beta1}, {config.adam_beta2})，输出头分块={config.loss_chunk_size}")

    print_rank0(f"每 epoch 步数: {steps_per_epoch}")
    print_rank0(f"总训练步数: {train_iters} ({config.epochs} epochs)")
    print_rank0(f"Warmup 步数: {warmup_iters} ({config.warmup_ratio * 100:.0f}%)")
    print_rank0(f"有效 batch size: {effective_batch_size} (per GPU: {config.batch_size}, GPUs: {world_size}, accum: {config.accumulation_steps})")
    print_rank0(f"有效 tokens/step: {effective_batch_size * config.max_seq_len:,}")
    if steps_per_epoch == 0:
        raise ValueError(
            "每个 epoch 的训练步数为 0，请增大 data_fraction 或减小 batch_size/accumulation_steps。"
        )

    # 设置随机种子（每个进程不同的种子）
    torch.manual_seed(config.random_seed + get_rank())

    # 创建分布式采样器
    train_sampler = ResumableBatchSampler(len(train_dataset), config.batch_size,
                                          config.accumulation_steps, config.random_seed, get_rank(), world_size)
    train_loader_kwargs = {
        "batch_sampler": train_sampler, "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "generator": torch.Generator().manual_seed(config.random_seed + get_rank()),
    }
    if config.num_workers > 0:
        train_loader_kwargs.update(prefetch_factor=config.prefetch_factor, persistent_workers=True)
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    # 评估使用固定、无重复的 rank 分片，不改变训练数据游标或 dropout 随机数。
    def eval_loader(ds):
        return DataLoader(Subset(ds, range(get_rank(), len(ds), world_size)),
                          batch_size=config.batch_size, num_workers=0,
                          generator=torch.Generator().manual_seed(config.random_seed))
    train_eval_loader, test_loader = eval_loader(train_dataset), eval_loader(test_dataset)
    eval_loaders = {"train": train_eval_loader, "test": test_loader}
    for name, folder in config.extra_val_memmap_dirs.items():
        if name in eval_loaders:
            raise ValueError(f"额外验证集名称重复: {name}")
        dataset = MemmapDataset(os.path.join(folder, "train.bin"), os.path.join(folder, "meta.json"), pad_id)
        for key in ("vocab_size", "pad_id", "train_seq_len", "tokenizer_sha256"):
            if dataset.meta.get(key) != full_dataset.meta.get(key):
                raise ValueError(f"{name} 与训练集的 {key} 不一致")
        if not can_skip_attention_mask(dataset.meta):
            raise ValueError(f"{name} 必须为 packed 且无 padding 的数据")
        eval_loaders[name] = eval_loader(dataset)
        print_rank0(f"独立监测验证集: {name}，路径={folder}，块数={len(dataset)}")

    # 创建模型
    architecture = {}
    if config.model_config_path:
        import json
        with open(config.model_config_path, encoding="utf-8") as source:
            architecture = json.load(source)
        for name in ("vocab_size", "max_seq_len", "pad_token_id", "bos_token_id", "eos_token_id",
                     "use_gradient_checkpointing", "use_native_gqa", "loss_chunk_size"):
            architecture.pop(name, None)
        print_rank0(f"使用指定架构配置: {config.model_config_path}")
    model_config = RiverValleyConfig(
        vocab_size=vocab_size,
        max_seq_len=config.max_seq_len,
        pad_token_id=pad_id,
        bos_token_id=get_im_start_id(tokenizer),
        eos_token_id=get_eos_id(tokenizer),
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        use_native_gqa=config.use_native_gqa,
        loss_chunk_size=config.loss_chunk_size,
        **architecture,
    )

    if config.compile_benchmark:
        run_compile_benchmark(
            model_config,
            train_loader,
            device,
            config,
            skip_attention_mask,
            use_ddp,
            local_rank,
            world_size,
        )
        cleanup_ddp()
        return

    # 初始化 swanlab（只在主进程，benchmark 模式不创建实验记录）
    global swanlab
    if config.use_swanlab and is_main:
        import swanlab as _swanlab
        swanlab = _swanlab

        run_name = config.swanlab_run_name or f"Pretrain-{config.epochs}epochs"
        swanlab.init(
            project=config.swanlab_project,
            experiment_name=run_name,
            config=config.to_dict(),
        )

    print_rank0("训练精度: CUDA 使用 BF16 autocast，参数和优化器状态保持 FP32；CPU 使用 FP32")

    # 恢复训练（必须在 compile 和 DDP 之前加载 checkpoint）
    start_iter = 0
    best_val_loss = float("inf")
    resume_path = resolve_checkpoint_path(config.resume_path or f"{config.checkpoint_dir}/latest")
    training_state = None

    if config.resume:
        if not os.path.isdir(resume_path):
            raise FileNotFoundError(f"续训模型目录不存在: {resume_path}")
        # 直接加载 checkpoint，避免同时在显存中保留两份完整模型。
        validate_checkpoint_tokenizer(resume_path, tokenizer)
        model, training_state = load_model_hf(resume_path, device)
        if model.config.vocab_size != vocab_size or model.config.pad_token_id != pad_id:
            raise ValueError("checkpoint 与当前 tokenizer 的词表大小或 pad_id 不一致")
        model.config.max_seq_len = config.max_seq_len
        model.config.use_gradient_checkpointing = config.use_gradient_checkpointing
        model.gradient_checkpointing = config.use_gradient_checkpointing
        model.config.use_native_gqa = config.use_native_gqa
        model.config.loss_chunk_size = config.loss_chunk_size
        for block in model.blocks:
            block.sa.use_native_gqa = config.use_native_gqa
        if config.new_stage_from_complete:
            from training_utils import validate_new_pretrain_stage
            config._stage_origin = validate_new_pretrain_stage(
                training_state, config._resume_signature, config.initial_checkpoint_expected_step)
            config._stage_origin['source_checkpoint'] = resume_path
            print_rank0(f"启动显式新预训练阶段：来源={resume_path}，新游标=0，优化器与 RNG 保留")
        elif config.resume_weights_only:
            training_state = None
            print_rank0(f"仅加载权重: {resume_path}，优化器、数据游标和学习率计划重新开始")
        else:
            validate_resume(training_state, config._resume_signature)
            start_iter = training_state.get("next_iter")
            if start_iter is None:
                start_iter = training_state.get("iter_num", 0) + 1
            best_val_loss = training_state.get("best_val_loss", float("inf"))
            config._stage_origin = training_state.get('stage_origin')
        print_rank0(f"恢复训练: next_iter={start_iter}, best_val_loss={best_val_loss:.4f}")
    else:
        model = RiverValleyModel(model_config).to(device)
    if start_iter > stop_iter:
        raise ValueError(f"checkpoint 已完成 {start_iter} 步，超过停止位置 {stop_iter}")

    # 先 compile 再包装 DDP（PyTorch 不支持 compile DDP wrapper）
    model = compile_model_if_needed(model, config)

    if use_ddp:
        model = wrap_model_ddp(model, local_rank, use_ddp=True)
        print_rank0("已启用 DDP 分布式训练")

    num_params = sum(p.numel() for p in get_raw_model(model).parameters())
    print_rank0(f"\n模型参数量: {num_params:,}")

    # 创建优化器（使用原始模型参数）
    raw_model = get_raw_model(model)
    decay_params = []
    no_decay_params = []
    for name, param in raw_model.named_parameters():
        if param.requires_grad:
            if param.ndim < 2:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    adamw_foreach = getattr(config, "adamw_foreach", False)
    adamw_fused = bool(getattr(config, "adamw_fused", False) and device.type == "cuda" and not use_ddp)
    optimizer_kwargs = {"lr": config.learning_rate, "betas": (config.adam_beta1, config.adam_beta2), "eps": config.adam_eps}
    if adamw_fused:
        optimizer_kwargs["fused"] = True
        print_rank0("使用 fused AdamW 提升单卡 optimizer step 吞吐")
    else:
        optimizer_kwargs["foreach"] = adamw_foreach
    if use_ddp:
        print_rank0("使用 ZeroRedundancyOptimizer (AdamW) 节省显存")
        from torch.distributed.optim import ZeroRedundancyOptimizer
        optimizer = ZeroRedundancyOptimizer(
            [
                {"params": decay_params, "weight_decay": config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            optimizer_class=torch.optim.AdamW,
            **optimizer_kwargs
        )
    else:
        optimizer = torch.optim.AdamW([
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], **optimizer_kwargs)

    def estimate_loss():
        return evaluate_loss(model, eval_loaders, device, config.eval_iters, skip_attention_mask)

    # 恢复优化器状态（模型权重已在 compile 前加载）
    if training_state and training_state.get("optimizer"):
        optimizer.load_state_dict(training_state["optimizer"])
        restore_rng_states(training_state)
        del training_state

    # 训练循环
    print_rank0("\n" + "=" * 50)
    print_rank0(f"开始预训练 (从 iter {start_iter} 到 {train_iters})")
    print_rank0("=" * 50)

    train_iter = None
    t0 = time.time()
    throughput_start_step = start_iter
    model.train()
    stopper = TrainingStopper(device)

    for iter_num in range(start_iter, stop_iter):
        if stopper.requested():
            stop_iter = iter_num
            print_rank0(f"收到预算或外部停止请求：已完成 {stop_iter} 步，开始验证并保存完整断点")
            if is_main:
                append_metrics(config.checkpoint_dir, {"event": "stop_requested", "step": stop_iter})
            break
        if train_iter is None or iter_num % steps_per_epoch == 0:
            train_sampler.set_step(iter_num)
            train_iter = iter(train_loader)
            print_rank0(f"数据游标: epoch={train_sampler.epoch}, 起始 micro-batch={train_sampler.start_batch}")

        lr = get_lr(iter_num, schedule_iters, warmup_iters, config.learning_rate, config.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_num % config.eval_interval == 0:
            losses = estimate_loss()
            if is_main:
                append_metrics(config.checkpoint_dir, {"event": "eval", "step": iter_num, "lr": lr, **losses})
                if config.extra_val_memmap_dirs:
                    print_rank0(f"分语言验证 step={iter_num}: " + ", ".join(f"{name}={losses[name]:.4f}" for name in config.extra_val_memmap_dirs))

            # 计算 throughput（只在主进程）
            if is_main:
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                measured_updates = iter_num - throughput_start_step
                throughput_start_step = iter_num
                if measured_updates > 0:
                    # 断点可能位于两个验证点之间，必须按实际完成的更新数计算吞吐。
                    tokens_per_sec = (measured_updates * effective_batch_size * config.max_seq_len) / dt
                    print_rank0(f"[{now_str()}] Step {iter_num:5d} | train loss: {losses['train']:.4f} | test loss: {losses['test']:.4f} | lr: {lr:.2e} | {tokens_per_sec:.0f} tokens/s")
                else:
                    print_rank0(f"[{now_str()}] Step {iter_num:5d} | train loss: {losses['train']:.4f} | test loss: {losses['test']:.4f} | lr: {lr:.2e}")

            if config.use_swanlab and swanlab and is_main:
                swanlab.log({
                    "train_loss": losses["train"],
                    "test_loss": losses["test"],
                    "learning_rate": lr,
                    "iter": iter_num,
                })

            if config.save_best and losses["test"] < best_val_loss:
                best_val_loss = losses["test"]
                barrier()  # 等待所有进程
                optimizer_state = None
                if should_save_optimizer_state(config, "best"):
                    optimizer_state = get_optimizer_state_for_save(optimizer, use_ddp, is_main)
                save_model_hf(model, tokenizer, f"{config.checkpoint_dir}/best",
                              optimizer_state, iter_num, best_val_loss, config, next_iter=iter_num)
                barrier()  # 等待保存完成

        if iter_num > 0 and iter_num % config.save_interval == 0:
            barrier()
            save_latest = getattr(config, "save_latest_on_interval", True)
            include_interval_state = should_save_optimizer_state(config, "interval")
            include_latest_state = save_latest and should_save_optimizer_state(config, "latest")
            optimizer_state = None
            if include_interval_state or include_latest_state:
                optimizer_state = get_optimizer_state_for_save(optimizer, use_ddp, is_main)
            save_model_hf(model, tokenizer, f"{config.checkpoint_dir}/ckpt_{iter_num}",
                          optimizer_state if include_interval_state else None,
                          iter_num, best_val_loss, config, next_iter=iter_num)
            if save_latest:
                save_model_hf(model, tokenizer, f"{config.checkpoint_dir}/latest",
                              optimizer_state if include_latest_state else None,
                              iter_num, best_val_loss, config, next_iter=iter_num)
            if is_main:
                prune_interval_checkpoints(config.checkpoint_dir, config.keep_interval_checkpoints)
            barrier()

        batches = [next(train_iter) for _ in range(config.accumulation_steps)]
        update_started = time.perf_counter()
        stats = train_update(model, optimizer, batches, device, config.grad_clip, skip_attention_mask)

        log_interval = getattr(config, "log_interval", 0)
        if is_main and log_interval and (iter_num + 1) % log_interval == 0:
            append_metrics(config.checkpoint_dir, {"event": "train", "step": iter_num + 1, "lr": lr,
                **stats, "update_seconds": time.perf_counter() - update_started,
                "input_tokens_seen": (iter_num + 1) * effective_batch_size * config.max_seq_len})
        if log_interval and iter_num % log_interval == 0 and iter_num % config.eval_interval != 0:
            print_rank0(f"[{now_str()}] Step {iter_num + 1:5d} | lr: {lr:.2e} | loss: {stats['loss']:.4f} | grad_norm: {stats['grad_norm']:.4f}")

    # 最后一次更新也必须评估，保证 best 包括本轮末尾状态。
    losses = estimate_loss()
    if is_main:
        append_metrics(config.checkpoint_dir, {"event": "eval_final", "step": stop_iter, **losses})
    if config.save_best and losses["test"] < best_val_loss:
        best_val_loss = losses["test"]
        state = get_optimizer_state_for_save(optimizer, use_ddp, is_main) if should_save_optimizer_state(config, "best") else None
        save_model_hf(model, tokenizer, f"{config.checkpoint_dir}/best", state,
                      stop_iter - 1, best_val_loss, config, next_iter=stop_iter)
    # 保存最终模型
    barrier()
    optimizer_state = None
    if should_save_optimizer_state(config, "latest"):
        optimizer_state = get_optimizer_state_for_save(optimizer, use_ddp, is_main)
    save_model_hf(model, tokenizer, f"{config.checkpoint_dir}/latest",
                  optimizer_state, stop_iter - 1, best_val_loss, config, next_iter=stop_iter)
    barrier()

    print_rank0("\n预训练完成!")
    print_rank0(f"最终 train loss: {losses['train']:.4f} | test loss: {losses['test']:.4f}")
    print_rank0(f"最佳 test loss: {best_val_loss:.4f}")

    if config.use_swanlab and swanlab and is_main:
        swanlab.finish()

    print_rank0(f"\n模型保存位置: {config.checkpoint_dir}/")

    # 清理 DDP
    cleanup_ddp()


if __name__ == "__main__":
    import argparse
    import json
    import logging
    from pathlib import Path
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="RiverValley 预训练")
    parser.add_argument("--config", help="覆盖 PretrainConfig 默认值的 JSON 文件")
    args = parser.parse_args()
    main(PretrainConfig(**json.loads(Path(args.config).read_text(encoding="utf-8"))) if args.config else None)
