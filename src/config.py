# -*- coding: utf-8 -*-
"""
RiverValley 配置模块

包含:
- RiverValleyConfig: HuggingFace 兼容的模型配置类
- TrainingConfig: 训练配置数据类
- 默认超参数常量
"""

import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from transformers import PretrainedConfig


# ============================================================================
# 默认超参数
# ============================================================================

# 数据配置
DEFAULT_PRETRAIN_DATA = "datasets/pretrain_hq.jsonl"
DEFAULT_SFT_DATA = "datasets/sft_512.jsonl"
DEFAULT_BPE_VOCAB_SIZE = 32000
DEFAULT_BPE_MODEL_PATH = "/root/autodl-tmp/tokenizer.json"
DEFAULT_RANDOM_SEED = 1337

# ChatML 特殊 token
SPECIAL_TOKENS = {
    "pad_token": "<|pad|>",
    "eos_token": "<|endoftext|>",
    "im_start_token": "<|im_start|>",
    "im_end_token": "<|im_end|>",
}

# 模型架构默认值（当前约 1.01B 参数）
DEFAULT_N_EMBED = 2048
DEFAULT_N_HEADS = 16
DEFAULT_N_KV_HEADS = 4
DEFAULT_N_LAYERS = 21
DEFAULT_HEAD_DIM = 128
DEFAULT_INTERMEDIATE_SIZE = 5632
DEFAULT_DROPOUT = 0.0
DEFAULT_MAX_SEQ_LEN = 1024
DEFAULT_RMS_NORM_EPS = 1e-6
DEFAULT_ROPE_THETA = 1000000.0

# 训练默认值
DEFAULT_BATCH_SIZE = 2
DEFAULT_ACCUMULATION_STEPS = 32
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_MIN_LR = 3e-5
DEFAULT_EPOCHS = 1
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_EVAL_ITERS = 20
DEFAULT_EVAL_INTERVAL = 500
DEFAULT_LOG_INTERVAL = 50
DEFAULT_WEIGHT_DECAY = 0.1
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_DATA_FRACTION = 1.0  # 数据使用比例

# DataLoader 默认值
DEFAULT_NUM_WORKERS = 8
DEFAULT_PIN_MEMORY = True
DEFAULT_PREFETCH_FACTOR = 2

# 推理默认值
DEFAULT_MAX_TOKENS = 200
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.9
DEFAULT_REPETITION_PENALTY = 1.2


# ============================================================================
# RiverValley 配置类 (HuggingFace 兼容)
# ============================================================================

class RiverValleyConfig(PretrainedConfig):
    """
    RiverValley 模型配置类，继承自 HuggingFace PretrainedConfig。

    支持通过 config.json 保存和加载配置。
    """
    model_type = "rivervalley"

    def __init__(
        self,
        vocab_size: int = DEFAULT_BPE_VOCAB_SIZE,
        n_embed: int = DEFAULT_N_EMBED,
        n_heads: int = DEFAULT_N_HEADS,
        n_kv_heads: int = DEFAULT_N_KV_HEADS,
        n_layers: int = DEFAULT_N_LAYERS,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        dropout: float = DEFAULT_DROPOUT,
        # 独立的 head_dim 和 intermediate_size（Qwen3 兼容）
        head_dim: Optional[int] = DEFAULT_HEAD_DIM,
        intermediate_size: Optional[int] = DEFAULT_INTERMEDIATE_SIZE,
        rms_norm_eps: float = DEFAULT_RMS_NORM_EPS,
        rope_theta: float = DEFAULT_ROPE_THETA,
        # MoE 配置
        use_moe: bool = False,
        num_experts: int = 8,
        num_shared_experts: int = 1,
        moe_top_k: int = 1,
        moe_freq: int = 2,
        aux_loss_coef: float = 0.01,
        # YaRN 配置
        rope_scaling_type: str = "none",  # 'none' | 'yarn'
        rope_scaling_factor: float = 1.0,
        yarn_beta_fast: int = 32,
        yarn_beta_slow: int = 1,
        yarn_original_max_seq_len: Optional[int] = None,
        # HF 兼容参数
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 1,
        tie_word_embeddings: bool = True,
        use_cache: bool = True,
        # Gradient checkpointing
        use_gradient_checkpointing: bool = False,
        use_native_gqa: bool = False,
        loss_chunk_size: int = 0,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            use_cache=use_cache,
            **kwargs
        )
        effective_n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        effective_head_dim = head_dim if head_dim is not None else n_embed // n_heads
        if n_embed % n_heads != 0 and head_dim is None:
            raise ValueError("n_embed 必须能被 n_heads 整除，除非显式指定 head_dim")
        if n_heads % effective_n_kv_heads != 0:
            raise ValueError("n_heads 必须能被 n_kv_heads 整除")
        if effective_head_dim % 2 != 0:
            raise ValueError("head_dim 必须是偶数，才能应用 RoPE")
        if moe_top_k > num_experts:
            raise ValueError("moe_top_k 不能大于 num_experts")
        if moe_freq <= 0:
            raise ValueError("moe_freq 必须大于 0")
        if vocab_size > 65535:
            raise ValueError("vocab_size 不能超过 65535，因为当前 memmap 使用 uint16 存储 token id")

        self.vocab_size = vocab_size
        self.use_cache = use_cache
        self.n_embed = n_embed
        self.n_heads = n_heads
        self.n_kv_heads = effective_n_kv_heads
        self.n_layers = n_layers
        self.hidden_size = n_embed  # HF 兼容别名
        self.num_attention_heads = n_heads  # HF 兼容别名
        self.num_key_value_heads = effective_n_kv_heads  # HF 兼容别名
        self.num_hidden_layers = n_layers  # HF 兼容别名
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        # 独立的 head_dim 和 intermediate_size（Qwen3 兼容）
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        # MoE 配置
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.moe_top_k = moe_top_k
        self.moe_freq = moe_freq
        self.aux_loss_coef = aux_loss_coef
        # YaRN 配置
        self.rope_scaling_type = rope_scaling_type
        self.rope_scaling_factor = rope_scaling_factor
        self.yarn_beta_fast = yarn_beta_fast
        self.yarn_beta_slow = yarn_beta_slow
        self.yarn_original_max_seq_len = yarn_original_max_seq_len or max_seq_len
        # Gradient checkpointing
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_native_gqa = use_native_gqa
        if loss_chunk_size < 0:
            raise ValueError("loss_chunk_size 不能为负数")
        self.loss_chunk_size = loss_chunk_size


# ============================================================================
# 训练配置数据类
# ============================================================================

@dataclass
class PretrainConfig:
    """预训练配置"""

    # 数据配置
    memmap_dir: str = "/root/autodl-tmp/pretrain_memmap"  # memmap 数据目录
    val_memmap_dir: Optional[str] = None  # 正式重训应使用按文档隔离后单独打包的验证集
    extra_val_memmap_dirs: Dict[str, str] = field(default_factory=dict)  # 分别监测中文、英文等固定验证分片
    data_fraction: float = DEFAULT_DATA_FRACTION  # 使用数据的比例 (0.0-1.0)
    bpe_vocab_size: int = DEFAULT_BPE_VOCAB_SIZE
    bpe_model_path: str = DEFAULT_BPE_MODEL_PATH
    random_seed: int = DEFAULT_RANDOM_SEED

    # 训练配置
    model_config_path: Optional[str] = None  # 小实验可指定微型架构，默认仍为完整 1.01B
    batch_size: int = 4  # 1B 正式预训练优先面向 2x RTX PRO 6000 96GB，提高 tokens/step
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN
    learning_rate: float = 2e-4
    min_lr: float = 2e-5
    epochs: int = DEFAULT_EPOCHS
    warmup_ratio: float = 0.03
    eval_iters: int = DEFAULT_EVAL_ITERS
    eval_interval: int = 2000
    log_interval: int = DEFAULT_LOG_INTERVAL
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    grad_clip: float = DEFAULT_GRAD_CLIP
    accumulation_steps: int = DEFAULT_ACCUMULATION_STEPS
    adamw_foreach: bool = False  # 关闭 foreach 以减少优化器峰值显存
    adamw_fused: bool = True  # 单卡 CUDA 优先使用 fused AdamW 提升 optimizer step 吞吐
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999  # 保留已测基线，0.95 仅作为后续对照候选
    adam_eps: float = 1e-8
    schedule_steps: Optional[int] = None  # 明确学习率计划，和临时 max_steps 停止位置分离
    loss_chunk_size: int = 0  # 0 保留已验证路径；正数启用待 GPU 对照的输出头重算分块

    # DataLoader 配置
    num_workers: int = DEFAULT_NUM_WORKERS
    pin_memory: bool = DEFAULT_PIN_MEMORY
    prefetch_factor: int = DEFAULT_PREFETCH_FACTOR

    # 训练加速
    use_compile: bool = False  # 编译收益取决于实际 PyTorch、GPU 和批量配置，实测后开启
    compile_mode: str = "default"  # torch.compile 模式，benchmark 后再决定是否改
    compile_benchmark: bool = False  # 只跑 compile 吞吐 benchmark，不进入正式训练
    compile_benchmark_steps: int = 100
    compile_benchmark_warmup_steps: int = 10
    use_gradient_checkpointing: bool = False  # 只在 OOM 时开启，省显存但会增加重算
    use_native_gqa: bool = False  # 先用小实验验证当前 GPU 的原生 GQA 内核
    max_steps: Optional[int] = None  # 小实验只限制执行步数，不改变完整训练的学习率计划

    # DDP 配置
    use_ddp: bool = False  # 是否使用 DDP（torchrun 启动时自动启用）

    # Checkpoint 配置
    checkpoint_dir: str = "/root/autodl-tmp/checkpoints"
    save_interval: int = 5000
    keep_interval_checkpoints: int = 2  # 只清理本入口创建的 ckpt_数字 目录，不影响 best/latest
    save_best: bool = True
    save_latest_on_interval: bool = True  # 每次周期保存时滚动更新 latest，用于精确断点续训
    save_optimizer_state: bool = True  # 只有需要精确断点续训时才保存 AdamW 状态
    save_optimizer_state_for_best: bool = False  # best 主要用于评估/下游续训，默认只保存权重
    save_optimizer_state_for_interval: bool = False  # ckpt_xxx 默认只保存权重，避免多份 7GB+ optimizer state
    resume: bool = False  # 从头开始训练
    resume_path: Optional[str] = None  # 默认使用 checkpoint_dir/latest；512 -> 1024 续训请显式传 best/latest 并设 resume_weights_only=True
    resume_weights_only: bool = False  # 只加载模型权重，不恢复 optimizer/iter
    new_stage_from_complete: bool = False  # 仅允许从已结束的预训练完整状态开始新数据和新学习率阶段
    initial_checkpoint_expected_step: Optional[int] = None  # 新阶段必须显式确认前一阶段最终步数
    require_packed_fast_path: bool = True  # 正式预训练要求 packed 且无 partial block

    # Swanlab 配置
    swanlab_project: str = "rivervalley-pretrain"
    swanlab_run_name: Optional[str] = None
    use_swanlab: bool = True

    # 生成配置
    generate_tokens: int = 100
    generate_prompt: str = "你好"

    # 设备
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class SFTConfig:
    """SFT 微调配置"""

    # 数据配置
    data_path: str = DEFAULT_SFT_DATA
    bpe_model_path: str = DEFAULT_BPE_MODEL_PATH
    random_seed: int = DEFAULT_RANDOM_SEED
    val_data_path: Optional[str] = None  # 使用预处理时冻结的独立验证集

    # 模型配置
    pretrain_checkpoint: str = "/root/autodl-tmp/checkpoints/best"  # 从本轮重新预训练得到的最佳权重开始

    # 训练配置
    batch_size: int = 2
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN
    learning_rate: float = 2e-5
    min_lr: float = 2e-6
    epochs: int = DEFAULT_EPOCHS
    warmup_ratio: float = DEFAULT_WARMUP_RATIO
    eval_iters: int = 100
    eval_interval: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    accumulation_steps: int = 32  # 通过累积得到有效批量 64，避免单次前向显存过大
    log_interval: int = 10
    max_steps: Optional[int] = None  # 小实验限制实际更新步数，保留完整训练的调度计划
    schedule_steps: Optional[int] = None
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    loss_chunk_size: int = 0

    # DataLoader 配置
    num_workers: int = DEFAULT_NUM_WORKERS
    pin_memory: bool = DEFAULT_PIN_MEMORY
    prefetch_factor: int = DEFAULT_PREFETCH_FACTOR

    # 训练加速
    use_compile: bool = False  # 动态裁剪产生不同长度，先实测再启用编译
    use_gradient_checkpointing: bool = False  # A100 80GB 上优先避免重算，显存不足再开启
    use_native_gqa: bool = False
    dynamic_padding: bool = True  # 每批裁掉公共的右侧 padding
    adamw_fused: bool = True

    # DDP 配置
    use_ddp: bool = False  # 是否使用 DDP（torchrun 启动时自动启用）

    # Checkpoint 配置
    checkpoint_dir: str = "checkpoints/sft"
    save_optimizer_state: bool = True  # 正式续训保留；只测流水线时可关闭以减少存盘
    save_interval: int = 10000
    save_best: bool = True
    resume: bool = False  # 是否恢复训练
    resume_path: str = "checkpoints/sft/latest"  # latest 保存完整续训状态
    resume_weights_only: bool = False

    # Swanlab 配置
    swanlab_project: str = "rivervalley-sft"
    swanlab_run_name: Optional[str] = None
    use_swanlab: bool = True

    # 设备
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class DPOConfig:
    """DPO 训练配置"""

    # 数据配置
    data_path: str = "datasets/dpo.jsonl"
    bpe_model_path: str = DEFAULT_BPE_MODEL_PATH
    random_seed: int = DEFAULT_RANDOM_SEED

    # 模型配置
    sft_checkpoint: str = "checkpoints/sft/best"  # SFT模型路径

    # DPO 特有参数
    beta: float = 0.1  # DPO 温度参数，控制偏离 reference model 的惩罚程度

    # 训练配置
    batch_size: int = 32  # DPO需要加载两个模型，batch较小
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN
    learning_rate: float = 5e-6  # DPO 通常使用较小学习率
    min_lr: float = 1e-7
    epochs: int = 1
    warmup_ratio: float = DEFAULT_WARMUP_RATIO
    eval_iters: int = 20
    eval_interval: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    accumulation_steps: int = 1  # 梯度累积步数

    # DataLoader 配置
    num_workers: int = DEFAULT_NUM_WORKERS
    pin_memory: bool = DEFAULT_PIN_MEMORY
    prefetch_factor: int = DEFAULT_PREFETCH_FACTOR

    # 训练加速
    use_compile: bool = False  # DPO 暂不使用 compile

    # DDP 配置
    use_ddp: bool = False  # 是否使用 DDP（torchrun 启动时自动启用）

    # Checkpoint 配置
    checkpoint_dir: str = "checkpoints/dpo"
    save_interval: int = 500
    save_best: bool = True

    # Swanlab 配置
    swanlab_project: str = "rivervalley-dpo"
    swanlab_run_name: Optional[str] = None
    use_swanlab: bool = True

    # 设备
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class InferenceConfig:
    """推理配置"""

    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ReasonDistillConfig:
    """思维链蒸馏训练配置"""

    # 数据配置
    data_path: str = "datasets/r1_mix_1024.jsonl"
    bpe_model_path: str = DEFAULT_BPE_MODEL_PATH
    random_seed: int = DEFAULT_RANDOM_SEED

    # 模型配置
    dpo_checkpoint: str = "checkpoints/dpo/best"  # 基于 DPO 模型

    # 思维链特殊标签权重
    reason_tag_weight: float = 10.0  # 对 <think>, </think>, <answer>, </answer> 标签的权重

    # 训练配置
    batch_size: int = 32
    max_seq_len: int = 1024
    learning_rate: float = 5e-5  # 适当增大学习率
    min_lr: float = 1e-6
    epochs: int = 1
    warmup_ratio: float = 0.03  # 减少 warmup 比例
    eval_iters: int = 20
    eval_interval: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    accumulation_steps: int = 1  # 梯度累积

    # DataLoader 配置
    num_workers: int = DEFAULT_NUM_WORKERS
    pin_memory: bool = DEFAULT_PIN_MEMORY
    prefetch_factor: int = DEFAULT_PREFETCH_FACTOR

    # 训练加速
    use_compile: bool = False

    # DDP 配置
    use_ddp: bool = False  # 是否使用 DDP（torchrun 启动时自动启用）

    # Checkpoint 配置
    checkpoint_dir: str = "checkpoints/reason"
    save_interval: int = 100
    save_best: bool = True
    resume: bool = False
    resume_path: str = "checkpoints/reason/best"

    # Swanlab 配置
    swanlab_project: str = "rivervalley-reason"
    swanlab_run_name: Optional[str] = None
    use_swanlab: bool = True

    # 设备
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class GRPOConfig:
    """GRPO (Group Relative Policy Optimization) 训练配置"""

    # 数据配置
    data_path: str = "datasets/rlaif-mini.jsonl"
    bpe_model_path: str = DEFAULT_BPE_MODEL_PATH
    random_seed: int = DEFAULT_RANDOM_SEED

    # 模型配置
    reason_checkpoint: str = "checkpoints1"  # 基于 Reason 模型
    reward_model_path: str = "/root/autodl-tmp/internlm2-1_8b-reward"  # Reward 模型路径

    # GRPO 参数
    num_generations: int = 8  # 每个 prompt 生成的样本数
    beta: float = 0.02  # KL 惩罚系数
    max_seq_len: int = 66  # Prompt 最大长度
    max_gen_len: int = 1536  # 生成最大长度
    reasoning: bool = True  # 是否为推理模型（使用格式奖励）

    # 训练配置
    batch_size: int = 2
    learning_rate: float = 8e-8
    epochs: int = 1
    accumulation_steps: int = 1
    grad_clip: float = 1.0
    log_interval: int = 1
    save_interval: int = 10

    # DataLoader 配置
    num_workers: int = 1
    pin_memory: bool = True

    # DDP 配置
    use_ddp: bool = False  # 是否使用 DDP（torchrun 启动时自动启用）

    # Checkpoint 配置
    checkpoint_dir: str = "checkpoints/grpo"
    save_best: bool = True

    # Swanlab 配置
    swanlab_project: str = "rivervalley-grpo"
    swanlab_run_name: Optional[str] = None
    use_swanlab: bool = True

    # 设备
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}
