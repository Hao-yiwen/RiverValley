# -*- coding: utf-8 -*-
"""
模型测试
"""

import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "scripts"))

import pytest
import torch

from config import (
    DEFAULT_BPE_VOCAB_SIZE,
    DEFAULT_HEAD_DIM,
    DEFAULT_INTERMEDIATE_SIZE,
    DEFAULT_N_EMBED,
    DEFAULT_N_HEADS,
    DEFAULT_N_KV_HEADS,
    DEFAULT_N_LAYERS,
    RiverValleyConfig,
)
from ddp_utils import get_raw_model
from model import RiverValleyModel
from model.embeddings import RMSNorm


class TestRiverValleyConfig:
    """测试配置类"""

    def test_default_config(self):
        """测试默认配置"""
        config = RiverValleyConfig()
        assert config.vocab_size == DEFAULT_BPE_VOCAB_SIZE
        assert config.n_embed == DEFAULT_N_EMBED
        assert config.n_heads == DEFAULT_N_HEADS
        assert config.n_kv_heads == DEFAULT_N_KV_HEADS
        assert config.n_layers == DEFAULT_N_LAYERS
        assert config.head_dim == DEFAULT_HEAD_DIM
        assert config.intermediate_size == DEFAULT_INTERMEDIATE_SIZE
        assert config.n_kv_heads == 4
        assert config.n_layers == 21
        assert config.intermediate_size == 5632
        assert config.hidden_size == config.n_embed
        assert config.num_attention_heads == config.n_heads
        assert config.num_key_value_heads == config.n_kv_heads
        assert config.num_hidden_layers == config.n_layers

    def test_default_config_parameter_count_formula(self):
        """用公式验证 1B 默认参数量，避免在单测里实例化完整模型。"""
        config = RiverValleyConfig()
        q_dim = config.n_heads * config.head_dim
        kv_dim = config.n_kv_heads * config.head_dim

        embedding_params = config.vocab_size * config.n_embed
        attention_params = (
            config.n_embed * q_dim
            + config.n_embed * kv_dim
            + config.n_embed * kv_dim
            + q_dim * config.n_embed
        )
        ffn_params = config.n_embed * config.intermediate_size * 3
        norm_params = 2 * config.n_embed
        block_params = attention_params + ffn_params + norm_params
        total_params = embedding_params + config.n_layers * block_params + config.n_embed

        assert total_params == 1_012_488_192

    def test_custom_config(self):
        """测试自定义配置"""
        config = RiverValleyConfig(
            vocab_size=5000,
            n_embed=512,
            n_heads=8,
            n_layers=6,
        )
        assert config.vocab_size == 5000
        assert config.n_embed == 512
        assert config.n_heads == 8
        assert config.n_layers == 6


class TestRiverValleyModel:
    """测试模型"""

    @pytest.fixture
    def small_config(self):
        """小型配置用于测试"""
        return RiverValleyConfig(
            vocab_size=1000,
            n_embed=64,
            n_heads=4,
            n_kv_heads=2,
            n_layers=2,
            max_seq_len=32,
            head_dim=16,
            intermediate_size=128,
            dropout=0.0,
        )

    @pytest.fixture
    def model(self, small_config):
        """创建测试模型"""
        return RiverValleyModel(small_config)

    def test_model_creation(self, model, small_config):
        """测试模型创建"""
        assert model is not None
        assert model.config.vocab_size == small_config.vocab_size

    def test_forward_pass(self, model, small_config):
        """测试前向传播"""
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

        outputs = model(input_ids=input_ids, use_cache=False)

        assert outputs.logits.shape == (batch_size, seq_len, small_config.vocab_size)

    def test_forward_with_labels(self, model, small_config):
        """测试带标签的前向传播"""
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

        outputs = model(input_ids=input_ids, labels=labels, use_cache=False)

        assert outputs.loss is not None
        assert outputs.loss.ndim == 0  # scalar

    def test_forward_with_attention_mask(self, model, small_config):
        """测试带注意力掩码的前向传播"""
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, -4:] = 0  # 最后 4 个位置是 padding

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        assert outputs.logits.shape == (batch_size, seq_len, small_config.vocab_size)

    def test_kv_cache(self, model, small_config):
        """测试 KV Cache"""
        model.eval()
        batch_size = 1
        seq_len = 8
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

        # 第一次前向（构建缓存）
        outputs1 = model(input_ids=input_ids, use_cache=True)
        past_key_values = outputs1.past_key_values

        assert past_key_values is not None
        assert len(past_key_values) == small_config.n_layers

        # 第二次前向（使用缓存）
        new_token = torch.randint(0, small_config.vocab_size, (batch_size, 1))
        outputs2 = model(
            input_ids=new_token,
            past_key_values=past_key_values,
            use_cache=True,
        )

        assert outputs2.logits.shape == (batch_size, 1, small_config.vocab_size)

    def test_generate(self, model, small_config):
        """测试生成"""
        batch_size = 1
        prompt_len = 4
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, prompt_len))
        model.eval()

        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=0,
            )

        assert generated.shape[0] == batch_size
        assert generated.shape[1] == prompt_len + 10

    def test_rmsnorm_keeps_input_dtype(self):
        """测试 RMSNorm 使用 FP32 计算但输出保持输入 dtype"""
        norm = RMSNorm(8)
        x = torch.randn(2, 4, 8, dtype=torch.bfloat16)

        out = norm(x)

        assert out.dtype == torch.bfloat16

    def test_get_raw_model_unwraps_nested_wrappers(self, small_config):
        """测试 DDP 和 torch.compile 风格包装会被递归解开"""
        raw = RiverValleyModel(small_config)

        class CompileWrapper:
            def __init__(self, module):
                self._orig_mod = module

        class DDPWrapper:
            def __init__(self, module):
                self.module = module

        wrapped = DDPWrapper(CompileWrapper(raw))

        assert get_raw_model(wrapped) is raw

    def test_grpo_logps_align_completion_logits(self):
        """测试 GRPO 只用最后 completion_len 个 logits 对齐 completion token"""
        from train_grpo import get_per_token_logps

        class DummyOutput:
            def __init__(self, logits):
                self.logits = logits
                self.aux_loss = None

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.full((1, 5, 10), -10.0)
                self.logits[0, 2, 3] = 8.0
                self.logits[0, 3, 4] = 7.0
                self.seen_attention_mask = None
                self.seen_use_cache = None

            def forward(self, input_ids, attention_mask=None, use_cache=None):
                self.seen_attention_mask = attention_mask
                self.seen_use_cache = use_cache
                return DummyOutput(self.logits.to(input_ids.device))

        dummy = DummyModel()
        input_ids = torch.tensor([[5, 6, 7, 3, 4]])
        completion_ids = input_ids[:, -2:]
        attention_mask = torch.ones_like(input_ids)

        logps = get_per_token_logps(dummy, input_ids, completion_ids, attention_mask)
        expected = torch.stack(
            [
                torch.log_softmax(dummy.logits[0, 2].float(), dim=-1)[3],
                torch.log_softmax(dummy.logits[0, 3].float(), dim=-1)[4],
            ]
        )

        assert dummy.seen_use_cache is False
        assert dummy.seen_attention_mask is attention_mask
        assert torch.allclose(logps[0], expected)


class TestMoEModel:
    """测试 MoE 模型"""

    @pytest.fixture
    def moe_config(self):
        """MoE 配置"""
        return RiverValleyConfig(
            vocab_size=1000,
            n_embed=64,
            n_heads=4,
            n_kv_heads=2,
            n_layers=4,
            max_seq_len=32,
            head_dim=16,
            intermediate_size=128,
            dropout=0.0,
            use_moe=True,
            num_experts=4,
            num_shared_experts=1,
            moe_top_k=1,
            moe_freq=2,
        )

    @pytest.fixture
    def moe_model(self, moe_config):
        """创建 MoE 模型"""
        return RiverValleyModel(moe_config)

    def test_moe_forward(self, moe_model, moe_config):
        """测试 MoE 前向传播"""
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, moe_config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, moe_config.vocab_size, (batch_size, seq_len))

        outputs = moe_model(input_ids=input_ids, labels=labels, use_cache=False)

        assert outputs.logits.shape == (batch_size, seq_len, moe_config.vocab_size)
        assert outputs.loss is not None
        assert outputs.aux_loss is not None
