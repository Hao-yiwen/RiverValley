# -*- coding: utf-8 -*-
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "scripts"))

import pytest

from config import PretrainConfig
from pretrain import can_skip_attention_mask, select_data_fraction_indices, should_save_optimizer_state


def test_data_fraction_indices_are_seeded_and_not_prefix():
    """data_fraction 小比例抽样应可复现，且不再固定取前 N 条。"""
    indices_a = select_data_fraction_indices(total_samples=100, data_fraction=0.1, random_seed=1337)
    indices_b = select_data_fraction_indices(total_samples=100, data_fraction=0.1, random_seed=1337)

    assert indices_a == indices_b
    assert len(indices_a) == 10
    assert indices_a != list(range(10))


def test_data_fraction_rejects_invalid_fraction():
    """非法抽样比例要尽早报错，避免静默跑空数据。"""
    with pytest.raises(ValueError):
        select_data_fraction_indices(total_samples=100, data_fraction=0.0, random_seed=1)
    with pytest.raises(ValueError):
        select_data_fraction_indices(total_samples=100, data_fraction=1.1, random_seed=1)


def test_packed_fast_path_requires_no_partial_blocks():
    """只有 packed 且无 partial block 的 memmap 才能跳过 attention_mask。"""
    assert can_skip_attention_mask({"packed": True, "partial_blocks": 0}) is True
    assert can_skip_attention_mask({"packed": True, "partial_blocks": 1}) is False
    assert can_skip_attention_mask({"packed": False, "partial_blocks": 0}) is False


def test_pretrain_defaults_prioritize_long_run_throughput():
    """预训练默认值面向正式长跑，减少 eval/save 干扰并提高 tokens/step。"""
    config = PretrainConfig()

    assert config.batch_size == 4
    assert config.accumulation_steps == 32
    assert config.learning_rate == 2e-4
    assert config.min_lr == 2e-5
    assert config.warmup_ratio == 0.03
    assert config.eval_interval == 2000
    assert config.eval_iters == 20
    assert config.save_interval == 5000
    assert config.save_latest_on_interval is True
    assert config.save_optimizer_state is True
    assert config.save_optimizer_state_for_best is False
    assert config.save_optimizer_state_for_interval is False
    assert config.require_packed_fast_path is True
    assert config.use_gradient_checkpointing is False


def test_pretrain_optimizer_state_save_policy_defaults_to_latest_only():
    """默认只让 latest 保存 optimizer state，避免 best/ckpt_xxx 反复写入大状态。"""
    config = PretrainConfig()

    assert should_save_optimizer_state(config, "latest") is True
    assert should_save_optimizer_state(config, "best") is False
    assert should_save_optimizer_state(config, "interval") is False

    config.save_optimizer_state = False
    assert should_save_optimizer_state(config, "latest") is False
