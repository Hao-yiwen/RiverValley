# -*- coding: utf-8 -*-
"""
RiverValley 语言模型

包含:
- RiverValleyModel: 主模型类（HuggingFace 兼容）
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from config import RiverValleyConfig
from model.block import Block
from model.embeddings import RMSNorm
from model.losses import chunked_linear_cross_entropy


@dataclass
class RiverValleyCausalLMOutputWithPast(CausalLMOutputWithPast):
    """扩展 HF 输出，额外携带 MoE 负载均衡损失。"""

    aux_loss: Optional[torch.FloatTensor] = None


class RiverValleyModel(PreTrainedModel, GenerationMixin):
    """
    RiverValley 语言模型，继承自 HuggingFace PreTrainedModel 和 GenerationMixin

    特性：
    - GQA 分组查询注意力 + Flash Attention
    - KV Cache 推理加速
    - RoPE 旋转位置编码
    - 标准权重初始化
    - 兼容 HuggingFace save_pretrained / from_pretrained
    - 兼容 HuggingFace generate()
    """
    config_class = RiverValleyConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["Block"]
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: RiverValleyConfig):
        super().__init__(config)
        self.config = config

        # Token 嵌入 (使用 RoPE，不需要位置嵌入)
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embed)

        # Transformer 层（使用 GQA + KV Cache）
        self.blocks = nn.ModuleList([
            Block(config, i) for i in range(config.n_layers)
        ])
        self.ln_f = RMSNorm(config.n_embed, eps=config.rms_norm_eps)

        # 输出头
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # 权重共享
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding_table.weight

        # Gradient checkpointing
        self.gradient_checkpointing = config.use_gradient_checkpointing

        # 初始化权重
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        """标准权重初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)

        if isinstance(module, nn.Linear) and hasattr(module, "weight"):
            for pn, p in self.named_parameters():
                if (pn.endswith("c_proj.weight") or pn.endswith("w3.weight")) and p is module.weight:
                    torch.nn.init.normal_(
                        p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers)
                    )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.token_embedding_table

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.token_embedding_table = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, RiverValleyModel):
            module.gradient_checkpointing = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        loss_targets: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> RiverValleyCausalLMOutputWithPast:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if self.training or (self.gradient_checkpointing and self.training):
            use_cache = False

        B, T = input_ids.shape

        # 计算 start_pos
        start_pos = 0
        if past_key_values is not None and len(past_key_values) > 0:
            first_layer = past_key_values[0]
            if first_layer is not None and len(first_layer) > 0 and first_layer[0] is not None:
                start_pos = first_layer[0].shape[2]

        # 嵌入
        tok_emb = self.token_embedding_table(input_ids)
        x = tok_emb

        # Transformer 处理
        new_kv_caches = [] if use_cache else None
        total_aux_loss = None

        for i, block in enumerate(self.blocks):
            kv_cache = None
            if past_key_values is not None and i < len(past_key_values):
                kv_cache = past_key_values[i]

            if self.gradient_checkpointing and self.training:
                # Gradient checkpointing 模式（训练时使用，不支持 KV cache）
                if block.use_moe:
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)
                        return custom_forward

                    x, new_kv_cache, aux_loss = checkpoint(
                        create_custom_forward(block),
                        x, attention_mask, start_pos, None,
                        use_reentrant=False
                    )
                    total_aux_loss = aux_loss if total_aux_loss is None else total_aux_loss + aux_loss
                else:
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)
                        return custom_forward

                    x, new_kv_cache = checkpoint(
                        create_custom_forward(block),
                        x, attention_mask, start_pos, None,
                        use_reentrant=False
                    )
            else:
                # 普通前向传播
                if block.use_moe:
                    x, new_kv_cache, aux_loss = block(x, attention_mask, start_pos, kv_cache)
                    total_aux_loss = aux_loss if total_aux_loss is None else total_aux_loss + aux_loss
                else:
                    x, new_kv_cache = block(x, attention_mask, start_pos, kv_cache)

            if use_cache:
                new_kv_caches.append(new_kv_cache)

        x = self.ln_f(x)

        # 训练专用标签已与输入位置对齐，不得再按 HF labels 的语义移动一次。
        if loss_targets is not None:
            if labels is not None:
                raise ValueError("labels 与 loss_targets 不能同时传入")
            loss = chunked_linear_cross_entropy(x, self.lm_head.weight, loss_targets,
                                                self.config.loss_chunk_size)
            logits = None
        else:
            logits = self.lm_head(x)
            loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            ce_loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )

            if self.config.use_moe and total_aux_loss is not None:
                loss = ce_loss + self.config.aux_loss_coef * total_aux_loss
            else:
                loss = ce_loss

        if not return_dict:
            output = (logits, new_kv_caches)
            return ((loss,) + output) if loss is not None else output

        return RiverValleyCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=new_kv_caches,
            hidden_states=None,
            attentions=None,
            aux_loss=total_aux_loss,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if past_key_values is not None and len(past_key_values) > 0:
            first_layer = past_key_values[0]
            if first_layer is not None and len(first_layer) > 0 and first_layer[0] is not None:
                # 暂不裁剪 KV cache，避免 RoPE 绝对位置在滑窗后重复。
                # 长上下文滑窗需要显式维护 cache_position 后再重新启用。
                input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
            "attention_mask": attention_mask,
        }

    def _reorder_cache(
        self,
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        beam_idx: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        reordered_past = []
        for layer_past in past_key_values:
            if layer_past is None:
                reordered_past.append(None)
            else:
                reordered_past.append(
                    tuple(
                        past_state.index_select(0, beam_idx.to(past_state.device))
                        if past_state is not None else None
                        for past_state in layer_past
                    )
                )
        return reordered_past
