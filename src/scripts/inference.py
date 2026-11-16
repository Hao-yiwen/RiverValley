# -*- coding: utf-8 -*-
"""
RiverValley 推理脚本

使用训练好的模型生成文本
"""

import os
import sys
import argparse
import time

# 获取项目根目录和 src 目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# 添加 src 到路径
sys.path.insert(0, SRC_DIR)

# 切换到项目根目录，确保相对路径正确
os.chdir(PROJECT_ROOT)

import torch
from transformers import PreTrainedTokenizerFast

from config import InferenceConfig
from model import RiverValleyModel


def load_model(model_path: str, device: str = None):
    """从 HuggingFace 格式加载模型"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    if device_obj.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    print(f"加载模型: {model_path}")
    print(f"推理精度: {dtype}")

    model = RiverValleyModel.from_pretrained(
        model_path,
        dtype=dtype,
    ).to(device_obj)
    model.eval()

    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)

    print(f"模型加载完成，参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"词汇表大小: {tokenizer.vocab_size}")

    return model, tokenizer


def get_generation_eos_ids(tokenizer) -> list:
    """获取生成停止 token，ChatML 下同时识别 <|endoftext|> 和 <|im_end|>。"""
    eos_ids = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id not in eos_ids:
        eos_ids.append(im_end_id)
    return eos_ids


def generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 1.0,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    device: str = None,
    chat: bool = False,
    do_sample: bool = True,
    messages=None,
    return_metadata: bool = False,
):
    """生成单轮或完整对话；可返回实际生成数量和停止原因供评估审计。"""
    if device is None:
        device = next(model.parameters()).device

    if messages is not None and (not chat or not messages or messages[-1].get('role') != 'user'):
        raise ValueError('多轮生成需要 ChatML，并以用户问题结束')
    if chat:
        input_ids = tokenizer.apply_chat_template(
            messages if messages is not None else [{"role": "user", "content": prompt}], add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
    else:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    if input_ids.shape[1] + max_tokens > model.config.max_seq_len:
        raise ValueError('提示与生成预算超出模型上下文，拒绝默默截断验收题')

    print(f"Prompt: {prompt}")
    print(f"参数: max_tokens={max_tokens}, temperature={temperature}, top_p={top_p}")
    print("-" * 50)

    started = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            **({"temperature": temperature, "top_p": top_p} if do_sample else {}),
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=get_generation_eos_ids(tokenizer),
        )

    tokens = output_ids[0, input_ids.shape[1]:] if chat else output_ids[0]
    generated_text = tokenizer.decode(tokens, skip_special_tokens=chat)
    if return_metadata:
        generated = output_ids[0, input_ids.shape[1]:]
        return {'answer': generated_text, 'generated_tokens': len(generated), 'prompt_tokens': input_ids.shape[1],
                'elapsed_seconds': time.perf_counter() - started,
                'finish_reason': 'stop' if len(generated) and generated[-1].item() in get_generation_eos_ids(tokenizer) else 'length',
                'do_sample': do_sample, 'repetition_penalty': repetition_penalty,
                'dtype': str(next(model.parameters()).dtype)}
    return generated_text


def interactive_mode(model, tokenizer, config: InferenceConfig = None):
    """交互式生成模式（续写）"""
    if config is None:
        config = InferenceConfig()

    print("\n" + "=" * 50)
    print("交互式生成模式（续写）")
    print("输入 prompt 开始生成，输入 'quit' 或 'exit' 退出")
    print("=" * 50 + "\n")

    while True:
        try:
            prompt = input("Prompt> ").strip()
            if prompt.lower() in ["quit", "exit", "q"]:
                print("退出")
                break
            if not prompt:
                continue

            result = generate(
                model,
                tokenizer,
                prompt,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
            )
            print("\n生成结果:")
            print(result)
            print("\n" + "-" * 50 + "\n")

        except KeyboardInterrupt:
            print("\n退出")
            break


def chat_mode(model, tokenizer, config: InferenceConfig = None, reasoning: bool = False):
    """对话模式（ChatML 格式）"""
    if config is None:
        config = InferenceConfig()

    device = next(model.parameters()).device

    # 推理模式的 system prompt
    if reasoning:
        system_prompt = {
            "role": "system",
            "content": "你是一个善于思考的助手。请在回答问题时，先在<think>标签内进行详细的思考和分析，然后在<answer>标签内给出最终答案。格式如下：\n<think>\n你的思考过程...\n</think>\n<answer>\n你的最终答案...\n</answer>"
        }
        messages = [system_prompt]
    else:
        messages = []

    print("\n" + "=" * 50)
    if reasoning:
        print("对话模式 (ChatML + 推理)")
    else:
        print("对话模式 (ChatML)")
    print("输入消息开始对话，输入 'quit' 或 'exit' 退出")
    print("输入 'clear' 清空对话历史")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("You> ").strip()
            if user_input.lower() in ["quit", "exit", "q"]:
                print("退出")
                break
            if user_input.lower() == "clear":
                if reasoning:
                    messages = [system_prompt]
                else:
                    messages = []
                print("对话历史已清空\n")
                continue
            if not user_input:
                continue

            # 添加用户消息
            messages.append({"role": "user", "content": user_input})

            # 构建 ChatML 格式的 prompt
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            max_len = getattr(model.config, "max_seq_len", None)
            if max_len is not None and input_ids.size(1) > max_len:
                print(f"上下文超过模型长度 {max_len}，已截断最早的 token")
                input_ids = input_ids[:, -max_len:]

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=config.max_tokens,
                    do_sample=True,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    repetition_penalty=config.repetition_penalty,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=get_generation_eos_ids(tokenizer),
                )

            # 解码生成的部分
            generated_ids = output_ids[0][input_ids.shape[1]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=False)

            # 清理 response：移除 <|im_end|> 及之后的内容
            if "<|im_end|>" in response:
                response = response.split("<|im_end|>")[0]
            response = response.strip()

            # 添加助手回复到历史
            messages.append({"role": "assistant", "content": response})

            print(f"Assistant> {response}\n")

        except KeyboardInterrupt:
            print("\n退出")
            break


def main():
    parser = argparse.ArgumentParser(description="RiverValley 推理脚本")
    parser.add_argument("--model", type=str, default="checkpoints/sft/best", help="模型目录路径")
    parser.add_argument("--prompt", type=str, default=None, help="输入提示词（不指定则进入交互模式）")
    parser.add_argument("--complete", action="store_true", help="续写模式（默认为对话模式）")
    parser.add_argument("--reason", action="store_true", help="推理模式（添加思考标签引导）")
    parser.add_argument("--max_tokens", type=int, default=512, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=1.0, help="温度参数")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p 采样阈值")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="重复惩罚系数")

    parser.add_argument("--greedy", action="store_true", help="单次生成使用贪心解码，便于复现评测")
    args = parser.parse_args()

    config = InferenceConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    model, tokenizer = load_model(args.model, config.device)

    if args.prompt:
        result = generate(
            model,
            tokenizer,
            args.prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
            chat=not args.complete,
            do_sample=not args.greedy,
        )
        print("\n生成结果:")
        print(result)
    elif args.complete:
        # 续写模式
        interactive_mode(model, tokenizer, config)
    else:
        # 默认对话模式
        chat_mode(model, tokenizer, config, reasoning=args.reason)


if __name__ == "__main__":
    main()
