# -*- coding: utf-8 -*-
"""有界训练实验：微型模型端到端验证、完整 1B 架构的单卡吞吐对照。"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src/scripts")]
os.chdir(ROOT)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import transformers
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from config import RiverValleyConfig, PretrainConfig, SFTConfig
from model import RiverValleyModel
from training_utils import train_update
from pretrain import main as pretrain_main
from sft import main as sft_main
from inference import load_model, generate

# 人工构造的冒烟样本只验证记忆和问答格式，不能代表开放域问答能力。
QA = [
    ("你好", "你好！很高兴和你交流。"),
    ("你叫什么名字？", "我叫 RiverValley，是一个小型语言模型。"),
    ("一加一等于几？", "一加一等于二。"),
    ("二加三等于几？", "二加三等于五。"),
    ("中国的首都是哪里？", "中国的首都是北京。"),
    ("法国的首都是哪里？", "法国的首都是巴黎。"),
    ("水的化学式是什么？", "水的化学式是 H2O。"),
    ("太阳从哪边升起？", "太阳从东方升起。"),
    ("一周有几天？", "一周有七天。"),
    ("一年有几个月？", "一年有十二个月。"),
    ("请把 hello 翻译成中文。", "hello 的中文是你好。"),
    ("请把谢谢翻译成英文。", "谢谢的英文是 thank you。"),
    ("猫通常怎么叫？", "猫通常发出喵喵的叫声。"),
    ("植物生长需要什么？", "植物生长需要阳光、水分、空气和适宜的温度。"),
    ("为什么要洗手？", "洗手可以清除手上的污物和病菌，降低传播疾病的风险。"),
    ("请给出一个水果的例子。", "苹果是一种水果。"),
    ("请给出一个蔬菜的例子。", "胡萝卜是一种蔬菜。"),
    ("零乘以五等于几？", "零乘以五等于零。"),
    ("怎么表达感谢？", "可以说谢谢，也可以具体说明对方给了你什么帮助。"),
    ("请说一句鼓励的话。", "每天进步一点点，坚持下去，你会看到自己的成长。"),
    ("遇到不知道的问题怎么办？", "我会说明不确定，并建议查找可靠的资料。"),
    ("阅读有什么好处？", "阅读可以帮助我们学习知识、开阔视野和提高表达能力。"),
    ("请按顺序说出三个数字。", "一、二、三。"),
    ("什么是三角形？", "三角形是由三条线段首尾相连围成的图形。"),
]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run_pilot(output):
    """从新的 tokenizer 和随机权重开始，经过真实预训练与 SFT 入口。"""
    folder = output / "pilot"; folder.mkdir(parents=True, exist_ok=True)
    texts = [f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n" for q, a in QA]
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=768, min_frequency=1,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<|pad|>", "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<unk>"]))
    tokenizer_path = folder / "tokenizer.json"; tokenizer.save(str(tokenizer_path))
    # 预训练和 SFT 的验证问题都不进入训练语料；tokenizer 的词表构造不更新语言模型权重。
    order = torch.randperm(len(QA), generator=torch.Generator().manual_seed(1337)).tolist()
    heldout = set(order[int(0.9 * len(QA)):])
    corpus = [f"{q}{a}" for index, (q, a) in enumerate(QA) if index not in heldout]
    tokens = []
    for _ in range(50):
        for text in corpus:
            tokens.extend(tokenizer.encode(text).ids + [1])
    seq_len = 128
    count = len(tokens) // (seq_len + 1)
    memmap = folder / "memmap"; memmap.mkdir(exist_ok=True)
    np.asarray(tokens[:count * (seq_len + 1)], dtype=np.uint16).tofile(memmap / "train.bin")
    write_json(memmap / "meta.json", {"num_samples": count, "seq_len": seq_len + 1,
        "train_seq_len": seq_len, "vocab_size": tokenizer.get_vocab_size(), "pad_id": 0,
        "packed": True, "partial_blocks": 0, "purpose": "人工冒烟样本，含重复文档，不用于泛化评测"})
    architecture = folder / "architecture.json"
    write_json(architecture, {"n_embed": 128, "n_heads": 4, "n_kv_heads": 2, "head_dim": 32,
                              "intermediate_size": 352, "n_layers": 4})
    pretrain_config = PretrainConfig(memmap_dir=str(memmap), bpe_model_path=str(tokenizer_path),
        model_config_path=str(architecture), checkpoint_dir=str(folder / "pretrain"),
        batch_size=4, accumulation_steps=1, max_seq_len=seq_len, epochs=20, max_steps=120,
        learning_rate=2e-3, min_lr=2e-4, warmup_ratio=0.02, eval_interval=40, eval_iters=4,
        log_interval=20, save_interval=10000, num_workers=0, use_swanlab=False, use_native_gqa=True)
    write_json(folder / "pretrain_config.json", pretrain_config.to_dict())
    pretrain_main(pretrain_config)
    dataset = folder / "sft.jsonl"
    dataset.write_text("\n".join(json.dumps({"conversations": [{"role": "user", "content": q},
                          {"role": "assistant", "content": a}]}, ensure_ascii=False) for q, a in QA), encoding="utf-8")
    sft_config = SFTConfig(data_path=str(dataset), bpe_model_path=str(tokenizer_path),
        pretrain_checkpoint=str(folder / "pretrain/best"), checkpoint_dir=str(folder / "sft"),
        batch_size=4, accumulation_steps=1, max_seq_len=seq_len, epochs=100, max_steps=240,
        learning_rate=2e-3, min_lr=2e-4, warmup_ratio=0.02, eval_interval=60, eval_iters=8,
        log_interval=30, save_interval=10000, num_workers=0, use_swanlab=False, use_native_gqa=True)
    write_json(folder / "sft_config.json", sft_config.to_dict())
    losses = sft_main(sft_config)
    model, hf_tokenizer = load_model(str(folder / "sft/latest"))
    answers = []
    for index, (question, expected) in enumerate(QA):
        answer = generate(model, hf_tokenizer, question, max_tokens=80, repetition_penalty=1,
                          chat=True, do_sample=False).strip()
        answers.append({"question": question, "expected": expected, "actual": answer,
                        "split": "heldout" if index in heldout else "train", "exact": answer == expected})
    result = {"purpose": "流程与记忆能力测试，训练样本答对不等于泛化能力", "losses": losses,
              "parameters": sum(p.numel() for p in model.parameters()), "answers": answers}
    write_json(output / "pilot_results.json", result)
    del model
    gc.collect(); torch.cuda.empty_cache()
    return result


def run_benchmarks(output, include_compile=False, only_padding=False, capacity=False, compile_capacity=False):
    """使用完整默认 1.01B 架构和实际更新函数，固定输入测量稳定阶段计算吞吐。"""
    variants = [
        ("manual_unfused", 4, 512, False, False, False, False),
        ("manual_fused", 4, 512, False, False, True, False),
        ("native_fused", 4, 512, True, False, True, False),
        ("native_b8_512", 8, 512, True, False, True, False),
        ("native_b4_1024", 4, 1024, True, False, True, False),
        ("native_b8_1024", 8, 1024, True, False, True, False),
        ("native_b8_1024_checkpoint", 8, 1024, True, True, True, False),
    ]
    if capacity:
        variants = [("native_b16_512", 16, 512, True, False, True, False),
                    ("native_b16_1024", 16, 1024, True, False, True, False),
                    ("native_b24_1024", 24, 1024, True, False, True, False)]
    if compile_capacity:
        variants = [("native_b16_1024_compile", 16, 1024, True, False, True, True),
                    ("native_b24_1024_compile", 24, 1024, True, False, True, True)]
    if only_padding:
        variants = [("sft_fixed_padding", 4, 1024, True, False, True, False),
                    ("sft_trimmed_padding", 4, 1024, True, False, True, False)]
    if include_compile:
        variants.append(("native_b8_1024_compile", 8, 1024, True, False, True, True))
    results = []
    for name, batch_size, seq_len, native, checkpoint, fused, compile_model in variants:
        print(f"开始 1B 实验: {name}", flush=True)
        torch.manual_seed(1337); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        model = optimizer = batches = None
        result = dict(name=name, batch_size=batch_size, seq_len=seq_len, native_gqa=native,
                      gradient_checkpointing=checkpoint, fused=fused, compile=compile_model)
        try:
            config = RiverValleyConfig(max_seq_len=seq_len, use_native_gqa=native,
                                       use_gradient_checkpointing=checkpoint)
            model = RiverValleyModel(config).cuda().train()
            result["parameters"] = sum(p.numel() for p in model.parameters())
            optimizer = torch.optim.AdamW([
                {"params": [p for p in model.parameters() if p.ndim >= 2], "weight_decay": 0.1},
                {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0}],
                lr=2e-4, fused=fused, **({} if fused else {"foreach": False}))
            if compile_model:
                model = torch.compile(model)
            x = torch.randint(4, config.vocab_size, (batch_size, seq_len), device="cuda")
            y = torch.randint(4, config.vocab_size, x.shape, device="cuda")
            mask, attention = torch.ones_like(x), torch.ones_like(x)
            if only_padding:
                mask[:, :128] = 0; mask[:, 256:] = 0
                attention[:, 256:] = 0; x[:, 256:] = 0; y[:, 256:] = 0
                if name == "sft_trimmed_padding":
                    x, y, mask, attention = [value[:, :256].contiguous() for value in (x, y, mask, attention)]
            batches = [(x, y, mask, attention)]
            started = time.perf_counter()
            for _ in range(3):
                train_update(model, optimizer, batches, torch.device("cuda"), 1.0, not only_padding)
            torch.cuda.synchronize()
            result["warmup_s"] = time.perf_counter() - started
            times, losses = [], []
            torch.cuda.reset_peak_memory_stats()
            for _ in range(8):
                started = time.perf_counter()
                stats = train_update(model, optimizer, batches, torch.device("cuda"), 1.0, not only_padding)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - started); losses.append(stats["loss"])
            result.update(status="ok", median_step_s=float(np.median(times)),
                          tokens_per_s=x.numel() / float(np.median(times)),
                          supervised_tokens_per_s=int(mask.sum()) / float(np.median(times)),
                          peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                          peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
                          times_s=times, losses=losses)
        except Exception as error:
            import traceback
            traceback.print_exc()
            result.update(status="error", error=f"{type(error).__name__}: {error}")
        finally:
            del model, optimizer, batches
            gc.collect(); torch.cuda.empty_cache()
        results.append(result)
        write_json(output / ("padding_results.json" if only_padding else "capacity_results.json" if capacity else "compile_capacity_results.json" if compile_capacity else "benchmark_results.json"), results)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return results


def run_preset(output, loss_head=False, measure_steps=5):
    """保持有效批量和随机输入一致，3 步预热后测量指定的完整更新数。"""
    import statistics
    rows = []
    torch.set_float32_matmul_precision("high")
    variants = [("baseline_b4_acc32", 4, 32, False, False),
                ("recommended_b16_acc8", 16, 8, True, True)]
    if loss_head:
        variants = [("full_logits", 16, 8, True, True),
                    ("chunked_512", 16, 8, True, True), ("chunked_1024", 16, 8, True, True)]
    for name, batch_size, accumulation, compiled, native in variants:
        torch.manual_seed(1337)
        torch.cuda.empty_cache()
        chunk_size = int(name.split("_")[-1]) if name.startswith("chunked_") else 0
        model = RiverValleyModel(RiverValleyConfig(max_seq_len=1024, use_native_gqa=native,
                                                  loss_chunk_size=chunk_size)).cuda().train()
        optimizer = torch.optim.AdamW([
            {"params": [p for p in model.parameters() if p.ndim >= 2], "weight_decay": 0.1},
            {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0},
        ], lr=2e-4, fused=True)
        if compiled:
            model = torch.compile(model)
        batches = []
        for _ in range(accumulation):
            x = torch.randint(4, 32000, (batch_size, 1024), device="cuda")
            y = torch.randint(4, 32000, x.shape, device="cuda")
            batches.append((x, y, torch.ones_like(x), torch.ones_like(x)))
        times = []
        warmup = time.perf_counter()
        first_update = None
        for step in range(3 + measure_steps):
            if step == 3:
                warmup = time.perf_counter() - warmup
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            stats = train_update(model, optimizer, batches, torch.device("cuda"), 1.0, True)
            if first_update is None:
                first_update = stats
            torch.cuda.synchronize()
            if step >= 3:
                times.append(time.perf_counter() - started)
        effective_tokens = batch_size * accumulation * 1024
        result = {
            "name": name, "batch": batch_size, "accumulation_steps": accumulation,
            "effective_tokens": effective_tokens, "warmup_s": warmup,
            "median_step_s": statistics.median(times),
            "tokens_per_s": effective_tokens / statistics.median(times),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "times_s": times,
            "first_update": first_update, "measured_updates": measure_steps,
        }
        rows.append(result)
        write_json(output / ("loss_head_results.json" if loss_head else "preset_results.json"), rows)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        del model, optimizer, batches
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行有界的训练验证实验")
    parser.add_argument("--mode", choices=["pilot", "benchmark", "padding", "capacity", "compile-capacity", "preset", "loss-head", "all"], default="all")
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-compile", action="store_true")
    parser.add_argument("--measure-steps", type=int, default=5, help="preset/loss-head 模式的计时更新数")
    args = parser.parse_args()
    if args.measure_steps <= 0:
        parser.error("--measure-steps 必须为正数")
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("该实验需要支持 BF16 的 CUDA GPU")
    write_json(output / "environment.json", {"torch": torch.__version__, "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
        "vram_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "measurement": f"输入已在显存；3 步预热、{args.measure_steps if args.mode in {'preset', 'loss-head'} else 8} 步测量、中位数吞吐；完整优化器更新，不含下载和存盘"})
    if args.mode in {"pilot", "all"}:
        run_pilot(output)
    if args.mode in {"benchmark", "all"}:
        run_benchmarks(output, args.include_compile)

    if args.mode == "padding":
        run_benchmarks(output, only_padding=True)

    if args.mode == "capacity":
        run_benchmarks(output, capacity=True)

    if args.mode == "compile-capacity":
        run_benchmarks(output, compile_capacity=True)

    if args.mode == "preset":
        run_preset(output, measure_steps=args.measure_steps)
    if args.mode == "loss-head":
        run_preset(output, loss_head=True, measure_steps=args.measure_steps)
