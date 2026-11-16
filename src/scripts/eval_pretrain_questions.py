# -*- coding: utf-8 -*-
"""
预训练基座 30 题综合探针批量生成脚本。

默认以续写方式评估预训练模型，不使用 ChatML。
"""

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

sys.path.insert(0, SRC_DIR)
os.chdir(PROJECT_ROOT)


REQUIRED_QUESTION_FIELDS = ("id", "category", "prompt")
MANUAL_FLAGS = ["重复", "乱码", "特殊token", "跑题", "事实错误", "截断", "危险回答"]
SPECIAL_TOKEN_STRINGS = ["<|pad|>", "<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """读取 UTF-8 JSONL 文件。"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} 不是合法 JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} 必须是 JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    """写入 UTF-8 JSONL 文件。"""
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_questions(
    questions: List[Dict[str, Any]],
    expected_count: Optional[int] = 30,
) -> None:
    """校验题集字段、ID 唯一性和固定 prompt 格式。"""
    if expected_count is not None and len(questions) != expected_count:
        raise ValueError(f"题集数量应为 {expected_count}，实际为 {len(questions)}")

    seen_ids = set()
    for index, question in enumerate(questions, 1):
        for field in REQUIRED_QUESTION_FIELDS:
            if not isinstance(question.get(field), str) or not question[field].strip():
                raise ValueError(f"第 {index} 题缺少有效字段: {field}")

        question_id = question["id"]
        if question_id in seen_ids:
            raise ValueError(f"题目 ID 重复: {question_id}")
        seen_ids.add(question_id)

        prompt = question["prompt"]
        if not prompt.startswith("问题：") or not prompt.rstrip().endswith("回答："):
            raise ValueError(f"{question_id} 的 prompt 必须使用“问题：...\\n回答：”格式")


def normalize_flags(value: Any) -> List[str]:
    """兼容人工填写的 list 或逗号分隔字符串 flags。"""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        parts = value.replace("，", ",").split(",")
        return [part.strip() for part in parts if part.strip()]
    return [str(value).strip()] if str(value).strip() else []


def normalize_score(value: Any) -> Optional[float]:
    """解析人工评分；空值表示未评分。"""
    if value is None or value == "":
        return None
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"非法评分: {value}") from exc
    if score < 0 or score > 5:
        raise ValueError(f"评分必须在 0-5 之间: {value}")
    return score


def score_conclusion(percent: float, flag_counts: Counter) -> str:
    """按预设阈值给出结论。"""
    redlines = []
    if flag_counts.get("重复", 0) > 3:
        redlines.append("超过 3 题严重重复")
    if flag_counts.get("特殊token", 0) > 2:
        redlines.append("超过 2 题特殊 token 泄漏")
    if redlines:
        return "触发红线，建议先排查生成/训练问题：" + "；".join(redlines)
    if percent >= 80:
        return "预训练质量较好，可继续 1024 适配或 SFT"
    if percent >= 70:
        return "可继续，但需要关注低分类别"
    if percent >= 60:
        return "边缘，建议先查数据/tokenizer/loss 曲线"
    return "不建议直接进入下一阶段，优先排查训练或数据问题"


def build_eval_metrics(args: argparse.Namespace) -> Dict[str, Any]:
    """整理训练日志中的辅助指标；不主动读取 checkpoint 的大 optimizer state。"""
    metrics = {}
    if args.eval_step is not None:
        metrics["eval_step"] = args.eval_step
    if args.train_loss is not None:
        metrics["train_loss"] = args.train_loss
    if args.test_loss is not None:
        metrics["test_loss"] = args.test_loss
        metrics["test_ppl"] = math.exp(args.test_loss) if args.test_loss < 100 else float("inf")
    return metrics


def summarize_scores(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """如果全部题目已有人工评分，则计算总分、分类均分和 flags 汇总。"""
    scores = [normalize_score(row.get("score")) for row in rows]
    if not rows or any(score is None for score in scores):
        return None

    total = float(sum(score for score in scores if score is not None))
    max_total = len(rows) * 5
    percent = total / max_total * 100 if max_total else 0.0

    category_scores = defaultdict(list)
    flag_counts = Counter()
    for row, score in zip(rows, scores):
        category_scores[row.get("category", "未分类")].append(score)
        flag_counts.update(normalize_flags(row.get("flags")))

    lowest_score = min(scores)
    lowest_rows = [
        row for row, score in zip(rows, scores)
        if score == lowest_score
    ]

    return {
        "total": total,
        "max_total": max_total,
        "percent": percent,
        "category_avg": {
            category: sum(values) / len(values)
            for category, values in sorted(category_scores.items())
        },
        "lowest_score": lowest_score,
        "lowest_ids": [row["id"] for row in lowest_rows],
        "flag_counts": flag_counts,
        "conclusion": score_conclusion(percent, flag_counts),
    }


def markdown_escape_cell(value: Any) -> str:
    """转义 Markdown 表格单元格。"""
    text = "" if value is None else str(value)
    text = text.replace("|", "\\|")
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")
    return text


def write_markdown_report(
    path: str,
    rows: List[Dict[str, Any]],
    checkpoint: str,
    generation_config: Dict[str, Any],
    eval_metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """写出包含 rubric、摘要和人工评分列的 Markdown 报告。"""
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    summary = summarize_scores(rows)
    category_counts = Counter(row.get("category", "未分类") for row in rows)

    lines = [
        "# RiverValley 1B 预训练 30 题评估报告",
        "",
        f"- checkpoint: `{checkpoint}`",
        f"- generated_at: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- question_count: `{len(rows)}`",
        f"- generation_config: `{json.dumps(generation_config, ensure_ascii=False)}`",
        "",
        "## 辅助指标",
        "",
    ]
    if eval_metrics:
        if "eval_step" in eval_metrics:
            lines.append(f"- eval_step: `{eval_metrics['eval_step']}`")
        if "train_loss" in eval_metrics:
            lines.append(f"- train_loss: `{eval_metrics['train_loss']:.4f}`")
        if "test_loss" in eval_metrics:
            lines.append(f"- test_loss: `{eval_metrics['test_loss']:.4f}`")
        if "test_ppl" in eval_metrics:
            lines.append(f"- test_ppl: `{eval_metrics['test_ppl']:.2f}`")
    else:
        lines.append("- 未提供。可从训练日志读取最终 `test loss`，运行脚本时用 `--test_loss` 填入，报告会自动计算 ppl。")

    lines.extend([
        "",
        "## 题目分布",
        "",
    ])
    for category, count in sorted(category_counts.items()):
        lines.append(f"- {category}: {count}")

    lines.extend([
        "",
        "## 评分规则",
        "",
        "- 每题 0-5 分，总分 150，百分制为 `总分 / 150 * 100`。",
        "- 5 分：准确、流畅、完整，没有明显幻觉或格式问题。",
        "- 4 分：基本正确，只有轻微遗漏、措辞或格式问题。",
        "- 3 分：部分正确，但有明显不完整、跑题或推理缺口。",
        "- 2 分：语言尚可，但主要答案错误或任务没有完成。",
        "- 1 分：大量重复、乱码、胡乱续写，只能看出少量相关内容。",
        "- 0 分：不可读、空输出、严重特殊 token 泄漏、完全无关。",
        f"- flags 可选：{'、'.join(MANUAL_FLAGS)}。",
        "",
        "## 评分摘要",
        "",
    ])

    if summary is None:
        lines.append("- 评分状态：未评分。填写 JSONL 中的 `score` 和 `flags` 后，可用 `--report_only` 重新生成摘要。")
    else:
        lines.extend([
            f"- 总分：{summary['total']:.1f} / {summary['max_total']}",
            f"- 百分制：{summary['percent']:.2f}",
            f"- 最低分：{summary['lowest_score']:.1f}，题目：{', '.join(summary['lowest_ids'])}",
            f"- 结论：{summary['conclusion']}",
            "- 分类均分：" + "；".join(
                f"{category} {avg:.2f}" for category, avg in summary["category_avg"].items()
            ),
            "- flags 汇总：" + (
                "；".join(f"{flag} {count}" for flag, count in summary["flag_counts"].most_common())
                if summary["flag_counts"] else "无"
            ),
        ])

    lines.extend([
        "",
        "## 待评分表",
        "",
        "| ID | 分类 | Prompt | Output | 自动提示 | 分数(0-5) | Flags | 备注 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for row in rows:
        lines.append(
            "| {id} | {category} | {prompt} | {output} | {auto_flags} | {score} | {flags} | {notes} |".format(
                id=markdown_escape_cell(row.get("id", "")),
                category=markdown_escape_cell(row.get("category", "")),
                prompt=markdown_escape_cell(row.get("prompt", "")),
                output=markdown_escape_cell(row.get("output", "")),
                auto_flags=markdown_escape_cell(",".join(normalize_flags(row.get("auto_flags")))),
                score=markdown_escape_cell("" if row.get("score") is None else row.get("score")),
                flags=markdown_escape_cell(",".join(normalize_flags(row.get("flags")))),
                notes=markdown_escape_cell(row.get("notes", "")),
            )
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def derive_markdown_path(out_path: str) -> str:
    """从 JSONL 输出路径推导 Markdown 路径。"""
    base, _ = os.path.splitext(out_path)
    return base + ".md"


def get_generation_eos_ids(tokenizer) -> List[int]:
    """获取生成停止 token，兼容 <|endoftext|> 和 <|im_end|>。"""
    eos_ids = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id not in eos_ids:
        eos_ids.append(im_end_id)
    return eos_ids


def load_model_and_tokenizer(model_path: str, device: Optional[str] = None):
    """按 checkpoint 加载模型和 tokenizer。"""
    import torch
    from transformers import PreTrainedTokenizerFast

    from model import RiverValleyModel

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    if device_obj.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    print(f"加载模型: {model_path}")
    print(f"推理设备: {device_obj}")
    print(f"推理精度: {dtype}")

    model = RiverValleyModel.from_pretrained(model_path, torch_dtype=dtype).to(device_obj)
    model.eval()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)

    print(f"模型加载完成，参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"词汇表大小: {tokenizer.vocab_size}")
    return model, tokenizer, device_obj


def has_repetition(text: str, window: int = 8, threshold: int = 4) -> bool:
    """粗略检测短片段重复，用作人工评分提示。"""
    compact = "".join(text.split())
    if len(compact) < window * threshold:
        return False
    counts = Counter(compact[i:i + window] for i in range(0, len(compact) - window + 1))
    return any(count >= threshold for count in counts.values())


def detect_auto_flags(output: str, input_truncated: bool) -> List[str]:
    """生成自动提示 flags，不替代人工评分。"""
    flags = []
    if input_truncated:
        flags.append("截断")
    if any(token in output for token in SPECIAL_TOKEN_STRINGS):
        flags.append("特殊token")
    if has_repetition(output):
        flags.append("重复")
    if not output.strip():
        flags.append("空输出")
    return flags


def generate_one(
    model,
    tokenizer,
    device,
    prompt: str,
    generation_config: Dict[str, Any],
) -> Dict[str, Any]:
    """对单个 prompt 生成续写结果。"""
    import torch

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    original_input_tokens = int(input_ids.size(1))
    max_seq_len = getattr(model.config, "max_seq_len", None)
    input_truncated = False
    if max_seq_len is not None and input_ids.size(1) > max_seq_len:
        input_ids = input_ids[:, -max_seq_len:]
        input_truncated = True

    gen_kwargs = {
        "max_new_tokens": generation_config["max_new_tokens"],
        "do_sample": generation_config["do_sample"],
        "repetition_penalty": generation_config["repetition_penalty"],
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": get_generation_eos_ids(tokenizer),
    }
    if generation_config["do_sample"]:
        gen_kwargs["temperature"] = generation_config["temperature"]
        gen_kwargs["top_p"] = generation_config["top_p"]

    with torch.no_grad():
        output_ids = model.generate(input_ids, **gen_kwargs)

    generated_ids = output_ids[0][input_ids.shape[1]:]
    output = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)

    finish_reason = "length"
    stop_token_id = None
    if generated_ids.numel() > 0:
        last_id = int(generated_ids[-1].item())
        if last_id in gen_kwargs["eos_token_id"]:
            finish_reason = "eos"
            stop_token_id = last_id

    return {
        "output": output,
        "full_text": full_text,
        "original_input_tokens": original_input_tokens,
        "used_input_tokens": int(input_ids.size(1)),
        "input_truncated": input_truncated,
        "generated_tokens": int(generated_ids.numel()),
        "finish_reason": finish_reason,
        "stop_token_id": stop_token_id,
        "auto_flags": detect_auto_flags(output, input_truncated),
    }


def run_generation(args: argparse.Namespace, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """批量生成评估结果。"""
    model, tokenizer, device = load_model_and_tokenizer(args.model, args.device)
    generation_config = build_generation_config(args)
    eval_metrics = build_eval_metrics(args)

    rows = []
    for index, question in enumerate(questions, 1):
        print(f"[{index:02d}/{len(questions):02d}] {question['id']} {question['category']}")
        result = generate_one(model, tokenizer, device, question["prompt"], generation_config)
        rows.append({
            "id": question["id"],
            "category": question["category"],
            "prompt": question["prompt"],
            "output": result["output"],
            "full_text": result["full_text"],
            "checkpoint": args.model,
            "generation_config": generation_config,
            "eval_metrics": eval_metrics,
            "score": None,
            "flags": [],
            "notes": "",
            "auto_flags": result["auto_flags"],
            "meta": {
                "original_input_tokens": result["original_input_tokens"],
                "used_input_tokens": result["used_input_tokens"],
                "input_truncated": result["input_truncated"],
                "generated_tokens": result["generated_tokens"],
                "finish_reason": result["finish_reason"],
                "stop_token_id": result["stop_token_id"],
            },
        })
    return rows


def build_generation_config(args: argparse.Namespace) -> Dict[str, Any]:
    """整理可追溯的生成参数。"""
    return {
        "mode": "completion",
        "chat_template": False,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量生成预训练基座 30 题评估结果")
    parser.add_argument("--model", type=str, default="/root/autodl-tmp/checkpoints_1b_512/best", help="checkpoint 目录")
    parser.add_argument("--questions", type=str, default="evals/pretrain_30_questions.jsonl", help="题集 JSONL")
    parser.add_argument("--out", type=str, default="/root/autodl-tmp/evals/pretrain_1b_512_best_outputs.jsonl", help="输出 JSONL")
    parser.add_argument("--markdown", type=str, default=None, help="输出 Markdown；默认与 --out 同名")
    parser.add_argument("--device", type=str, default=None, help="推理设备，例如 cuda/cpu；默认自动选择")
    parser.add_argument("--max_new_tokens", type=int, default=192, help="每题最大生成 token 数")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="重复惩罚")
    parser.add_argument("--do_sample", action="store_true", help="开启采样；默认贪心解码")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度，仅 --do_sample 时使用")
    parser.add_argument("--top_p", type=float, default=0.9, help="top-p，仅 --do_sample 时使用")
    parser.add_argument("--eval_step", type=int, default=None, help="训练日志中的最终评估 step，可选")
    parser.add_argument("--train_loss", type=float, default=None, help="训练日志中的最终 train loss，可选")
    parser.add_argument("--test_loss", type=float, default=None, help="训练日志中的最终 test loss，可选；报告会计算 ppl")
    parser.add_argument("--report_only", action="store_true", help="不生成文本，只从 --out 读取结果并重新生成 Markdown")
    parser.add_argument("--allow_any_count", action="store_true", help="允许题集不是 30 题，便于调试")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    markdown_path = args.markdown or derive_markdown_path(args.out)
    expected_count = None if args.allow_any_count else 30
    generation_config = build_generation_config(args)
    eval_metrics = build_eval_metrics(args)

    if args.report_only:
        rows = read_jsonl(args.out)
        checkpoint = rows[0].get("checkpoint", args.model) if rows else args.model
        if rows and isinstance(rows[0].get("generation_config"), dict):
            generation_config = rows[0]["generation_config"]
        if rows and isinstance(rows[0].get("eval_metrics"), dict):
            eval_metrics = rows[0]["eval_metrics"]
        write_markdown_report(markdown_path, rows, checkpoint, generation_config, eval_metrics)
        print(f"Markdown 报告已写入: {markdown_path}")
        return

    questions = read_jsonl(args.questions)
    validate_questions(questions, expected_count=expected_count)
    rows = run_generation(args, questions)
    write_jsonl(args.out, rows)
    write_markdown_report(markdown_path, rows, args.model, generation_config, eval_metrics)
    print(f"JSONL 结果已写入: {args.out}")
    print(f"Markdown 报告已写入: {markdown_path}")


if __name__ == "__main__":
    main()
