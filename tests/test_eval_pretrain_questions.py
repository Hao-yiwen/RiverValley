# -*- coding: utf-8 -*-
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "scripts"))

from eval_pretrain_questions import (
    read_jsonl,
    summarize_scores,
    validate_questions,
    write_markdown_report,
)


def test_pretrain_question_set_has_expected_distribution():
    """固定 30 题探针集应保持类别分布稳定。"""
    questions = read_jsonl(os.path.join(ROOT, "evals", "pretrain_30_questions.jsonl"))

    validate_questions(questions, expected_count=30)

    counts = {}
    for question in questions:
        counts[question["category"]] = counts.get(question["category"], 0) + 1

    assert counts == {
        "中文表达": 5,
        "英文翻译": 5,
        "数学逻辑": 5,
        "常识科普": 5,
        "代码技术": 5,
        "长文本鲁棒性": 5,
    }


def test_scored_report_summary_and_markdown(tmp_path):
    """人工评分字段填好后，报告应能给出总分、分类均分和结论。"""
    rows = [
        {
            "id": "a",
            "category": "中文表达",
            "prompt": "问题：A\n回答：",
            "output": "答案 A",
            "score": 5,
            "flags": [],
            "auto_flags": [],
            "notes": "",
        },
        {
            "id": "b",
            "category": "数学逻辑",
            "prompt": "问题：B\n回答：",
            "output": "答案 B",
            "score": 3,
            "flags": ["事实错误"],
            "auto_flags": ["重复"],
            "notes": "示例备注",
        },
    ]

    summary = summarize_scores(rows)
    assert summary["total"] == 8
    assert summary["max_total"] == 10
    assert summary["percent"] == 80
    assert summary["category_avg"] == {"中文表达": 5, "数学逻辑": 3}
    assert summary["flag_counts"]["事实错误"] == 1
    assert summary["lowest_ids"] == ["b"]

    report_path = tmp_path / "report.md"
    write_markdown_report(
        str(report_path),
        rows,
        checkpoint="/tmp/checkpoint",
        generation_config={"max_new_tokens": 192, "do_sample": False},
        eval_metrics={"test_loss": 2.0, "test_ppl": 7.389},
    )

    report = report_path.read_text(encoding="utf-8")
    assert "总分：8.0 / 10" in report
    assert "百分制：80.00" in report
    assert "test_loss: `2.0000`" in report
    assert "test_ppl: `7.39`" in report
    assert "预训练质量较好" in report
    assert "| b | 数学逻辑 |" in report
