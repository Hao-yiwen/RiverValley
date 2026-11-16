"""用冻结的中英文验收题生成回答；自动检查与人工质量评分分开记录。"""
import argparse
from collections import defaultdict
import hashlib
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src/scripts")]
logger = logging.getLogger(__name__)


def validate_evaluation_rows(rows, questions, question_hash):
    """生成中断或人工评分时误删题目，不能让部分结果冒充完整验收。"""
    expected = {row["id"]: row for row in questions}
    if len(rows) != len(expected) or {row["id"] for row in rows} != set(expected):
        raise ValueError("评估结果题目缺失、重复或多出，不能汇总为完整验收")
    for row in rows:
        if row.get("question_sha256") != question_hash:
            raise ValueError("评估题摘要不一致，请使用生成回答时的冻结题集")
        if any(row.get(key) != expected[row["id"]].get(key)
               for key in ("prompt", "reference", "scoring", "language", "messages")):
            raise ValueError(f"评估题内容被修改: {row['id']}")


def automatic_score(answer, reference, scoring):
    """只自动判定完全可核验的格式题；开放题保持未评分。"""
    if not answer.strip():
        return 0
    if scoring == 'exact':
        return 2 if answer.strip() == reference else 0
    if scoring == 'json':
        def unique(pairs):
            if len(dict(pairs)) != len(pairs):
                raise ValueError('重复 JSON 键')
            return dict(pairs)
        try:
            actual = json.loads(answer, object_pairs_hook=unique)
            expected = json.loads(reference)
            canonical = lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
            return 2 if canonical(actual) == canonical(expected) else 0
        except (TypeError, ValueError):
            return 0
    return None


def summarize(rows):
    """人工题未评完不产生通过结论；精确题只用于受约束的小任务。"""
    groups = defaultdict(list)
    for row in rows:
        groups[row["language"]].append(row)
    result = {}
    for language, items in groups.items():
        scored = [row for row in items if row.get("score") is not None]
        if any(row["score"] not in (0, 1, 2) for row in scored):
            raise ValueError("人工评分只能为 0、1、2 或 null")
        result[language] = {"questions": len(items), "scored": len(scored),
            "blank": sum(not row.get("answer", "").strip() for row in items),
            "score_fraction": sum(row["score"] for row in scored) / (2 * len(scored)) if scored else None,
            "review_complete": len(scored) == len(items)}
    return {"languages": result, "all_reviewed": all(value["review_complete"] for value in result.values()) if result else False,
            "note": "这是项目内基础能力验收，不能等同于通用能力排行榜；翻译成另一语言的题按目标语言判断。"}


def main():
    parser = argparse.ArgumentParser(description="运行中英文 ChatML 问答验收或汇总人工评分")
    parser.add_argument("--model")
    parser.add_argument("--questions", default="evals/bilingual_acceptance_v1.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--device")
    args = parser.parse_args()
    output = Path(args.output)
    if not args.score_only:
        from colab_overnight import defer_pipeline_evaluation
        if defer_pipeline_evaluation(output):
            return
    questions = Path(args.questions)
    digest = hashlib.sha256(questions.read_bytes()).hexdigest()
    question_rows = [json.loads(line) for line in questions.read_text(encoding="utf-8").splitlines()]
    if args.score_only:
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    else:
        if not args.model:
            parser.error("生成回答需要 --model")
        if output.exists():
            raise FileExistsError(f"结果已存在，请使用新文件名保留旧评估: {output}")
        from inference import load_model, generate
        rows = [dict(row) for row in question_rows]
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError("验收题 ID 不唯一")
        model, tokenizer = load_model(args.model, args.device)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as sink:
            for row in rows:
                result = generate(model, tokenizer, row["prompt"], max_tokens=160,
                                  chat=True, do_sample=False, repetition_penalty=1.0,
                                  messages=row.get('messages'), return_metadata=True)
                answer = result['answer']
                row.update(**result, score=automatic_score(answer, row['reference'], row['scoring']),
                           question_sha256=digest, model=args.model)
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.flush()
                logger.info("完成问答: id=%s，语言=%s，回答字符=%d，评分=%s", row["id"], row["language"], len(answer), row["score"])
    validate_evaluation_rows(rows, question_rows, digest)
    result = summarize(rows)
    output.with_suffix(".summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
