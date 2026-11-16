"""验证新输出头、正式训练参数及中英文材料的独立性。"""
import copy
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src/scripts"), str(ROOT / "datasets")]
from config import RiverValleyConfig, PretrainConfig
from model import RiverValleyModel
from training_utils import train_update, validate_training_options, prune_interval_checkpoints
from eval_bilingual import summarize


@pytest.mark.parametrize("chunk_size", [1, 7, 100])
def test_chunked_head_matches_tied_weight_gradients(chunk_size):
    """包含全屏蔽微批和余数分块，检查共享 embedding/head 的最终更新。"""
    torch.manual_seed(19)
    config = RiverValleyConfig(vocab_size=64, n_embed=24, n_heads=3, n_kv_heads=1,
                              head_dim=8, intermediate_size=48, n_layers=2, max_seq_len=16)
    baseline = RiverValleyModel(config)
    chunked = copy.deepcopy(baseline)
    chunked.config.loss_chunk_size = chunk_size
    x, y = torch.randint(4, 64, (3, 9)), torch.randint(4, 64, (3, 9))
    mask = torch.ones_like(x)
    mask[0] = 0
    mask[1, :5] = 0
    batches = [tuple(t[i:i + 1] for t in (x, y, mask, torch.ones_like(x))) for i in range(3)]
    results = []
    for model in (baseline, chunked):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        results.append(train_update(model, optimizer, batches, torch.device("cpu"), 100))
    assert results[0]["loss"] == pytest.approx(results[1]["loss"], rel=2e-6)
    for a, b in zip(baseline.parameters(), chunked.parameters()):
        torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-5)
    with torch.no_grad():
        assert chunked(x).logits.shape == (3, 9, 64)
        assert chunked(x, loss_targets=y).logits is None


@pytest.mark.parametrize("options", [{"warmup_ratio": 1}, {"min_lr": 1}, {"adam_beta2": 1},
                                     {"schedule_steps": 0}, {"grad_clip": 0}, {"learning_rate": float("nan")}])
def test_invalid_training_plan_fails_early(options):
    with pytest.raises(ValueError):
        validate_training_options(PretrainConfig(**options))


def test_checkpoint_retention_preserves_non_owned_directories(tmp_path):
    for name in ["ckpt_1", "ckpt_20", "ckpt_100", "best", "latest", "ckpt_manual"]:
        (tmp_path / name).mkdir()
    prune_interval_checkpoints(tmp_path, 2)
    assert {p.name for p in tmp_path.iterdir()} == {"ckpt_20", "ckpt_100", "best", "latest", "ckpt_manual"}


def test_unreviewed_answers_cannot_be_reported_as_complete():
    report = summarize([{"language": "zh", "answer": "有文字不等于正确", "score": None},
                        {"language": "en", "answer": "7", "score": 2}])
    assert not report["all_reviewed"]
    assert not report["languages"]["zh"]["review_complete"]


def test_corpus_dedup_language_and_contamination():
    pytest.importorskip("pyarrow")
    from prepare_bilingual import NearDeduplicator, language_ok, reject_reason, normalized
    text = "小溪旁边有一座图书馆，人们可以在这里借书，阅读故事并了解自然。" * 9
    index = NearDeduplicator()
    assert index.add(text)[1] is None
    assert index.add(" \n" + text + " ")[1] == "exact_duplicate"
    assert language_ok(text, "zh") and not language_ok(text, "en")
    probe = normalized("图书馆，人们可以在这里借书")
    assert reject_reason(text, "zh", [probe]) == "evaluation_overlap"


def test_acceptance_set_is_balanced_and_unique():
    rows = [json.loads(line) for line in (ROOT / "evals/bilingual_acceptance_v1.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len({row["id"] for row in rows}) == 100
    assert sum(row["language"] == "zh" for row in rows) == 50
    assert all(row["prompt"] and row["reference"] for row in rows)


def test_compiled_chunked_head_matches_eager():
    """在 CPU AOTAutograd 验证重算与编译组合；不把它当作 CUDA 吞吐证明。"""
    torch.manual_seed(13)
    config = RiverValleyConfig(vocab_size=32, n_embed=16, n_heads=2, n_kv_heads=1,
                              head_dim=8, intermediate_size=32, n_layers=1, max_seq_len=8,
                              loss_chunk_size=3)
    eager = RiverValleyModel(config)
    compiled = torch.compile(copy.deepcopy(eager), backend="aot_eager")
    x = torch.randint(4, 32, (1, 7))
    labels = x.clone()
    labels[:, :2] = -100
    first, second = eager(x, loss_targets=labels).loss, compiled(x, loss_targets=labels).loss
    first.backward()
    second.backward()
    torch.testing.assert_close(first, second)
    for a, b in zip(eager.parameters(), compiled.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-5)


def test_memmap_rejects_same_size_content_change(tmp_path):
    """即使字节数没变，换过 token 的文件也不能继续沿用旧 metadata。"""
    import hashlib
    import numpy as np
    from data.dataset import MemmapDataset
    data = np.arange(12, dtype=np.uint16)
    path = tmp_path / "train.bin"
    data.tofile(path)
    metadata = tmp_path / "meta.json"
    metadata.write_text(json.dumps({"num_samples": 3, "seq_len": 4,
        "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}), encoding="utf-8")
    assert len(MemmapDataset(str(path), str(metadata))) == 3
    data[5] = 2
    data.tofile(path)
    with pytest.raises(ValueError, match="内容校验失败"):
        MemmapDataset(str(path), str(metadata))


def test_instruction_adapter_preserves_context_and_provenance():
    """指令源的 input 不能丢掉，否则抽取任务会变成无条件猜答案。"""
    pytest.importorskip("pyarrow")
    from prepare_bilingual import sft_record
    row = {"instruction": "请提取数字", "input": "这里有 12 本书", "output": "12",
           "copyright": "来源说明", "human_verified": False}
    result = sft_record(row, "sft_zh_cqia_coig_pc_0000.parquet", "zh")
    assert result["conversations"][0]["content"] == "请提取数字\n\n这里有 12 本书"
    assert result["conversations"][1]["content"] == "12"
    assert result["source"] == "m-a-p/COIG-CQIA/coig_pc"
    assert result["source_copyright"] == "来源说明"
    assert result["source_human_verified"] is False


def test_first_turn_adapter_rejects_incomplete_answer():
    pytest.importorskip("pyarrow")
    from prepare_bilingual import sft_record
    assert sft_record({"messages": [{"role": "user", "content": "你好"}]}, "sft_en_0000.parquet", "en") is None
    result = sft_record({"messages": [{"role": "system", "content": "请简短回答"},
                                      {"role": "user", "content": "你好"},
                                      {"role": "assistant", "content": "你好呀"},
                                      {"role": "user", "content": "第二问"}]}, "sft_en_0000.parquet", "en")
    assert len(result["conversations"]) == 3


def test_partial_evaluation_cannot_masquerade_as_complete():
    from eval_bilingual import validate_evaluation_rows
    questions = [{"id": "zh_1", "prompt": "问题一", "language": "zh"},
                 {"id": "en_1", "prompt": "Question one", "language": "en"}]
    complete = [{**row, "question_sha256": "固定摘要", "score": 2} for row in questions]
    validate_evaluation_rows(complete, questions, "固定摘要")
    with pytest.raises(ValueError, match="题目缺失"):
        validate_evaluation_rows(complete[:1], questions, "固定摘要")
    complete[0]["prompt"] = "更换问题"
    with pytest.raises(ValueError, match="内容被修改"):
        validate_evaluation_rows(complete, questions, "固定摘要")


def test_sharegpt_adapter_and_short_qa_scope_filters():
    pytest.importorskip("pyarrow")
    from prepare_bilingual import sft_record, sft_reject_reason
    row = {"conversations": [{"from": "human", "value": "云是什么？"},
                              {"from": "gpt", "value": "云由悬浮的小水滴或冰晶组成。"}]}
    record = sft_record(row, "sft_zh_alpaca_gpt4_0000.parquet", "zh")
    assert record["group"] == "alpaca_gpt4_zh"
    assert sft_reject_reason(record) is None
    record["conversations"][1]["content"] = "这个疾病需要确诊以后才能决定用药。"
    assert sft_reject_reason(record) == "scope_or_noise"


def test_answer_already_inside_prompt_is_filtered():
    pytest.importorskip("pyarrow")
    from prepare_bilingual import sft_reject_reason
    answer = "请先明确目标，再安排时间，并通过反馈调整下一次学习任务。" * 3
    record = {"group": "smoltalk_zh", "language": "zh", "conversations": [
        {"role": "user", "content": "应该如何安排学习？\n" + answer},
        {"role": "assistant", "content": answer}]}
    assert sft_reject_reason(record) == "answer_inside_prompt"


def test_latin_script_does_not_imply_english():
    pytest.importorskip("langid")
    from langid.langid import LanguageIdentifier, model
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    identifier.set_languages(["en", "zh", "ja", "ko", "fr", "de", "it", "pt", "es", "ru", "ar"])
    assert identifier.classify("Per tradurre la frase dall'arabo all'italiano, è necessario seguire questi passaggi.")[0] != "en"


def test_review_filters_are_order_independent_and_drop_broken_tasks():
    pytest.importorskip("pyarrow")
    from prepare_bilingual import finalize_sft_rows
    rows = []
    for i, (language, question) in enumerate([("zh", "什么是云？"), ("zh", "请解释下雨。"),
                                             ("zh", "为以下产品提供描述词："), ("zh", "将中文单词按字母排序。"),
                                             ("en", "What is rain?"), ("en", "Role: Play a robot.")]):
        rows.append({"id": str(i), "language": language, "group": "alpaca_gpt4_zh" if language == "zh" else "smol",
                     "conversations": [{"role": "user", "content": question}, {"role": "assistant", "content": "回答"}]})
    first, stats = finalize_sft_rows(rows, 1337)
    second, _ = finalize_sft_rows(list(reversed(rows)), 1337)
    assert first == second
    assert not {"2", "3", "5"}.intersection(r["id"] for r in first)
    assert stats["review_zh_missing_context"] == 1
