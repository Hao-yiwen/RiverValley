"""验证因果性、监督标签、累计梯度和中断恢复的真实行为。"""
import copy
import json
import os
import sys
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src/scripts")]
from config import RiverValleyConfig, SFTConfig
from model import RiverValleyModel
from data.dataset import SFTDataset
from training_utils import (ResumableBatchSampler, trim_padding_collate, token_loss_sum,
                            train_update, evaluate_loss, validate_resume)
from tokenizer import CHAT_TEMPLATE
from pretrain import save_model_hf
from sft import main as sft_main


def tiny_config(**kwargs):
    return RiverValleyConfig(vocab_size=96, n_embed=32, n_heads=4, n_kv_heads=2,
                            head_dim=8, intermediate_size=64, n_layers=2, max_seq_len=128, **kwargs)


@pytest.mark.parametrize("masked", [False, True])
def test_cached_chunk_matches_full_causal_forward(masked):
    """缓存后一次追加多个 token，不得看到同一块中的未来 token。"""
    torch.manual_seed(7)
    model = RiverValleyModel(tiny_config()).eval()
    tokens = torch.randint(4, 96, (2, 11))
    attention = torch.ones_like(tokens) if masked else None
    with torch.no_grad():
        full = model(tokens, attention_mask=attention).logits
        prefix = model(tokens[:, :6], attention_mask=attention[:, :6] if masked else None, use_cache=True)
        chunk = model(tokens[:, 6:], attention_mask=attention, past_key_values=prefix.past_key_values).logits
    torch.testing.assert_close(chunk, full[:, 6:], atol=2e-6, rtol=2e-5)


def test_accumulation_matches_one_batch_with_unequal_answer_lengths():
    """累计不同答案长度的微批，梯度应等于一次处理完整有效批量。"""
    torch.manual_seed(5)
    first = RiverValleyModel(tiny_config())
    second = copy.deepcopy(first)
    x = torch.randint(4, 96, (2, 9)); y = torch.randint(4, 96, (2, 9))
    mask = torch.ones_like(x); mask[0, :7] = 0
    batch = (x, y, mask, torch.ones_like(x))
    opt1, opt2 = [torch.optim.SGD(model.parameters(), lr=0.1) for model in (first, second)]
    train_update(first, opt1, [tuple(v[i:i+1] for v in batch) for i in range(2)], torch.device("cpu"), 100)
    train_update(second, opt2, [batch], torch.device("cpu"), 100)
    for a, b in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-5)


def test_sampler_resume_is_exact_and_ranks_do_not_overlap():
    samplers = [ResumableBatchSampler(59, 2, 3, seed=17, rank=rank, world_size=2) for rank in range(2)]
    batches = [list(sampler) for sampler in samplers]
    assert set(sum(batches[0], [])).isdisjoint(sum(batches[1], []))
    samplers[0].set_step(2)
    assert list(samplers[0]) == batches[0][6:]
    samplers[0].set_step(samplers[0].steps_per_epoch)
    assert list(samplers[0]) != batches[0]


def make_tokenizer(texts):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    from tokenizers import decoders
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=96,
        special_tokens=["<|pad|>", "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<unk>"]))
    return tokenizer


def test_sft_full_encoding_and_dynamic_padding_match():
    sample = {"conversations": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "yes\nthanks"}]}
    text = "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\nyes\nthanks<|im_end|>\n"
    tokenizer = make_tokenizer([text] * 5)
    ds = SFTDataset([sample], tokenizer, 128, 0)
    x, y, mask, attention = ds[0]
    expected = tokenizer.encode(text).ids
    assert x[:len(expected)].tolist() == expected
    trimmed = trim_padding_collate([ds[0]])
    assert trimmed[0].shape[1] == len(expected)
    assert trimmed[2].sum() == mask.sum()
    model = RiverValleyModel(tiny_config()).eval()
    with torch.no_grad():
        full = model(x[None], attention_mask=attention[None]).logits
        short = model(trimmed[0], attention_mask=trimmed[3]).logits
    torch.testing.assert_close(full[:, :len(expected)], short, atol=2e-6, rtol=2e-5)


def test_sft_filters_missing_answers():
    tokenizer = make_tokenizer(["hello"])
    data = [{"conversations": [{"role": "user", "content": "only question"}]},
            {"conversations": [{"role": "assistant", "content": " "}]},
            {"conversations": [{"role": "assistant", "content": 5}]}]
    assert len(SFTDataset(data, tokenizer, 64, 0)) == 0


def test_no_supervision_aborts_update():
    model = RiverValleyModel(tiny_config())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.ones((1, 4), dtype=torch.long)
    with pytest.raises(ValueError, match="监督 token"):
        train_update(model, optimizer, [(x, x, x * 0, x)], torch.device("cpu"), 1)


def test_gradient_checkpointing_matches_gradients():
    torch.manual_seed(11)
    first = RiverValleyModel(tiny_config())
    second = copy.deepcopy(first); second.gradient_checkpointing = True
    x = torch.randint(4, 96, (2, 10))
    for model in (first, second):
        model(x, labels=x).loss.backward()
    for a, b in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("external_stop", [False, True])
def test_sft_stop_resume_matches_uninterrupted(tmp_path, monkeypatch, external_stop):
    """穿过真实 SFT 入口、HF 保存和 AdamW 恢复，验证含 dropout 的连续训练一致性。"""
    torch.set_num_threads(1)
    samples = [{"conversations": [{"role": "user", "content": f"question {i}"},
                                  {"role": "assistant", "content": f"answer {i}"}]} for i in range(12)]
    tokenizer = make_tokenizer([json.dumps(samples)] + ["<|im_start|>user\nquestion\nassistant\nanswer<|im_end|>"])
    tokenizer_path = tmp_path / "tokenizer.json"; tokenizer.save(str(tokenizer_path))
    data_path = tmp_path / "data.jsonl"
    data_path.write_text("\n".join(json.dumps(s) for s in samples), encoding="utf-8")
    config = tiny_config(dropout=0.1); config.vocab_size = tokenizer.get_vocab_size()
    model = RiverValleyModel(config)
    initial = tmp_path / "initial"; save_model_hf(model, tokenizer, str(initial))
    common = dict(data_path=str(data_path), bpe_model_path=str(tokenizer_path),
                  pretrain_checkpoint=str(initial), batch_size=2, accumulation_steps=1,
                  max_seq_len=64, epochs=2, learning_rate=1e-3, min_lr=1e-4, warmup_ratio=0,
                  num_workers=0, pin_memory=False, use_swanlab=False, device="cpu",
                  eval_iters=2, eval_interval=100, save_interval=100, log_interval=0)
    whole = tmp_path / "whole"; split = tmp_path / "split"
    sft_main(SFTConfig(**common, checkpoint_dir=str(whole), max_steps=4))
    with monkeypatch.context() as patch:
        if external_stop:
            from training_runtime import TrainingStopper
            calls = iter([False, False, True])
            patch.setattr(TrainingStopper, "requested", lambda self: next(calls))
        sft_main(SFTConfig(**common, checkpoint_dir=str(split), max_steps=4 if external_stop else 2))
    assert torch.load(split / "latest/training_state.pt", weights_only=False)["next_iter"] == 2
    sft_main(SFTConfig(**common, checkpoint_dir=str(split), max_steps=4,
                       resume=True, resume_path=str(split / "latest")))
    a, b = [RiverValleyModel.from_pretrained(str(path / "latest")) for path in (whole, split)]
    for name, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[name], rtol=0, atol=0)
    state = torch.load(split / "latest/training_state.pt", weights_only=False)
    assert state["next_iter"] == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA 验证原生 GQA")
def test_native_gqa_cuda_forward_backward():
    torch.manual_seed(31)
    first = RiverValleyModel(tiny_config()).cuda()
    second = copy.deepcopy(first)
    for block in second.blocks:
        block.sa.use_native_gqa = True
    x = torch.randint(4, 96, (2, 17), device="cuda")
    for model in (first, second):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(x, labels=x).loss.backward()
    for a, b in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=2e-3, rtol=0.05)


@pytest.mark.parametrize("external_stop", [False, True])
def test_pretrain_stop_resume_matches_uninterrupted(tmp_path, monkeypatch, external_stop):
    """预训练也必须跨保存恢复模型、AdamW、dropout 和 memmap 游标。"""
    import numpy as np
    from config import PretrainConfig
    from pretrain import main as pretrain_main
    torch.set_num_threads(1)
    tokenizer = make_tokenizer(["the quick brown fox and the small river valley"])
    tokenizer_path = tmp_path / "tokenizer.json"; tokenizer.save(str(tokenizer_path))
    memmap = tmp_path / "memmap"; memmap.mkdir()
    np.random.default_rng(7).integers(5, tokenizer.get_vocab_size(), (40, 17), dtype=np.uint16).tofile(memmap / "train.bin")
    (memmap / "meta.json").write_text(json.dumps({"num_samples": 40, "seq_len": 17,
        "train_seq_len": 16, "vocab_size": tokenizer.get_vocab_size(), "pad_id": 0,
        "packed": True, "partial_blocks": 0}))
    architecture = tmp_path / "architecture.json"
    architecture.write_text(json.dumps({"n_embed": 32, "n_heads": 4, "n_kv_heads": 2,
        "head_dim": 8, "intermediate_size": 64, "n_layers": 2, "dropout": 0.1}))
    common = dict(memmap_dir=str(memmap), bpe_model_path=str(tokenizer_path),
        model_config_path=str(architecture), batch_size=2, accumulation_steps=1,
        max_seq_len=16, epochs=2, learning_rate=1e-3, min_lr=1e-4, warmup_ratio=0,
        num_workers=0, pin_memory=False, use_swanlab=False, device="cpu",
        eval_iters=2, eval_interval=100, save_interval=100, log_interval=0)
    whole = tmp_path / "whole"; split = tmp_path / "split"
    pretrain_main(PretrainConfig(**common, checkpoint_dir=str(whole), max_steps=4))
    with monkeypatch.context() as patch:
        if external_stop:
            from training_runtime import TrainingStopper
            calls = iter([False, False, True])
            patch.setattr(TrainingStopper, "requested", lambda self: next(calls))
        pretrain_main(PretrainConfig(**common, checkpoint_dir=str(split), max_steps=4 if external_stop else 2))
    assert torch.load(split / "latest/training_state.pt", weights_only=False)["next_iter"] == 2
    pretrain_main(PretrainConfig(**common, checkpoint_dir=str(split), max_steps=4,
        resume=True, resume_path=str(split / "latest")))
    a, b = [RiverValleyModel.from_pretrained(str(path / "latest")) for path in (whole, split)]
    for name, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[name], rtol=0, atol=0)


def test_new_pretrain_stage_preserves_adam_and_resumes(tmp_path):
    """穿过真实入口验证新语料游标从零开始，但 Adam 步数延续，阶段内恢复完全一致。"""
    import numpy as np
    from config import PretrainConfig
    from pretrain import main as pretrain_main
    torch.set_num_threads(1)
    tokenizer = make_tokenizer(['the quick brown fox and the small river valley'])
    token_path = tmp_path / 'tokenizer.json'; tokenizer.save(str(token_path))
    for version in (1, 2):
        folder = tmp_path / f'data{version}'; folder.mkdir()
        np.random.default_rng(version).integers(5, tokenizer.get_vocab_size(), (40, 17), dtype=np.uint16).tofile(folder / 'train.bin')
        (folder / 'meta.json').write_text(json.dumps({'num_samples': 40, 'seq_len': 17,
            'train_seq_len': 16, 'vocab_size': tokenizer.get_vocab_size(), 'pad_id': 0,
            'packed': True, 'partial_blocks': 0, 'data_version': version}))
    architecture = tmp_path / 'architecture.json'
    architecture.write_text(json.dumps({'n_embed': 32, 'n_heads': 4, 'n_kv_heads': 2,
        'head_dim': 8, 'intermediate_size': 64, 'n_layers': 2, 'dropout': .1}))
    common = dict(bpe_model_path=str(token_path), model_config_path=str(architecture), batch_size=2,
        accumulation_steps=1, max_seq_len=16, epochs=1, learning_rate=1e-3, min_lr=1e-4,
        warmup_ratio=0, num_workers=0, pin_memory=False, use_swanlab=False, device='cpu',
        eval_iters=2, eval_interval=100, save_interval=100, log_interval=0)
    old = tmp_path / 'old'
    pretrain_main(PretrainConfig(**common, memmap_dir=str(tmp_path / 'data1'),
        checkpoint_dir=str(old), schedule_steps=2, max_steps=2))
    new = dict(common, memmap_dir=str(tmp_path / 'data2'), schedule_steps=4, resume=True)
    whole, split = tmp_path / 'whole_new', tmp_path / 'split_new'
    for target, stop in [(whole, 4), (split, 2)]:
        pretrain_main(PretrainConfig(**new, checkpoint_dir=str(target), max_steps=stop,
            resume_path=str(old / 'latest'), new_stage_from_complete=True, initial_checkpoint_expected_step=2))
    state = torch.load(split / 'latest/training_state.pt', weights_only=False)
    assert state['next_iter'] == 2 and state['stage_origin']['source_completed_steps'] == 2
    assert all(item['step'].item() == 4 for item in state['optimizer']['state'].values())
    pretrain_main(PretrainConfig(**new, checkpoint_dir=str(split), max_steps=4, resume_path=str(split / 'latest')))
    first, second = [RiverValleyModel.from_pretrained(str(path / 'latest')) for path in (whole, split)]
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)
    state = torch.load(split / 'latest/training_state.pt', weights_only=False)
    assert state['next_iter'] == 4
    assert all(item['step'].item() == 6 for item in state['optimizer']['state'].values())


def _ddp_update_worker(rank, init_path, output):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_path}", rank=rank, world_size=2)
    try:
        torch.manual_seed(41)
        model = DistributedDataParallel(RiverValleyModel(tiny_config()))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        generator = torch.Generator().manual_seed(19)
        x = torch.randint(4, 96, (4, 7), generator=generator)
        y = torch.randint(4, 96, (4, 7), generator=generator)
        mask = torch.ones_like(x); mask[0, :5] = 0; mask[2, :3] = 0
        batch = (x, y, mask, torch.ones_like(x))
        train_update(model, optimizer, [tuple(value[i:i+1] for value in batch)
                     for i in (rank, rank + 2)], torch.device("cpu"), 100)
        torch.save(model.module.state_dict(), Path(output) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_ddp_unequal_token_counts_match_global_batch(tmp_path):
    """两个 CPU rank 的答案长度不同，DDP 平均后仍应等价于全局 token 损失。"""
    import torch.multiprocessing as mp
    mp.spawn(_ddp_update_worker, args=(str(tmp_path / "ddp_init"), str(tmp_path)), nprocs=2, join=True)
    torch.manual_seed(41)
    model = RiverValleyModel(tiny_config())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    generator = torch.Generator().manual_seed(19)
    x = torch.randint(4, 96, (4, 7), generator=generator)
    y = torch.randint(4, 96, (4, 7), generator=generator)
    mask = torch.ones_like(x); mask[0, :5] = 0; mask[2, :3] = 0
    train_update(model, optimizer, [(x, y, mask, torch.ones_like(x))], torch.device("cpu"), 100)
    for rank in range(2):
        saved = torch.load(tmp_path / f"rank{rank}.pt", weights_only=True)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, saved[name], atol=2e-7, rtol=2e-5)


def test_single_prompt_uses_chat_template_and_returns_answer_only():
    from inference import generate
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=make_tokenizer(["hello\nyes assistant user"]),
        eos_token="<|endoftext|>", pad_token="<|pad|>")
    tokenizer.chat_template = CHAT_TEMPLATE

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight = torch.nn.Parameter(torch.zeros(1))
            from types import SimpleNamespace
            self.config = SimpleNamespace(max_seq_len=1024)

        def generate(self, ids, **kwargs):
            self.input = ids
            assert kwargs["do_sample"] is False
            return torch.cat([ids, torch.tensor([tokenizer.encode("yes", add_special_tokens=False)])], dim=1)

    model = Model()
    assert generate(model, tokenizer, "hello", chat=True, do_sample=False) == "yes"
    assert tokenizer.decode(model.input[0]).endswith("<|im_start|>assistant\n")
    messages = [{'role': 'user', 'content': 'hello'}, {'role': 'assistant', 'content': 'yes'},
                {'role': 'user', 'content': 'hello again'}]
    result = generate(model, tokenizer, 'hello again', chat=True, do_sample=False,
                      messages=messages, return_metadata=True)
    expected = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')
    torch.testing.assert_close(model.input, expected)
    assert result['answer'] == 'yes' and result['generated_tokens'] > 0
    with pytest.raises(ValueError, match='上下文'):
        generate(model, tokenizer, 'hello', chat=True, max_tokens=1024)


def test_rejects_same_size_but_different_tokenizer(tmp_path):
    from training_utils import validate_checkpoint_tokenizer
    tokenizer = make_tokenizer(["hello world"])
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    saved = json.loads((tmp_path / "tokenizer.json").read_text())
    vocab = saved["model"]["vocab"]
    keys = [key for key, value in vocab.items() if value > 4][:2]
    vocab[keys[0]], vocab[keys[1]] = vocab[keys[1]], vocab[keys[0]]
    (tmp_path / "tokenizer.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="tokenizer"):
        validate_checkpoint_tokenizer(tmp_path, tokenizer)


def test_failed_checkpoint_write_preserves_previous_complete_state(tmp_path):
    from training_utils import write_checkpoint_directory, resolve_checkpoint_path
    target = tmp_path / "latest"
    write_checkpoint_directory(target, lambda path: (Path(path) / "value").write_text("first"))
    def broken_writer(path):
        (Path(path) / "value").write_text("partial")
        raise OSError("模拟磁盘写入失败")
    with pytest.raises(OSError):
        write_checkpoint_directory(target, broken_writer)
    assert (target / "value").read_text() == "first"
    write_checkpoint_directory(target, lambda path: (Path(path) / "value").write_text("second"))
    assert (target / "value").read_text() == "second"
    assert (tmp_path / "latest.previous/value").read_text() == "first"
    assert resolve_checkpoint_path(tmp_path / "absent") == str(tmp_path / "absent")
    (tmp_path / "recover.previous").mkdir()
    assert resolve_checkpoint_path(tmp_path / "recover") == str(tmp_path / "recover.previous")
