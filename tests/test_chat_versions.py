"""检查交付注册与切换边界，防止加载未完成权重或打断正在生成的回答。"""
import json
from pathlib import Path
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts'))
from chat_versions import discover_deliveries, VersionedEngine


def delivery(root, name, status='weights_downloaded_verified'):
    folder = root / name
    for relative in ['sft_best/model.safetensors', 'sft_best/tokenizer.json', 'sft_best/config.json', 'code/river-valley/src/model/gpt.py']:
        path = folder / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('测试文件')
    (folder / 'delivery_receipt.json').write_text(json.dumps({'status': status, 'sha256': 'a' * 64, 'best_sft_step': 980}))
    return folder


def test_only_complete_verified_deliveries_appear(tmp_path):
    delivery(tmp_path, 'delivery_valid')
    partial = delivery(tmp_path, 'delivery_partial')
    (partial / 'sft_best/model.safetensors').unlink()
    delivery(tmp_path, 'delivery_running', status='training')
    versions = discover_deliveries(tmp_path)
    assert list(versions) == ['delivery_valid']
    assert versions['delivery_valid']['sft_step'] == 980


def test_unknown_and_busy_version_switches_never_start_process(tmp_path):
    delivery(tmp_path, 'delivery_valid')
    engine = object.__new__(VersionedEngine)
    engine.root, engine.lock = tmp_path, threading.RLock()
    engine.versions = discover_deliveries(tmp_path)
    engine.cached = {'state': 'ready'}
    engine.active = {'id': 'busy'}
    engine.current = 'delivery_valid'
    with pytest.raises(ValueError, match='完整下载'):
        engine.switch('delivery_missing')
    with pytest.raises(RuntimeError, match='回答结束'):
        engine.switch('delivery_valid')
    engine.active = None
    assert engine.switch('delivery_valid') is None


def test_model_status_does_not_invent_sft_step(tmp_path):
    from local_chat import ChatEngine
    model = tmp_path / 'model'
    model.mkdir()
    assert ChatEngine(model, tmp_path, 'cpu').status()['sft_step'] is None
    (tmp_path / 'delivery_receipt.json').write_text(json.dumps({'best_sft_step': 980}))
    assert ChatEngine(model, tmp_path, 'cpu').status()['sft_step'] == 980


def test_shared_initial_context_is_removed_even_with_different_later_turns():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'datasets'))
    from prepare_formal_sft_v2 import isolated_validation
    train = [{'split_group': 'train', 'conversations': [{'role': 'user', 'content': '我喜欢蓝色。'}]}]
    reused = {'split_group': 'val-a', 'conversations': [{'role': 'user', 'content': '我喜欢蓝色。'},
               {'role': 'user', 'content': '更正为绿色。'}]}
    distinct = {'split_group': 'val-b', 'conversations': [{'role': 'user', 'content': '记录两盒铅笔。'}]}
    kept, removed = isolated_validation(train, [reused, distinct])
    assert kept == [distinct] and removed == [reused]
