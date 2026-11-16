"""确保延期验收不加载模型、只接受完整断点，释放保险不会干扰训练。"""
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts'))
from colab_train_pipeline import publish_snapshot, write_json
from colab_overnight import defer_pipeline_evaluation, effective_status, release_due


def prepared_run(tmp_path):
    root, persistent, source = tmp_path / 'run', tmp_path / 'drive', tmp_path / 'source'
    source.mkdir()
    for name in ('model.safetensors', 'config.json', 'tokenizer.json', 'training_state.pt'):
        (source / name).write_bytes(b'test checkpoint')
    for name, step in (('pretrain', 7629), ('pretrain_best', 7400), ('sft', 324), ('sft_best', 320)):
        publish_snapshot(source, persistent / name, step)
    output = root / 'bilingual_outputs.jsonl'
    write_json(root / 'completion_policy.json', {
        'defer_qa_evaluation': True, 'evaluation_output': str(output),
        'pipeline_pid': os.getppid(), 'persistent_root': str(persistent), 'expected_sft_updates': 324})
    write_json(root / 'status.json', {'status': 'running', 'stage': 'bilingual_eval',
                                    'completed_pretrain_updates': 7629, 'completed_sft_updates': 324})
    return root, persistent, output


def test_deferral_exits_before_questions_or_inference(tmp_path, monkeypatch):
    import eval_bilingual
    root, persistent, output = prepared_run(tmp_path)
    monkeypatch.setitem(sys.modules, 'inference', None)
    monkeypatch.setattr(sys, 'argv', ['eval_bilingual.py', '--model', 'unused', '--output', str(output),
                                    '--questions', str(tmp_path / 'missing_questions.jsonl')])
    eval_bilingual.main()
    assert not output.exists()
    receipt = json.loads((persistent / 'completion_receipt.json').read_text())
    assert receipt['qa_answers_generated'] is False
    assert receipt['checkpoints']['sft']['step'] == 324
    corrected = effective_status({'status': 'completed_needs_manual_review', 'note': '旧固定文案'}, receipt)
    assert corrected['status'] == 'completed_saved_for_manual_test'
    assert corrected['evaluation_deferred'] is True


def test_incomplete_backup_cannot_publish_completion(tmp_path):
    root, persistent, output = prepared_run(tmp_path)
    (persistent / 'sft/step_000324/model.safetensors').write_bytes(b'broken')
    with pytest.raises(ValueError, match='清单不符'):
        defer_pipeline_evaluation(output)
    assert not (root / 'completion_receipt.json').exists()


def test_manual_eval_and_incomplete_training_are_not_completed(tmp_path):
    root, persistent, output = prepared_run(tmp_path)
    assert not defer_pipeline_evaluation(root / 'tomorrow_outputs.jsonl')
    policy = json.loads((root / 'completion_policy.json').read_text())
    write_json(root / 'completion_policy.json', dict(policy, pipeline_pid=-1))
    assert not defer_pipeline_evaluation(output)
    write_json(root / 'completion_policy.json', policy)
    write_json(root / 'status.json', {'stage': 'sft', 'completed_pretrain_updates': 7629})
    with pytest.raises(ValueError, match='尚未完成'):
        defer_pipeline_evaluation(output)


def test_guard_only_releases_after_terminal_grace_period():
    assert release_due({'status': 'running'}, None, 1000) == (None, False)
    assert release_due({'status': 'backing_up'}, 0, 1000) == (None, False)
    assert release_due({'status': 'completed_needs_manual_review'}, 1000, 2000) == (None, False)
    assert release_due({'status': 'failed'}, 1000, 2000) == (None, False)
    ready = {'status': 'completed_saved_for_manual_test', 'release_ready': True}
    assert release_due(ready, None, 1000) == (1000, False)
    assert release_due(ready, 1000, 1179) == (1000, False)
    assert release_due(ready, 1000, 1180) == (1000, True)
