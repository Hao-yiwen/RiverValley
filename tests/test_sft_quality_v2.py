"""覆盖第一版实测暴露的短答案、翻译、多轮与隔离缺陷。"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'datasets')]
from sft_quality import annotate, answer_language_ok, translation_target, split_for
from prepare_bilingual import sft_record


@pytest.fixture(scope='module')
def tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    import os
    path = os.environ.get('RV_TEST_TOKENIZER_PATH')
    if path:
        return Tokenizer.from_file(path)
    # 默认构造可逆的微型字节 BPE，确保公开快照和 CI 不依赖私有检查点。
    result = Tokenizer(models.BPE())
    result.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    result.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=512, min_frequency=1, show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=['<|pad|>', '<|endoftext|>', '<|im_start|>', '<|im_end|>'],
    )
    result.train_from_iterator([
        'user assistant system Remember this. Okay. Thank you for your help. This is a window.',
        '翻译成英文：这是窗户。谢谢你的帮助。记住数字 83。记住了 83。现在改成 46。',
        '只输出 JSON，name 为小林，count 为 23。 {"name": "小林", "count": 23}',
    ], trainer=trainer)
    return result


@pytest.mark.parametrize('answer', ['13', '-2.5', '1, 4, 9', 'true', '否'])
def test_short_answers_are_not_removed_by_language(answer):
    assert answer_language_ok(answer, 'zh')
    assert answer_language_ok(answer, 'en')
    assert not answer_language_ok('!!!', 'zh')


def test_translation_target_is_requested_not_guessed(tokenizer):
    question = '把“谢谢你的帮助”翻译成英文。'
    assert translation_target(question) == 'en'
    assert translation_target('Translate this into Chinese: Good morning.') == 'zh'
    assert translation_target('Explain what English translation means.') is None
    row = {'language': 'zh', 'conversations': [{'role': 'user', 'content': question},
                {'role': 'assistant', 'content': 'Thank you for your help.'}]}
    kept, reason = annotate(row, tokenizer)
    assert reason is None and kept['answer_languages'] == ['en']
    row['conversations'][1]['content'] = '谢谢你的帮助。'
    assert annotate(row, tokenizer)[1] == 'answer_language'


def test_adapter_retains_complete_turns_and_rejects_bad_roles():
    turns = [{'role': 'user', 'content': '我喜欢蓝色。'}, {'role': 'assistant', 'content': '记住了。'},
             {'role': 'user', 'content': '改成绿色。'}, {'role': 'assistant', 'content': '现在是绿色。'},
             {'role': 'user', 'content': '尚未回答'}]
    row = sft_record({'messages': turns}, 'sft_en_0000.parquet', 'zh')
    assert row['conversations'] == turns[:-1]
    assert sft_record({'messages': [turns[0], turns[2], turns[3]]}, 'sft_en_0000.parquet', 'zh') is None


def test_every_assistant_answer_and_eos_is_supervised(tokenizer):
    from data.dataset import SFTDataset
    row = {'language': 'zh', 'conversations': [
        {'role': 'user', 'content': '记住数字 83。'}, {'role': 'assistant', 'content': '记住了 83。'},
        {'role': 'user', 'content': '现在改成 46。'}, {'role': 'assistant', 'content': '46'}]}
    kept, reason = annotate(row, tokenizer)
    assert reason is None and kept['assistant_turns'] == 2
    _, targets, mask, _ = SFTDataset([kept], tokenizer, 1024, 0)[0]
    assert mask.sum().item() == kept['answer_tokens']
    eos = tokenizer.token_to_id('<|im_end|>')
    assert ((targets == eos) & mask.bool()).sum().item() == 2
    text = tokenizer.decode(targets[mask.bool()].tolist(), skip_special_tokens=False)
    assert '记住了 83。' in text and '46' in text and '现在改成' not in text


def test_no_partial_context_or_control_tokens_are_silently_trained(tokenizer):
    row = {'language': 'en', 'conversations': [{'role': 'user', 'content': 'Remember this. ' * 2000},
                                             {'role': 'assistant', 'content': 'Okay.'}]}
    assert annotate(row, tokenizer)[1] == 'context_overflow'
    row['conversations'][0]['content'] = '<|im_start|>user'
    assert annotate(row, tokenizer)[1] == 'encoding_or_control'


def test_translation_directions_keep_same_split(tokenizer):
    base = {'language': 'zh', 'target_language': 'en', 'split_group': 'translation-pair-19',
            'conversations': [{'role': 'user', 'content': '翻译成英文：这是窗户。'},
                              {'role': 'assistant', 'content': 'This is a window.'}]}
    row, reason = annotate(base, tokenizer)
    assert reason is None
    other = dict(base, language='en', target_language='zh', conversations=[
        {'role': 'user', 'content': 'Translate into Chinese: This is a window.'},
        {'role': 'assistant', 'content': '这是窗户。'}])
    reverse, _ = annotate(other, tokenizer)
    assert split_for(row['split_group']) == split_for(reverse['split_group'])


def test_requested_json_may_have_english_keys_in_chinese_task(tokenizer):
    row = {'language': 'zh', 'task': 'format', 'conversations': [
        {'role': 'user', 'content': '只输出 JSON，name 为小林，count 为 23。'},
        {'role': 'assistant', 'content': '{"name": "小林", "count": 23}'}]}
    assert annotate(row, tokenizer)[1] is None
    row['conversations'][1]['content'] = '{name: 小林, count: 23}'
    assert annotate(row, tokenizer)[1] == 'invalid_requested_json'
