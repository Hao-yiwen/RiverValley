"""正式候选不能重新混入已发现的错误监督，且简单有效样本仍保留。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'datasets'))
from refine_formal_sft_v2 import rejection


def sample(turns):
    return {'id': 'new', 'source': 'independent/checked', 'conversations': turns}


def test_multiturn_length_instruction_checks_every_assistant_turn():
    row = sample([{'role': 'system', 'content': 'Always respond with a response shorter than 10 characters'},
                  {'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hello!'},
                  {'role': 'user', 'content': 'More'}, {'role': 'assistant', 'content': 'This answer is too long.'}])
    assert rejection(row) == 'explicit_length_constraint_violation'


def test_simple_valid_dialogue_stays_available():
    row = sample([{'role': 'user', 'content': 'The box contains 12 pens.'},
                  {'role': 'assistant', 'content': 'Recorded: 12 pens.'},
                  {'role': 'user', 'content': 'Correction: 7 pens. How many now?'},
                  {'role': 'assistant', 'content': '7'}])
    assert rejection(row) is None


def test_wrong_router_sample_is_quarantined_before_formal_training():
    row = sample([{'role': 'user', 'content': 'IP数据报首部校验失败，路由器会怎么做？'},
                  {'role': 'assistant', 'content': '一定发ICMP目标不可达消息。'}])
    assert rejection(row) == 'unverified_network_specialist'


def test_long_form_cap_uses_supervised_tokens_not_sample_count():
    from balance_formal_sft_v2 import cap_long_form_share
    core = [{'id': 'c'+str(i), 'task': 'general', 'answer_tokens': 100} for i in range(2)]
    long_form = [{'id':'r'+str(i), 'task':'rewrite', 'answer_tokens':n} for i,n in enumerate([10,20,30,100])]
    kept, excluded = cap_long_form_share(core+long_form, .15)
    assert all(row in kept for row in core)
    selected = sum(r['answer_tokens'] for r in kept if r['task']=='rewrite')
    assert selected/(selected+200) <= .15
    assert len(kept)+len(excluded)==6
    assert cap_long_form_share(core+long_form, .15)==(kept,excluded)


def test_rewrite_is_not_mislabeled_as_general_knowledge():
    from balance_formal_sft_v2 import label_task
    original={'source':'HuggingFaceTB/smol-smoltalk/smollm-rewrite-30k','task':'general'}
    assert label_task(original)['task']=='rewrite'
    assert original['task']=='general'


def test_foreign_translation_rejected_but_mention_in_english_text_is_allowed():
    foreign = sample([{'role':'user','content':'Translate the Arabic sentence into Italian.\nنعم'},
                      {'role':'assistant','content':'Sì.'}])
    bilingual = sample([{'role':'user','content':'Translate this into Chinese:\nI study French a lot.'},
                        {'role':'assistant','content':'我经常学习法语。'}])
    assert rejection(foreign) == 'translation_outside_zh_en'
    assert rejection(bilingual) is None


def test_current_weather_requires_grounding_or_limitations():
    prompt = {'role':'user','content':'Did it snow in New York yesterday?'}
    assert rejection(sample([prompt, {'role':'assistant','content':'Yes, it received 2 inches.'}])) == 'ungrounded_current_weather'
    assert rejection(sample([prompt, {'role':'assistant','content':"I don't have access to current weather observations."}])) is None


def test_weather_rule_preserves_translation_and_does_not_match_training():
    translated = sample([{'role':'user','content':'请将下面这句话翻译成英文，只输出译文：\n今天天气不错！'},
                         {'role':'assistant','content':'The weather is fine today!'}])
    translated['task'] = 'translation'
    training = sample([{'role':'user','content':'What is the current training stage?'},
                       {'role':'assistant','content':'The stage is SFT.'}])
    assert rejection(translated) is None
    assert rejection(training) is None
