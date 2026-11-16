"""版本计划与阶段迁移必须防止错误数据、错误断点和学习率重置。"""
import copy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'src/scripts')]
from train_version import stage_segments, verify_plan
from training_utils import validate_new_pretrain_stage, validate_resume


def states():
    old = dict(memmap_dir='old', random_seed=1337, batch_size=16, accumulation_steps=8,
               max_seq_len=1024, world_size=1, adam_beta1=.9, adam_beta2=.999, adam_eps=1e-8,
               weight_decay=.1, grad_clip=1, tokenizer_sha256='same', metadata_sha256='old', schedule_steps=7629)
    state = dict(next_iter=7629, has_optimizer_state=True, optimizer={'state': {1: 'moment'}},
                 rng_states=[{'cpu': 'rng'}], resume_signature=old)
    return state, dict(old, memmap_dir='new', metadata_sha256='new', schedule_steps=15258)


def test_new_stage_is_explicit_and_regular_resume_remains_strict():
    state, signature = states()
    before = copy.deepcopy(state)
    result = validate_new_pretrain_stage(state, signature, 7629)
    assert result['source_completed_steps'] == 7629 and state == before
    with pytest.raises(ValueError, match='不一致'):
        validate_resume(state, signature)


@pytest.mark.parametrize('change', ['cursor', 'optimizer', 'rng', 'same_data', 'tokenizer', 'batch', 'sft'])
def test_invalid_stage_transition_fails(change):
    state, signature = states()
    if change == 'cursor': state['next_iter'] = 7500
    if change == 'optimizer': state['optimizer'] = None
    if change == 'rng': state['rng_states'] = []
    if change == 'same_data': signature['metadata_sha256'] = 'old'
    if change == 'tokenizer': signature['tokenizer_sha256'] = 'different'
    if change == 'batch': signature['batch_size'] = 8
    if change == 'sft': state['resume_signature'].pop('memmap_dir')
    with pytest.raises(ValueError): validate_new_pretrain_stage(state, signature, 7629)


def test_segments_never_repeat_completed_updates():
    assert stage_segments(80, 160, 80) == [160]
    assert stage_segments(160, 160, 80) == []
    assert stage_segments(3814, 15258, 3814) == [7628, 11442, 15256, 15258]


def test_plan_rejects_hidden_resume_or_changed_schedule():
    plan = {'stages': [{'id': 'sft_b', 'kind': 'sft', 'target_steps': 160,
                       'init_folder_id': 'source', 'config': {'schedule_steps': 160}}]}
    assert verify_plan(plan) is plan
    plan['stages'][0]['config']['resume'] = True
    with pytest.raises(ValueError): verify_plan(plan)
    plan['stages'][0]['config'] = {'schedule_steps': 100}
    with pytest.raises(ValueError): verify_plan(plan)


def test_json_scoring_checks_boolean_type_duplicates_and_code_fences():
    from eval_bilingual import automatic_score
    expected = '{"ready": true, "code": "Q9"}'
    assert automatic_score('{"code":"Q9", "ready":true}', expected, 'json') == 2
    assert automatic_score('{"code":"Q9", "ready":1}', expected, 'json') == 0
    assert automatic_score('{"code":"Q9", "code":"Q9", "ready":true}', expected, 'json') == 0
    assert automatic_score('```json\n'+expected+'\n```', expected, 'json') == 0


def test_upload_freezes_log_that_changes_during_its_own_upload(tmp_path):
    from train_version import upload_evidence_snapshot
    source = tmp_path / 'pipeline.log'
    source.write_text('训练完成\n')

    class Store:
        root_id = 'root'

        def upload(self, path, parent, name):
            with source.open('a') as stream:
                stream.write('开始上传日志\n')
            assert path.read_text() == '训练完成\n'
            assert name == 'pipeline.log' and parent == 'root'
            return {'verified': True}

    assert upload_evidence_snapshot(Store(), source, tmp_path / 'snapshot') == {'verified': True}


def test_epoch_answers_are_reused_only_for_same_weights_and_questions(tmp_path):
    import json
    from train_version import evaluate_checkpoint
    from colab_train_pipeline import sha256_file
    model = tmp_path / 'model'
    model.mkdir()
    for name in ('model.safetensors', 'tokenizer.json', 'config.json'):
        (model / name).write_text('first')
    question = {'id': 'one', 'prompt': 'Write 7', 'reference': '7', 'language': 'en', 'scoring': 'exact'}
    questions = tmp_path / 'questions.jsonl'
    questions.write_text(json.dumps(question)+'\n')
    calls = []
    def execute(name, command):
        calls.append(name)
        output = Path(command[command.index('--output')+1])
        output.write_text(json.dumps(dict(question, answer='7', question_sha256=sha256_file(questions)))+'\n')
        output.with_suffix('.summary.json').write_text('{}')
    first = evaluate_checkpoint(tmp_path, 'sft_step_10', model, questions, execute)
    assert len(calls) == 1 and first['questions'] == 1
    assert evaluate_checkpoint(tmp_path, 'sft_step_10', model, questions, execute) == first
    (model / 'model.safetensors').write_text('second')
    second = evaluate_checkpoint(tmp_path, 'sft_step_10', model, questions, execute)
    assert len(calls) == 2 and first['model_sha256'] != second['model_sha256']
    assert list(tmp_path.glob('sft_step_10_answers.jsonl.prior_*'))


def test_partial_epoch_evaluation_never_creates_completion_binding(tmp_path):
    import json
    from train_version import evaluate_checkpoint
    model = tmp_path / 'model';model.mkdir()
    for name in ('model.safetensors', 'tokenizer.json', 'config.json'):(model/name).write_text('x')
    questions=tmp_path/'questions.jsonl';questions.write_text(json.dumps({'id':'one'})+'\n')
    def execute(name, command):
        output=Path(command[command.index('--output')+1]);output.write_text('');output.with_suffix('.summary.json').write_text('{}')
    with pytest.raises(ValueError,match='缺失'):
        evaluate_checkpoint(tmp_path,'epoch',model,questions,execute)
    assert not (tmp_path/'epoch_evaluation_binding.json').exists()


def test_formal_sft_requires_review_and_accounts_for_tail_samples():
    from prepare_version_sft import build_plan
    manifest={'hashes':{'sft_train.jsonl':'a','sft_val.jsonl':'b'},'train':{'rows':30619,'answer_tokens':100},'val':{'rows':813}}
    decision={'status':'approved_for_formal_sft','data_hashes':manifest['hashes'],'pilot_comparison_sha256':'proof',
              'selection_reason':'真实开发结果支持','learning_rate':2e-5}
    origin={'step':15258,'drive_folder_id':'new_pretrain','files':{'model.safetensors':{}}}
    plan=build_plan(manifest,decision,origin,'/content/run/')
    assert plan['purpose']=='final_sft_candidate'
    assert plan['stages'][0]['target_steps']==956
    assert plan['data_accounting']['dropped_tail_per_epoch']==27
    assert plan['stages'][0]['evaluate_milestones']
    decision['data_hashes']={}
    with pytest.raises(ValueError,match='摘要'):
        build_plan(manifest,decision,origin,'/content/run/')
