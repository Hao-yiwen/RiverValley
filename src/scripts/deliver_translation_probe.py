"""验证同精度开发对照、锁定候选及一次新留出后发布翻译实验版本。"""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import shutil
import sys
import tarfile
import time

from deliver_version import load_completed
from drive_checkpoint_store import fingerprints
from eval_bilingual import validate_evaluation_rows
from launch_version import usage
from prepare_version_pilot import ROOT, sha

log = logging.getLogger(__name__)


def verify_frozen_development(evidence, plan):
    """合并开发题必须对应冻结的新48题和原100题，不能换成更容易的题集。"""
    data = ROOT/'checkpoints/curriculum_20261110/data'
    if sha(data/'manifest.json') != plan['input_sha256']['/content/river-valley-run/curriculum_manifest.json']:
        raise ValueError('开发数据清单偏离训练时冻结摘要')
    manifest = json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    if sha(data/'development.jsonl') != manifest['hashes']['development.jsonl']:
        raise ValueError('新开发题偏离冻结数据清单')
    bootstrap = json.loads((evidence/'bootstrap.json').read_text(encoding='utf-8'))
    if sha(evidence/'version_code.tar') != bootstrap['code_sha256']:
        raise ValueError('原100题所在冻结源码包摘要不一致')
    with tarfile.open(evidence/'version_code.tar') as archive:
        old = [dict(json.loads(s), regression=True) for s in archive.extractfile('river-valley/evals/bilingual_acceptance_v1.jsonl').read().decode('utf-8').splitlines()]
    new = [json.loads(s) for s in (data/'development.jsonl').read_text(encoding='utf-8').splitlines()]
    actual = [json.loads(s) for s in (evidence/'development.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(new) != 48 or len(old) != 100 or actual != new + old:
        raise ValueError('合并开发题必须严格为冻结48+100题及原顺序')


def verify_review(review_path, answers_path, questions_path, model_sha):
    """复核分数必须逐题绑定原始回答，不仅相信一个汇总数字。"""
    review = json.loads(review_path.read_text(encoding='utf-8'))
    rows = [json.loads(s) for s in answers_path.read_text(encoding='utf-8').splitlines()]
    questions = [json.loads(s) for s in questions_path.read_text(encoding='utf-8').splitlines()]
    validate_evaluation_rows(rows, questions, sha(questions_path))
    binding_path = answers_path.with_name(answers_path.name.replace('_answers.jsonl', '_binding.json'))
    binding = json.loads(binding_path.read_text(encoding='utf-8'))
    if (binding.get('status') != 'completed' or binding.get('model_sha256') != model_sha
            or binding.get('questions_sha256') != sha(questions_path)
            or binding.get('answers_sha256') != sha(answers_path)
            or binding.get('device') != 'mps' or binding.get('dtype') != 'float32'
            or binding.get('do_sample') is not False or binding.get('repetition_penalty') != 1.0
            or binding.get('max_tokens') != 160 or review.get('binding_sha256') != sha(binding_path)):
        raise ValueError('本机问答运行绑定不完整、模型不符或前后推理设置不一致')
    if (not review.get('complete') or review.get('model_sha256') != model_sha
            or review.get('answers_sha256') != sha(answers_path) or review.get('questions_sha256') != sha(questions_path)):
        raise ValueError('复核模型或原始回答摘要不一致，或复核尚未完成')
    scores = review.get('scores', {})
    if set(scores) != {r['id'] for r in rows} or any(type(v.get('score')) is not int or v['score'] not in (0, 1, 2) or not v.get('reason') for v in scores.values()):
        raise ValueError('人工评分缺题、重复、越界或缺少理由')
    totals = {}
    for row in rows:
        key = 'old' if row.get('regression') else 'translation'
        totals.setdefault(key, {'rows': 0, 'points': 0, 'correct': 0})
        totals[key]['rows'] += 1
        totals[key]['points'] += scores[row['id']]['score']
        totals[key]['correct'] += scores[row['id']]['score'] == 2
    if review.get('groups') != totals:
        raise ValueError('复核汇总与逐题实际分数不一致')
    return review


def eligible(plan, baseline, candidate):
    """失败保留旧默认版本，不能降低预先冻结门槛。"""
    rule = plan['selection_rule']
    for review in (baseline, candidate):
        if review['groups']['translation']['rows'] != 48 or review['groups']['old']['rows'] != 100:
            raise ValueError('前后开发对照必须包含完整48+100题')
    gain = candidate['groups']['translation']['points'] - baseline['groups']['translation']['points']
    drop = baseline['groups']['old']['points'] - candidate['groups']['old']['points']
    return gain >= rule['minimum_translation_point_gain'] and drop <= rule['maximum_old_dev_point_drop']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--staging', required=True)
    parser.add_argument('--output', default='checkpoints/delivery_curriculum_20261110')
    args = parser.parse_args()
    evidence, staging, output = map(Path, (args.evidence, args.staging, args.output))
    store, plan, receipt = load_completed(evidence, str(Path.home()/'.config/colab-cli/token.json'))
    if plan.get('purpose') != 'translation_repair_probe' or receipt['stages'] != {'translation_sft': 160}:
        raise ValueError('不是冻结的160步翻译实验')
    selected = receipt['checkpoints']['translation_sft']
    model_sha = selected['files']['model.safetensors']['sha256']
    verify_frozen_development(evidence, plan)
    baseline = verify_review(evidence/'baseline_review.json', evidence/'baseline_answers.jsonl', evidence/'development.jsonl', plan['initialization']['source_model_sha256'])
    candidate = verify_review(evidence/'candidate_review.json', evidence/'candidate_answers.jsonl', evidence/'development.jsonl', model_sha)
    if not eligible(plan, baseline, candidate):
        raise ValueError('未达到冻结开发门槛，保留旧默认版并保存失败实验，不发布改善版本')
    selection = json.loads((evidence/'candidate_selection.json').read_text(encoding='utf-8'))
    if (selection.get('model_sha256') != model_sha or selection.get('step') != selected['step']
            or selection.get('drive_folder_id') != selected['drive_folder_id']
            or not selection.get('eligible') or selection.get('basis') != 'development'
            or selection.get('development_review_sha256') != sha(evidence/'candidate_review.json')
            or selection.get('baseline_review_sha256') != sha(evidence/'baseline_review.json')):
        raise ValueError('候选锁定未绑定实际完整开发复核')
    questions = ROOT/'checkpoints/curriculum_20261110/data/holdout.jsonl'
    if sha(questions) != plan['holdout_sha256']:
        raise ValueError('新留出偏离冻结题集')
    holdout = verify_review(evidence/'holdout_review.json', evidence/'holdout_answers.jsonl', questions, model_sha)
    if (holdout.get('selection_sha256') != sha(evidence/'candidate_selection.json')
            or holdout['groups']['translation']['rows'] != 48 or not holdout.get('quality_status')):
        raise ValueError('新留出未绑定先前选择，或未完成48题及真实质量说明')
    from colab_cli.common import state
    account = usage(str(Path(sys.executable).with_name('colab')))
    if account['count'] or account['rate'] or state.client.list_assignments():
        raise RuntimeError('仍有真实计费实例，拒绝提前发布交付')
    assets = {n: v for n, v in selected['files'].items() if n != 'training_state.pt'}
    for name, expected in assets.items():
        if Path(name).name != name:
            raise ValueError('非法资产名称')
        actual = fingerprints(staging/name)
        if actual['bytes'] != expected['bytes'] or actual['sha256'] != expected['sha256']:
            raise ValueError('本机暂存资产校验失败：'+name)
    bootstrap = json.loads((evidence/'bootstrap.json').read_text())
    if sha(evidence/'version_code.tar') != bootstrap['code_sha256']:
        raise ValueError('冻结源码包摘要错误')
    if output.exists():
        raise FileExistsError('拒绝覆盖已有版本')
    output = output.resolve()
    model = output/'sft_best'
    model.mkdir(parents=True)
    for name in assets:
        shutil.copy2(staging/name, model/name)
    with tarfile.open(evidence/'version_code.tar') as archive:
        archive.extractall(output/'code', filter='data')
    for item in json.loads((evidence/'code_manifest.json').read_text()):
        if sha(output/'code/river-valley'/item['path']) != item['sha256']:
            raise ValueError('冻结源码单文件摘要错误')
    for name in ('baseline_answers.jsonl','baseline_review.json','candidate_answers.jsonl','candidate_review.json',
                 'development.jsonl','holdout_answers.jsonl','holdout_review.json','candidate_selection.json',
                 'baseline_binding.json','candidate_binding.json','holdout_binding.json',
                 'candidate_manual_scores.json','holdout_manual_scores.json','baseline_categories.json',
                 'plan.json','code_manifest.json','data_audit.json','origin_verified.json','run_limits.json',
                 'guard_recalibration.json','bundle_test_receipt.json','release_evidence.json',
                 'release_confirmed.json',
                 'checkpoints_translation_sft_metrics.jsonl'):
        shutil.copy2(evidence/name, output/name)
    shutil.copy2(questions, output/'holdout_questions.jsonl')
    shutil.copytree(ROOT/'checkpoints/curriculum_20261110/data', output/'training_data')
    tools_dir = output/'evaluation_tools'
    tools_dir.mkdir()
    for name in ('run_local_stage.py','make_review.py','after_training.py'):
        shutil.copy2(evidence.parent/name, tools_dir/name)
    shutil.copy2(Path(__file__), tools_dir/Path(__file__).name)
    shutil.copy2(ROOT/'requirements-chat.txt', output/'requirements-chat.txt')
    for name, value in [('selected_checkpoint.json',selected),('completion_receipt.json',receipt)]:
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (output/'README_本地测试.md').write_text(f'''# RiverValley 翻译修复实验

本版本从行为修复250步继续新建SFT阶段并完成160步，自研1.01B和tokenizer保留。
问答生成采用本机MPS FP32，真实结果：{holdout['quality_status']}
开发翻译分数 {baseline['groups']['translation']['points']}/96 → {candidate['groups']['translation']['points']}/96。
旧100题 {baseline['groups']['old']['points']}/200 → {candidate['groups']['old']['points']}/200。
新留出 {holdout['groups']['translation']['points']}/96，仅24组双语短句；不是通用聊天验收。

页面 http://127.0.0.1:8765/ ，模型菜单保留旧版。服务未运行时执行项目start_chat.command。
在本交付目录安装requirements-chat.txt，并使用随包的冻结源码加载；不要假设通用AutoModel自动注册了此自研模型。
复现本轮生成设置时，在交付目录执行单次提示词：
`python code/river-valley/src/scripts/inference.py --model "$PWD/sft_best" --prompt "请把我正在学习英语翻译成英文" --greedy --max_tokens 160 --repetition_penalty 1.0`
页面默认确定性生成、重复惩罚1.0，回答长度可在设置中调整；命令行交互默认参数与上述评估命令不同。
权重SHA256：{model_sha}。全部文件摘要见SHA256SUMS。
完整优化器/RNG/游标和最终/最佳权重保留Drive目录 {store.root_id}；原始回答及人工评分随包保存。
用户人工验收尚未进行；不以训练完成冒充通用质量通过。
''',encoding='utf-8')
    delivery={'status':'weights_downloaded_verified','version':plan['version'],'label':'翻译修复实验版',
        'time':time.time(),'directory':str(output),'model':str(model/'model.safetensors'),'sha256':model_sha,
        'sft_step':160,'best_sft_step':160,'source_sft_step':250,'pretrain_tokens':2999844864,
        'qa_evaluated':True,'quality_status':holdout['quality_status'],'gpu_released':True,'release_evidence':account,
        'used_this_authorization_cu':round(10.63-account['balance'],2),'drive_folder_id':store.root_id,
        'best_folder_id':selected['drive_folder_id'],'selection_basis':'development',
        'selection_sha256':sha(evidence/'candidate_selection.json'),'acceptance_report_sha256':sha(evidence/'holdout_review.json')}
    payload=(json.dumps(delivery,ensure_ascii=False,indent=2)+'\n').encode()
    checks=[f'{sha(p)}  {p.relative_to(output)}' for p in sorted(output.rglob('*')) if p.is_file()]
    checks.append(f'{hashlib.sha256(payload).hexdigest()}  delivery_receipt.json')
    (output/'SHA256SUMS').write_text('\n'.join(checks)+'\n',encoding='utf-8')
    pending=output/'.delivery_receipt.pending';pending.write_bytes(payload);pending.replace(output/'delivery_receipt.json')
    log.info('翻译实验交付已发布：路径=%s SHA256=%s 质量=%s', output,model_sha,holdout['quality_status'])


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    main()
