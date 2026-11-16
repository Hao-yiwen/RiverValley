"""核对修复实验选优、留出及释放证据后，发布可供本地页面加载的独立交付。"""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import shutil
import sys
import tarfile
import time

from deliver_version import load_completed, selected_checkpoint
from drive_checkpoint_store import fingerprints
from launch_version import usage
from prepare_version_pilot import ROOT, sha
from eval_bilingual import validate_evaluation_rows

log = logging.getLogger(__name__)


def validate_repair_delivery(plan, selection, review, questions_sha, answers_sha):
    """开发门槛必须满足，且留出评估必须绑定此前锁定的同一权重。"""
    if plan.get('purpose') != 'behavior_repair_probe' or plan['stages'][0]['target_steps'] != 500:
        raise ValueError('不是已冻结的500步行为修复计划')
    if not selection.get('eligible') or selection.get('basis') != 'development':
        raise ValueError('开发评估未达到修复候选门槛')
    rule = plan['selection_rule']
    if (selection['new_dev_accuracy_gain'] < rule['minimum_new_dev_accuracy_gain'] or
            selection['old_dev_points'] < selection['baseline_old_dev_points'] - rule['maximum_old_dev_point_drop']):
        raise ValueError('候选的实际开发指标不满足冻结门槛')
    if (not review.get('complete') or review.get('questions') != 384 or
            review.get('model_sha256') != selection['model_sha256'] or
            review.get('question_set_sha256') != questions_sha or review.get('answers_sha256') != answers_sha):
        raise ValueError('新留出不完整，或模型、题集、原始回答摘要不一致')
    if not review.get('quality_status'):
        raise ValueError('留出报告缺少真实质量及局限结论')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--selection', required=True)
    parser.add_argument('--review', required=True)
    parser.add_argument('--answers', required=True)
    parser.add_argument('--staging', required=True)
    parser.add_argument('--output', default='checkpoints/delivery_behavior_20261026')
    args = parser.parse_args()
    evidence, output, staging = map(Path, (args.evidence, args.output, args.staging))
    selection_path, review_path, answers_path = map(Path, (args.selection, args.review, args.answers))
    selection, review = [json.loads(p.read_text(encoding='utf-8')) for p in (selection_path, review_path)]
    development_path=evidence/f"behavior_sft_step_{selection['step']}_review_summary.json"
    baseline_path=evidence/'initial_review_report.json'
    if (sha(development_path)!=selection['development_review_sha256'] or
            sha(baseline_path)!=selection['baseline_review_sha256']):
        raise ValueError('选优所依据的开发复核报告已变化')
    development=json.loads(development_path.read_text(encoding='utf-8'))
    baseline=json.loads(baseline_path.read_text(encoding='utf-8'))
    if (development['groups']['new']['fully_correct']!=selection['new_dev_correct'] or
            development['groups']['old']['points']!=selection['old_dev_points'] or
            baseline['groups']['old']['points']!=selection['baseline_old_dev_points'] or
            abs((development['groups']['new']['fully_correct']-baseline['groups']['new']['fully_correct'])/192-selection['new_dev_accuracy_gain'])>1e-12):
        raise ValueError('选优指标与实际开发报告不一致')
    credential = str(Path.home()/'.config/colab-cli/token.json')
    store, plan, receipt = load_completed(evidence, credential)
    questions = ROOT/'checkpoints/behavior_repair_20261026/data/holdout.jsonl'
    validate_repair_delivery(plan, selection, review, sha(questions), sha(answers_path))
    if review.get('selection_sha256') != sha(selection_path):
        raise ValueError('留出报告没有绑定先前冻结的候选选择')
    rows = [json.loads(x) for x in answers_path.read_text(encoding='utf-8').splitlines()]
    validate_evaluation_rows(rows, [json.loads(x) for x in questions.read_text(encoding='utf-8').splitlines()], sha(questions))
    selected = selected_checkpoint(store, plan, receipt, 'behavior_sft', selection)
    previous = json.loads((ROOT/'checkpoints/delivery_v2_20261019/delivery_receipt.json').read_text(encoding='utf-8'))
    if plan['initialization']['source_model_sha256'] != previous['sha256']:
        raise ValueError('修复起点不对应旧第二版交付')
    from colab_cli.common import state
    account = usage(str(Path(sys.executable).with_name('colab')))
    if account['count'] or account['rate'] or state.client.list_assignments():
        raise RuntimeError('GPU仍未核实释放，不能发布交付回执')
    if output.exists():
        raise FileExistsError('拒绝覆盖已有本地交付目录')
    bootstrap = json.loads((evidence/'bootstrap.json').read_text(encoding='utf-8'))
    if sha(evidence/'version_code.tar') != bootstrap['code_sha256']:
        raise ValueError('冻结源码包摘要不一致')
    # 先检查暂存区全部资产；阶段完整状态已在Drive核验，本地聊天无需下载8GB优化器。
    assets = {name: value for name, value in selected['files'].items() if name != 'training_state.pt'}
    for name, expected in assets.items():
        actual = fingerprints(staging/name)
        if actual['sha256'] != expected['sha256'] or actual['bytes'] != expected['bytes']:
            raise ValueError('暂存权重或配置摘要不一致：'+name)
    output = output.resolve()
    model = output/'sft_best'
    model.mkdir(parents=True)
    for name in assets:
        shutil.copy2(staging/name, model/name)
    with tarfile.open(evidence/'version_code.tar') as archive:
        archive.extractall(output/'code', filter='data')
    for item in json.loads((evidence/'code_manifest.json').read_text(encoding='utf-8')):
        if sha(output/'code/river-valley'/item['path']) != item['sha256']:
            raise ValueError('交付源码与冻结清单不一致：'+item['path'])
    shutil.copy2(ROOT/'requirements-chat.txt', output/'requirements-chat.txt')
    for source, name in [(selection_path,'candidate_selection.json'), (review_path,'acceptance_report.json'),
                          (answers_path,'acceptance_answers.jsonl'), (questions,'acceptance_questions.jsonl')]:
        shutil.copy2(source, output/name)
    for name, value in [('selected_checkpoint.json',selected), ('completion_receipt.json',receipt),
                        ('plan.json',plan), ('previous_delivery_receipt.json',previous)]:
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    development = output/'development'
    development.mkdir()
    for pattern in ('initial_review*', 'manual_review_*.json', 'behavior_sft*answers.jsonl',
                    'behavior_sft*reviewed.jsonl', 'behavior_sft*review_summary.json', 'behavior_sft*evaluation_binding.json'):
        for source in evidence.glob(pattern):
            if source.is_file():
                shutil.copy2(source, development/source.name)
    for name in ('checkpoints_behavior_sft_metrics.jsonl','launch_gate.json','bundle_test_receipt.json'):
        shutil.copy2(evidence/name, output/name)
    (output/'README_本地测试.md').write_text(f'''# RiverValley 行为修复实验版本地测试

本次额外SFT已经完成500步，依据开发评估选择第{selected['step']}步权重。
起点是旧第二版SFT864；模型仍是自研1.01B，预训练累计{previous['pretrain_tokens']:,} token。
实际质量：{review['quality_status']}

页面：http://127.0.0.1:8765/ ，在“模型与设置”中选择“行为修复实验版”。
原有版本保留用于对照。若页面服务未运行，从项目根目录运行start_chat.command。

独立运行需安装requirements-chat.txt，入口为本目录code/river-valley/src/scripts/inference.py，
使用 --model "{model}" 指定权重。中文说明和模型源代码均随包保存。
权重约4.05GB，额外需要推理内存；本机已采用MPS FP32运行测试。

SHA256：{selection['model_sha256']}
文件清单：SHA256SUMS。原始回答、评分、选优依据和完整训练计划均包含在本目录。
新384题仅覆盖内部程序任务，不能证明通用常识、翻译和开放聊天可靠；用户人工验收尚未进行。
完整最终/最佳权重及optimizer/RNG/数据游标保留在Drive根 {store.root_id}。
本次真实费用及GPU释放证据见delivery_receipt.json。
''', encoding='utf-8')
    delivery = {'status':'weights_downloaded_verified','version':'behavior_repair_20261026',
        'label':'行为修复实验版','time':time.time(),'directory':str(output),'model':str(model/'model.safetensors'),
        'sha256':selection['model_sha256'],'sft_step':500,'best_sft_step':selected['step'],
        'source_sft_step':864,'pretrain_tokens':previous['pretrain_tokens'],'qa_evaluated':True,
        'quality_status':review['quality_status'],'gpu_released':True,'release_evidence':account,
        'used_this_authorization_cu':round(18.57-account['balance'],2),'drive_folder_id':store.root_id,
        'best_folder_id':selected['drive_folder_id'],'selection_basis':'development_milestone',
        'selection_sha256':sha(selection_path),'acceptance_report_sha256':sha(review_path)}
    payload=(json.dumps(delivery,ensure_ascii=False,indent=2)+'\n').encode()
    checks=[f'{sha(path)}  {path.relative_to(output)}' for path in sorted(output.rglob('*')) if path.is_file()]
    checks.append(f'{hashlib.sha256(payload).hexdigest()}  delivery_receipt.json')
    (output/'SHA256SUMS').write_text('\n'.join(checks)+'\n',encoding='utf-8')
    pending=output/'.delivery_receipt.pending';pending.write_bytes(payload);pending.replace(output/'delivery_receipt.json')
    log.info('行为修复实验版交付发布：目录=%s step=%d SHA256=%s；质量=%s', output,selected['step'],selection['model_sha256'],review['quality_status'])


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    main()
