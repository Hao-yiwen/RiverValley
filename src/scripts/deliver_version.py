"""交付绑定完整预训练、SFT 和问答证据的新版本，保留旧版并核对实例释放。"""
import argparse
import json
import logging
from pathlib import Path
import shutil
import sys
import tarfile
import time

from colab_train_pipeline import sha256_file
from deliver_training_weights import download
from drive_checkpoint_store import DriveStore
from launch_version import usage
from train_version import read_remote_json

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]


def selected_checkpoint(store, plan, receipt, stage_name, selection):
    """只允许选择本阶段已验证的每轮断点，不能用其他实验的权重替换交付对象。"""
    if not selection:
        return receipt['checkpoints'][stage_name + '_best']
    stage = next(s for s in plan['stages'] if s['id'] == stage_name)
    step = selection.get('step')
    if (selection.get('basis') != 'development' or not selection.get('reason')
            or not selection.get('development_review_sha256')
            or type(step) is not int or step not in stage.get('milestone_steps', [])):
        raise ValueError('候选选择必须依据开发评估，且来自冻结的每轮里程碑')
    parent = store.child(store.root_id, stage_name)
    folder = store.child(parent, f'step_{step:06d}') if parent else None
    if not folder or folder != selection.get('drive_folder_id'):
        raise ValueError('候选不属于本次正式 SFT 的指定里程碑')
    marker = store.child(folder, 'BACKUP_COMPLETE.json')
    if not marker:
        raise ValueError('候选没有完整服务器备份清单')
    manifest = read_remote_json(store, marker)
    if (manifest.get('step') != step or manifest.get('drive_folder_id') != folder
            or 'training_state.pt' not in manifest.get('files', {})
            or manifest['files'].get('model.safetensors', {}).get('sha256') != selection.get('model_sha256')):
        raise ValueError('候选权重、完整状态或里程碑游标不一致')
    store.verify(manifest)
    log.info('按开发评估选择正式 SFT 断点：step=%d，原因=%s', step, selection['reason'])
    return manifest


def validate_delivery(plan, receipt, pretrain_receipt, review, stage_name, candidate=None):
    """拒绝把小实验、仅训练完成或与评估权重不同的文件当成最终候选。"""
    if plan.get('purpose') != 'final_sft_candidate' or receipt.get('status') != 'plan_completed':
        raise ValueError('缺少正式 SFT 候选计划和完整完成回执')
    stages = {s['id']: s for s in plan['stages']}
    stage = stages[stage_name]
    if stage['kind'] != 'sft' or receipt['stages'][stage_name] != stage['target_steps']:
        raise ValueError('SFT 未达到冻结目标')
    full = receipt['checkpoints'][stage_name]
    best = candidate or receipt['checkpoints'][stage_name + '_best']
    if full['step'] != stage['target_steps'] or not 0 < best['step'] <= full['step']:
        raise ValueError('最终或最佳 SFT 游标不正确')
    if 'training_state.pt' not in full['files']:
        raise ValueError('缺少完整 SFT 恢复状态')
    if pretrain_receipt.get('status') != 'plan_completed':
        raise ValueError('继续预训练尚未完成')
    origins = {m['drive_folder_id']: m for m in pretrain_receipt['checkpoints'].values()}
    if stage.get('init_folder_id') not in origins:
        raise ValueError('SFT 初始化来源未绑定本轮预训练断点')
    expected_sha = best['files']['model.safetensors']['sha256']
    if not review.get('complete') or review.get('model_sha256') != expected_sha:
        raise ValueError('问答复核尚未完成或评估权重与待交付权重不一致')
    if not review.get('question_set_sha256') or not review.get('quality_status'):
        raise ValueError('问答报告缺少题集摘要或实际质量结论')
    return stage, best, origins[stage['init_folder_id']]


def load_completed(evidence, credential):
    bootstrap = json.loads((evidence / 'bootstrap.json').read_text())
    plan_path = evidence / 'plan.json'
    if sha256_file(plan_path) != bootstrap['plan_sha256']:
        raise ValueError('本机计划偏离冻结摘要')
    store = DriveStore(bootstrap['drive_root_id'], credential, bootstrap['quota_project'])
    receipt_id = store.child(store.root_id, 'completion_receipt.json')
    if not receipt_id:
        raise ValueError('服务器没有完成回执')
    receipt = read_remote_json(store, receipt_id)
    if receipt.get('plan_sha256') != bootstrap['plan_sha256']:
        raise ValueError('服务器完成回执属于其他计划')
    plan = json.loads(plan_path.read_text())
    if receipt.get('status') != 'plan_completed':
        raise ValueError('服务器回执未标记计划完成')
    for stage in plan['stages']:
        full = receipt['checkpoints'][stage['id']]
        if (receipt['stages'][stage['id']] != stage['target_steps'] or full['step'] != stage['target_steps']
                or 'training_state.pt' not in full['files']):
            raise ValueError('阶段未达到目标或完整恢复状态缺失：' + stage['id'])
    for manifest in receipt['checkpoints'].values():
        store.verify(manifest)
    return store, plan, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sft-evidence', required=True)
    parser.add_argument('--pretrain-evidence', required=True)
    parser.add_argument('--stage', default='sft')
    parser.add_argument('--review', required=True)
    parser.add_argument('--answers', required=True)
    parser.add_argument('--selection', help='可选：在留出评估前冻结、依据开发集选择每轮断点的 JSON')
    parser.add_argument('--output', default='checkpoints/delivery_v2_20261019')
    args = parser.parse_args()
    sft_evidence, pretrain_evidence = Path(args.sft_evidence), Path(args.pretrain_evidence)
    credential = str(Path.home() / '.config/colab-cli/token.json')
    store, plan, receipt = load_completed(sft_evidence, credential)
    _, pretrain_plan, pretrain_receipt = load_completed(pretrain_evidence, credential)
    review = json.loads(Path(args.review).read_text())
    selection = json.loads(Path(args.selection).read_text()) if args.selection else None
    if selection and review.get('selection_sha256') != sha256_file(Path(args.selection)):
        raise ValueError('留出复核报告没有绑定先前冻结的候选选择')
    candidate = selected_checkpoint(store, plan, receipt, args.stage, selection)
    stage, best, origin = validate_delivery(plan, receipt, pretrain_receipt, review, args.stage, candidate)
    if review.get('answers_sha256') != sha256_file(Path(args.answers)):
        raise ValueError('复核报告与原始回答摘要不符')
    account = usage(str(Path(sys.executable).with_name('colab')))
    from colab_cli.common import state
    if account['count'] or account['rate'] or state.client.list_assignments():
        raise RuntimeError('仍有实际付费实例，必须先核实保存并释放，再作最终交付')
    code_bundle = sft_evidence / 'version_code.tar'
    bootstrap = json.loads((sft_evidence / 'bootstrap.json').read_text())
    if sha256_file(code_bundle) != bootstrap['code_sha256']:
        raise ValueError('冻结源码压缩包摘要不符')
    output = Path(args.output).resolve()
    if (output / 'delivery_receipt.json').exists():
        previous = json.loads((output / 'delivery_receipt.json').read_text())
        if previous.get('sha256') != best['files']['model.safetensors']['sha256']:
            raise ValueError('目标已有不同的已交付权重，请使用新的版本目录')
    model = output / 'sft_best'
    model.mkdir(parents=True, exist_ok=True)
    for name, expected in best['files'].items():
        if Path(name).name != name:
            raise ValueError('服务器清单包含非法文件名')
        if name == 'training_state.pt':
            # 推理交付不下载数 GB 的 Adam 状态，云端完整副本仍由完成回执绑定。
            continue
        download(store, expected['id'], model / name, expected)
    code = output / 'code'
    code.mkdir(exist_ok=True)
    with tarfile.open(code_bundle) as archive:
        archive.extractall(code, filter='data')
    shutil.copy2(ROOT / 'requirements-chat.txt', output / 'requirements-chat.txt')
    shutil.copy2(args.review, output / 'acceptance_report.json')
    shutil.copy2(args.answers, output / 'acceptance_answers.jsonl')
    if args.selection:
        shutil.copy2(args.selection, output / 'candidate_selection.json')
    (output / 'selected_checkpoint.json').write_text(json.dumps(best, ensure_ascii=False, indent=2))
    (output / 'completion_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    (output / 'pretrain_completion_receipt.json').write_text(json.dumps(pretrain_receipt, ensure_ascii=False, indent=2))
    previous_tokens = pretrain_plan['token_accounting']['previous']
    config = next(s['config'] for s in pretrain_plan['stages'] if s['kind'] == 'pretrain')
    selected_tokens = previous_tokens + origin['step'] * config['batch_size'] * config['accumulation_steps'] * config['max_seq_len']
    delivery = {'status': 'weights_downloaded_verified', 'version': 'v2_20261019', 'label': '第二版 · 继续预训练 + SFT',
                'time': time.time(), 'directory': str(output), 'model': str(model / 'model.safetensors'),
                'sha256': best['files']['model.safetensors']['sha256'], 'sft_step': stage['target_steps'],
                'best_sft_step': best['step'], 'pretrain_selected_step': origin['step'], 'pretrain_tokens': selected_tokens,
                'pretrain_total_completed_tokens': pretrain_plan['token_accounting']['cumulative'],
                'qa_evaluated': True, 'quality_status': review['quality_status'], 'gpu_released': True,
                'release_evidence': account, 'used_this_authorization_cu': round(258.45 - account['balance'], 2),
                'drive_folder_id': store.root_id, 'best_folder_id': best['drive_folder_id']}
    delivery['selection_basis'] = 'development_milestone' if selection else 'validation_loss_best'
    (output / 'delivery_receipt.json').write_text(json.dumps(delivery, ensure_ascii=False, indent=2) + '\n')
    (output / 'README_本地测试.md').write_text(f'''# RiverValley 第二版本地测试

真实模型来源：继续预训练第 {origin['step']} 步，SFT 最佳第 {best['step']} 步。
本候选对应预训练输入 {selected_tokens:,} token。实际质量：{review['quality_status']}
原始回答与复核分别在 acceptance_answers.jsonl、acceptance_report.json；训练完成不代表所有问题都能答对。

运行项目根目录 start_chat.command，在右上角“模型与设置”选择第二版。旧版保留用于比较。
独立运行：安装本目录 requirements-chat.txt 后，使用 code/river-valley/src/scripts/inference.py，传入 --model 指向本目录 sft_best。

权重 SHA256：{delivery['sha256']}
最终权重、最佳权重及完整恢复状态： https://drive.google.com/drive/folders/{store.root_id}
已核实 Colab 活动实例和计费均为零。
''', encoding='utf-8')
    log.info('第二版真实权重交付完成：%s，最佳SFT=%d，质量=%s', model, best['step'], review['quality_status'])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
