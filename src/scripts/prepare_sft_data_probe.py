"""冻结同起点、同160步计划的数据配方对照；不放宽正式SFT的预训练完成门禁。"""
import argparse
import copy
import hashlib
import json
import logging
from pathlib import Path
import shutil
import tarfile
import time

from prepare_milestone_probe import ROOT, sha

log = logging.getLogger(__name__)
STAGE = 'sft_data_7629_160'
REMOTE = '/content/river-valley-run/'


def build_plan(baseline, manifest, approval):
    """训练数学完全沿用160步对照，只换数据内容和独立证据标识。"""
    if baseline.get('purpose') != 'milestone_diagnostic' or len(baseline['stages']) != 1:
        raise ValueError('需要已完成的单阶段里程碑对照')
    stage, origin = baseline['stages'][0], baseline['origin']
    if (stage['kind'] != 'sft' or stage['target_steps'] != 160
            or stage['config']['schedule_steps'] != 160
            or origin['stage'] != 'continued_pretrain' or origin['step'] != 7629
            or stage['init_folder_id'] != origin['drive_folder_id']):
        raise ValueError('必须使用同一CPT7629来源和160步学习率计划')
    if (approval.get('status') != 'approved_for_formal_sft'
            or approval['data_hashes'] != manifest['hashes']):
        raise ValueError('数据缺少对应的内容复核决策')
    config = stage['config']
    effective_batch = config['batch_size'] * config['accumulation_steps']
    if effective_batch != 64 or manifest['train']['rows'] // effective_batch < 160:
        raise ValueError('必须保持有效批量64，且新数据可在一遍内覆盖160步')
    if manifest['tokenizer_sha256'] != baseline['input_sha256'][REMOTE + 'tokenizer.json']:
        raise ValueError('不能在数据对照中更换tokenizer')
    result = copy.deepcopy(baseline)
    result['stages'][0].update(id=STAGE, milestone_steps=[160], evaluate_milestones=True)
    result.update(
        version='v2_20261019_' + STAGE, purpose='sft_data_diagnostic',
        description='同一CPT7629初始化，保持160步学习率与训练数学，比较清洗后的候选05数据',
        monitor_id='rv1b-v2-sft-data-7629-160',
        comparison={
            'baseline': 'probe_7629', 'baseline_steps': 160,
            'unchanged': '完整CPT来源、冻结代码、tokenizer、全部训练配置、解码与100题开发集',
            'changed': '训练/验证材料及其清单；单独保留固定160步与验证最佳权重的回答',
            'limitations': '等更新预算，非等监督token；新旧验证集不同，loss不直接横比。固定160步开发回答为主要对照，最佳权重另报。'},
        data_accounting={
            'available_rows': manifest['train']['rows'],
            'available_answer_tokens_per_pass': manifest['train']['answer_tokens'],
            'sample_presentations': 160 * effective_batch,
            'validation_rows': manifest['val']['rows'],
            'validation_batches_used': config['eval_iters']},
        following_stage='按开发回答和实际任务学习证据作质量决策；原CPT仅从原完整状态接续，正式SFT仍需CPT完成门禁。')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--dose-evidence', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--candidate-decision', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    baseline, dose, candidate, output = map(Path, (
        args.baseline, args.dose_evidence, args.candidate, args.output))
    if output.exists():
        raise FileExistsError('禁止覆盖既有诊断材料')
    decision_path = dose / 'quality_decision.json'
    decision = json.loads(decision_path.read_text(encoding='utf-8'))
    verified = json.loads((dose / 'latest_server_verification.json').read_text(encoding='utf-8'))
    baseline_verified = json.loads((baseline / 'latest_server_verification.json').read_text(encoding='utf-8'))
    baseline_review = json.loads((baseline / 'probe_7629_review_summary.json').read_text(encoding='utf-8'))
    if (decision['decision'] != 'compare_clean_data_same_schedule' or not verified['all_done']
            or not baseline_verified['all_done'] or not baseline_review['all_reviewed']
            or baseline_review['raw_sha256'] != baseline_verified['binding_verified']['answers_sha256']
            or decision['baseline_plan_sha256'] != sha(baseline / 'plan.json')
            or decision['baseline_review_sha256'] != sha(baseline / 'probe_7629_review_summary.json')
            or decision['dose_plan_sha256'] != sha(dose / 'plan.json')
            or decision['dose_verification_sha256'] != sha(dose / 'latest_server_verification.json')):
        raise ValueError('既有实验未完成复核，或质量决策绑定不一致')
    for name, evidence in decision['dose_reports'].items():
        report_path = dose / (name + '_review_summary.json')
        report = json.loads(report_path.read_text(encoding='utf-8'))
        if (sha(report_path) != evidence['report_sha256'] or not report['all_reviewed']
                or report['raw_sha256'] != verified['bindings'][name]['answers_sha256']):
            raise ValueError('监督量诊断的逐题评分不完整或未绑定服务器回答')
    manifest = json.loads((candidate / 'manifest.json').read_text(encoding='utf-8'))
    approval = json.loads(Path(args.candidate_decision).read_text(encoding='utf-8'))
    if (sha(candidate / 'manifest.json') != decision['candidate_manifest_sha256']
            or sha(args.candidate_decision) != decision['candidate_decision_sha256']
            or approval['manifest_sha256'] != sha(candidate / 'manifest.json')):
        raise ValueError('候选材料清单或审核决策已改变')
    for filename, expected in manifest['hashes'].items():
        if sha(candidate / filename) != expected:
            raise ValueError('候选数据摘要不符：' + filename)
    boot = json.loads((baseline / 'bootstrap.json').read_text(encoding='utf-8'))
    for filename, key in [('version_code.tar', 'code_sha256'),
                          ('version_materials.tar', 'materials_sha256'), ('plan.json', 'plan_sha256')]:
        if sha(baseline / filename) != boot[key]:
            raise ValueError('原对照冻结文件摘要不符：' + filename)
    original = json.loads((baseline / 'plan.json').read_text(encoding='utf-8'))
    plan = build_plan(original, manifest, approval)
    with tarfile.open(baseline / 'version_materials.tar') as archive:
        for name, expected in original['input_sha256'].items():
            body = archive.extractfile(name.removeprefix(REMOTE)).read()
            if hashlib.sha256(body).hexdigest() != expected:
                raise ValueError('原对照内部材料摘要不符：' + name)
    files = {
        'data/sft_pilot.jsonl': candidate / 'sft_train.jsonl',
        'data/sft_val.jsonl': candidate / 'sft_val.jsonl',
        'sft_manifest.json': candidate / 'manifest.json',
        'sft_review_decision.json': Path(args.candidate_decision),
        'tokenizer.json': candidate.parent / 'tokenizer.json',
        'development.jsonl': ROOT / 'evals/bilingual_acceptance_v1.jsonl'}
    for name in ('tokenizer.json', 'development.jsonl'):
        if sha(files[name]) != original['input_sha256'][REMOTE + name]:
            raise ValueError('对照中不得变更：' + name)
    plan['input_sha256'] = {REMOTE + name: sha(path) for name, path in files.items()}
    plan['baseline_plan_sha256'] = sha(baseline / 'plan.json')
    plan['quality_decision_sha256'] = sha(decision_path)
    from drive_checkpoint_store import DriveStore
    from milestone_handoff import validate_origin
    roots = json.loads((ROOT / 'evals/results/v2_20261019/drive_roots.json').read_text(encoding='utf-8'))
    store = DriveStore(roots['version_root_id'], str(Path.home() / '.config/colab-cli/token.json'),
                       boot['quota_project'])
    marker = store.child(plan['origin']['drive_folder_id'], 'BACKUP_COMPLETE.json')
    if not marker:
        raise ValueError('原完整CPT来源缺失')
    origin = store.request('GET', 'https://www.googleapis.com/drive/v3/files/' + marker,
                           params={'alt': 'media'}).json()
    validate_origin(origin, plan)
    store.verify(origin)
    folder = store.folder(store.root_id, STAGE)
    output.mkdir(parents=True)
    for name in ('version_code.tar', 'code_manifest.json'):
        shutil.copy2(baseline / name, output / name)
    with tarfile.open(output / 'version_materials.tar', 'w') as archive:
        for name, path in files.items():
            archive.add(path, arcname=name)
    (output / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    boot.update(drive_root_id=folder, materials_sha256=sha(output / 'version_materials.tar'),
                plan_sha256=sha(output / 'plan.json'))
    (output / 'bootstrap.json').write_text(json.dumps(boot, indent=2) + '\n', encoding='utf-8')
    receipt = {'time': time.time(), 'status': 'prepared_origin_server_verified', 'origin': origin,
               'candidate_manifest_sha256': sha(candidate / 'manifest.json'),
               'decision_sha256': sha(decision_path), 'code_unchanged': True,
               'training_config_unchanged': plan['stages'][0]['config'] == original['stages'][0]['config'],
               'formal_sft_completion_gate_unchanged': True}
    (output / 'preparation.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    log.info('数据配方对照材料已冻结：阶段=%s 来源=%d 候选行数=%d 目标160步 输出=%s；未申请GPU',
             STAGE, origin['step'], manifest['train']['rows'], output)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    try:
        main()
    except Exception as error:
        log.error('数据对照准备失败，未申请GPU：异常类型=%s', type(error).__name__)
        raise
