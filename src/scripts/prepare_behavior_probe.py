"""冻结 500 步行为修复实验，从已交付 SFT 初始化，阶段内仍使用完整状态恢复。"""
import argparse
import json
import logging
from pathlib import Path
import tarfile

from prepare_version_pilot import ROOT, sha
from deliver_version import load_completed

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--drive-root-id', required=True)
    parser.add_argument('--quota-project', required=True, help='使用者自己的 Google Cloud 计费项目')
    args = parser.parse_args()
    data, output = Path(args.data), Path(args.output)
    if output.exists():
        raise FileExistsError('拒绝覆盖已冻结修复计划')
    manifest = json.loads((data/'manifest.json').read_text())
    for name, expected in manifest['hashes'].items():
        if sha(data/name) != expected:
            raise ValueError('修复材料摘要不一致：'+name)
    if manifest['splits']['train']['rows'] != 32000 or manifest['generator_sha256'] != sha(ROOT/'datasets/prepare_behavior_repair.py'):
        raise ValueError('材料规模或生成规则偏离已检查实验')
    old_data = ROOT/'checkpoints/bilingual_v2_20261019/formal_candidate_05'
    old_manifest = json.loads((old_data/'manifest.json').read_text())
    if sha(old_data/'sft_val.jsonl') != old_manifest['hashes']['sft_val.jsonl']:
        raise ValueError('原验证回放来源摘要不匹配')
    credential = str(Path.home()/'.config/colab-cli/token.json')
    _, _, receipt = load_completed(ROOT/'evals/results/v2_20261019/formal_sft_01', credential)
    origin = receipt['checkpoints']['sft_best']
    delivery = ROOT/'checkpoints/delivery_v2_20261019'
    local_receipt = json.loads((delivery/'delivery_receipt.json').read_text())
    if origin['step'] != 864 or origin['files']['model.safetensors']['sha256'] != local_receipt['sha256']:
        raise ValueError('修复起点不对应已诊断的第二版权重')
    remote = '/content/river-valley-run/'
    config = json.loads((ROOT/'configs/colab_a100_1b_sft.json').read_text())
    for key in ('checkpoint_dir', 'pretrain_checkpoint', 'max_steps'):
        config.pop(key, None)
    config.update(data_path=remote+'data/sft_train.jsonl', val_data_path=remote+'data/sft_val.jsonl',
        bpe_model_path=remote+'tokenizer.json', random_seed=20261026, num_workers=2,
        learning_rate=1e-5, min_lr=1e-6, epochs=1, schedule_steps=500, eval_interval=125,
        eval_iters=103, save_interval=250, batch_size=4, accumulation_steps=16)
    stage = {'id': 'behavior_sft', 'kind': 'sft', 'target_steps': 500, 'backup_interval': 250,
        'milestone_steps': [250, 500], 'evaluate_milestones': True, 'config': config,
        'init_folder_id': origin['drive_folder_id'], 'eval_questions': remote+'development.jsonl',
        'initial_eval_questions': remote+'development.jsonl'}
    plan = {'version': 'behavior_repair_20261026', 'purpose': 'behavior_repair_probe', 'atomic_resume_pairs': True,
        'description': '第二版之后的位置、格式和多轮行为修复小实验', 'monitor_id': 'rv1b-behavior-repair-20261026',
        'stages': [stage], 'initialization': {'source_model_sha256': local_receipt['sha256'],
          'source_step': 864, 'new_stage_optimizer': '显式新建AdamW；不是原864步中断恢复',
          'in_stage_resume': '完整optimizer/RNG/数据游标/最佳模型成对恢复，严格校验签名'},
        'budget_estimate': {'setup_training_evaluation_cu': 10, 'save_release_reserve_cu': 4,
          'basis': '原864步正式SFT全实例9.03CU；本轮500步且样本最长556token，增加起点评估与两次里程碑；保守预估10CU并另留4CU。此估算不是保证。'},
        'selection_rule': {'new_dev_questions': 192, 'old_dev_questions': 100,
          'minimum_new_dev_accuracy_gain': 0.15, 'maximum_old_dev_point_drop': 5,
          'rule': '先按开发集选择250或500；新任务准确率至少提高15个百分点且旧100题总分下降不超过5/200才作为改进候选。先人工复核旧开放题再锁定；384题新留出只在锁定后评估。未达门槛保留旧版，不把小实验冒充成功。'},
        'holdout_sha256': manifest['hashes']['holdout.jsonl'],
        'diagnosis': '训练抽取固定首项、JSON键固定导致捷径；小样本诊断不证明唯一根因；模型基础能力仍弱。'}
    files = {'data/sft_train.jsonl': data/'sft_train.jsonl', 'data/sft_val.jsonl': data/'sft_val.jsonl',
             'development.jsonl': data/'development.jsonl', 'tokenizer.json': delivery/'sft_best/tokenizer.json',
             'repair_manifest.json': data/'manifest.json'}
    if sha(files['tokenizer.json']) != origin['files']['tokenizer.json']['sha256']:
        raise ValueError('初始化与训练tokenizer不同')
    plan['input_sha256'] = {remote+name: sha(path) for name, path in files.items()}
    output.mkdir(parents=True)
    (output/'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (output/'origin_verified.json').write_text(json.dumps(origin, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    with tarfile.open(output/'version_materials.tar', 'w') as archive:
        for name, path in files.items():
            archive.add(path, arcname=name)
    sources = []
    with tarfile.open(output/'version_code.tar', 'w') as archive:
        for folder in ('src', 'datasets', 'tests'):
            for path in sorted((ROOT/folder).rglob('*.py')):
                if '__pycache__' not in path.parts:
                    relative = str(path.relative_to(ROOT))
                    archive.add(path, arcname='river-valley/'+relative)
                    sources.append({'path': relative, 'sha256': sha(path)})
        config_path = ROOT/'configs/colab_a100_1b_sft.json'
        archive.add(config_path, arcname='river-valley/configs/colab_a100_1b_sft.json')
        sources.append({'path': 'configs/colab_a100_1b_sft.json', 'sha256': sha(config_path)})
        archive.add(ROOT/'evals/bilingual_acceptance_v1.jsonl', arcname='river-valley/evals/bilingual_acceptance_v1.jsonl')
    bootstrap = {'code_sha256': sha(output/'version_code.tar'), 'materials_sha256': sha(output/'version_materials.tar'),
                 'drive_root_id': args.drive_root_id, 'quota_project': args.quota_project, 'plan_sha256': sha(output/'plan.json')}
    (output/'bootstrap.json').write_text(json.dumps(bootstrap, indent=2)+'\n', encoding='utf-8')
    (output/'code_manifest.json').write_text(json.dumps(sources, indent=2)+'\n', encoding='utf-8')
    log.info('已冻结修复实验：500步、32000行、学习率1e-5、新优化器；原权重与tokenizer校验通过。未分配GPU。')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
