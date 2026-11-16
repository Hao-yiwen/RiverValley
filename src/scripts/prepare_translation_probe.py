"""冻结160步翻译修复；从已交付行为修复权重新建明确的SFT阶段。"""
import argparse
import json
import logging
from pathlib import Path
import tarfile

from deliver_version import load_completed, selected_checkpoint
from prepare_version_pilot import ROOT, sha

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--quota-project', required=True, help='使用者自己的 Google Cloud 计费项目')
    parser.add_argument('--drive-parent-id', required=True, help='存放本次实验的 Drive 父目录')
    args = parser.parse_args()
    output = Path(args.output)
    if (output/'plan.json').exists():
        raise FileExistsError('拒绝覆盖冻结计划')
    data = ROOT/'checkpoints/curriculum_20261110/data'
    manifest = json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    for name, digest in manifest['hashes'].items():
        if sha(data/name) != digest:
            raise ValueError('材料摘要不匹配：'+name)
    if manifest['train_rows'] != 10240 or manifest['generator_sha256'] != sha(ROOT/'datasets/prepare_translation_curriculum.py'):
        raise ValueError('训练规模或材料生成器偏离审查版本')
    credential = str(Path.home()/'.config/colab-cli/token.json')
    old = ROOT/'evals/results/quality_repair_20261026/probe_02'
    store, old_plan, old_receipt = load_completed(old, credential)
    selection = json.loads((old/'candidate_selection.json').read_text(encoding='utf-8'))
    origin = selected_checkpoint(store, old_plan, old_receipt, 'behavior_sft', selection)
    delivery = ROOT/'checkpoints/delivery_behavior_20261026'
    receipt = json.loads((delivery/'delivery_receipt.json').read_text(encoding='utf-8'))
    if origin['files']['model.safetensors']['sha256'] != receipt['sha256'] or sha(delivery/'sft_best/model.safetensors') != receipt['sha256']:
        raise ValueError('新阶段起点与已交付权重不一致')
    remote = '/content/river-valley-run/'
    config = json.loads((ROOT/'configs/colab_a100_1b_sft.json').read_text())
    for key in ('checkpoint_dir', 'pretrain_checkpoint', 'max_steps'):
        config.pop(key, None)
    config.update(data_path=remote+'data/sft_train.jsonl', val_data_path=remote+'data/sft_val.jsonl',
        bpe_model_path=remote+'tokenizer.json', random_seed=20261110, num_workers=2,
        learning_rate=1e-5, min_lr=1e-6, epochs=1, schedule_steps=160, eval_interval=80,
        eval_iters=32, save_interval=160, batch_size=4, accumulation_steps=16)
    stage = {'id': 'translation_sft', 'kind': 'sft', 'target_steps': 160, 'backup_interval': 160,
        'milestone_steps': [160], 'evaluate_milestones': False, 'config': config,
        'init_folder_id': origin['drive_folder_id']}
    # GPU只做训练、原有验证与持久备份；问答前后对照使用本机相同MPS FP32设置。
    plan = {'version': 'translation_curriculum_20261110', 'purpose': 'translation_repair_probe',
        'atomic_resume_pairs': True, 'description': '短句翻译指令多样化与旧任务回放实验',
        'monitor_id': 'rv1b-translation-repair-20261110', 'stages': [stage],
        'initialization': {'source_model_sha256': receipt['sha256'], 'source_step': 250,
            'new_stage_optimizer': '新SFT阶段显式新建AdamW及160步学习率计划，保留1024上下文；不重跑预训练。',
            'in_stage_resume': '严格完整optimizer/RNG/数据游标/最佳权重恢复与签名校验'},
        'budget_estimate': {'setup_training_evaluation_cu': 5.5, 'save_release_reserve_cu': 4,
            'basis': '旧500步、四组GPU问答与两次持久保存合计7.94CU；本轮160步、最长427token、一次最终持久保存，GPU不生成问答，保守估计5.5CU且另留4CU；估算非保证。'},
        'balance_policy': {'soft_balance_cu': 4, 'hard_balance_cu': 2},
        'selection_rule': {'new_dev_questions': 48, 'new_independent_groups': 12, 'old_dev_questions': 100,
            'minimum_translation_point_gain': 12, 'maximum_old_dev_point_drop': 5,
            'rule': '唯一160步候选与起点在相同MPS FP32/greedy下比较；48题人工语义总分至少提升12/96，旧100题下降不超过5/200。通过才锁定候选SHA，再一次运行48题新留出；失败保留旧默认版和实验权重，不继续盲目花费。'},
        'holdout_sha256': manifest['hashes']['holdout.jsonl'],
        'quality_limits': '内部12组开发与24组留出不是通用翻译基准，公开句对非逐条核验；不承诺通用聊天能力达标。'}
    files = {'data/sft_train.jsonl': data/'sft_train.jsonl', 'data/sft_val.jsonl': data/'sft_val.jsonl',
             'tokenizer.json': delivery/'sft_best/tokenizer.json', 'curriculum_manifest.json': data/'manifest.json'}
    if sha(files['tokenizer.json']) != origin['files']['tokenizer.json']['sha256']:
        raise ValueError('初始化tokenizer与本轮训练不一致')
    plan['input_sha256'] = {remote+name: sha(path) for name, path in files.items()}
    output.mkdir(parents=True, exist_ok=True)
    (output/'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (output/'origin_verified.json').write_text(json.dumps(origin, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    with tarfile.open(output/'version_materials.tar', 'w') as archive:
        for name, path in files.items():
            archive.add(path, arcname=name)
    sources = []
    with tarfile.open(output/'version_code.tar', 'w') as archive:
        paths = [p for folder in ('src', 'datasets', 'tests') for p in sorted((ROOT/folder).rglob('*.py')) if '__pycache__' not in p.parts]
        paths += [ROOT/'configs/colab_a100_1b_sft.json', ROOT/'evals/bilingual_acceptance_v1.jsonl']
        for path in paths:
            relative = str(path.relative_to(ROOT))
            archive.add(path, arcname='river-valley/'+relative)
            sources.append({'path': relative, 'sha256': sha(path)})
    root_id = store.folder(args.drive_parent_id, 'translation_curriculum_20261110')
    bootstrap = {'code_sha256': sha(output/'version_code.tar'), 'materials_sha256': sha(output/'version_materials.tar'),
        'drive_root_id': root_id, 'quota_project': args.quota_project, 'plan_sha256': sha(output/'plan.json')}
    (output/'bootstrap.json').write_text(json.dumps(bootstrap, indent=2)+'\n', encoding='utf-8')
    (output/'code_manifest.json').write_text(json.dumps(sources, indent=2)+'\n', encoding='utf-8')
    log.info('160步实验已冻结：起点250步、10240行、全新优化器、独立Drive根=%s；尚未分配GPU', root_id)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
