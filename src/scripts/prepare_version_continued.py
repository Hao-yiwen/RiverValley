"""材料全部完成后冻结继续预训练计划；不申请算力，也不提前批准全量 SFT。"""
import argparse
import json
from pathlib import Path
import tarfile

from prepare_version_pilot import ROOT, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--drive-root-id', required=True)
    parser.add_argument('--quota-project', required=True, help='使用者自己的 Google Cloud 计费项目')
    parser.add_argument('--init-folder-id', required=True, help='已核验的初始化检查点 Drive 目录')
    args = parser.parse_args()
    output = ROOT / 'evals/results/v2_20261019/continued_01'
    output.mkdir(exist_ok=True)
    new = ROOT / 'checkpoints/bilingual_v2_20261019'
    old = ROOT / 'checkpoints/bilingual_20261011'
    manifest = json.loads((new / 'pretrain_manifest.json').read_text())
    target = manifest['training_updates']
    if target != 15258 or manifest['actual_training_tokens'] != 1999896576:
        raise ValueError('材料实际量未达到本轮20亿token计划，不能按文件名猜测完成')
    if sha(new / 'tokenizer.json') != sha(old / 'tokenizer.json'):
        raise ValueError('继续预训练不允许更换 tokenizer')
    files = {'tokenizer.json': new / 'tokenizer.json', 'pretrain_manifest.json': new / 'pretrain_manifest.json',
             'pretrain_sources.lock.json': ROOT / 'configs/v2_pretrain_sources.lock.json'}
    for part in ['train', 'val', 'val_zh', 'val_en']:
        folder = new / 'data/pretrain' / part
        meta = json.loads((folder / 'meta.json').read_text())
        if sha(folder / 'train.bin') != meta['data_sha256']:
            raise ValueError('增量memmap摘要不符：' + part)
        for name in ['train.bin', 'meta.json']:
            files[f'data/pretrain/{part}/{name}'] = folder / name
    for part in ['val_zh', 'val_en']:
        for name in ['train.bin', 'meta.json']:
            files[f'data/pretrain/v1_{part}/{name}'] = old / 'data/pretrain' / part / name
    remote = '/content/river-valley-run/'
    config = json.loads((ROOT / 'configs/colab_a100_1b_pretrain.json').read_text())
    config.pop('checkpoint_dir', None)
    config.pop('max_steps', None)
    config.update(learning_rate=1e-4, min_lr=1e-5, schedule_steps=target, save_interval=500,
                  new_stage_from_complete=True, initial_checkpoint_expected_step=7629,
                  extra_val_memmap_dirs={name: remote + 'data/pretrain/' + name
                     for name in ['val_zh', 'val_en', 'v1_val_zh', 'v1_val_en']})
    stage = {'id': 'continued_pretrain', 'kind': 'pretrain', 'target_steps': target, 'backup_interval': 500,
             'milestone_steps': [3814, 7629, 11443, 15258], 'config': config,
             'init_folder_id': args.init_folder_id}
    plan = {'version': 'v2_20261019_continued', 'description': '第二版：新增20亿token继续预训练', 'atomic_resume_pairs': True,
            'monitor_id': 'rv1b-v2-continued-20261019', 'stages': [stage],
            'input_sha256': {remote + name: sha(path) for name, path in files.items()},
            'stage_transition': '从完成的7629步保留Adam/RNG，新的数据游标与warmup/余弦计划；新峰值lr1e-4，末端1e-5',
            'token_accounting': {'previous': 999948288, 'added': 1999896576, 'cumulative': 2999844864},
            'following_stage': '本计划完成后核验验证指标与对照实验，冻结完整SFT材料，再执行SFT；不能将本回执当作第二版交付完成'}
    (output / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    with tarfile.open(output / 'version_materials.tar', 'w') as archive:
        for name, path in files.items():
            archive.add(path, arcname=name)
    sources = []
    with tarfile.open(output / 'version_code.tar', 'w') as archive:
        for folder in ['src', 'datasets', 'tests']:
            for path in sorted((ROOT / folder).rglob('*.py')):
                if '__pycache__' not in path.parts:
                    archive.add(path, arcname='river-valley/' + str(path.relative_to(ROOT)))
                    sources.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path)})
        archive.add(ROOT / 'evals/bilingual_acceptance_v1.jsonl', arcname='river-valley/evals/bilingual_acceptance_v1.jsonl')
        config_path = ROOT / 'configs/colab_a100_1b_sft.json'
        archive.add(config_path, arcname='river-valley/configs/colab_a100_1b_sft.json')
        sources.append({'path': 'configs/colab_a100_1b_sft.json', 'sha256': sha(config_path)})
    bootstrap = {'code_sha256': sha(output / 'version_code.tar'), 'materials_sha256': sha(output / 'version_materials.tar'),
                 'drive_root_id': args.drive_root_id, 'quota_project': args.quota_project,
                 'plan_sha256': sha(output / 'plan.json')}
    (output / 'bootstrap.json').write_text(json.dumps(bootstrap, indent=2) + '\n')
    (output / 'code_manifest.json').write_text(json.dumps(sources, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'target_steps': target, 'token_accounting': plan['token_accounting']}, indent=2))


if __name__ == '__main__':
    main()
