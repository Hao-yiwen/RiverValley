"""冻结三组对照实验的代码、数据、起点和相同更新计划，供单实例顺序执行。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quota-project', required=True, help='使用者自己的 Google Cloud 计费项目')
    args = parser.parse_args()
    output = ROOT / 'evals/results/v2_20261019/pilot_01'
    output.mkdir(exist_ok=True)
    new = ROOT / 'checkpoints/bilingual_v2_20261019'
    old = ROOT / 'checkpoints/bilingual_20261011'
    manifest = json.loads((new / 'sft_manifest.json').read_text())
    review = json.loads((new / 'sft_review_decision.json').read_text())
    if manifest.get('refinement_code_sha256') != sha(ROOT / 'datasets/refine_sft_v2.py'):
        raise ValueError('精筛未按当前规则成功完成，禁止把旧输出当成新材料')
    if review['status'] != 'approved_for_pilot' or review['data_sha256'] != manifest['hashes']:
        raise ValueError('当前数据尚未完成对应版本的抽检，不能冻结启动')
    files = {'data/old_sft_train.jsonl': old / 'data/sft_train.jsonl',
             'data/sft_pilot.jsonl': new / 'data/sft_pilot.jsonl',
             'data/sft_val.jsonl': new / 'data/sft_val.jsonl',
             'tokenizer.json': old / 'tokenizer.json',
             'development.jsonl': ROOT / 'evals/bilingual_acceptance_v1.jsonl',
             'sft_manifest.json': new / 'sft_manifest.json',
             'sft_review_decision.json': new / 'sft_review_decision.json'}
    remote = '/content/river-valley-run/'
    config = {'bpe_model_path': remote + 'tokenizer.json', 'val_data_path': remote + 'data/sft_val.jsonl',
              'random_seed': 1337, 'batch_size': 4, 'accumulation_steps': 16, 'max_seq_len': 1024,
              'learning_rate': 2e-5, 'min_lr': 2e-6, 'warmup_ratio': .03, 'epochs': 2,
              'schedule_steps': 160, 'eval_iters': 100, 'eval_interval': 80, 'save_interval': 80,
              'log_interval': 10, 'weight_decay': .1, 'grad_clip': 1., 'num_workers': 2,
              'pin_memory': True, 'prefetch_factor': 2, 'use_compile': False,
              'use_gradient_checkpointing': False, 'use_native_gqa': True, 'dynamic_padding': True,
              'adamw_fused': True, 'use_swanlab': False, 'device': 'cuda'}
    roots = json.loads((output.parent / 'drive_roots.json').read_text())
    stages = []
    for name, data, initial in [('old_data_control', 'old_sft_train', roots['old_sft_best_id']),
                                ('repaired_sft', 'sft_pilot', roots['old_sft_best_id']),
                                ('clean_from_pretrain', 'sft_pilot', roots['old_pretrain_final_id'])]:
        stage = {'id': name, 'kind': 'sft', 'target_steps': 160, 'backup_interval': 80,
                 'init_folder_id': initial, 'eval_questions': remote + 'development.jsonl',
                 'config': dict(config, data_path=remote + 'data/' + data + '.jsonl')}
        if name == 'old_data_control':
            stage['initial_eval_questions'] = remote + 'development.jsonl'
        stages.append(stage)
    plan = {'version': 'v2_20261019_pilot', 'description': '第二版：旧数据、新数据、预训练起点三组对照',
            'monitor_id': 'rv1b-v2-pilot-20261019', 'stages': stages,
            'input_sha256': {remote + name: sha(path) for name, path in files.items()},
            'comparison': '同 seed、更新数、batch、学习率；材料长度不同，监督 token 数另行报告。旧版同 GPU 精度重测。'}
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
        for name in ['evals/bilingual_acceptance_v1.jsonl']:
            archive.add(ROOT / name, arcname='river-valley/' + name)
    bootstrap = {'code_sha256': sha(output / 'version_code.tar'),
                 'materials_sha256': sha(output / 'version_materials.tar'),
                 'drive_root_id': roots['pilot_root_id'], 'quota_project': args.quota_project,
                 'plan_sha256': sha(output / 'plan.json')}
    (output / 'bootstrap.json').write_text(json.dumps(bootstrap, indent=2) + '\n')
    (output / 'code_manifest.json').write_text(json.dumps(sources, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'files': len(sources), 'bootstrap': bootstrap}, indent=2))


if __name__ == '__main__':
    main()
