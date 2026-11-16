"""通过数据与预训练完成门禁后冻结正式 SFT 计划，先不申请任何 GPU。"""
import argparse
import json
import logging
from pathlib import Path
import tarfile

from prepare_version_pilot import ROOT, sha
from deliver_version import load_completed

log = logging.getLogger(__name__)


def build_plan(manifest, decision, origin, remote):
    """显式绑定数据决策和新预训练来源，不允许沿用旧版权重冒充新版本。"""
    if decision.get('status') != 'approved_for_formal_sft' or decision.get('data_hashes') != manifest['hashes']:
        raise ValueError('正式材料尚未完成内容复核与实验决策，或决策摘要不匹配')
    if not decision.get('pilot_comparison_sha256') or not decision.get('selection_reason'):
        raise ValueError('正式决策缺少三组对照依据')
    if not 0 < origin['step'] or 'model.safetensors' not in origin['files']:
        raise ValueError('新预训练来源不完整')
    effective_batch = 64
    per_epoch = manifest['train']['rows'] // effective_batch
    if per_epoch < 1:
        raise ValueError('训练样本不足一个完整有效批量')
    learning_rate = decision['learning_rate']
    if not 0 < learning_rate <= 1e-3:
        raise ValueError('SFT学习率不在受支持范围')
    target = per_epoch * 2
    config = json.loads((ROOT / 'configs/colab_a100_1b_sft.json').read_text())
    for key in ('checkpoint_dir', 'pretrain_checkpoint', 'max_steps'):
        config.pop(key, None)
    config.update(data_path=remote+'data/sft_train.jsonl', val_data_path=remote+'data/sft_val.jsonl',
                  bpe_model_path=remote+'tokenizer.json', random_seed=1337, num_workers=2,
                  learning_rate=learning_rate, min_lr=learning_rate/10, epochs=2,
                  schedule_steps=target, eval_interval=max(1, per_epoch//4),
                  eval_iters=(manifest['val']['rows']+3)//4, save_interval=per_epoch,
                  batch_size=4, accumulation_steps=16)
    stage = {'id': 'sft', 'kind': 'sft', 'target_steps': target, 'backup_interval': per_epoch,
             'milestone_steps': [per_epoch, target], 'evaluate_milestones': True, 'config': config,
             'init_folder_id': origin['drive_folder_id'], 'eval_questions': remote+'development.jsonl'}
    return {'version': 'v2_20261019_formal_sft', 'purpose': 'final_sft_candidate', 'atomic_resume_pairs': True,
            'description': '第二版：新增预训练后的正式中英文SFT', 'monitor_id': 'rv1b-v2-formal-sft-20261019',
            'stages': [stage], 'epoch_comparison': {'steps': [per_epoch, target],
                'rule': '每轮完整断点保留；用开发题及验证指标选择候选，锁定后进行独立留出验收'},
            'data_accounting': {'available_rows': manifest['train']['rows'],
                'rows_per_epoch': per_epoch*effective_batch, 'dropped_tail_per_epoch': manifest['train']['rows']%effective_batch,
                'available_answer_tokens_per_pass': manifest['train']['answer_tokens']},
            'selection_reason': decision['selection_reason']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--decision', required=True)
    parser.add_argument('--pretrain-evidence', required=True)
    parser.add_argument('--pretrain-stage', default='continued_pretrain')
    parser.add_argument('--use-best-pretrain', action='store_true')
    parser.add_argument('--drive-root-id', required=True)
    parser.add_argument('--quota-project', required=True, help='使用者自己的 Google Cloud 计费项目')
    parser.add_argument('--output', default='evals/results/v2_20261019/formal_sft_01')
    args = parser.parse_args()
    candidate, output = Path(args.candidate), Path(args.output)
    if output.exists():
        raise FileExistsError('不能覆盖已经冻结的正式SFT目录')
    manifest = json.loads((candidate/'manifest.json').read_text())
    decision = json.loads(Path(args.decision).read_text())
    for name, digest in manifest['hashes'].items():
        if sha(candidate/name) != digest:
            raise ValueError('候选文件已修改：'+name)
    credential = str(Path.home()/'.config/colab-cli/token.json')
    _, _, receipt = load_completed(Path(args.pretrain_evidence), credential)
    key = args.pretrain_stage + ('_best' if args.use_best_pretrain else '')
    origin = receipt['checkpoints'][key]
    remote = '/content/river-valley-run/'
    plan = build_plan(manifest, decision, origin, remote)
    files = {'data/sft_train.jsonl': candidate/'sft_train.jsonl', 'data/sft_val.jsonl': candidate/'sft_val.jsonl',
             'tokenizer.json': candidate.parent/'tokenizer.json', 'sft_manifest.json': candidate/'manifest.json',
             'sft_review_decision.json': Path(args.decision),
             'development.jsonl': ROOT/'evals/bilingual_acceptance_v1.jsonl'}
    if sha(files['tokenizer.json']) != origin['files']['tokenizer.json']['sha256']:
        raise ValueError('新预训练与SFT tokenizer摘要不一致')
    plan['input_sha256'] = {remote+name: sha(path) for name,path in files.items()}
    plan['pretrain_plan_sha256'] = receipt['plan_sha256']
    output.mkdir(parents=True)
    (output/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    with tarfile.open(output/'version_materials.tar','w') as archive:
        for name,path in files.items():archive.add(path,arcname=name)
    sources=[]
    with tarfile.open(output/'version_code.tar','w') as archive:
        for folder in ('src','datasets','tests'):
            for path in sorted((ROOT/folder).rglob('*.py')):
                if '__pycache__' in path.parts:continue
                relative=str(path.relative_to(ROOT));archive.add(path,arcname='river-valley/'+relative)
                sources.append({'path':relative,'sha256':sha(path)})
        config_path = ROOT/'configs/colab_a100_1b_sft.json'
        archive.add(config_path,arcname='river-valley/configs/colab_a100_1b_sft.json')
        sources.append({'path':'configs/colab_a100_1b_sft.json','sha256':sha(config_path)})
        archive.add(files['development.jsonl'],arcname='river-valley/evals/bilingual_acceptance_v1.jsonl')
    bootstrap={'code_sha256':sha(output/'version_code.tar'),'materials_sha256':sha(output/'version_materials.tar'),
               'drive_root_id':args.drive_root_id,'quota_project':args.quota_project,'plan_sha256':sha(output/'plan.json')}
    (output/'bootstrap.json').write_text(json.dumps(bootstrap,indent=2)+'\n')
    (output/'code_manifest.json').write_text(json.dumps(sources,indent=2)+'\n')
    log.info('正式SFT材料冻结完成：目录=%s 每轮=%d步 总目标=%d步；未申请GPU',output,
             plan['epoch_comparison']['steps'][0],plan['stages'][0]['target_steps'])


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    main()
