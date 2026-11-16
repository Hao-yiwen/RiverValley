"""冻结同起点、同数据的SFT监督量诊断；保留160步对照，不覆盖既有计划。"""
import argparse
import copy
import hashlib
import json
import logging
import math
from pathlib import Path
import shutil
import tarfile
import time

from prepare_milestone_probe import ROOT, sha

log = logging.getLogger(__name__)


def build_plan(baseline, train_rows):
    """只延长新诊断的训练时长和余弦计划，其他训练数学保持一致。"""
    if baseline.get('purpose') != 'milestone_diagnostic' or len(baseline['stages']) != 1:
        raise ValueError('必须以已经完成的单阶段里程碑诊断为对照')
    stage = baseline['stages'][0]
    origin = baseline['origin']
    if (stage['kind'] != 'sft' or stage['target_steps'] != 160
            or stage['config']['schedule_steps'] != 160 or train_rows != 10000):
        raise ValueError('对照必须是冻结的一万条数据和160步SFT')
    if (origin['stage'] != 'continued_pretrain' or origin['step'] != 7629
            or origin['drive_folder_id'] != stage['init_folder_id']):
        raise ValueError('本次诊断必须重新初始化同一个CPT7629来源，不能接诊断权重')
    config = stage['config']
    if config['batch_size'] * config['accumulation_steps'] != 64:
        raise ValueError('有效批量必须保持64')
    result = copy.deepcopy(baseline)
    probe = result['stages'][0]
    probe.update(id='sft_dose_7629_640', target_steps=640, backup_interval=320,
                 milestone_steps=[320, 640], evaluate_milestones=True)
    probe['config'].update(schedule_steps=640,
                           epochs=math.ceil(640 / (train_rows // 64)))
    result.update(version='v2_20261019_sft_dose_7629_640',
                  purpose='sft_supervision_diagnostic',
                  description='CPT7629同起点同一万条数据：延长SFT至640步，比较320与640步开发回答',
                  monitor_id='rv1b-v2-sft-dose-7629-640',
                  comparison={
                      'baseline': 'probe_7629', 'baseline_steps': 160,
                      'checkpoints': [320, 640], 'new_optimizer': True,
                      'unchanged': 'CPT来源、代码、数据、tokenizer、batch、seed、峰值/末端学习率及解码',
                      'changed': '目标更新160→640；余弦时长160→640；允许5个epoch以覆盖640步',
                      'limitations': '训练量与学习率衰减时长同时变化，不能单独归因于epoch；重复数据不计新预训练token。原验证共享首句及100题开发偏差保留。'},
                  following_stage='复核开发题和重复行为后决定SFT训练量；CPT仅从原完整状态恢复，正式SFT和240题留出仍未完成')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--decision', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    baseline, output = Path(args.baseline), Path(args.output)
    if output.exists():
        raise FileExistsError('诊断目录已存在，禁止覆盖冻结材料')
    decision = json.loads(Path(args.decision).read_text())
    b = json.loads((baseline / 'bootstrap.json').read_text())
    plan = json.loads((baseline / 'plan.json').read_text())
    review = json.loads((baseline / 'probe_7629_review_summary.json').read_text())
    verified = json.loads((baseline / 'latest_server_verification.json').read_text())
    if (decision['decision'] != 'investigate_sft_supervision_amount'
            or decision['baseline_plan_sha256'] != sha(baseline / 'plan.json')
            or decision['baseline_review_sha256'] != sha(baseline / 'probe_7629_review_summary.json')
            or not verified['all_done'] or not review['all_reviewed']
            or review['raw_sha256'] != verified['binding_verified']['answers_sha256']):
        raise ValueError('开发质量决策、已完成对照或回答绑定不一致')
    for name, key in [('version_code.tar', 'code_sha256'),
                      ('version_materials.tar', 'materials_sha256'), ('plan.json', 'plan_sha256')]:
        if sha(baseline / name) != b[key]:
            raise ValueError('冻结材料摘要不符：' + name)
    with tarfile.open(baseline / 'version_materials.tar') as archive:
        for remote, expected in plan['input_sha256'].items():
            body = archive.extractfile(remote.removeprefix('/content/river-valley-run/')).read()
            if hashlib.sha256(body).hexdigest() != expected:
                raise ValueError('材料内部文件摘要不符：' + remote)
        train = archive.extractfile('data/sft_pilot.jsonl').read().decode('utf-8')
        train_rows = len([line for line in train.splitlines() if line.strip()])
    result = build_plan(plan, train_rows)
    from drive_checkpoint_store import DriveStore
    from milestone_handoff import validate_origin
    roots = json.loads((ROOT / 'evals/results/v2_20261019/drive_roots.json').read_text())
    store = DriveStore(roots['version_root_id'], str(Path.home() / '.config/colab-cli/token.json'),
                       b['quota_project'])
    marker = store.child(plan['origin']['drive_folder_id'], 'BACKUP_COMPLETE.json')
    if not marker:
        raise ValueError('同起点完整预训练来源缺失')
    origin = store.request('GET', 'https://www.googleapis.com/drive/v3/files/' + marker,
                           params={'alt': 'media'}).json()
    validate_origin(origin, plan)
    store.verify(origin)
    remote = store.folder(store.root_id, 'sft_dose_probe_7629_640')
    output.mkdir(parents=True)
    for name in ('version_code.tar', 'version_materials.tar', 'code_manifest.json'):
        shutil.copy2(baseline / name, output / name)
    result['baseline_plan_sha256'] = sha(baseline / 'plan.json')
    result['quality_decision_sha256'] = sha(args.decision)
    (output / 'plan.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    bootstrap = dict(b, plan_sha256=sha(output / 'plan.json'), drive_root_id=remote)
    (output / 'bootstrap.json').write_text(json.dumps(bootstrap, indent=2) + '\n')
    receipt = {'time': time.time(), 'status': 'prepared_origin_server_verified',
               'origin': origin, 'train_rows': train_rows, 'decision_sha256': sha(args.decision),
               'baseline_review_sha256': sha(baseline / 'probe_7629_review_summary.json'),
               'code_unchanged': True, 'materials_unchanged': True}
    (output / 'preparation.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    log.info('监督量诊断已冻结：来源=%d 训练行=%d 目标=%d 输出=%s；尚未申请GPU',
             origin['step'], train_rows, result['stages'][0]['target_steps'], output)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
