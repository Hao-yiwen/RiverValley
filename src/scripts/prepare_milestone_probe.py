"""冻结里程碑短SFT对照材料；只建独立诊断计划，不改正在运行的预训练计划。"""
import argparse
import copy
import hashlib
import json
import logging
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger(__name__)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build_plan(pilot, cpt, step, source_folder):
    stage = cpt['stages'][0]
    if step not in stage['milestone_steps'] or stage['kind'] != 'pretrain':
        raise ValueError('诊断来源必须是冻结预训练计划中的里程碑')
    baseline = next(s for s in pilot['stages'] if s['id'] == 'clean_from_pretrain')
    probe = copy.deepcopy(baseline)
    probe.update(id=f'probe_{step}', init_folder_id=source_folder)
    if probe['target_steps'] != 160 or probe['config']['schedule_steps'] != 160:
        raise ValueError('短SFT必须沿用已完成C组的160步计划')
    return {
        'version': f'v2_20261019_probe_{step}', 'purpose': 'milestone_diagnostic',
        'description': f'新增预训练第{step}步接相同160步短SFT，和原C组对照',
        'monitor_id': f'rv1b-v2-probe-{step}-20261019', 'atomic_resume_pairs': True,
        'stages': [probe], 'input_sha256': copy.deepcopy(pilot['input_sha256']),
        'origin': {'stage': stage['id'], 'step': step, 'drive_folder_id': source_folder,
                   'rule': '启动前校验完整预训练来源；诊断SFT权重绝不用于继续预训练'},
        'comparison': {'baseline': 'pilot_01/clean_from_pretrain',
                      'baseline_steps': 160, 'data_and_config': '沿用原C组冻结输入和完整SFT配置',
                      'limitations': '原C组验证含9条共享首句；这是开发诊断，不是独立验收。'},
        'following_stage': '读取100条开发回答并评分作质量决策；继续预训练只恢复其原完整状态。'
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    base = ROOT / 'evals/results/v2_20261019'
    pilot_dir, cpt_dir, output = base / 'pilot_01', base / 'continued_01', Path(args.output)
    if output.exists():
        raise FileExistsError('诊断目录已存在，不能覆盖冻结材料')
    pilot = json.loads((pilot_dir / 'plan.json').read_text())
    cpt = json.loads((cpt_dir / 'plan.json').read_text())
    bootstrap = json.loads((cpt_dir / 'bootstrap.json').read_text())
    if sha(cpt_dir / 'version_code.tar') != bootstrap['code_sha256']:
        raise ValueError('当前预训练冻结源码摘要不符')
    with tarfile.open(pilot_dir / 'version_materials.tar') as archive:
        for name, expected in pilot['input_sha256'].items():
            relative = name.removeprefix('/content/river-valley-run/')
            actual = hashlib.sha256(archive.extractfile(relative).read()).hexdigest()
            if actual != expected:
                raise ValueError('原对照材料摘要不符：' + relative)
    # 对照需保持实际训练数学一致；允许流水线备份和证据绑定逻辑升级。
    with tarfile.open(pilot_dir / 'version_code.tar') as old, tarfile.open(cpt_dir / 'version_code.tar') as new:
        names = [m.name for m in old.getmembers() if m.isfile() and (
            m.name.startswith(('river-valley/src/model/', 'river-valley/src/data/', 'river-valley/src/tokenizer/'))
            or m.name in ('river-valley/src/scripts/sft.py', 'river-valley/src/config.py',
                          'river-valley/src/training_utils.py', 'river-valley/src/training_runtime.py',
                          'river-valley/src/scripts/eval_bilingual.py', 'river-valley/src/ddp_utils.py'))]
        if not {'river-valley/src/scripts/sft.py', 'river-valley/src/training_runtime.py',
                'river-valley/src/scripts/eval_bilingual.py'} <= set(names):
            raise ValueError('冻结包缺少关键训练或评估源码，不能声称与C组一致')
        for name in names:
            if old.extractfile(name).read() != new.extractfile(name).read():
                raise ValueError('与C组训练数学源码不同，必须先调查：' + name)
    from drive_checkpoint_store import DriveStore
    roots = json.loads((base / 'drive_roots.json').read_text())
    store = DriveStore(roots['version_root_id'], str(Path.home() / '.config/colab-cli/token.json'), bootstrap['quota_project'])
    parent = store.child(bootstrap['drive_root_id'], 'continued_pretrain')
    source = store.child(parent, f'step_{args.step:06d}') if parent else None
    if not source:
        raise ValueError('冻结计划预建的里程碑来源目录不存在')
    remote = store.folder(store.root_id, f'milestone_probe_{args.step}')
    plan = build_plan(pilot, cpt, args.step, source)
    output.mkdir(parents=True)
    for filename, origin in [('version_code.tar', cpt_dir), ('code_manifest.json', cpt_dir),
                             ('version_materials.tar', pilot_dir)]:
        shutil.copy2(origin / filename, output / filename)
    plan['cpt_plan_sha256'] = sha(cpt_dir / 'plan.json')
    plan['baseline_plan_sha256'] = sha(pilot_dir / 'plan.json')
    (output / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    b = {'code_sha256': sha(output / 'version_code.tar'), 'materials_sha256': sha(output / 'version_materials.tar'),
         'plan_sha256': sha(output / 'plan.json'), 'drive_root_id': remote, 'quota_project': bootstrap['quota_project']}
    (output / 'bootstrap.json').write_text(json.dumps(b, indent=2) + '\n')
    receipt = {'status': 'prepared_waiting_for_verified_origin', 'milestone_step': args.step,
               'source_folder_id': source, 'drive_root_id': remote,
               'math_files_compared': names, 'math_matches_baseline': True,
               'baseline_review': str(pilot_dir / 'clean_from_pretrain_review_summary.json'),
               'note': '预建来源目录不表示权重已保存；启动控制器必须读取服务器完整清单再放行'}
    (output / 'preparation.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    log.info('短SFT材料就绪：步数=%d 输出=%s 源码一致文件=%d；未申请GPU', args.step, output, len(names))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
